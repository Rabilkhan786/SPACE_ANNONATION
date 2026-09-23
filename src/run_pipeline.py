"""Question 1: inspect FITS, generate reviewable masks, tile and export privately."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
from skimage.segmentation import watershed

from image_ops import (
    read_fits,
    statistics,
    sha256,
    write_json,
    standardize,
    display,
    overlay,
    measurements,
    mask_polygons,
    rasterize,
)


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_inputs(path):
    extensions = {".fits", ".fit", ".fts"}
    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in extensions
    )


def inspect_source(path, folder):
    raw, hdu = read_fits(path)
    info = statistics(raw)
    info.update(file=path.name, hdu=hdu, sha256=sha256(path))
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "statistics.json", info)
    limits = [
        info["percentiles"]["1"],
        max(info["percentiles"]["99.99"], info["percentiles"]["1"] + 1),
    ]
    info["display_limits"] = limits
    # Thumbnails are for navigation; crops retain original pixels.
    thumb = Image.fromarray(display(raw[::5, ::5], limits))
    thumb.save(folder / "overview.png")
    h, w = raw.shape
    for index, (x, y) in enumerate(
        [
            (0, 0),
            (max(0, w // 2 - 256), max(0, h // 2 - 256)),
            (max(0, w - 512), max(0, h - 512)),
        ]
    ):
        Image.fromarray(display(raw[y : y + 512, x : x + 512], limits)).save(
            folder / f"crop_{index}_x{x}_y{y}.png"
        )
    write_json(folder / "statistics.json", info)
    return raw, info


def detect(raw, cfg):
    """Detect across the original image, then clip the SAME masks into tiles."""
    finite = np.isfinite(raw)
    work = raw.astype(np.float32)
    work[~finite] = np.median(work[finite])
    z, noise_info = standardize(work, cfg["background_cell_px"], cfg["noise_floor"])
    del work
    smooth = ndimage.gaussian_filter(
        z, cfg["detection_smoothing_sigma_px"], mode="reflect"
    )
    score, smooth_info = standardize(smooth, cfg["background_cell_px"], 0.05)
    del smooth
    support = (score >= cfg["grow_sigma"]) & finite
    seeds = (score >= cfg["seed_sigma"]) & finite
    del score
    labels, count = ndimage.label(support, structure=np.ones((3, 3)))
    del support
    seed_ids = np.unique(labels[seeds])
    del seeds
    allowed = np.zeros(count + 1, bool)
    allowed[seed_ids] = True
    allowed[0] = False
    slices = ndimage.find_objects(labels)
    instances = np.zeros(raw.shape, np.int32)
    candidates = []
    next_id = 1
    for old_id, sl in enumerate(slices, 1):
        if sl is None or not allowed[old_id]:
            continue
        component = labels[sl] == old_id
        if component.sum() < cfg["minimum_area_px"]:
            continue
        component = ndimage.binary_fill_holes(component) & finite[sl]
        metric = measurements(component, z[sl], raw[sl])
        if metric["peak_snr"] < cfg["minimum_peak_snr"]:
            continue
        ratio = metric["elongation"]
        major = metric["major_axis_px"]
        cls = int(
            ratio >= cfg["streak_elongation"]
            and major >= cfg["streak_min_major_px"]
            and metric["eccentricity"] >= 0.85
        )
        reasons = []
        if cls:
            reasons.append("streak_candidate")
        if cfg["ambiguous_elongation_min"] <= ratio < cfg["streak_elongation"]:
            reasons.append("ambiguous_shape")
        if metric["peak_snr"] < cfg["faint_review_snr"]:
            reasons.append("faint_source")
        if metric["area_px"] > cfg["large_area_review_px"]:
            reasons.append("large_or_blended")
        if (
            sl[0].start // 1024 != (sl[0].stop - 1) // 1024
            or sl[1].start // 1024 != (sl[1].stop - 1) // 1024
        ):
            reasons.append("tile_crossing")
        if (
            sl[0].start == 0
            or sl[1].start == 0
            or sl[0].stop == raw.shape[0]
            or sl[1].stop == raw.shape[1]
        ):
            reasons.append("source_edge")
        if metric["area_px"] <= 10 or metric["minor_axis_px"] < 1.5:
            reasons.append("small_or_impulsive")
        instances[sl][component] = next_id
        candidates.append(
            dict(
                object_id=next_id,
                class_id=cls,
                x0=sl[1].start,
                y0=sl[0].start,
                x1=sl[1].stop,
                y1=sl[0].stop,
                **metric,
                review_status="pending",
                review_reason=";".join(reasons) or "routine",
                review_note="",
            )
        )
        next_id += 1
    return instances, candidates, dict(raw=noise_info, smoothed=smooth_info)


def apply_corrections(instances, candidates, corrections, source_hash):
    """Replay explicit review edits. Full-source polygons use pixel coordinates."""
    if not corrections:
        return []
    if corrections["source_sha256"] != source_hash:
        raise ValueError("Corrections refer to another image")
    by_id = {r["object_id"]: r for r in candidates}
    applied = []
    for edit in corrections["edits"]:
        operation = edit["action"]
        oid = int(edit.get("object_id", max(by_id, default=0) + 1))
        if not edit.get("note", "").strip():
            raise ValueError("Every correction requires a review note")
        if operation != "add" and oid not in by_id:
            raise ValueError("Unknown correction object ID")
        if operation == "add" and oid in by_id:
            raise ValueError("New object ID already exists")
        if operation == "split":
            if int(edit["class_id"]) not in (0, 1):
                raise ValueError("Class ID must be 0 or 1")
            row = by_id[oid]
            x0, y0, x1, y1 = [int(row[k]) for k in ["x0", "y0", "x1", "y1"]]
            region = instances[y0:y1, x0:x1]
            mask = region == oid
            markers = np.zeros(mask.shape, np.int32)
            for index, (x, y) in enumerate(edit["seeds_xy"], 1):
                xx, yy = int(x) - x0, int(y) - y0
                if (
                    not (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1])
                    or not mask[yy, xx]
                ):
                    raise ValueError("Split seed outside object")
                if markers[yy, xx]:
                    raise ValueError("Duplicate split seeds")
                markers[yy, xx] = index
            if markers.max() < 2:
                raise ValueError("Split requires at least two seeds")
            parts = watershed(-ndimage.distance_transform_edt(mask), markers, mask=mask)
            child_ids = []
            for index in range(1, int(markers.max()) + 1):
                new_id = max(by_id) + 1
                child = dict(row)
                child.update(
                    object_id=new_id,
                    class_id=int(edit["class_id"]),
                    review_status="visually_reviewed",
                    review_note=edit["note"],
                )
                by_id[new_id] = child
                candidates.append(child)
                child_ids.append(new_id)
                region[parts == index] = new_id
            row.update(
                review_status="deleted",
                review_note=edit["note"] + " Split into " + str(child_ids),
            )
            applied.append(dict(edit, child_ids=child_ids))
            continue
        if operation in ("replace", "add"):
            pts = np.asarray(edit["polygon_xy"], float)
            if (
                pts.ndim != 2
                or pts.shape[1] != 2
                or len(pts) < 3
                or not np.isfinite(pts).all()
            ):
                raise ValueError("Invalid correction polygon")
            h, w = instances.shape
            if (
                (pts < 0).any()
                or (pts[:, 0] >= w).any()
                or (pts[:, 1] >= h).any()
                or cv2.contourArea(pts.astype(np.float32)) <= 0
            ):
                raise ValueError("Correction polygon outside source or degenerate")
            x0, y0 = np.floor(pts.min(axis=0)).astype(int)
            x1, y1 = np.floor(pts.max(axis=0)).astype(int) + 1
            painted = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(painted, [(pts - [x0, y0]).astype(np.int32)], 1)
            existing = instances[y0:y1, x0:x1]
            if np.any((existing != 0) & (existing != oid) & (painted != 0)):
                raise ValueError(
                    "Correction overlaps another instance; review the conflict"
                )
            instances[instances == oid] = 0
            existing[painted != 0] = oid
            if operation == "add":
                by_id[oid] = (
                    dict(
                        object_id=oid,
                        **{key: "" for key in candidates[0] if key != "object_id"},
                    )
                    if candidates
                    else dict(object_id=oid)
                )
                candidates.append(by_id[oid])
                by_id[oid]["review_reason"] = "manual_addition"
            by_id[oid].update(
                x0=int(x0),
                y0=int(y0),
                x1=int(x1),
                y1=int(y1),
                area_px=int(painted.sum()),
            )
        elif operation == "delete":
            instances[instances == oid] = 0
        elif operation not in ("accept", "reclassify"):
            raise ValueError(f"Unknown action {operation}")
        if operation in ("add", "replace", "reclassify"):
            cls = int(edit["class_id"])
            if cls not in (0, 1):
                raise ValueError("Class ID must be 0 or 1")
            by_id[oid]["class_id"] = cls
        by_id[oid].update(
            review_status="deleted" if operation == "delete" else "visually_reviewed",
            review_note=edit["note"],
        )
        applied.append(edit)
    return applied


def refresh_edited_measurements(raw, instances, candidates, cfg):
    """Recompute evidence after changed masks; do not retain stale area/shape."""
    z, _ = standardize(raw, cfg["background_cell_px"], cfg["noise_floor"])
    slices = ndimage.find_objects(instances)
    for row in candidates:
        oid = row["object_id"]
        if row["review_status"] == "deleted":
            continue
        sl = slices[oid - 1]
        if sl is None:
            raise ValueError("Active reviewed object has no pixels")
        row.update(measurements(instances[sl] == oid, z[sl], raw[sl]))
        row.update(x0=sl[1].start, y0=sl[0].start, x1=sl[1].stop, y1=sl[0].stop)


def export_source(raw, info, instances, candidates, output, cfg):
    size = cfg["tile_size"]
    h, w = raw.shape
    source = Path(info["file"]).stem
    class_lookup = np.zeros(
        max([r["object_id"] for r in candidates], default=0) + 1, np.uint8
    )
    for r in candidates:
        class_lookup[r["object_id"]] = r["class_id"] + 1
    class_full = class_lookup[instances]
    np.save(output / "instances" / f"{source}.npy", instances)
    Image.fromarray(class_full).save(output / "repatched" / f"{source}_mask.png")
    rebuilt_gray = np.zeros((h, w), np.uint8)
    rebuilt_mask = np.zeros((h, w), np.uint8)
    metadata = []
    object_map = []
    covered = 0
    exact_raw = True
    exact_polygons = True
    for y in range(0, h, size):
        for x in range(0, w, size):
            vh, vw = min(size, h - y), min(size, w - x)
            name = f"{source}_x{x:05d}_y{y:05d}"
            scientific = np.zeros((size, size), raw.dtype)
            scientific[:vh, :vw] = raw[y : y + vh, x : x + vw]
            np.save(output / "scientific_tiles" / f"{name}.npy", scientific)
            restored = np.load(output / "scientific_tiles" / f"{name}.npy")
            exact_raw &= bool(
                np.array_equal(
                    restored[:vh, :vw], raw[y : y + vh, x : x + vw], equal_nan=True
                )
            )
            valid = np.zeros((size, size), np.uint8)
            valid[:vh, :vw] = 255
            Image.fromarray(valid).save(output / "valid_pixels" / f"{name}.png")
            gray = np.zeros((size, size), np.uint8)
            gray[:vh, :vw] = display(scientific[:vh, :vw], info["display_limits"])
            mask = np.zeros((size, size), np.uint8)
            mask[:vh, :vw] = class_full[y : y + vh, x : x + vw]
            local = instances[y : y + vh, x : x + vw]
            rows = []
            raster_mask = np.zeros((size, size), np.uint8)
            # Each clipped fragment retains its full-image parent object ID/class.
            for oid in np.unique(local):
                if oid == 0:
                    continue
                ry, rx = np.nonzero(local == oid)
                x0, x1 = int(rx.min()), int(rx.max() + 1)
                y0, y1 = int(ry.min()), int(ry.max() + 1)
                component = local[y0:y1, x0:x1] == oid
                for poly in mask_polygons(component, (x0, y0), size):
                    cls = int(class_lookup[oid] - 1)
                    row = str(cls) + " " + " ".join(f"{v:.8f}" for v in poly.ravel())
                    # Test the serialized coordinates, not the unrounded polygon.
                    parsed = np.array(list(map(float, row.split()[1:]))).reshape(-1, 2)
                    raster_mask[rasterize(parsed, size)] = cls + 1
                    rows.append(row)
                    object_map.append(
                        dict(
                            tile=name,
                            label_row=len(rows),
                            object_id=int(oid),
                            class_id=cls,
                        )
                    )
            exact_polygons &= bool(np.array_equal(raster_mask, mask))
            (output / "labels" / f"{name}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
            )
            Image.fromarray(gray).save(output / "images" / f"{name}.png")
            Image.fromarray(mask).save(output / "masks" / f"{name}.png")
            Image.fromarray(overlay(gray, mask)).save(
                output / "overlays" / f"{name}.png"
            )
            rebuilt_gray[y : y + vh, x : x + vw] = gray[:vh, :vw]
            rebuilt_mask[y : y + vh, x : x + vw] = mask[:vh, :vw]
            metadata.append(
                dict(
                    tile=name,
                    source=info["file"],
                    row=y // size,
                    column=x // size,
                    x_offset=x,
                    y_offset=y,
                    original_width=w,
                    original_height=h,
                    valid_width=vw,
                    valid_height=vh,
                    pad_right=size - vw,
                    pad_bottom=size - vh,
                )
            )
            covered += vh * vw
    Image.fromarray(rebuilt_gray).save(output / "repatched" / f"{source}_processed.png")
    rgb = overlay(rebuilt_gray, rebuilt_mask)
    Image.fromarray(rgb).save(output / "repatched" / f"{source}_overlay.png")
    thumb = Image.fromarray(rgb)
    thumb.thumbnail((1600, 1100))
    thumb.save(output / "inspection" / source / "overlay_overview.png")
    checks = dict(
        scientific_pixels_lossless=exact_raw,
        polygons_match_masks=exact_polygons,
        masks_repatch_exactly=bool(np.array_equal(rebuilt_mask, class_full)),
        original_pixel_count=h * w,
        covered_original_pixels=covered,
        no_source_pixels_lost=covered == h * w,
        original_sha256_unchanged=info["sha256"],
    )
    if not all(
        checks[k]
        for k in [
            "scientific_pixels_lossless",
            "polygons_match_masks",
            "masks_repatch_exactly",
            "no_source_pixels_lost",
        ]
    ):
        raise AssertionError(f"Export validation failed: {checks}")
    return metadata, object_map, checks


def run(args):
    cfg = yaml.safe_load(args.config.read_text())
    paths = discover_inputs(args.input)
    if not paths:
        raise ValueError("No FITS inputs found")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Duplicate source stems would collide")
    if not args.inspect_only and not args.diagnostic and len(paths) != 10:
        raise ValueError(
            f"Question 1 requires 10 FITS images; found {len(paths)}. Use --inspect-only for inspection, or --diagnostic for an explicitly incomplete test run."
        )
    if cfg["tile_size"] != 1024:
        raise ValueError("Question 1 requires exact 1024px tiles")
    if args.output.exists():
        raise FileExistsError(
            "Use a new output directory; previous work will not be overwritten"
        )
    args.output.mkdir(parents=True)
    write_json(
        args.output / "inventory.json",
        dict(
            found=len(paths),
            required=10,
            dataset_complete=len(paths) == 10,
            mode=(
                "inspection"
                if args.inspect_only
                else "diagnostic" if args.diagnostic else "full_dataset"
            ),
            files=[p.name for p in paths],
        ),
    )
    for folder in [
        "inspection",
        "images",
        "labels",
        "masks",
        "overlays",
        "scientific_tiles",
        "valid_pixels",
        "instances",
        "repatched",
        "qa",
    ]:
        (args.output / folder).mkdir()
    all_rows = []
    all_metadata = []
    all_map = []
    sources = []
    for path in paths:
        print(f"Inspecting {path.name}", flush=True)
        raw, info = inspect_source(path, args.output / "inspection" / path.stem)
        if args.inspect_only:
            continue
        print("Detecting full-image candidates before clipping to tiles", flush=True)
        instances, rows, noise = detect(raw, cfg)
        for row in rows:
            row["source"] = path.name
        correction_data = None
        if args.corrections:
            if args.corrections.is_dir():
                review_path = args.corrections / f"{path.stem}.json"
                if review_path.exists():
                    correction_data = json.loads(review_path.read_text())
            else:
                if len(paths) != 1:
                    raise ValueError(
                        "For multiple images, supply a corrections folder with one JSON file per source stem"
                    )
                correction_data = json.loads(args.corrections.read_text())
        if correction_data and correction_data.get("config") != json.loads(
            json.dumps(cfg)
        ):
            raise ValueError(
                "Corrections must include the exact detection config to keep object IDs stable"
            )
        edits = apply_corrections(instances, rows, correction_data, info["sha256"])
        for row in rows:
            row["source"] = path.name
        if edits:
            refresh_edited_measurements(raw, instances, rows, cfg)
        print(f"Exporting {len(rows)} candidates", flush=True)
        metadata, mapping, checks = export_source(
            raw, info, instances, rows, args.output, cfg
        )
        if sha256(path) != info["sha256"]:
            raise AssertionError("Source FITS changed during processing")
        sources.append(
            dict(
                file=path.name,
                statistics=info,
                noise=noise,
                checks=checks,
                corrections_applied=edits,
            )
        )
        all_rows += rows
        all_metadata += metadata
        all_map += mapping
        del raw, instances
    if args.inspect_only:
        return
    write_csv(args.output / "tile_metadata.csv", all_metadata)
    write_csv(
        args.output / "qa" / "candidates.csv",
        all_rows,
        fields=(
            list(all_rows[0])
            if all_rows
            else ["object_id", "class_id", "review_status"]
        ),
    )
    write_csv(
        args.output / "qa" / "label_instances.csv",
        all_map,
        fields=["tile", "label_row", "object_id", "class_id"],
    )
    write_json(
        args.output / "qa" / "run_summary.json",
        dict(
            status="draft_pending_review",
            input_count=len(paths),
            required_input_count=10,
            tile_count=len(all_metadata),
            candidate_count=sum(r["review_status"] != "deleted" for r in all_rows),
            audit_rows=len(all_rows),
            pending_review=sum(r["review_status"] == "pending" for r in all_rows),
            visually_reviewed=sum(
                r["review_status"] == "visually_reviewed" for r in all_rows
            ),
            class_counts={
                str(c): sum(
                    r["class_id"] == c and r["review_status"] != "deleted"
                    for r in all_rows
                )
                for c in [0, 1]
            },
            sources=sources,
            config=cfg,
        ),
    )
    # No fictitious train/validation split: these are annotation outputs only.
    (args.output / "data.yaml").write_text(
        yaml.safe_dump(
            dict(
                path=str(args.output.resolve()),
                train="images",
                val=None,
                names={0: "star_blob", 1: "streak_object"},
            ),
            sort_keys=False,
        )
    )
    print(
        f"Completed draft export to {args.output}; manual annotation QA remains separate.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.yaml",
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Incomplete dataset test; never marks assessment complete",
    )
    parser.add_argument("--corrections", type=Path)
    run(parser.parse_args())
