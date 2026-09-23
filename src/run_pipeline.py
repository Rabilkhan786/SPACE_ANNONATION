"""Run Question 1 of the Digantara AI/ML Data Annotation assessment.

The pipeline follows the three required Question 1 stages:
A1. Inspect FITS imagery and apply justified preprocessing.
A2. Divide processed imagery into exact 1024 x 1024 tiles without losing data.
A3. Create two-class pixel masks and YOLO segmentation labels for:
    0 = star/blob
    1 = streak/object

The automatic output is a first-pass annotation. Ambiguous, faint, blended, and
boundary cases are flagged for manual QA before the dataset is treated as final.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

from image_ops import (
    image_statistics,
    make_display_image,
    mask_to_polygons,
    measure_component,
    overlay_mask,
    rasterize_polygon,
    read_fits,
    standardize_local_background,
)

CLASS_NAMES = {
    0: "star_blob",
    1: "streak_object",
}


def write_json(path: Path, value: Any) -> None:
    """Write a JSON file using readable indentation."""
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write a CSV file with a fixed column order."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_fits_files(path: Path) -> list[Path]:
    """Return FITS files from a file path or directory."""
    extensions = {".fits", ".fit", ".fts"}

    if path.is_file():
        return [path] if path.suffix.lower() in extensions else []

    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    )


def inspect_source(
    path: Path,
    inspection_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Inspect one FITS image and save statistics plus representative previews."""
    image = read_fits(path)
    info = image_statistics(image)
    info["file"] = path.name

    inspection_dir.mkdir(parents=True, exist_ok=True)

    lower = info["percentiles"]["1"]
    upper = max(info["percentiles"]["99.99"], lower + 1.0)
    info["display_limits"] = [lower, upper]

    display = make_display_image(image, lower, upper)

    overview = Image.fromarray(display)
    overview.thumbnail((1600, 1200))
    overview.save(inspection_dir / "overview.png")

    height, width = image.shape
    crop_size = 512
    crop_origins = [
        (0, 0),
        (
            max(0, width // 2 - crop_size // 2),
            max(0, height // 2 - crop_size // 2),
        ),
        (
            max(0, width - crop_size),
            max(0, height - crop_size),
        ),
    ]

    for index, (x0, y0) in enumerate(crop_origins, start=1):
        crop = display[y0 : y0 + crop_size, x0 : x0 + crop_size]
        Image.fromarray(crop).save(
            inspection_dir / f"crop_{index}_x{x0}_y{y0}.png"
        )

    write_json(inspection_dir / "statistics.json", info)
    return image, info


def _review_reasons(
    metrics: dict[str, float | int],
    bounds: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    config: dict[str, Any],
    is_streak: bool,
) -> list[str]:
    """Return reasons that make a candidate worth manual review."""
    x0, y0, x1, y1 = bounds
    height, width = image_shape
    review = config["review"]
    reasons: list[str] = []

    if is_streak:
        reasons.append("streak_candidate")

    if (
        review["ambiguous_elongation_min"]
        <= metrics["elongation"]
        < config["streak"]["elongation_min"]
    ):
        reasons.append("ambiguous_shape")

    if metrics["peak_snr"] < review["faint_peak_snr"]:
        reasons.append("faint_source")

    if metrics["area_px"] > review["large_area_px"]:
        reasons.append("large_or_blended")

    tile_size = config["tile_size"]
    crosses_tile = (
        y0 // tile_size != (y1 - 1) // tile_size
        or x0 // tile_size != (x1 - 1) // tile_size
    )
    if crosses_tile:
        reasons.append("tile_crossing")

    if x0 == 0 or y0 == 0 or x1 == width or y1 == height:
        reasons.append("source_edge")

    if metrics["area_px"] <= 10 or metrics["minor_axis_px"] < 1.5:
        reasons.append("small_or_impulsive")

    return reasons


def detect_candidates(
    raw: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Detect candidate stars/blobs and streaks on the full source image.

    Detection is performed before tiling so that an object crossing a 1024-pixel
    tile boundary is measured once, then clipped consistently into output tiles.
    """
    finite = np.isfinite(raw)
    if not finite.any():
        raise ValueError("Image contains no finite pixels.")

    work = raw.astype(np.float32)
    work[~finite] = float(np.median(work[finite]))

    snr, background_info = standardize_local_background(
        work,
        config["background_cell_px"],
        config["noise_floor"],
    )
    del work

    smoothed = ndimage.gaussian_filter(
        snr,
        sigma=config["detection_smoothing_sigma_px"],
        mode="reflect",
    )
    score, smoothed_info = standardize_local_background(
        smoothed,
        config["background_cell_px"],
        0.05,
    )
    del smoothed

    support = (score >= config["grow_sigma"]) & finite
    seeds = (score >= config["seed_sigma"]) & finite
    del score

    labels, component_count = ndimage.label(
        support,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    seeded_ids = np.unique(labels[seeds])
    seeded_ids = seeded_ids[seeded_ids != 0]
    del seeds, support

    allowed = np.zeros(component_count + 1, dtype=bool)
    allowed[seeded_ids] = True

    instances = np.zeros(raw.shape, dtype=np.int32)
    candidates: list[dict[str, Any]] = []
    next_object_id = 1

    for old_id, component_slice in enumerate(
        ndimage.find_objects(labels),
        start=1,
    ):
        if component_slice is None or not allowed[old_id]:
            continue

        component = labels[component_slice] == old_id
        if int(component.sum()) < config["minimum_area_px"]:
            continue

        component = (
            ndimage.binary_fill_holes(component)
            & finite[component_slice]
        )
        metrics = measure_component(
            component,
            snr[component_slice],
            raw[component_slice],
        )

        if metrics["peak_snr"] < config["minimum_peak_snr"]:
            continue

        streak = config["streak"]
        is_streak = (
            metrics["elongation"] >= streak["elongation_min"]
            and metrics["major_axis_px"] >= streak["major_axis_min_px"]
            and metrics["eccentricity"] >= streak["eccentricity_min"]
        )

        y0 = component_slice[0].start
        y1 = component_slice[0].stop
        x0 = component_slice[1].start
        x1 = component_slice[1].stop

        reasons = _review_reasons(
            metrics,
            (x0, y0, x1, y1),
            raw.shape,
            config,
            is_streak,
        )

        region = instances[component_slice]
        region[component] = next_object_id

        candidates.append(
            {
                "object_id": next_object_id,
                "class_id": int(is_streak),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                **metrics,
                "needs_review": bool(reasons),
                "review_reason": ";".join(reasons),
            }
        )
        next_object_id += 1

    detection_info = {
        "raw_background": background_info,
        "smoothed_background": smoothed_info,
        "candidate_count": len(candidates),
    }
    return instances, candidates, detection_info


def export_source(
    raw: np.ndarray,
    info: dict[str, Any],
    instances: np.ndarray,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Export tiles, masks, YOLO labels, overlays, and repatched images."""
    tile_size = config["tile_size"]
    height, width = raw.shape
    source_stem = Path(info["file"]).stem

    class_lookup = np.zeros(
        max((row["object_id"] for row in candidates), default=0) + 1,
        dtype=np.uint8,
    )
    for row in candidates:
        class_lookup[row["object_id"]] = row["class_id"] + 1

    full_class_mask = class_lookup[instances]
    lower, upper = info["display_limits"]
    full_display = make_display_image(raw, lower, upper)

    metadata: list[dict[str, Any]] = []

    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            valid_height = min(tile_size, height - y0)
            valid_width = min(tile_size, width - x0)
            tile_name = f"{source_stem}_x{x0:05d}_y{y0:05d}"

            image_tile = np.zeros(
                (tile_size, tile_size),
                dtype=np.uint8,
            )
            mask_tile = np.zeros(
                (tile_size, tile_size),
                dtype=np.uint8,
            )

            image_tile[:valid_height, :valid_width] = full_display[
                y0 : y0 + valid_height,
                x0 : x0 + valid_width,
            ]
            mask_tile[:valid_height, :valid_width] = full_class_mask[
                y0 : y0 + valid_height,
                x0 : x0 + valid_width,
            ]

            local_instances = instances[
                y0 : y0 + valid_height,
                x0 : x0 + valid_width,
            ]

            label_rows: list[str] = []
            rasterized_labels = np.zeros(
                (tile_size, tile_size),
                dtype=np.uint8,
            )

            for object_id in np.unique(local_instances):
                if object_id == 0:
                    continue

                pixel_y, pixel_x = np.nonzero(
                    local_instances == object_id
                )
                local_x0 = int(pixel_x.min())
                local_x1 = int(pixel_x.max() + 1)
                local_y0 = int(pixel_y.min())
                local_y1 = int(pixel_y.max() + 1)

                fragment = (
                    local_instances[
                        local_y0:local_y1,
                        local_x0:local_x1,
                    ]
                    == object_id
                )
                polygons = mask_to_polygons(
                    fragment,
                    offset_xy=(local_x0, local_y0),
                    tile_size=tile_size,
                )

                class_id = int(class_lookup[object_id] - 1)

                for polygon in polygons:
                    values = " ".join(
                        f"{value:.8f}"
                        for value in polygon.ravel()
                    )
                    label_rows.append(f"{class_id} {values}")

                    parsed = np.asarray(
                        [float(value) for value in values.split()],
                        dtype=np.float32,
                    ).reshape(-1, 2)
                    rasterized_labels[
                        rasterize_polygon(parsed, tile_size)
                    ] = class_id + 1

            if not np.array_equal(
                rasterized_labels,
                mask_tile,
            ):
                raise AssertionError(
                    "YOLO polygons do not reproduce mask for tile: "
                    f"{tile_name}"
                )

            Image.fromarray(image_tile).save(
                output_dir / "images" / f"{tile_name}.png"
            )
            Image.fromarray(mask_tile).save(
                output_dir / "masks" / f"{tile_name}.png"
            )
            Image.fromarray(
                overlay_mask(image_tile, mask_tile)
            ).save(
                output_dir / "overlays" / f"{tile_name}.png"
            )
            (
                output_dir / "labels" / f"{tile_name}.txt"
            ).write_text(
                "\n".join(label_rows)
                + ("\n" if label_rows else ""),
                encoding="utf-8",
            )

            metadata.append(
                {
                    "tile": tile_name,
                    "source": info["file"],
                    "row": y0 // tile_size,
                    "column": x0 // tile_size,
                    "x_offset": x0,
                    "y_offset": y0,
                    "original_width": width,
                    "original_height": height,
                    "valid_width": valid_width,
                    "valid_height": valid_height,
                    "pad_right": tile_size - valid_width,
                    "pad_bottom": tile_size - valid_height,
                }
            )

    Image.fromarray(full_display).save(
        output_dir
        / "repatched"
        / f"{source_stem}_processed.png"
    )
    Image.fromarray(full_class_mask).save(
        output_dir
        / "repatched"
        / f"{source_stem}_mask.png"
    )
    full_overlay = overlay_mask(full_display, full_class_mask)
    Image.fromarray(full_overlay).save(
        output_dir
        / "repatched"
        / f"{source_stem}_overlay.png"
    )

    overview = Image.fromarray(full_overlay)
    overview.thumbnail((1600, 1200))
    overview.save(
        output_dir
        / "inspection"
        / source_stem
        / "annotation_overview.png"
    )

    return metadata


def run(args: argparse.Namespace) -> None:
    """Run inspection and annotation export for all discovered FITS files."""
    config = yaml.safe_load(
        args.config.read_text(encoding="utf-8")
    )
    fits_files = discover_fits_files(args.input)

    if not fits_files:
        raise ValueError("No FITS files found.")

    if config["tile_size"] != 1024:
        raise ValueError(
            "The assessment requires tile_size: 1024."
        )

    if args.output.exists():
        raise FileExistsError(
            f"Output folder already exists: {args.output}. "
            "Use a new folder to avoid overwriting previous work."
        )

    args.output.mkdir(parents=True)
    required_folders = ["inspection"]

    if not args.inspect_only:
        required_folders += [
            "images",
            "labels",
            "masks",
            "overlays",
            "repatched",
            "qa",
        ]

    for folder in required_folders:
        (args.output / folder).mkdir()

    inventory = {
        "found": len(fits_files),
        "required_by_assessment": 10,
        "dataset_complete": len(fits_files) == 10,
        "files": [path.name for path in fits_files],
    }
    write_json(
        args.output / "inventory.json",
        inventory,
    )

    if len(fits_files) != 10:
        print(
            f"NOTE: found {len(fits_files)} FITS file(s). "
            "The final assessment run must use all 10 supplied images.",
            flush=True,
        )

    all_candidates: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    for path in fits_files:
        print(
            f"Inspecting: {path.name}",
            flush=True,
        )
        raw, info = inspect_source(
            path,
            args.output / "inspection" / path.stem,
        )

        if args.inspect_only:
            continue

        print(
            f"Detecting candidates: {path.name}",
            flush=True,
        )
        instances, candidates, detection_info = (
            detect_candidates(raw, config)
        )

        for row in candidates:
            row["source"] = path.name

        metadata = export_source(
            raw,
            info,
            instances,
            candidates,
            args.output,
            config,
        )

        all_candidates.extend(candidates)
        all_metadata.extend(metadata)
        source_summaries.append(
            {
                "file": path.name,
                "shape_yx": info["shape_yx"],
                "candidate_count": len(candidates),
                "review_required": sum(
                    bool(row["needs_review"])
                    for row in candidates
                ),
                "detection": detection_info,
            }
        )

    if args.inspect_only:
        print(
            f"Inspection outputs written to: {args.output}",
            flush=True,
        )
        return

    candidate_fields = [
        "source",
        "object_id",
        "class_id",
        "x0",
        "y0",
        "x1",
        "y1",
        "area_px",
        "width_px",
        "height_px",
        "major_axis_px",
        "minor_axis_px",
        "elongation",
        "eccentricity",
        "orientation_deg",
        "compactness",
        "peak_intensity",
        "mean_intensity",
        "peak_snr",
        "mean_snr",
        "needs_review",
        "review_reason",
    ]
    write_csv(
        args.output / "qa" / "candidates.csv",
        all_candidates,
        candidate_fields,
    )

    metadata_fields = [
        "tile",
        "source",
        "row",
        "column",
        "x_offset",
        "y_offset",
        "original_width",
        "original_height",
        "valid_width",
        "valid_height",
        "pad_right",
        "pad_bottom",
    ]
    write_csv(
        args.output / "tile_metadata.csv",
        all_metadata,
        metadata_fields,
    )

    class_counts = {
        str(class_id): sum(
            row["class_id"] == class_id
            for row in all_candidates
        )
        for class_id in CLASS_NAMES
    }
    write_json(
        args.output / "qa" / "run_summary.json",
        {
            "status": "automatic_draft_pending_manual_qa",
            "input_count": len(fits_files),
            "assessment_input_count": 10,
            "dataset_complete": len(fits_files) == 10,
            "tile_count": len(all_metadata),
            "candidate_count": len(all_candidates),
            "review_required_count": sum(
                bool(row["needs_review"])
                for row in all_candidates
            ),
            "class_counts": class_counts,
            "sources": source_summaries,
            "config": config,
        },
    )

    data_yaml = {
        "path": ".",
        "train": "images",
        "val": None,
        "names": CLASS_NAMES,
    }
    (
        args.output / "data.yaml"
    ).write_text(
        yaml.safe_dump(
            data_yaml,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Question 1 draft written to: {args.output}",
        flush=True,
    )
    print(
        "Next: review flagged candidates, correct inaccurate masks/classes, "
        "then run validate_outputs.py.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="FITS file or folder containing FITS images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New output folder for this run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "config.yaml"
        ),
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect FITS files; do not generate annotations.",
    )
    run(parser.parse_args())
