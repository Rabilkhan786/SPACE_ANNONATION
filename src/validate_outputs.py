"""Independently load every saved Q1 tile, mask and label and check alignment."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from image_ops import read_fits, sha256, write_json


def validate(root, source_folder=None):
    rows = list(csv.DictReader((root / "tile_metadata.csv").open(encoding="utf-8")))
    failures = []
    names = [r["tile"] for r in rows]
    mappings = list(
        csv.DictReader((root / "qa" / "label_instances.csv").open(encoding="utf-8"))
    )
    mapping = {(m["tile"], int(m["label_row"])): int(m["object_id"]) for m in mappings}
    instances = {
        stem: np.load(root / "instances" / f"{stem}.npy", mmap_mode="r")
        for stem in {Path(r["source"]).stem for r in rows}
    }
    if len(names) != len(set(names)):
        failures.append("duplicate tile names")
    for directory, extension in [
        ("images", ".png"),
        ("masks", ".png"),
        ("labels", ".txt"),
        ("scientific_tiles", ".npy"),
        ("valid_pixels", ".png"),
    ]:
        found = {p.stem for p in (root / directory).glob("*" + extension)}
        if found != set(names):
            failures.append(f"{directory}: missing or extra files")
    polygon_count = 0
    mask_pixels = 0
    max_difference = 0
    for row in rows:
        name = row["tile"]
        vh = int(row["valid_height"])
        vw = int(row["valid_width"])
        gray = np.array(Image.open(root / "images" / f"{name}.png"))
        mask = np.array(Image.open(root / "masks" / f"{name}.png"))
        validity = np.array(Image.open(root / "valid_pixels" / f"{name}.png"))
        raw = np.load(root / "scientific_tiles" / f"{name}.npy")
        if any(a.shape != (1024, 1024) for a in [gray, mask, validity, raw]):
            failures.append(f"{name}: wrong shape")
            continue
        if not set(np.unique(mask)).issubset({0, 1, 2}):
            failures.append(f"{name}: invalid class mask")
        if mask[vh:, :].any() or mask[:, vw:].any():
            failures.append(f"{name}: padding annotated")
        expected_valid = np.zeros((1024, 1024), np.uint8)
        expected_valid[:vh, :vw] = 255
        if not np.array_equal(validity, expected_valid):
            failures.append(f"{name}: invalid validity mask")
        rebuilt = np.zeros((1024, 1024), np.uint8)
        tile_instances = instances[Path(row["source"]).stem][
            int(row["y_offset"]) : int(row["y_offset"]) + vh,
            int(row["x_offset"]) : int(row["x_offset"]) + vw,
        ]
        per_instance = {}
        for line_index, line in enumerate(
            (root / "labels" / f"{name}.txt").read_text().splitlines(), 1
        ):
            values = line.split()
            if len(values) < 7 or len(values) % 2 != 1 or values[0] not in ("0", "1"):
                failures.append(f"{name}: bad label row")
                continue
            points = np.asarray(values[1:], np.float32).reshape(-1, 2)
            if (
                not np.isfinite(points).all()
                or (points < 0).any()
                or (points > 1).any()
            ):
                failures.append(f"{name}: coordinates outside [0,1]")
                continue
            if cv2.contourArea(points) <= 0 or len(np.unique(points, axis=0)) < 3:
                failures.append(f"{name}: degenerate polygon")
            cv2.fillPoly(
                rebuilt, [(points * 1024).astype(np.int32)], int(values[0]) + 1
            )
            oid = mapping.get((name, line_index))
            if oid is None:
                failures.append(f"{name}: missing parent instance ID")
            else:
                binary = per_instance.setdefault(oid, np.zeros((1024, 1024), np.uint8))
                cv2.fillPoly(binary, [(points * 1024).astype(np.int32)], 1)
            polygon_count += 1
        for oid, binary in per_instance.items():
            if not np.array_equal(binary[:vh, :vw] != 0, tile_instances == oid):
                failures.append(f"{name}: polygon differs from instance {oid}")
        diff = int(np.count_nonzero(rebuilt != mask))
        max_difference = max(max_difference, diff)
        if diff:
            failures.append(f"{name}: {diff} polygon/mask pixel mismatches")
        mask_pixels += int(np.count_nonzero(mask))
    source_checks = []
    for source in sorted(set(r["source"] for r in rows)):
        tiles = [r for r in rows if r["source"] == source]
        h = int(tiles[0]["original_height"])
        w = int(tiles[0]["original_width"])
        seen = np.zeros(((h + 1023) // 1024, (w + 1023) // 1024), bool)
        reconstruction = np.zeros((h, w), np.uint8)
        reconstruction_gray = np.zeros((h, w), np.uint8)
        original = None
        if source_folder:
            original, _ = read_fits(source_folder / source)
            if original.shape != (h, w):
                failures.append(f"{source}: dimensions differ from FITS")
        for tile in tiles:
            x = int(tile["x_offset"])
            y = int(tile["y_offset"])
            vh = int(tile["valid_height"])
            vw = int(tile["valid_width"])
            name = tile["tile"]
            if x % 1024 or y % 1024 or seen[y // 1024, x // 1024]:
                failures.append(f"{name}: invalid or duplicate placement")
            seen[y // 1024, x // 1024] = True
            reconstruction[y : y + vh, x : x + vw] = np.array(
                Image.open(root / "masks" / f"{name}.png")
            )[:vh, :vw]
            reconstruction_gray[y : y + vh, x : x + vw] = np.array(
                Image.open(root / "images" / f"{name}.png")
            )[:vh, :vw]
            if original is not None:
                saved = np.load(root / "scientific_tiles" / f"{name}.npy")[:vh, :vw]
                if not np.array_equal(
                    saved, original[y : y + vh, x : x + vw], equal_nan=True
                ):
                    failures.append(f"{name}: scientific pixels altered")
        stem = Path(source).stem
        if not seen.all():
            failures.append(f"{source}: missing tiles")
        if not np.array_equal(
            reconstruction,
            np.array(Image.open(root / "repatched" / f"{stem}_mask.png")),
        ):
            failures.append(f"{source}: mask repatch differs")
        if not np.array_equal(
            reconstruction_gray,
            np.array(Image.open(root / "repatched" / f"{stem}_processed.png")),
        ):
            failures.append(f"{source}: processed repatch differs")
        if Image.open(root / "repatched" / f"{stem}_overlay.png").size != (w, h):
            failures.append(f"{source}: overlay dimensions")
        check = dict(
            source=source,
            all_tile_positions_present=bool(seen.all()),
            original_pixels_compared=original is not None,
        )
        if original is not None:
            digest = sha256(source_folder / source)
            expected = json.loads(
                (root / "inspection" / stem / "statistics.json").read_text()
            )["sha256"]
            if digest != expected:
                failures.append(f"{source}: source hash mismatch")
            check["source_sha256_unchanged"] = digest == expected
        source_checks.append(check)
    result = dict(
        passed=not failures,
        tiles_checked=len(rows),
        polygons_checked=polygon_count,
        mask_pixels=mask_pixels,
        maximum_polygon_mask_pixel_difference=max_difference,
        failures=failures,
        sources=source_checks,
        scope="Technical integrity only. This does not establish detection recall, class accuracy or completed manual QA.",
    )
    write_json(root / "qa" / "validation.json", result)
    print(json.dumps(result, indent=2))
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-folder", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if validate(args.run, args.source_folder) else 1)
