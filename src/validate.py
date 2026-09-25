"""Validate prepared tiles, reviewed annotations and reconstructed outputs."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_tiles import ROOT, TILE_SIZE, display_image, find_fits, read_fits
from process_annotations import (
    find_labels,
    load_metadata,
    overlay,
    polygon_mask,
)


def validate(root: Path = ROOT) -> None:
    """Check structure, geometry, masks, padding and reconstruction."""
    files = find_fits(root)
    rows = load_metadata(root)
    labels = find_labels(rows, root)

    if {row["source_file"] for row in rows} != {path.name for path in files}:
        raise ValueError("Metadata does not cover all 10 FITS images.")

    total_tiles = 0

    for path in files:
        raw = read_fits(path)
        height, width = raw.shape
        display = display_image(raw)
        del raw

        group = [row for row in rows if row["source_file"] == path.name]
        expected = ((width + TILE_SIZE - 1) // TILE_SIZE) * (
            (height + TILE_SIZE - 1) // TILE_SIZE
        )
        total_tiles += expected

        if len(group) != expected:
            raise ValueError(
                f"{path.name}: expected {expected} tiles, found {len(group)}."
            )

        full_mask = np.zeros((height, width), dtype=np.uint8)

        for row in group:
            name = row["tile_file"]
            stem = Path(name).stem
            x = row["x"]
            y = row["y"]
            valid_width = row["valid_width"]
            valid_height = row["valid_height"]

            with Image.open(root / "tiles" / name) as image_file:
                image = np.asarray(image_file.convert("L"))

            if image.shape != (TILE_SIZE, TILE_SIZE):
                raise ValueError(f"{name}: tile must be 1024x1024.")

            if image[valid_height:, :].any() or image[:, valid_width:].any():
                raise ValueError(f"{name}: image padding is not zero.")

            if not np.array_equal(
                image[:valid_height, :valid_width],
                display[
                    y : y + valid_height,
                    x : x + valid_width,
                ],
            ):
                raise ValueError(f"{name}: tile differs from the prepared FITS display.")

            mask_path = root / "annotations/masks" / name
            overlay_path = root / "outputs/overlays" / name

            if not mask_path.exists() or not overlay_path.exists():
                raise ValueError(f"{name}: mask or overlay is missing.")

            mask = np.asarray(Image.open(mask_path))

            if mask.shape != (TILE_SIZE, TILE_SIZE):
                raise ValueError(f"{name}: mask must be 1024x1024.")

            if not set(np.unique(mask)).issubset({0, 1, 2}):
                raise ValueError(f"{name}: mask contains an invalid class value.")

            if mask[valid_height:, :].any() or mask[:, valid_width:].any():
                raise ValueError(f"{name}: padding contains annotations.")

            expected_mask = polygon_mask(
                labels.get(stem),
                valid_width,
                valid_height,
            )

            if not np.array_equal(mask, expected_mask):
                raise ValueError(f"{name}: YOLO polygon and mask do not match.")

            actual_overlay = np.asarray(Image.open(overlay_path))

            if not np.array_equal(actual_overlay, overlay(image, mask)):
                raise ValueError(f"{name}: overlay does not match image and mask.")

            full_mask[
                y : y + valid_height,
                x : x + valid_width,
            ] = mask[:valid_height, :valid_width]

        for suffix, expected_array in [
            ("processed", display),
            ("mask", full_mask),
            ("overlay", overlay(display, full_mask)),
        ]:
            output = root / "outputs/repatched" / f"{path.stem}_{suffix}.png"

            if not output.exists():
                raise ValueError(f"Missing reconstructed output: {output.name}")

            actual = np.asarray(Image.open(output))

            if not np.array_equal(actual, expected_array):
                raise ValueError(f"{output.name}: reconstruction is incorrect.")

        print(f"Checked {path.name}: {width}x{height}, {expected} tiles.")

    print(f"Checked {len(files)} FITS files and {total_tiles} tiles.")
    print(
        "VALIDATION PASSED. "
        "File consistency is correct; manual visual QA is still required."
    )


if __name__ == "__main__":
    try:
        validate()
    except (OSError, ValueError, KeyError, csv.Error) as error:
        raise SystemExit(f"VALIDATION FAILED: {error}")
