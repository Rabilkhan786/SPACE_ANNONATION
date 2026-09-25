"""Check tiles, human annotation exports, masks and full-size reconstruction."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_tiles import ROOT, TILE_SIZE, find_fits, read_fits, display_image
from process_annotations import load_metadata, match_labels, polygon_mask, overlay


def validate(root: Path = ROOT, prepared_only: bool = False) -> None:
    """Raise a clear error on the first inconsistency; never alter files."""
    files = find_fits(root)
    rows = load_metadata(root)
    if {r["source_file"] for r in rows} != {p.name for p in files}:
        raise ValueError("Metadata does not cover all 10 FITS images.")
    labels = {} if prepared_only else match_labels(rows, root)
    expected_total = 0
    for path in files:
        raw = read_fits(path)
        height, width = raw.shape
        display = display_image(raw)
        del raw
        group = [r for r in rows if r["source_file"] == path.name]
        expected = ((width + TILE_SIZE - 1) // TILE_SIZE) * (
            (height + TILE_SIZE - 1) // TILE_SIZE
        )
        expected_total += expected
        if len(group) != expected:
            raise ValueError(
                f"{path.name}: expected {expected} tiles, found {len(group)}."
            )
        full_mask = np.zeros((height, width), dtype=np.uint8)
        for row in group:
            name = row["tile_file"]
            if (row["original_width"], row["original_height"]) != (width, height):
                raise ValueError(f"{name}: metadata dimensions differ from FITS.")
            x, y, w, h = (row[k] for k in ["x", "y", "valid_width", "valid_height"])
            image = np.asarray(Image.open(root / "tiles" / name))
            if image.shape != (TILE_SIZE, TILE_SIZE) or image.dtype != np.uint8:
                raise ValueError(f"{name}: tile must be 1024x1024 8-bit grayscale.")
            if image[h:, :].any() or image[:, w:].any():
                raise ValueError(f"{name}: image padding is not zero.")
            if not np.array_equal(image[:h, :w], display[y : y + h, x : x + w]):
                raise ValueError(f"{name}: tile differs from FITS display pixels.")
            if prepared_only:
                continue
            mask = np.asarray(Image.open(root / "annotations/masks" / name))
            if mask.shape != image.shape or not set(np.unique(mask)).issubset(
                {0, 1, 2}
            ):
                raise ValueError(f"{name}: invalid mask shape or class values.")
            if mask[h:, :].any() or mask[:, w:].any():
                raise ValueError(f"{name}: padding contains annotations.")
            expected_mask = polygon_mask(labels[Path(name).stem], w, h)
            if not np.array_equal(mask, expected_mask):
                raise ValueError(f"{name}: YOLO polygons and mask do not match.")
            actual_overlay = np.asarray(Image.open(root / "outputs/overlays" / name))
            if not np.array_equal(actual_overlay, overlay(image, mask)):
                raise ValueError(f"{name}: overlay does not match image and mask.")
            full_mask[y : y + h, x : x + w] = mask[:h, :w]
        if not prepared_only:
            for suffix, expected_array in [
                ("processed", display),
                ("mask", full_mask),
                ("overlay", overlay(display, full_mask)),
            ]:
                output = root / "outputs/repatched" / f"{path.stem}_{suffix}.png"
                actual = np.asarray(Image.open(output))
                if not np.array_equal(actual, expected_array):
                    raise ValueError(
                        f"{output.name}: incorrect dimensions or reconstruction."
                    )
        print(f"Checked {path.name}: {width}x{height}, {expected} tiles.", flush=True)
    print(f"Checked {len(files)} FITS files and {expected_total} tiles.")
    if prepared_only:
        print("PREPARATION CHECK PASSED. Annotation validation is still pending.")
    else:
        print(
            "VALIDATION PASSED. File consistency only; human visual QA is still required."
        )


if __name__ == "__main__":
    try:
        validate()
    except (OSError, ValueError, KeyError, csv.Error) as error:
        raise SystemExit(f"VALIDATION FAILED: {error}")
