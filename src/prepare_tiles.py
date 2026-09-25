"""Prepare display-only PNG tiles from the 10 confidential FITS images."""

import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TILE_SIZE = 1024
EXPECTED_SOURCES = 10
FIELDS = [
    "source_file",
    "tile_file",
    "x",
    "y",
    "original_width",
    "original_height",
    "valid_width",
    "valid_height",
]


def find_fits(root: Path = ROOT) -> list[Path]:
    """Find inputs and reject duplicate names or an incomplete dataset."""
    paths = sorted(
        p
        for p in (root / "data/raw").rglob("*")
        if p.is_file() and p.suffix.lower() in {".fits", ".fit", ".fts"}
    )
    if len(paths) != EXPECTED_SOURCES:
        raise ValueError(f"Expected {EXPECTED_SOURCES} FITS files; found {len(paths)}.")
    if len({p.stem.casefold() for p in paths}) != len(paths):
        raise ValueError("FITS filenames must have unique stems.")
    return paths


def read_fits(path: Path) -> np.ndarray:
    """Reuse the first 2D image HDU; memmap=False supports unsigned FITS."""
    with fits.open(path, memmap=False) as hdus:
        for hdu in hdus:
            if (
                isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                and hdu.data is not None
                and hdu.data.ndim == 2
            ):
                return np.asarray(hdu.data).copy()
    raise ValueError(f"No 2D image found in {path.name}.")


def display_image(raw: np.ndarray) -> np.ndarray:
    """Reuse per-image percentile clipping and arcsinh; never change raw data."""
    if not np.isfinite(raw).all() or raw.size == 0:
        raise ValueError("Empty or non-finite FITS image; inspect the source first.")
    low, high = np.percentile(raw, [1, 99.99])
    high = max(high, low + 1.0)
    scaled = np.clip((raw.astype(np.float32) - low) / (high - low), 0, 1)
    return np.rint(np.arcsinh(5 * scaled) / np.arcsinh(5) * 255).astype(np.uint8)


def prepare(root: Path = ROOT) -> None:
    paths = find_fits(root)
    tiles_dir = root / "tiles"
    metadata_path = root / "tile_metadata.csv"
    if metadata_path.exists() or (tiles_dir.exists() and any(tiles_dir.iterdir())):
        raise ValueError(
            "Tiles/metadata already exist. Preserve them elsewhere before rerunning."
        )
    tiles_dir.mkdir(exist_ok=True)
    rows = []
    for index, path in enumerate(paths, 1):
        print(f"Processing FITS {index}/{len(paths)}: {path.name}", flush=True)
        raw = read_fits(path)
        image = display_image(raw)
        height, width = image.shape
        count = 0
        for y in range(0, height, TILE_SIZE):
            for x in range(0, width, TILE_SIZE):
                valid_width = min(TILE_SIZE, width - x)
                valid_height = min(TILE_SIZE, height - y)
                tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
                tile[:valid_height, :valid_width] = image[
                    y : y + valid_height, x : x + valid_width
                ]
                name = f"{path.stem}_x{x:04d}_y{y:04d}.png"
                Image.fromarray(tile).save(tiles_dir / name)
                rows.append(
                    dict(
                        zip(
                            FIELDS,
                            [
                                path.name,
                                name,
                                x,
                                y,
                                width,
                                height,
                                valid_width,
                                valid_height,
                            ],
                        )
                    )
                )
                count += 1
        print(
            f"  {width} x {height}; created {count} tiles; "
            f"padding right={(-width) % TILE_SIZE}, bottom={(-height) % TILE_SIZE}.",
            flush=True,
        )
    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Total tiles: {len(rows)}. Metadata: {metadata_path}")


if __name__ == "__main__":
    try:
        prepare()
    except (OSError, ValueError) as error:
        raise SystemExit(f"PREPARATION FAILED: {error}")
