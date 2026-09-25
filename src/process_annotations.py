"""Convert human-reviewed YOLO polygons to class masks and full-size overlays."""

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from prepare_tiles import ROOT, TILE_SIZE, FIELDS


def load_metadata(root: Path = ROOT) -> list[dict]:
    """Read the tile layout and check complete, non-overlapping source grids."""
    with (root / "tile_metadata.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Unexpected tile_metadata.csv columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("Tile metadata is empty.")
    names = set()
    sources = {}
    for row in rows:
        for key in FIELDS[2:]:
            row[key] = int(row[key])
        name, source = row["tile_file"], row["source_file"]
        if Path(name).name != name or Path(source).name != source:
            raise ValueError("Metadata must contain filenames, not paths.")
        x, y = row["x"], row["y"]
        w, h = row["original_width"], row["original_height"]
        if (
            name in names
            or name != f"{Path(source).stem}_x{x:04d}_y{y:04d}.png"
            or not (0 <= x < w and 0 <= y < h)
            or x % TILE_SIZE
            or y % TILE_SIZE
            or row["valid_width"] != min(TILE_SIZE, w - x)
            or row["valid_height"] != min(TILE_SIZE, h - y)
        ):
            raise ValueError(f"Invalid or duplicate tile metadata: {name}")
        names.add(name)
        sources.setdefault(source, []).append(row)
    for source, group in sources.items():
        w, h = group[0]["original_width"], group[0]["original_height"]
        expected = ((w + TILE_SIZE - 1) // TILE_SIZE) * (
            (h + TILE_SIZE - 1) // TILE_SIZE
        )
        if len(group) != expected or any(
            (r["original_width"], r["original_height"]) != (w, h) for r in group
        ):
            raise ValueError(f"Incomplete or inconsistent tile grid: {source}")
    actual = {p.name for p in (root / "tiles").glob("*.png")}
    if actual != names:
        raise ValueError("PNG files do not match tile_metadata.csv.")
    return rows


def match_labels(rows: list[dict], root: Path = ROOT) -> dict[str, Path]:
    """Require one label per tile; missing labels are NOT empty annotations."""
    expected = {Path(r["tile_file"]).stem for r in rows}
    matches = {}
    for path in sorted((root / "annotations/labels").rglob("*.txt")):
        stem = path.stem
        # Accept Roboflow's filename suffix, but never guess a different image.
        if stem not in expected:
            stem = stem.split("_png.rf.", 1)[0]
        if stem not in expected:
            raise ValueError(f"Label has no matching tile: {path.name}")
        if stem in matches:
            raise ValueError(f"More than one label for tile: {stem}")
        matches[stem] = path
    missing = sorted(expected - matches.keys())
    if missing:
        raise ValueError(
            f"Missing {len(missing)} label files (first: {missing[0]}.txt). "
            "Finish manual annotation/export; reviewed empty tiles need empty TXT files."
        )
    return matches


def polygon_mask(path: Path, valid_width: int, valid_height: int) -> np.ndarray:
    """Rasterize YOLO segmentation; reject invalid polygons or padded annotations."""
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        prefix = f"{path.name}, row {number}"
        if values[0] not in {"0", "1"} or len(values) < 7 or len(values) % 2 == 0:
            raise ValueError(
                f"{prefix}: expected class 0/1 and at least 3 polygon points, not boxes."
            )
        points = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError(
                f"{prefix}: coordinates must be finite and between 0 and 1."
            )
        if (
            len(np.unique(points, axis=0)) < 3
            or cv2.contourArea(points.astype(np.float32)) <= 0
        ):
            raise ValueError(
                f"{prefix}: polygon needs three distinct points and positive area."
            )
        points *= TILE_SIZE
        if (points[:, 0] > valid_width + 1e-5).any() or (
            points[:, 1] > valid_height + 1e-5
        ).any():
            raise ValueError(f"{prefix}: polygon enters the zero-padding area.")
        # A vertex on the outer image boundary belongs to its last valid pixel.
        pixels = np.floor(points).astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, valid_width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, valid_height - 1)
        region = np.zeros_like(mask)
        cv2.fillPoly(region, [pixels], 1)
        value = int(values[0]) + 1
        if ((region > 0) & (mask > 0) & (mask != value)).any():
            raise ValueError(
                f"{prefix}: different classes overlap; review the polygons."
            )
        mask[region > 0] = value
    return mask


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Show blobs in cyan and streaks in orange."""
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    for value, color in [(1, [70, 220, 255]), (2, [255, 100, 70])]:
        selected = mask == value
        rgb[selected] = (0.4 * rgb[selected] + 0.6 * np.array(color)).astype(np.uint8)
    return rgb


def process(root: Path = ROOT) -> None:
    rows = load_metadata(root)
    labels = match_labels(rows, root)
    destinations = [
        root / p for p in ["annotations/masks", "outputs/overlays", "outputs/repatched"]
    ]
    for path in destinations:
        if path.exists() and any(path.iterdir()):
            raise ValueError(
                f"Preserve existing outputs elsewhere before rerunning: {path}"
            )
    # Check all labels and inputs before creating any final outputs.
    for row in rows:
        name = row["tile_file"]
        with Image.open(root / "tiles" / name) as image:
            if image.size != (TILE_SIZE, TILE_SIZE) or image.mode != "L":
                raise ValueError(f"Expected a 1024x1024 grayscale tile: {name}")
            array = np.asarray(image)
            if (
                array[row["valid_height"] :, :].any()
                or array[:, row["valid_width"] :].any()
            ):
                raise ValueError(f"Nonzero image padding: {name}")
        polygon_mask(labels[Path(name).stem], row["valid_width"], row["valid_height"])
    for path in destinations:
        path.mkdir(parents=True, exist_ok=True)
    print("Processing human annotations...", flush=True)
    for source in dict.fromkeys(r["source_file"] for r in rows):
        group = [r for r in rows if r["source_file"] == source]
        full_image = np.zeros(
            (group[0]["original_height"], group[0]["original_width"]), dtype=np.uint8
        )
        full_mask = np.zeros_like(full_image)
        for row in group:
            name = row["tile_file"]
            image = np.asarray(Image.open(root / "tiles" / name))
            mask = polygon_mask(
                labels[Path(name).stem], row["valid_width"], row["valid_height"]
            )
            Image.fromarray(mask).save(destinations[0] / name)
            Image.fromarray(overlay(image, mask)).save(destinations[1] / name)
            x, y, w, h = (row[k] for k in ["x", "y", "valid_width", "valid_height"])
            full_image[y : y + h, x : x + w] = image[:h, :w]
            full_mask[y : y + h, x : x + w] = mask[:h, :w]
        stem = Path(source).stem
        Image.fromarray(full_image).save(destinations[2] / f"{stem}_processed.png")
        Image.fromarray(full_mask).save(destinations[2] / f"{stem}_mask.png")
        Image.fromarray(overlay(full_image, full_mask)).save(
            destinations[2] / f"{stem}_overlay.png"
        )
        print(f"Reconstructed {source}", flush=True)
    print("Mask processing complete. Run validate.py and visually review the overlays.")


if __name__ == "__main__":
    try:
        process()
    except (OSError, ValueError, KeyError, csv.Error) as error:
        raise SystemExit(f"PROCESSING FAILED: {error}")
