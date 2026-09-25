"""Convert reviewed YOLO polygons into masks, overlays and full-size outputs."""

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from prepare_tiles import FIELDS, ROOT, TILE_SIZE


def load_metadata(root: Path = ROOT) -> list[dict]:
    """Read tile metadata and verify the source grids."""
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

        name = row["tile_file"]
        source = row["source_file"]
        x, y = row["x"], row["y"]
        width = row["original_width"]
        height = row["original_height"]

        valid_width = min(TILE_SIZE, width - x)
        valid_height = min(TILE_SIZE, height - y)

        if (
            Path(name).name != name
            or Path(source).name != source
            or name in names
            or name != f"{Path(source).stem}_x{x:04d}_y{y:04d}.png"
            or x < 0
            or y < 0
            or x >= width
            or y >= height
            or x % TILE_SIZE
            or y % TILE_SIZE
            or row["valid_width"] != valid_width
            or row["valid_height"] != valid_height
        ):
            raise ValueError(f"Invalid tile metadata: {name}")

        names.add(name)
        sources.setdefault(source, []).append(row)

    for source, group in sources.items():
        width = group[0]["original_width"]
        height = group[0]["original_height"]
        expected = ((width + TILE_SIZE - 1) // TILE_SIZE) * (
            (height + TILE_SIZE - 1) // TILE_SIZE
        )
        if len(group) != expected:
            raise ValueError(f"Incomplete tile grid: {source}")

    actual = {path.name for path in (root / "tiles").glob("*.png")}
    if actual != names:
        raise ValueError("PNG tiles do not match tile_metadata.csv.")

    return rows


def find_labels(rows: list[dict], root: Path = ROOT) -> dict[str, Path]:
    """Find YOLO labels. Missing TXT files are treated as reviewed empty tiles."""
    expected = {Path(row["tile_file"]).stem for row in rows}
    labels_dir = root / "annotations/labels"

    if not labels_dir.exists():
        raise ValueError("annotations/labels does not exist.")

    labels = {}

    for path in sorted(labels_dir.rglob("*.txt")):
        stem = path.stem

        if stem not in expected:
            stem = stem.split("_png.rf.", 1)[0]

        if stem not in expected:
            raise ValueError(f"Label has no matching tile: {path.name}")

        if stem in labels:
            raise ValueError(f"Duplicate label file for tile: {stem}")

        labels[stem] = path

    return labels


def polygon_mask(
    label_path: Path | None,
    valid_width: int,
    valid_height: int,
) -> np.ndarray:
    """Convert one YOLO segmentation label file into a class mask."""
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

    if label_path is None:
        return mask

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(),
        1,
    ):
        values = line.split()

        if not values:
            continue

        prefix = f"{label_path.name}, row {line_number}"

        if values[0] not in {"0", "1"}:
            raise ValueError(f"{prefix}: class must be 0 or 1.")

        if len(values) < 7 or len(values) % 2 == 0:
            raise ValueError(f"{prefix}: expected a segmentation polygon.")

        points = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)

        if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError(f"{prefix}: coordinates must be between 0 and 1.")

        if len(np.unique(points, axis=0)) < 3:
            raise ValueError(f"{prefix}: polygon needs at least three distinct points.")

        points *= TILE_SIZE

        if (points[:, 0] > valid_width + 1e-5).any() or (
            points[:, 1] > valid_height + 1e-5
        ).any():
            raise ValueError(f"{prefix}: polygon enters the padding area.")

        pixels = np.floor(points).astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, valid_width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, valid_height - 1)

        if cv2.contourArea(pixels.astype(np.float32)) <= 0:
            raise ValueError(f"{prefix}: polygon has no area.")

        region = np.zeros_like(mask)
        cv2.fillPoly(region, [pixels], 1)

        mask[region > 0] = int(values[0]) + 1

    return mask


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a simple RGB annotation overlay."""
    rgb = np.repeat(image[:, :, None], 3, axis=2)

    for value, color in [(1, [70, 220, 255]), (2, [255, 100, 70])]:
        selected = mask == value
        rgb[selected] = (
            0.4 * rgb[selected] + 0.6 * np.asarray(color)
        ).astype(np.uint8)

    return rgb


def process(root: Path = ROOT) -> None:
    """Create tile masks/overlays and repatch every source image."""
    rows = load_metadata(root)
    labels = find_labels(rows, root)

    masks_dir = root / "annotations/masks"
    overlays_dir = root / "outputs/overlays"
    repatched_dir = root / "outputs/repatched"

    for path in [masks_dir, overlays_dir, repatched_dir]:
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"Output already exists: {path}")

    for path in [masks_dir, overlays_dir, repatched_dir]:
        path.mkdir(parents=True, exist_ok=True)

    print(
        f"Found {len(labels)} label files for {len(rows)} tiles. "
        "Tiles without a TXT file are treated as empty annotations."
    )

    sources = dict.fromkeys(row["source_file"] for row in rows)

    for source in sources:
        group = [row for row in rows if row["source_file"] == source]
        height = group[0]["original_height"]
        width = group[0]["original_width"]

        full_image = np.zeros((height, width), dtype=np.uint8)
        full_mask = np.zeros((height, width), dtype=np.uint8)

        for row in group:
            name = row["tile_file"]
            stem = Path(name).stem

            with Image.open(root / "tiles" / name) as image_file:
                image = np.asarray(image_file.convert("L"))

            if image.shape != (TILE_SIZE, TILE_SIZE):
                raise ValueError(f"Tile must be 1024x1024: {name}")

            mask = polygon_mask(
                labels.get(stem),
                row["valid_width"],
                row["valid_height"],
            )

            Image.fromarray(mask).save(masks_dir / name)
            Image.fromarray(overlay(image, mask)).save(overlays_dir / name)

            x = row["x"]
            y = row["y"]
            valid_width = row["valid_width"]
            valid_height = row["valid_height"]

            full_image[
                y : y + valid_height,
                x : x + valid_width,
            ] = image[:valid_height, :valid_width]

            full_mask[
                y : y + valid_height,
                x : x + valid_width,
            ] = mask[:valid_height, :valid_width]

        stem = Path(source).stem

        Image.fromarray(full_image).save(
            repatched_dir / f"{stem}_processed.png"
        )
        Image.fromarray(full_mask).save(
            repatched_dir / f"{stem}_mask.png"
        )
        Image.fromarray(overlay(full_image, full_mask)).save(
            repatched_dir / f"{stem}_overlay.png"
        )

        print(f"Reconstructed {source}")

    print("Annotation processing complete.")


if __name__ == "__main__":
    try:
        process()
    except (OSError, ValueError, KeyError, csv.Error) as error:
        raise SystemExit(f"PROCESSING FAILED: {error}")
