"""Create local raw/overlay crop sheets for prioritized visual annotation QA."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image_ops import overlay


def build(root, source=None):
    rows = list(csv.DictReader((root / "qa" / "candidates.csv").open(encoding="utf-8")))
    if source is None:
        for source_name in sorted({r["source"] for r in rows}):
            build(root, source_name)
        return
    rows = [
        r for r in rows if r["source"] == source and r["review_status"] != "deleted"
    ]
    groups = {
        "reviewed": [r for r in rows if r["review_status"] == "visually_reviewed"],
        "streaks": [
            r for r in rows if r["class_id"] == "1" and r["review_status"] != "deleted"
        ],
        "ambiguous": sorted(
            [r for r in rows if "ambiguous_shape" in r["review_reason"]],
            key=lambda r: float(r["elongation"]),
            reverse=True,
        )[:36],
        "faint": sorted(
            [r for r in rows if "faint_source" in r["review_reason"]],
            key=lambda r: float(r["peak_snr"]),
        )[:24],
        "boundaries": [
            r
            for r in rows
            if "tile_crossing" in r["review_reason"]
            or "source_edge" in r["review_reason"]
        ],
        "bright": sorted(
            [r for r in rows if r["class_id"] == "0"],
            key=lambda r: float(r["peak_intensity"]),
            reverse=True,
        )[:12],
    }
    folder = root / "qa" / "review_sheets" / Path(source).stem
    folder.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for group, selected in groups.items():
        for start in range(0, len(selected), 12):
            page = Image.new("RGB", (1200, 900), "#17202a")
            draw = ImageDraw.Draw(page)
            for i, row in enumerate(selected[start : start + 12]):
                stem = Path(row["source"]).stem
                if stem not in loaded:
                    loaded[stem] = (
                        Image.open(root / "repatched" / f"{stem}_processed.png"),
                        Image.open(root / "repatched" / f"{stem}_mask.png"),
                    )
                gray, mask = loaded[stem]
                x0, y0, x1, y1 = [int(row[k]) for k in ["x0", "y0", "x1", "y1"]]
                margin = 10
                side = max(40, x1 - x0 + 2 * margin, y1 - y0 + 2 * margin)
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2
                box = (
                    cx - side // 2,
                    cy - side // 2,
                    cx - side // 2 + side,
                    cy - side // 2 + side,
                )
                a = np.array(gray.crop(box))
                b = np.array(mask.crop(box))
                pair = Image.new("RGB", (384, 192))
                pair.paste(
                    Image.fromarray(a)
                    .convert("RGB")
                    .resize((192, 192), Image.Resampling.NEAREST),
                    (0, 0),
                )
                pair.paste(
                    Image.fromarray(overlay(a, b)).resize(
                        (192, 192), Image.Resampling.NEAREST
                    ),
                    (192, 0),
                )
                left = i % 3 * 400
                top = i // 3 * 225
                draw.text(
                    (left + 5, top + 3),
                    f"ID {row['object_id']} class {row['class_id']} ratio {float(row['elongation']):.1f} peak SNR {float(row['peak_snr']):.1f}",
                    fill="white",
                )
                page.paste(pair, (left + 5, top + 25))
            page.save(folder / f"{group}_{start//12+1:02d}.png")
    print(source, {k: len(v) for k, v in groups.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--source", help="Optional source FITS filename")
    args = parser.parse_args()
    build(args.run, args.source)
