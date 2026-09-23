"""Create visual review sheets for Question 1 annotation QA.

The pipeline flags faint, ambiguous, large/blended, boundary, and streak
candidates. This script creates contact sheets so those cases can be checked
quickly before annotations are treated as final.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image_ops import overlay_mask


def load_candidates(run_dir: Path) -> list[dict[str, str]]:
    """Load the candidate table produced by run_pipeline.py."""
    csv_path = run_dir / "qa" / "candidates.csv"
    with csv_path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def crop_pair(
    processed: Image.Image,
    mask: Image.Image,
    row: dict[str, str],
    crop_size: int = 128,
) -> Image.Image:
    """Return a side-by-side source/overlay crop around one candidate."""
    x0, y0, x1, y1 = [
        int(row[key])
        for key in ("x0", "y0", "x1", "y1")
    ]
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    half = crop_size // 2

    box = (
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
    )

    gray = np.asarray(processed.crop(box))
    class_mask = np.asarray(mask.crop(box))
    overlay = overlay_mask(gray, class_mask)

    pair = Image.new(
        "RGB",
        (crop_size * 2, crop_size),
    )
    pair.paste(
        Image.fromarray(gray).convert("RGB"),
        (0, 0),
    )
    pair.paste(
        Image.fromarray(overlay),
        (crop_size, 0),
    )
    return pair


def make_group_sheets(
    run_dir: Path,
    source: str,
    group_name: str,
    rows: list[dict[str, str]],
) -> None:
    """Write contact sheets containing up to 12 candidates per page."""
    if not rows:
        return

    stem = Path(source).stem
    processed = Image.open(
        run_dir / "repatched" / f"{stem}_processed.png"
    )
    mask = Image.open(
        run_dir / "repatched" / f"{stem}_mask.png"
    )

    output_dir = (
        run_dir
        / "qa"
        / "review_sheets"
        / stem
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_per_page = 12

    for start in range(
        0,
        len(rows),
        rows_per_page,
    ):
        selected = rows[
            start : start + rows_per_page
        ]
        page = Image.new(
            "RGB",
            (1200, 920),
            "#17202a",
        )
        draw = ImageDraw.Draw(page)

        for index, row in enumerate(selected):
            pair = crop_pair(
                processed,
                mask,
                row,
            )
            pair = pair.resize(
                (360, 180),
                Image.Resampling.NEAREST,
            )

            column = index % 3
            page_row = index // 3
            left = column * 400 + 20
            top = page_row * 225 + 35

            label = (
                f"ID {row['object_id']} | "
                f"class {row['class_id']} | "
                f"elong {float(row['elongation']):.2f} | "
                f"SNR {float(row['peak_snr']):.1f}"
            )
            draw.text(
                (left, top - 22),
                label,
                fill="white",
            )
            page.paste(
                pair,
                (left, top),
            )

        page_number = (
            start // rows_per_page + 1
        )
        page.save(
            output_dir
            / f"{group_name}_{page_number:02d}.png"
        )


def build_review_sheets(run_dir: Path) -> None:
    """Create prioritized review sheets for each source image."""
    rows = load_candidates(run_dir)
    sources = sorted(
        {row["source"] for row in rows}
    )

    for source in sources:
        source_rows = [
            row
            for row in rows
            if row["source"] == source
        ]

        groups = {
            "priority": [
                row
                for row in source_rows
                if row["needs_review"].lower()
                == "true"
            ],
            "streaks": [
                row
                for row in source_rows
                if row["class_id"] == "1"
            ],
            "ambiguous": [
                row
                for row in source_rows
                if "ambiguous_shape"
                in row["review_reason"]
            ],
            "faint": [
                row
                for row in source_rows
                if "faint_source"
                in row["review_reason"]
            ],
            "boundaries": [
                row
                for row in source_rows
                if (
                    "tile_crossing"
                    in row["review_reason"]
                    or "source_edge"
                    in row["review_reason"]
                )
            ],
        }

        for group_name, group_rows in groups.items():
            make_group_sheets(
                run_dir,
                source,
                group_name,
                group_rows,
            )

        print(
            source,
            {
                name: len(group_rows)
                for name, group_rows
                in groups.items()
            },
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help=(
            "Output folder produced by "
            "run_pipeline.py."
        ),
    )
    build_review_sheets(
        parser.parse_args().run
    )
