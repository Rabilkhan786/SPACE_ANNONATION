# Digantara AI/ML Data Annotation Intern Assessment

This repository covers the assessment workflow for annotating astronomical FITS images.

Two classes are used:

- `star_blob` — compact, point-like or blob-like source
- `streak_object` — elongated, continuous source

The solution is intentionally simple: Python prepares the FITS data, Roboflow is used for manual instance-segmentation annotation, and Python converts/validates the reviewed annotations.

## Assessment workflow

```text
10 FITS images
    ↓
display preprocessing
    ↓
1024×1024 PNG tiles
    ↓
Roboflow manual instance segmentation
    ↓
YOLO segmentation export
    ↓
pixel masks + overlays
    ↓
full-size reconstruction
    ↓
validation
```

### A1 — Inspect and preprocess

`src/prepare_tiles.py` reads the first 2D FITS image plane and creates an 8-bit display image using percentile clipping and an arcsinh stretch.

This is only for visualization and annotation. The original FITS files are never modified.

### A2 — Create 1024×1024 tiles

The display image is split without resizing or cropping valid pixels.

For a 9568×6380 image:

- right padding: 672 pixels
- bottom padding: 788 pixels
- padded size: 10240×7168
- tiles: 10 × 7 = 70

Padding is added only outside the original image. `tile_metadata.csv` stores each tile position and valid size so the final result can be reconstructed correctly.

### A3 — Annotate two classes

Create a **Roboflow Instance Segmentation** project and use:

| Class | YOLO ID | Raster mask |
|---|---:|---:|
| background | — | 0 |
| star_blob | 0 | 1 |
| streak_object | 1 | 2 |

Annotation rules:

- Draw a separate polygon around each clearly visible compact star/blob.
- Draw a tight polygon around each clearly elongated continuous streak.
- Do not label random noise.
- Do not merge nearby stars into one streak.
- Annotate faint sources only when they can be distinguished from the background.
- At tile edges, annotate only the visible part of the object.
- Never annotate the black padding area.

After review, export **YOLOv8/Ultralytics instance segmentation** labels from Roboflow.

## Project files

```text
src/
  prepare_tiles.py
  process_annotations.py
  validate.py

data/raw/              # 10 FITS files, ignored by Git
tiles/                 # generated 1024×1024 PNG tiles
annotations/labels/    # reviewed YOLO segmentation labels
annotations/masks/     # generated class masks
outputs/overlays/      # tile overlays
outputs/repatched/     # full-size image/mask/overlay
tile_metadata.csv
requirements.txt
README.md
```

Generated data is ignored by Git.

## Run

Install dependencies:

```bat
python -m pip install -r requirements.txt
```

Prepare the annotation tiles:

```bat
python src\prepare_tiles.py
```

Upload only the PNG files from `tiles/` to Roboflow and complete manual review.

Place the exported YOLO segmentation TXT files in:

```text
annotations/labels/
```

Then process the annotations:

```bat
python src\process_annotations.py
```

Finally validate:

```bat
python src\validate.py
```

A successful final check prints:

```text
VALIDATION PASSED
```

Validation checks files, tile geometry, polygon coordinates, mask values, padding and reconstruction. It does **not** replace human visual QA.

## Written reasoning

### 1. Why preprocessing is used

The FITS images have different intensity ranges. A simple adaptive display stretch makes faint and bright sources easier to inspect while leaving the original FITS values unchanged. One fixed raw-intensity threshold is not used.

### 2. How boundary tiles avoid data loss

The image is never resized and valid pixels are never cropped. The right and bottom edges are zero-padded to the next multiple of 1024. For 9568×6380, this gives 10240×7168 and 70 tiles. Metadata records the valid part of every tile, and padding is removed during reconstruction.

### 3. Short/thick streak versus blob

A blob is compact and approximately point-like. A streak is visibly elongated and continuous. Short or thick ambiguous objects should be judged using shape and visual context instead of relying on one automatic threshold.

### 4. Faint/small stars

A faint source is annotated only when it is distinguishable from the surrounding noise and has a compact star-like appearance. Uncertain objects remain a manual-review decision.

## Confidentiality

The assessment data is confidential. Keep the GitHub repository and Roboflow project private, do not commit raw FITS files, and do not upload the raw FITS files to Roboflow.
