# Digantara assessment — Question 1

Identify compact stars (`star_blob`) and elongated sources (`streak_object`) in
10 astronomical FITS images. This is a **Python data-preparation + Roboflow
auto-label proposals + review/correction + validation workflow**. No model
training or Question 2 is included.

## Current status — 25 September 2026

All 10 FITS files were processed into 700 PNG tiles. Preparation checks passed
for every tile, including metadata, dimensions, exact display pixels and padding.
The original FITS checksums are unchanged. Representative tiles were inspected
visually. The three scripts compile and the installed dependencies are compatible.

Roboflow MCP is connected. After private-project creation was rejected by this
workspace, the user explicitly chose a public project for these sample image
tiles and approved the `BY-NC-SA 4.0` license.
Project: [Sky Annotation Q1](https://app.roboflow.com/ahammed-rabil/sky-annotation-q1),
type **Instance Segmentation**, visibility **public**. Upload was verified:
**700 PNG tiles imported, 0 duplicates, 0 upload errors, 0 annotations**.
The user subsequently requested Auto Label and Review. A two-tile pilot completed
successfully with **171 draft polygons: 170 star blobs and 1 streak**. Its job is
in the **Review** queue with **0 approved**; 698 tiles remain unassigned and the
approved Dataset count is 0. Initial overlay inspection is not final QA.
Raw FITS and old automatic labels were not uploaded. Full-batch annotation,
detailed correction, export, mask processing and final validation remain pending.

Pilot: [Auto-label review job](https://app.roboflow.com/ahammed-rabil/sky-annotation-q1/annotate/job/D0c6FKE4qLouH2qcBixd?startReviewing=1).
Auto-label task ID: `7cfe80c6-44c2-4d42-b5ab-8b1c3a03b488`.
The batch run reported 2 labeled images, 171 annotations and 0 resized images.
The Auto Label job's Review screen was opened and both overlays inspected.
This is preliminary agent inspection, not completed human QA or approval.
The standalone team-wide "Enable Review Mode" control is plan-locked, but
the auto-label job itself exposes working Approve/Reject review controls.
Scaling is paused pending a spending limit and a confirmed full-run estimate:
the UI shows a 15-credit allowance, with recent usage potentially delayed.
No extra credits were purchased and no subscription was changed.

## Why Roboflow?

The old heuristic pipeline produced many uncertain detections. Old automatic
labels are not imported or treated as ground truth. At the user's request,
Roboflow Auto Label now generates editable proposals for review and correction.
On a representative tile, SAM 3 with descriptive prompts found no objects;
short prompts at confidence 0.1 found only one star and one streak. The
`gpt-6-astra-sam3-polygons` preview found 59 stars and one streak, so it was used
for the two-image pilot with classes `star_blob` and `streak_object`. This hosted
vision-model use is limited to the requested auto-label step, not a new local
LLM application. Preview counts differ from the subsequent batch predictions.
Neither model confidence nor a successful job establishes annotation accuracy.
Check the workspace's current cost estimate before scaling; do not purchase
credits or upgrade a plan automatically.

## Workflow and files

FITS → display stretch → 1024×1024 PNG tiles → Roboflow auto-label proposals →
review and manual correction →
YOLO segmentation export → pixel masks and overlays → full-size reconstruction →
validation.

| File/folder | Purpose |
|---|---|
| `data/raw/` | The 10 original, confidential FITS files |
| `src/prepare_tiles.py` | Read FITS, improve display contrast and create tiles |
| `tiles/` | PNG tiles for auto-labeling and visual review |
| `tile_metadata.csv` | Source, tile, coordinates and valid dimensions |
| `annotations/labels/` | Human-reviewed YOLO segmentation TXT files |
| `src/process_annotations.py` | Convert polygons to masks, overlays and full images |
| `annotations/masks/` | One class mask per tile |
| `outputs/overlays/` | Tile images with colored annotations |
| `outputs/repatched/` | Full-size display images, masks and overlays |
| `src/validate.py` | Check files, coordinates, padding and reconstruction |

## Prepare

Run from this project folder with Python 3.10+ (the existing `.venv` uses 3.14).
Activate the existing environment first: `.venv\Scripts\activate` in Windows CMD.

```bat
python -m pip install -r requirements.txt
python src\prepare_tiles.py
```

The script reads the first 2D image HDU in each FITS file. Each image's 1st and
99.99th percentiles plus an arcsinh stretch produce an 8-bit grayscale display.
This adapts to different brightness ranges; it is not scientific calibration or
a lossless conversion of the original intensity values. FITS files stay unchanged.

Tiles are never resized and no valid spatial pixels are cropped. Zero padding is
added only on the right/bottom. A 9568×6380 image gets 672 right and 788 bottom
padding pixels, giving 70 tiles. All 10 supplied images give 700 tiles.
Counts and padding are calculated from actual dimensions.

Existing tiles/metadata are not overwritten. Preserve them elsewhere before
rerunning preparation. The same safety rule applies to existing processed masks.

## Annotate in Roboflow

This run uses the **public Instance Segmentation** project explicitly approved by
the user above. Upload only `tiles/*.png`, never raw FITS or the old automatic
labels. Public visibility applies to the uploaded tiles; do not assume they are
private. This approval does not cover other company materials or future datasets.
For confidential material without explicit public-sharing approval, use a private
destination and confirm its visibility before uploading.

| Class | YOLO ID | Raster mask value |
|---|---:|---:|
| Background | No polygon | 0 |
| star_blob | 0 | 1 |
| streak_object | 1 | 2 |

- Draw a separate polygon around each clearly visible compact source.
- Draw tight polygons around clearly elongated, continuous streaks.
- Do not turn neighboring stars into one streak or label random noise.
- Annotate faint sources only when distinguishable from background; leave
  uncertain cases for human review.
- At tile edges, annotate the visible portion in each neighboring tile.
- Do not annotate black padding. Metadata gives the valid width/height.

Zoom in and review polygon boundaries. Cyan overlays show blobs; orange shows
streaks. Human review, not a successful script run, establishes annotation quality.

The connected MCP supports project inspection/creation, image-upload preparation,
auto-label previews and jobs, review status, saving supplied annotations and export. These
operations do not replace visual judgment. There is no dedicated class-creation
action in the inspected tools; use Project Settings → Classes when needed.
Never create fake annotations just to create class names.

## Export, process and validate

After all tiles are annotated and reviewed, export **YOLOv8 instance segmentation
(Ultralytics polygons)**, not bounding boxes. Include all original images,
including reviewed empty tiles. Disable resizing, cropping, tiling, augmentation
and filtering in the exported version so coordinates still match these tiles.

Check the export's `data.yaml`: ID 0 must be `star_blob` and ID 1 must be
`streak_object`. TXT rows alone cannot verify the meaning of a numeric ID.
Keep the export ZIP and YAML for this check; do not guess or silently swap IDs.

Put TXT files from all export splits into `annotations/labels/` (subfolders are
accepted). Filenames must match PNG stems; the common `_png.rf.<hash>` suffix is
also accepted. Unexpected or duplicate names cause an error. Each reviewed empty
tile needs an empty TXT file. Missing files are **not** assumed to mean background;
confirm empty tiles manually if the export omits their files.

```bat
python src\process_annotations.py
python src\validate.py
```

Each TXT row is `class_id x1 y1 x2 y2 x3 y3 ...`, with coordinates in [0,1].
Polygons are filled as pixel masks without changing the labels. A vertex on the
outer image boundary maps to the last valid pixel. Different-class overlaps and
polygons entering padding are rejected for review. Class masks combine same-class
overlaps; separate instances remain in the original TXT polygons. Reconstruction
removes padding using the metadata.

Validation checks all 10 sources, tile counts/sizes, original display pixels,
class IDs, polygon points, mask values, padding, polygon/mask agreement, overlays
and exact reconstruction. It prints `VALIDATION PASSED` or a clear
`VALIDATION FAILED` message. It does not measure annotation accuracy.

Until the reviewed export exists, mask processing and final validation are pending.
No blank labels or old automatic annotations are generated to bypass this step.

## Confidentiality and previous work

Raw data, tiles, labels, outputs, metadata, ZIPs and credentials are ignored by Git.
Share only with authorized reviewers. `.gitignore` does not make a cloud project
private; check Roboflow privacy separately before upload.

The old scripts, runs and private ZIP were preserved outside this project at
`../../work/automatic_pipeline_backup_20260925/`. They are reference only.
