# Digantara assessment: Question 1

This project implements A1 (inspect and preprocess), A2 (exact 1024 x 1024 tiles), and A3 (two-class pixel masks). Question 2 and its report are excluded.

## Start here

`run_question1_corrected/` is the current **diagnostic draft for one supplied image**. Its technical integrity is checked; it is not a completed ten-image assessment. Most candidate annotations still require review. The full dataset link returned HTTP 403, and only one FITS image was available locally.

Read `FILE_AUDIT.md` for the corrections made. Inspect `run_question1_corrected/qa/validation.json` for the actual file checks and `qa/candidates.csv` for review status. Source FITS files are unchanged.

## Environment and commands

Use Python 3.12 on Windows. The dependencies are CPU-only and pinned in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Inspect inputs without generating annotations:

```powershell
.\.venv\Scripts\python src\run_pipeline.py --input "C:\path\to\Datasets_Assessment" --output run_inspection --inspect-only
```

Generate a full-dataset draft after all ten FITS files are available. The command fails before processing if the count is not ten:

```powershell
.\.venv\Scripts\python src\run_pipeline.py --input "C:\path\to\Datasets_Assessment" --output run_all_images
```

Reproduce the current single-image diagnostic and recorded visual corrections:

```powershell
.\.venv\Scripts\python src\run_pipeline.py --input "C:\path\to\1a600998-b97c-4307-8632-6fcf573621ff.fits" --output run_reproduced --diagnostic --corrections reviews\supplied_image.json
.\.venv\Scripts\python src\validate_outputs.py --run run_reproduced --source-folder "C:\path\to"
.\.venv\Scripts\python src\review_candidates.py --run run_reproduced
```

Always use a new output folder. The pipeline refuses to overwrite an existing run. `--diagnostic` explicitly permits an incomplete dataset for a preliminary run; it never marks the assessment complete.

## Files

```text
src/
  image_ops.py          FITS, statistics, masks and polygon helpers
  run_pipeline.py       inspect, detect, apply edits, tile and repatch
  validate_outputs.py   independently check every saved tile and annotation
  review_candidates.py  create local raw/overlay comparison sheets
reviews/
  supplied_image.json   recorded assistant visual edits, tied to source hash/config
run_question1_corrected/
  inventory.json       available input count and diagnostic status
  inspection/          source statistics, overview and native-pixel crops
  images/              exact 1024 x 1024 display PNGs
  labels/              same-stem YOLO segmentation text files
  masks/               per-tile class masks
  instances/           original-size instance-ID arrays
  scientific_tiles/    original FITS values/dtype, padded without resampling
  valid_pixels/        original pixels=255, added padding=0
  overlays/            per-tile inspection overlays
  repatched/           original-size processed image, class mask and overlay
  tile_metadata.csv    source, row/column, offsets, dimensions and padding
  data.yaml            annotation dataset paths and class names
  qa/                  candidate measurements, parent IDs, checks, review sheets
```

YOLO IDs: `0 = star_blob`, `1 = streak_object`. Raster class masks: `0 = background`, `1 = star_blob`, `2 = streak_object`. Instance arrays hold stable object IDs, with zero for background. Split objects have new child IDs and keep their deleted parent in the audit CSV.

`data.yaml` locates the current run by absolute path. If moving a run, update its `path` to the extracted run folder. `val: null` is intentional: no training/validation split or model training is claimed. PNG images and label folders are siblings as required by the [Ultralytics segmentation format](https://docs.ultralytics.com/datasets/segment/).

## Review and correction

Review sheets pair the unannotated image on the left with the overlay on the right. Cyan marks blob/star masks; orange marks streak masks. They cover all initial streak candidates and boundary cases plus selected bright, faint and ambiguous examples; they do not certify every object or detection recall. The recorded visual review was performed by the assistant, not by an independent human annotator.

`reviews/supplied_image.json` shows the edit format. Each review file must contain the exact source SHA-256 and detection config. Supported actions are `accept`, `delete`, `reclassify`, `replace`, `add`, and `split`; every action requires a note. Replacement/addition polygons and split seeds use full-source pixel coordinates. Class-changing actions require `class_id`. Replacement polygons must not overlap another instance. For multiple sources, pass a folder with files named `<source_stem>.json` to `--corrections`.

Corrected masks, measurements, labels and overlays are regenerated together. Visual review status is separate from technical checks. Keep ambiguous detections pending until resolved; do not describe the present draft labels as final ground truth.

## Confidentiality

Keep this package and all imagery private. Raw files, run folders, review records and ZIP archives are ignored by Git. Only source, configuration and non-sensitive documentation should be shared through an authorized private repository. No remote repository was created or published.
