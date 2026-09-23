# Digantara AI/ML Data Annotation Intern — Question 1

This repository contains a focused implementation for **Question 1 only** of the Digantara AI/ML Data Annotation Intern assessment.

## What Question 1 requires

The supplied real-sky FITS images must be:

1. inspected and preprocessed when required,
2. divided into exact **1024 × 1024** tiles without losing source pixels,
3. annotated with pixel-level masks for two classes:
   - **0 — star/blob**
   - **1 — streak/object**

The final annotation output is exported in **Ultralytics YOLO segmentation format**, and the full-size processed imagery is repatched for visual inspection.

## Approach

The implementation uses a small semi-automatic computer-vision pipeline:

FITS image → inspect statistics → local background/noise normalization → candidate detection → morphology measurement → provisional star/streak class → pixel mask → 1024×1024 tiles → YOLO polygons → visual QA → validation

Important design choices:

- Detection is performed on the **full image before tiling** so objects crossing tile boundaries are measured once.
- Images are **not resized**.
- Right/bottom edge tiles are zero-padded to 1024×1024, while metadata records the valid source area.
- Streak classification uses multiple morphology features: elongation, major-axis length, and eccentricity.
- Faint, ambiguous, large/blended, boundary, and streak candidates are flagged for manual review.
- Automatic annotations are treated as a **first-pass draft** until visual QA is complete.

## Project structure

~~~text
SPACE_ANNONATION/
├── config.yaml
├── requirements.txt
├── README.md
└── src/
    ├── image_ops.py
    ├── run_pipeline.py
    ├── review_candidates.py
    └── validate_outputs.py
~~~

Generated data, FITS images, run folders, review artifacts, and ZIP files are intentionally excluded from Git because the assessment material is confidential.

## Setup

### 1. Create a virtual environment

~~~bat
python -m venv .venv
.venv\Scripts\activate
~~~

### 2. Install dependencies

~~~bat
python -m pip install -r requirements.txt
~~~

## Commands

### Inspect the FITS images first

This saves image statistics, an overview, and representative crops without creating annotations.

~~~bat
python src\run_pipeline.py --input "D:\path\to\Datasets_Assessment" --output "runs\inspection" --inspect-only
~~~

### Run Question 1 annotation pipeline

For the final assessment run, the input folder should contain all **10 supplied FITS images**.

~~~bat
python src\run_pipeline.py --input "D:\path\to\Datasets_Assessment" --output "runs\question1"
~~~

### Create visual review sheets

~~~bat
python src\review_candidates.py --run "runs\question1"
~~~

### Validate the exported dataset

~~~bat
python src\validate_outputs.py --run "runs\question1"
~~~

## Output structure

~~~text
runs/question1/
├── inventory.json
├── tile_metadata.csv
├── data.yaml
├── inspection/
├── images/       # 1024×1024 processed tiles
├── labels/       # YOLO segmentation text files
├── masks/        # pixel-level class masks
├── overlays/     # tile-level visual QA overlays
├── repatched/    # full-size processed image, mask, overlay
└── qa/
    ├── candidates.csv
    ├── run_summary.json
    ├── review_sheets/
    └── validation.json
~~~

## Class convention

YOLO class IDs:

~~~text
0 = star_blob
1 = streak_object
~~~

Raster mask values:

~~~text
0 = background
1 = star_blob
2 = streak_object
~~~

## Manual QA

The code intentionally flags uncertain cases instead of pretending every automatic decision is ground truth. Review at least:

- faint/small sources,
- short or thick streak candidates,
- ambiguous shapes,
- large/blended objects,
- objects crossing tile boundaries,
- possible false positives and missed objects.

If a UI annotation tool such as CVAT or Roboflow is used for final correction, the assessment asks for a short screen recording of that annotation workflow.

## Scope

This repository is for **Question 1 only**. Question 2 should be written later from the actual final implementation and observed results, not from generic assumptions.

## Confidentiality

Do not commit or publicly share Digantara-provided FITS images, generated assessment outputs, or private submission archives. Keep the repository private when it contains assessment-related material.
