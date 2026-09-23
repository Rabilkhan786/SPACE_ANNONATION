# Question 1 file-by-file correction log

This is a code/output audit and work record, not a Question 2 response.

| File or output | Finding | Correction / verification |
|---|---|---|
| `src/run_pipeline.py` | Local crop coordinates omitted offsets; independent tile detection truncated source shapes; no input-count guard; no working correction import. | Full-image detection, exact tiling, ten-file guard, explicit diagnostic mode, source/config-bound edit replay, and coherent re-export. |
| `src/image_ops.py` | Original polygon contours could be open or degenerate, and vertex striding changed shapes. Quantized data could make MAD zero. | Offset-aware closed polygons that also represent thin/one-pixel fragments; serialized polygons tested against OpenCV rasterization. Clipped-noise fallback and finite floor. |
| `config.yaml` | Background setting and README disagreed; detection was applied before meaningful input inspection. | Config corresponds to the revised code; the supplied image's actual statistics and crops were inspected. Thresholds remain candidate-generation parameters, not proven accuracy on unseen images. |
| `requirements.txt` | Loose dependency ranges; report-only dependency mixed into Q1. | Exact installed runtime-package versions; no report generator or report dependency. |
| `src/validate_outputs.py` | Previous validation checked only coordinate ranges, so misplaced polygons passed. | Read every image/mask/label; compare polygon rasters to class and parent-instance masks; compare raw tile values to FITS; verify padding, tile placement, dimensions and reconstruction. |
| `src/review_candidates.py` | Review CSV contained no linked visual evidence. | Local before/overlay crop sheets with IDs, shape ratios and SNR; review selection is disclosed. |
| `reviews/supplied_image.json` | Some star pairs were labelled as single streaks. | Six visually identified blends split into two star instances each; six coherent streaks retained. Edits are labelled assistant visual review, not human sign-off. |
| `images/`, `labels/`, `data.yaml` | Original directory layout would hide labels from Ultralytics; train/val reused identical data. | Matching sibling image/label directories; annotation metadata with no fictitious validation split. |
| `scientific_tiles/`, `valid_pixels/`, `tile_metadata.csv` | Only lossy display tiles existed; edge-replicated padding could create detections; tile manifest absent. | Lossless scientific-value tiles, explicit validity masks, source dimensions, exact offsets and right/bottom padding metadata. Detection excludes all synthetic padding. |
| `repatched/` | Overlay used different normalization from individual tiles. | One display scale per source; reconstruct from exported tiles and verify exact equality. |
| `README.md`, `SUBMISSION_NOTES.md` | Outdated paths, unsupported correction instructions and inflated completion wording. | Tested commands, current paths, supported edit operations and explicit remaining work. |
| `.gitignore` | Old generated paths and additional FITS extensions were incompletely covered. | All run folders, private review records, FITS extensions, generated arrays/images and ZIPs excluded. |

## Input evidence

The supplied file is a 9568 x 6380 uint16 FITS. Its physical values range from 0 to 1001, median is 1, and most background pixels are 0, 1 or 2. No non-finite pixels were found. These observations motivated checking quantization handling. Exact statistics and source hashes are retained inside the private run's `inspection/` folder.

## Limits of this correction

Final checked run: 70 image tiles and matching mask/label files; 5,119 exported polygon fragments; zero polygon-to-class-mask pixel differences; parent-instance checks passed; all scientific tile pixels match the FITS source; reconstructed dimensions are 9568 x 6380; source SHA-256 unchanged.

The corrected candidate catalogue contains 5,065 active objects (5,051 provisional star/blob and 14 provisional streak/object labels). Six original blended objects were replaced by twelve child star instances. Eighteen active objects have recorded assistant visual review; 5,047 remain pending. Counts are candidates, not established astronomical ground truth. Multiple polygon fragments can refer to one parent object, especially at tile boundaries.

Pixel-perfect export establishes structural consistency, not perfect astronomical labels. Low-SNR candidates, remaining ambiguous shapes, possible blends and missed sources still need review. The complete dataset is unavailable (one local input; company link returned HTTP 403). This package therefore remains a diagnostic draft for Question 1.
