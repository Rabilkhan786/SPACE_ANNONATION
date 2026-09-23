# Question 1 handoff status

Use `run_question1_corrected/`; previous runs and ZIPs were superseded. This package contains source code, configuration, reproducible review edits, images, pixel masks, YOLO labels, scientific tiles, tile metadata, reconstructed imagery and annotation QA records. Automated test files/results, Question 2 and any PDF response are excluded.

The technical checks compare every saved tile to the original FITS, rasterize the actual saved labels back to masks, check parent instances, reject padding annotations, and compare reconstructed masks/images with their tiles. See `qa/validation.json` for results, not merely a successful script exit.

The assessment is not complete: the brief requires ten images, only one was accessible here, and most annotations remain pending visual review. The dataset link returned HTTP 403. Before submission, provide/access the remaining nine images, inspect them, process all ten, resolve uncertain masks/classes and missed sources, and validate the final run. These are Question 1 tasks; Question 2 remains deferred.

Do not send an older ZIP: its polygons had incorrect crop offsets and its data layout was wrong. Superseded artifacts remain recoverable outside the active deliverable folder.
