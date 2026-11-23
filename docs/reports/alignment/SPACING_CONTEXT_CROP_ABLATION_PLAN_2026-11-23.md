# Spacing Context-Crop Ablation Plan - 23 November 2026

## Hypothesis

AI WideTech v2 contains useful visible-gap instances, but many exported canvases combine multiple photos. A detector trained on the full canvas sees unrelated panels while the annotation identifies only one local gap. This experiment restricts the field of view to a local window around the largest target box while keeping pixel scale and box width/height unchanged. It tests whether reduced off-target context helps the existing spacing task. The crop does not guarantee that every neighboring panel or label inconsistency is removed; the visual pilot still contains split views and letterboxing.

This is a train-time data transformation, not a new clinical label, a paper reproduction, or a guaranteed gain.

## Fixed data and transformation

| Item | Locked choice |
|---|---|
| Source | AI WideTech Dental Disease v2, CC BY 4.0 |
| Original source SHA-256 | `5f4d9a6c974b6a43521119b1edd6f3c17cf67c24aa62e1eebb6c5b0b9b95521a` |
| Candidate train package SHA-256 | `becfefe26a9fd7080d86048430897aee38eac30f3740272110c3bf7836f4a299` |
| Source content candidates | 572 train images, 917 diastema boxes after duplicate/overlap/conflict exclusions |
| Crop | One 320 x 320 context crop around the largest target box per source image |
| Canvas | Paste crop at native pixel scale in the center of a black 640 x 640 canvas |
| Box transform | Translation only; preserve width and height; retain only fully visible boxes |
| Quarantine | Exclude a crop if another target box would be partially visible; 13 excluded in local smoke |
| Raw context crops before crop-level deduplication | 559 images / 690 boxes |
| Crop-level repeated panels | 7 visually reviewed mirror-invariant pHash pairs at distance 0 or 2, plus 16 confirmed same-photo variants from 31 pHash-distance 3-8 candidates; the latter also pass SIFT/pixel alignment checks |
| Final train-only add-on | 536 unique crops / 662 boxes after removing 23 repeated-photo variants; 32 source-box instances and 28 crop-box instances removed with those copies |
| Other boxes omitted by the crop | Before crop deduplication, 184 of 874 boxes on retained source images fall fully outside the chosen crop. After deduplication, 842 source boxes remain; 180 lie outside the selected crop, and the 536 training crops contain 662 supervised boxes |
| Source valid/test | Neither partition is attached to Kaggle; no source test payload was opened |

The crop is not resized or zoomed. The transformed labels remain a geometric translation of the source labels. After trimming black padding, mirror-invariant pHash surfaces candidates; seven <=2-distance pairs and 31 distance-3-to-8 pairs were visually inspected. Only 16 of the latter also passed SIFT RANSAC (at least 30 inliers, at least 0.80 inlier ratio) and aligned grayscale correlation (at least 0.98). The remaining candidates were not auto-excluded. See the [extended train-only duplicate audit](GHQWDO_CONTEXT_CROP_DUPLICATE_AUDIT_2026-11-23.md), including the adjudication CSVs and contact sheets. Duplicate grouping does not establish patient identity. The original 12-example crop overlay remains under `.venv/work/audit_ghqwdo/panel_crop_dataset/`.

## Matched experiment

| Arm | Training images / boxes | Method |
|---|---:|---|
| Project control | 15,030 / 8,223 | YOLO26s, common tooth-anatomy initialization, CIoU-NWD |
| Context-crop | 15,566 / 8,885 | Same project set and method, plus the 536 deduplicated crops / 662 boxes |

Both arms use the same seed, optimizer, augmentation, image size, schedule, NWD constant computed only from project train, and two T4 devices (one worker per GPU). Validation membership and evaluation code are identical: primary 477 images plus the previously defined 473-image sensitivity cohort. The full project validation remains unchanged. External validation and test are not used.

Execution guard: 60-epoch maximum with early stopping, per-epoch log heartbeats, checkpoint every five epochs, 15-minute worker no-progress watchdog, 5-hour epoch-boundary worker cap, and 6.25-hour paired-run cap. The Kaggle session timeout is set to 12 hours when submitted, leaving recovery time before the platform limit.

## Reporting and promotion

Report AP50, AP50-95, precision, recall, epochs, per-arm checkpoint hashes, crop counts, and both fixed project cohorts. Promote the context-crop arm only if primary AP50 improves by at least 1 percentage point, AP50-95 does not fall on either cohort, and recall does not fall by more than 1.5 points. A failed gate is a valid negative result.

Because the 477-image cohort has been repeatedly used in development and patient IDs are unavailable, even a promoted result remains a development comparison, not a generalization or clinical claim.

## Execution state

- Notebook source: `notebooks/alignment/16_70_spacing_context_crop_ablation.ipynb`.
- Staged Kaggle notebook SHA-256: `090c89462cff7bea285f2da4dfb58e3d9c7b8b0a15a32c5f1d6d4757814a2171`; metadata pins visible-alignment v2 and GHQWDO train-only v1 and requests GPU (`NvidiaTeslaT4`, Kaggle's T4 x2 identifier).
- The revised notebook crop/dedup cell passed a local smoke: 572 package inputs -> 559 eligible crops -> 536 crops / 662 labels after 23 confirmed repeated-photo pairs. The smoke verified all 31 pHash 3-8 candidates, 16 extra SIFT/pixel-confirmed pairs, image/label pairing and counts; project validation membership is unchanged. See [revised local smoke](SPACING_CONTEXT_CROP_REVISED_LOCAL_SMOKE_2026-11-23.json) and the [extended duplicate audit](GHQWDO_CONTEXT_CROP_DUPLICATE_AUDIT_2026-11-23.md).
- Public train-only source package: [Kaggle dataset](https://www.kaggle.com/datasets/hngphc73/ghqwdo-v2-diastema-train-only), version 1.
- Notebook has not been pushed to Kaggle or executed. The substantive 16.67 train-add-on ablation was re-pushed as version 2 at 15:48 UTC; it remained `QUEUED` at 16:08 UTC after version 1 was `CANCEL_ACKNOWLEDGED`. Defer this second paired-GPU experiment until 16.67 is collected and its result determines whether crop adaptation is still justified. Next 16.67 status poll: no earlier than 16:28 UTC.
