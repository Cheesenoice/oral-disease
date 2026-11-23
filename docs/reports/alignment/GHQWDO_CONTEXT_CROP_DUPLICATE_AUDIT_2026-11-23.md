# GHQWDO Context-Crop Duplicate Audit - 23 November 2026

## Purpose

Screen the **train-only** context crops used by notebook 16.70 for repeated photos and panel variants. This is data-quality review, not clinical relabeling. The fixed project validation and source validation/test partitions were not opened or changed.

## Result

| Stage | Images | Target boxes | Decision |
|---|---:|---:|---|
| Eligible context crops before crop-level deduplication | 559 | 690 | All arise from 572 train-only candidate images; 13 crops with partially visible targets were already excluded |
| pHash distance <= 2 | 7 pairs | - | All seven previously reviewed visually; confirmed duplicates |
| pHash distance 3-8 | 31 candidate pairs | - | Contact-sheet review; 16 confirmed with strong geometric and pixel evidence, 15 left untouched |
| Final add-on after grouping confirmed copies | **536** | **662** | Lexicographically first source image retained in each duplicate pair |

The extended pass removes 23 repeated-photo variants, 32 source-box instances and 28 crop-box instances. The remaining crops carry 842 source target boxes; 180 lie completely outside the chosen context window, leaving 662 supervised crop boxes. The fixed train/validation membership for the project source remains the same.

## Review method

1. Compute mirror-invariant pHash on each crop after trimming the black canvas padding and normalizing the foreground to 128 x 128. The exact ImageHash 4.3.2 implementation is pinned in notebook 16.70.
2. Inspect all pHash-distance <= 2 and 3-8 candidate pairs in contact sheets.
3. Treat the second distance band only as candidate generation. Confirm a repeated-photo variant only when visual review agrees with SIFT RANSAC (at least 30 inliers, at least 0.80 inlier ratio) and aligned grayscale pixel correlation (at least 0.98).
4. Keep the lexicographically first source filename in each confirmed group. Do not infer patient identity; do not remove ambiguous candidates.
5. Verify the image and label file hashes and every retained YOLO box before using the reviewed manifest.

The 31 candidates, SIFT evidence, accepted duplicate edges, deterministic keep/drop manifest and summary are saved alongside this report:

- [All pHash 3-8 candidates and review decisions](ghqwdo_context_crop_duplicates/all_phash_3_to_8_reviewed.csv)
- [Confirmed duplicate edges and evidence](ghqwdo_context_crop_duplicates/confirmed_duplicate_edges.csv)
- [Reviewed crop manifest](ghqwdo_context_crop_duplicates/crop_manifest_reviewed.csv)
- [Machine-readable summary](ghqwdo_context_crop_duplicates/summary.json)

### Contact sheets

Every pHash 3-8 candidate pair was visually inspected. The row captions show the pHash distance; individual pair evidence is in the CSV above.

![Near-duplicate candidates 1-16](ghqwdo_context_crop_duplicates/phash_review_01.jpg)

![Near-duplicate candidates 17-31](ghqwdo_context_crop_duplicates/phash_review_02.jpg)

## Reproducibility and limits

From the project root, run `python src/scripts/prepare_ghqwdo_context_crops.py` and then `python src/scripts/adjudicate_ghqwdo_context_crop_duplicates.py` to reproduce the crops, pair verification, graph groups, retained manifest and checksums. The second script verifies crop image/label SHA-256 values and normalized box geometry. Both steps read only the local train-only package.

The review removes strong repeated-photo evidence from a weak, heterogeneous external set. It does not establish patient-level independence, prove every remaining image is unique, or guarantee a metric gain. No model training or evaluation metric is claimed here. Notebook 16.70 must still pass its local smoke before it is pushed to Kaggle.
