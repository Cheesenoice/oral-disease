# YOLOv8n with the existing ResNet18 region filter

Notebook 15.5 evaluated all 2,060 saved YOLOv8n v4 detections at confidence ≥0.25 on the 160 locked D2 validation images. It projected each normalized box onto the checksum-verified original image and applied the R1 raw-crop preprocessing. The existing ResNet18 removed a box when its highest-scoring class was grade 0. Both arms use the original detector confidence for one-to-one positive matching at IoU ≥0.50. The experiment uses existing checkpoints and the original validation split.

| Outcome | YOLOv8n | YOLOv8n + R1 |
|---|---:|---:|
| Positive regions found | 1,814/1,870 (97.01%) | 1,776/1,870 (94.97%) |
| Mild regions found | 908/931 (97.53%) | 875/931 (93.98%) |
| Marked regions found | 905/938 (96.48%) | 900/938 (95.95%) |
| Grade-0 regions flagged | 49/49 (100%) | 40/49 (81.63%) |
| Unmatched predictions | 246 | 229 |

The filter removes 55 predicted boxes, and the matched-positive count decreases by 38 regions, including 33 mild regions. The grade-0 flagged count improves by nine regions, with 40 still flagged. The ambiguous positive box remains matched in both arms. The current filter provides insufficient improvement to select as the default: most grade-0 regions remain flagged and mild recall drops by 3.54 percentage points. These metrics are annotation-relative and do not estimate false alarms on wholly negative photos.

The next comparison requires independently reviewed negative and mixed images. If the grade-0 issue persists, test a binary region classifier trained with reviewed negatives, then compare it against both saved arms. The prediction scores and per-image counts are stored in `classified_boxes.csv` and `per_image.csv`.
