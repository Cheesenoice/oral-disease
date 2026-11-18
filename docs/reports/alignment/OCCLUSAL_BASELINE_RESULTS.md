# Occlusal source benchmark results

YOLO11n version 1 completed 60 epochs with one Tesla T4 worker. Best checkpoint was selected at epoch 12 by mAP50–95. Final best-checkpoint validation metrics are below. The SHA-256 of the retrieved checkpoint matches the run log and its stored epoch metrics match results.csv.

| Target | Validation images | Boxes | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| Midline diastema | 6 | 6 | 49.6% | 100.0% | 83.0% | 41.3% |
| Source crowding | 32 | 36 | 59.6% | 57.5% | 57.0% | 24.5% |
| Mean | 38 | 42 | 54.63% | 78.74% | 70.01% | 32.89% |

Training completed in approximately 414 seconds. The reported best checkpoint SHA-256 is `30fc101bdad08d5b34b1dd83082e489b30f6c0130f5429d9313a9bbf0ac8b22f`. Final checkpoint revalidation differs slightly from the epoch CSV because the saved checkpoint is validated again.

Validation contains target-positive occlusal images, some with source crowding boxes on posterior teeth. Midline diastema has only six annotated validation instances. Predictions show repeated-looking validation scenes. The source includes visually similar images, and dHash was used conservatively only to exclude candidate cross-split training images. Unknown near duplicates and validation dependencies remain. Absence of a target source label is not an independently verified negative. These results cannot establish frontal camera performance or clinical accuracy.

Next prioritise acquiring or annotating frontal photographs with explicit midline diastema and anterior crowding boxes, along with reviewed target-negative images. Preserve patient or image-family grouping when verifiable and lock validation before training. The YOLOv11 malocclusion paper provides box definitions, but its dataset and performance differ. Notebook 16.2 is prepared for reviewed frontal data; this occlusal checkpoint is an exploratory source-view baseline.
