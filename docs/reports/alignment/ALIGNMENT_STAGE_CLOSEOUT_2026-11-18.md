# Alignment Stage Closeout - 18 November 2026

## Kaggle work completed

The three submitted alignment kernels use the published OCCLSAL ORTHO AI v3 source dataset (CC BY 4.0), its source train/validation split, and YOLO11. All kernels completed on Kaggle with a Tesla T4 visible. Outputs and checkpoints are collected under `data/runs/`.

| Run | Configuration | mAP50 | mAP50-95 | Diastema AP50 | Crowding AP50 |
|---|---|---:|---:|---:|---:|
| YOLO11n baseline | 640, batch 16 | 70.01% | 32.89% | 83.0% | 57.0% |
| YOLO11s comparison | 640, batch 16 | 66.69% | 33.73% | 69.9% | 63.5% |
| YOLO11n resolution comparison | 960, batch 8 | 56.60% | 33.54% | 61.0% | 52.2% |

The larger model trades away diastema AP50 while gaining 6.5 points on source crowding; the 960-pixel run is lower on both classes. The 960 comparison also changes batch size. These are single-seed source-validation results with only six diastema and 36 crowding validation boxes; repeated-looking scenes and positive-only images limit generalization claims. The source images are occlusal, and some crowding annotations are posterior. These models do not establish frontal phone-camera accuracy.

## Scope and data status

- The CC BY 4.0 source data and training notebooks are already published on Kaggle; the three GPU runs are complete. Kaggle account authentication is working.
- The intended frontal OMNI annotation pilot remains a separate, untrained local research set. Current v4 has 14 provisional frontal diastema boxes in train, no independently reviewed validation annotations, and two provisional frontal crowding boxes that are excluded from training. The current pilot is not a valid training/validation package.
- OMNI's paper and repository make the dataset available for research, but the dataset files have no explicit license or redistribution grant identified in the official repository. Therefore, OMNI images were not uploaded to Kaggle. The paper's academic-use consent statement is not treated as a data redistribution license.
- Phone feasibility is not established. Existing phone photos are not an independent labeled evaluation set.

## Conclusion

The available permissively licensed alignment dataset has been uploaded, the notebooks have been published, and the three planned YOLO11 experiments have been trained and collected. The deliverable is an **occlusal-source research baseline**, not a frontal phone-ready detector. Do not describe it as detecting crowding in frontal selfies. A frontal experiment can resume only after there is an authorized, sufficiently labeled frontal training and validation set; do not derive labels from model predictions.

See [source audit](SOURCE_AUDIT.md), [comparison results](OCCLUSAL_COMPARISON_RESULTS.md), and [frontal view screen](FRONTAL_VIEW_SCREEN_2026-11-18.md) for provenance and annotation constraints.
