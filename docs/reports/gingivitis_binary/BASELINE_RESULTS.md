# Binary gingival detector baselines

Both latest Kaggle versions completed 60 epochs on one Tesla T4 each, with two T4 devices visible per session. The downloaded input manifests match the published preflight hash. Best checkpoint hashes match the execution logs, and stored training metrics match the selected epoch rows. The saved training arguments differ only in model and output names or paths.

| Model | Kaggle version | Precision | Recall | mAP50 | mAP50–95 | Best epoch | Training time | Checkpoint size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 4 | 93.25% | 96.11% | 95.45% | 55.78% | 50 | 9m 51s | 6.23 MB |
| YOLOv8s | 3 | 94.01% | 95.94% | 95.19% | 56.33% | 26 | 14m 45s | 22.50 MB |

Metrics come from final best-checkpoint revalidation. Ultralytics removed five duplicate training boxes and one duplicate validation box during loading. Final validation covers 160 images with 1,870 boxes. Validation has zero completely annotation-negative images. The scores describe localization of source-annotated regions. Image-level false alarms on wholly negative photos and performance on phone captures remain unmeasured. One seed was run per model.

YOLOv8n is the provisional model for the next evaluation. Its mAP50 is 0.26 percentage points higher and recall is 0.17 points higher. YOLOv8s gains 0.55 points at mAP50–95 and the checkpoint is approximately 3.6 times larger. Checkpoint size does not measure device latency.

Collected metrics, training configuration, plots, checkpoints and logs are under `data/runs/gum_binary_v8n_v4` and `data/runs/gum_binary_v8s_v3`. `baseline_results.csv` records the scores and hashes. The next evaluation should examine prediction overlap with source grade-0 regions and misses on mild versus marked regions. A fully negative image cohort needs separate verified annotations.
