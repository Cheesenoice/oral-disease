# Combined 09.1-09.3 Findings - 22 October 2026

## Version resolution and integrity

09.1 v1 and 09.2 v1 contain four completed fine-tuning runs. The valid 09.3 run is **v3**, confirmed COMPLETE using version_label=v3 in the authenticated SDK. The latest owner version **v4 is CANCEL_ACKNOWLEDGED**, has no log and no output. CLI status/output methods discard numeric version suffixes, so resolving the version explicitly was necessary. The earlier 09.3 v2 execution had training disabled; it remains historical evidence, not the probe result.

Version-3 submitted source has RUN_TRAINING=True and the same reviewed embedded-worker hash as the planned notebook. Its source was screened for common credential patterns before saving. No token was printed or copied into this report. Both probes passed their smoke and generated full results. For each of six full trials, the 1,920 validation crop IDs and labels match the locked manifests. Confusion matrices and accuracy/macro-F1/balanced accuracy/MAE/kappa were recomputed; probe probabilities decode exactly to predictions and the selected linear-head C matches the declared grid. Test images opened: zero.

## Combined results

| Trial | Training | Accuracy | Macro-F1 | Balanced accuracy | Recall no-visible / mild / marked | Worker minutes |
|---|---|---:|---:|---:|---|---:|
| R0_old_CE | finetune | 68.07% | 56.61% | 60.11% | 42.86% / 68.67% / 68.80% | 11.83 |
| R1_raw_CE | finetune | 68.70% | 57.26% | 58.61% | 36.73% / 71.46% / 67.63% | 9.48 |
| R2_raw_CBCE | finetune | 69.11% | 57.40% | 60.16% | 40.82% / 63.41% / 76.25% | 10.14 |
| R3_raw_BalancedSoftmax | finetune | 67.03% | 53.36% | 56.15% | 32.65% / 59.44% / 76.36% | 14.68 |
| P0_raw_frozen_ResNet18 | probe | 66.67% | 50.08% | 50.10% | 14.29% / 61.48% / 74.55% | 0.91 |
| P1_raw_frozen_DINOv3 | probe | 67.97% | 47.83% | 47.77% | 4.08% / 61.70% / 77.53% | 0.87 |

Historical C2 (6.3) achieved 69.58% accuracy and 55.76% macro-F1 with a different recipe. R2 is the highest observed macro-F1 at 57.40%, still below the project aspiration of 75%. Fine-tuning results and frozen-feature probes answer different questions; compare P0/P1 as the representation experiment, rather than attributing their gap to fine-tuned R1 solely to architecture.

## What the completed cycle shows

1. Original-source crop preservation: R1 vs matched R0 improves macro-F1 by only 0.66 percentage points. The grouped-bootstrap interval [-2.88,+4.43] includes zero. It is a small descriptive change, not a demonstrated solution.
2. Class-balanced CE: R2 vs R1 adds 0.13 points, grouped interval [-2.92,+3.02]. Balanced Softmax reduces macro-F1 by 3.90 points in this run. Minority handling alone did not solve class separability.
3. Frozen representation: P0 frozen ImageNet ResNet18 achieves 50.08% macro-F1; P1 frozen DINOv3 ViT-S/16 achieves 47.83%. DINOv3 is 2.25 points lower; paired source-group bootstrap interval is [-6.89,+2.38]. This is not evidence of DINOv3 superiority and not a universal conclusion about the model.
4. The DINOv3 probe recognizes only 2/49 no-visible ROIs (4.08% recall), versus ResNet18 probe 7/49 (14.29%). DINOv3 no-visible precision is 2/21=9.52%. Accuracy 67.97% therefore hides failure on the minority class.
5. Mild/marked recall for DINOv3 is 61.70%/77.53%. Its major error is 343 mild ROIs predicted marked, plus 206 marked predicted mild. Label overlap/domain difficulty persists.
6. P0 selected C=10, the upper end of the four-value grid; P1 selected C=1. No convergence warning appears in the saved worker logs. Selected C at a boundary is a tuning-budget limitation, not permission to retrospectively expand only one model's grid.
7. The frozen DINOv3 experiment uses a 384-dimensional CLS feature with a train-fitted StandardScaler and logistic classifier, no domain adaptation, no patch aggregation and no encoder fine-tuning. Frozen ResNet18 uses 512-dimensional pooled features. Both use deterministic 224px whole-crop inputs with ImageNet normalization and the same head grid. Model/pretraining differences are part of this comparison.
8. Probe worker times are 54.5 and 52.3 seconds, after model construction/loading and input verification; these are not complete kernel runtimes or phone inference latency. Fine-tune worker times were 9.5-14.7 minutes plus setup/smoke.

## Decision

The current experiments do not support a strong teacher for distillation or a useful three-class gingival demo. Do not start distillation, ensemble or further gating/ordinal combinations merely to add complexity. Keep R1 raw-crop CE as the simple reference and R2 as a near-tied candidate; do not promote R2 on a 0.13-point one-seed difference. The 160 observable validation groups and 49 no-visible ROIs limit reliability; this validation set has also been reused for many choices. No claim of clinical usability or independent patient-level accuracy is supported.

Next recommended work is a bounded task-feasibility review: source-class composition and original MGI semantics, expert review of difficult/ambiguous examples, and full-image caries error/generalization checks. Any clinical label correction requires qualified review. Source images remain immutable; test remains locked. Caries remains the provisional demo anchor because detection validation is stronger, but independent phone-photo behavior must be checked. If needed, declare a simpler gingival endpoint as a new task with new metrics, rather than claiming the old three-class goal was met. Healthy-vs-inflamed grouping does not eliminate healthy-class scarcity.

A later partial DINOv3 fine-tune or richer patch probe would be a new controlled hypothesis, not a guaranteed rescue; it is not started by this reporting task. Mobile implementation/export remains deferred.

## Thesis narrative

Scratch architecture controls -> ImageNet initialization -> unsuccessful context/ordinal screening -> verified native-resolution recovery -> class-imbalance ablations -> modern foundation-feature test -> evidence-based scope decision. Negative results are retained. The model family being newer did not produce a better linear-probe outcome on these crops; this is an experimentally grounded finding.

## Artifacts

- `data/runs/ROI_09_3_v3/`: exact versioned output, probe heads, grids, predictions and logs.
- `docs/reports/training/ROI_09_3_v3/`: versioned status, log and screened submitted source.
- `roi_foundation_probe_comparison.json`: verified metrics and paired grouped uncertainty.
- `roi_09_combined_summary.json`: combined six-trial summary.
- `ROI_RECOVERY_FINDINGS_2026-10-22.md`: earlier crop/loss analysis, written before v3 collection.
