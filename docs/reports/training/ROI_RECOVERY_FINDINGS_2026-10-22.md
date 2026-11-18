# ROI Recovery Findings - 22 October 2026

## Execution evidence

09.1 v1 and 09.2 v1 are COMPLETE with four full-training results. Each has 1,920 validation predictions and its own successful SMOKE_OK checkpoint-restore log. Prediction identities/labels, probability decoding, confusion matrices, saved metrics and best-epoch selection were verified locally. No test pixels were opened.

09.3 v2 is COMPLETE as a notebook execution only: the downloaded source still declares RUN_TRAINING=False. The execution log explicitly says Training disabled and no full result yet for P0/P1. The output contains recovery_config.json and the kernel log, with no probe metrics/features/predictions. It must not be counted as an executed ResNet18/DINOv3 experiment.

## Matched fine-tuning comparison

| Trial | Accuracy | Macro-F1 | Balanced accuracy | No-visible / mild / marked recall | Best / total epochs | Worker minutes |
|---|---:|---:|---:|---|---:|---:|
| R0_old_CE | 68.07% | 56.61% | 60.11% | 42.9% / 68.7% / 68.8% | 10/18 | 11.8 |
| R1_raw_CE | 68.70% | 57.26% | 58.61% | 36.7% / 71.5% / 67.6% | 5/13 | 9.5 |
| R2_raw_CBCE | 69.11% | 57.40% | 60.16% | 40.8% / 63.4% / 76.3% | 5/13 | 10.1 |
| R3_raw_BalancedSoftmax | 67.03% | 53.36% | 56.15% | 32.7% / 59.4% / 76.4% | 14/22 | 14.7 |

## Interpretation

- R1 vs R0 is the controlled preprocessing comparison. Original-source crops add 0.66 macro-F1 percentage points and 0.63 accuracy points. Minority recall falls from 42.86% to 36.73%, while mild recall rises. Native-pixel preservation alone does not explain or solve the large performance deficit.
- R2 class-balanced CE adds only 0.13 macro-F1 points versus raw-crop CE (R1). It shifts recall toward marked inflammation (76.25%) and away from mild (63.41%). This small one-seed difference does not establish superiority.
- R3 Balanced Softmax trails R1 by 3.90 macro-F1 points. No-visible precision is 18.18%, recall 32.65%; stronger prior correction did not yield the desired tradeoff in this run.
- The highest measured macro-F1 is now R2 at 57.40%, versus historical C2 55.76%. Only the new matched controls support component comparisons: historical C2 has different training details. Current results remain below the engineering aspiration of 75% macro-F1.
- All four workers stopped normally after eight stale epochs. R0/R1/R2/R3 selected epochs were 10/5/5/14 and totals 18/13/13/22. Worker times were 9.5-14.7 minutes; smoke and setup add kernel wall time.
- Train/validation losses diverge: R0 final CE 0.089/1.272, R1 0.237/0.924. More epochs of the same fine-tuning recipe are not presently justified. CB/softmax-adjusted losses have different definitions and their magnitudes should not be ranked against CE.

## Uncertainty

Paired group bootstrap, 2,000 repetitions, resampling all ROIs of each observable source group together:

| Difference | Macro-F1 change | 95% grouped bootstrap interval, percentage points |
|---|---:|---|
| R1 - R0 | +0.66 pp | [-2.88, +4.43] |
| R2 - R1 | +0.13 pp | [-2.92, +3.02] |
| R3 - R1 | -3.90 pp | [-7.71, -0.42] |

There are 160 validation source groups. These intervals are conditional on the selected validation checkpoints; they exclude training-seed and model-selection uncertainty and are not a substitute for a frozen test evaluation. Group IDs are not verified patient identities.

## Decision and next executable action

Keep the raw-crop CE run as the simple provisional reference for the pending frozen-feature study; do not promote a complex combination or claim the loss change solves the task. Finish the planned 09.3 frozen ResNet18/DINOv3 comparison before deciding whether representation learning can improve class separation. This is the missing branch, not a repeat of completed 09.1/09.2. Official DINOv3 weights access plus RUN_TRAINING=True and GPU T4 x2 must be verified. No additional GPU job was started in this collection turn.

If the user ran a fork on another account, collect its exact kernel URL/version instead of assigning the owner kernel's disabled run its metrics. Local HF credentials remain unavailable at collection time; no secret was read or printed.

If probes also remain weak, stop adding losses/modules. Review label/image ambiguity and consider a bounded scope decision while retaining caries as the provisional demo anchor. Classifier crop metrics still do not establish full-image detection or phone-photo performance.

## Artifacts

Raw source, metadata and logs: `docs/reports/training/ROI_09_1_v1/`, `ROI_09_2_v1/`, `ROI_09_3_v2/`. Measured output folders: corresponding names under `data/runs/`. Recomputed results: `roi_recovery_comparison.json`; grouped uncertainty: `roi_recovery_group_bootstrap.json`.
