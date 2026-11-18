# ROI Method Findings - 21 October 2026

Both public version-1 Kaggle kernels completed. All four sets of 1,920 validation predictions were rechecked against the locked manifest and labels; confusion matrices, accuracy, macro-F1, balanced accuracy, ordinal MAE and quadratic weighted kappa match the saved metrics. Ordinal probabilities are monotone and decode to the saved predictions. No test data was read.

## Measured comparison

| Experiment | Accuracy | Macro-F1 | Balanced accuracy | MAE | QWK | Best / total epochs | Minutes |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 scratch CNN | 71.61% | 48.32% | 48.98% | 0.284 | 0.484 | 19/27 | 20.4 |
| C1 scratch ResNet18 | 69.79% | 55.42% | 54.20% | 0.303 | 0.476 | 29/30 | 26.4 |
| C2 ImageNet ResNet18 | 69.58% | 55.76% | 54.71% | 0.304 | 0.476 | 11/19 | 26.6 |
| M1_concat_CE | 69.53% | 53.60% | 53.34% | 0.306 | 0.482 | 4/9 | 22.1 |
| M2_gate_CE | 67.03% | 53.99% | 59.37% | 0.334 | 0.493 | 7/12 | 25.1 |
| M3_local_CORAL | 63.28% | 51.46% | 63.86% | 0.373 | 0.478 | 7/12 | 23.1 |
| M4_gate_CORAL | 63.65% | 49.48% | 56.37% | 0.373 | 0.438 | 10/15 | 26.4 |

## Findings

- C2 remains the best measured macro-F1 reference (55.76%). M1, M2, M3 and M4 trail it by 2.16, 1.77, 4.30 and 6.28 percentage points respectively. These single-run differences are descriptive, not causal evidence.
- M2 gating slightly exceeds M1 concatenation in macro-F1 (53.99% vs 53.60%) and increases no-visible recall, but its accuracy decreases. The average gate value near 0.45 does not establish which anatomical context is useful.
- M3 has the highest balanced accuracy (63.86%) and no-visible recall (32/49 = 65.31%), but mild recall falls to 39.38%. No-visible precision is only 18.08% (32/177). Predicting the rare class more often is not the same as improving overall grading.
- M4 has the highest marked recall (88.60%), but the lowest method macro-F1 (49.48%). It predicts marked for 462/932 mild ROIs. The combined method is not the current winner.
- CORAL did not improve measured ordinal error: MAE is 0.373 for M3/M4 vs 0.304 for C2. Extreme 0-to-2 or 2-to-0 errors are 0 for C2, 3 for M1, 8 for M2, 12 for M3 and 18 for M4. Monotone cumulative outputs do not guarantee better ordinal classification.
- All four workers stopped normally after five stale epochs: best/total epochs 4/9, 7/12, 7/12 and 10/15. Worker runtimes were 22.1-26.4 minutes. Kernel logs end near 26.3 and 27.4 minutes. Neither the 20-epoch ceiling nor 70-minute worker budget was reached.
- Training loss kept decreasing while validation loss increased. For example M2 ends at train/validation CE 0.117/1.080. This is evidence of a growing generalization gap; extending the same runs blindly is not supported. CE and CORAL loss magnitudes should not be compared directly.

## Protocol audit and limitations

The version-1 experiments are exploratory screening, not a fully controlled factorial ablation. The implementation introduced the following confounders; correct them before a final method claim:

1. C2 uses 30 max epochs, patience 8, color jitter and DataParallel; 07 uses 20 max epochs, patience 5, no color jitter, one GPU per trial and gradient accumulation. An effective batch of 64 does not guarantee the same BatchNorm or optimizer dynamics (including the partial final batch).
2. M1/M3 declare seed 42, while M2/M4 declare seed 43. Concurrent threads also reset global Python/NumPy/PyTorch RNG state, so the declared per-worker seeds are not reliably isolated. This prevents an exact same-seed comparison and bitwise reproducibility.
3. The fusion heads introduce dropout and different parameter counts. Shared-encoder paired forwards update BatchNorm twice per optimizer microbatch. These are additional architectural/training changes beyond the loss label.
4. The ordinal head is a monotone shared-score adaptation, not an exact published CORAL reproduction. The published 07.2 v1 creates two score channels but uses only channel zero; the unused row is redundant and should be removed. It does not invalidate the recomputed predictions, but the implementation must be documented and cleaned up.
5. Only 49 validation ROIs belong to no-visible inflammation, and multiple ROIs come from each of 160 source photographs. They are not 1,920 independent patients. No clinical diagnostic sensitivity or specificity is inferred.
6. ROI grading assumes annotation-provided localization. These numbers are not detector mAP or end-to-end phone-camera results. Test remains locked.

## Next executable action

Keep these v1 artifacts as real screening evidence. First prepare a matched local-CE control and matched local-ordinal control with the same encoder/head conventions, seed list, augmentation, microbatch/accumulation, early stopping and validation rule. Isolate worker RNGs (prefer separate processes) and clean the unused ordinal output. Then compare context concatenation/gating under that identical recipe. Use additional seeds for the selected comparison; do not start a new run or tune on test as part of this analysis request.

If class imbalance is investigated, declare it as a separate train-derived loss/sampling ablation. Do not add it silently and attribute any improvement to context or ordinal supervision. Keep C2 as the provisional model rather than selecting M4 for its complexity.

The full artifact bundles are in `data/runs/ROI_07_1_v1/` and `data/runs/ROI_07_2_v1/`. Raw public source, metadata and execution logs are in matching folders under `docs/reports/training/`. The first CLI downloads returned a Windows log-encoding error after saving data and checkpoints; logs were recovered with UTF-8 enabled. No training failure is implied.
