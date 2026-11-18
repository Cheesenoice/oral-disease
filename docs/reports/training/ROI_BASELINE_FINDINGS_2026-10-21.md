# ROI baseline findings — 21 October 2026

## Execution and protocol

All three GPU runs completed on Kaggle with two Tesla T4 GPUs. Each used seed 42 and the same pinned ROI manifest (`8ed4c05240a2d23e6c254ee74adf13af48ff45a4418f1e2024bd38498b02cab1`): 8,969 training crops and 1,920 validation crops from 907 parent images. Model selection used validation macro-F1. The test split remained unopened. These are crop-classification results with supplied ROIs, not end-to-end detection or clinical diagnostic performance.

| Notebook | Model and initialization | Best epoch / completed | Runtime | Parameters | Val accuracy | Val macro-F1 | Val balanced accuracy | Recall: none / mild / marked |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 6.1 | Custom CNN, random | 19 / 27 | 20.4 min | 389,667 | 0.7161 | 0.4832 | 0.4898 | 0.000 / 0.6953 / 0.7742 |
| 6.2 v3 | ResNet18, random | 29 / 30 | 26.4 min | 11,178,051 | 0.6979 | 0.5542 | 0.5420 | 0.2041 / 0.7339 / 0.6880 |
| 6.3 v2 | ResNet18, ImageNet IMAGENET1K_V1, all layers fine-tuned | 11 / 19 | 26.6 min | 11,178,051 | 0.6958 | **0.5576** | **0.5471** | 0.2245 / **0.7575** / 0.6592 |

## What the results say

**ResNet18 from scratch scored higher than the small CNN on the primary metric in this run.** It gained 0.071 macro-F1 and 0.052 balanced accuracy over 6.1, while ordinary accuracy fell by 0.018. This observed difference is consistent with an architecture benefit, but one seed does not establish a reliable gain. The reversal shows why accuracy alone gives a misleading ranking on the imbalanced validation set.

**ImageNet initialization produced only a small single-run gain over scratch ResNet18.** The macro-F1 difference is 0.0034 and balanced-accuracy difference is 0.0051; at one seed these do not establish a reliable transfer-learning advantage. 6.3's training macro-F1 was 0.7916 at its selected epoch versus validation 0.5576, then rose to 0.964 while final validation macro-F1 fell to 0.5356. This is a clear overfitting signal during full fine-tuning. 6.2's selected-epoch training/validation macro-F1 was 0.5882/0.5542, a smaller gap. The pretrained run reached its best validation epoch earlier, but total measured runtime was similar (about 26.6 vs 26.4 minutes).

**The no-visible-inflammation class is the central failure for all three.** It has only 49 validation ROIs. The CNN recognized none of them (recall 0); scratch ResNet18 recognized 10 (0.204); pretrained ResNet18 recognized 11 (0.224). Mild and marked inflammation are also confused in both directions. Per-class counts and confusion matrices are in `roi_baseline_comparison.json` and the original Kaggle outputs. This imbalance is consistent with the training distribution (210 no-visible vs 3,829 mild and 4,930 marked), but does not by itself prove that imbalance is the only cause.

**Do not claim the pretrained ResNet18 is a meaningful winner yet.** This is one fixed seed and one validation split, with small metric differences between 6.2 and 6.3. The test set was not used, and no patient-level independence is claimed. A repeat on additional seeds is reserved for the final selected comparisons.

## Correction retained in the experiment record

The original 6.2 version 2 was mislabeled: its initialization cell loaded ImageNet weights when training was enabled. Its 19-epoch history and validation report exactly matched 6.3 v2. It is excluded from the comparison. The source was corrected, public version 3 explicitly prints `pretrained_weights_loaded: False`, and its collected run metadata confirms random initialization. Do not erase the original v2 incident record in `roi_baselines_publication.json`.

## What comes next

Proceed to **notebook 07, `07_context_ordinal_ablation.ipynb`**, as planned. Use the corrected 6.3 recipe as the common pretrained reference and compare the predeclared M1–M4 changes: local-plus-context concatenation, learned gated fusion, CORAL ordinal supervision alone, and gate plus CORAL. Keep splits and preprocessing fixed, select on validation macro-F1, and report class-wise recall, confusion matrix, ordinal metrics, train/validation curves, parameter count and runtime. Because class 0 recall is poor, include a validation-only error-gallery and a clearly isolated class-imbalance control only if it can be added without confounding the M1–M4 ablation. Do not open test until the method and thresholds are frozen.

## Local outputs

- 6.1: `.venv/work/kaggle/results/6-1/6.1/`
- 6.2 v3: `.venv/work/kaggle/results/6-2-v3/6.2/`
- 6.3 v2: `.venv/work/kaggle/results/6-3/6.3/`
- Machine-readable comparison: `docs/reports/training/roi_baseline_comparison.json`
- Kernel versions, integrity correction and run receipt: `docs/reports/training/roi_baselines_publication.json`
