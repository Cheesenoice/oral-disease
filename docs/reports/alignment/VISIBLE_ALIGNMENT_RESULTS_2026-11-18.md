# Visible alignment YOLO11 results - 18 November 2026

## Summary

Both Kaggle training runs completed model training and saved usable checkpoints. Their notebooks then failed in the post-training bookkeeping cell because Ultralytics returned `None` from `model.train()` in these distributed/time-budget runs, while the notebook assumed it could read `result.save_dir`. The exception was `AttributeError: 'NoneType' object has no attribute 'save_dir'`. It happened after the trainer saved the checkpoints, `results.csv`, curves, confusion matrices, and validation batch plots. The separate final-checkpoint validation cell did not run.

The values below are the best epoch rows from the saved per-epoch source-validation `results.csv`; they are not independent phone or clinical performance.

## Results

| Model | Epochs completed | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Training time | `best.pt` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n, COCO initialized | 46 | 37 | 50.47% | 41.43% | 41.30% | 18.31% | 1.751 h | 5.47 MB |
| YOLO11s, COCO initialized | 32 | 24 | 50.21% | 46.62% | 44.09% | 20.85% | 1.435 h | 19.18 MB |

YOLO11s improves the source-validation mAP50 by 2.80 percentage points and mAP50-95 by 2.54 points. Recall rises by 5.19 points; precision is 0.46 points lower. This is one run per architecture, so it is a directional comparison, not evidence of a statistically reliable gain. YOLO11s has about 3.64 times as many parameters and a 3.5 times larger checkpoint.

The YOLO11n run hit its configured 1.75-hour training budget and completed 46 epochs. YOLO11s stopped after 32 epochs with patience 8; its best row was epoch 24.

## Integrity and failure audit

- Both results contain only the two source classes: `gap_or_spacing` and `malposition_present`.
- The exact source train/validation manifest was verified before training; the held-out test was excluded.
- Kaggle logs confirm two Tesla T4 GPUs were visible for YOLO11n.
- Both `best.pt` files load locally and expose the expected class names.
- YOLO11n `best.pt` SHA-256: `9d1df64d9debe0932ed25746f356227523975ebe6f4900e5ce085ae1af1a42e2`.
- YOLO11s `best.pt` SHA-256: `d238b1fa2496648c08e4aae42dce655df7087d55d1c4795fd3f51aaa130720a1`.
- Run artifacts and compact checksums are under `data/runs/alignment_visible_yolo11n_r3/`, `data/runs/alignment_visible_yolo11s_r2/`, and `data/runs/alignment_visible_results_2026-11-18.json`.
- Notebooks 16.17-16.19 now derive the run folder from a fixed `run_name` rather than dereferencing the `model.train()` return value. No retraining was triggered by this local fix.

## Can these be used?

**For the thesis:** yes, as completed source-domain baselines with saved checkpoints and per-epoch validation evidence. The training failure does not erase those artifacts.

**For an app or clinical demonstration:** not as a reliable diagnostic detector. mAP50 of 41-44% and mAP50-95 of 18-21% are low; the source labels are broad annotations rather than clinician-confirmed midline diastema/crowding. The validation set is part of the training workflow for epoch/checkpoint selection, has no verified patient IDs, and does not establish ordinary-phone generalization. The results support a baseline comparison, not a claim of dependable disease detection.

## Next action

Run final validation directly on each saved `best.pt` using the same locked source validation split, or collect a corrected Kaggle evaluation-only run. Then compare class-specific AP and review the saved validation prediction panels. Keep the held-out test locked and do not describe the source classes as clinical diagnoses.
