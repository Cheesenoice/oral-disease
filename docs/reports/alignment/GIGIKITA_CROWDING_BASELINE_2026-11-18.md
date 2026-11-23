# GigiKita crowding baseline

## Frozen decision

Use only GigiKita Roboflow version 1 and its source label `crowded`. Do not merge FIPDZ or other candidate labels: reviewed cross-source examples showed inconsistent box counts and overlapping scenes. The source declares CC BY 4.0; preserve attribution and treat this as the source's declaration, not a legal opinion.

## Data and split

| Item | Result |
|---|---:|
| Source train + validation inspected | 7,143 images |
| Exact/strong visual duplicate groups | 225 groups |
| Redundant images removed | 265 |
| Conflicting duplicate groups quarantined | 5 |
| Train | 5,516 images; 2,224 positive; 4,165 boxes |
| Validation | 1,357 images; 571 positive; 1,093 boxes |
| Duplicate groups shared across splits | 0 |
| Invalid boxes / missing image-label pairs | 0 / 0 |
| Source test accessed | No |

The split is a deterministic 80/20 stratified split by target presence, after visual-duplicate grouping and conflict quarantine (seed 51324). Duplicate grouping covers strong pHash/ORB candidate matches found in the audit; it is not proof that every near-duplicate has been found. There are no patient identifiers; patient-level independence is not claimed. A negative means no `crowded` annotation in this seven-class export, not clinically healthy. The source label/box rule is not clinician-adjudicated here, so the experiment is a source-label baseline only.

Manifest SHA-256: `44d39f856f80508da95eafcaecd49c362f3e34b9d8b2b0f8729281c54229c488`.

## One experiment

Notebook [16.21](../../../notebooks/alignment/16_21_gigikita_crowding_yolo11s_baseline.ipynb) runs one YOLO11s COCO-initialized single-class detector at 640 pixels, seed 51324, with early stopping, an 8-hour training time cap, periodic checkpoints and T4 x2. It reports precision, recall, mAP50 and mAP50-95 on the fixed validation split. These are model-selection metrics, not a final untouched estimate. No other model/data branch is included in this run.

## Kaggle handoff

- Public, train/validation-only dataset: `trihuynhviprovcl/dentalselfie-crowding-gigikita-v1`.
- Kernel: [`trihuynhviprovcl/ds-crowding-yolo11s-v1`](https://www.kaggle.com/code/trihuynhviprovcl/ds-crowding-yolo11s-v1), version 4 (successful; script 371105706; 3,461.8 seconds). Version 1 stopped before training because the assumed input mount contained neither the expected manifest nor ZIP. Version 4 searches recursively from `/kaggle/input`; the actual mount is `/kaggle/input/datasets/trihuynhviprovcl/dentalselfie-crowding-gigikita-v1/gigikita_crowding_train_valid`. Remote logs confirm manifest/inventory checks passed, both T4s initialized, YOLO11s trained, and artifact packaging ran. The previous `StopIteration` came from the superseded guessed-root lookup and occurred before training.
- Archive SHA-256: `2c103696ee2389a1919bc36f0befa9aaf51ce341d95225679a047d3dfe75ea10`.
- The source test directory and personal images are excluded.

The task is useful as a reproducible engineering baseline. It is not enough to claim clinical reliability, patient-independent performance, or performance on casual phone captures. A low or unstable score should be reported as a data/label limitation, not repaired by filtering validation cases.

## Collected run results

The output archive at C:\Users\huynh\Downloads\results.zip contains 64 files; ZIP CRC verification passed for every member. Archive SHA-256: ba1fd4d668e9fa4ec8783830ae54f4f7721e3315cabf5ae0a9a97d752f72e96f. The frozen manifest hash in summary.json matches the prepared dataset, and source_test_opened is false. Twenty selected artifacts are extracted under data/runs/alignment_gigikita_crowding_v4/; the original archive remains intact in Downloads.

| Metric | Best-checkpoint re-evaluation | Peak epoch row in results.csv (epoch 31) |
|---|---:|---:|
| Precision | 61.56% | 61.57% |
| Recall | 61.48% | 61.57% |
| mAP50 | 62.82% | 62.91% |
| mAP50-95 | 27.44% | 27.54% |

The report uses the best.pt re-evaluation as the selected checkpoint result; the CSV value is the training-time validation row. The small difference is retained rather than rounded into an assumed exact match. The CSV has 37 epoch rows (1 through 37); its maximum mAP50-95 validation row is epoch 31. The run used COCO-initialized YOLO11s, 640px input, batch 32, AdamW, seed 51324, T4 x2, and stopped with 0.888 training hours under the time/early-stopping configuration. The checkpoint loads locally as a one-class detection model with Ultralytics 8.4.183 on CPU.

A visual spot-check of saved validation prediction/label mosaics shows missed and overlapping/low-confidence boxes in some views. This is qualitative evidence only; no per-image error-rate estimate is claimed from the mosaics.

![Kaggle YOLO11s training curves](../../../data/runs/alignment_gigikita_crowding_v4/results.png)

![Kaggle validation prediction mosaic](../../../data/runs/alignment_gigikita_crowding_v4/val_batch0_pred.jpg) The result is a moderate source-label baseline. It does not demonstrate patient-independent performance, clinical validity, or robustness on casual phone photographs. The data has no verified patient IDs and labels were not clinician-adjudicated; the source-declared CC BY 4.0 should be preserved with attribution.

The locked source test was not accessed. Do not tune or claim final generalization from it until the split protocol and annotation limitations are reviewed. The extracted checkpoint, metrics, training curve, arguments, confusion matrices and validation overlays are retained locally for the next controlled method comparison.

## Matched resolution control: 640 px versus 960 px

Notebook [16.22 v2](../../../notebooks/alignment/16_22_gigikita_crowding_yolo11s_960_resolution_control.ipynb) trained the 960 px candidate with 37 epochs and `time=None`. An audit showed that the older 640 px baseline used a different 25-epoch/time-capped schedule, so its apparent score gap could not be attributed to image resolution. Notebook [16.23 v1](../../../notebooks/alignment/16_23_gigikita_crowding_matched_640_control.ipynb) therefore repeated the 640 px setting under the 16.22 recipe on Kaggle T4 x2.

| Setting/result | 640 px matched control | 960 px reference |
|---|---:|---:|
| Epochs | 37 | 37 |
| Seed / optimizer / batch | 51324 / AdamW / 32 | 51324 / AdamW / 32 |
| Training-cell time | 0.761 h | 1.536 h |
| Precision | 64.18% | 65.56% |
| Recall | 65.07% | 63.49% |
| mAP50 | 66.63% | 65.95% |
| mAP50-95 | 30.70% | 30.69% |

The `lr/pg0`, `lr/pg1`, and `lr/pg2` columns match exactly for every one of the 37 epoch rows. The model, initialization, dataset manifest, seed, batch, optimizer, augmentation settings, and epoch schedule are matched; only `imgsz` changes among training choices. Artifact retention differs (`save_period` and output paths), which affects saved files and not the learning schedule. The reported score deltas for 640 minus 960 are -1.38 pp precision, +1.57 pp recall, +0.68 pp mAP50, and +0.01 pp mAP50-95. These are one-seed validation differences; near-identical mAP50-95 does not establish an accuracy gain. The shorter measured training time makes 640 the practical candidate for the next review, but inference latency and memory were not measured.

The selected metrics are the best-checkpoint validation values written in each run's `summary.json`. Both use the same frozen manifest SHA-256 (`44d39f856f80508da95eafcaecd49c362f3e34b9d8b2b0f8729281c54229c488`); `source_test_opened` is false. The 640 checkpoint SHA-256 is `b6dd62a8b3593488323272c8abc9117ad9ad99e05cf0bb6e3c3ac7fcc6aa3834`; it loads locally as a one-class YOLO `DetectionModel`. The complete selected artifact inventory and hashes are in [the download receipt](../../../data/runs/alignment_gigikita_crowding_640_matched_v1/download_receipt.json), and the machine-readable comparison is in [resolution_comparison.json](../../../data/runs/alignment_gigikita_crowding_640_matched_v1/resolution_comparison.json). Kaggle kernel: [16.23 v1](https://www.kaggle.com/code/trihuynhviprovcl/ds-crowding-yolo11s-640-matched-control-20261119/versions/1).

The saved validation mosaics were spot-checked against their label mosaics. Several crowded boxes overlap the source boxes, while some are partial or loosely placed; this is visual inspection of sampled batches only, not an error-rate estimate. In submitted Kaggle v1, an early informational dictionary displayed `imgsz=960` from a stale constant. The actual `model.train` call explicitly passed `imgsz=640`; resolved trainer arguments, `summary.json`, and the checkpoint agree on 640. The local notebook source now defines and asserts 640. This output-label mismatch did not affect the run or metrics.

### Interpretation and next action

This controlled run repairs the schedule confound in the prior resolution comparison. It does not show a meaningful mAP improvement at 640 px; rather, both resolutions deliver practically tied validation mAP under the matched schedule, and 640 uses about half the measured training time. A visual spot-check of the saved mosaics found broad overlap on several positive cases, with some partial or loose boxes; no systematic sample error rate is claimed. Keep 640 as the provisional training candidate on efficiency grounds, not as a more accurate or faster-on-device model. Keep the original source test unopened until any next evaluation protocol is frozen. The evidence remains source-label validation only: no patient-level independence, clinical adjudication, or casual-phone generalization is established. A second seed would be needed before claiming the resolution difference is stable.
## 960 px candidate and protocol audit

Notebook 16.22 v2 completed 37 epochs on T4 x2. The 960 px checkpoint is a candidate result, but comparison with the old 640 px v4 baseline is not controlled. The baseline `args.yaml` records `epochs=25,time=8.0`; 16.22 records `epochs=37,time=null`. Both `results.csv` files have 37 rows, but their cosine learning-rate columns diverge after warmup (for example at epoch 31, 640 LR is about 0.000984 while 960 LR is about 0.0000949). Therefore, the observed metric difference cannot be attributed to input resolution. A matched 640 px control is prepared as notebook 16.23 with `epochs=37,time=None` and the same remaining recipe as 16.22; only `imgsz` changes in that pair.

Paper: [Nebiaj et al., Multi-Class Malocclusion Detection on Standardized Intraoral Photographs Using YOLOv11 (2026)](https://www.mdpi.com/2304-6767/14/1/60). Notebook 16.22 compares 640 px with 960 px. Kaggle v1 used Ultralytics' dynamic `time` epoch budgeting and is excluded from the controlled comparison. Corrected v2 removed that override and completed 37 epochs on T4 x2 (Kaggle reports 5,619.1 seconds). The live log confirms `time=None`, `epochs=37`, the frozen manifest hash, 5,516/1,357 train/validation images, and `source_test_opened=false`.

The Kaggle output reports 1.18 GB; Kaggle CLI completed the download of 65 files (1,179,006,176 bytes) into `.venv/work/kaggle/crowding_960_v2/`. Thirty selected artifacts (25,839,980 bytes) are retained at `data/runs/alignment_gigikita_crowding_960_v2/` with SHA-256 entries in `download_receipt.json`. The standalone and run-folder best checkpoints are byte-identical (SHA-256 `a0cec814cd1ba6e6f6828e2934b541765b245b8e25996922adf22b8420365456`); local Ultralytics loading confirms a one-class `crowded` detection model. The local `summary.json` matches the frozen manifest, has `source_test_opened=false`, and records 5,516 train / 1,357 validation images.

Best-checkpoint re-evaluation recorded in the collected summary: P 65.5553%, R 63.4950%, mAP50 65.9456%, mAP50-95 30.6928%. Old 640 px v4: P 61.5611%, R 61.4822%, mAP50 62.8246%, mAP50-95 27.4443%. Although raw deltas are +3.121 pp mAP50 and +3.249 pp mAP50-95, these are confounded by different training schedules and are descriptive only. The source test remains unopened. Next execute 16.23 and decide resolution effect from the matched 640/960 pair.
