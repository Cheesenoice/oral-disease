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
- Kernel: [`trihuynhviprovcl/ds-crowding-yolo11s-v1`](https://www.kaggle.com/code/trihuynhviprovcl/ds-crowding-yolo11s-v1), version 4 (running). Version 1 stopped before training because the assumed input mount contained neither the expected manifest nor ZIP. Version 4 searches all of `/kaggle/input` and prints the mounted entries; its data-integrity cell has not yet been confirmed in the new run.
- Archive SHA-256: `2c103696ee2389a1919bc36f0befa9aaf51ce341d95225679a047d3dfe75ea10`.
- The source test directory and personal images are excluded.

The task is useful as a reproducible engineering baseline. It is not enough to claim clinical reliability, patient-independent performance, or performance on casual phone captures. A low or unstable score should be reported as a data/label limitation, not repaired by filtering validation cases.

No model metrics are available yet. Collect its artifacts after it reaches a terminal state, then update this report with the actual run status and validation results.
