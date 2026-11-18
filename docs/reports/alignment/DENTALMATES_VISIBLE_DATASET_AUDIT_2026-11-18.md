# Visible alignment dataset audit and Kaggle run plan - 18 November 2026

## Decision

The owner supplied `Dental.v1-dataset-version-2.yolov8.zip`. The archive was audited and prepared as a two-class, source-faithful visible alignment experiment. The classes are kept as broad dataset labels: `gap_or_spacing` and `malposition_present`. They are **not** renamed to clinician-defined midline diastema or crowding. The literature anchor is the 2026 YOLOv11 intraoral malocclusion study, but this dataset is not that paper's dataset and this work is not a reproduction.

## Data integrity and scope

| Item | Verified result |
|---|---|
| Original archive | `data/raw/alignment/Dental.v1-dataset-version-2.yolov8.zip` |
| Original archive SHA-256 | `5BA4FC0888D5EF007E3AE826CA7BBD12C254E77DDF7CF66BABE48D1D4597E2B6` |
| Prepared train/validation ZIP | `data/processed/alignment_visible_v2/alignment_visible_v2_train_valid.zip` |
| Prepared ZIP SHA-256 | `32c344cf993b9d4c48d452d78de63359325a2cccfeba6f04bb565c1052ff5f25` |
| Source train | 15,034 image views; 3,818 filename-derived groups; 8,225 spacing boxes; 4,161 malposition boxes |
| Source validation | 477 views/groups; 215 spacing boxes; 149 malposition boxes |
| Test | Excluded from Kaggle package and all planned experiments |
| Kaggle dataset | `trihuynhviprovcl/dentalselfie-visible-alignment-v2`, public version 1, approximately 722 MB |
| License | Source archive declares CC BY 4.0; source attribution is embedded |

The archive says Dataset Version 2, while its embedded `data.yaml` points to version 1 and the README image total conflicts with archive members. This remains unresolved and is recorded in the data card. The public source includes identifiable faces and smartphone-interface screenshots. Package filenames are hashed; pixels are not anonymized. No personal/pilot images are included.

Source filename groups are disjoint between train and validation, but group IDs derive from filenames and do not prove patient independence. There are 15,034 train views for 3,818 groups, so augmented views remain. Annotation completeness is unverified: images with no retained target box are background only under the source annotations, not verified healthy negatives.

## Experiments

| Notebook | Method | Purpose | Kaggle kernel |
|---|---|---|---|
| `16_17_visible_yolo11n_baseline.ipynb` | COCO-pretrained YOLO11n, 640 px | Compact baseline | `trihuynhviprovcl/dentalselfie-visible-alignment-yolo11n-20261118` |
| `16_18_visible_yolo11s_comparison.ipynb` | COCO-pretrained YOLO11s, 640 px | Matched capacity comparison | `trihuynhviprovcl/dentalselfie-visible-alignment-yolo11s-20261118` |
| `16_19_visible_cross_view_transfer.ipynb` | OCCLSAL YOLO11n non-head representation transfer; fresh target Detect head | Cross-view transfer ablation | Prepared; push pending GPU slot |

Both submitted notebooks use the same pinned prepared data, seed, image size, batch, optimizer, augmentation, 50-epoch cap, patience 8, and 1.75-hour training budget. The 16.19 method explicitly excludes the OCCLSAL Detect head because its class meanings differ. Kaggle accepted 16.17 and 16.18; both were `QUEUED` on last check. Kaggle rejected the third push because this account reached its two-session GPU limit. Launch 16.19 once a slot becomes available.

## Interpretation boundary

The results can only measure localization against this source's broad labels on its source validation split. They do not establish phone-camera accuracy, clinical validity, exact diastema/crowding detection, patient-level generalization, or readiness for an app. The independent phone probe remains non-evaluable; do not turn source validation into a phone-performance claim.

## Reproducibility artifacts

- Executed source audit notebook (local only; includes identifiable sample previews): `notebooks/alignment/16_16_roboflow_candidate_audit.ipynb`.
- Public train notebooks: `notebooks/alignment/16_17_visible_yolo11n_baseline.ipynb`, `notebooks/alignment/16_18_visible_yolo11s_comparison.ipynb`, and, after method review, `notebooks/alignment/16_19_visible_cross_view_transfer.ipynb`.
- Hashed manifest: `data/manifests/alignment_visible_v2_manifest.csv`.
- Training package includes attribution, data card, manifest, train and validation only.

## Kaggle first-run failures and correction - 18 November 2026

The first YOLO11n and YOLO11s submissions stopped before training at notebook cell 3 because Kaggle materializes the uploaded ZIP as individual dataset files; no ZIP is present under `/kaggle/input`. The repaired input discovery locates `manifest.csv` and the attached dataset root, verifies the manifest hash, confirms counts/split group disjointness, and checks file hashes. The next attempt found the mounted `data.yaml` is read-only, so the notebook now writes its runtime-specific YOLO YAML under `/kaggle/working`, leaving the attached dataset immutable. Both errors occurred before an epoch started. A further push was rejected because its notebook title exceeded Kaggle's 50-character title limit; short matching slugs now avoid that API validation error.

Corrected short-ID YOLO11n kernel `trihuynhviprovcl/ds-align-y11n-r2-20261118` was RUNNING at last check; it was in the manifest/file-integrity cell and had not yet reported training epochs. `trihuynhviprovcl/ds-align-y11s-20261118` was still QUEUED and contains the read-only YAML write, so expect it to stop before training. Once a GPU slot frees, submit corrected YOLO11s under a short new kernel ID and then submit the corrected non-head cross-view transfer notebook. No metrics or model checkpoints have been produced by these runs yet.

## Corrected runs launched - 18 November 2026

The full-file SHA pass proved too slow on the mounted dataset, so it was cancelled before training and replaced by a bounded integrity gate: exact manifest SHA, full split file counts, train/validation group disjointness, and SHA-256 checks for eight fixed-seed image/label pairs per split. This gate passed in the corrected YOLO11n run. Kaggle reports two Tesla T4 GPUs visible; YOLO11n is now scanning its training labels and advancing toward the first epoch. The matching corrected YOLO11s kernel `trihuynhviprovcl/ds-align-y11s-r2-20261118` has also been accepted and is RUNNING. The earlier 11s run failed exactly at the read-only `/kaggle/input/.../data.yaml` write, before training. No model metrics are available yet. The cross-view run remains prepared and awaits a GPU slot.
