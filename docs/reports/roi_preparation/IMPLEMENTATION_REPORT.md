# Notebook 05 implementation report — 21 October 2026

## Purpose and actual scope

Notebook 05 prepares evidence and data for the next CNN/ResNet classification experiments. It does not train a CNN, modify the original clinical labels, evaluate the held-out test split, or establish phone-selfie generalization.

## Executed operations

| Operation | Actual output |
|---|---|
| Read saved YOLOv8n/YOLO26n v4 results | Detector history CSVs, overall metrics, final rounded per-class console metrics and validation curves |
| Inspect existing validation figures | Confusion matrices and label/prediction mosaics displayed in the executed notebook; no new prediction run |
| Verify source integrity | Parent split lock plus train/validation image and label SHA-256 checks |
| Derive paired views | Tight annotated rectangle and centered 1.5x context; floor/ceil integer bounds, explicitly recorded padding, 224px RGB JPEG quality 95 |
| Preserve provenance | Parent IDs/groups/splits, box index, native size, crop bounds, padding and source/crop hashes in manifest |
| Inspect output | Class counts, native ROI size quantiles, smallest/largest examples per training class and paired overlays |
| Record conclusion | Real saved notebook output, receipt, conclusion JSON and geometry review |

## Measured counts

| Split | No visible inflammation | Mild | Marked | Total ROIs |
|---|---:|---:|---:|---:|
| Train | 210 | 3,829 | 4,930 | 8,969 |
| Validation | 49 | 932 | 939 | 1,920 |

Total: 10,889 ROIs, 21,778 image views, 907 parent images. Test images opened by this stage: zero. The parent split SHA-256 remains `17951a893d64d81ad5d8318abac5fb004c791cc3b411d07adaa387cb1ca39bd3`.

The manifest is `data/processed/gum_roi_v1/manifest.csv`; view files are under its train/val local/context directories. Notebook and report paths are relative to the project root. Context uses the central ROI label; it is not a new annotation for all tissue visible in the wider crop. The boxes cover tooth/gingival-margin appearance, not segmentation masks.

## Findings and limits

- The no-visible-inflammation class is rare (210 training ROIs, 49 validation ROIs). Macro-F1 and per-class recall are essential in the next classification stage.
- Small native regions become visibly blurred at 224px. Resizing adds no new diagnostic information; no low-resolution regions were silently removed.
- Correlated crops within an image are not independent patients. Grouped uncertainty estimation and inherited split isolation are required.
- Saved validation mosaics do not provide enough data for exact per-instance localization-versus-classification error attribution. Export predictions in the later pipeline comparison stage.
- CNN crop performance will require a separate complete-image evaluation using predicted boxes; it cannot be ranked directly against YOLO mAP.

## Engineering and publication work

Shared implementation is in `src/dentalselfie/roi.py`, config in `src/configs/roi.json`, notebook creation in `src/scripts/prepare_roi_notebook.py`. Six ROI tests cover geometry, padding, source/crop tampering, test exclusion and group leakage. Two publication tests prevent stale ready-version receipts. Three public-export tests cover archive contents/CRC, crop tampering and credential-signature rejection. All eleven focused tests pass.

A publication race was fixed: Kaggle may report an old version as ready immediately after a new upload. The publisher now requires the expected next version and matching visibility. A v11 receipt was recovered from remote filenames/sizes; subsequent v12 publication was verified against 48 expanded members. These were inventory checks, not remote content SHA-256 claims.

During public ROI creation, another asynchronous boundary was observed: exact dataset status/metadata can be ready before the search index lists the new dataset. Post-publication verification now checks the exact metadata identity and privacy field rather than requiring immediate search indexing. Public archives are reused when their verified input manifests are unchanged, avoiding duplicate uploads caused only by ZIP timestamps. Notebook execution saves completed cells progressively so publication does not package an older failed output snapshot.

The user subsequently requested public artifacts for authorized multi-account use. Public policy is explicit in `src/configs/kaggle.json`; credentials and personal/pilot photos remain excluded. Prepared data and code visibility was updated, and 03/04 received public QUICK_SAVE version 5 without retraining. Their real training results remain v4. Derived ROI files and complete collected v4 evidence are packaged as separate reusable research datasets. Consult `docs/reports/data_preparation/kaggle_upload_receipt.json` for the actual latest publication versions/readiness.

## Next stage

Verified public handoff is complete: prepared data v1, code v13, ROI v1 and YOLO-results v1. Public 03/04 review versions are v5 (actual training remains v4); public notebook 05 v1 is a lightweight local-execution review, with full figures retained in the code dataset. Publication started no new training session. Exact URLs and verification details are in `docs/reports/publication/PUBLIC_HANDOFF.json`.

Implement notebook 06 and the custom CNN/ResNet baseline workers, then benchmark 2–3 epochs before full scheduling. Distribute model/seed tracks across authorized accounts with shared pinned public inputs. Method ablations in 07 and the complete pipeline evaluation remain future work; no CNN results are claimed here.
