# DentalSelfie

New collaborators and agents: start with [`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md) for project decisions, dataset links, reproducibility, artifact retrieval, and current next steps.

For the reusable, evidence-based training workflow across the current labels, see the [Model Training Playbook](docs/TRAINING_PLAYBOOK.md). It defines task-appropriate metrics, data/split integrity, controlled method comparisons, Kaggle execution, promotion gates, and the verified findings and limitations for each label.

For the selected highest validation results across all current labels, including cohort-specific headline/secondary choices and overlap caveats, see [Final Metric Selection](docs/FINAL_METRIC_SELECTION_2026-11-24.md).

The same report now includes the per-label Kaggle input, notebook sequence, result artifacts, and clone-to-reproduce checklist. Start there when rerunning a selected experiment.

An offline dental-screening research project. Data preparation and initial YOLO training are complete; notebook 05 adds paired gingival crops for the CNN/ResNet research extension. Public research publication was explicitly requested on 21 October. Android implementation is deferred.

Alignment improvement work resumed on 20 November 2026. For the thesis narrative and earlier executed experiments, read the [English progress report](report/README.md), including notebook roles, results and reproducible figures. Latest fixed-split results: [Spacing YOLOv8s vs P2](docs/reports/alignment/SPACING_P2_FULL_SPLIT_RESULTS_2026-11-22.md) and [Crowding YOLO26s/WBF comparison](docs/reports/alignment/CROWDING_CURRENT_WBF_RESULTS_2026-11-22_V2.md). P2 is not promoted; WBF gains mAP50 but loses mAP50-95 versus the single-model arm.

For the Vietnamese, slide-style technical walkthrough of **caries, calculus and gingival appearance**, see [the three-label report](docs/reports/three_label_technical/BAO_CAO_KY_THUAT_BA_NHAN.md). It includes ten generated figures, actual validation prediction mosaics, the data/model workflow, results and speaker notes.

For the Vietnamese, slide-style technical walkthrough of **orthodontic crowding and tooth spacing**, see [the alignment report](docs/reports/alignment/BAO_CAO_KY_THUAT_CHEN_CHUC_KHE_THUA.md). It explains the label definitions, data audit, YOLO box/image processing, baseline and resolution comparison, validation results, and limits. Its 14 figures can be regenerated with `src/scripts/generate_alignment_technical_report.py`.

For the detailed Vietnamese account of **how the alignment datasets were sourced, audited, compared, deduplicated and prepared**, see [the data process report](docs/reports/alignment/BAO_CAO_QUY_TRINH_DATA_CHEN_CHUC_KHE_THUA.md). It includes source/label EDA, split and merge decisions, duplicate evidence and clear labeled image examples; figures are reproducible with `src/scripts/generate_alignment_data_report.py`.

## Start with the executed notebooks

| Notebook | Purpose | State |
|---|---|---|
| [01 — Data audit](notebooks/01_data_audit.ipynb) | Extract archives, inspect annotations, verify geometry, report exclusions and duplicates | Executed; tables and figures saved |
| [02 — Prepare splits](notebooks/02_prepare_splits.ipynb) | Normalize images/labels, lock groups and splits, verify every output, generate the data card | Executed; datasets saved |
| [02b — Upload to Kaggle](notebooks/02b_upload_kaggle.ipynb) | Publish allowlisted data, code, ROI and results under configured visibility | Executed; latest versions and visibility in publication receipt |
| [03 — YOLOv8n baseline](notebooks/03_yolov8n_baseline.ipynb) | Baseline experiments on T4 ×2 | v4 completed; executed evidence in data/runs/E0_full_v4 and docs/reports/training/E0_full_v4 |
| [04 — YOLO26n comparison](notebooks/04_yolo26n_comparison.ipynb) | Comparison under the same protocol | v4 completed; executed evidence in data/runs/E1_full_v4 and docs/reports/training/E1_full_v4 |
| [05 — Validation audit and paired ROI data](notebooks/05_error_audit_and_roi_data.ipynb) | Inspect saved detector evidence and derive train/validation local/context crops | Executed; 10,889 ROIs, 21,778 view files; test images unopened |
| [6.1 — Custom CNN baseline](notebooks/dentalselfie-06-1-cnn.ipynb) | From-scratch convolutional classifier on local gingival ROIs | Public version 3; T4 x2 run complete |
| [6.2 - ResNet18 from scratch](notebooks/dentalselfie-06-2-resnet18-scratch.ipynb) | Manually implemented residual network with random initialization | Public version 3 corrected scratch run complete; version 2 excluded |
| [6.3 - ImageNet ResNet18](notebooks/dentalselfie-06-3-resnet18-pretrained.ipynb) | Same ResNet18 structure with ImageNet initialization | Public version 2 complete; validation results collected |
| [07.1 - Context and gated fusion](notebooks/07_1_context_gated_fusion.ipynb) | Paired local/context features: concatenation vs learned gate, cross-entropy | Public version 1 complete; outputs collected |
| [07.2 - Ordinal combined method](notebooks/07_2_ordinal_combined_method.ipynb) | Local CORAL vs gated local/context CORAL | Public version 1 complete; outputs collected |
| [08 - Source-resolution audit](notebooks/08_source_resolution_and_error_audit.ipynb) | Verified original-source local crops and error analysis | Executed; 10,889 crops; raw ROI dataset public v1 |
| [09.1 - Resolution control](notebooks/09_1_resolution_control.ipynb) | Matched old-crop vs raw-crop ResNet18 CE | Public v1 complete; results verified |
| [09.2 - Imbalance ablation](notebooks/09_2_imbalance_ablation.ipynb) | Class-balanced CE vs Balanced Softmax; R1 shared control | Public v1 complete; results verified |
| [09.3 - Foundation-feature probe](notebooks/09_3_foundation_feature_probe.ipynb) | Frozen ResNet18 vs DINOv3 linear probes | Valid v3 complete; probes verified; latest v4 cancelled |
| [10.1 - Gingival failure audit](notebooks/10_1_gingival_failure_audit.ipynb) | Prediction verification, shared errors, source support, quality strata and review galleries | Executed; technical checks passed; clinical review pending |
| [10.2 - Caries full-image audit](notebooks/10_2_caries_full_image_audit.ipynb) | Existing checkpoints, full-image errors, lesion size and fixed synthetic stress tests | Executed; eight CPU evaluations, 327 validation images each |
| [11.1 - DINOv3 patch readout](notebooks/11_1_dinov3_patch_probe.ipynb) | Frozen CLS versus CLS plus mean spatial patch features | Full v8 complete and collected; patch macro-F1 52.21% |
| [11.2 - DINOv3 partial adaptation](notebooks/11_2_dinov3_partial_adaptation.ipynb) | Frozen CE head versus final two blocks plus LayerNorm adaptation | Full v8 complete and collected; adapted macro-F1 47.64% |
| [12.1 - Research evidence synthesis](notebooks/12_1_research_evidence_synthesis.ipynb) | Completed experiment matrix, figures, presentation narrative and evidence gaps | Executed locally; 17 classification rows and eight detector audit rows |
| [12.2 - Local phone-photo feasibility](notebooks/12_2_phone_photo_feasibility.ipynb) | Private capture inspection and existing-weight caries inference | Local-only; two owner captures tested on 9 November; private findings remain under data/pilot |

The data notebooks contain English methodology, readable code, executed outputs and implementation appendices. Shared functions stay in `src/dentalselfie/` to avoid divergent copies.

The research extension is defined in [Project Plan, sections 8–14](KE_HOACH_DO_AN.md): scratch CNN/ResNet controls, paired local/context inputs, ordinal grading, ablations and complete-pipeline comparison. Notebooks 6.1–6.3 are self-contained and publicly available on Kaggle. All three GPU baseline runs are complete. The 6.2 version 2 initialization error is documented and excluded; corrected version 3 is the valid scratch baseline. Method notebooks 07.1 and 07.2 are complete and collected. C2 remains the provisional macro-F1 reference; see [method findings](docs/reports/training/ROI_METHOD_FINDINGS_2026-10-21.md) for the measured tradeoffs and protocol limitations.

Public research handoff: [prepared data](https://www.kaggle.com/datasets/trihuynhviprovcl/dentalselfie-data-v1), [code/notebooks](https://www.kaggle.com/datasets/trihuynhviprovcl/dentalselfie-code), [paired gingival ROIs](https://www.kaggle.com/datasets/trihuynhviprovcl/dentalselfie-gum-roi-v1), and [YOLO v4 model evidence](https://www.kaggle.com/datasets/trihuynhviprovcl/dentalselfie-yolo-v4-results). Latest verified versions and readiness are in the publication receipt. Fork notebooks or attach these public inputs from authorized runner accounts; publishing scripts remain owned by the dataset owner.

## Measured prepared data

| Task | Original images | Retained | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Caries | 6,265 | 2,175 | 1,522 | 327 | 326 |
| Gingival appearance | 1,096 | 1,067 | 747 | 160 | 160 |

Missing annotations are not healthy examples. Invalid/unknown annotations and conflicting exact duplicates are quarantined. Group isolation covers observable filename/hash/visual constraints; independently verified patient-disjointness is not claimed.

## Alignment branch update

Three owner-supplied Roboflow candidate archives were moved into immutable `data/raw/alignment_external/` and audited on train/validation only. See the executed [notebook 16.20](notebooks/alignment/16_20_external_candidate_audit.ipynb) and [candidate audit report](docs/reports/alignment/EXTERNAL_ALIGNMENT_CANDIDATE_AUDIT_2026-11-18.md). GigiKita and FIPDZ contain repeated-scene candidates across sources and splits, so do not combine their frozen splits or treat their validation images as independent evidence. No new training or test access occurred.

The focused crowding branch now uses only duplicate-grouped GigiKita train/validation data. Notebook 16.23 is the matched 640 px control against 16.22 at 960 px; both use the same 37-epoch learning-rate schedule. See [notebooks 16.21-16.23](notebooks/alignment/), the [split and comparison record](docs/reports/alignment/GIGIKITA_CROWDING_BASELINE_2026-11-18.md), and the [Vietnamese process report for crowding and spacing](docs/reports/alignment/BAO_CAO_QUA_TRINH_RANG_THUA_CHEN_CHUC_VI.md). Kaggle outputs are collected under `data/runs/alignment_gigikita_crowding_640_matched_v1/` and `data/runs/alignment_gigikita_crowding_960_v2/`.

Current alignment results: spacing's official matched reference is YOLO26s + CIoU-NWD at 60.68%/24.29% mAP50/mAP50-95 on the fixed 477-image development view (60.33%/24.10% on the 473-image sensitivity view). The 16.70 context-crop arm is the best AP candidate so far at 62.29%/24.75%, but recall drops 1.86 pp and it fails the predeclared promotion gate. The 16.73 negative-filter arm lowers AP and recall and is not promoted. Notebook 16.74 tests two-class auxiliary `malposition_present` supervision; its short T4 x2 smoke has been submitted, while full training is gated on account quota. See the [context-crop results](docs/reports/alignment/SPACING_CONTEXT_CROP_ABLATION_RESULTS_2026-11-24.md), [negative-filter results](docs/reports/alignment/SPACING_NEGATIVE_FILTER_ABLATION_RESULTS_2026-11-24.md), and [current spacing plan](docs/reports/alignment/SPACING_NEXT_METHOD_PLAN_2026-11-23.md). Crowding's current reference remains YOLO26s + screened FIPDZ at 69.76%/32.76%. Spacing TTA and WBF comparisons did not pass their gates. Source test remains unopened.

## Minimal layout

```text
data/          original archives, extracted sources, prepared datasets and manifests
notebooks/     methodology, processing code and saved execution results
src/           shared code, configuration, utilities and tests
docs/          status, data/Kaggle guides, reports and historical plans
```

- Prepared YOLO inputs: `data/processed/{caries,gum}/dataset/`.
- Split manifests and locks: `data/manifests/`.
- [Data card and results](docs/reports/data_preparation/DATA_CARD.md).
- [Current status](docs/STATUS.md) · [Data guide](docs/DATA.md) · [Kaggle guide](docs/KAGGLE.md).
- `AGENTS.md` contains agent instructions. `.venv` contains the local environment and temporary work; no extra top-level output directories are needed.

## Reproduce locally

```powershell
uv sync --extra dev --extra research
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name dentalselfie-local --display-name 'DentalSelfie (Python 3.11)'
.\.venv\Scripts\python.exe src/scripts/execute_notebook.py notebooks/01_data_audit.ipynb
.\.venv\Scripts\python.exe src/scripts/execute_notebook.py notebooks/02_prepare_splits.ipynb
.\.venv\Scripts\python.exe src/scripts/execute_notebook.py notebooks/02b_upload_kaggle.ipynb
```

The last notebook performs an external upload using existing Kaggle authentication. It follows the visibility in `src/configs/kaggle.json` (currently public) and does not start GPU kernels. Locked data is reused only with matching audit/config fingerprints.

Training uses independent task workers on T4 ×2. The current bounded screening protocol uses 30 epochs, patience 8, a one-hour per-worker budget, a 10-minute per-worker no-progress watchdog and a two-hour Kaggle session cap. Every epoch saves a resumable checkpoint plus scheduler/process RNG state; worker/prefetch states are not serialized, so resumption is not bitwise identical. See the current status before submitting a run.

Latest combined 09.x evidence: [report](docs/reports/training/ROI_09_COMBINED_FINDINGS_2026-10-22.md). Raw-source CB-CE reaches 57.40% macro-F1; frozen ResNet18/DINOv3 probes reach 50.08%/47.83%. The engineering target remains unmet.

## Local web inference demo

Run the disease and aesthetics interface with one selected checkpoint per source label. Follow [`src/web_demo/README.md`](src/web_demo/README.md). Local validation-only sample images are in `data/demo_images/images/`; their source/split manifest is intentionally local-only.
