# DentalSelfie - Repository and Research Handoff

This document is the starting point for a new coding or research agent. Treat `docs/STATUS.md` as the source of truth for the latest state, then read the task-specific plan and report linked below. This repository is public; never add credentials, personal phone captures, or private/pilot review images.

## 1. Project at a glance

DentalSelfie is an offline-first oral screening research project. The current work compares image detection/classification methods for caries, gingival appearance, calculus, and visible tooth crowding/alignment. It is a student prototype, not a diagnostic device. Keep image-level detection, crop-level classification, and usability evidence separate; do not present source-domain validation as clinical performance.

The long-term demo target is Android on a OnePlus Ace 5. Mobile implementation is deferred. Do not add Android code, export models, or start device setup unless the user reopens that phase.

## 2. Read these files in order

| Order | File | Why |
|---|---|---|
| 1 | `README.md` | Project entry point and notebook map |
| 2 | `docs/STATUS.md` | Latest work, evidence, blockers, and next executable action |
| 3 | `KE_HOACH_DO_AN.md` | Original thesis plan; preserve it as historical context |
| 4 | `docs/RESEARCH_NEXT_STEPS.md`, `docs/POST_09_PLAN.md` | Gingival research decisions and completed method stages |
| 5 | `docs/KAGGLE.md` | Published datasets, kernels, execution/recovery workflow |
| 6 | Task-specific report under `docs/reports/` | Evidence and limitations for the active branch |
| 7 | Relevant notebook | The executed notebook is the reproducible experiment record |

Before any Kaggle work, inspect `docs/reports/training/EXECUTION.json` and live remote status. Never replace or duplicate an active/uncollected version.

## 3. Fixed research decisions

| Topic | Decision |
|---|---|
| Caries | Separate detector; Zenodo source; do not infer negatives from missing labels |
| Gingival appearance | Separate detector/classification studies; source labels are not independently clinically adjudicated |
| Calculus | Keep the grouped source experiment separate from other diseases and datasets |
| Crowding/alignment | Keep source/split provenance separate; do not combine conflicting candidate datasets |
| Evaluation | Fixed source splits; validation for selection; no tuning on test; state patient-independence only if verified |
| Core detector progression | YOLOv8n baseline, then matched YOLO26n comparison where the data branch supports it |
| Compute | Local audit/preparation; Kaggle T4 x2 for authorized GPU work; one bounded worker per GPU |
| Mobile | Offline Android is the eventual direction, currently deferred |
| Language | New code, active notebook text, and active docs in English; historical documents remain unchanged |

## 4. Dataset sources and recovery

Raw and prepared datasets are intentionally not committed to Git. Use the publisher's terms and source manifests; do not redistribute a source archive unless its license/terms allow it. Source archives and processed image sets total over 21 GB locally and would make an unsuitable Git repository.

| ID / task | Source and retrieval | Project record / caveat |
|---|---|---|
| D1 - caries | [Zenodo record 14827784](https://zenodo.org/records/14827784); published archive `Dataset.zip` | CC BY 4.0 per `src/configs/sources.json`; API archive URL is recorded there. Verify checksum after download. |
| D2 - gingivitis | [Mendeley Data 3253gj88rr, v1](https://data.mendeley.com/datasets/3253gj88rr/1) | CC BY 4.0 per project source record. Large archive may require browser download. No publisher checksum was recorded; use the local receipt/checksum evidence. |
| D3 - calculus | [Kaggle source dataset](https://www.kaggle.com/datasets/singh868/multi-class-oral-disease-detection-dataset) | Download through Kaggle CLI after local Kaggle authentication. See calculus audit reports before using labels; source includes classes beyond calculus and invalid annotation rows were quarantined. |
| A1 - OCCLSAL orthodontic annotations | [Roboflow dataset v3](https://universe.roboflow.com/dental-app/dental-app/dataset/8) | Read `docs/reports/alignment/` audit before use. Small crowding support; do not claim strong crowding performance from sparse boxes. |
| A2 - GigiKita crowding | [Roboflow dataset v1](https://universe.roboflow.com/gigikita/discloration-and-6-cases-yaubs/dataset/1) | Current focused crowding source. Fixed train/validation derivative, source test untouched; see `GIGIKITA_CROWDING_BASELINE_2026-11-18.md`. Check license/terms before redistribution. |
| A3 - FIPDZ crowded teeth | [Roboflow dataset v1](https://universe.roboflow.com/dental-examination/crowded-teeth-fipdz-o7tqa/dataset/1) | Candidate audit found cross-source/split repeated-scene concerns. Do not merge frozen splits. |

### Project-published Kaggle artifacts

These are recovery/compute handoffs, not a substitute for the source citation or license record:

| Artifact | Link |
|---|---|
| Prepared caries/gingival data v1 | [dentalselfie-data-v1](https://www.kaggle.com/datasets/trihuynhviprovcl/dentalselfie-data-v1) |
| Code and notebooks | [dentalselfie-code](https://www.kaggle.com/datasets/trihuynhviprovcl/dentalselfie-code) |
| Paired gingival ROI crops | [dentalselfie-gum-roi-v1](https://www.kaggle.com/datasets/trihuynhviprovcl/dentalselfie-gum-roi-v1) |
| YOLO v4 model/results package | [dentalselfie-yolo-v4-results](https://www.kaggle.com/datasets/trihuynhviprovcl/dentalselfie-yolo-v4-results) |

Download public Kaggle datasets after configuring the Kaggle CLI locally (never commit `kaggle.json`):

```powershell
kaggle datasets download -d trihuynhviprovcl/dentalselfie-data-v1 -p .venv/work/kaggle/data --unzip
kaggle datasets download -d trihuynhviprovcl/dentalselfie-gum-roi-v1 -p .venv/work/kaggle/roi --unzip
kaggle datasets download -d trihuynhviprovcl/dentalselfie-yolo-v4-results -p .venv/work/kaggle/results --unzip
kaggle datasets download -d singh868/multi-class-oral-disease-detection-dataset -p .venv/work/kaggle/calculus --unzip
```

The CLI downloads only the published dataset versions. Consult `docs/reports/data_preparation/kaggle_upload_receipt.json` and `docs/KAGGLE.md` for version numbers and receipts. For model/kernel outputs, use the exact kernel slug and version recorded in the relevant report; Kaggle CLI example:

```powershell
kaggle kernels output trihuynhviprovcl/dentalselfie-03-baseline -p .venv/work/kaggle/kernel-output
```

Do not assume every result folder has been published as a Kaggle dataset. Compare the recovered files with local `data/runs/` and the run manifest before declaring a complete restore.

## 5. Local layout and artifact policy

| Path | Contents | Git policy |
|---|---|---|
| `src/` | Reusable pipeline, scripts, configs, tests | Commit source; never commit Kaggle credentials |
| `notebooks/` | Audits and executed experiments | Commit public notebooks; exclude notebooks marked local-only / phone feasibility |
| `docs/` | Status, plans, methods, reports | Commit research evidence; exclude personal/pilot photo reports |
| `report/` | Technical walkthrough and figures | Commit if no personal data is embedded |
| `data/raw/`, `data/processed/`, `data/manifests/` | Source archives, normalized images, split files | Not in Git; reconstruct/download as documented |
| `data/pilot/` | Owner/device captures and local feasibility evidence | Never publish |
| `data/runs/` | Training outputs, checkpoints, predictions and metrics | Results/checkpoints are tracked selectively; pilot-named folders stay excluded |
| `.venv/` | Local Python environment and temporary Kaggle staging | Never commit |

Current local inventory: approximately 26 GB under `data/`; `data/runs/` contains 160 `.pt` checkpoints (about 4.2 GB) plus metrics, logs and visual results. Checkpoints use Git LFS. GitHub Git LFS free allowance is 10 GiB; if upload is rejected for quota, do not delete history or silently omit checkpoints. Report the precise failure and use the published Kaggle result packages or a user-selected artifact host.

## 6. Setup and safe Git workflow

Requirements: Windows PowerShell, Python 3.11, Git, Git LFS, and Kaggle CLI only for Kaggle tasks.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,research]"
.\.venv\Scripts\python.exe -m dentalselfie --help
git lfs install
git lfs pull
```

For a clean clone, LFS files are fetched by `git lfs pull`. Review `.gitignore` and `git status --ignored` before adding files. Never use an unrestricted `git add -f data` command. Before every public push, verify:

```powershell
git status --short
git diff --cached --stat
git lfs ls-files
git check-ignore -v src/configs/kaggle.json data/pilot/phone_feasibility
```

## 7. Notebook map

| Group | Purpose / entry points |
|---|---|
| `01`-`02b` | Source audit, deterministic preparation, publication receipts |
| `03`-`04` | YOLOv8n baseline and YOLO26n comparison |
| `05`, `06.1`-`06.3` | Gingival ROI creation; custom CNN, ResNet18 from scratch, ImageNet initialization |
| `07.1`-`09.3` | Context/ordinal methods, error/resolution/imbalance controls, foundation-feature probes |
| `10.1`-`12.1` | Error audits, representation/adaptation study, evidence synthesis |
| `13`-`14` | Calculus source audit and detector experiments |
| `15` | Binary gingival studies, error audit and filtering feasibility |
| `16` | Alignment/crowding source audits, baselines and candidate evaluation |

Executed notebooks retain outputs because they are part of the presentation and audit trail. Do not replace them with blank templates. Notebook `12_2_phone_photo_feasibility.ipynb` and `alignment/16_15_local_phone_alignment_probe.ipynb` are local-only and must never be published.

## 8. Current research state and next action

As of 18 November 2026, the latest focused branch is one-class GigiKita crowding. The train/validation derivative contains 6,873 images (5,516 train / 1,357 validation), zero duplicate-group overlap, zero malformed boxes and zero missing image-label pairs. Five conflicting groups were quarantined; source test remains unopened. There are no verified patient IDs and labels are not clinically adjudicated.

The YOLO11s baseline notebook 16.21 was submitted to Kaggle T4 x2; dataset indexing/submission state must be checked live. The next action is to verify the current kernel state, collect its output, verify metrics/checkpoint, then update `docs/STATUS.md` and the branch report. Do not start a second model until that baseline is collected and reviewed.

Evidence paths:

- `docs/reports/alignment/GIGIKITA_CROWDING_BASELINE_2026-11-18.md`
- `docs/reports/alignment/EXTERNAL_ALIGNMENT_CANDIDATE_AUDIT_2026-11-18.md`
- `docs/reports/alignment/DENTALMATES_VISIBLE_DATASET_AUDIT_2026-11-18.md`
- `docs/reports/alignment/ALIGNMENT_STAGE_CLOSEOUT_2026-11-18.md`
- `docs/reports/training/ROI_09_COMBINED_FINDINGS_2026-10-22.md`
- `docs/reports/training/EXECUTION.json`

## 9. Reporting standards

- Record actual source/version, split fingerprint, config, seed, hardware, runtime, checkpoint, and metrics.
- Distinguish mAP50 from mAP50-95; report per-class support/metrics and false positives/negatives where available.
- Keep benchmark comparisons matched. Do not imply that a method is novel because it was reimplemented from a paper.
- Report failed/negative experiments. Never fabricate clinical labels, results, or paper-reproduction status.
- After an authorized task, update `docs/STATUS.md` with evidence and one next executable action.
- Do not run tests unless requested; if a code change alters a meaningful data conversion, split, archive, or worker path, validate that changed behavior.
