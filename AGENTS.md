# DentalSelfie — agent instructions

## Workflow
- Read `README.md`, `docs/STATUS.md`, then the relevant section of `KE_HOACH_DO_AN.md`. Inspect existing files and Git changes before editing.
- Current scope: completed data audit/normalization and YOLO baselines; active work covers ROI classifier baselines and controlled method comparisons. **Mobile is deferred by the user.** Do not create Android files, device setup tasks or export notebooks now.
- Keep four main folders: `data`, `notebooks`, `src`, `docs`. Create output folders only when needed. Generated Kaggle staging goes under `.venv/work/kaggle/`.
- Work autonomously on authorized tasks. Ask only for missing access or actions requiring a human. Update `docs/STATUS.md` with real results and the next executable action.

## Project decisions
- Fixed sources: D1 Zenodo 14827784, `Dataset.zip`; D2 Mendeley 3253gj88rr v1. Preserve immutable originals and provenance; do not silently substitute datasets.
- Separate caries and gingivitis detectors. Missing cross-disease annotations are not negatives.
- YOLOv8n baseline, YOLO26n comparison; local preprocessing/analysis, expensive training on Kaggle T4 x2, one bounded worker per task/GPU.
- Current notebooks: 01 audit, 02 preparation and 02b publication executed; 03/04 v4 training completed and collected. Create later notebooks only when implementing that stage.
- Research extension requested 21 October: follow sections 8–14 of `KE_HOACH_DO_AN.md`. Notebook 05 is executed: 10,889 train/validation gingival ROI pairs, 21,778 views; source split locks retained and test images unopened. Notebooks 6.1–6.3 now contain independent, step-by-step baselines (custom CNN, manual ResNet18 from scratch, and the same ResNet18 with ImageNet initialization). All 6.x baselines and 07.1/07.2 v1 are complete and collected. Notebook 08 is executed; 09.1/09.2 v1 are complete and collected (verify live status before any action). 09.3 valid v3 completed and both full probes are collected (ResNet18 macro-F1 50.08%, DINOv3 47.83%); latest v4 was cancelled with no outputs. Use SDK version_label=v3 for its results; CLI defaults to latest. Combined findings are in `docs/reports/training/ROI_09_COMBINED_FINDINGS_2026-10-22.md`. Follow `docs/RESEARCH_NEXT_STEPS.md` and `docs/reports/training/roi_recovery_publication.json`; do not replace active or uncollected versions. Preserve locked splits and distinguish crop classification from end-to-end detection.
- Reusable pipeline code in `src/dentalselfie/`; config in `src/configs/`; utilities in `src/scripts/`; tests in `src/tests/`. Notebooks 6.1–6.3 are intentionally self-contained and repeat their teaching code cell-by-cell, as requested for thesis presentation; preserve that standalone readability.
- Post-09 implementation follows `docs/POST_09_PLAN.md`: 10.1 local gingival failure audit and 10.2 existing-weight caries full-image audit. Reports live in `docs/reports/post09/`; temporary stress images in `.venv/work/audit10/`. Read current execution state in STATUS before rerunning. The 11.x representation/adaptation notebooks are conditional future work, not yet implemented.
- Use English for all active documentation, notebook methodology, code and outputs, as requested by the user. Preserve historical source documents unchanged. Long-term app inference remains offline Android; OnePlus Ace 5 is the planned demo phone.
- Latest stage: 11.x full v8 results are collected. Notebooks 12.1 evidence synthesis and 12.2 local phone feasibility are executed; this supersedes the older reserved number for conditional finalist repeats. Read `docs/PHONE_PHOTO_FEASIBILITY.md`. Phone captures are pending. Never publish 12.2 or personal outputs; `package_code` excludes its reserved filename and notebooks marked `dentalselfie.local_only`. Keep all capture inputs/results under ignored `data/pilot/phone_feasibility/`. Do not start new GPU training, open test, or implement mobile in this stage.
- Data-stage notebooks must contain readable processing code, saved real outputs and implementation appendices. Shared modules remain the single implementation; do not replace executed notebooks with empty templates.
- After local data is fully verified, publish the requested private Kaggle data/code artifacts through notebook 02b. This authorizes dataset upload, not automatic GPU training.
- The subsequent user request explicitly authorizes CLI dataset attachment, T4 x2 selection, smoke/full execution and result collection. The active local workflow is `src/scripts/run_kaggle_tracks.py`; inspect `docs/reports/training/EXECUTION.json` and live remote status before starting any additional run.

## Integrity
- Verify source checksums. Before approving `src/configs/label_mappings.json`, inspect actual classes, box conventions and overlays. Agent approval means evidence-backed schema validation, not a user permission request.
- No guessed class indices, silent clipping, invented patient IDs or missing-label negatives. Record quarantine reasons.
- Group exact duplicates and inspect near duplicates/sessions before locking splits. Claim patient independence only with verified IDs.
- Never tune on test. Keep split hashes fixed; future quantization calibration uses train only.
- Never fabricate metrics, clinical labels or paper-reproduction claims. Notebook scaffolds are not completed experiments.
- Never upload personal/pilot photos or credentials. Package by allowlist. The user explicitly requested public research datasets/notebooks on 21 October: `src/configs/kaggle.json` now declares public visibility. Keep prior private-run reports historically accurate. `kernels push` can start GPU execution; publishing completed notebooks uses QUICK_SAVE instead of retraining. Experiments may be distributed across the user's authorized accounts, with fixed input versions and unique model/seed assignments; one account's quota does not limit the research matrix.

## Engineering
- Python 3.11 project venv; keep local tooling light and avoid global package changes. Use `python -m dentalselfie` with editable installation.
- Resolve paths from the actual root. Test meaningful conversion/split/archive/worker behavior when those components change.
- Preserve the original plan in `docs/archive/`. Keep audit evidence in `docs/reports/`. Do not add unrelated model branches, a backend or LLM.
