# DINOv3 launch workflow repair

## Problems and changes

| Observed problem | Implemented repair | Verification |
|---|---|---|
| CLI-created executions could not retrieve the existing `HF_TOKEN`, although editor-created smoke versions could | `stage_dinov3_kernels.py --publish` now builds an explicit SDK `QUICK_SAVE` request. It never starts a GPU session. Gated runs launch through the authenticated Kaggle editor after checking the existing secret attachment | Both staged full notebooks produced `QUICK_SAVE` requests in a local no-network check. Version 8 of both kernels was actually imported and committed using the web editor; existing `HF_TOKEN` checkboxes were checked, with values masked |
| Installing Transformers into global site-packages downgraded Hub and conflicted with Kaggle's Gradio/Diffusers | Install four pinned HF packages with `--target /kaggle/temp/dentalselfie_hf_runtime --no-deps`; assert import paths; reuse Kaggle's Torch/NumPy/scientific packages | Full 11.1 v8 passed the isolated import assertions, loaded the pinned official weights and started full train feature extraction without dependency-resolver conflict messages |
| Worker processes independently queried Kaggle Secrets | 11.2 reads the secret once in the parent process and passes it only in the child process environment. No credential enters config, files or printed output | Code and reconstructed worker source parse; runtime preflight appears before worker launch |
| One failed worker could leave its sibling running | Launcher terminates and waits for remaining workers in `finally`; closes logs and clears the parent's token variable | Cleanup paths inspected and all notebook code cells statically compiled |

HF runtime pins: Transformers 4.57.1, huggingface-hub 0.36.2, tokenizers 0.22.1, safetensors 0.6.2. Model revision, crops, split locks, augmentations, losses, optimizer and selection rules are unchanged. Tokens are never stored in the notebook or this report.

## Actual full submissions

| Notebook | Version | Kaggle script version ID | Launch method |
|---|---:|---:|---|
| 11.1 frozen representation comparison | 8 | 361137081 | Existing kernel editor: Import Notebook, verify HF_TOKEN attachment, Save & Run All with GPU T4 x2 |
| 11.2 frozen versus partial adaptation | 8 | 361136686 | Same editor workflow; two one-GPU workers |

Both full sources set `RUN_EXPERIMENT=True` and `PHASE='full'`, use raw ROI dataset v1, and keep test unopened. At the initial verification, 11.1 was extracting all 8,969 training crops and 11.2 was queued. Collection must target these exact script version IDs rather than the default notebook page, which may display the last successful smoke version.

Kaggle CLI credentials became invalid during this turn. No replacement credential or new OAuth permission was created; the authenticated browser was sufficient for submission. CLI monitoring/publication requires the owner to restore CLI authentication later. Do not use automatic push/retry loops while these versions are active.

## Collected full result

11.1 v8 completed in 783.2 seconds. The actual executed notebook and output archive are saved under `data/runs/ROI_11_1_full_v8/`; small result tables and the collection verification receipt are in `11_1_full_v8/` beside this report. Both prediction tables contain exactly the 1,920 locked validation crop IDs and correct labels; independently recomputed confusion matrices and metrics agree with the saved results. No test images were opened.

| Frozen representation | Accuracy | Macro-F1 |
|---|---:|---:|
| CLS | 67.97% | 47.83% |
| CLS plus patch mean | 68.54% | 52.21% |

The patch readout improves its matched CLS control by 4.38 percentage points in macro-F1, but remains below historical R2 (57.40%). This is an exploratory validation comparison, not evidence of clinical readiness or statistical significance. Full 11.2 v8 verified all 10,889 crop files in 682.5 seconds and passed HF_TOKEN preflight. Both GPU workers are now training: at run time 784.8 seconds, frozen_CE on GPU 0 had heartbeat age 0 seconds and last2_blocks_norm_CE on GPU 1 had heartbeat age 7 seconds, with neither process exited. No traceback was present. Full metrics are pending.

Next action: monitor 11.2 to completion, then collect the completed outputs and recompute metrics from all 1,920 validation predictions. Never interpret smoke metrics as full results. Do not rerun the collected 11.1 full experiment.

## Final execution update

11.2 v8 is now COMPLETE (2054.6 seconds) and collected. Both 1,920-row predictions and selected epochs are verified. Frozen/adapted macro-F1: 44.16%/47.64%; no-visible recall is zero in both. See `FULL_FINDINGS_2026-10-22.md` for final interpretation and the missing exported-checkpoint limitation. Earlier running-state text above is historical startup evidence. No new run is required for collection.
