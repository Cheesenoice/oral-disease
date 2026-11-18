# DINOv3 smoke and full-start findings — 22 October 2026

Latest verification: 11.2 v7 completed both smoke arms. Gradient routing and checkpoint restoration passed, and both 48-row prediction tables reproduce their saved metrics. Frozen/adapted macro-F1: 27.72%/26.54%; these are one-epoch pipeline diagnostics. The pip messages concern conflicts with unused Gradio/Diffusers packages and were non-fatal. See `11_2_smoke_v7/verification.json`. 11.1 latest v7 still fails at `HF_TOKEN` retrieval; no full job is active. Historical queue states below are superseded by this update.

## Observed execution states

| Track / version | Result | Evidence and interpretation |
|---|---|---|
| 11.1 v5 | Failed before loading DINOv3 | Kaggle Secrets returned HTTP 400 for `HF_TOKEN`; no metric |
| 11.1 v6 smoke | Completed | Model loaded at pinned revision. 48 balanced validation crops per representation; predictions and scores recomputed exactly from the saved CSVs. CLS: accuracy 39.58%, macro-F1 38.96%. CLS+mean-patch: accuracy 41.67%, macro-F1 40.86%. This is only a smoke diagnostic, not a research result or evidence of improvement. Outputs are saved under `11_1_smoke_v6/`. |
| 11.1 v7 full | Failed before feature extraction | New CLI-published version again received HTTP 400 reading `HF_TOKEN`; no full prediction table or selected probe. See `11_1_full_v7_failure.log`. The secret binding may not have carried to this newly published version. |
| 11.2 v5 smoke | Frozen arm passed; adaptation arm failed | Frozen CE: 48 balanced validation examples, accuracy 37.50%, macro-F1 27.72%, and `SMOKE_CHECKPOINT_RESTORE_OK`. The adaptation arm exposed an incorrect module path before training. Its traceback is preserved under `11_2_smoke_v5/`. |
| 11.2 v6 smoke | Queued at last check | Source corrects the model path and verifies gradient routing. No result has yet been collected. |

All smokes use a 48-row balanced validation sample (16 per class). Treat their scores only as execution diagnostics. Test remains unopened.

## Implementation correction

The failing path was `backbone.model.layer`. The pinned Hugging Face Transformers 4.57.1 DINOv3 ViT implementation defines the transformer blocks directly as `DINOv3ViTModel.layer`; the corrected path is `backbone.layer`. Parameter-name assertions were updated to match the actual `backbone.layer.10.*` and `backbone.layer.11.*` names, while gradient checks use the corresponding backbone-local `layer.10.*` and `layer.11.*` names. See the [version-pinned upstream implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py).

## Full-run gate

Do not treat 11.1's smoke score as the full result. Its full version failed because the current version could not access the attached Hugging Face secret. The user's secret worked for 11.1 v6 and 11.2 v5, but the later CLI-created versions failed to retrieve it. Reattach `HF_TOKEN` to the latest 11.1 v7 and 11.2 v6 kernel revisions, confirm 11.2 v6 completes smoke, and then launch bounded full runs. Current Kaggle quota was 2.80 hours after the last reading. Never store or print the secret.
