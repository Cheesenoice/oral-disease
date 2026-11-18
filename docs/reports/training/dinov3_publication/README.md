# DINOv3 notebook publication record

Updated 22 October 2026. Both experiments use public raw ROI dataset `trihuynhviprovcl/dentalselfie-gum-raw-roi-v1/1`, fixed train/validation splits and DINOv3 ViT-S/16 revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`.

| Track | Public notebook | Latest version | State |
|---|---|---:|---|
| 11.1 | [Frozen CLS vs patch mean](https://www.kaggle.com/code/trihuynhviprovcl/dentalselfie-11-1-dinov3-patch-probe?scriptVersionId=361137081) | 8 | Full COMPLETE; collected and verified all 1,920 validation predictions per arm |
| 11.2 | [Frozen vs partial adaptation](https://www.kaggle.com/code/trihuynhviprovcl/dentalselfie-11-2-dinov3-partial-adaptation?scriptVersionId=361136686) | 8 | Full COMPLETE; both arms collected and all 1,920 predictions verified |

See [runner repair](RUNNER_FIX_2026-10-22.md) and [browser submission receipt](full_v8_browser_submission.json) for the current executions. Full 11.1 results are verified: CLS accuracy/macro-F1 67.97%/47.83%; CLS plus patch mean 68.54%/52.21%. Full 11.2 frozen/adapted macro-F1: 44.16%/47.64%; both have zero no-visible recall. See [combined full findings](FULL_FINDINGS_2026-10-22.md). Source publication now uses QUICK_SAVE and never starts GPU work; launch gated runs through the Kaggle editor with the existing secret checked. CLI credentials are currently unavailable. Never put the token in notebook source or outputs. [Smoke findings](SMOKE_FINDINGS_2026-10-22.md) preserve earlier failure history and verified diagnostic results.

Submission and source records are versioned as JSON in this directory. Collected smoke outputs and failed logs are saved under `11_1_smoke_v6/`, `11_2_smoke_v5/`, and the corresponding versioned `.log` files. The earlier `v5` DINO run errors were followed by the successful 11.1 v6 smoke and the corrected 11.2 v6 source.
