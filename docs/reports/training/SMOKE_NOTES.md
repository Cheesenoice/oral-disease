# Smoke integration evidence

The first submitted versions used data v1 and code v5. Both GPU workers completed three training epochs for each model. The notebook then failed in the source-code appendix because the running notebook kernel had not added the editable installation's `src` directory to its import path. Fresh worker processes resolved the package correctly; the existing notebook process did not. Logs and outputs from these failed notebook versions are preserved, not relabeled as successful runs.

| First smoke | Caries seconds/epoch after first epoch | Gum seconds/epoch after first epoch | Notebook state |
|---|---:|---:|---|
| YOLOv8n | 11.9106 | 6.4399 | ERROR in appendix, both workers completed |
| YOLO26n | 13.8693 | 8.22575 | ERROR in appendix, both workers completed |

These are two-interval estimates from three-epoch integration runs, not final training speed benchmarks or disease-screening performance claims. Cache/loading/startup dominates total elapsed time. Warm-up scheduling also differs from a 100-epoch run, so smoke metrics are excluded from final comparisons.

Fix: bootstrap explicitly inserts the current project's `src` into the notebook process's import path. Add eight bounded parallel copy workers to stage the prepared inputs onto `/tmp`, checking each image/label against the locked manifest SHA-256. This preserves images, labels and splits while avoiding repeated input-mount file access. `/tmp` staging is excluded from published notebook outputs.

Version 2 smoke submissions use data v1/code v7, including these fixes. Require complete notebook execution and two successful workers before full training. Local tests: 27 passed. The complete first baseline's checkpoint and process-state hashes and ZIP CRC were verified after download.
