# Occlusal source comparison results

Both Kaggle version-1 experiments completed 60 epochs on one Tesla T4 worker. Each session displayed two T4 devices. Output hashes match the execution logs, and checkpoint training metrics match the selected epoch in each results.csv. The final evaluation runs again on the saved best checkpoint, producing small numerical differences from epoch metrics. Notebook code validates the image and label hashes recorded in the attached dataset manifest. The manifest hash itself was not pinned independently by the notebook, so identity across runs rests on the attached dataset version and recorded source checks rather than an independently saved manifest fingerprint.

| Experiment | Best epoch | Input / batch | Precision | Recall | mAP50 | mAP50–95 | Diastema AP50 | Source crowding AP50 | Training time |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n baseline | 12 | 640 / 16 | 54.63% | 78.74% | 70.01% | 32.89% | 83.0% | 57.0% | 414 s |
| YOLO11s | 13 | 640 / 16 | 72.64% | 72.22% | 66.69% | 33.73% | 69.9% | 63.5% | 703 s |
| YOLO11n | 18 | 960 / 8 | 68.90% | 41.67% | 56.60% | 33.54% | 61.0% | 52.2% | 951 s |

All values above use the final best-checkpoint validation output. YOLO11s increases crowding AP50 by 6.5 percentage points relative to the baseline while decreasing diastema AP50 by 13.1 points. YOLO11n 960 decreases both class AP50 values. Its batch size was changed, so the observed difference cannot be isolated to resolution. These are single-seed experiments on a small, dependent source validation set; they do not establish stable improvements.

The 38 target-positive validation images contain only six diastema boxes and 36 broader crowding boxes, and all reviewed examples use occlusal views. Some source crowding examples include posterior teeth. Similar validation photos may represent the same underlying scene. The 960 run changes batch size and uses the library's automatic optimizer. Threshold and per-class precision/recall are source-evaluation outputs, not patient-level diagnostic measures. Independent frontal phone evaluation and reviewed negatives remain unavailable.

Artifacts are stored under `data/runs/alignment_s_v1` and `data/runs/alignment_n960_v1`. Their best checkpoint SHA-256 values are `98271347b8c2996010ffc1e8acb316a1c0da7798828a6cb0daac68e8a2bf3dc7` and `0ad9c8f4a0a261d0fe204bd34eeb566cfa5827a35a27cebd1f7f7b4ba6605d8f` respectively.
