# Binary gingival detector error audit

Notebook 15.3 ran both verified checkpoints on all 160 locked D2 validation images using local CPU inference (Ultralytics 8.4.183, 640 input, NMS IoU 0.70). Predictions were matched in confidence order to positive regions at IoU ≥ 0.50, one prediction per region. A grade-0 overlap is a positive-unmatched prediction with IoU ≥ 0.50 against a region annotated as no visible inflammation. Fixed confidence cutoffs 0.10, 0.25 and 0.50 were diagnostic. No deployment threshold was selected.

| Model | Confidence | Mild regions found | Marked regions found | Unmatched predictions | Grade-0 regions overlapped |
|---|---:|---:|---:|---:|---:|
| YOLOv8n v4 | 0.25 | 908/931 (97.53%) | 905/938 (96.48%) | 246 | 49/49 |
| YOLOv8s v3 | 0.25 | 902/931 (96.89%) | 904/938 (96.38%) | 131 | 49/49 |
| YOLOv8n v4 | 0.50 | 869/931 (93.34%) | 861/938 (91.79%) | 105 | 45/49 |
| YOLOv8s v3 | 0.50 | 865/931 (92.91%) | 864/938 (92.11%) | 96 | 43/49 |

One identical positive region carries conflicting mild and marked source grades. It is counted once as ambiguous and excluded from both grade-specific denominators. Both models detected it at all three cutoffs. Total positive support is 1,870 boxes. The per-prediction overlap count can exceed the number of overlapped grade-0 regions because multiple unmatched predictions can overlap a single region. Complete counts and overlaid-box coordinates are in the audit CSVs.

Both models localize source positive regions well, but overlap almost every annotated grade-0 region, including at confidence 0.50. The pattern is consistent with learning expected gingival box placement as a shortcut. Source labels and missed or overlapping boxes need visual review before attributing the pattern to that mechanism. The 49 annotated grade-0 regions are sparse, and every validation image has at least one positive region, preventing a whole-image false-alarm estimate.

Next review grade-0 overlaps alongside matched positive regions in the same images. Prepare a separately reviewed evaluation cohort with photos that contain no visible inflammation and with mixed grade-0 and positive regions. Report false boxes per wholly negative image and region recall. Keep the locked D2 split as the historical comparison. Consider targeted training with reviewed negative regions and images before illumination or architecture experiments. Keep both checkpoints available for the next evaluation.
