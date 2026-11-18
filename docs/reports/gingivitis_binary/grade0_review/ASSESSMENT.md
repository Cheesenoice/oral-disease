# Visual and geometric assessment of grade-0 overlaps

Reviewed the source panels for all 23 mixed-label D2 validation images, with detailed comparisons of both models on representative cases. The 23 photos include 49 grade-0 and 227 positive annotated regions. The annotated boxes follow the gingival margin around individual teeth in both groups.

| Model | Grade-0 regions flagged at confidence 0.25 | Median confidence | Median IoU |
|---|---:|---:|---:|
| YOLOv8n | 49/49 | 0.721 | 0.811 |
| YOLOv8s | 49/49 | 0.630 | 0.822 |

The table uses the highest-confidence unmatched prediction with grade-0 IoU ≥0.50 for each region. Positive matches are assigned first using confidence-ordered one-to-one matching. The median IoU values indicate tight localization on the grade-0 annotations rather than incidental overlap. At confidence 0.50, 45/49 and 43/49 regions remain flagged, respectively. `overlap_evidence.csv` records the evidence for each model and region.

In image `137f62b565f0671a8d28`, both models follow a sequence of five adjacent grade-0 boxes on the lower gingival margin. In `8ecbc6c1f164afaa7aa7`, the same behavior is visible on both arches, including around the upper incisors. These patterns are consistent with learning where gingival-region boxes belong. A controlled comparison is needed to determine how much the model relies on location versus appearance.

This is a technical review against original D2 annotations. Individual clinical review decisions remain pending. The observed grade-0 errors occur in mixed images and cannot estimate false-alarm rates on wholly negative photos. Keep both checkpoints as localization baselines. Prioritize an independent reviewed negative and mixed-image cohort for evaluating inflammation discrimination, then use the results to decide whether to add negative images or test a region classification stage.
