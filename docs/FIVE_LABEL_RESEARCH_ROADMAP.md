# Five-label research roadmap | 17 November 2026

## Selected direction

The product targets visible caries, visible calculus, visible gingival inflammation, upper midline diastema and anterior crowding. All displayed findings need localization and dentist review. Improve existing detectors while preparing new-label data in parallel. Historical experiments remain references with their own splits and metrics.

Literature search used SciSpace plus primary author repositories and PubMed. Search summaries and extracted fields are secondary descriptions; explicit primary-source verification is recorded below. Paper metrics are not project results. No method described here has been implemented by this research/planning update.

## Sources and adoption decisions

| Study | Relevant contribution | Verification and adoption |
|---|---|---|
| [HFSA-DETR](https://pubmed.ncbi.nlm.nih.gov/41687952/) | RT-DETR with spatial-frequency transformation in the backbone and adaptive sparse attention in the neck; 2300 internal images and 1096 external images, external P/R/AP50 92.8/91.5/93.2% | Abstract verified. Exact external dataset identity, label protocol, code and weights unresolved. Read full methods and resolve these dependencies before implementing. The 1096 count alone does not prove it is our D2 version. Use as a method candidate, not a completed reproduction. |
| [GumAI external validation](https://scispace.com/papers/external-validation-of-a-mhealth-tool-for-detecting-gingival-ah8ftmwpu6it) | Smartphone frontal gingival screening with external validation; abstract reports sensitivity 0.91, specificity 0.87, accuracy 0.89 against visual annotations | SciSpace abstract only. These are screening metrics, not bounding-box AP. Use the capture/external-validation idea; code, weights and local runtime remain unverified. |
| [Dual CNN periodontal assessment](https://scispace.com/papers/automated-periodontal-assessment-in-orthodontic-patients-a-30pkz0l6i94v) | Specialist-annotated calculus, plaque and inflammation on 1000 frontal/lateral orthodontic photos | Supports localized detection framing. Exact per-class detection metrics and reusable dataset/model access remain unverified. Avoid quoting abstract-level accuracy as calculus AP. |
| [YOLOv11 malocclusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC12839579/) | Structured photograph-based box definitions; AP50 97.9% diastema and 90.1% crowding | Primary article verified; original data by request. Use box definitions adapted and reviewed for our frontal MVP; do not claim these scores transfer to our narrowed classes or phone captures. |
| [OMNI](https://github.com/RoundFaceJ/OMNI) | Public dataset link, COCO layout, six detection baselines, 4166 multi-view photographs | Source previously checked. Misalignment combines crowding and spacing; tooth boxes differ from gap/group boxes. Reannotation needed for our exact labels. No new download occurred in this update. |
| [DCC-Net](https://scispace.com/papers/dental-crowding-categorization-network-dcc-net-explainable-b1hz6jtznmsr) | Segmentation, ROI extraction and crown-centroid-based crowding categorization | SciSpace abstract: upper/lower crowding categorization accuracy around 72-74%. Occlusal-view severity categorization differs from frontal box detection. Geometry is a possible follow-up after localization. |
| [AlphaDent](https://github.com/ZFTurbo/AlphaDent) | Dataset, segmentation train/validation code and released YOLOv8x weights at 640/960; official utility merges classes | Author repository checked. Inspect all six caries definitions before any merge. DSLR domain. Potential secondary source for training and cross-source evaluation; audit source license and groups. Models are large and segmentation-based; deriving boxes or distilling requires its own experiment. |
| [Caries-DETR](https://github.com/XuefenLiu-SZU/Caries-DETR) | Structural-query initialization, intraoral structural pretraining and lesion-aware adaptive loss | Author repository checked for source files/configuration. Requires MMDetection/MMCV and additional pretraining. Code execution, released weight availability and loss integration are not validated. Suitable research comparator or future teacher; mobile suitability unmeasured. |

AlphaDent mirrors: [Zenodo 16582489](https://zenodo.org/records/16582489), [Hugging Face](https://huggingface.co/datasets/ZFTurbo/AlphaDent), author GitHub releases. Metadata lists AlphaDent.zip around 4.9 GB. Access/license verification precedes repackaging. Author code includes patient-oriented dataset handling; inspect real identifiers and overlap before asserting independence.

## Executable experiment order

### Calculus

1. Collect the running 14.4 YOLOv8n 960 result and compare against v3 640. Batch 8 versus 16 and any differing manifests must be disclosed.
2. Complete 14.2 audit: cross-split image hashes, original filename families, visual near-duplicates and box sizes after resize. Keep historical v3 results; a revised grouped split starts a separately named protocol with rerun controls. Boxes around entire teeth may limit strict-IoU AP independently of network capacity.
3. Train matched 640 and P2 controls only if the error/size analysis supports the small-object hypothesis. Record weight transfer coverage and inference costs. P2 is a general engineering ablation; it does not reproduce HFSA-DETR.
4. Try a stronger small-scale teacher (YOLO small versus nano) on the same locked data. Only if it improves validation, consider a separate supervised distillation experiment after checking a concrete published implementation. Measure the extra benefit against the same student trained without distillation.
5. Add dentist-reviewed true negative phone cases and report false boxes per image. Source class absence alone is not a reviewed negative.

### Visible gingival inflammation

1. Build a binary detection dataset from D2 region labels: mild/marked become visible-inflammation positives. Preserve existing groups/splits, quarantine ambiguous labels and keep no-visible regions as reviewed background for this target only.
2. Start with a full-image YOLO nano binary detector. Measure detection/localization and region sensitivity; the previous three-grade ROI macro-F1 is a different task.
3. Compare plain training with restrained illumination augmentation under matched seeds. Overaggressive hue changes may corrupt inflammation cues.
4. Inspect HFSA-DETR full methodology for exact spatial-frequency operator, sparse-attention definition and labels. Implement baseline RT-DETR then each method component separately only after those details are verified. If details stay unavailable, select an explicitly named, reproducible alternative rather than attributing a generic attention module to this paper.
5. Compare each model on new phone images with dentist boxes. Keep capture-quality rejection and uncertainty visible.

### Caries

1. Retain current D1 YOLO26n as the anchor. Audit phone errors and tiny-region recall.
2. Inspect AlphaDent training masks and caries subtype definitions; derive one-visible-caries boxes only after semantic approval. Retain the dataset's own grouping and separate benchmark. Use a distinct untouched cohort for cross-source transfer evaluation.
3. Compare D1-only versus training-source enrichment on fixed evaluation data with cross-source overlap checks. Reusing evaluation photos for pretraining invalidates the holdout.
4. If more capacity is needed, evaluate a small model first. Caries-DETR structural prior and loss weighting are a later explicit reproduction effort with baseline/TSQI/LDLR ablations. A larger teacher is conditional on verified improvement over the student.

### Diastema and anterior crowding

1. Source audit before training: MGR CDE AI candidate frontal views, rights and box definitions; then OMNI frontal training views if reannotation is needed. The existing candidates do not yet supply a verified, ready-to-train dataset for our exact classes.
2. Define one box around the visible upper midline gap and one around the relevant anterior crowded group, following clinician-adapted instructions. Record insufficient views and reviewed target-negative images. Do not infer negatives from missing source labels or use classifier heatmaps as ground truth.
3. Fit one YOLO11n two-class detector as the direct model-family reference to the malocclusion paper. Compare a small model and 960 resolution using the same fully annotated dataset, then select by per-label AP/recall and phone latency.
4. Landmark/spacing geometry is conditional future work requiring tooth identities, correct views and measurement validation. Start with visible presence and location.

## Notebook allocation

| Folder | Planned notebooks | Dependency |
|---|---|---|
| calculus | 14.2 error/split audit; 14.4 existing 960 run; 14.5 matched P2 or small-model comparator | Collect 14.4 and finish audit |
| gingivitis | 15.1 binary data conversion; 15.2 detector baseline; 15.3 illumination/frequency experiment | Review binary mapping; verify paper methods for advanced arm |
| alignment | 16.1 source/box audit; 16.2 two-class baseline; 16.3 matched improvement | Obtain permitted target boxes |
| caries | 17.1 AlphaDent audit; 17.2 controlled source enrichment | Label semantics, rights and duplicate checks |

These are reserved plans. Create train notebooks at their implementation stage with readable Markdown, code, figures and saved real outputs. Publish allowed source artifacts only; personal phone captures stay local. Parallel independent work is calculus result collection, gum conversion and alignment source audit. New model selection requires per-class localization metrics, negative-image false positives, independent phone evidence and actual device measurements.
