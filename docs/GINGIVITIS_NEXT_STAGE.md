# Visible gingival inflammation stage

## Scope and handoff

The user selected historical YOLOv8s calculus as the current candidate and requested progression to the next label. Preserve both historical 78.03% and conservative-group 49.33% mAP50 findings. Return to calculus phone validation after preparing the remaining label branches. Mobile implementation remains deferred.

## Data decision

Reuse D2 Mendeley 3253gj88rr version 1 and the existing audited normalization. Verify the locked split manifest and each image and label checksum. Use the 747 training and 160 validation images. Source test remains unopened. Prepared mild and marked classes become visible_gingival_inflammation; source no-visible regions become background for this target. Empty annotations mean no positive among source-annotated regions. Source annotation coverage is not a clinical guarantee of image-wide absence.

Source definitions and geometry were approved in src/configs/label_mappings.json with Table 2, caption checks and overlays. Notebook 15.1 creates a separate derived dataset, manifests, counts and seeded training overlays without changing original labels. Patient independence has not been established.

## Research basis and limits

The D2 dataset publication is the basis for label interpretation and region geometry. Its source is https://data.mendeley.com/datasets/3253gj88rr/1 and publication is https://pmc.ncbi.nlm.nih.gov/articles/PMC11462013/. The publication endpoint returned a browser challenge during this turn; prior locally recorded label audit supplies the current mapping evidence.

Chen et al., 2026, HFSA-DETR introduces a spatial-frequency module in RT-DETR and adaptive sparse attention. The abstract reports an internal cohort of 2300 images, external evaluation on 1096 images and external mAP50 93.2%. These are author-reported grading results under their protocol. The abstract alone does not establish equivalent labels, identical D2 splits or a reproducible implementation. Source: https://pubmed.ncbi.nlm.nih.gov/41687952/ .

## Controlled model sequence

| Stage | Implementation | Decision evidence |
|---|---|---|
| 15.1 | Executed binary conversion and visual audit | Counts, hashes, region geometry and background semantics |
| 15.2n / 15.2s | Matched YOLOv8n and YOLOv8s at 640 | Per-region AP, recall and localization; identical split and optimizer |
| 15.3 | Executed baseline error audit | Misses by source grade and overlap with grade-0 regions |
| 15.4 | Executed grade-0 review packet and technical assessment | 49 grade-0 regions across 23 mixed validation images |
| 15.5 | Executed YOLOv8n plus existing ResNet18 ROI filter | Grade-0 regions flagged 49 → 40 of 49; mild regions found 908 → 875 of 931 |
| Conditional | Matched illumination augmentation experiment | First establish errors under uneven lighting and address the observed grade-0 overlap |
| Later | RT-DETR baseline followed by verified HFSA components | Obtain exact full methodology and usable implementation first |

The initial recipe uses 60 epochs, batch 16, seed 42, AdamW, restrained hue/saturation/value augmentation, one T4 worker and periodic checkpoints. This recipe is a project choice. It is not HFSA-DETR reproduction. Changing from three-grade crop classification to one-label full-image localization changes the task; compare each new arm on this fixed task.

## Next milestones

1. Verify executed 15.1 outputs and annotation overlays; package only derived train/validation data and provenance for Kaggle.
2. Add publication receipt and dataset-version attachment, then launch the two prepared baseline notebooks with checksum checks.
3. Collect selected checkpoints and per-image validation predictions. Assess positive-region recall and source-annotation-negative false positives separately.
4. Select a justified method comparison from measured errors.
5. Continue alignment source audit. Existing OMNI tooth boxes require target reannotation for midline diastema and anterior crowding; exact-label dataset readiness remains unresolved. No alignment model has been trained.

Readiness means verified files and executed processing. Unexecuted training notebooks are prepared experiments with no metric claims.

## Region filter experiment

Run the saved binary YOLOv8n followed by the existing R1 ResNet18 classifier on its predicted crops. Preserve the R1 crop and input conventions and use the original three-class argmax collapsed to grade 0 versus grades 1–2 for the first comparison. The fixed detector confidence is 0.25 and the matching IoU is 0.50 as in notebook 15.3. Report grade-0 regions flagged out of 49 and matched positive regions out of 1,870 for both methods, with separate mild and marked recall. Use the current locked validation images and the single-count ambiguous-region convention from 15.3. This is a feasibility run with saved weights. It requires no new training.

An initial check of saved R1 validation predictions on ground-truth crops retained 18 of 49 grade-0 labels and 1,826 of 1,871 positive annotation rows after collapsing grade labels. The historical crop outputs include the two source rows for one conflicting annotation box. They are evidence about the classifier with annotated crops, and the numbers cannot be used as predicted-box pipeline metrics. D2 train has 210 grade-0 regions across 87 images and neither train nor validation contains a fully negative image according to source annotations.

In parallel, prepare a separately reviewed evaluation cohort with wholly negative and mixed images. Use it to measure whole-image false alarms and verify any improvement with thresholds fixed in advance. If the filter helps reduce grade-0 errors while retaining positive recall, retain it for that evaluation. If it does not, prioritize additional reviewed negative training examples and a balanced binary region classifier before architecture or illumination experiments.

The implemented filter leaves 40/49 grade-0 regions flagged and reduces positive recall from 97.01% to 94.97%, with most additional misses in mild regions. Keep both arms as research baselines. The next milestone is a separately reviewed cohort and a targeted negative-data experiment. See [the executed result](reports/gingivitis_binary/roi_filter/RESULTS.md).
