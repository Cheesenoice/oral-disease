# Grade-0 review and follow-up evaluation

Open `index.html` to review 49 numbered source grade-0 regions across 23 locked D2 validation images. The left panel shows the source labels. The middle and right panels show unmatched YOLOv8n and YOLOv8s predictions at confidence ≥0.25 and IoU ≥0.50 with a grade-0 box. The gallery is local. Notebook 15.4 verifies the source image checksums and builds the gallery from saved predictions.

None of the 49 grade-0 source boxes overlaps a positive source box at IoU ≥0.50. This geometry check rules out identical or highly overlapping positive source annotations at that threshold. A human review of the visible regions is pending.

## Record a review

Use `review_queue.csv` to record one status for each numbered region. Inspect the source image before the model panels. Record `confirmed_no_visible_inflammation`, `visible_inflammation`, or `uncertain` in `review_status`. Use `review_notes` for the reason, such as insufficient visibility, a questionable annotation boundary, or a source label mismatch. The original D2 annotation stays in the provenance record.

## Prepare independent evaluation data

Collect consented photos with fully visible target regions, including images reviewed as having no visible inflammation anywhere in the intended field of view and images with both visible inflammation and grade-0 regions. Record subject grouping, view and capture conditions. Record ungradable images separately and report their count. An image needs positive and grade-0 region annotations as applicable, with uncertain regions explicitly marked. Keep this evaluation cohort separate from training, including any later negative-data experiment.

Use a declared confidence cutoff and IoU rule for both checkpoints. Report false boxes per reviewed wholly negative image, proportion of wholly negative images with any box, recall of annotated positive regions, and grade-0 overlap on mixed images. Include the number of images and regions with each result. Decide on a training change after the review and comparison. No clinical review labels or external evaluation photos have been added by this notebook.
