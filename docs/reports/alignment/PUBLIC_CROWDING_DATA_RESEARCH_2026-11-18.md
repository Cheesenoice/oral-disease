# Public data and paper audit for crowding/spacing - 18 November 2026

Expanded worldwide source search update, 18 November 2026: Kaggle CLI searches for dental crowding/spacing and malocclusion returned no independent dataset. Web search covered Roboflow Universe, Kaggle, Hugging Face, Zenodo, GitHub, and recent papers. Several new candidates surfaced; none is accepted into the project until its actual images, class boxes, duplicates, and source rights are audited.

## Decision summary

No verified public photo dataset found in this search simultaneously provides (1) clean, clinically reviewed intraoral images, (2) exact crowding/spacing labels or boxes, (3) a reusable license, and (4) a published 70-80+ benchmark that can be reproduced. The strongest exact-task papers use private institutional cohorts. Do not report their metrics as this project's results.

The best next reproducible experiment remains a carefully audited benchmark on the already licensed OCCLSAL source, but its validation set is too small to support a stable crowding score. First improve evidence quality and annotation support; do not start another architecture sweep on the same 36 crowding boxes.

## Candidates

| Source / paper | Exact task and evidence | Data / access finding | Decision |
|---|---|---|---|
| Roboflow `crowded-teeth-fipdz` (Dental Examination) | Frozen v1: 1,064 images; one class `Crowded - v1`; train/valid/test 801/147/116. | Page states CC BY 4.0 and one dataset version, but no source/protocol description, patient IDs, or label guidance. | Useful single-class comparator with simpler schema. Candidate for local download/visual and duplicate audit; test must stay unopened. |
| Roboflow `discloration and 6 cases` (GigiKita) | Project overview lists 7,948; frozen v1 export has 7,933 images, seven classes including `crowded`; train/valid/test 5,559/1,584/790. | CC BY 4.0; one version; no provenance, source protocol or annotation description. Overview/version count mismatch of 15 images. | Largest surfaced exact crowding-class candidate. Audit source mixing, overlap, and image relevance before use; do not treat all 7,933 as crowding positives. |
| Roboflow `caries_crowded_calculus` (GigiKita) | 2,730 images; classes `calculus`, `caries`, `crowded`. | CC BY 4.0; one dataset version; no documented provenance. | Candidate for multi-class transfer, but likely related to other GigiKita sets; check duplicates first. |
| Roboflow `caries_crowded2` (GigiKita) | 2,380 images; classes `caries`, `crowded`. | Search index confirms count/classes; project page export details were not independently retrieved in this pass. | Secondary crowding candidate; do not add before same-uploader overlap audit. |
| Roboflow `Crowded` (GigiKita) | 1,167 images; classes `calculus`, `crowded`. | CC BY 4.0; two versions (v1 126, v2 1,167); no provenance/method description. May be a merged/reused source. | Candidate crowding source, but check overlap against `crowded-teeth-fipdz`, the 7.95k GigiKita multi-class set, and our current sources before merging. |
| Roboflow `dental app` | Frozen v8: 167 images; train/valid/test 117/30/20; classes include `diastema`, `spacing`, `misaligned teeth` plus disease labels. | CC BY 4.0; no source provenance or annotation rules. v8 has no preprocessing/augmentation; historical v7 has 401 images, only 30 validation and 20 test, with auto-contrast/augmentation. | Use v8 as the safer fixed candidate; keep `diastema`, `spacing`, and `misaligned teeth` separate pending overlay review. |
| Roboflow `teeth_gap` (random) | Frozen v1: 314 images; classes `misaligned`, `spaced`; train/valid/test 249/45/20. | CC BY 4.0; preprocessing resizes to 512x512 and remaps one class; no source description or clinical definition. | Candidate only for spacing/misalignment audit. `misaligned` must not be relabeled as crowding. |
| Roboflow `Dental Dataset` (Deteksi Kesehatan Gigi) | 10,957 images, 12 object classes including `mal aligned tooth` and `tooth gap`; train/val/test 7,815/2,068/1,074. | Page states CC BY 4.0; no dataset description, class semantics, annotation protocol, or source lineage. | Largest surfaced detection candidate, but its classes are not equivalent to crowding/diastema. Do not count `mal aligned tooth` as crowding or `tooth gap` as diastema without box-level review. |
| Roboflow `oral-disease-union` (Dental Examination) | 1,471 images, five versions; includes `dentalCrowding` among 14 classes. | CC BY 4.0; no dataset description or box definition. No explicit spacing/diastema class. | Potential independent crowding candidate; crowding label only, license/provenance/image quality still need local audit. |
| Roboflow `dentalmate-dataset` (DentalMate6v) | 3,488 images; includes `misaligned-teeth` and `spaced-teeth`. | CC BY 4.0; three versions, no description/provenance. Likely related by name/content to public DentalMates source already used in this project; cannot assume independent. | Do not treat as new independent source until perceptual duplicate/source-lineage comparison against existing DentalMates data. `misaligned` ≠ crowding. |
| Roboflow `SMILE` (New Workspace) | 120 images; classes include `crowding`, `spacing`, duplicate `Spacing`, `SMILE`, `normal`, `0`, `overbite`. | CC BY 4.0; one version; inconsistent class vocabulary, no provenance. | Reject for direct benchmark unless label schema and images are manually audited; too small and likely noisy. |
| Bite2Text / IOP-Compass (2026) | 1,000 orthodontic cases / 5,000 standardized intraoral photos; clinician reports describe crowding/spacing. IOP-Compass adds FDI tooth instance masks and frozen patient splits. | Official university release requires an account. IOP-Compass labels teeth/view, not crowding or spacing boxes; Bite2Text findings are report text, not region boxes. | High-value research lead for weakly supervised image-level labels or tooth-segmentation pretraining, not a drop-in detector dataset. Confirm terms and inspect report/photo pairing before use. |
| MetaDent (2026) | 60,669 dental images; 2,588 expert-reviewed semi-structured annotations; paper reports 18 visual classes and interdental spacing as a concept. | Hugging Face card states CC BY-NC 4.0. Labels are image/text-level and sources combine clinical, public and web data; not object boxes; subpopulation/provenance and patient grouping need inspection. | Possible image-level transfer/weak-label research, not an object-detection benchmark or automatically independent external test. Non-commercial license only. |
| COde (2026) | 50,000 intraoral photos plus reports; case-level disease benchmark includes broad malocclusion. | Public Hugging Face release states CC BY 4.0; labels are extracted/verified from clinical text and do not provide spacing/crowding boxes or a fine-grained consistent orthodontic class. | Potential image-level malocclusion research after access/privacy audit; not same-label detector data. |
| Nebiaj et al., YOLOv11 malocclusion (2026) | AP50 90.1% for crowding and 97.9% for diastema | 5,854 standardized clinical photos; paper explicitly says data are not public and available only by request. | Strong metric reference, not reproducible without authorized data access. |
| Wang et al., DCC-Net (2026) | Crowding categorization accuracy 72.3% maxilla / 73.5% mandible; F1 69.6% / 70.2%. Uses tooth masks, crown centroids and ResNet50. | 2,604 intraoral photos with scan-derived ground truth, plus 100 external test photos; no public dataset release identified. | Good 70% range method reference; private data prevents exact replication. |
| Hertig et al., quantitative crowding (2025) | Lower anterior crowding/Little Irregularity Index: ICC 0.900 and MAD 0.36 mm; LII ICC 0.930 and MAD 0.74 mm. | 125 people with paired intraoral scans/photos; authors publish code/model but not the source cohort. | Best paper to follow for a meaningful geometric crowding task; not a public training-set solution. |
| OCCLSAL ORTHO AI v3 | Current project YOLO11n: overall mAP50 70.01%; diastema AP50 83.0%, crowding AP50 57.0%. | CC BY 4.0 source, 624 train images / 38 validation images after conservative train-side duplicate exclusion; validation contains just 6 diastema and 36 crowding boxes, all occlusal, some posterior. | Most directly reusable licensed source; useful for a bounded source benchmark, not enough evidence for a stable improvement claim. |
| OMNI (2025) | Public paper benchmark tops at mAP50 66.39%; its `TM` tooth-misalignment class AP50 is 51.81%. No direct spacing/diastema target. | 4,166 screened photos and dentist annotations; official repo links the dataset, but no explicit image redistribution license was found. | Do not merge/upload. It is not a better exact-label benchmark and reuse rights remain unresolved. |
| IO150K / Teeth-SEG (2024) | Tooth instance segmentation; includes crowded dentitions and has a small `RGB0.8K` actual-photo subset. | The author page provides a Drive download link, but release is a mixture of 3D-rendered, plaster-model and 800 RGB photos; no explicit data license was verified. Labels are tooth identity/masks, not crowding-grade boxes. | Possible future tooth-segmentation pretraining only after license and label audit; not a direct crowding metric source. |
| GingivitisSeg (Nov 2026) | Reports overall mIoU 76.49%, but this is gingivitis segmentation, not malocclusion. | Linked GitHub repository was empty when checked. | Not a candidate for this branch. |

## Search conclusion and ranked leads

There is no confirmed clean, source-transparent public dataset with **both** crowding and spacing/diastema box labels that can directly replace the current source. The largest new *crowding* lead is GigiKita `discloration and 6 cases` (7,948 images), but it has an undocumented lineage and may share images with same-uploader subsets. A smaller single-class option is `crowded-teeth-fipdz` (1,064 images); `oral-disease-union` (1,471 images) is another candidate. The best new *spacing/diastema* lead is `dental app` (small; inconsistent version sizes), then `teeth_gap` (314 images; only `spaced` and `misaligned`). These are Roboflow user projects, not clinical benchmark datasets; their CC BY badge is useful but does not establish label quality, source provenance, absence of overlap, or patient independence.

The potentially strongest academic lead is Bite2Text/IOP-Compass: standardized clinical photographs, clinician-written crowding/spacing findings, 1,000 patient cases, and patient-disjoint splits. It does **not** provide disease bounding boxes, so a defensible study would be image-level weak-label classification or tooth-mask/geometry adaptation, not the same YOLO detection task. MetaDent and COde similarly provide image/text labels rather than local crowding boxes.

Recommended executable next step: obtain the three frozen exports (`crowded-teeth-fipdz` v1, `discloration and 6 cases` v1, and `dental app` v8) into the ignored `data/inbox/external_alignment/` area. The Roboflow download endpoint returned HTTP 403 without account access and no API key is configured locally; do not request or send credentials in chat. After the archives arrive, inspect only train/valid during this stage, inventory overlays, exact/near duplicates against existing sources and among candidate sources, view types, and per-class boxes. Keep every source test split unopened and excluded. Do not download all GigiKita subsets before overlap checks. If there is no defensible combined train/validation protocol, report source-domain results separately.

## What can be tested without changing the question

1. Keep the two source labels explicit; do not translate broad source labels into clinical diagnoses.
2. Audit and visually review every OCCLSAL validation annotation, image group and box before any new run. The 38-image validation support is too small to claim a stable gain; expand validation only with licensed, reviewed images and preserve groups.
3. Use a fixed, patient/group-disjoint train/validation protocol, record both mAP50 and mAP50-95, per-class AP, precision/recall, false detections on reviewed negatives, and bootstrap uncertainty by source group.
4. Only after data support is sufficient, compare one paper-guided model change at a time. DCC-Net suggests tooth segmentation/ROI and crown-centroid cues; Hertig et al. suggest estimating contact points and an interpretable irregularity index. These require tooth masks/landmarks and proper crowding reference labels, which the current box datasets do not provide.
5. Keep the original full-source score beside any quality-filtered subset result. Define image-quality criteria before examining model predictions, state retained coverage, and never remove difficult validation examples to raise the score.

## Sources

- [Roboflow `crowded-teeth-fipdz`](https://universe.roboflow.com/dental-examination/crowded-teeth-fipdz-o7tqa)
- [Roboflow `discloration and 6 cases`](https://universe.roboflow.com/gigikita/discloration-and-6-cases-yaubs)
- [Roboflow `caries_crowded_calculus`](https://universe.roboflow.com/gigikita/caries_crowded_calculus)
- [Roboflow `Crowded` by GigiKita](https://universe.roboflow.com/gigikita/crowded-n5gle)
- [Roboflow `dental app`](https://universe.roboflow.com/dental-app/dental-app)
- [Roboflow `teeth_gap`](https://universe.roboflow.com/random-dznie/teeth_gap)
- [Roboflow `Dental Dataset` 10,957-image version](https://universe.roboflow.com/deteksi-kesehatan-gigi/dental-dataset-wq8vm-hnipe)
- [Roboflow `oral-disease-union`](https://universe.roboflow.com/dental-examination/oral-disease-union-xfe9b)
- [Roboflow `dentalmate-dataset`](https://universe.roboflow.com/dentalmate6v/dentalmate-dataset)
- [Bite2Text official data page](https://ditto.ing.unimore.it/bite2text/)
- [IOP-Compass official dataset and benchmark page](https://ditto.ing.unimore.it/iop-compass/)
- [MetaDent official data card](https://huggingface.co/datasets/dengwenhui/MetaDent)
- [COde official data card](https://huggingface.co/datasets/SMH-DEV-AI/COde)
- [Nebiaj et al. (2026), YOLOv11 malocclusion](https://www.mdpi.com/2304-6767/14/1/60)
- [Wang et al. (2026), DCC-Net](https://pubmed.ncbi.nlm.nih.gov/41572857/)
- [Hertig et al. (2025), quantitative crowding from intraoral photos](https://academic.oup.com/ejo/article/47/3/cjaf025/8139825)
- [OMNI paper](https://arxiv.org/abs/2505.15637) and [official repository](https://github.com/RoundFaceJ/OMNI)
- [Teeth-SEG / IO150K authors' data page](https://zoubo9034.github.io/TeethSEG/)
- Existing OCCLSAL results: `OCCLUSAL_BASELINE_RESULTS.md` and `OCCLUSAL_COMPARISON_RESULTS.md` in this folder.

## Experiment state

Literature and source audit only. No dataset was added, no new model was trained, and no test split was opened. A new training run is not justified until there is enough reviewed, group-separated validation support and confirmed data rights for any added source.

## Archive intake update - 18 November 2026

The owner later supplied the three candidate exports. They have now been moved to `data/raw/alignment_external/` and audited train/validation-only. The prior download blocker and “no data added” state above are historical and are superseded by the executed audit in [EXTERNAL_ALIGNMENT_CANDIDATE_AUDIT_2026-11-18.md](EXTERNAL_ALIGNMENT_CANDIDATE_AUDIT_2026-11-18.md). Key new finding: GigiKita and FIPDZ have substantial perceptual duplicate candidates, including cross-split conflicts, so their frozen splits must not be merged as independent validation. No test content was opened and no training was started.
