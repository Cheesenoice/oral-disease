# Alignment source and annotation audit

Notebook 16.1 inspected the author-linked OMNI ZIP and the public Roboflow source metadata on 18 November 2026. The two project targets are upper midline diastema and visible anterior crowding on frontal photographs.

| Source | Verified evidence | Readiness |
|---|---|---|
| MGR CDE AI OCCLSAL ORTHO AI | Universe declares CC BY 4.0 and lists `Crowding`, `Midline diastema`, and four other classes across 338 project images. Export v3 lists 834 train, 42 validation and 18 test images, and applies noise augmentation with three outputs per training example. | Best candidate for existing labels. Official COCO export endpoint returned HTTP 401 without a Roboflow credential. Frontal coverage, class counts, box semantics and grouping have yet to be verified in the export. |
| OMNI | Author ZIP downloaded, fingerprinted and inspected. COCO category 7 `TM` combines several forms of tooth misalignment. Both inspected COCO license lists are empty. | Source for a frontal reannotation pilot if reuse terms are confirmed. No verified separate target labels. |
| Multi-Class Malocclusion Detection Using YOLOv11 | Paper details photo-based box rules for diastema and crowding. The institutional dataset is available by reasonable request. | Use its annotation procedure as a reference. Data is unavailable for this audit. |

Sources: [Roboflow project](https://universe.roboflow.com/mgr-cde-ai/occlsal-ortho-ai), [Roboflow v3 metadata](https://universe.roboflow.com/mgr-cde-ai/occlsal-ortho-ai/dataset/3), [OMNI release](https://github.com/RoundFaceJ/OMNI), [OMNI paper](https://arxiv.org/html/2505.15637v1), [YOLOv11 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12839579/).

## Measured OMNI archive

| Split | Images | Boxes | Images with `TM` | `TM` boxes | Invalid boxes |
|---|---:|---:|---:|---:|---:|
| Train | 2,481 | 36,449 | 76 | 85 | 2 |
| Validation | 857 | 12,482 | 30 | 42 | 2 |

Archive SHA-256 is `3ea84e0b7f28526f80b000dfd9bfdf4677f54d66e24d5a60eedfc93afe87f525`. There are zero exact byte duplicates and zero shared first-level folder keys across the two development splits. Folder keys do not verify patient identities or exclude near duplicates. Source test content was not opened. The archived annotations do not specify a view per image. The authors report 534 frontal train and 187 frontal validation photos. A seeded twelve-image `TM` gallery shows tooth-level boxes and a mixture of frontal, lateral and occlusal views. Keep the four invalid annotations recorded for quarantine before conversion.

The published OMNI cohort spans ages 3–48 with mean 10.4 years. Frontal examples and dentition stage need visual review for the intended photo scope. Authors describe academic consent and an author-linked download. No dataset redistribution license was verified in the release metadata. The archive and gallery remain local while the terms are unresolved.

## Next executable check

Obtain the official Roboflow v3 COCO ZIP via an authorized Roboflow session. Inspect train and validation only. For each photo record view, visible tooth region, original-file family, target box semantics and both target annotation statuses. Keep augmented versions grouped by original. Filter the two source classes to the defined frontal targets after visual verification. Empty target labels require explicit review.

For a reannotation pilot, specify the upper central incisors and the intervening gap for diastema, and the visibly affected anterior cluster for crowding. The paper marks the upper central incisors for diastema and groups the affected tooth with adjacent teeth for crowding. Have a reviewer confirm how tight and how broad the project boxes should be on sample photos before labeling at scale. Count candidates and verified positives separately. If the Roboflow export lacks suitable frontal support, sample OMNI frontal images for reannotation after verifying reuse terms. Keep uncertain views and ambiguous cases out of the accepted set.

## COCO export intake

Run the following after obtaining the official export. The command creates a review CSV with source class counts, invalid geometry, cross-split byte hash matches and filename-family candidates. Frontal view and both target box decisions start as pending. The published paper reports AP50 of 97.9% for diastema and 90.1% for crowding on its own standardized photographs. Use its annotation method as the reference for the pilot and measure our performance on our own locked evaluation split.

```powershell
.venv/Scripts/python.exe src/scripts/audit_alignment_export.py data/raw/alignment_roboflow/OCCLSAL_ORTHO_AI_v3.coco.zip --output docs/reports/alignment/roboflow_v3_review.csv
```

Validated the intake script on a controlled fixture with a valid box, an out-of-bounds box and a cross-split duplicate. The real export is pending authenticated access.

## Actual Roboflow v3 export

User supplied the official COCO ZIP. The preserved copy is data/raw/alignment_roboflow/OCCLSAL_ORTHO_AI_v3.coco.zip with SHA-256 f318b7336bab8a73dcda93ad99bb7f13d70f79a30f71eca03f87f9200d17fec4.

| Split | Images | Filename families | Diastema boxes | Crowding boxes | Invalid target boxes |
|---|---:|---:|---:|---:|---:|
| Train | 834 | 142 | 183 | 615 | 0 |
| Validation | 42 | 42 | 6 | 36 | 0 |

COCO category IDs are 1 for Crowding and 3 for Midline diastema. Source-positive filename-family counts are 32/6 for diastema and 86/32 for crowding in train/validation. Filename families are provisional. The contact sheets contain 42 sampled slots across classes and splits, with possible repeated images across label sheets. All inspected views are occlusal. Diastema boxes are narrow gap boxes and the crowding examples include posterior boxes. Some validation examples appear visually repeated despite distinct filename families. There are zero identical image hashes and zero shared filename families across train and validation. A 64-bit dHash screen at distance <=4 flagged 210 candidate cross-split pairs involving 12 validation images. Candidates require visual confirmation. Source test content was not read.

Decision: reject direct use of the v3 export for the frontal baseline. Prioritize an independently reviewed frontal photo source with explicit gap and anterior-crowding boxes. OMNI frontal reannotation remains a candidate after dataset reuse terms are verified. Do not count unreviewed absent labels as negatives.

An inspected montage `roboflow_cross_split_candidates.jpg` contains same-scene examples across original train/validation, including train 114 and validation 27. These train examples were in the conservative dHash exclusion used for notebook 16.3. This verification does not establish that every candidate is a duplicate or that other near duplicates are absent.

## OMNI frontal review pilot

The author repository publishes the archive for benchmarking and requests citation. The paper describes consent for academic use and publication. No explicit license for redistributing the dental images or derived datasets was found in the repository or COCO license records. Use the following local audit materials to assess suitability. Confirm publication and distribution terms with the authors before any external dataset publication.

A seeded 30-image contact sheet showed that suffix 1 is a useful screening clue. All six sampled suffix-1 photos looked frontal. Some frontal-looking photos also had suffix 2 or 3. The archive has 538 train and 186 validation suffix-1 photos, while the paper reports 534 and 187 frontal photos. Use the suffix as a review priority and verify each view visually.

The local omni_frontal_review_queue.csv preserves all 3,338 train and validation images. Priority 1 includes 36 suffix-1 images with broad TM annotations (26 train and 10 validation). Priority 2 includes 688 remaining suffix-1 images. Priority 3 includes 2,614 images of other suffixes. No target class or absence is inferred from TM. The two pilot contact sheets include 36 photos across the three priority levels and both splits.

For each photo record view and dentition. Review both target statuses independently as present, absent, or uncertain. Include the upper central incisors and intervening gap in a diastema box. Mark the anterior tooth group when displacement or overlap is visible, using the YOLOv11 study's affected-tooth-plus-neighbors rule as the reference. Use uncertain when the needed area is obscured or the finding is ambiguous. Review the first pilot cases with a dental annotator before drawing training boxes at scale. Preserve train and validation membership and keep source test closed.
Additional candidate: Roboflow Rocky midline-shift-segementation-ns2ce declares CC BY 4.0 and includes MIDLINE-DIASTEMA among 18 classes across 42 images. Its displayed 87.4% mAP50 covers all classes. Source: https://universe.roboflow.com/rocky-g7hdj/midline-shift-segementation-ns2ce . Image view, exact target support and annotation coverage remain unverified. Crowding is absent from its class list.
