# Frontal Alignment Dataset and Paper Check - 18 November 2026

## Research question

Can we reproduce a paper-level frontal diastema detector with data that is both suitable for the phone-photo task and eligible for Kaggle upload?

## Findings

| Source | Evidence | Fit for this project | Decision |
|---|---|---|---|
| Nebiaj et al., *Multi-Class Malocclusion Detection on Standardized Intraoral Photographs Using YOLOv11* (2026) | 5,854 clinical intraoral photos; 340 diastema boxes overall, including 211 frontal instances; frontal diastema AP50 97.8%. Diastema is defined as a distinct gap between maxillary central incisors; its box covers those central incisors. Crowding is a different, mostly occlusal task: 0 frontal, 4 buccal, 5,628 occlusal instances. | Excellent method/label-definition anchor; source images were acquired in a clinic, not phone selfies. | Do not claim reproduction or use its scores as our expected result: its data is explicitly not public and is only available by request due to institutional rules. |
| OMNI dataset (Xue et al., 2025 preprint and author repository) | 4,166 multi-view intraoral photos from 384 participants; the official README reports 903 frontal occlusal photos and professional-dentist annotations. | Plausible local source for the frontal task; our local candidate archive is a derived suffix-1 subset and still needs provisional visual labels and a stronger independent validation cohort. | The author repository links the archive but shows no dataset license or redistribution grant. Keep images local; do not upload them to Kaggle until the rights are clarified. |
| OCCLSAL ORTHO AI v3 (Rocky, Roboflow Universe) | The source page declares CC BY 4.0 and exposes 42 images in its separate midline-shift/diastema project. The already-audited OCCLSAL v3 export contains 834 images and two target classes, but our visual audit found occlusal views and posterior crowding. | License is clear; view is unsuitable as a frontal-phone validation set. The 42-image side project is too small to assume useful support without auditing its image views and target boxes. | Retain as the existing occlusal baseline; do not describe its scores as frontal performance. |
| Okashas Workspace, *Dental Computer Vision Dataset* (Roboflow Universe, 2026) | Page lists 4,773 images, CC BY 4.0, with `gap_or_spacing` and `malposition_present` among 11 classes; the project description does not specify source provenance or annotation rules. A direct unauthenticated dataset request returned HTTP 403. | Potential supplemental source, but its label semantics may combine midline diastema with other spacing and its view mix is unknown. | Do not merge into the target labels until the export, class counts, view composition, and overlays are audited. Browser inventory shows no Roboflow-authenticated session and no API key is configured. |

## Consequence for the experiment

The 2026 YOLOv11 paper supports separating frontal diastema from occlusal crowding and gives a reproducible visual definition. It does not provide data to train on. Our three completed Kaggle runs therefore remain useful only as the licensed occlusal branch; they cannot establish the requested frontal phone capability.

The current local OMNI pilot has provisional train annotations but lacks enough independently reviewed validation labels and has unresolved redistribution terms. The newly found larger Roboflow source is a candidate to audit, not a ready-to-train dataset. The small 42-image source is also only a candidate. No additional frontal Kaggle run is justified until one of these paths produces an auditable, permitted train/validation package.

## Next executable actions

1. Obtain the Roboflow export through an authorized account or API key; audit its train/validation/test split hashes, exact class counts, view types, box semantics, duplicates, and representative overlays before any conversion.
2. If that source is unsuitable, ask the OMNI authors for explicit redistribution permission or use owner-provided, consented phone images with provisional annotations kept local.
3. Build one frontal diastema dataset with a frozen source split and a separately reviewed validation cohort; keep crowding on the occlusal branch in accordance with the paper's view-wise evidence.
4. Only then publish the eligible dataset/notebooks and run matched YOLO11n/YOLO11s plus the planned occlusal-initialized transfer arm on Kaggle.

## Sources

- Nebiaj et al. (2026), *Multi-Class Malocclusion Detection on Standardized Intraoral Photographs Using YOLOv11*, https://www.mdpi.com/2304-6767/14/1/60
- Xue et al. (2025), *Oral Imaging for Malocclusion Issues Assessments: OMNI Dataset, Deep Learning Baselines and Benchmarking*, https://arxiv.org/abs/2505.15637
- OMNI official repository and dataset link, https://github.com/RoundFaceJ/OMNI
- OCCLSAL/Roboflow midline project, https://universe.roboflow.com/rocky-g7hdj/midline-shift-segementation-ns2ce
- Roboflow dental object-detection candidate, https://universe.roboflow.com/okasha/dental-ktzqf

## Roboflow source recheck - 23 November 2026

The official Universe page lists the Okashas Workspace project as **4,773 images, two dataset versions, and 11 classes**, including `gap_or_spacing` and `malposition_present`, with a source-declared CC BY 4.0 license. The local archive `data/raw/alignment/Dental.v1-dataset-version-2.yolov8.zip` already belongs to this same workspace/project: its embedded `data.yaml` records `okasha/dental-ktzqf`, version 1, and the identical 11-class list. Its executed audit records 15,034 train views and 477 validation views; the source README and archive member count contain a version/image-count inconsistency already documented in `DENTALMATES_VISIBLE_DATASET_AUDIT_2026-11-18.md`.

Therefore version 2 is a **new revision of an already-used source**, not a new independent dataset. It may contain edits or changed images, but exact image-level overlap cannot be proven without exporting its ZIP and hashing/near-duplicate screening. Since the user reports the preview looks identical and the source class taxonomy is unchanged, no duplicate download or training is justified now. Reconsider only if a specific v2 delta (new clean images, corrected annotations or useful split support) can be shown. The v2 page remains [Roboflow Dental Computer Vision Dataset v2](https://universe.roboflow.com/okasha/dental-ktzqf/dataset/2); no v2 archive has been downloaded or audited.
