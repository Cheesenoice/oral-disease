# Frontal alignment research roadmap

## Evidence and target

The user-facing targets are upper midline diastema and anterior crowding on a frontal phone photograph, each displayed with a bounding box. The existing 16.3 YOLO11n experiment used occlusal photos and the broader source crowding label. Its final source validation AP50 was 83.0% for diastema on six boxes and 57.0% for crowding on 36 boxes. These are exploratory results with small support and repeated-looking scenes, not frontal-camera performance.

The YOLO11 malocclusion paper reports AP50 of 97.9% for diastema and 90.1% for crowding in its own institutional data. The study uses 5,854 multi-view standardized clinical photographs and its data are not public. Its box protocol covers the upper central incisors for diastema, and the affected tooth and adjacent teeth for crowding. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12839579/ . The protocol is a reference for a project-specific annotation guide, not a claim of reproduction or expected performance.

OMNI authors report 4,166 photographs from 384 participants, including 534 frontal training and 187 frontal validation photos. OMNI TM annotations include various forms of misalignment and use tooth boxes. The category does not provide our two target boxes. Source: https://github.com/RoundFaceJ/OMNI and https://arxiv.org/abs/2505.15637 . A license for image redistribution has not been verified; the archive is local. Author-reported cohort age ranges from 3 to 48 and may differ from the intended demo population.

## Stage 1: prepare target annotations

1. Audit frontal view, image quality, dentition stage and visible anterior teeth in the OMNI train/validation candidates. The suffix-based queue is a review priority and does not decide view.
2. Annotate both target statuses independently as present, absent or uncertain. Record ungradable images and use tight target boxes following agreed source-adapted rules. Do not treat an omitted label as a negative.
3. Review source annotations and definitions on a pilot set with a dental reviewer. Record independent disagreements and resolve them before bulk annotation. Store annotation provenance, review state and image hashes.
4. Keep source validation held out from training. Group same-patient or same-scene images using verified metadata where available, and compare hashes and visual candidates. State remaining uncertainty. Reserve a separate independent phone-photo evaluation with consent and reviewed labels.
5. Run the dataset audit, class-support and box-size plots, and annotation overlays. The original source test stays unopened. Publish a derivative only after confirming applicable redistribution terms.

## Stage 2: matched frontal baseline comparisons

| Arm | Initialization | Model | Purpose |
|---|---|---|---|
| A | Standard pretrained | YOLO11n | Frontal baseline; notebook 16.2 |
| B | Standard pretrained | YOLO11s | Capacity comparison under the same split and settings |
| C | 16.3 occlusal checkpoint | YOLO11n | Explore source-view transfer on the same frontal training split |

Keep source manifest, epochs, seed, augmentation, batch and image size fixed within a comparison. Record GPU, checkpoint hash, training duration and library version. Compare per-class AP50, AP50–95, precision, recall and false boxes per reviewed negative image. Inspect data coverage and grading ambiguity before tuning thresholds. The 16.3 checkpoint contains a broader crowding concept and only provides initialization; transfer may help or hurt.

## Stage 3: controlled improvements

Choose experiments from observed errors. For small midline gaps compare 640 and 960 input while holding batch and the remaining recipe fixed when hardware allows. For missed crowded groups inspect box rules and add reviewed examples before trying higher model capacity. If localization rather than classification is the bottleneck, compare a published multi-scale detection architecture with an accessible implementation under the same protocol. Keep a per-class decision table and a no-improvement outcome. The thresholds and experimental selection are set on validation only.

An additional paper combines YOLO11 localization with a Swin Transformer for molar occlusion recognition on 1,101 buccal photographs. It supports a two-stage localization and classification design, but the target, view and task differ. Any separate classifier experiment requires matching labels and a new end-to-end comparison. Source: https://www.nature.com/articles/s41598-026-62092-3 .

## Stage 4: independent evaluation and deployment handoff

After choosing model and thresholds, evaluate on reviewed, patient-separated phone photos, including whole-image negatives and ungradable views. Record per-class misses, false boxes per negative image and capture conditions. Then measure latency and memory on OnePlus Ace 5 and verify export parity when the user starts the deferred mobile stage. Do not report the occlusal paper or baseline scores as phone performance.
