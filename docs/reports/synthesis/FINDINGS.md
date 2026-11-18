# Evidence synthesis and capture readiness

Notebooks 12.1 and 12.2 were implemented and executed locally on 22 October 2026. No training, public upload or held-out test evaluation was started.

| Deliverable | Verified output |
|---|---|
| Research synthesis | 17 classification comparison rows from 6.x-11.x, including the explicitly repeated 09.3/11.1 CLS control; eight detector audit rows from 10.2 |
| Presentation material | ROI progression/minority-recall figure, detector stress figure, question/outcome table, evidence-gap table and source-hash receipt |
| Local inference plumbing | Both existing caries checkpoints verified against the 10.2 audit hashes; three deterministic validation images processed per model |
| Phone capture study | Awaiting real captures; zero phone images processed, no phone accuracy calculated |
| Publication boundary | Photo notebook excluded from code packages; pilot data stays ignored and local |

## Research interpretation

Scratch CNN, residual learning, initialization, context fusion, gating, ordinal learning, source-detail recovery, imbalance objectives and foundation representations have all been investigated. Complexity did not produce monotonic improvement. Historical R1/R2 remain the strongest observed references (macro-F1 57.26%/57.40%); their small difference is not an established winner. Partial DINOv3 adaptation improved its matched frozen CE control but recognized none of the 49 no-visible validation crops. No method establishes a usable three-class clinical screen.

The source-validation caries detector audit is stronger as a provisional demonstration anchor: YOLOv8n/YOLO26n mAP50-95 72.73%/73.01%, with different fixed-confidence precision/recall trade-offs. AP is not classification accuracy and source validation is not ordinary-phone validation. Synthetic stress does not resolve the capture-domain question.

## Actual execution limits

The 12.2 source-validation smoke processed three images twice; it did not rerun the complete 327-image audit. Every image/model result is retained, including a zero-box case if present. Desktop CPU time excludes decoding, warm-up, loading and rendering. The three images are a plumbing check, not a latency benchmark or representative sample.

The real capture inbox is `data/pilot/phone_feasibility/images/`. See `docs/PHONE_PHOTO_FEASIBILITY.md` for the owner handoff. Future galleries and review tables remain local. The 10.1 clinical reviewer sheet is still pending; no clinical labels were invented or changed.

## Next action

The owner supplies ordinary local photos and confirms permission. The agent records view/permission metadata, executes the existing 12.2 notebook and reports capture feasibility/failure modes. Choose a new data or model intervention only after those observations and the pending annotation review. Finalist repeats remain deferred; this 12.x stage supersedes the previously reserved notebook number without implementing repeat training.
