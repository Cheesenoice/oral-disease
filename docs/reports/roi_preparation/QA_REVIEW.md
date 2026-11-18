# ROI geometry review — 21 October 2026

The agent inspected `paired_roi_examples.png` after executing notebook 05. The figure includes deterministic smallest/largest-area training ROIs for each of the three classes, with local/context bounds overlaid on the parent image and both actual saved crops.

The reviewed boxes and crops are geometrically consistent with the supplied rectangles. They contain tooth/gingival-margin appearance rather than pixel-level gingival segmentation. Small native ROIs are visibly blurred when resized to 224 pixels; resizing must not be described as recovering diagnostic detail. Class-name text is inherited from source labels; this review does not independently validate clinical grades.

All 10,889 ROI rows retain their parent split/group; 21,778 view files have saved hashes. Train counts are 210/3,829/4,930; validation counts are 49/932/939 in no-visible/mild/marked order. Geometry, explicit padding, source/crop tamper detection and parent-group isolation checks pass in six focused tests. The held-out test images were not opened by this stage.

The v4 detector's final console tables and saved confusion/example figures were inspected through notebook 05. Exact per-instance localization-versus-classification error attribution is not available from those artifacts alone and remains pending prediction export. No change to source annotations, split assignments or clinical labels was made.

Next gate: implement notebook 06 and measure classifier throughput on the unchanged derived recipe. Keep crop metrics distinct from complete-image detection metrics.
