# Raw-source ROI QA review - 21 October 2026

Notebook 08 executed successfully. All 10,889 derived 224px RGB local crops were checked against their saved SHA-256 values. Crop IDs, mapped class labels, parent IDs, observable groups and train/validation membership exactly match ROI v1. All 907 selected original source images and annotation files were checksum-verified. Test pixels opened: zero. No quality-based exclusions or clinical relabeling were introduced.

The agent inspected `aligned_crop_examples.png`: paired crops are anatomically aligned; raw-source versions retain visibly different fine texture in several examples. This visual review validates the geometry and motivates the hypothesis, not clinical labels or an accuracy improvement. Some original crops are still visibly blurred. The gallery uses deterministic training examples across classes and native size strata; it is not a favorable performance subset.

Nine focused raw/old ROI conversion tests pass, including original-coordinate extraction, unknown-class rejection, source/crop tampering and unopened test files. Seven additional publication tests pass (16 relevant tests total). The public dataset inventory verifies 10,895 expanded files with zero name/byte-size mismatches; this is not a download-based remote content hash verification.

Next: run the matched crop/loss study with fixed input hashes. Keep frozen probes separate, and retain all outcomes.
