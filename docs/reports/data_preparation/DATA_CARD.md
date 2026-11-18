# DentalSelfie Data Card — Prepared Dataset v1

## Provenance

- Caries: https://doi.org/10.5281/zenodo.14827784 (Dataset.zip; publisher MD5 verified).
- Gingivitis: https://doi.org/10.17632/3253gj88rr.1 (original RAR; local SHA-256 recorded).
- Sources declare CC BY 4.0. Retain attribution when sharing derived data.

## Measured retained dataset

| Task | Source images | Retained | Excluded |
|---|---:|---:|---:|
| caries | 6265 | 2175 | 4090 |
| gum | 1096 | 1067 | 29 |

## Split images

| Task | Split | Images | Groups | Boxes |
|---|---|---:|---:|---:|
| caries | test | 326 | 188 | 952 |
| caries | train | 1522 | 887 | 4651 |
| caries | val | 327 | 189 | 1054 |
| gum | test | 160 | 160 | 1920 |
| gum | train | 747 | 747 | 8969 |
| gum | val | 160 | 160 | 1920 |

## Transformations

Named caries rectangles map D/d to visible_caries. Gingivitis source 0/1 jaw ROIs are excluded; 2 maps to no_visible_inflammation, 3/4 to mild_inflammation, 5/6 to marked_inflammation.

Stored pixel orientation is retained for source EXIF 0/1 after dimension/overlay checks. Defined non-identity rotations, unknown source classes/shapes and invalid boxes are quarantined. Images become RGB JPEG quality 95, 4:4:4, longest edge at most 1280; no upsampling or augmentation.

New approximately 70/15/15 grouped splits use seed 42. These differ from published source splits. Original paths, source split names, raw/prepared hashes and class evidence remain traceable.

## Limitations

Missing caries annotations are not negative labels. Unknown M/m classes are not interpreted. Conflicting exact-duplicate gingivitis annotations are excluded in their entirety. Observable grouping uses filename case tokens, byte hashes and conservative visual similarity. Patient-disjointness is not independently certified; residual undetected repeated patients remain possible.

Sources contain supported clinical photography, not a validated phone-selfie population. Class mappings reflect local visible appearance and are not a full clinical diagnosis. A separate real-phone study is needed later. Test data must not be used for model selection.

## Reproduction

Run notebooks/01_data_audit.ipynb then notebooks/02_prepare_splits.ipynb in the pinned local environment. See data/manifests/*.lock.json and docs/reports/data_preparation/ for exact fingerprints and exclusions.
