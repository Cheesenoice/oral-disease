# Data Preparation

## Sources and measured outputs

| Task | Source | Original images | Prepared images |
|---|---|---:|---:|
| Visible caries | [Zenodo 14827784](https://zenodo.org/records/14827784), `Dataset.zip` | 6,265 | 2,175 |
| Gingival appearance | [Mendeley 3253gj88rr v1](https://data.mendeley.com/datasets/3253gj88rr/1), `Dataset.rar` | 1,096 | 1,067 |

Both sources declare CC BY 4.0. The original archives and extracted files are immutable. Caries has a matched publisher MD5; gingivitis has a locally recorded SHA-256, not a claimed publisher checksum.

## Processing decisions

1. Use named LabelMe caries rectangles: D/d → `visible_caries`. Do not reuse numeric COCO category IDs across files.
2. Gingivitis: exclude source jaw ROIs 0/1; map 2 → `no_visible_inflammation`, 3/4 → `mild_inflammation`, 5/6 → `marked_inflammation`.
3. Quarantine whole images with missing/unknown/invalid labels or unresolved rotation metadata. Missing labels never become negatives.
4. Preserve stored pixel orientation for EXIF 0/1 after dimension/overlay review; strip EXIF in prepared outputs.
5. Join filename case tokens, identical bytes and conservative pHash/dHash candidates into same-split constraints. Conflicting exact duplicates are quarantined.
6. Encode RGB JPEGs, maximum edge 1280, quality 95, 4:4:4. No upsampling or augmentation is applied.
7. Create approximately 70/15/15 group-aware splits with seed 42 and record all fingerprints.

The revised splits differ from the source papers' benchmark splits. No claim of independently verified patient-disjointness is made.

## Files to inspect

| Output | Location |
|---|---|
| Executed extraction/audit | `notebooks/01_data_audit.ipynb` |
| Executed normalization/split checks | `notebooks/02_prepare_splits.ipynb` |
| Prepared YOLO inputs | `data/processed/{caries,gum}/dataset/` |
| Original/prepared provenance | `data/manifests/*.normalized.csv` |
| Split membership and locks | `data/manifests/*.splits.csv`, `*.lock.json` |
| All excluded image paths/reasons | `docs/reports/data_preparation/quarantine.csv` |
| Figures, class counts and storage measurements | `docs/reports/data_preparation/` |
| [Dataset card](reports/data_preparation/DATA_CARD.md) | Measured counts, transformations, sources and limitations |

Prepared gum images are much smaller than the source images. Caries re-encoding at fixed high quality increases its byte size; the storage table reports this openly rather than claiming compression for every source.
