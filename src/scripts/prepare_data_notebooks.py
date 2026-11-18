"""Author English research notebooks; shared processing code stays in the package."""
from pathlib import Path
import argparse
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
ONLY = set()

SETUP = '''from pathlib import Path
import json
import sys
import platform
import pandas as pd
from IPython.display import display

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / "src/configs/sources.json").is_file())
REPORT = ROOT / "docs/reports/data_preparation"
REPORT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_colwidth", 85)
print("Python:", platform.python_version())
print("Project:", ROOT)
'''


def write_notebook(name, cells):
    if ONLY and name not in ONLY:
        return
    if name.startswith(("01_", "02_")):
        names = ("audit_annotations", "assign_groups") if name.startswith("01_") else ("prepare_task", "validate_prepared")
        cells = cells + [
            ("md", """## Implementation appendix — readable processing code

The following cell displays the exact shared implementation used above, with syntax
highlighting. Its saved output makes the processing code readable inside this notebook.
The shared module remains the single implementation so fixes do not diverge between notebooks."""),
            ("code", "import inspect\nfrom IPython.display import Code\nfrom dentalselfie import research\n\n"
             + f"for function_name in {names!r}:\n"
             + "    print(function_name)\n    display(Code(inspect.getsource(getattr(research, function_name)), language='python'))\n")]
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(body) if kind == "md"
                                        else nbf.v4.new_code_cell(body) for kind, body in cells])
    notebook.metadata.update(kernelspec={"display_name": "DentalSelfie (Python 3.11)",
                                         "language": "python", "name": "dentalselfie-local"},
                              language_info={"name": "python"})
    target = ROOT / "notebooks" / name
    nbf.validate(notebook)
    nbf.write(notebook, target)
    print("Authored:", target.name)


def main():
    write_notebook("01_data_audit.ipynb", [
        ("md", """# 01 — Source Extraction and Data Audit

## Research objective
Establish what the two original datasets actually contain before choosing training examples.
This notebook preserves source files, records checksums, checks annotation geometry and
produces visual evidence. **Missing labels are never interpreted as healthy cases.**

| Source | Task | Original archive |
|---|---|---|
| Zenodo 14827784 | Visible dental caries | `data/raw/caries/Dataset.zip` |
| Mendeley 3253gj88rr, version 1 | Gingival appearance | `data/raw/gum/Dataset.rar` |

## Workflow
1. Extract and verify original archives.
2. Inventory images and annotation files.
3. Inspect annotation formats, image geometry and label semantics.
4. Report exclusions, duplicates and source limitations.

This is data preparation, not a clinical validation study or a training run."""),
        ("md", "## 1. Reproducible local environment\n\nUse the project Python 3.11 environment with the `research` dependency extra. No GPU is required."),
        ("code", SETUP),
        ("md", "## 2. Extract and verify both sources\n\nCaries uses a publisher MD5. Gingivitis receives a local SHA-256 fingerprint; this is not misrepresented as publisher verification. Extraction is skipped only when every expected member already exists at the expected size."),
        ("code", '''from dentalselfie.research import extract_sources, source_inventory

extraction = extract_sources(ROOT)
extraction.to_csv(REPORT / "archive_verification.csv", index=False)
display(extraction)
'''),
        ("md", "## 3. Inventory the extracted files\n\nCounts are measured from disk, rather than copied from dataset descriptions. Original captions are retained for provenance, but will not be model inputs."),
        ("code", '''inventory = source_inventory(ROOT)
inventory.to_csv(REPORT / "source_inventory.csv", index=False)
display(inventory.groupby(["task", "extension"]).agg(files=("path", "size"), bytes=("bytes", "sum")))
'''),
        ("md", "## 4. Annotation schema inspection\n\nInspect real source rows before the full checks below. No source mapping is approved from a paper alone."),
        ("code", '''for task in ("caries", "gum"):
    source = ROOT / "data/raw" / task / "Dataset"
    label_files = sorted(source.rglob("*.txt"))
    print(f"{task}: {len(label_files):,} text annotation files")
    for path in label_files[:2]:
        print(path.relative_to(ROOT).as_posix())
        print(path.read_text(encoding="utf-8-sig")[:900])
'''),
        ("md", """## 5. Image readability and reproducible fingerprints

Every extracted image is checked. File SHA-256 detects byte-identical files;
64-bit pHash and dHash identify conservative visual similarity candidates.
JPEG draft decoding accelerates preview analysis without changing source pixels or labels.
The cache is reused only when the source path, file size and modification timestamp match."""),
        ("code", '''from dentalselfie.research import inspect_images, audit_annotations, caries_format_summary

images = inspect_images(ROOT, inventory, REPORT / "image_inventory.csv", workers=4)
display(images.groupby("task").agg(images=("image_path", "size"),
                                    min_width=("width", "min"), max_width=("width", "max"),
                                    min_height=("height", "min"), max_height=("height", "max")))
display(images.loc[images.decode_error != "", ["task", "image_path", "decode_error"]])
'''),
        ("md", """## 6. Resolve source formats before combining labels

**Caries:** the canonical source is the named rectangle annotation in LabelMe.
`D` and `d` both map to `visible_caries`. COCO category IDs must be interpreted
inside each JSON file; a numeric ID is not a universal disease label.

**Gingiva:** source classes 0 and 1 are anatomical jaw ROIs. Classes 2–6 correspond
to MGI 0–4 according to [Table 2 and the labeling section of the source paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11462013/).
We cross-check those semantics with actual caption/label examples and visual geometry.
Captions are audit evidence only, never model features."""),
        ("code", '''format_coverage, coco_categories = caries_format_summary(ROOT)
display(format_coverage)
display(coco_categories)
coco_categories.to_csv(REPORT / "caries_coco_categories.csv", index=False)

records, annotations = audit_annotations(ROOT, images)
annotations.to_csv(REPORT / "source_annotations.csv", index=False)
display(annotations.groupby(["task", "source_class"]).agg(
    boxes=("box_index", "size"), valid_boxes=("valid_geometry", "sum")))
'''),
        ("md", """## 7. Verify gingivitis semantics against independent source descriptions

The three expected histograms below were transcribed from the original captions for
these image IDs. Jaw ROI labels are omitted from the histogram. A mismatch would stop
this semantic check rather than silently shifting class indices.
Published table counts are not substituted for measured archive counts."""),
        ("code", '''from collections import Counter

expected_examples = {
    "00929": {2: 5, 3: 7},
    "00001": {3: 4, 4: 4, 5: 4},
    "00733": {5: 6, 6: 6},
}
checks = []
gum_records = records[records.task == "gum"]
for image_id, expected in expected_examples.items():
    matches = gum_records[gum_records.image_path.map(lambda value: Path(value).stem == image_id)]
    assert len(matches) == 1
    observed = dict(sorted(Counter(int(box["source_class"]) for box in matches.iloc[0].source_boxes
                                   if int(box["source_class"]) >= 2).items()))
    checks.append({"image_id": image_id, "caption_expected": json.dumps(expected),
                   "annotation_observed": json.dumps(observed), "matches": observed == expected})
caption_checks = pd.DataFrame(checks)
display(caption_checks)
assert caption_checks.matches.all(), "Investigate the source class interpretation before preparing data"
caption_checks.to_csv(REPORT / "gingivitis_caption_checks.csv", index=False)

measured_gum = annotations[annotations.task == "gum"].groupby(
    ["source_class", "source_split"]).size().unstack(fill_value=0)
display(measured_gum)
measured_gum.to_csv(REPORT / "gingivitis_measured_class_counts.csv")
'''),
        ("md", """## 8. Inspect box geometry visually

We render deterministic source examples across the file list. For gingivitis,
the ordinary YOLO width/height interpretation is compared with a swapped-column
hypothesis because the paper's prose is inconsistent. The selected interpretation
must align with the actual anatomy. These overlays verify annotation geometry,
not the clinical correctness of the original diagnoses.

If any disease box is invalid, its entire image is quarantined. We never repair
out-of-frame boxes by clipping or silently remove one lesion while keeping its image."""),
        ("code", '''import matplotlib.pyplot as plt
from dentalselfie.research import plot_annotation_grid

plot_annotation_grid(ROOT, records, REPORT / "caries_valid_overlays.png", "caries")
plt.show()
plot_annotation_grid(ROOT, records, REPORT / "caries_quarantine_overlays.png", "caries", invalid=True)
plt.show()
plot_annotation_grid(ROOT, records, REPORT / "gingivitis_standard_overlays.png", "gum")
plt.show()
plot_annotation_grid(ROOT, records, REPORT / "gingivitis_swapped_overlays.png", "gum", swap_gum_dimensions=True)
plt.show()
orientation_zero = records[(records.task == "caries") & (records.exif_orientation == 0) & records.eligible]
plot_annotation_grid(ROOT, orientation_zero, REPORT / "caries_orientation0_review.png", "caries")
plt.show()
'''),
        ("md", """### Recorded interpretation after visual review

The standard gingivitis width/height interpretation aligns with the source regions;
the swapped hypothesis visibly misplaces the jaw and gingival boxes. The source
class semantics match the three independent caption examples above.

Caries LabelMe boxes align with stored image pixels. Orientation tag 0 does not
define a rotation: these images are kept in stored-pixel orientation after dimension
checks and the dedicated overlay review above. Defined rotations other than identity
remain quarantined. Prepared JPEGs contain no EXIF orientation metadata. Unknown
source labels `M`/`m` and non-rectangle shapes are quarantined, not guessed."""),
        ("md", """## 9. Audit exclusions and missing annotations

Unannotated caries images remain available in the immutable source directory but
are excluded from supervised training. Explicitly empty, valid annotation files
may represent negative examples. No missing file is converted into an empty label."""),
        ("code", '''reason_counts = (records[["task", "reasons"]].explode("reasons")
                 .dropna().groupby(["task", "reasons"]).size().rename("images").reset_index())
display(reason_counts)
reason_counts.to_csv(REPORT / "annotation_issue_counts.csv", index=False)
display(records.groupby("task").agg(source_images=("sample_id", "size"),
                                     geometry_eligible=("eligible", "sum")))
'''),
        ("md", """## 10. Duplicate candidates and same-split constraints

Before splitting, we join shared filename case tokens, byte-identical images and
visual candidates with pHash distance ≤ 4 **and** dHash distance ≤ 4, with similar
aspect ratios. Similarity creates a conservative same-split constraint, not a claim
that two images show the same patient. Filename-derived case tokens are also not
independently verified patient IDs. Exact duplicate examples are deduplicated;
conflicting duplicate annotations are quarantined.

The resulting holdout is group-isolated under these observable constraints.
It cannot certify that every unknown repeated patient has been detected."""),
        ("code", '''from dentalselfie.research import duplicate_pairs, assign_groups

pairs = duplicate_pairs(records, radius=4)
pairs.to_csv(REPORT / "duplicate_candidates.csv", index=False)
display(pairs.groupby(["task", "relation"]).size().rename("pairs").reset_index()
        if len(pairs) else pd.DataFrame(columns=["task", "relation", "pairs"]))
records = assign_groups(records, pairs)
display(records.groupby("task").agg(source_images=("sample_id", "size"),
                                     retained_candidates=("eligible", "sum"),
                                     constraint_groups=("group_id", "nunique")))
'''),
        ("md", "## 11. Inspect representative visual pairs\n\nThese examples make the duplicate rule reviewable. All detected pairs remain in the CSV, including those not shown here."),
        ("code", '''from PIL import Image

examples = pairs.sort_values(["phash_distance", "dhash_distance", "sample_a"]).head(4)
lookup = records.set_index("sample_id")
if len(examples):
    fig, axes = plt.subplots(len(examples), 2, figsize=(12, 3 * len(examples)), squeeze=False)
    for row_index, pair in enumerate(examples.to_dict("records")):
        for column_index, key in enumerate(("sample_a", "sample_b")):
            image_row = lookup.loc[pair[key]]
            with Image.open(ROOT / image_row.image_path) as image:
                image.draft("RGB", (700, 700))
                axes[row_index, column_index].imshow(image.convert("RGB"))
            axes[row_index, column_index].axis("off")
            axes[row_index, column_index].set_title(
                f"{pair['task']} | pHash={pair['phash_distance']}, dHash={pair['dhash_distance']} | {key}", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORT / "duplicate_pair_examples.png", dpi=140)
    plt.show()
'''),
        ("md", "## 12. Export the audit handoff\n\nNotebook 02 consumes the complete per-image audit, not only the displayed samples. Raw files remain unchanged."),
        ("code", '''from dentalselfie.research import write_json
from IPython.display import Markdown

records.to_json(REPORT / "audit_records.jsonl", orient="records", lines=True, double_precision=15)
quarantine = records[~records.eligible][["task", "sample_id", "image_path", "label_path", "reasons"]].copy()
quarantine["reasons"] = quarantine.reasons.map(lambda items: "; ".join(items))
quarantine.to_csv(REPORT / "quarantine.csv", index=False)
summary = records.groupby("task").agg(source_images=("sample_id", "size"),
                                       retained_candidates=("eligible", "sum")).reset_index()
summary["excluded_images"] = summary.source_images - summary.retained_candidates
summary.to_csv(REPORT / "audit_summary.csv", index=False)
display(summary)
final_reasons = (records[["task", "reasons"]].explode("reasons").dropna()
                 .groupby(["task", "reasons"]).size().rename("images").reset_index())
final_reasons.to_csv(REPORT / "final_exclusion_reasons.csv", index=False)
display(final_reasons)
write_json(REPORT / "audit_protocol.json", {
    "version": "1", "caries_source": "LabelMe named rectangles", "gum_columns": "class cx cy width height",
    "near_duplicate_rule": "pHash<=4 AND dHash<=4 AND aspect ratio within 10%",
    "quarantine_policy": "Any invalid annotation excludes the whole image; missing is not negative",
    "independence_limit": "Observable filename/hash/visual groups; not certified patient-disjoint",
    "orientation_policy": "Stored pixels for EXIF 0/1 with dimension checks; quarantine other rotations",
})
display(Markdown(f"""### Executed conclusion

- Audited **{len(records):,} source images** from both extracted archives.
- Retained **{int(records.eligible.sum()):,} candidates** before final normalization.
- Recorded **{int((~records.eligible).sum()):,} excluded images** with explicit reasons.
- Saved **{len(pairs):,} duplicate/similarity constraints** for grouped splitting.
- No model was trained and no missing label was invented.

Next: inspect the saved geometry evidence, record the verified class mapping, then execute notebook 02."""))
'''),
    ])
    write_notebook("02_prepare_splits.ipynb", [
        ("md", """# 02 — Cleaning, Normalization and Locked Evaluation Splits

## Research objective
Convert only verified source annotations into two reproducible object-detection
datasets. Every retained image must have provenance, valid geometry, an explicit
class mapping and a fixed split. The two diseases remain independent tasks.

## Input and output

| Input | Output |
|---|---|
| Executed notebook 01 and its per-image audit | Compact RGB JPEGs and YOLO labels |
| Reviewed class interpretation and overlays | Group-isolated train/validation/test splits |
| Duplicate constraints and exclusion reasons | CSV manifests, split locks and a data card |

No training or test-set model evaluation occurs in this notebook."""),
        ("md", "## 1. Load the audited handoff and reviewed mapping\n\nBoth sources were inspected in notebook 01. Approval records verify schema interpretation, not clinical diagnoses."),
        ("code", SETUP + '''
from dentalselfie.acquire import digest
from dentalselfie.research import prepare_task, validate_prepared, plot_prepared_grid, write_json

AUDIT_PATH = REPORT / "audit_records.jsonl"
records = pd.read_json(AUDIT_PATH, lines=True)
pairs = pd.read_csv(REPORT / "duplicate_candidates.csv", keep_default_na=False)
mappings = json.loads((ROOT / "src/configs/label_mappings.json").read_text())
assert all(mappings[task]["approved"] for task in ("caries", "gum"))
display(pd.DataFrame([{"task": task, "source_format": item["box_format"],
                       "classes": json.dumps(item["classes"]), "evidence": item["evidence"]}
                      for task, item in mappings.items()]))
print("Audit SHA-256:", digest(AUDIT_PATH))
'''),
        ("md", """## 2. Apply the recorded exclusions

Quarantine whole images with unknown labels, invalid geometry or unsupported
orientation. Keep the audit trail and raw files. Exact duplicate annotation
conflicts remain unresolved source issues, not arbitrary label choices.
Explicitly empty valid annotations may remain as negatives; missing annotations may not."""),
        ("code", '''selected = records[records.eligible].copy()
summary = records.groupby("task").agg(source_images=("sample_id", "size"),
                                       retained_images=("eligible", "sum")).reset_index()
summary["excluded_images"] = summary.source_images - summary.retained_images
display(summary)
display(pd.read_csv(REPORT / "final_exclusion_reasons.csv"))
assert selected.label_path.ne("").all()
assert selected.groupby(["task", "sha256"]).size().max() == 1
'''),
        ("md", """## 3. Declare preprocessing and split settings before materialization

- Resize only when needed: longest edge at most 1,280 pixels, with aspect ratio preserved.
- Convert stored pixels to RGB JPEG, quality 95, 4:4:4 chroma; strip EXIF metadata.
- Preserve normalized box coordinates under resizing. No color enhancement or augmentation.
- Target 70%/15%/15% by observable groups, seed 42. Group sizes can prevent exact ratios.
- Retain original source split names for provenance; these new splits are not the paper's benchmark.

Compact images reduce later upload and GPU input costs. This is one fixed preprocessing
protocol for both old and new models; it is not a claimed accuracy improvement."""),
        ("code", '''SEED = 42
MAX_EDGE = 1280
JPEG_QUALITY = 95
LOCAL_WORKERS = 4

display(pd.DataFrame([{"seed": SEED, "max_edge": MAX_EDGE, "jpeg_quality": JPEG_QUALITY,
                       "local_workers": LOCAL_WORKERS, "target_split": "70/15/15"}]))
'''),
        ("md", "## 4. Materialize both task datasets\n\nThe helper validates source hashes, writes images/labels and creates same-volume hard links for split folders where supported. Locked outputs are reused only under the identical audit/config fingerprint."),
        ("code", '''from time import perf_counter

started = perf_counter()
manifests = []
for task in ("caries", "gum"):
    print(f"Preparing {task} ...", flush=True)
    manifest = prepare_task(ROOT, task, selected, AUDIT_PATH, seed=SEED,
                            max_edge=MAX_EDGE, quality=JPEG_QUALITY, workers=LOCAL_WORKERS)
    manifests.append(manifest)
prepared = pd.concat(manifests, ignore_index=True)
print(f"Prepared {len(prepared):,} images in {perf_counter() - started:.1f} seconds.")
display(prepared[["task", "split", "sample_id", "width", "height", "label_count"]].head(10))
'''),
        ("md", "## 5. Inspect split sizes, negative examples and class balance\n\nClass counts are reported, not artificially balanced by duplicating validation/test examples. A no-inflammation gingival box is a local appearance label, not proof of a healthy patient."),
        ("code", '''split_summary = prepared.groupby(["task", "split"]).agg(
    images=("sample_id", "size"), groups=("group_id", "nunique"),
    boxes=("label_count", "sum"), bytes=("bytes", "sum")).reset_index()
split_summary["image_fraction"] = split_summary.images / split_summary.groupby("task").images.transform("sum")
split_summary["image_mb"] = split_summary.bytes / 1_000_000
split_summary.to_csv(REPORT / "split_summary.csv", index=False)
display(split_summary)
display(prepared.assign(explicit_negative=prepared.label_count.eq(0)).groupby("task").explicit_negative.sum())
'''),
        ("md", """## 6. Verify the complete output, not only samples

For every output, verify image/label hashes, class IDs, normalized box validity and
split materialization. Assert zero cross-split overlap of group IDs, original byte
hashes and prepared image hashes. Every retained visual-candidate pair must stay
in one split. Assert that each declared class appears in every split.

These checks establish observable isolation; unknown repeated patients may still
exist because the sources do not provide independently verifiable patient identifiers."""),
        ("code", '''class_counts = validate_prepared(ROOT, prepared, pairs)
class_counts.to_csv(REPORT / "class_distribution.csv", index=False)
display(class_counts)

from dentalselfie.data import assign_splits
for task in ("caries", "gum"):
    expected = assign_splits(selected[selected.task == task].sort_values("sample_id").to_dict("records"), SEED)
    expected_by_id = {row["sample_id"]: row["split"] for row in expected}
    actual_by_id = prepared[prepared.task == task].set_index("sample_id").split.to_dict()
    assert expected_by_id == actual_by_id
print("All output-integrity, observed-overlap, class-coverage and seed-reproducibility checks passed.")
'''),
        ("md", "## 7. Visualize the prepared labels and balance\n\nThese plots show the actual resized outputs. The class IDs correspond to the target mapping, not the original source IDs."),
        ("code", '''import matplotlib.pyplot as plt

split_summary.pivot(index="task", columns="split", values="images").plot.bar(
    figsize=(8, 4), ylabel="Images", title="Locked dataset split sizes", rot=0)
plt.tight_layout()
plt.savefig(REPORT / "split_sizes.png", dpi=140)
plt.show()

for task in ("caries", "gum"):
    table = class_counts[class_counts.task == task].pivot(index="class_name", columns="split", values="boxes")
    table.plot.bar(figsize=(10, 4), ylabel="Boxes", title=f"{task}: target class distribution", rot=15)
    plt.tight_layout()
    plt.savefig(REPORT / f"{task}_class_balance.png", dpi=140)
    plt.show()
    plot_prepared_grid(ROOT, prepared, REPORT / f"{task}_prepared_overlays.png", task)
    plt.show()
'''),
        ("md", "## 8. Record storage reduction, provenance and split locks\n\nResize/re-encoding changes image bytes, so manifests store both source and prepared SHA-256 fingerprints. Split-lock files bind the source audit, mapping and output CSV."),
        ("code", '''source_bytes = selected.groupby("task").bytes.sum().rename("source_eligible_bytes")
output_bytes = prepared.groupby("task").bytes.sum().rename("prepared_image_bytes")
storage = pd.concat([source_bytes, output_bytes], axis=1).reset_index()
storage["reduction_factor"] = storage.source_eligible_bytes / storage.prepared_image_bytes
storage.to_csv(REPORT / "storage_summary.csv", index=False)
display(storage)

lock_rows = []
for task in ("caries", "gum"):
    path = ROOT / "data/manifests" / f"{task}.lock.json"
    lock = json.loads(path.read_text())
    lock_rows.append({"task": task, "split_sha256": lock["splits_sha256"],
                      "lock_sha256": digest(path), "dataset_yaml": f"data/processed/{task}/dataset/data.yaml"})
locks = pd.DataFrame(lock_rows)
display(locks)
locks.to_csv(REPORT / "dataset_fingerprints.csv", index=False)
'''),
        ("md", "## 9. Publish the data card and executed conclusion\n\nPrepared datasets are local outputs. Training, GPU benchmarking and Kaggle uploads belong to the next stage."),
        ("code", '''from importlib.metadata import version
from IPython.display import Markdown

environment = {name: version(name) for name in
               ("Pillow", "numpy", "pandas", "matplotlib", "ImageHash", "nbclient", "rarfile")}
write_json(REPORT / "preparation_environment.json", environment)

lines = ["# DentalSelfie Data Card — Prepared Dataset v1", "",
         "## Provenance", "",
         "- Caries: https://doi.org/10.5281/zenodo.14827784 (Dataset.zip; publisher MD5 verified).",
         "- Gingivitis: https://doi.org/10.17632/3253gj88rr.1 (original RAR; local SHA-256 recorded).",
         "- Sources declare CC BY 4.0. Retain attribution when sharing derived data.", "",
         "## Measured retained dataset", "",
         "| Task | Source images | Retained | Excluded |", "|---|---:|---:|---:|"]
for row in summary.to_dict("records"):
    lines.append(f"| {row['task']} | {row['source_images']} | {row['retained_images']} | {row['excluded_images']} |")
lines += ["", "## Split images", "", "| Task | Split | Images | Groups | Boxes |", "|---|---|---:|---:|---:|"]
for row in split_summary.to_dict("records"):
    lines.append(f"| {row['task']} | {row['split']} | {row['images']} | {row['groups']} | {row['boxes']} |")
lines += ["", "## Transformations", "",
          "Named caries rectangles map D/d to visible_caries. Gingivitis source 0/1 jaw ROIs are excluded; "
          "2 maps to no_visible_inflammation, 3/4 to mild_inflammation, 5/6 to marked_inflammation.", "",
          "Stored pixel orientation is retained for source EXIF 0/1 after dimension/overlay checks. "
          "Defined non-identity rotations, unknown source classes/shapes and invalid boxes are quarantined. "
          "Images become RGB JPEG quality 95, 4:4:4, longest edge at most 1280; no upsampling or augmentation.", "",
          "New approximately 70/15/15 grouped splits use seed 42. These differ from published source splits. "
          "Original paths, source split names, raw/prepared hashes and class evidence remain traceable.", "",
          "## Limitations", "",
          "Missing caries annotations are not negative labels. Unknown M/m classes are not interpreted. "
          "Conflicting exact-duplicate gingivitis annotations are excluded in their entirety. "
          "Observable grouping uses filename case tokens, byte hashes and conservative visual similarity. "
          "Patient-disjointness is not independently certified; residual undetected repeated patients remain possible.", "",
          "Sources contain supported clinical photography, not a validated phone-selfie population. "
          "Class mappings reflect local visible appearance and are not a full clinical diagnosis. "
          "A separate real-phone study is needed later. Test data must not be used for model selection.", "",
          "## Reproduction", "",
          "Run notebooks/01_data_audit.ipynb then notebooks/02_prepare_splits.ipynb in the pinned local environment. "
          "See data/manifests/*.lock.json and docs/reports/data_preparation/ for exact fingerprints and exclusions."]
card_path = REPORT / "DATA_CARD.md"
card_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
display(Markdown(card_path.read_text()))

print(f"COMPLETED: {len(prepared):,} local prepared images across two tasks; all integrity checks passed.")
print("No model training or Kaggle upload was performed.")
'''),
    ])
    write_notebook("02b_upload_kaggle.ipynb", [
        ("md", """# 02b — Package and Publish the Prepared Data to Kaggle

## Objective
Publish only the verified local outputs of notebooks 01 and 02. Both the data and
the updated code bundle remain **private** under the configured account. This notebook
does not launch a kernel or consume GPU training quota.

## Publication contract
1. Re-check split locks, all image/label hashes and observed overlap constraints.
2. Package approved images/labels, manifests, source attribution and the data card.
3. Upload the data dataset and verify its remote version, readiness and private visibility.
4. Update the code dataset for the current project layout and executed data notebooks.

Raw archives, unannotated/quarantined photos, personal pilot photos and credentials
are excluded from the upload. Upload receipts bind the remote version to a local SHA-256."""),
        ("md", "## 1. Load and revalidate the prepared dataset"),
        ("code", SETUP + '''
from dentalselfie.research import validate_prepared

manifests = pd.concat([pd.read_csv(ROOT / "data/manifests" / f"{task}.splits.csv", keep_default_na=False)
                       for task in ("caries", "gum")], ignore_index=True)
pairs = pd.read_csv(REPORT / "duplicate_candidates.csv", keep_default_na=False)
checks = validate_prepared(ROOT, manifests, pairs)
display(checks.groupby("task").agg(checked_classes=("class_id", "size"), status=("validation", "first")))
display(manifests.groupby(["task", "split"]).size().rename("images").reset_index())
'''),
        ("md", "## 2. Inspect packaging and upload implementation\n\nThese are the actual shared implementations used by the cells below. Uploads are idempotent when the local artifact and recorded remote version match."),
        ("code", '''import inspect
from IPython.display import Code
from dentalselfie.package import package_data, package_code
from dentalselfie.publishing import publish_stage, verify_remote_data

display(Code(inspect.getsource(package_data), language="python"))
display(Code(inspect.getsource(publish_stage), language="python"))
display(Code(inspect.getsource(verify_remote_data), language="python"))
'''),
        ("md", "## 3. Build the data package\n\nThe package preserves two separate YOLO datasets. CRC and SHA-256 checks protect the generated archive; reruns reuse the same verified package."),
        ("code", '''data_stage = package_data(ROOT)
package_receipt = json.loads((data_stage / "package_receipt.json").read_text())
display(pd.DataFrame([package_receipt]))
display(pd.DataFrame([{"file": path.name, "bytes": path.stat().st_size}
                      for path in sorted(data_stage.iterdir()) if path.is_file()]))
'''),
        ("md", "## 4. Publish the private data dataset\n\nExisting authentication is reused. No credential is embedded in the notebook. This cell performs the requested upload and waits for a verified ready/private state."),
        ("code", '''data_receipt = publish_stage(ROOT, "data", data_stage)
display(pd.DataFrame([data_receipt]))
remote_verification = verify_remote_data(ROOT)
display(pd.DataFrame([remote_verification]))
'''),
        ("md", "## 5. Publish reusable public ROI and model evidence\n\nWhen public publication is configured, export the verified train/validation ROI views and collected v4 outputs. No GPU job is launched."),
        ("code", '''from dentalselfie.public_artifacts import package_research
publication_config = json.loads((ROOT / "src/configs/kaggle.json").read_text())
if not publication_config.get("private", True):
    for kind in ("roi", "results"):
        stage = package_research(ROOT, kind)
        display(pd.DataFrame([publish_stage(ROOT, kind, stage)]))
'''),
        ("md", "## 6. Refresh the code dataset\n\nPublish the current source layout and executed data notebooks under the configured visibility."),
        ("code", '''code_stage = package_code(ROOT)
code_receipt = publish_stage(ROOT, "code", code_stage)
display(pd.DataFrame([code_receipt]))
'''),
        ("md", "## 7. Verified publication handoff"),
        ("code", '''from IPython.display import Markdown

receipts = json.loads((REPORT / "kaggle_upload_receipt.json").read_text())
display(pd.DataFrame(receipts).T[["dataset", "version", "private", "status", "artifact_sha256"]])
expected_private = json.loads((ROOT / "src/configs/kaggle.json").read_text()).get("private", True)
assert all(item["private"] == expected_private and item["status"] == "ready" for item in receipts.values())
display(Markdown(f"""### Publication completed

- [Prepared data]({data_receipt['url']}), version **{data_receipt['version']}**, private and ready.
- [Matching code]({code_receipt['url']}), version **{code_receipt['version']}**, private and ready.
- **{len(manifests):,} prepared images** across two independent tasks.
- Local audit, split-lock and upload receipts are saved for reproduction.
- **No training job was launched.** GPU benchmarking and baseline runs are the next stage."""))
'''),
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="Regenerate only the named notebook(s)")
    ONLY = set(parser.parse_args().only or [])
    main()
