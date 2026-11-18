"""Create notebook 05 once; never overwrite executed evidence."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[2]
target = ROOT / "notebooks/05_error_audit_and_roi_data.ipynb"
if target.exists():
    raise FileExistsError(target)

cells = []
def md(s): cells.append(nb.v4.new_markdown_cell(s))
def code(s): cells.append(nb.v4.new_code_cell(s))

md('''# 05 — Validation evidence and paired gingival ROI preparation

## Research question
Can local detail, surrounding context and ordered labels improve gingival appearance grading?
This notebook establishes the data and error evidence before building the CNN baselines.
It reads completed v4 detector results, prepares train/validation crops, and preserves the parent split.
It does not evaluate the test split or train a model.

## Protocol
1. Verify the parent split and image/label SHA-256 values.
2. Inspect aggregate validation evidence and saved detector examples.
3. Convert each annotated gingival box to a local crop and a 1.5× context crop.
4. Resize to 224×224 with explicit constant padding, save both views and provenance.
5. Inspect class balance, native ROI sizes, examples and the limitations of the derived task.

Three labels denote **local appearance groups**, not patient-level diagnoses.
All crops from one source image retain its original split and group.''')
code('''from pathlib import Path
import json, inspect, sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/configs/roi.json").is_file())
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython.display import display
from dentalselfie.roi import prepare_rois, audit_training_reports
REPORT = ROOT / "docs/reports/roi_preparation"
OUTPUT = ROOT / "data/processed/gum_roi_v1"
DATASET = ROOT / "data/processed/gum/dataset"
REPORT.mkdir(parents=True, exist_ok=True)
config = json.loads((ROOT / "src/configs/roi.json").read_text())
display(pd.Series(config, name="ROI protocol"))''')
md('''## 1. Completed detector evidence

The final best-checkpoint validation console table supplies rounded per-class metrics.
The CSV history describes epoch development. Neither provides a complete saved per-instance
prediction table. We therefore do not estimate exact localization/classification error shares here.
Saved confusion matrices and example mosaics support qualitative inspection only.''')
code('''overall, classes = audit_training_reports(ROOT, REPORT)
display(overall)
display(classes)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for experiment in ["E0", "E1"]:
    h = pd.read_csv(REPORT / f"{experiment}_gum_history.csv")
    axes[0].plot(h["epoch"], h["metrics/mAP50-95(B)"], label=experiment)
    axes[1].plot(h["epoch"], h["val/cls_loss"], label=experiment)
axes[0].set(xlabel="Epoch", ylabel="Validation mAP50–95")
axes[1].set(xlabel="Epoch", ylabel="Validation classification loss")
for ax in axes: ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(REPORT / "detector_curves.png", dpi=150)
plt.show()''')
code('''for experiment in ["E0", "E1"]:
    source = ROOT / "data/runs" / f"{experiment}_full_v4/experiments" / f"{experiment}_gum_seed42"
    print(experiment, "saved validation evidence; images are not a new prediction run")
    for name in ["confusion_matrix_normalized.png", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
        with Image.open(source / name) as im:
            preview = im.copy(); preview.thumbnail((1000, 650)); display(preview)''')
md('''## 2. Deterministic derived data

Only train and validation rows are processed. Parent image and label checksums are checked before cropping.
Local integer bounds use floor/ceil around the original floating-point rectangle.
The wider context can exceed the image: its padding is explicitly stored in the manifest.
Context preserves the **central ROI's** label; it is not labelled as a whole context region.
No augmentation, class resampling or new clinical labels are introduced here.

The shared function processes each image once, creates its two views, records hashes, and checks
parent/group isolation. On rerun it verifies existing crop hashes rather than overwriting them.''')
code('''frame, receipt = prepare_rois(DATASET, ROOT / "data/manifests", OUTPUT,
                              ROOT / "src/configs/roi.json", workers=8)
display(pd.Series(receipt, name="Verified preparation receipt"))
display(frame.head())
counts = pd.crosstab(frame["split"], frame["class_name"])
display(counts)
print("Derived images:", len(frame) * 2, "from parents:", frame.parent_id.nunique())
print("Native ROI width/height distribution in prepared 1280-edge source pixels:")
display(frame.groupby(["split", "class_name"])[["box_width", "box_height"]].quantile([.1, .5, .9]))
counts.plot.bar(rot=0, ylabel="Annotated ROI count", title="Class imbalance inherited from the source")
plt.tight_layout(); plt.savefig(REPORT / "roi_class_counts.png", dpi=150); plt.show()''')
md('''## 3. Inspect paired examples

Select a small and large training ROI per class deterministically. These are quality-control examples,
not independent patients or model-selected errors. Red shows the local box; blue shows context bounds.
Inspect the native ROI size before interpreting 224-pixel crops: resizing does not add image information.''')
code('''examples = []
for class_id in [0, 1, 2]:
    subset = frame[(frame.split == "train") & (frame.class_id == class_id)].copy()
    subset["area"] = subset.box_width * subset.box_height
    examples.extend(subset.sort_values(["area", "crop_id"]).iloc[[0, -1]].to_dict("records"))
fig, axes = plt.subplots(len(examples), 3, figsize=(12, 3 * len(examples)))
for i, row in enumerate(examples):
    with Image.open(DATASET / row["image_path"]) as im:
        parent = im.convert("RGB")
    draw = ImageDraw.Draw(parent)
    draw.rectangle(json.loads(row["context_bounds"]), outline="blue", width=3)
    draw.rectangle(json.loads(row["local_bounds"]), outline="red", width=3)
    axes[i, 0].imshow(parent); axes[i, 0].set_title(row["class_name"])
    for j, view in enumerate(["local", "context"], 1):
        with Image.open(OUTPUT / row[f"{view}_path"]) as im:
            axes[i, j].imshow(im.copy())
        axes[i, j].set_title(f"{view}: {row['crop_id']}")
    for ax in axes[i]: ax.axis("off")
fig.tight_layout(); fig.savefig(REPORT / "paired_roi_examples.png", dpi=120); plt.show()''')
md('''## 4. Interpretation and next gate

Class imbalance and low-resolution source ROIs can affect minority-class grading.
Ground-truth crops quantify classification with known localization; final comparison must also use
predicted detector boxes and count missed proposals. A refiner cannot recover regions the detector misses.
Use macro-F1 as the primary ROI screening metric, report every class's recall, and keep test closed.
Do not combine CNN accuracy and detector mAP in one numerical ranking.

Notebook 06 will implement scratch CNN, scratch ResNet18 and ImageNet-initialized ResNet18.
Pretraining, architecture and any later context/ordinal change must be distinct experiments.
Measure 2–3 epochs before scheduling full tracks; do not reuse detector throughput estimates for ROI training.''')
code('''minority = classes[classes.class_name == "no_visible_inflammation"]
conclusion = {
    "protocol": config["protocol"], "roi_rows": len(frame), "view_files": 2 * len(frame),
    "parents": int(frame.parent_id.nunique()), "splits": sorted(frame.split.unique().tolist()),
    "test_images_opened": receipt["test_images_opened"],
    "parent_split_sha256": receipt["signature"]["parent_splits_sha256"],
    "manifest_sha256": receipt["manifest_sha256"],
    "validation_minority_instances": minority.instances.tolist(),
    "limitations": ["Per-instance detection error attribution is pending prediction export",
                    "Correlated crops are not independent patients", "No real phone generalization evidence"],
    "next": "Implement notebook 06 and benchmark ROI classification throughput on Kaggle T4 x2"
}
from dentalselfie.research import write_json
write_json(REPORT / "conclusion.json", conclusion)
display(pd.Series(conclusion, name="Executed conclusion"))''')
md('''## Implementation appendix

The following is the actual shared module used above. It is displayed for review instead of maintaining
a second implementation in this notebook.''')
code('''from IPython.display import Code
import dentalselfie.roi as roi_module
display(Code(inspect.getsource(roi_module), language="python"))''')
notebook = nb.v4.new_notebook(cells=cells)
notebook.metadata.update(kernelspec={"display_name":"DentalSelfie (Python 3.11)","language":"python","name":"dentalselfie-local"},language_info={"name":"python"})
nb.write(notebook, target)
print(target)
