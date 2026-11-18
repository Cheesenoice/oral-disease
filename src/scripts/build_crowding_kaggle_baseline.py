"""Build the single reproducible crowding baseline notebook and Kaggle input."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/alignment_crowding_gigikita_v1"
STAGE = ROOT / ".venv/work/kaggle/crowding_gigikita_v1"
NOTEBOOK = ROOT / "notebooks/alignment/16_21_gigikita_crowding_yolo11s_baseline.ipynb"
OWNER = "trihuynhviprovcl"
DATA_SLUG = "dentalselfie-crowding-gigikita-v1"
KERNEL_SLUG = "ds-crowding-y11s-v1"


def build_data_stage() -> dict:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    archive = STAGE / "gigikita_crowding_train_valid.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for split in ("train", "valid"):
            for sub in ("images", "labels"):
                for path in sorted((DATA / split / sub).iterdir()):
                    zf.write(path, Path(split) / sub / path.name)
        for name in ("data.yaml", "manifest.csv", "split_summary.json", "annotation_conflicts.json"):
            zf.write(DATA / name, name)
    summary = json.loads((DATA / "split_summary.json").read_text(encoding="utf-8"))
    card = f"""# GigiKita crowding train/validation derivative\n\n- Source: GigiKita Roboflow Universe, dataset version 1: {summary['attribution']}\n- License declared by source: {summary['license_declared_by_source']} (attribution retained; source declaration, not independently legal-verified).\n- Task: one-class YOLO detection, source label `crowded`. This is not a clinical diagnosis or a claim of patient-level/generalization performance.\n- Contents: duplicate-grouped train/validation only. The source test split was not opened or included.\n- Deduplicated manifest SHA-256: `{summary['manifest_sha256']}`\n- Split: train {summary['train_images']} images ({summary['train_positive_images']} positive), validation {summary['valid_images']} ({summary['valid_positive_images']} positive).\n- Exclusions: {summary['excluded_duplicate_label_conflict_groups']} duplicate groups with conflicting target box counts.\n- Negatives mean no `crowded` annotation in this seven-class export; they must not be interpreted as clinically healthy.\n- Grouping uses observed visual duplicates; there are no verified patient IDs, so patient independence is not claimed.\n"""
    (STAGE / "README.md").write_text(card, encoding="utf-8")
    metadata = {
        "title": "DentalSelfie GigiKita Crowding Train Validation v1",
        "id": f"{OWNER}/{DATA_SLUG}",
        "licenses": [{"name": "CC-BY-4.0"}],
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"zip": str(archive), "zip_bytes": archive.stat().st_size,
            "zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "stage": str(STAGE)}


def make_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "kaggle": {"accelerator": "NvidiaTeslaT4", "isInternetEnabled": True,
                   "isGpuEnabled": True, "dataset_sources": [f"{OWNER}/{DATA_SLUG}"],
                   "competition_sources": [], "kernel_sources": []},
    }
    def md(s): nb.cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):
        s = s.replace("\n+", "\n")
        s = s.replace("str(train_result.save_dir) if train_result else str(RUN_ROOT/'train')", "str(RUN_ROOT/'train')")
        nb.cells.append(nbf.v4.new_code_cell(s))

    md("""# 16.21 - GigiKita single-class crowding baseline\n\nThis is one bounded, source-specific YOLO11s detection experiment. It trains on the duplicate-grouped GigiKita train split and selects checkpoints using only the new validation split. It does not open the source test split. The source's `crowded` labels are used as provided; metrics describe this export, not clinical or phone-camera accuracy.\n\n**Why this baseline:** YOLO11s is the stronger visible YOLO11 checkpoint in the existing project and is still small enough for a practical mobile-export investigation later. The study keeps one dataset, one architecture, one seed and one protocol to avoid score shopping. Research anchor: the 2026 YOLOv11 dental-malocclusion study in the project research report motivates YOLO11 family detection, but its private-data scores are not comparable to this dataset.\n\n**Expected result:** source-validation precision, recall, mAP50 and mAP50-95 plus curves and confusion matrix. Do not treat a high score as evidence of external validity.""")
    md("## 1. Runtime and reproducibility\n\nUse Kaggle T4 x2. The time guard leaves room inside the 12-hour session. The same seed and fixed train/validation manifest are recorded with artifacts.")
    code("""import os, sys, json, time, random, hashlib, zipfile, shutil\n+from pathlib import Path\n+import numpy as np\n+\n+SEED = 51324\n+random.seed(SEED); np.random.seed(SEED)\n+os.environ['PYTHONHASHSEED'] = str(SEED)\n+print({'python': sys.version.split()[0], 'seed': SEED, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})\n+!nvidia-smi""")
    md("## 2. Recover and verify the frozen input\n\nOnly the packaged train/validation derivative is attached. Check the expected manifest hash and image/label inventories before training.")
    code("""INPUT_ROOT=Path('/kaggle/input')\n+assert INPUT_ROOT.exists(), f'Kaggle input mount not found: {INPUT_ROOT}'\n+top_level=sorted(p.name for p in INPUT_ROOT.iterdir())\n+print('Mounted input entries:', top_level)\n+EXPECTED_MANIFEST_SHA256='44d39f856f80508da95eafcaecd49c362f3e34b9d8b2b0f8729281c54229c488'\n+existing_manifests=list(INPUT_ROOT.rglob('manifest.csv'))\n+if existing_manifests:\n+    DATA_ROOT=existing_manifests[0].parent\n+else:\n+    archives=list(INPUT_ROOT.rglob('*.zip'))\n+    if not archives:\n+        raise FileNotFoundError(f'No manifest.csv or zip under /kaggle/input. Top level: {top_level}')\n+    DATA_ROOT=Path('/kaggle/working/crowding_dataset')\n+    DATA_ROOT.mkdir(parents=True, exist_ok=True)\n+    with zipfile.ZipFile(archives[0]) as zf: zf.extractall(DATA_ROOT)\n+summary=json.loads((DATA_ROOT/'split_summary.json').read_text())\n+manifest_sha=hashlib.sha256((DATA_ROOT/'manifest.csv').read_bytes()).hexdigest()\n+assert manifest_sha==EXPECTED_MANIFEST_SHA256, (manifest_sha, summary)\n+assert summary['source_test_opened'] is False and not (DATA_ROOT/'test').exists()\n+print({'data_root':str(DATA_ROOT), **{k:summary[k] for k in ['train_images','valid_images','train_positive_images','valid_positive_images','train_boxes','valid_boxes','excluded_duplicate_label_conflict_groups','manifest_sha256']}})\n+for split,expected in [('train',5516),('valid',1357)]:\n+    images=list((DATA_ROOT/split/'images').glob('*')); labels=list((DATA_ROOT/split/'labels').glob('*.txt'))\n+    assert len(images)==len(labels)==expected\n+print('Integrity checks passed: image-label pairing, frozen manifest, no test directory.')""")
    md("## 3. Audit labels and split isolation\n\nThis verifies that all boxes use the single class index 0 and normalized YOLO coordinates, and that visual duplicate groups do not cross splits.")
    code("""import csv, collections\n+rows=list(csv.DictReader((DATA_ROOT/'manifest.csv').open(encoding='utf-8')))\n+groups={s:{r['group_id'] for r in rows if r['split']==s} for s in ['train','valid']}\n+assert not groups['train'] & groups['valid']\n+bad=[]; box_counts=collections.Counter()\n+for split in ['train','valid']:\n+    for p in (DATA_ROOT/split/'labels').glob('*.txt'):\n+        for line_no,line in enumerate(p.read_text().splitlines(),1):\n+            v=line.split(); box_counts[split]+=1\n+            if len(v)!=5 or v[0]!='0' or any(not 0<=float(x)<=1 for x in v[1:]): bad.append((p.name,line_no,line))\n+assert not bad, bad[:5]\n+print({'train_groups':len(groups['train']), 'validation_groups':len(groups['valid']),\n+       'cross_split_group_overlap':0, 'box_counts':dict(box_counts), 'invalid_boxes':len(bad)})""")
    md("## 4. Prepare the single YOLO11s run\n\nThe transfer-learning baseline uses the official COCO-pretrained YOLO11s checkpoint, one target class and a fixed seed. Mild geometric/color augmentation is used; vertical flips are disabled because they create implausible mouth orientation. A time cap and early stopping bound the session. The full run is one experiment, not a hyperparameter search.")
    code("""%pip -q install ultralytics==8.3.203\n+from ultralytics import YOLO\n+import torch, yaml\n+assert torch.cuda.is_available(), 'Kaggle GPU is not available'\n+gpu_count=torch.cuda.device_count()\n+print({'torch':torch.__version__, 'cuda':torch.version.cuda, 'gpu_count':gpu_count,\n+       'gpus':[torch.cuda.get_device_name(i) for i in range(gpu_count)]})\n+assert gpu_count >= 1\n+runtime_yaml=Path('/kaggle/working/crowding_data.yaml')\n+runtime_yaml.write_text(yaml.safe_dump({'path':str(DATA_ROOT), 'train':'train/images', 'val':'valid/images', 'nc':1, 'names':['crowded']}))\n+RUN_ROOT=Path('/kaggle/working/runs/crowding_yolo11s_seed51324')\n+RUN_ROOT.mkdir(parents=True, exist_ok=True)\n+model=YOLO('yolo11s.pt')\n+train_start=time.time()\n+train_result=model.train(data=str(runtime_yaml), imgsz=640, epochs=25, patience=6,\n+    batch=32, workers=4, device=0 if gpu_count==1 else [0,1], seed=SEED, deterministic=True,\n+    optimizer='AdamW', lr0=0.001, cos_lr=True, amp=True, time=8.0,\n+    fliplr=0.5, flipud=0.0, degrees=5.0, translate=0.05, scale=0.10,\n+    hsv_h=0.005, hsv_s=0.25, hsv_v=0.15, mosaic=0.5, close_mosaic=5,\n+    save=True, save_period=2, plots=True, project=str(RUN_ROOT), name='train', exist_ok=True)\n+train_hours=(time.time()-train_start)/3600\n+print({'wall_hours':round(train_hours,3), 'run_dir':str(train_result.save_dir) if train_result else str(RUN_ROOT/'train')})""")
    md("## 5. Validate the selected best checkpoint\n\nReport source-validation metrics from the best checkpoint. The validation set was fixed before training; it is used for early stopping and checkpoint selection, so these are model-selection metrics rather than an untouched final test estimate.")
    code("""run_dir=RUN_ROOT/'train'\n+best_path=run_dir/'weights'/'best.pt'\n+assert best_path.exists(), f'Missing best checkpoint: {best_path}'\n+best=YOLO(str(best_path))\n+metrics=best.val(data=str(runtime_yaml), split='val', imgsz=640, batch=32,\n+                 device=0 if gpu_count==1 else 0, plots=True, project=str(RUN_ROOT), name='best_validation')\n+metric_row={'precision':float(metrics.box.mp), 'recall':float(metrics.box.mr),\n+            'mAP50':float(metrics.box.map50), 'mAP50_95':float(metrics.box.map),\n+            'best_checkpoint':str(best_path), 'training_hours':round(train_hours,3)}\n+print(json.dumps(metric_row, indent=2))""")
    md("## 6. Save compact, inspectable evidence\n\nKeep the training log, results table, validation plots and best checkpoint together. These are source-validation results; do not tune repeatedly on this same split and call it a test result.")
    code("""import shutil, json\n+ARTIFACTS=Path('/kaggle/working/crowding_baseline_artifacts')\n+ARTIFACTS.mkdir(parents=True, exist_ok=True)\n+for name in ['results.csv','results.png','confusion_matrix.png','confusion_matrix_normalized.png','PR_curve.png','F1_curve.png','P_curve.png','R_curve.png']:\n+    src=run_dir/name\n+    if src.exists(): shutil.copy2(src, ARTIFACTS/name)\n+shutil.copy2(best_path, ARTIFACTS/'best.pt')\n+(ARTIFACTS/'summary.json').write_text(json.dumps({\n+    'source':'GigiKita Roboflow version 1, declared CC BY 4.0', 'task':'source-defined crowded object detection',\n+    'seed':SEED, 'manifest_sha256':manifest_sha, 'source_test_opened':False,\n+    'train_images':5516, 'validation_images':1357, 'metrics':metric_row}, indent=2))\n+print('Artifacts:', [(p.name, round(p.stat().st_size/1e6,2)) for p in ARTIFACTS.iterdir()])""")
    nbf.write(nb, NOTEBOOK)


if __name__ == "__main__":
    print(json.dumps(build_data_stage(), indent=2))
    make_notebook()
    print("Notebook:", NOTEBOOK)
