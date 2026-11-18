"""Hash-verified paired gingival crops, derived without changing parent splits."""
from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from .acquire import digest
from .research import write_json


def pixel_box(values, width, height):
    """Validate YOLO centre/size values; return a pixel rectangle without clipping."""
    x, y, w, h = map(float, values)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("Non-finite or empty box")
    box = ((x - w / 2) * width, (y - h / 2) * height,
           (x + w / 2) * width, (y + h / 2) * height)
    if box[0] < -1e-6 or box[1] < -1e-6 or box[2] > width + 1e-6 or box[3] > height + 1e-6:
        raise ValueError("Original annotation exceeds image bounds; do not silently clip")
    return box


def view_bounds(box, multiplier=1.0):
    if multiplier < 1:
        raise ValueError("Context multiplier must be at least one")
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = (x1 - x0) * multiplier / 2, (y1 - y0) * multiplier / 2
    return (math.floor(cx - half_w), math.floor(cy - half_h),
            math.ceil(cx + half_w), math.ceil(cy + half_h))


def padded_view(image, bounds, size, fill):
    """Explicit constant padding for derived views; source geometry stays intact."""
    x0, y0, x1, y1 = bounds
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Empty derived view")
    canvas = Image.new("RGB", (x1 - x0, y1 - y0), tuple(fill))
    intersection = (max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1))
    if intersection[2] > intersection[0] and intersection[3] > intersection[1]:
        canvas.paste(image.crop(intersection), (intersection[0] - x0, intersection[1] - y0))
    padding = (max(0, -x0), max(0, -y0), max(0, x1 - image.width), max(0, y1 - image.height))
    return canvas.resize((size, size), Image.Resampling.BILINEAR), padding


def prepare_rois(dataset: Path, manifests: Path, output: Path, config_path: Path, workers=8):
    """Reusable locally and on Kaggle's prepared gum dataset; train/val only."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["splits"] != ["train", "val"]:
        raise ValueError("This stage may process train/validation only")
    split_path, lock_path = manifests / "gum.splits.csv", manifests / "gum.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if digest(split_path) != lock["splits_sha256"]:
        raise ValueError("Parent split hash differs from its lock")
    parents = pd.read_csv(split_path, dtype={"sample_id": str})
    if parents.groupby("group_id")["split"].nunique().max() != 1:
        raise ValueError("Parent groups cross splits")
    selected = parents[parents.split.isin(config["splits"])].copy()
    signature = {"protocol": config["protocol"], "config_sha256": digest(config_path),
                 "parent_splits_sha256": digest(split_path), "parent_lock_sha256": digest(lock_path)}
    receipt_path = output / "receipt.json"
    if output.exists() and any(output.iterdir()):
        if not receipt_path.is_file():
            raise ValueError("Existing ROI output is incomplete; preserve it and select a fresh output directory")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["signature"] != signature or digest(output / "manifest.csv") != receipt["manifest_sha256"]:
            raise ValueError("Existing ROI protocol or manifest differs")
        for parent in selected.to_dict("records"):
            image = dataset / "images" / parent["split"] / f"{parent['sample_id']}.jpg"
            label = dataset / "labels" / parent["split"] / f"{parent['sample_id']}.txt"
            if digest(image) != parent["sha256"] or digest(label) != parent["label_sha256"]:
                raise ValueError("Parent checksum mismatch on ROI reuse")
        frame = pd.read_csv(output / "manifest.csv")
        for row in frame.to_dict("records"):
            for view in ("local", "context"):
                if digest(output / row[f"{view}_path"]) != row[f"{view}_sha256"]:
                    raise ValueError("Existing crop checksum differs")
        return frame, receipt
    output.mkdir(parents=True, exist_ok=True)
    for split in config["splits"]:
        for view in ("local", "context"):
            (output / split / view).mkdir(parents=True, exist_ok=True)

    def process(parent):
        split, sample_id = parent["split"], parent["sample_id"]
        relative_image = f"images/{split}/{sample_id}.jpg"
        image_path = dataset / relative_image
        label_path = dataset / "labels" / split / f"{sample_id}.txt"
        if digest(image_path) != parent["sha256"] or digest(label_path) != parent["label_sha256"]:
            raise ValueError(f"Parent checksum mismatch: {sample_id}")
        rows = []
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            if image.size != (int(parent["width"]), int(parent["height"])):
                raise ValueError("Parent image dimensions differ from manifest")
            lines = label_path.read_text(encoding="utf-8").splitlines()
            if len(lines) != int(parent["label_count"]):
                raise ValueError("Parent box count differs")
            for index, line in enumerate(lines):
                fields = line.split()
                if len(fields) != 5 or fields[0] not in {"0", "1", "2"}:
                    raise ValueError(f"Unexpected ROI label: {sample_id}:{index}")
                class_id = int(fields[0])
                box = pixel_box(fields[1:], image.width, image.height)
                crop_id = f"{sample_id}_{index:02d}"
                row = {"crop_id": crop_id, "parent_id": sample_id, "group_id": parent["group_id"],
                       "split": split, "class_id": class_id, "class_name": config["classes"][class_id],
                       "image_path": relative_image, "parent_sha256": parent["sha256"],
                       "label_sha256": parent["label_sha256"], "box_index": index,
                       "box_width": box[2] - box[0], "box_height": box[3] - box[1]}
                for view, scale in (("local", 1.0), ("context", config["context_multiplier"])):
                    bounds = view_bounds(box, scale)
                    crop, padding = padded_view(image, bounds, config["size"], config["padding_rgb"])
                    relative = f"{split}/{view}/{crop_id}.jpg"
                    crop.save(output / relative, quality=config["jpeg_quality"], subsampling=config["jpeg_subsampling"])
                    row.update({f"{view}_path": relative, f"{view}_bounds": json.dumps(bounds),
                                f"{view}_padding": json.dumps(padding),
                                f"{view}_sha256": digest(output / relative)})
                rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [row for batch in pool.map(process, selected.to_dict("records")) for row in batch]
    frame = pd.DataFrame(rows).sort_values(["split", "parent_id", "box_index"]).reset_index(drop=True)
    if frame.crop_id.duplicated().any() or frame.groupby("parent_id").split.nunique().max() != 1:
        raise ValueError("Duplicate crop ID or parent split leakage")
    if digest(split_path) != signature["parent_splits_sha256"]:
        raise ValueError("Parent split changed during processing")
    frame.to_csv(output / "manifest.csv", index=False, lineterminator="\n")
    counts = frame.groupby(["split", "class_name"]).size().rename("crops").reset_index()
    counts.to_csv(output / "class_counts.csv", index=False)
    receipt = {"signature": signature, "manifest_sha256": digest(output / "manifest.csv"),
               "crops": len(frame), "views": 2 * len(frame), "parents": len(selected),
               "counts": counts.to_dict("records"), "test_images_opened": 0,
               "independence": "Parent/group isolation inherited; no patient independence claim"}
    write_json(receipt_path, receipt)
    return frame, receipt


def audit_training_reports(root: Path, report: Path):
    """Read saved validation evidence; do not invent per-instance error attribution."""
    report.mkdir(parents=True, exist_ok=True)
    rows, per_class = [], []
    for experiment in ("E0", "E1"):
        source = root / "data/runs" / f"{experiment}_full_v4/experiments"
        run = source / f"{experiment}_gum_seed42"
        history = pd.read_csv(run / "results.csv")
        history.columns = history.columns.str.strip()
        result = json.loads((source / f"{experiment}_gum_seed42.result.json").read_text())
        rows.append({"experiment": experiment, "epochs": result["epochs_completed"],
                     "seconds": result["elapsed_seconds"], **result["metrics"]})
        log = (source / f"{experiment}_gum_seed42.log").read_text(encoding="utf-8", errors="replace")
        for name in ("no_visible_inflammation", "mild_inflammation", "marked_inflammation"):
            matches = re.findall(rf"^\s*{name}\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", log, re.MULTILINE)
            if not matches:
                raise ValueError(f"Final class metrics absent: {experiment}/{name}")
            images, instances, precision, recall, ap50, ap = matches[-1]
            per_class.append({"experiment": experiment, "class_name": name, "images": int(images),
                              "instances": int(instances), "precision": float(precision),
                              "recall": float(recall), "ap50": float(ap50), "ap50_95": float(ap),
                              "provenance": "Final best-checkpoint validation console table; rounded by trainer"})
        history.to_csv(report / f"{experiment}_gum_history.csv", index=False)
    overall, classes = pd.DataFrame(rows), pd.DataFrame(per_class)
    overall.to_csv(report / "detector_summary.csv", index=False)
    classes.to_csv(report / "detector_class_metrics.csv", index=False)
    return overall, classes
