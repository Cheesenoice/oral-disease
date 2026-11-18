"""Derive raw-source local ROIs with unchanged annotation identities and split locks."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .acquire import digest
from .roi import pixel_box, view_bounds, padded_view


def retained_annotations(text, mapping):
    rows = []
    for line_index, line in enumerate(text.splitlines()):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5 or fields[0] not in mapping:
            raise ValueError(f"Unknown source annotation at line {line_index}")
        if mapping[fields[0]] is not None:
            rows.append((line_index, mapping[fields[0]], list(map(float, fields[1:]))))
    return rows


def prepare_raw_rois(root: Path, workers=2):
    root = Path(root)
    old_root = root / "data/processed/gum_roi_v1"
    output = root / "data/processed/gum_raw_roi_v1"
    splits = root / "data/manifests/gum.splits.csv"
    lock_path = root / "data/manifests/gum.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if digest(splits) != lock["splits_sha256"]:
        raise ValueError("Split lock differs")
    old_receipt = json.loads((old_root / "receipt.json").read_text(encoding="utf-8"))
    if digest(old_root / "manifest.csv") != old_receipt["manifest_sha256"]:
        raise ValueError("ROI-v1 manifest fingerprint differs")
    old = pd.read_csv(old_root / "manifest.csv")
    parents = pd.read_csv(splits, dtype={"sample_id": str})
    parents = parents[parents.split.isin(["train", "val"])].copy()
    if set(old.parent_id) != set(parents.sample_id):
        raise ValueError("Parent membership differs")
    mapping = json.loads((root / "src/configs/label_mappings.json").read_text())["gum"]["classes"]
    config = json.loads((root / "src/configs/roi.json").read_text())
    signature = {"protocol": "gingival-raw-local-roi-v1", "parent_split_sha256": digest(splits),
                 "parent_lock_sha256": digest(lock_path), "roi_v1_manifest_sha256": digest(old_root / "manifest.csv"),
                 "mapping_sha256": digest(root / "src/configs/label_mappings.json"),
                 "size": config["size"], "interpolation": "BILINEAR", "jpeg_quality": 95,
                 "jpeg_subsampling": 0, "pillow": Image.__version__}
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "receipt.json"
    existing = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if existing and existing["signature"] != signature:
        raise ValueError("Existing derived-data signature differs; use a new version")
    grouped = {k: v.sort_values("box_index") for k, v in old.groupby("parent_id")}

    def process(parent):
        source = root / parent["source_image"]
        labels = root / parent["source_label"]
        if digest(source) != parent["source_sha256"] or digest(labels) != parent["source_label_sha256"]:
            raise ValueError(f"Source fingerprint mismatch: {parent['sample_id']}")
        annotations = retained_annotations(labels.read_text(), mapping)
        previous = grouped[parent["sample_id"]]
        if len(annotations) != len(previous):
            raise ValueError("Retained annotation membership differs")
        prepared_lines = (root / parent["label"]).read_text().splitlines()
        rows = []
        with Image.open(source) as opened:
            orientation = opened.getexif().get(274, 1)
            if orientation not in (0, 1):
                raise ValueError("Unverified EXIF orientation")
            if opened.size != (int(parent["source_width"]), int(parent["source_height"])):
                raise ValueError("Source dimensions differ")
            image = opened if existing else opened.convert("RGB")
            for (_, old_row), (line_index, class_id, coords) in zip(previous.iterrows(), annotations):
                prepared = prepared_lines[int(old_row.box_index)].split()
                if int(prepared[0]) != class_id or int(old_row.class_id) != class_id:
                    raise ValueError("Class identity/order differs")
                if not np.allclose(coords, list(map(float, prepared[1:])), rtol=0, atol=1.1e-6):
                    raise ValueError("Source/prepared box correspondence differs")
                if old_row.split != parent["split"] or old_row.group_id != parent["group_id"]:
                    raise ValueError("Split/group membership differs")
                box = pixel_box(coords, image.width, image.height)
                bounds = view_bounds(box)
                path = f"{parent['split']}/local/{old_row.crop_id}.jpg"
                destination = output / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not existing:
                    crop, padding = padded_view(image, bounds, config["size"], config["padding_rgb"])
                    crop.save(destination, quality=95, subsampling=0)
                else:
                    padding = (0, 0, 0, 0)
                rows.append({"crop_id": old_row.crop_id, "parent_id": old_row.parent_id,
                             "group_id": old_row.group_id, "split": old_row.split,
                             "class_id": class_id, "class_name": old_row.class_name,
                             "local_path": path, "local_sha256": digest(destination),
                             "source_image": parent["source_image"], "source_sha256": parent["source_sha256"],
                             "source_label_sha256": parent["source_label_sha256"],
                             "source_annotation_line": line_index, "box_index": int(old_row.box_index),
                             "raw_box_width": box[2]-box[0], "raw_box_height": box[3]-box[1],
                             "raw_bounds": json.dumps(bounds), "padding": json.dumps(padding),
                             "old_local_sha256": old_row.local_sha256,
                             "old_box_width": old_row.box_width, "old_box_height": old_row.box_height})
        return rows

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [row for group in pool.map(process, parents.to_dict("records")) for row in group]
    frame = pd.DataFrame(rows).sort_values(["split", "parent_id", "box_index"]).reset_index(drop=True)
    if len(frame) != len(old) or not frame.crop_id.is_unique or set(frame.crop_id) != set(old.crop_id):
        raise ValueError("ROI identity differs")
    if frame.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("Group leakage")
    if existing:
        saved = pd.read_csv(output / "manifest.csv")
        if not frame.local_sha256.equals(saved.local_sha256):
            raise ValueError("Derived crop checksum mismatch")
        if digest(output / "manifest.csv") != existing["manifest_sha256"]:
            raise ValueError("Derived manifest checksum mismatch")
        return saved, existing
    if digest(splits) != signature["parent_split_sha256"]:
        raise ValueError("Split changed during derivation")
    frame.to_csv(output / "manifest.csv", index=False, lineterminator="\n")
    frame.groupby(["split", "class_id", "class_name"]).size().rename("count").to_csv(output / "class_counts.csv")
    receipt = {"signature": signature, "manifest_sha256": digest(output / "manifest.csv"),
               "crops": len(frame), "parents": len(parents), "test_images_opened": 0,
               "membership": "Same crop IDs, labels, groups and split assignments as ROI v1; local view only"}
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return frame, receipt
