"""Strict YOLO normalization and reproducible, group-aware split generation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml

from .acquire import digest


def parse_yolo(text: str) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        pieces = line.split()
        if len(pieces) != 5:
            raise ValueError(f"Line {number}: expected five detection columns")
        class_id = int(pieces[0])
        x, y, w, h = map(float, pieces[1:])
        if class_id < 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"Line {number}: non-finite coordinate or negative class")
        eps = 1e-6
        if not (w > 0 and h > 0 and x - w / 2 >= -eps and y - h / 2 >= -eps
                and x + w / 2 <= 1 + eps and y + h / 2 <= 1 + eps):
            raise ValueError(f"Line {number}: invalid normalized bounding box")
        rows.append((class_id, x, y, w, h))
    return rows


def convert_index(root: Path, task: str, index_path: Path) -> Path:
    """Input CSV paths are root-relative; original annotations remain unchanged."""
    mapping = json.loads((root / "src/configs/label_mappings.json").read_text(encoding="utf-8"))[task]
    if not mapping["approved"] or mapping["box_format"] != "yolo_cxcywh_normalized":
        raise ValueError("Source schema/mapping must first be verified with rendered overlays")
    if not mapping["classes"] or not mapping.get("evidence"):
        raise ValueError("Explicit class mapping and audit evidence are required")
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        index = list(csv.DictReader(stream))
    if not index:
        raise ValueError("Source index is empty")
    planned = []
    ids = set()
    for row in index:
        image = (root / row["image_path"]).resolve()
        label = (root / row["label_path"]).resolve()
        if not image.is_relative_to(root.resolve()) or not label.is_relative_to(root.resolve()):
            raise ValueError("Source paths must stay within the project root")
        if not image.is_file() or not label.is_file():
            raise ValueError("Missing image or label; missing is not a negative annotation")
        converted = []
        for class_id, *box in parse_yolo(label.read_text(encoding="utf-8-sig")):
            if str(class_id) not in mapping["classes"]:
                raise ValueError(f"Unmapped source class {class_id} in {label}")
            new_class = mapping["classes"][str(class_id)]
            if new_class is None:  # Explicitly excluded anatomy ROI.
                continue
            if type(new_class) is not int or not 0 <= new_class < len(mapping["names"]):
                raise ValueError("Invalid destination class")
            converted.append(" ".join([str(new_class), *(f"{v:.8f}" for v in box)]))
        with Image.open(image) as picture:
            picture.verify()
        sample_id = hashlib.sha256(row["image_path"].encode()).hexdigest()[:20]
        if sample_id in ids:
            raise ValueError("Duplicate sample in source index")
        ids.add(sample_id)
        image_hash = digest(image)
        planned.append({"sample_id": sample_id, "source_image": row["image_path"],
                        "source_label": row["label_path"], "sha256": image_hash,
                        "group_id": row.get("group_id", ""), "view": row.get("view", "unknown"),
                        "image": str(image.relative_to(root.resolve())).replace("\\", "/"),
                        "label": f"data/processed/{task}/labels/{sample_id}.txt", "task": task,
                        "_text": "\n".join(converted) + ("\n" if converted else "")})
        planned[-1]["label_sha256"] = hashlib.sha256(planned[-1]["_text"].encode("utf-8")).hexdigest()
    manifest = root / "data/manifests" / f"{task}.normalized.csv"
    if manifest.exists():
        raise FileExistsError("Normalized manifest already exists; do not overwrite provenance silently")
    for row in planned:
        destination = root / row["label"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(row.pop("_text").encode("utf-8"))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(planned[0]))
        writer.writeheader()
        writer.writerows(planned)
    return manifest


def assign_splits(rows: list[dict], seed: int = 42) -> list[dict]:
    """Connected components keep both known groups and exact duplicates together."""
    if not rows:
        raise ValueError("Cannot split an empty manifest")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in [("hash", row["sha256"]), ("group", row.get("group_id", ""))]:
            if not key[1]:
                continue
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required")
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    targets = {"train": len(rows) * 0.7, "val": len(rows) * 0.15, "test": len(rows) * 0.15}
    counts = Counter()
    result = [dict(row) for row in rows]
    for indices in ordered:
        split = max(targets, key=lambda s: (targets[s] - counts[s]) / targets[s])
        counts[split] += len(indices)
        for i in indices:
            result[i]["split"] = split
    if any(counts[s] == 0 for s in targets):
        raise ValueError("Group sizes yielded an empty split; review grouping")
    return result


def prepare_splits(root: Path, task: str, review_path: Path, seed: int = 42) -> Path:
    review = json.loads(review_path.read_text(encoding="utf-8"))
    manifest = root / "data/manifests" / f"{task}.normalized.csv"
    if not review.get("reviewed") or review.get("manifest_sha256") != digest(manifest):
        raise ValueError("Grouping/near-duplicate audit must refer to the exact normalized manifest")
    if not review.get("grouping_evidence") or not review.get("independence_claim"):
        raise ValueError("Record the grouping evidence and actual independence limitation")
    lock = root / "data/manifests" / f"{task}.lock.json"
    output = root / "data/processed" / task / "dataset"
    if lock.exists() or output.exists():
        raise FileExistsError("Dataset is already prepared; preserve locked splits")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = assign_splits(list(csv.DictReader(stream)), seed)
    for row in rows:
        if digest(root / row["image"]) != row["sha256"] or digest(root / row["label"]) != row["label_sha256"]:
            raise ValueError("Normalized image/label changed after conversion")
    # YOLO resolves labels by replacing /images/ with /labels/; materialize both.
    for row in rows:
        stem = row["sample_id"]
        image = root / row["image"]
        image_out = output / "images" / row["split"] / f"{stem}{image.suffix.lower()}"
        label_out = output / "labels" / row["split"] / f"{stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_out)
        shutil.copy2(root / row["label"], label_out)
    split_manifest = root / "data/manifests" / f"{task}.splits.csv"
    with split_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = json.loads((root / "src/configs/label_mappings.json").read_text(encoding="utf-8"))[task]["names"]
    (output / "data.yaml").write_text(yaml.safe_dump({"train": "images/train", "val": "images/val",
                                                     "test": "images/test", "names": names}), encoding="utf-8")
    lock.write_text(json.dumps({"task": task, "seed": seed, "manifest_sha256": digest(manifest),
                                "splits_sha256": digest(split_manifest), "review": review,
                                "counts": dict(Counter(row["split"] for row in rows))}, indent=2), encoding="utf-8")
    return output
