"""Build a duplicate-grouped, one-class crowding train/validation dataset.

Only GigiKita v1 train/valid photos are eligible. The source test split is not
present in the extracted work area and is never read by this script.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from audit_external_alignment_candidates import parse_yolo_label

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".venv/work/alignment_external_audit/gigikita_discloration_v1"
AUDIT_DIR = ROOT / "docs/reports/alignment/external_candidates_2026-11-18"
SUMMARY_PATH = AUDIT_DIR / "audit_summary.json"
REPEAT_PAIRS_PATH = AUDIT_DIR / "dhash_candidate_pair_refinement.csv"
OUTPUT = ROOT / "data/processed/alignment_crowding_gigikita_v1"
SEED = 51324
VAL_FRACTION = 0.20
SOURCE_ID = "gigikita_discloration_v1"


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def normalized_target_boxes(row: dict, class_names: list[str], target_id: int) -> list[list[float]]:
    relative = Path(row["relative_image"])
    image_path = WORK / relative
    label_path = WORK / relative.parent.parent / "labels" / f"{relative.stem}.txt"
    with Image.open(image_path) as image:
        width, height = image.size
    objects, errors = parse_yolo_label(label_path, len(class_names), width, height)
    if errors:
        raise ValueError(f"Source annotation audit error for {relative}: {errors}")
    boxes = []
    for obj in objects:
        if obj["class_id"] != target_id:
            continue
        x1, y1, x2, y2 = obj["xyxy"]
        boxes.append([x1/width, y1/height, x2/width, y2/height])
    return boxes


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(area_a+area_b-inter, 1e-12)


def labels_agree(group: list[dict], class_names: list[str], target_id: int):
    boxes_by_image = {row["relative_image"]: normalized_target_boxes(row, class_names, target_id) for row in group}
    values = list(boxes_by_image.values())
    presence = {bool(boxes) for boxes in values}
    if len(presence) > 1:
        return False, "duplicate_target_presence_conflict", boxes_by_image
    counts = {len(boxes) for boxes in values}
    if len(counts) > 1:
        return False, "duplicate_target_box_count_conflict", boxes_by_image
    if values and values[0]:
        reference = values[0]
        for other in values[1:]:
            unused = set(range(len(other)))
            for box in reference:
                if not unused:
                    return False, "duplicate_target_box_match_conflict", boxes_by_image
                best_idx = max(unused, key=lambda idx: iou(box, other[idx]))
                if iou(box, other[best_idx]) < 0.50:
                    return False, "duplicate_target_box_match_conflict", boxes_by_image
                unused.remove(best_idx)
    return True, "", boxes_by_image


def stable_split(group_id: str, positive: bool) -> str:
    """Deterministic stratified 80/20 assignment using class-stratified hashes."""
    key = f"{SEED}|{'positive' if positive else 'negative'}|{group_id}".encode()
    value = int(hashlib.sha256(key).hexdigest()[:16], 16) / 2**64
    return "valid" if value < VAL_FRACTION else "train"


def prepare() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {OUTPUT}")
    audit = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    source = next(s for s in audit["sources"] if s["source_id"] == SOURCE_ID)
    class_names = source["class_names_in_source_order"]
    target_ids = [i for i, name in enumerate(class_names) if name.strip().casefold() == "crowded"]
    if len(target_ids) != 1:
        raise ValueError(f"Expected one exact source class 'crowded', found {target_ids}")
    target_id = target_ids[0]

    with (AUDIT_DIR / "train_valid_image_manifest.csv").open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if r["source"] == SOURCE_ID and r["split"] in ("train", "valid")]
    if len(rows) != 7143:
        raise ValueError(f"Unexpected train/validation source count: {len(rows)}")
    images = {r["relative_image"]: r for r in rows}

    # Join only visually strong pair matches from the original train/valid pool.
    groups = UnionFind()
    for row in rows:
        groups.find(row["relative_image"])
    strong_edges_used = 0
    with REPEAT_PAIRS_PATH.open(newline="", encoding="utf-8") as stream:
        for pair in csv.DictReader(stream):
            if pair["review_tier"] != "strong_visual_match_candidate":
                continue
            if pair["source_a"] == pair["source_b"] == SOURCE_ID:
                if pair["image_a"] in images and pair["image_b"] in images:
                    groups.union(pair["image_a"], pair["image_b"])
                    strong_edges_used += 1
    components = defaultdict(list)
    for row in rows:
        components[groups.find(row["relative_image"])].append(row)

    canonical, excluded = [], []
    for members in components.values():
        agree, reason, boxes_by_image = labels_agree(members, class_names, target_id)
        if not agree:
            excluded.append({"group_members": [r["relative_image"] for r in members], "reason": reason})
            continue
        # Prefer the largest source resolution; stable image hash breaks ties.
        representative = sorted(members, key=lambda r: (-int(r["width"])*int(r["height"]),
                                                            -int(r["width"]), -int(r["height"]), r["sha256"]))[0]
        boxes = boxes_by_image[representative["relative_image"]]
        group_id = hashlib.sha256("|".join(sorted(r["sha256"] for r in members)).encode()).hexdigest()[:20]
        canonical.append({"representative": representative, "members": members,
                          "group_id": group_id, "boxes": boxes})

    if not canonical:
        raise ValueError("No images survived duplicate-label agreement checks")
    for record in canonical:
        record["split"] = stable_split(record["group_id"], bool(record["boxes"]))

    OUTPUT.mkdir(parents=True)
    manifest_rows = []
    class_box_counts = {"train": 0, "valid": 0}
    positives = {"train": 0, "valid": 0}
    negatives = {"train": 0, "valid": 0}
    for record in canonical:
        row = record["representative"]
        split = record["split"]
        source_path = WORK / row["relative_image"]
        sample_id = hashlib.sha256(f"{SOURCE_ID}|{row['sha256']}".encode()).hexdigest()[:20]
        image_name = f"{sample_id}{source_path.suffix.casefold()}"
        image_out = OUTPUT / split / "images" / image_name
        label_out = OUTPUT / split / "labels" / f"{sample_id}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, image_out)
        lines = []
        for x1, y1, x2, y2 in record["boxes"]:
            xc, yc, bw, bh = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
            lines.append(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
        label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lines:
            positives[split] += 1
            class_box_counts[split] += len(lines)
        else:
            negatives[split] += 1
        manifest_rows.append({
            "sample_id": sample_id, "group_id": record["group_id"], "split": split,
            "source_split": row["split"], "source_image": row["relative_image"],
            "source_image_sha256": row["sha256"], "width": row["width"], "height": row["height"],
            "target_boxes": len(lines), "target_present": bool(lines),
            "deduplicated_member_count": len(record["members"]),
            "deduplicated_members": "|".join(sorted(r["relative_image"] for r in record["members"])),
        })

    fields = list(manifest_rows[0])
    with (OUTPUT / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(manifest_rows)
    (OUTPUT / "data.yaml").write_text(
        "train: train/images\nval: valid/images\nnc: 1\nnames: ['crowded']\n", encoding="utf-8")
    (OUTPUT / "annotation_conflicts.json").write_text(json.dumps(excluded, indent=2), encoding="utf-8")

    split_groups = defaultdict(set)
    for row in manifest_rows:
        split_groups[row["split"]].add(row["group_id"])
    if split_groups["train"] & split_groups["valid"]:
        raise AssertionError("Duplicate groups cross the new train/validation split")
    if sum(positives.values()) + sum(negatives.values()) != len(manifest_rows):
        raise AssertionError("Split image totals do not match manifest")
    for split in ("train", "valid"):
        image_files = list((OUTPUT / split / "images").glob("*.jpg"))
        label_files = list((OUTPUT / split / "labels").glob("*.txt"))
        if len(image_files) != len(label_files):
            raise AssertionError(f"Image/label count mismatch in {split}")

    metadata = {
        "source_id": SOURCE_ID, "source_archive_sha256": source["archive_sha256"],
        "source_classes": class_names, "target_source_class": class_names[target_id],
        "target_model_class": "crowded", "source_test_opened": False,
        "source_input_splits": ["train", "valid"], "source_input_images": len(rows),
        "strong_duplicate_pair_edges_used": strong_edges_used,
        "image_group_count_before_conflict_quarantine": len(components),
        "groups_with_duplicates": sum(len(members) > 1 for members in components.values()),
        "redundant_images_removed": sum(len(members)-1 for members in components.values() if len(members) > 1),
        "retained_groups": len(canonical),
        "excluded_duplicate_label_conflict_groups": len(excluded),
        "train_images": sum(r["split"] == "train" for r in manifest_rows),
        "valid_images": sum(r["split"] == "valid" for r in manifest_rows),
        "train_positive_images": positives["train"], "valid_positive_images": positives["valid"],
        "train_negative_images": negatives["train"], "valid_negative_images": negatives["valid"],
        "train_boxes": class_box_counts["train"], "valid_boxes": class_box_counts["valid"],
        "seed": SEED, "validation_fraction_per_target_presence_stratum": VAL_FRACTION,
        "split_basis": "connected components of strong pHash/ORB-matched duplicate candidates; one canonical image per group",
        "no_patient_id_claim": True,
        "negative_definition": "No source 'crowded' annotation in this same seven-class export; treated as source-defined target-negative, not clinically healthy.",
        "license_declared_by_source": source["license_from_data_yaml"],
        "attribution": source["roboflow_url"],
        "manifest_sha256": hashlib.sha256((OUTPUT / "manifest.csv").read_bytes()).hexdigest(),
    }
    (OUTPUT / "split_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
