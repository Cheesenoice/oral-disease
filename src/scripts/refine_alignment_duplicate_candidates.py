"""Refine equal-dHash alignment duplicate candidates using pHash and ORB.

Uses only the train/validation images listed in the external candidate audit
manifest. A visual-match score is an engineering review cue, not a patient ID.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from audit_external_alignment_candidates import parse_yolo_label

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT / "docs/reports/alignment/external_candidates_2026-11-18"
WORK = ROOT / ".venv/work/alignment_external_audit"
PAIR_CSV = AUDIT_DIR / "dhash_candidate_pair_refinement.csv"
SUMMARY_JSON = AUDIT_DIR / "duplicate_refinement_summary.json"
SHEET = AUDIT_DIR / "duplicate_refinement_contact_sheet.jpg"
LABEL_AUDIT_CSV = AUDIT_DIR / "cross_source_crowding_label_agreement.csv"


def image_phash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    values = low.flatten()
    median = np.median(values[1:])
    bits = values > median
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def orb_match(path_a: Path, path_b: Path) -> dict:
    a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise ValueError(f"Cannot read pair: {path_a} / {path_b}")
    # Keep feature extraction bounded while retaining high-resolution dental details.
    def bounded(gray):
        h, w = gray.shape
        scale = min(1.0, 720 / max(h, w))
        return cv2.resize(gray, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA) if scale < 1 else gray
    a, b = bounded(a), bounded(b)
    orb = cv2.ORB_create(nfeatures=1600, scaleFactor=1.2, nlevels=8, fastThreshold=8)
    key_a, des_a = orb.detectAndCompute(a, None)
    key_b, des_b = orb.detectAndCompute(b, None)
    if des_a is None or des_b is None or len(des_a) < 4 or len(des_b) < 4:
        return {"keypoints_a": len(key_a), "keypoints_b": len(key_b), "ratio_matches": 0, "homography_inliers": 0, "inlier_fraction": 0.0}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_a, des_b, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75*n.distance]
    inliers = 0
    if len(good) >= 4:
        src = np.float32([key_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([key_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.sum())
    return {"keypoints_a": len(key_a), "keypoints_b": len(key_b), "ratio_matches": len(good),
            "homography_inliers": inliers, "inlier_fraction": inliers / max(1, len(good))}


def make_pairs(rows: list[dict]) -> list[dict]:
    by_dhash = defaultdict(list)
    for row in rows:
        if row.get("dhash64") and row["dhash64"].casefold() != "nan":
            by_dhash[row["dhash64"]].append(row)
    pairs = {}
    for dhash, group in by_dhash.items():
        by_source = defaultdict(list)
        for row in group:
            by_source[row["source"]].append(row)
        sources = sorted(by_source)
        for i, source_a in enumerate(sources):
            for source_b in sources[i+1:]:
                for row_a in by_source[source_a]:
                    for row_b in by_source[source_b]:
                        key = tuple(sorted((f'{row_a["source"]}::{row_a["relative_image"]}',
                                            f'{row_b["source"]}::{row_b["relative_image"]}')))
                        pairs[key] = {"row_a": row_a, "row_b": row_b, "candidate_scope": "cross_source"}
            source_rows = by_source[source_a]
            train = [r for r in source_rows if r["split"] == "train"]
            valid = [r for r in source_rows if r["split"] == "valid"]
            for row_a in train:
                for row_b in valid:
                    key = tuple(sorted((f'{row_a["source"]}::{row_a["relative_image"]}',
                                        f'{row_b["source"]}::{row_b["relative_image"]}')))
                    pairs[key] = {"row_a": row_a, "row_b": row_b, "candidate_scope": "within_source_train_valid"}
            for same_split in (train, valid):
                for index_a, row_a in enumerate(same_split):
                    for row_b in same_split[index_a+1:]:
                        key = tuple(sorted((f'{row_a["source"]}::{row_a["relative_image"]}',
                                            f'{row_b["source"]}::{row_b["relative_image"]}')))
                        pairs[key] = {"row_a": row_a, "row_b": row_b, "candidate_scope": "within_source_same_split"}
    return list(pairs.values())


def target_boxes(row: dict, source_summary: dict) -> list[list[float]]:
    """Read source target boxes while deriving class IDs from the archived YAML."""
    label_path = WORK / row["source"] / Path(row["relative_image"]).parent.parent / "labels" / f"{Path(row['relative_image']).stem}.txt"
    image_path = WORK / row["source"] / row["relative_image"]
    with Image.open(image_path) as image:
        width, height = image.size
    names = source_summary[row["source"]]
    target_ids = {i for i, name in enumerate(names) if name.strip().casefold() == "crowded"
                  or name.strip().casefold().startswith("crowded -")}
    objects, _ = parse_yolo_label(label_path, len(names), width, height)
    return [[obj["xyxy"][0]/width, obj["xyxy"][1]/height,
             obj["xyxy"][2]/width, obj["xyxy"][3]/height]
            for obj in objects if obj["class_id"] in target_ids]


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def audit_cross_source_labels(results: list[dict]) -> dict:
    source_summaries = json.loads((AUDIT_DIR / "audit_summary.json").read_text(encoding="utf-8"))["sources"]
    source_names = {s["source_id"]: s["class_names_in_source_order"] for s in source_summaries}
    rows = []
    for pair in results:
        if pair["candidate_scope"] != "cross_source" or pair["review_tier"] != "strong_visual_match_candidate":
            continue
        boxes_a = target_boxes({"source": pair["source_a"], "relative_image": pair["image_a"]}, source_names)
        boxes_b = target_boxes({"source": pair["source_b"], "relative_image": pair["image_b"]}, source_names)
        ious = sorted((box_iou(a, b) for a in boxes_a for b in boxes_b), reverse=True)
        matched = []
        used_a, used_b = set(), set()
        for iou in ious:
            if iou < 0.5:
                break
            # Recover indices for the current overlap deterministically.
            candidates = [(box_iou(a, b), ia, ib) for ia, a in enumerate(boxes_a) for ib, b in enumerate(boxes_b)
                          if ia not in used_a and ib not in used_b]
            if not candidates:
                break
            best, ia, ib = max(candidates)
            if best < 0.5:
                break
            matched.append(best); used_a.add(ia); used_b.add(ib)
        rows.append({
            "source_a": pair["source_a"], "split_a": pair["split_a"], "image_a": pair["image_a"],
            "source_b": pair["source_b"], "split_b": pair["split_b"], "image_b": pair["image_b"],
            "train_validation_conflict": pair["split_a"] != pair["split_b"],
            "target_boxes_a": len(boxes_a), "target_boxes_b": len(boxes_b),
            "same_target_presence": bool(boxes_a) == bool(boxes_b),
            "same_target_box_count": len(boxes_a) == len(boxes_b),
            "greedy_box_matches_iou50": len(matched),
            "mean_matched_iou50": sum(matched)/len(matched) if matched else None,
            "phash_hamming": pair["phash_hamming"], "orb_inliers": pair["homography_inliers"],
        })
    with LABEL_AUDIT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)
    conflicts = [r for r in rows if r["train_validation_conflict"]]
    both = [r for r in rows if r["target_boxes_a"] and r["target_boxes_b"]]
    return {
        "strong_cross_source_pairs_audited": len(rows),
        "train_validation_conflict_pairs": len(conflicts),
        "target_presence_agreement_fraction": sum(r["same_target_presence"] for r in rows) / max(1, len(rows)),
        "target_box_count_agreement_fraction": sum(r["same_target_box_count"] for r in rows) / max(1, len(rows)),
        "pairs_with_target_boxes_in_both_sources": len(both),
        "both_source_pairs_with_at_least_one_iou50_box_match": sum(r["greedy_box_matches_iou50"] > 0 for r in both),
        "mean_matched_iou50_when_match_exists": (sum(r["mean_matched_iou50"] for r in both if r["mean_matched_iou50"] is not None)
                                                   / max(1, sum(r["mean_matched_iou50"] is not None for r in both))),
        "details_csv": str(LABEL_AUDIT_CSV.relative_to(ROOT)),
        "interpretation": "Annotation similarity is measured on visually matched images; low agreement can reflect different box rules, annotation incompleteness, or transform differences. It is not clinical adjudication.",
    }


def audit_within_source_labels(results: list[dict]) -> dict:
    source_summaries = json.loads((AUDIT_DIR / "audit_summary.json").read_text(encoding="utf-8"))["sources"]
    source_names = {s["source_id"]: s["class_names_in_source_order"] for s in source_summaries}
    rows = []
    for pair in results:
        if not pair["candidate_scope"].startswith("within_source") or pair["review_tier"] != "strong_visual_match_candidate":
            continue
        if pair["source_a"] != "gigikita_discloration_v1":
            continue
        boxes_a = target_boxes({"source": pair["source_a"], "relative_image": pair["image_a"]}, source_names)
        boxes_b = target_boxes({"source": pair["source_b"], "relative_image": pair["image_b"]}, source_names)
        pairs = sorted(((box_iou(a, b), ia, ib) for ia, a in enumerate(boxes_a) for ib, b in enumerate(boxes_b)), reverse=True)
        used_a, used_b, matched = set(), set(), []
        for iou, ia, ib in pairs:
            if iou < 0.5:
                break
            if ia not in used_a and ib not in used_b:
                used_a.add(ia); used_b.add(ib); matched.append(iou)
        rows.append({
            "split_a": pair["split_a"], "image_a": pair["image_a"], "split_b": pair["split_b"], "image_b": pair["image_b"],
            "target_boxes_a": len(boxes_a), "target_boxes_b": len(boxes_b),
            "same_target_presence": bool(boxes_a) == bool(boxes_b), "same_target_box_count": len(boxes_a) == len(boxes_b),
            "greedy_box_matches_iou50": len(matched), "mean_matched_iou50": sum(matched)/len(matched) if matched else None,
            "candidate_scope": pair["candidate_scope"], "phash_hamming": pair["phash_hamming"],
            "orb_inliers": pair["homography_inliers"],
        })
    output = AUDIT_DIR / "gigikita_duplicate_crowding_label_agreement.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)
    return {
        "strong_gigikita_duplicate_pairs": len(rows),
        "train_validation_pairs": sum(r["candidate_scope"] == "within_source_train_valid" for r in rows),
        "same_split_pairs": sum(r["candidate_scope"] == "within_source_same_split" for r in rows),
        "target_presence_agreement_fraction": sum(r["same_target_presence"] for r in rows)/max(1, len(rows)),
        "box_count_agreement_fraction": sum(r["same_target_box_count"] for r in rows)/max(1, len(rows)),
        "both_positive_pairs_with_iou50_match": sum(r["greedy_box_matches_iou50"] > 0 and r["target_boxes_a"] > 0 and r["target_boxes_b"] > 0 for r in rows),
        "details_csv": str(output.relative_to(ROOT)),
    }


def refine() -> dict:
    with (AUDIT_DIR / "train_valid_image_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    pairs = make_pairs(rows)
    results = []
    for index, pair in enumerate(pairs, 1):
        a, b = pair["row_a"], pair["row_b"]
        path_a = WORK / a["source"] / a["relative_image"]
        path_b = WORK / b["source"] / b["relative_image"]
        phash_a, phash_b = image_phash(path_a), image_phash(path_b)
        phash_distance = (phash_a ^ phash_b).bit_count()
        orb = orb_match(path_a, path_b)
        # Conservative review triage only; the original label/split is untouched.
        strong = phash_distance <= 4 or (orb["homography_inliers"] >= 20 and orb["inlier_fraction"] >= 0.25)
        result = {
            "source_a": a["source"], "split_a": a["split"], "image_a": a["relative_image"], "sha256_a": a["sha256"],
            "source_b": b["source"], "split_b": b["split"], "image_b": b["relative_image"], "sha256_b": b["sha256"],
            "candidate_scope": pair["candidate_scope"], "equal_dhash64": a["dhash64"] == b["dhash64"],
            "phash_hamming": phash_distance, **orb,
            "review_tier": "strong_visual_match_candidate" if strong else "dhash_collision_or_weak_match",
        }
        results.append(result)
        if index % 100 == 0:
            print(f"Refined {index}/{len(pairs)} candidate image pairs", flush=True)
    fields = list(results[0]) if results else []
    with PAIR_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "input_manifest_rows": len(rows), "evaluated_candidate_pairs": len(results),
        "review_tiers": {tier: sum(r["review_tier"] == tier for r in results)
                         for tier in ("strong_visual_match_candidate", "dhash_collision_or_weak_match")},
        "cross_source": {"pairs": sum(r["candidate_scope"] == "cross_source" for r in results),
                          "strong_candidates": sum(r["candidate_scope"] == "cross_source" and r["review_tier"] == "strong_visual_match_candidate" for r in results),
                          "cross_split_pairs": sum(r["candidate_scope"] == "cross_source" and r["split_a"] != r["split_b"] for r in results),
                          "cross_split_strong_candidates": sum(r["candidate_scope"] == "cross_source" and r["split_a"] != r["split_b"] and r["review_tier"] == "strong_visual_match_candidate" for r in results)},
        "within_source_train_valid": {source: {"pairs": sum(r["candidate_scope"] == "within_source_train_valid" and r["source_a"] == source for r in results),
                                               "strong_candidates": sum(r["candidate_scope"] == "within_source_train_valid" and r["source_a"] == source and r["review_tier"] == "strong_visual_match_candidate" for r in results)}
                                      for source in sorted({r["source_a"] for r in results if r["candidate_scope"] == "within_source_train_valid"})},
        "threshold_note": "Threshold is an engineering triage rule, not proof of same patient or identical capture. Visual review remains required.",
        "test_content_opened": False,
    }
    summary["cross_source_crowding_annotation_agreement"] = audit_cross_source_labels(results)
    summary["gigikita_within_source_crowding_annotation_agreement"] = audit_within_source_labels(results)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    render_contact_sheet(results)
    return summary


def render_contact_sheet(results: list[dict], max_pairs: int = 12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    chosen = sorted((r for r in results if r["review_tier"] == "strong_visual_match_candidate"),
                    key=lambda r: (-r["homography_inliers"], r["phash_hamming"]))[:max_pairs]
    if not chosen:
        return
    cols, rows = 2, len(chosen)
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(4, 2.7*rows)), squeeze=False)
    for i, pair in enumerate(chosen):
        for j, suffix in enumerate(("a", "b")):
            row = {"source": pair[f"source_{suffix}"], "image": pair[f"image_{suffix}"]}
            path = WORK / row["source"] / row["image"]
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            axes[i, j].imshow(image)
            axes[i, j].axis("off")
            axes[i, j].set_title(f"{row['source']} / {pair[f'split_{suffix}']}\n{Path(row['image']).name}", fontsize=7)
        axes[i, 0].text(1.02, .5, f"pHash d={pair['phash_hamming']}\nORB inliers={pair['homography_inliers']}",
                         transform=axes[i, 0].transAxes, va="center", fontsize=8)
    fig.suptitle("Strong visual duplicate candidates: train/validation data only", fontsize=13)
    fig.tight_layout()
    fig.savefig(SHEET, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    result = refine()
    print(json.dumps(result, indent=2))
