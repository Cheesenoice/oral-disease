"""Safely audit Roboflow YOLO archives using train/valid only.

The test members are never extracted, opened, or parsed. Original ZIP files
remain immutable under data/raw/alignment_external/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "gigikita_discloration_v1": {
        "archive": ROOT / "data/raw/alignment_external/gigikita_discloration_v1/original.zip",
        "roboflow_url": "https://universe.roboflow.com/gigikita/discloration-and-6-cases-yaubs/dataset/1",
    },
    "dental_app_v8": {
        "archive": ROOT / "data/raw/alignment_external/dental_app_v8/original.zip",
        "roboflow_url": "https://universe.roboflow.com/dental-app/dental-app/dataset/8",
    },
    "crowded_teeth_fipdz_v1": {
        "archive": ROOT / "data/raw/alignment_external/crowded_teeth_fipdz_v1/original.zip",
        "roboflow_url": "https://universe.roboflow.com/dental-examination/crowded-teeth-fipdz-o7tqa/dataset/1",
    },
}
WORK = ROOT / ".venv/work/alignment_external_audit"
REPORT_DIR = ROOT / "docs/reports/alignment/external_candidates_2026-11-18"
SUMMARY_PATH = REPORT_DIR / "audit_summary.json"
ROWS_PATH = REPORT_DIR / "train_valid_image_manifest.csv"
VALID_SPLITS = ("train", "valid")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(zf: zipfile.ZipFile) -> tuple[list[str], str, str]:
    yaml_members = [n for n in zf.namelist() if PurePosixPath(n).name.casefold() == "data.yaml"]
    if not yaml_members:
        raise ValueError("Archive does not contain data.yaml")
    raw = zf.read(yaml_members[0]).decode("utf-8", "replace")
    metadata = yaml.safe_load(raw) or {}
    names = metadata.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda k: int(k))]
    if not isinstance(names, list) or not names:
        raise ValueError("Cannot resolve class names from data.yaml")
    license_name = str((metadata.get("roboflow") or {}).get("license", "not_declared"))
    return [str(name) for name in names], raw, license_name


def inventory_archive(zf: zipfile.ZipFile) -> dict:
    """Read only central-directory metadata, not test member bytes."""
    members = zf.infolist()
    split_counts = {}
    for split in ("train", "valid", "test"):
        paths = [PurePosixPath(m.filename) for m in members
                 if PurePosixPath(m.filename).parts and PurePosixPath(m.filename).parts[0] == split]
        split_counts[split] = {
            "image_files": sum(p.suffix.casefold() in IMAGE_EXTS for p in paths),
            "label_txt_files": sum(p.suffix.casefold() == ".txt" for p in paths),
            "member_files": sum(not m.is_dir() for m in members
                                if PurePosixPath(m.filename).parts and PurePosixPath(m.filename).parts[0] == split),
        }
    return {"members": len(members), "split_inventory": split_counts}


def extract_train_valid(zf: zipfile.ZipFile, output: Path) -> int:
    """Extract only train/valid files with path traversal and test guards."""
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    count = 0
    for info in zf.infolist():
        parts = PurePosixPath(info.filename).parts
        if not parts or parts[0] not in VALID_SPLITS:
            continue
        if info.is_dir():
            continue
        if ".." in parts or PurePosixPath(info.filename).is_absolute():
            raise ValueError(f"Unsafe ZIP member: {info.filename}")
        if parts[0] == "test":
            raise AssertionError("Test split extraction is prohibited")
        destination = output.joinpath(*parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        count += 1
    if any(p.relative_to(output).parts and p.relative_to(output).parts[0] == "test"
           for p in output.rglob("*")):
        raise AssertionError("A test path was extracted")
    return count


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    gray = ImageOps.grayscale(image).resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = []
    for y in range(hash_size):
        row = y * (hash_size + 1)
        bits.extend(pixels[row + x] > pixels[row + x + 1] for x in range(hash_size))
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-12)


def parse_yolo_label(path: Path, n_classes: int, width: int, height: int):
    objects, errors = [], []
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            cls = int(fields[0])
            values = list(map(float, fields[1:]))
        except ValueError:
            errors.append(f"line {line_no}: non-numeric field")
            continue
        if cls < 0 or cls >= n_classes:
            errors.append(f"line {line_no}: class id {cls} outside [0,{n_classes-1}]")
            continue
        if not all(map(math.isfinite, values)):
            errors.append(f"line {line_no}: non-finite coordinates")
            continue
        if len(values) == 4:
            xc, yc, bw, bh = values
            if bw <= 0 or bh <= 0:
                errors.append(f"line {line_no}: box width/height must be positive")
                continue
            # Do not silently clip labels. Flag out-of-image geometry as invalid.
            x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
            if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                errors.append(f"line {line_no}: normalized box exceeds image bounds")
                continue
            objects.append({"class_id": cls, "format": "xywh_box", "xc": xc, "yc": yc, "w": bw, "h": bh,
                            "xyxy": [x1 * width, y1 * height, x2 * width, y2 * height]})
        elif len(values) >= 6 and len(values) % 2 == 0:
            # YOLO segmentation row: class followed by >=3 normalized (x, y) vertices.
            xs, ys = values[0::2], values[1::2]
            if any(value < 0 or value > 1 for value in values):
                errors.append(f"line {line_no}: polygon vertex outside normalized image bounds")
                continue
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if x2 <= x1 or y2 <= y1:
                errors.append(f"line {line_no}: degenerate polygon bounding box")
                continue
            objects.append({"class_id": cls, "format": "segmentation_polygon", "xc": (x1+x2)/2,
                            "yc": (y1+y2)/2, "w": x2-x1, "h": y2-y1,
                            "xyxy": [x1 * width, y1 * height, x2 * width, y2 * height]})
        else:
            errors.append(f"line {line_no}: unsupported YOLO row with {1+len(values)} fields")
    return objects, errors


def _overlay_samples(records: list[dict], output_path: Path, title: str, limit: int = 18):
    positive = [row for row in records if row["box_count"] > 0 and row["decode_error"] is None]
    # Stable sampling by content hash keeps the audit deterministic.
    selected = sorted(positive, key=lambda row: row["sha256"])[:limit]
    if not selected:
        return
    cols = 3
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, selected):
        image = Image.open(row["local_path"]).convert("RGB")
        ax.imshow(image)
        for obj in row["objects"]:
            x1, y1, x2, y2 = obj["xyxy"]
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linewidth=2, edgecolor="lime"))
            ax.text(x1, max(0, y1 - 4), row["class_names"][obj["class_id"]], color="black", fontsize=8,
                    bbox={"facecolor": "lime", "alpha": .85, "pad": 2})
        ax.set_title(f"{row['source']} | {row['split']} | boxes={row['box_count']}\n{Path(row['relative_image']).name}", fontsize=9)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def audit_source(source_id: str, config: dict) -> tuple[dict, list[dict]]:
    archive_path = config["archive"]
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    source_output = WORK / source_id
    with zipfile.ZipFile(archive_path) as zf:
        inventory = inventory_archive(zf)
        class_names, yaml_raw, declared_license = load_metadata(zf)
        extracted_members = extract_train_valid(zf, source_output)
    split_summary = {}
    class_counts = {split: Counter() for split in VALID_SPLITS}
    records = []
    errors = []
    for split in VALID_SPLITS:
        image_dir = source_output / split / "images"
        label_dir = source_output / split / "labels"
        image_files = sorted(p for p in image_dir.glob("*") if p.suffix.casefold() in IMAGE_EXTS)
        paired, missing_labels, orphan_labels = 0, [], []
        image_stems = {p.stem for p in image_files}
        if label_dir.exists():
            orphan_labels = sorted(p.name for p in label_dir.glob("*.txt") if p.stem not in image_stems)
        split_records = []
        for image_index, image_path in enumerate(image_files, 1):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(image_path.name)
            else:
                paired += 1
            try:
                with Image.open(image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    image.load()
                width, height = image.size
                image_hash = file_sha256(image_path)
                image_dhash = dhash(image)
                decode_error = None
            except Exception as exc:
                width = height = None
                image_hash = image_dhash = None
                decode_error = f"{type(exc).__name__}: {exc}"
                errors.append({"source": source_id, "split": split, "file": image_path.name,
                               "issue": "image_decode", "detail": decode_error})
                image = None
            objects, label_errors = parse_yolo_label(label_path, len(class_names), width or 1, height or 1)
            for obj in objects:
                class_counts[split][obj["class_id"]] += 1
            for detail in label_errors:
                errors.append({"source": source_id, "split": split, "file": label_path.name,
                               "issue": "label_geometry_or_format", "detail": detail})
            if label_errors:
                # Preserve invalid line count in record; never repair silently.
                pass
            record = {
                "source": source_id, "split": split, "relative_image": image_path.relative_to(source_output).as_posix(),
                "local_path": str(image_path), "sha256": image_hash, "dhash64": image_dhash,
                "width": width, "height": height, "aspect_ratio": width / height if width and height else None,
                "label_present": label_path.exists(), "empty_label": label_path.exists() and not label_path.read_text(encoding="utf-8", errors="replace").strip(),
                "box_count": len(objects), "invalid_label_lines": len(label_errors), "decode_error": decode_error,
                "objects": objects, "class_names": class_names,
            }
            records.append(record)
            split_records.append(record)
            if image_index % 1000 == 0:
                print(f"{source_id}: audited {split} image {image_index}/{len(image_files)}", flush=True)
        split_summary[split] = {
            "images": len(image_files), "paired_label_files": paired,
            "missing_labels": len(missing_labels), "orphan_label_files": len(orphan_labels),
            "images_with_boxes": sum(r["box_count"] > 0 for r in split_records),
            "empty_labels": sum(r["empty_label"] for r in split_records),
            "decode_errors": sum(r["decode_error"] is not None for r in split_records),
            "class_box_counts": {class_names[i]: class_counts[split][i] for i in range(len(class_names))},
            "class_image_counts": {class_names[i]: sum(any(o["class_id"] == i for o in r["objects"]) for r in split_records)
                                   for i in range(len(class_names))},
            "missing_label_examples": missing_labels[:20], "orphan_label_examples": orphan_labels[:20],
        }
        _overlay_samples(split_records, REPORT_DIR / f"{source_id}_{split}_overlay.jpg",
                         f"{source_id} - {split} bounding-box audit")

    # Exact/dHash overlaps within train-vs-validation for this source.
    cross_split_exact, cross_split_near = [], []
    by_hash = defaultdict(list)
    by_dhash = defaultdict(list)
    for row in records:
        if row["sha256"]:
            by_hash[row["sha256"]].append(row)
        if row["dhash64"]:
            by_dhash[row["dhash64"]].append(row)
    for key, group in by_hash.items():
        if len({r["split"] for r in group}) > 1:
            cross_split_exact.append({"sha256": key, "members": [r["relative_image"] for r in group]})
    # dHash identical is only a candidate; no auto-removal.
    for key, group in by_dhash.items():
        if len({r["split"] for r in group}) > 1 and len(group) > 1:
            cross_split_near.append({"dhash64": key, "members": [r["relative_image"] for r in group]})

    summary = {
        "source_id": source_id, "archive": str(archive_path.relative_to(ROOT)),
        "archive_sha256": file_sha256(archive_path), "archive_bytes": archive_path.stat().st_size,
        "roboflow_url": config["roboflow_url"], "license_from_data_yaml": declared_license,
        "class_names_in_source_order": class_names, "data_yaml": yaml_raw,
        "archive_inventory": inventory, "extracted_train_valid_members": extracted_members,
        "train_valid_only_extraction": True, "test_content_opened": False,
        "splits": split_summary, "cross_split_exact_duplicates": cross_split_exact,
        "cross_split_identical_dhash_candidates": cross_split_near,
        "label_image_geometry_errors": errors,
        "limitations": ["Source provenance and patient IDs are not provided in the export.",
                        "Source class names are retained; no clinical relabeling was performed.",
                        "A label file with no objects is not proof that all target classes were annotated absent."],
    }
    return summary, records


def compare_candidate_images(records: list[dict]) -> dict:
    by_hash = defaultdict(list)
    by_dhash = defaultdict(list)
    for row in records:
        if row["sha256"]:
            by_hash[row["sha256"]].append({"source": row["source"], "split": row["split"], "image": row["relative_image"]})
        if row["dhash64"]:
            by_dhash[row["dhash64"]].append({"source": row["source"], "split": row["split"], "image": row["relative_image"]})
    cross_exact = [v for v in by_hash.values() if len({x["source"] for x in v}) > 1]
    cross_dhash = [v for v in by_dhash.values() if len({x["source"] for x in v}) > 1]
    return {"candidate_cross_source_exact_duplicate_groups": cross_exact,
            "candidate_cross_source_identical_dhash_groups": cross_dhash,
            "note": "Exact SHA-256 duplicates are confirmed byte-identical. Equal 64-bit dHash groups are review candidates only; no images were removed."}


def compare_existing_visible(records: list[dict]) -> dict:
    """Compare only the already-published train/valid package, never its test."""
    manifest = ROOT / "data/manifests/alignment_visible_v2_manifest.csv"
    if not manifest.is_file():
        return {"status": "not_compared", "reason": "existing visible v2 manifest missing"}
    existing_hashes = set()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") in VALID_SPLITS and row.get("image_sha256"):
                existing_hashes.add(row["image_sha256"].casefold())
    overlaps = [r for r in records if r["sha256"] and r["sha256"].casefold() in existing_hashes]
    return {"status": "compared_against_existing_train_valid_only", "existing_image_hash_count": len(existing_hashes),
            "exact_overlap_count": len(overlaps),
            "exact_overlaps": [{"source": r["source"], "split": r["split"], "image": r["relative_image"], "sha256": r["sha256"]}
                               for r in overlaps]}


def run_audit() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summaries, all_records = [], []
    for source_id, config in SOURCES.items():
        summary, records = audit_source(source_id, config)
        summaries.append(summary)
        all_records.extend(records)
    summary = {
        "generated_by": "src/scripts/audit_external_alignment_candidates.py",
        "sources": summaries,
        "cross_source_overlap": compare_candidate_images(all_records),
        "existing_project_overlap": compare_existing_visible(all_records),
        "policy": {"splits_read": ["train", "valid"], "test_unopened": True,
                   "source_archives_immutable": True, "class_semantics_preserved": True,
                   "no_training_started": True},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["source", "split", "relative_image", "sha256", "dhash64", "width", "height", "aspect_ratio",
              "label_present", "empty_label", "box_count", "invalid_label_lines", "decode_error",
              "class_box_counts_json"]
    with ROWS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_records:
            per_class = Counter(row["class_names"][obj["class_id"]] for obj in row["objects"])
            writer.writerow({**{key: row[key] for key in fields if key != "class_box_counts_json"},
                             "class_box_counts_json": json.dumps(per_class, ensure_ascii=False)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="Print concise results after running the audit")
    args = parser.parse_args()
    summary = run_audit()
    if args.summary:
        for source in summary["sources"]:
            print(f"\n{source['source_id']} | classes={source['class_names_in_source_order']}")
            for split, values in source["splits"].items():
                print(split, json.dumps(values, ensure_ascii=False))
            print("cross-split exact duplicate groups:", len(source["cross_split_exact_duplicates"]))
        print("cross-source exact duplicate groups:", len(summary["cross_source_overlap"]["candidate_cross_source_exact_duplicate_groups"]))
        print("existing project exact overlap:", summary["existing_project_overlap"].get("exact_overlap_count"))
        print("summary:", SUMMARY_PATH)
        print("manifest:", ROWS_PATH)


if __name__ == "__main__":
    main()
