"""Readable shared utilities for the executed data preparation notebooks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict

import pandas as pd

from .acquire import digest
from .audit import safe_extract


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_sources(root: Path) -> pd.DataFrame:
    """Extract missing archives and verify all extracted member sizes; keep originals."""
    import rarfile

    rows = []
    caries_dir = root / "data/raw/caries"
    caries_archive = caries_dir / "Dataset.zip"
    expected = json.loads((root / "src/configs/sources.json").read_text())["caries"]
    actual_md5 = digest(caries_archive, "md5")
    if actual_md5 != expected["md5"]:
        raise ValueError("Caries archive does not match the publisher MD5")
    with zipfile.ZipFile(caries_archive) as archive:
        caries_members = [(info.filename, info.file_size) for info in archive.infolist() if not info.is_dir()]
    if not (caries_dir / "Dataset").exists():
        safe_extract(caries_archive, caries_dir)
    for name, size in caries_members:
        if not (caries_dir / name).is_file() or (caries_dir / name).stat().st_size != size:
            raise ValueError(f"Incomplete caries extraction: {name}")
    rows.append({"task": "caries", "archive": str(caries_archive.relative_to(root)),
                 "archive_bytes": caries_archive.stat().st_size, "extracted_files": len(caries_members),
                 "checksum": "Publisher MD5 matched", "status": "extracted and verified"})

    gum_dir = root / "data/raw/gum"
    gum_archive = gum_dir / "Dataset.rar"
    with rarfile.RarFile(gum_archive) as archive:
        gum_members = []
        for info in archive.infolist():
            portable = info.filename.replace("\\", "/")
            target = (gum_dir / portable).resolve()
            if not target.is_relative_to(gum_dir.resolve()) or ":" in portable:
                raise ValueError(f"Unsafe RAR path: {portable}")
            if info.is_symlink():
                raise ValueError("RAR symbolic links are not accepted")
            if not info.isdir():
                gum_members.append((portable, info.file_size))
    if not (gum_dir / "Dataset").exists():
        binary = shutil.which("unrar") or shutil.which("UnRAR")
        if binary is None:
            known = Path(r"C:\Program Files\WinRAR\UnRAR.exe")
            binary = str(known) if known.is_file() else None
        if binary is None:
            raise RuntimeError("UnRAR is required to extract the original gingivitis archive")
        print(f"Extracting {len(gum_members):,} gingivitis files with UnRAR ...", flush=True)
        result = subprocess.run([binary, "x", "-o-", "-idq", "-y", str(gum_archive), str(gum_dir) + os.sep],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"UnRAR failed: {result.stderr[-2000:]} {result.stdout[-2000:]}")
    for name, size in gum_members:
        if not (gum_dir / name).is_file() or (gum_dir / name).stat().st_size != size:
            raise ValueError(f"Incomplete gingivitis extraction: {name}")
    receipt_path = gum_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(extracted=True, extracted_files=len(gum_members),
                   sha256=digest(gum_archive), checksum_note="Locally computed SHA-256; no publisher checksum available")
    write_json(receipt_path, receipt)
    rows.append({"task": "gum", "archive": str(gum_archive.relative_to(root)),
                 "archive_bytes": gum_archive.stat().st_size, "extracted_files": len(gum_members),
                 "checksum": "Local SHA-256 recorded", "status": "extracted and verified"})
    return pd.DataFrame(rows)


def source_inventory(root: Path) -> pd.DataFrame:
    """List extracted source files without modifying their names or content."""
    rows = []
    for task in ("caries", "gum"):
        source = root / "data/raw" / task / "Dataset"
        for path in sorted(source.rglob("*")):
            if path.is_file():
                rows.append({"task": task, "path": path.relative_to(root).as_posix(),
                             "extension": path.suffix.lower(), "bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def inspect_images(root: Path, inventory: pd.DataFrame, cache: Path, workers: int = 4) -> pd.DataFrame:
    """Read dimensions, decode a JPEG preview, hash file bytes and compute visual hashes."""
    from PIL import Image
    import imagehash

    previous = {}
    if cache.is_file():
        previous = pd.read_csv(cache, keep_default_na=False).set_index("image_path").to_dict("index")
    images = inventory[inventory.extension.isin([".jpg", ".jpeg", ".png"])]

    def inspect(row):
        path = root / row["path"]
        stat = path.stat()
        prior = previous.get(row["path"])
        if prior and int(prior["bytes"]) == stat.st_size and int(prior["mtime_ns"]) == stat.st_mtime_ns:
            return dict(prior, image_path=row["path"])
        result = {"task": row["task"], "image_path": row["path"], "bytes": stat.st_size,
                  "mtime_ns": stat.st_mtime_ns, "sha256": digest(path), "decode_error": ""}
        try:
            with Image.open(path) as image:
                result.update(width=image.width, height=image.height,
                              exif_orientation=int(image.getexif().get(274, 1)))
                image.draft("RGB", (512, 512))
                image.load()
                preview = image.convert("RGB")
                result.update(phash=str(imagehash.phash(preview)), dhash=str(imagehash.dhash(preview)))
        except Exception as error:
            result.update(width=0, height=0, exif_orientation=0, phash="", dhash="",
                          decode_error=f"{type(error).__name__}: {error}")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = pd.DataFrame(pool.map(inspect, images.to_dict("records")))
    result = result.sort_values(["task", "image_path"]).reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache, index=False)
    return result


CARIES_NAMES = ["visible_caries"]
GUM_NAMES = ["no_visible_inflammation", "mild_inflammation", "marked_inflammation"]
GUM_CLASS_MAP = {2: 0, 3: 1, 4: 1, 5: 2, 6: 2}


def audit_annotations(root: Path, images: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use named LabelMe rectangles for caries and verified numeric YOLO rows for gingiva.

    Any malformed disease annotation quarantines the entire image, avoiding a
    false negative introduced by silently dropping one of its boxes.
    """
    records, annotations = [], []
    for info in images.to_dict("records"):
        path = root / info["image_path"]
        task = info["task"]
        sample_id = hashlib.sha256(info["image_path"].encode()).hexdigest()[:20]
        record = dict(info, sample_id=sample_id, label_path="", source_split="not_provided",
                      source_collection="", view="unknown", reasons=[], source_boxes=[], target_boxes=[])
        if info["decode_error"]:
            record["reasons"].append("image_decode_error")
        # Tag 0 defines no rotation. Reviewed LabelMe boxes align with stored
        # pixels, and source annotation dimensions are checked below.
        if int(info["exif_orientation"]) not in (0, 1):
            record["reasons"].append("nonstandard_exif_orientation_requires_review")
        try:
            if task == "caries":
                suffix = path.as_posix().split("/Images/", 1)[1].split("/")
                collection, view = suffix[:2]
                record.update(source_collection=collection, view=view)
                label = root / "data/raw/caries/Dataset/Annotations/Labelme" / collection / view / f"{path.stem}.json"
                match = re.search(r"anonymous_(\d{3}-\d{3}-\d+)-\d{2}_", path.stem)
                timestamp = re.search(r"\d{13}", path.stem)
                if match:
                    record.update(group_hint=f"caries:filename_case:{match.group(1)}",
                                  group_basis="Shared filename case token; patient identity is not independently verified")
                elif timestamp:
                    record.update(group_hint=f"caries:timestamp:{timestamp.group(0)}",
                                  group_basis="Shared filename timestamp only; sessions/patients may remain unlinked")
                else:
                    record.update(group_hint=f"caries:image:{sample_id}", group_basis="Image only; no reliable case token")
                if not label.is_file():
                    record["reasons"].append("missing_annotation_not_a_negative")
                else:
                    record["label_path"] = label.relative_to(root).as_posix()
                    document = json.loads(label.read_text(encoding="utf-8-sig"))
                    if (document["imageWidth"], document["imageHeight"]) != (info["width"], info["height"]):
                        record["reasons"].append("annotation_image_dimensions_mismatch")
                    if Path(document["imagePath"].replace("\\", "/")).name != path.name:
                        record["reasons"].append("annotation_image_name_mismatch")
                    for shape in document["shapes"]:
                        if shape["shape_type"] != "rectangle" or shape["label"] not in {"D", "d"}:
                            record["reasons"].append("unknown_caries_shape_or_class")
                            continue
                        (x1, y1), (x2, y2) = shape["points"]
                        record["source_boxes"].append({"source_class": shape["label"], "target_class": 0,
                                                       "xyxy": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]})
            else:
                record.update(source_split=path.parent.parent.name, source_collection="professional_frontal",
                              view="frontal", group_hint=f"gum:image_id:{path.stem}",
                              group_basis="Source image ID only; no participant identifier is supplied")
                label = path.parent.parent / "Labels" / f"{path.stem}.txt"
                if not label.is_file():
                    record["reasons"].append("missing_annotation_not_a_negative")
                else:
                    record["label_path"] = label.relative_to(root).as_posix()
                    for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                        if not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            raise ValueError(f"Expected five columns at line {line_number}")
                        source_class = int(parts[0])
                        if source_class not in range(7):
                            raise ValueError(f"Unexpected gingivitis class {source_class}")
                        cx, cy, width, height = map(float, parts[1:])
                        record["source_boxes"].append({"source_class": str(source_class),
                                                       "target_class": GUM_CLASS_MAP.get(source_class),
                                                       "xyxy": [(cx - width / 2) * info["width"],
                                                                (cy - height / 2) * info["height"],
                                                                (cx + width / 2) * info["width"],
                                                                (cy + height / 2) * info["height"]]})
                    if not any(box["target_class"] is not None for box in record["source_boxes"]):
                        record["reasons"].append("no_gingival_region_labels")
            import math
            for box_index, box in enumerate(record["source_boxes"]):
                x1, y1, x2, y2 = box["xyxy"]
                valid = (all(math.isfinite(v) for v in box["xyxy"]) and 0 <= x1 < x2 <= info["width"]
                         and 0 <= y1 < y2 <= info["height"])
                if not valid:
                    record["reasons"].append("invalid_box_geometry")
                annotations.append({"task": task, "sample_id": sample_id, "image_path": info["image_path"],
                                    "source_split": record["source_split"], "box_index": box_index,
                                    "source_class": box["source_class"], "target_class": box["target_class"],
                                    "valid_geometry": valid, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                if valid and box["target_class"] is not None:
                    record["target_boxes"].append([int(box["target_class"]),
                                                    (x1 + x2) / (2 * info["width"]),
                                                    (y1 + y2) / (2 * info["height"]),
                                                    (x2 - x1) / info["width"], (y2 - y1) / info["height"]])
        except Exception as error:
            record["reasons"].append(f"annotation_parse_error:{type(error).__name__}:{error}")
        record["reasons"] = sorted(set(record["reasons"]))
        record["eligible"] = not record["reasons"]
        record["label_count"] = len(record["target_boxes"])
        records.append(record)
    return pd.DataFrame(records), pd.DataFrame(annotations)


def caries_format_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report format coverage and COCO category IDs independently for every JSON file."""
    source = root / "data/raw/caries/Dataset/Annotations"
    formats = []
    for folder, suffix in (("Darknet_YOLO", ".txt"), ("Labelme", ".json"), ("pascal-voc", ".xml")):
        formats.append({"format": folder, "files": len(list((source / folder).rglob(f"*{suffix}")))})
    categories = []
    for path in sorted((source / "MS_coco").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        for item in document["categories"]:
            categories.append({"file": path.name, "category_id": item["id"], "name": item["name"],
                               "images_in_file": len(document["images"]), "boxes_in_file": len(document["annotations"])})
    return pd.DataFrame(formats), pd.DataFrame(categories)


def duplicate_pairs(records: pd.DataFrame, radius: int = 4) -> pd.DataFrame:
    """Conservative near-duplicate candidates: pHash <= 4 AND dHash <= 4, similar aspect.

    Candidates become same-split constraints; similarity is not proof of shared identity.
    A BK-tree avoids comparing every pair of images.
    """
    columns = ["task", "sample_a", "sample_b", "phash_distance", "dhash_distance", "relation"]
    pairs = []
    for task, subset in records.groupby("task", sort=True):
        tree = None
        lookup = {}
        for row in subset.sort_values("sample_id").to_dict("records"):
            if not row["phash"] or not row["dhash"]:
                continue
            value = int(row["phash"], 16)
            candidates = []
            stack = [tree] if tree else []
            while stack:
                node = stack.pop()
                distance = (value ^ node[0]).bit_count()
                if distance <= radius:
                    candidates.extend(node[1])
                stack.extend(child for edge, child in node[2].items()
                             if distance - radius <= edge <= distance + radius)
            for other_id in candidates:
                other = lookup[other_id]
                d_distance = (int(row["dhash"], 16) ^ int(other["dhash"], 16)).bit_count()
                aspect_ratio = (row["width"] / row["height"]) / (other["width"] / other["height"])
                if d_distance <= radius and 0.9 <= aspect_ratio <= 1.1:
                    pairs.append({"task": task, "sample_a": other_id, "sample_b": row["sample_id"],
                                  "phash_distance": (value ^ int(other["phash"], 16)).bit_count(),
                                  "dhash_distance": d_distance,
                                  "relation": "exact_bytes" if row["sha256"] == other["sha256"] else "visual_candidate"})
            lookup[row["sample_id"]] = row
            if tree is None:
                tree = [value, [row["sample_id"]], {}]
            else:
                node = tree
                while True:
                    edge = (value ^ node[0]).bit_count()
                    if edge == 0:
                        node[1].append(row["sample_id"])
                        break
                    if edge not in node[2]:
                        node[2][edge] = [value, [row["sample_id"]], {}]
                        break
                    node = node[2][edge]
    return pd.DataFrame(pairs, columns=columns)


def assign_groups(records: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Join filename hints, identical bytes and conservative visual candidates."""
    rows = records.to_dict("records")
    parent = {row["sample_id"]: row["sample_id"] for row in rows}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def join(a, b):
        x, y = sorted([find(a), find(b)])
        parent[y] = x

    seen = {}
    for row in rows:
        for key in [(row["task"], "hint", row["group_hint"]), (row["task"], "sha256", row["sha256"])]:
            if key in seen:
                join(row["sample_id"], seen[key])
            else:
                seen[key] = row["sample_id"]
    for pair in pairs.to_dict("records"):
        join(pair["sample_a"], pair["sample_b"])
    for row in rows:
        row["group_id"] = f"{row['task']}:constraint:{find(row['sample_id'])}"
    # Exact duplicates do not count as extra independent examples.
    for _, subset in records[records.eligible].groupby(["task", "sha256"]):
        if len(subset) < 2:
            continue
        signatures = {json.dumps(sorted(row["target_boxes"]), sort_keys=True) for row in subset.to_dict("records")}
        ids = set(subset.sample_id) if len(signatures) != 1 else set(sorted(subset.sample_id)[1:])
        for row in rows:
            if row["sample_id"] in ids:
                reason = "exact_duplicate_annotation_conflict" if len(signatures) != 1 else "redundant_exact_duplicate"
                row["reasons"] = sorted(set(row["reasons"] + [reason]))
                row["eligible"] = False
    return pd.DataFrame(rows)


def plot_annotation_grid(root: Path, records: pd.DataFrame, output: Path, task: str,
                         limit: int = 6, invalid: bool = False, swap_gum_dimensions: bool = False):
    """Draw original source boxes without altering or clipping annotation coordinates."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    subset = records[(records.task == task) & (records.eligible != invalid)]
    if invalid:
        subset = subset[subset.source_boxes.map(bool)]
    # Deterministic spread over views/source folders, not a hand-picked accuracy gallery.
    subset = subset.sort_values("image_path")
    if len(subset) > limit:
        import numpy as np
        subset = subset.iloc[np.linspace(0, len(subset) - 1, limit).astype(int)]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    palette = {"D": "lime", "d": "cyan", "0": "white", "1": "silver", "2": "cyan",
               "3": "lime", "4": "yellow", "5": "orange", "6": "red"}
    for axis, row in zip(axes.flat, subset.to_dict("records")):
        with Image.open(root / row["image_path"]) as image:
            image.draft("RGB", (900, 900))
            axis.imshow(image.convert("RGB"), extent=(0, row["width"], row["height"], 0))
        for box in row["source_boxes"]:
            x1, y1, x2, y2 = box["xyxy"]
            if swap_gum_dimensions:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = (y2 - y1) / row["height"] * row["width"], (x2 - x1) / row["width"] * row["height"]
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            color = palette.get(str(box["source_class"]), "magenta")
            axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5))
            axis.text(x1, y1, str(box["source_class"]), color="black", fontsize=7,
                      bbox={"facecolor": color, "alpha": 0.8, "pad": 1})
        axis.set_xlim(0, row["width"])
        axis.set_ylim(row["height"], 0)
        axis.set_title(Path(row["image_path"]).name[:45] + "\n" + "; ".join(row["reasons"]), fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(subset):]:
        axis.axis("off")
    figure.suptitle(f"{task}: {'quarantined source examples' if invalid else 'source annotation geometry'}"
                   + (" — swapped column hypothesis" if swap_gum_dimensions else ""))
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    return figure


def serialize_boxes(boxes: list[list]) -> str:
    from .data import parse_yolo
    text = "\n".join(" ".join([str(int(box[0])), *(f"{value:.8f}" for value in box[1:])]) for box in boxes)
    text += "\n" if text else ""
    parse_yolo(text)
    return text


def prepare_task(root: Path, task: str, selected: pd.DataFrame, audit_path: Path,
                 seed: int = 42, max_edge: int = 1280, quality: int = 95, workers: int = 4) -> pd.DataFrame:
    """Create compact RGB JPEGs and valid YOLO labels, then lock observable groups.

    Prepared split files are hard links on the same volume where supported, avoiding
    a second full image copy. Original files are never modified or moved.
    """
    from PIL import Image
    import PIL
    from .data import assign_splits

    if task not in ("caries", "gum"):
        raise ValueError("Unknown dataset task")
    root = root.resolve()
    manifest_dir = root / "data/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = root / "src/configs/label_mappings.json"
    mapping = json.loads(mapping_path.read_text())[task]
    if not mapping["approved"]:
        raise ValueError(f"{task}: source interpretation has not been verified")
    settings = {"protocol": "dentalselfie-data-v1", "task": task, "seed": seed,
                "max_edge": max_edge, "jpeg_quality": quality, "jpeg_subsampling": 0,
                "pillow": PIL.__version__, "audit_sha256": digest(audit_path),
                "mapping_sha256": digest(mapping_path)}
    lock_path = manifest_dir / f"{task}.lock.json"
    split_path = manifest_dir / f"{task}.splits.csv"
    destination = root / "data/processed" / task
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text())
        if lock["settings"] != settings or lock["splits_sha256"] != digest(split_path):
            raise ValueError("Locked dataset differs from this request; create a new protocol version explicitly")
        print(f"{task}: reusing the identical locked dataset", flush=True)
        return pd.read_csv(split_path, keep_default_na=False)
    stage = destination if destination.exists() else root / ".venv/work/data_preparation" / task
    if destination.exists() and not (destination / "preparation_settings.json").is_file():
        raise FileExistsError(f"Unrecognized prepared destination without a lock: {destination}")
    stage.mkdir(parents=True, exist_ok=True)
    stage_settings = stage / "preparation_settings.json"
    if stage_settings.is_file() and json.loads(stage_settings.read_text()) != settings:
        raise ValueError("Partial staging belongs to another preparation protocol")
    write_json(stage_settings, settings)
    selected = selected[selected.task == task].sort_values("sample_id")
    if len(selected) == 0:
        raise ValueError(f"{task}: no eligible images")
    assignment = assign_splits(selected.to_dict("records"), seed=seed)

    def normalize(row):
        source = root / row["image_path"]
        if digest(source) != row["sha256"]:
            raise ValueError(f"Source image changed after audit: {source}")
        image_out = stage / "images" / f"{row['sample_id']}.jpg"
        label_out = stage / "labels" / f"{row['sample_id']}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        text = serialize_boxes(row["target_boxes"])
        reuse = False
        if image_out.is_file() and label_out.is_file() and label_out.read_text() == text:
            try:
                with Image.open(image_out) as image:
                    image.load()
                    width, height = image.size
                    reuse = image.format == "JPEG" and image.mode == "RGB" and max(width, height) <= max_edge
            except OSError:
                reuse = False
        if not reuse:
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                width, height = image.size
                image.save(image_out, format="JPEG", quality=quality, subsampling=0, optimize=False)
        label_out.write_bytes(text.encode("utf-8"))
        prepared = {"sample_id": row["sample_id"], "task": task,
                    "image": f"data/processed/{task}/images/{image_out.name}",
                    "label": f"data/processed/{task}/labels/{label_out.name}",
                    "source_image": row["image_path"], "source_label": row["label_path"],
                    "source_sha256": row["sha256"], "source_label_sha256": digest(root / row["label_path"]),
                    "sha256": digest(image_out), "label_sha256": digest(label_out),
                    "group_id": row["group_id"], "group_basis": row["group_basis"], "view": row["view"],
                    "source_split": row["source_split"], "source_collection": row["source_collection"],
                    "width": width, "height": height, "source_width": row["width"], "source_height": row["height"],
                    "label_count": len(row["target_boxes"]),
                    "source_classes": json.dumps(sorted(box["source_class"] for box in row["source_boxes"])),
                    "bytes": image_out.stat().st_size, "split": row["split"]}
        for folder, file in (("images", image_out), ("labels", label_out)):
            link = stage / "dataset" / folder / row["split"] / file.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists():
                if digest(link) != digest(file):
                    raise ValueError(f"Staging file differs: {link}")
            else:
                try:
                    os.link(file, link)
                except OSError:
                    shutil.copy2(file, link)
        return prepared

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(normalize, assignment))
    import yaml
    (stage / "dataset/data.yaml").write_text(yaml.safe_dump({"train": "images/train", "val": "images/val",
                                                            "test": "images/test", "names": mapping["names"]}),
                                             encoding="utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    result.to_csv(stage / "preparation_manifest.csv", index=False)
    if stage != destination:
        if not stage.resolve().is_relative_to(root) or not destination.resolve().is_relative_to(root):
            raise ValueError("Prepared dataset promotion must stay within the project")
        # Windows scanners may briefly hold freshly written JPEGs open.
        for attempt in range(6):
            try:
                stage.rename(destination)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
    normalized_path = manifest_dir / f"{task}.normalized.csv"
    result.drop(columns="split").to_csv(normalized_path, index=False)
    result.to_csv(split_path, index=False)
    write_json(lock_path, {"task": task, "seed": seed, "settings": settings,
                           "manifest_sha256": digest(normalized_path), "splits_sha256": digest(split_path),
                           "counts": dict(Counter(result.split)),
                           "independence_claim": "Observable filename/exact/visual groups only; not certified patient-disjoint",
                           "calibration_policy": "Use train only; never validation or test"})
    return result


def validate_prepared(root: Path, manifests: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Check bytes, geometry, class coverage and cross-split group/duplicate isolation."""
    from .data import parse_yolo
    import yaml

    rows = []
    for task, task_rows in manifests.groupby("task"):
        lock_path = root / "data/manifests" / f"{task}.lock.json"
        lock = json.loads(lock_path.read_text())
        split_path = root / "data/manifests" / f"{task}.splits.csv"
        assert digest(split_path) == lock["splits_sha256"]
        dataset = root / "data/processed" / task / "dataset"
        names = yaml.safe_load((dataset / "data.yaml").read_text())["names"]
        class_counts = Counter()
        for row in task_rows.to_dict("records"):
            assert digest(root / row["image"]) == row["sha256"]
            assert digest(root / row["label"]) == row["label_sha256"]
            assert digest(dataset / "images" / row["split"] / f"{row['sample_id']}.jpg") == row["sha256"]
            assert digest(dataset / "labels" / row["split"] / f"{row['sample_id']}.txt") == row["label_sha256"]
            boxes = parse_yolo((root / row["label"]).read_text())
            assert len(boxes) == row["label_count"]
            for class_id, *coordinates in boxes:
                assert 0 <= class_id < len(names)
                class_counts[(row["split"], class_id)] += 1
        for key in ("group_id", "source_sha256", "sha256"):
            assert task_rows.groupby(key).split.nunique().max() == 1, f"Cross-split overlap: {task}/{key}"
        lookup = task_rows.set_index("sample_id").split.to_dict()
        checked_pairs = 0
        for pair in pairs[pairs.task == task].to_dict("records"):
            if pair["sample_a"] in lookup and pair["sample_b"] in lookup:
                assert lookup[pair["sample_a"]] == lookup[pair["sample_b"]]
                checked_pairs += 1
        for split in ("train", "val", "test"):
            assert len(task_rows[task_rows.split == split]) > 0
            for class_id, name in enumerate(names):
                assert class_counts[(split, class_id)] > 0, f"Class absent from split: {task}/{split}/{name}"
                rows.append({"task": task, "split": split, "class_id": class_id, "class_name": name,
                             "boxes": class_counts[(split, class_id)], "validation": "passed",
                             "same_split_visual_pairs_checked": checked_pairs})
    return pd.DataFrame(rows)


def plot_prepared_grid(root: Path, manifests: pd.DataFrame, output: Path, task: str):
    from .data import parse_yolo
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    subset = manifests[manifests.task == task].sort_values(["split", "sample_id"])
    subset = subset.iloc[np.linspace(0, len(subset) - 1, min(6, len(subset))).astype(int)]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    names = CARIES_NAMES if task == "caries" else GUM_NAMES
    colors = ["lime", "yellow", "red"]
    for axis, row in zip(axes.flat, subset.to_dict("records")):
        with Image.open(root / row["image"]) as image:
            axis.imshow(image)
            width, height = image.size
        for class_id, cx, cy, w, h in parse_yolo((root / row["label"]).read_text()):
            x, y = (cx - w / 2) * width, (cy - h / 2) * height
            axis.add_patch(Rectangle((x, y), w * width, h * height, fill=False,
                                    edgecolor=colors[class_id], linewidth=1.4))
            axis.text(x, y, str(class_id), color="black", fontsize=7,
                      bbox={"facecolor": colors[class_id], "alpha": 0.8, "pad": 1})
        axis.set_title(f"{row['split']} | {row['sample_id']} | {row['width']}x{row['height']}", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(subset):]:
        axis.axis("off")
    figure.suptitle(task + " prepared YOLO labels: " + ", ".join(f"{i}={name}" for i, name in enumerate(names)))
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    return figure
