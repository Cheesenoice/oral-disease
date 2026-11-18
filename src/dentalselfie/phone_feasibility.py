"""Local-only, unlabeled capture inspection and existing-weight inference."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIEWS = {"front_teeth", "open_mouth", "left_accessible", "right_accessible", "upper_accessible", "lower_accessible"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_photos(inbox, metadata, known_hashes=()):
    """No recursive file discovery; require local consent/view records for inference.

    Descriptive brightness and sharpness are not validated capture-quality gates.
    Source-byte overlap is a limited leakage check, not subject independence.
    """
    inbox = Path(inbox).resolve()
    records = pd.read_csv(metadata, keep_default_na=False, dtype=str)
    if not {"file", "view", "consent"} <= set(records.columns):
        raise ValueError("capture_manifest.csv needs file, view and consent columns")
    if records.file.duplicated().any():
        raise ValueError("Duplicate filenames in capture manifest")
    files = sorted(p for p in inbox.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())
    manifest_rows = records.set_index("file").to_dict("index")
    rows, seen_bytes, seen_pixels = [], {}, {}
    for path in files:
        row = dict(file=path.name, status="pending", reason="", view="", consent="")
        try:
            if not path.resolve().is_relative_to(inbox):
                raise ValueError("Path escapes local photo inbox")
            annotation = manifest_rows.get(path.name, {})
            row.update(annotation)
            row["sha256"] = sha(path)
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                row.update(width=rgb.width, height=rgb.height)
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                gray = np.asarray(rgb.convert("L").resize((256, 256)), dtype=float)
                laplacian = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
                row.update(mean_luma=float(gray.mean()), dark_fraction=float((gray < 20).mean()),
                           bright_fraction=float((gray > 235).mean()), laplacian_variance_256=float(laplacian.var()))
            reasons = []
            if row["sha256"] in known_hashes:
                reasons.append("exact_bytes_overlap_research_data")
            if row["sha256"] in seen_bytes or pixel_hash in seen_pixels:
                reasons.append("duplicate_capture")
            seen_bytes[row["sha256"]] = path.name
            seen_pixels[pixel_hash] = path.name
            if annotation.get("consent", "").lower() != "yes":
                reasons.append("consent_not_recorded")
            if annotation.get("view", "") not in VIEWS:
                reasons.append("view_not_recorded")
            row.update(status="ready" if not reasons else "pending", reason=";".join(reasons))
        except (OSError, ValueError) as error:
            row.update(status="unreadable_or_invalid", reason=str(error))
        rows.append(row)
    columns = ["file", "status", "reason", "view", "consent", "sha256", "width", "height",
               "mean_luma", "dark_fraction", "bright_fraction", "laplacian_variance_256"]
    return pd.DataFrame(rows, columns=columns)


def infer_images(root, image_paths, output, protocol):
    """Retain one row and annotated image for every input, including zero boxes."""
    import os
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    import ultralytics
    from ultralytics import YOLO
    assert ultralytics.__version__ == "8.4.183", "Reuse the verified local inference runtime"
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not image_paths:
        return pd.DataFrame(), pd.DataFrame()
    rows, boxes = [], []
    weights = {
        "YOLOv8n": root / "data/runs/E0_full_v4/experiments/E0_caries_seed42/weights/best.pt",
        "YOLO26n": root / "data/runs/E1_full_v4/experiments/E1_caries_seed42/weights/best.pt",
    }
    for name, weight in weights.items():
        receipt = json.loads((root / f"docs/reports/post09/caries/{name}_clean/audit_receipt.json").read_text())
        assert sha(weight) == receipt["identity"]["weight_sha256"], "Checkpoint changed"
        model = YOLO(str(weight))
        assert len(model.names) == 1
        # Warm-up is excluded from latency. Actual input decode is excluded too.
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device="cpu", imgsz=640, verbose=False)
        for index, path in enumerate(image_paths):
            path = Path(path)
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
            started = time.perf_counter()
            result = model.predict(rgb, device="cpu", imgsz=protocol["imgsz"], conf=protocol["conf"],
                                   iou=protocol["iou"], max_det=protocol["max_det"],
                                   augment=False, half=False, verbose=False)[0]
            elapsed_ms = (time.perf_counter() - started) * 1000
            annotated = f"{name}_{index:03d}.jpg"
            # plot() returns BGR. Re-encode RGB without the original EXIF metadata.
            Image.fromarray(result.plot()[..., ::-1]).save(output / annotated)
            count = len(result.boxes)
            rows.append(dict(model=name, file=path.name, sha256=sha(path), detections=count,
                             elapsed_ms=elapsed_ms, annotated=annotated, weight_sha256=sha(weight)))
            for box in result.boxes:
                assert int(box.cls.item()) == 0
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                boxes.append(dict(model=name, file=path.name, class_id=0, confidence=float(box.conf.item()),
                                  x1=x1, y1=y1, x2=x2, y2=y2))
        print(name, "completed", len(image_paths), "images", flush=True)
    images = pd.DataFrame(rows)
    detections = pd.DataFrame(boxes, columns=["model", "file", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    images.to_csv(output / "per_image.csv", index=False)
    detections.to_csv(output / "detections.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(dict(settings=protocol, torch=torch.__version__,
        ultralytics=ultralytics.__version__, device="desktop_cpu", metrics_status="unlabeled_no_accuracy",
        zero_detections_mean="No candidate at this threshold; not a healthy diagnosis"), indent=2), encoding="utf-8")
    return images, detections
