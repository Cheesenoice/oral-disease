"""Render train/validation examples for each source's alignment-related labels."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".venv/work/alignment_external_audit"
OUT = ROOT / "docs/reports/alignment/external_candidates_2026-11-18"
SUMMARY = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))

FOCUS = {
    "gigikita_discloration_v1": {"crowded": [2]},
    "dental_app_v8": {"diastema": [3], "spacing": [4]},
    "crowded_teeth_fipdz_v1": {"Crowded - v1 2025-10-29 10-35am": [0]},
}


def parse_rows(label_path: Path, class_ids: set[int]):
    objects = []
    if not label_path.exists():
        return objects
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            class_id = int(fields[0])
            coords = [float(x) for x in fields[1:]]
        except ValueError:
            continue
        if class_id not in class_ids or not all(0 <= x <= 1 for x in coords):
            continue
        if len(coords) == 4:
            xc, yc, w, h = coords
            xyxy = (xc-w/2, yc-h/2, xc+w/2, yc+h/2)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs, ys = coords[::2], coords[1::2]
            xyxy = (min(xs), min(ys), max(xs), max(ys))
        else:
            continue
        objects.append((class_id, xyxy))
    return objects


def render(source: str, label: str, class_ids: list[int], split: str, limit: int = 18):
    image_dir = WORK / source / split / "images"
    label_dir = WORK / source / split / "labels"
    candidates = []
    for image_path in image_dir.iterdir():
        if image_path.suffix.casefold() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        objects = parse_rows(label_dir / f"{image_path.stem}.txt", set(class_ids))
        if objects:
            candidates.append((image_path, objects))
    candidates = sorted(candidates, key=lambda item: item[0].name)[:limit]
    if not candidates:
        return None
    cols = 3
    rows = (len(candidates) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (path, objects) in zip(axes.flat, candidates):
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        width, height = image.size
        ax.imshow(image)
        for class_id, (x1, y1, x2, y2) in objects:
            color = "lime"
            ax.add_patch(Rectangle((x1*width, y1*height), (x2-x1)*width, (y2-y1)*height,
                                   fill=False, linewidth=2, edgecolor=color))
            ax.text(x1*width, max(0, y1*height-4), label, color="black", fontsize=9,
                    bbox={"facecolor": color, "alpha": .85, "pad": 2})
        ax.set_title(f"{source} | {split} | source boxes={len(objects)}\n{path.name}", fontsize=9)
    fig.suptitle(f"Source label focus: {label} ({split})", fontsize=15)
    fig.tight_layout()
    output = OUT / f"{source}_{label.lower().replace(' ', '_')}_{split}_focus.jpg"
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    for source, labels in FOCUS.items():
        for label, class_ids in labels.items():
            for split in ("train", "valid"):
                path = render(source, label, class_ids, split)
                print(path if path else f"No examples: {source}/{label}/{split}")
