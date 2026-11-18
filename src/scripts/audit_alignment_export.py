"""Inventory a Roboflow COCO ZIP without extracting or reading test images."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import pandas as pd


TARGETS = {"midline diastema", "crowding"}


def audit(archive, output):
    rows = []
    for split in ("train", "valid"):
        member = f"{split}/_annotations.coco.json"
        data = json.loads(archive.read(member))
        categories = {item["id"]: item["name"].strip().casefold() for item in data["categories"]}
        if not TARGETS <= set(categories.values()):
            raise ValueError(f"Missing expected source categories in {split}: {categories}")
        images = {item["id"]: item for item in data["images"]}
        targets = {image_id: [] for image_id in images}
        for annotation in data["annotations"]:
            if categories[annotation["category_id"]] in TARGETS:
                targets[annotation["image_id"]].append(annotation)
        for image_id, image in images.items():
            name = image["file_name"]
            path = PurePosixPath(name)
            if path.name != name or "\\" in name:
                raise ValueError(f"Expected a flat Roboflow export: {name}")
            counts = dict.fromkeys(TARGETS, 0)
            invalid = 0
            for annotation in targets[image_id]:
                x, y, w, h = annotation["bbox"]
                valid = (all(map(math.isfinite, (x, y, w, h))) and w > 0 and h > 0
                         and 0 <= x <= x+w <= image["width"] and 0 <= y <= y+h <= image["height"])
                if valid:
                    counts[categories[annotation["category_id"]]] += 1
                else:
                    invalid += 1
            rows.append({
                "split": split, "image_id": image_id, "file_name": name,
                "sha256": hashlib.sha256(archive.read(f"{split}/{name}")).hexdigest(),
                "filename_family_candidate": name.split(".rf.", 1)[0],
                "diastema_source_boxes": counts["midline diastema"],
                "crowding_source_boxes": counts["crowding"],
                "invalid_target_boxes": invalid,
            })
    frame = pd.DataFrame(rows)
    for column in ("sha256", "filename_family_candidate"):
        frame[f"{column}_cross_split"] = frame.groupby(column)["split"].transform("nunique") > 1
    frame["frontal_review"] = "pending"
    frame["diastema_box_review"] = "pending"
    frame["anterior_crowding_box_review"] = "pending"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.groupby("split").agg(images=("image_id", "size"),
          diastema_source_boxes=("diastema_source_boxes", "sum"),
          crowding_source_boxes=("crowding_source_boxes", "sum"),
          invalid_target_boxes=("invalid_target_boxes", "sum")).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path to preserve existing review decisions.")
    with zipfile.ZipFile(args.archive) as source:
        audit(source, args.output)
