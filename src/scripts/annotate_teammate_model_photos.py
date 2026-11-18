"""Run the teammate's mixed ResNet50 INT8 classifier and label local photos.

Example:
    python src/scripts/annotate_teammate_model_photos.py photo1.jpg photo2.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEAM_REPOSITORY = Path(r"C:\Users\huynh\Desktop\oral-disease-ai")
MODEL_DIR = TEAM_REPOSITORY / "mobile" / "assets" / "models" / "resnet50_mixed"
sys.path.insert(0, str(TEAM_REPOSITORY / "inference"))
from predict_onnx import OralClassifier  # noqa: E402


def make_labeled_image(path: Path, label: str, score: float, output: Path) -> None:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", max(18, image.width // 30))
    text = f"{label} | {score:.1%} | whole-image class"
    bounds = draw.textbbox((0, 0), text, font=font)
    pad = 12
    banner_height = bounds[3] - bounds[1] + 2 * pad
    draw.rectangle((0, 0, image.width, banner_height), fill=(15, 23, 42))
    draw.text((pad, pad - bounds[1]), text, fill=(255, 255, 255), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    output_dir = PROJECT_ROOT / "data" / "pilot" / "phone_feasibility" / "teammate_classifier_trial"
    model = OralClassifier(MODEL_DIR, int8=True)
    probabilities = model.predict(args.images)
    results = []
    for path, scores in zip(args.images, probabilities):
        index = int(np.argmax(scores))
        score = float(scores[index])
        image_output = output_dir / f"{path.stem}_classified.jpg"
        make_labeled_image(path, model.classes[index], score, image_output)
        results.append({"file": path.name, "top1": model.classes[index], "score": score,
                        "below_app_threshold": score < 0.5, "labeled_image": image_output.name})
    (output_dir / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
