"""Stage a private kernel locally. Does NOT upload or start a Kaggle session."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", choices=["03_yolov8n_baseline", "04_yolo26n_comparison"], required=True)
    parser.add_argument("--enable-training", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.full and not args.enable_training:
        parser.error("--full requires --enable-training")
    config = json.loads((ROOT / "src/configs/kaggle.json").read_text())
    receipts = json.loads((ROOT / "docs/reports/data_preparation/kaggle_upload_receipt.json").read_text())
    if not all(receipts[k]["private"] == config.get("private", True) and receipts[k]["status"] == "ready" for k in ("code", "data")):
        raise RuntimeError("Both inputs must match configured visibility and be verified ready")
    notebook = nbformat.read(ROOT / "notebooks" / f"{args.notebook}.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            if args.enable_training:
                cell.source = cell.source.replace("RUN_TRAINING = False", "RUN_TRAINING = True")
            if args.full:
                cell.source = cell.source.replace("SMOKE = True", "SMOKE = False")
    target = ROOT / ".venv/work/kaggle/runs" / args.notebook
    target.mkdir(parents=True, exist_ok=True)
    slug = "dentalselfie-03-baseline" if args.notebook.startswith("03") else "dentalselfie-04-yolo26"
    nbformat.write(notebook, target / f"{args.notebook}.ipynb")
    metadata = {
        "id": f"{config['owner']}/{slug}", "title": slug.replace("-", " ").title(),
        "code_file": f"{args.notebook}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": config.get("private", True), "enable_gpu": True, "enable_internet": True,
        "machine_shape": config["accelerator"],
        "dataset_sources": [f"{receipts[k]['dataset']}/{receipts[k]['version']}" for k in ("code", "data")],
        "competition_sources": [], "kernel_sources": []
    }
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Staged locally:", target)
    print("Training enabled:", args.enable_training, "Full run:", args.full)
    print("No upload or GPU run has been started.")


if __name__ == "__main__":
    main()
