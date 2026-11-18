"""Explicitly submit a private T4 x2 run after input, quota and smoke checks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["E0", "E1"], required=True)
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    parser.add_argument("--smoke-version", type=int, default=1)
    args = parser.parse_args()
    config = json.loads((ROOT / "src/configs/kaggle.json").read_text())
    notebook = "03_yolov8n_baseline" if args.experiment == "E0" else "04_yolo26n_comparison"
    slug = "dentalselfie-03-baseline" if args.experiment == "E0" else "dentalselfie-04-yolo26"
    kernel = f"{config['owner']}/{slug}"
    api = KaggleApi()
    api.authenticate()
    # These kernels are created by this project; do not overwrite an active run.
    status = api.kernels_status(kernel).to_dict()
    if str(status["status"]).upper() in {"RUNNING", "QUEUED"}:
        raise RuntimeError("The target kernel is active; collect it before submitting another version")
    quota = json.loads(subprocess.run(["kaggle", "quota", "--format", "json"],
                                    capture_output=True, text=True, check=True).stdout)
    remaining = float(next(q for q in quota if q["resource"] == "GPU")["remaining"].rstrip("h"))
    estimated_hours = 0.5
    if args.phase == "full":
        smoke = ROOT / "docs/reports/training" / f"{args.experiment}_smoke_v{args.smoke_version}"
        receipt = json.loads((smoke / "collection_receipt.json").read_text())
        if receipt["state"] != "COMPLETE" or receipt["completed_workers"] != 2:
            raise RuntimeError("A successful two-task smoke run must be collected first")
        throughput = pd.read_csv(smoke / "throughput.csv")
        if len(throughput) != 2 or throughput.estimated_100_epochs_hours.isna().any():
            raise RuntimeError("Need measured epoch times for both tasks")
        estimated_hours = float(throughput.estimated_100_epochs_hours.max()) * 1.5 + 0.5
    if remaining < 2 * estimated_hours:
        raise RuntimeError(f"Insufficient quota: {remaining}h left, conservative estimate {2 * estimated_hours:.2f}h")
    command = [sys.executable, str(ROOT / "src/scripts/prepare_kaggle_kernel.py"),
               "--notebook", notebook, "--enable-training"]
    if args.phase == "full":
        command.append("--full")
    subprocess.run(command, check=True)
    stage = ROOT / ".venv/work/kaggle/runs" / notebook
    metadata = json.loads((stage / "kernel-metadata.json").read_text())
    if metadata["is_private"] != config.get("private", True) or metadata["machine_shape"] != "NvidiaTeslaT4":
        raise RuntimeError("Expected configured visibility and T4 x2 kernel")
    for kind in ("data", "code"):
        slug_key = config[kind + "_slug"]
        found = next((d for d in api.dataset_list(mine=True, search=slug_key)
                      if d.ref == f"{config['owner']}/{slug_key}"), None)
        if found is None or bool(found.is_private) != config.get("private", True):
            raise RuntimeError(f"Cannot verify input visibility: {kind}")
    timeout = 3600 if args.phase == "smoke" else 7200
    response = api.kernels_push(str(stage), timeout=str(timeout), acc="NvidiaTeslaT4")
    if response.error or response.invalidDatasetSources:
        raise RuntimeError(f"Submission failed: {response.error}; invalid inputs: {response.invalidDatasetSources}")
    version = response.versionNumber
    report = ROOT / "docs/reports/training" / f"{args.experiment}_{args.phase}_v{version}"
    report.mkdir(parents=True, exist_ok=True)
    record = {"kernel": kernel, "version": version, "url": response.url, "phase": args.phase,
              "submitted_utc": datetime.now(timezone.utc).isoformat(), "quota_before": quota,
              "estimated_wall_hours_with_margin": estimated_hours, "timeout_seconds": timeout,
              "metadata": metadata}
    (report / "submission.json").write_text(json.dumps(record, indent=2))
    (report / "submitted.ipynb").write_bytes((stage / f"{notebook}.ipynb").read_bytes())
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
