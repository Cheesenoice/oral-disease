"""Collect one finished version, including raw notebook evidence and training outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["E0", "E1"], required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    if args.version < 1:
        parser.error("Version must be positive")
    owner = json.loads((ROOT / "src/configs/kaggle.json").read_text())["owner"]
    slug = "dentalselfie-03-baseline" if args.experiment == "E0" else "dentalselfie-04-yolo26"
    kernel = f"{owner}/{slug}/{args.version}"
    name = f"{args.experiment}_{args.phase}_v{args.version}"
    report = ROOT / "docs/reports/training" / name
    report.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = owner, slug
    with api.build_kaggle_client() as client:
        remote = client.kernels.kernels_api_client.get_kernel(request)
    # Current CLI status/output APIs ignore a /version suffix; enforce the version ourselves.
    if remote.metadata.current_version_number != args.version:
        raise RuntimeError("Latest kernel version differs; refuse to mislabel another run's outputs")
    status = api.kernels_status(kernel).to_dict()
    state = str(status["status"])
    (report / "status.json").write_text(json.dumps({"status": state,
                                                  "failure_message": status.get("failureMessage")}, indent=2))
    print(kernel, state)
    if state.upper() not in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED"}:
        return
    destination = ROOT / "data/runs" / name
    destination.mkdir(parents=True, exist_ok=True)
    token = None
    while True:
        files, token = api.kernels_output(
            kernel, str(destination), force=False, quiet=True, page_size=200, page_token=token,
            file_pattern=r"(^|/)(experiments/|__notebook__|.*\.log$)")
        print("Downloaded page:", len(files), "files")
        if not token:
            break
    api.kernels_pull(f"{owner}/{slug}", str(report), metadata=True)
    (report / "execution.log").write_text(api.kernels_logs(f"{owner}/{slug}"), encoding="utf-8")
    rows = []
    for path in destination.rglob("*.result.json"):
        item = json.loads(path.read_text())
        rows.append({"track": path.name.removesuffix(".result.json"), "phase": args.phase,
                     "status": item["status"], "epochs": item["epochs_completed"],
                     "seconds": item["elapsed_seconds"], **item["metrics"]})
    pd.DataFrame(rows).to_csv(report / "summary.csv", index=False)
    estimates = []
    for path in destination.rglob("results.csv"):
        history = pd.read_csv(path)
        history.columns = history.columns.str.strip()
        seconds = history["time"].diff().dropna().median() if len(history) > 1 else None
        estimates.append({"track": path.parent.name, "epochs": len(history),
                          "steady_epoch_seconds": seconds,
                          "estimated_100_epochs_hours": seconds * 100 / 3600 if seconds is not None else None})
    pd.DataFrame(estimates).to_csv(report / "throughput.csv", index=False)
    receipt = {"kernel": kernel, "state": state, "collected_utc": datetime.now(timezone.utc).isoformat(),
               "completed_workers": len(rows), "outputs": str(destination.relative_to(ROOT)),
               "phase": args.phase, "note": "Smoke metrics are not final experiment results"}
    (report / "collection_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(estimates).to_string(index=False))


if __name__ == "__main__":
    main()
