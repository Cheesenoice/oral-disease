"""Private Kaggle publication with artifact receipts and explicit remote verification."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from .acquire import digest
from .research import write_json


def publication_ready(status: dict, expected_version: int) -> bool:
    """An old ready version is not evidence that the newly uploaded version is ready."""
    version = int(status.get("current_version_number", 0))
    if version > expected_version:
        raise RuntimeError("Remote dataset advanced beyond this upload; verify concurrent publication")
    return status.get("status", "").lower() == "ready" and version == expected_version


def verify_remote_data(root: Path) -> dict:
    """Compare all uploaded file names and sizes with the verified local ZIP."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    import pandas as pd

    config = json.loads((root / "src/configs/kaggle.json").read_text())
    dataset = f"{config['owner']}/{config['data_slug']}"
    api = KaggleApi()
    api.authenticate()
    remote = {}
    token = None
    while True:
        page = api.dataset_list_files(dataset, page_token=token, page_size=200)
        remote.update({item.name: item.total_bytes for item in page.files})
        token = page.next_page_token
        if not token:
            break
    markers = [name for name in remote if name.endswith("caries/data.yaml")]
    if len(markers) != 1:
        raise RuntimeError("Cannot locate one expanded prepared dataset on Kaggle")
    prefix = markers[0][:-len("caries/data.yaml")]
    archive_path = root / ".venv/work/kaggle/staging/data/prepared.zip"
    with zipfile.ZipFile(archive_path) as archive:
        rows = [{"file": prefix + item.filename, "local_bytes": item.file_size,
                 "remote_bytes": remote.get(prefix + item.filename),
                 "matches": remote.get(prefix + item.filename) == item.file_size}
                for item in archive.infolist() if not item.is_dir()]
    report = root / "docs/reports/data_preparation"
    pd.DataFrame(rows).to_csv(report / "kaggle_remote_inventory.csv", index=False)
    result = {"dataset": dataset, "checked_files": len(rows), "remote_files": len(remote),
              "expanded_prefix": prefix, "mismatches": sum(not row["matches"] for row in rows),
              "verification": "Remote filenames and byte sizes; local ZIP separately checked by CRC and SHA-256"}
    write_json(report / "kaggle_remote_verification.json", result)
    if result["mismatches"]:
        raise RuntimeError("Remote inventory differs from the prepared archive; inspect the CSV report")
    return result


def publish_stage(root: Path, kind: str, stage: Path) -> dict:
    """Create/version an owned dataset under the explicitly configured visibility."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    if kind not in ("data", "code", "roi", "raw_roi", "results"):
        raise ValueError("Unknown publication kind")
    config = json.loads((root / "src/configs/kaggle.json").read_text())
    stage = stage.resolve()
    expected = (root / ".venv/work/kaggle/staging" / kind).resolve()
    if stage != expected or not stage.is_relative_to(root.resolve()):
        raise ValueError("Unexpected upload folder")
    metadata = json.loads((stage / "dataset-metadata.json").read_text())
    dataset = f"{config['owner']}/{config[kind + '_slug']}"
    if metadata["id"] != dataset:
        raise ValueError("Upload metadata points to a different account/dataset")
    artifact = stage / {"data": "prepared.zip", "code": "project.zip", "roi": "roi.zip", "raw_roi": "raw_roi.zip", "results": "results.zip"}[kind]
    artifact_sha = digest(artifact)
    api = KaggleApi()
    api.authenticate()  # Uses the existing local credential configuration; never print it.
    existing = next((item for item in api.dataset_list(mine=True, search=config[kind + "_slug"])
                     if item.ref == dataset), None)
    expected_private = bool(config.get("private", True))
    if existing is not None and bool(existing.is_private) != expected_private:
        raise RuntimeError("Existing visibility differs from project policy; update and verify it explicitly first")
    receipt_path = root / "docs/reports/data_preparation/kaggle_upload_receipt.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    previous = receipts.get(kind)
    if (existing is not None and previous and previous["artifact_sha256"] == artifact_sha
            and previous["version"] == existing.current_version_number):
        previous["private"] = expected_private
        receipts[kind] = previous
        write_json(receipt_path, receipts)
        print(f"{kind}: identical dataset version already uploaded; visibility verified", flush=True)
        return previous
    cli = shutil.which("kaggle")
    if cli is None:
        raise RuntimeError("Kaggle CLI is unavailable")
    mode = "version" if existing is not None else "create"
    expected_version = int(existing.current_version_number) + 1 if existing is not None else 1
    command = [cli, "datasets", mode, "-p", str(stage), "-q"]
    if mode == "version":
        command += ["-m", "Verified DentalSelfie notebook workflow and artifacts"]
    elif not expected_private:
        command.append("--public")
    log_dir = root / ".venv/work/kaggle/upload_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{kind}_{int(time.time())}.log"
    print(f"Uploading {kind}: {artifact.stat().st_size / 1_000_000:.1f} MB to {dataset} ({'private' if expected_private else 'public'})", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        last_notice = time.monotonic()
        while process.poll() is None:
            time.sleep(2)
            if time.monotonic() - last_notice >= 30:
                print(f"Upload still running: {time.monotonic() - started:.0f}s; log {log_path.name}", flush=True)
                last_notice = time.monotonic()
    result_text = log_path.read_text(encoding="utf-8", errors="replace")
    print(result_text[-3500:], flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"Kaggle upload failed; inspect {log_path}")
    for attempt in range(40):
        result = subprocess.run([cli, "datasets", "status", dataset, "--format", "json"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            status = json.loads(result.stdout)
            if publication_ready(status, expected_version):
                break
            if status.get("status", "").lower() in {"error", "failed"}:
                raise RuntimeError(f"Kaggle processing failed: {status}")
        print(f"Waiting for Kaggle to finish processing {kind} ...", flush=True)
        time.sleep(10)
    else:
        raise TimeoutError("Upload completed but the remote dataset has not become ready yet")
    # Newly created datasets may be ready before Kaggle's search index lists them.
    # Verify exact metadata rather than treating search visibility as publication state.
    from kaggle.api.kaggle_api_extended import ApiGetDatasetMetadataRequest
    verify_request = ApiGetDatasetMetadataRequest()
    verify_request.owner_slug, verify_request.dataset_slug = dataset.split("/", 1)
    with api.build_kaggle_client() as client:
        verified = client.datasets.dataset_api_client.get_dataset_metadata(verify_request)
    if (verified.error_message or bool(verified.info.is_private) != expected_private
            or verified.info.owner_user != config["owner"]
            or verified.info.dataset_slug != config[kind + "_slug"]):
        raise RuntimeError("Could not verify exact dataset identity and configured visibility")
    receipt = {"dataset": dataset, "url": f"https://www.kaggle.com/datasets/{dataset}",
               "version": status["current_version_number"], "private": expected_private, "status": "ready",
               "artifact_sha256": artifact_sha, "artifact_bytes": artifact.stat().st_size,
               "mode": mode, "log": log_path.relative_to(root).as_posix()}
    receipts[kind] = receipt
    write_json(receipt_path, receipts)
    return receipt
