"""Stage or publish the two bounded DINOv3 comparison notebooks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from datetime import datetime, timezone

import nbformat

ROOT = Path(__file__).resolve().parents[2]
TRACKS = {
    "11.1": ("11_1_dinov3_patch_probe", "dentalselfie-11-1-dinov3-patch-probe"),
    "11.2": ("11_2_dinov3_partial_adaptation", "dentalselfie-11-2-dinov3-partial-adaptation"),
}


def build_quick_save_request(notebook, metadata, timeout):
    """Build a non-executing request; this path must never reserve GPU time."""
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

    request = ApiSaveKernelRequest()
    request.slug = metadata["id"]
    request.new_title = metadata["title"]
    request.text = json.dumps(notebook)
    request.language = metadata["language"]
    request.kernel_type = metadata["kernel_type"]
    request.is_private = metadata["is_private"]
    request.enable_gpu = metadata["enable_gpu"]
    request.enable_internet = metadata["enable_internet"]
    request.machine_shape = metadata["machine_shape"]
    request.dataset_data_sources = metadata["dataset_sources"]
    request.kernel_data_sources = metadata["kernel_sources"]
    request.competition_data_sources = metadata["competition_sources"]
    request.session_timeout_seconds = timeout
    request.kernel_execution_type = KernelExecutionType.QUICK_SAVE
    return request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--phase", choices=("source", "smoke", "full"), default="source")
    parser.add_argument("--publish", action="store_true", help="QUICK_SAVE the source; never start a GPU session from this command")
    args = parser.parse_args()

    config = json.loads((ROOT / "src/configs/kaggle.json").read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / "docs/reports/data_preparation/kaggle_upload_receipt.json").read_text(encoding="utf-8"))
    raw = receipt["raw_roi"]
    if raw["status"] != "ready" or raw["private"] is not False:
        raise RuntimeError("Pinned public raw ROI input is not verified ready")
    name, slug = TRACKS[args.track]
    notebook = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    nbformat.validate(notebook)
    found_enable = found_phase = False
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        cell.execution_count = None
        if re.search(r"^RUN_EXPERIMENT\s*=", cell.source, re.M):
            cell.source = re.sub(r"^RUN_EXPERIMENT\s*=\s*False", "RUN_EXPERIMENT = " + str(args.phase != "source"), cell.source, count=1, flags=re.M)
            found_enable = True
        if re.search(r"^PHASE\s*=", cell.source, re.M):
            cell.source = re.sub(r"^PHASE\s*=\s*['\"](?:smoke|full)['\"]", f"PHASE = {args.phase if args.phase != 'source' else 'smoke'!r}", cell.source, count=1, flags=re.M)
            found_phase = True
    if not (found_enable and found_phase):
        raise RuntimeError("Could not update the experiment gate and phase")

    stage = ROOT / ".venv/work/kaggle/dinov3" / slug
    stage.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, stage / f"{name}.ipynb")
    metadata = {
        "id": f"{config['owner']}/{slug}",
        "title": slug.replace("-", " ").title(),
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": config["accelerator"],
        "dataset_sources": [f"{raw['dataset']}/{raw['version']}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Staged {args.track} {args.phase}: {stage}")
    if not args.publish:
        return

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    mine = api.kernels_list(mine=True, page_size=100) or []
    owned = any(getattr(item, "ref", None) == metadata["id"] for item in mine if item is not None)
    if owned:
        status = str(api.kernels_status(metadata["id"]).status).upper().split(".")[-1]
    else:
        status = "NOT_FOUND"
    if status in {"RUNNING", "QUEUED"}:
        raise RuntimeError(f"Refusing to replace active kernel {metadata['id']} ({status})")
    timeout = 7200 if args.phase == "full" else 3600
    # kernels_push defaults to a new execution and cannot carry Kaggle Secrets.
    # Publish without execution, then launch from the authenticated Kaggle editor.
    request = build_quick_save_request(notebook, metadata, timeout)
    with api.build_kaggle_client() as client:
        result = client.kernels.kernels_api_client.save_kernel(request)
    if result.error or result.invalidDatasetSources:
        raise RuntimeError(f"Kaggle push failed: {result.error}; invalid inputs={result.invalidDatasetSources}")
    record = {
        "track": args.track,
        "phase": args.phase,
        "publication_mode": "QUICK_SAVE",
        "execution_requested": False,
        "kernel": metadata["id"],
        "url": result.url,
        "version": result.versionNumber,
        "status_before_push": status,
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
    }
    report = ROOT / "docs/reports/training" / "dinov3_publication"
    report.mkdir(parents=True, exist_ok=True)
    (report / f"{args.track.replace('.', '_')}_{args.phase}_v{result.versionNumber}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    print("Source saved only. Launch from Kaggle's editor after attaching HF_TOKEN; do not publish again between attachment and execution.")


if __name__ == "__main__":
    main()
