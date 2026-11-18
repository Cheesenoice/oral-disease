"""Complete the currently authorized smoke -> full -> collection workflow."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/reports/training"
STATE = REPORT / "EXECUTION.json"
os.environ["PYTHONUTF8"] = "1"


def save(state):
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix(".part")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(STATE)
    path = ROOT / "docs/STATUS.md"
    text = path.read_text(encoding="utf-8").split("\n## Live Kaggle execution")[0]
    table = "\n## Live Kaggle execution\n\nUpdated automatically by the active local workflow. Times below are UTC.\n\n| Track | Phase | Version | State |\n|---|---|---:|---|\n"
    for experiment, track in state["tracks"].items():
        table += f"| {experiment} | {track['phase']} | {track['version']} | {track['status']} |\n"
    table += f"\nLast update: {state['updated_utc']}. "
    table += "Next action: inspect collected results." if state["status"] == "complete" else "Next action: the local workflow monitors active runs, collects terminal outputs and submits full runs only after successful smoke checks."
    if state.get("error"):
        table += f"\n\nWorkflow needs attention: `{state['error']}`. No replacement run is automatically submitted."
    path.write_text(text + table + "\n", encoding="utf-8")


def command(script, *arguments):
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "src/scripts" / script),
                    *arguments], cwd=ROOT, check=True)


def kernel_status(api, kernel, attempts=5):
    """Retry temporary Kaggle API outages instead of abandoning remote monitoring."""
    for attempt in range(attempts):
        try:
            return api.kernels_status(kernel).to_dict()["status"]
        except Exception as exc:
            if attempt + 1 == attempts:
                raise
            delay = min(60, 5 * (2 ** attempt))
            print(f"Temporary Kaggle status error for {kernel}: {exc}; retrying in {delay}s", flush=True)
            time.sleep(delay)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.is_file() else {
        "status": "running", "tracks": {e: {"phase": "smoke", "version": 2, "status": "submitted"}
                                         for e in ("E0", "E1")}}
    if state["status"] == "complete":
        return
    api = KaggleApi()
    api.authenticate()
    owner = json.loads((ROOT / "src/configs/kaggle.json").read_text())["owner"]
    started = time.monotonic()
    try:
        while time.monotonic() - started < 14 * 3600:
            for experiment, track in state["tracks"].items():
                if track["status"] == "collected":
                    continue
                slug = "dentalselfie-03-baseline" if experiment == "E0" else "dentalselfie-04-yolo26"
                status = kernel_status(api, f"{owner}/{slug}")
                if status != track["status"]:
                    print(experiment, track["phase"], track["version"], status, flush=True)
                    track["status"] = status
                    save(state)
                if status in {"CANCELLED", "CANCELED", "CANCEL_ACKNOWLEDGED"}:
                    state["status"] = "needs_attention"
                    state["error"] = f"{experiment} {track['phase']} v{track['version']} was cancelled; no full-run results are available."
                    save(state)
                    print(state["error"], flush=True)
                    return
                if status not in {"COMPLETE", "ERROR"}:
                    continue
                command("collect_kaggle_run.py", "--experiment", experiment, "--phase", track["phase"],
                        "--version", str(track["version"]))
                if status != "COMPLETE":
                    raise RuntimeError(f"{experiment} {track['phase']} v{track['version']} ended with {status}")
                if track["phase"] == "smoke":
                    smoke_version = track["version"]
                    command("submit_kaggle_run.py", "--experiment", experiment, "--phase", "full",
                            "--smoke-version", str(smoke_version))
                    request = ApiGetKernelRequest()
                    request.user_name, request.kernel_slug = owner, slug
                    with api.build_kaggle_client() as client:
                        remote = client.kernels.kernels_api_client.get_kernel(request)
                    track.update(phase="full", version=remote.metadata.current_version_number, status="submitted")
                else:
                    receipt = json.loads((REPORT / f"{experiment}_full_v{track['version']}" / "collection_receipt.json").read_text())
                    if receipt["completed_workers"] != 2:
                        raise RuntimeError("Completed notebook lacks two worker result files")
                    track["status"] = "collected"
                save(state)
            if all(t["status"] == "collected" for t in state["tracks"].values()):
                state["status"] = "complete"
                save(state)
                print("Both full model/task tracks collected", flush=True)
                return
            time.sleep(30)
        raise TimeoutError("Local monitoring budget exhausted; remote sessions retain their own timeout")
    except Exception as exc:
        state["status"] = "needs_attention"
        state["error"] = str(exc)
        save(state)
        raise


if __name__ == "__main__":
    main()
