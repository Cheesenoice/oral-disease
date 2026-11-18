"""Prepare only the pending training notebooks. Never overwrite notebooks with executed outputs."""
from pathlib import Path
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]

BOOTSTRAP = '''from pathlib import Path
import os, shutil, subprocess, sys, zipfile

if Path("/kaggle/input").exists():
    bundles = sorted(Path("/kaggle/input").rglob("project.zip"))
    expanded = [p.parent for p in Path("/kaggle/input").rglob("pyproject.toml")
                if (p.parent / "src/configs/sources.json").is_file()]
    ROOT = Path("/kaggle/working/dentalselfie")
    ROOT.mkdir(exist_ok=True)
    if len(expanded) == 1 and not bundles:
        shutil.copytree(expanded[0], ROOT, dirs_exist_ok=True)
    elif not expanded and len(bundles) == 1:
        with zipfile.ZipFile(bundles[0]) as archive:
            for member in archive.infolist():
                target = (ROOT / member.filename).resolve()
                if not target.is_relative_to(ROOT.resolve()) or ":" in member.filename:
                    raise ValueError("Unsafe code-bundle path")
            archive.extractall(ROOT)
    else:
        raise RuntimeError("Attach exactly one DentalSelfie Code dataset")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(ROOT),
                    "-r", str(ROOT / "requirements-kaggle.txt")], check=True)
else:
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / "src/configs/sources.json").is_file()), None)
    if ROOT is None:
        raise RuntimeError("Open this notebook inside the DentalSelfie project")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
import importlib
importlib.invalidate_caches()
print("Project:", ROOT)
'''

DATA_ROOT = '''if Path("/kaggle/input").exists():
    candidates = [p.parent for p in Path("/kaggle/input").rglob("sources.json")
                  if (p.parent / "caries/data.yaml").is_file()
                  and (p.parent / "gum/data.yaml").is_file()]
    if not candidates:
        archives = sorted(Path("/kaggle/input").rglob("prepared.zip"))
        if len(archives) == 1:
            from dentalselfie.audit import safe_extract
            destination = Path("/kaggle/working/prepared_data")
            if not destination.exists():
                safe_extract(archives[0], destination)
            candidates = [destination]
    if len(candidates) != 1:
        raise RuntimeError("Attach exactly one prepared DentalSelfie Data dataset")
    DATA_ROOT = candidates[0]
else:
    raise RuntimeError("Heavy training runs on Kaggle T4 x2; local data tooling has its own notebooks")
print("Prepared dataset:", DATA_ROOT)
from dentalselfie.kaggle_data import stage_prepared_dataset
# /tmp is excluded from the published /kaggle/working outputs.
DATA_ROOT = stage_prepared_dataset(DATA_ROOT, Path("/tmp/dentalselfie_prepared"), workers=8)
'''


def training(experiment):
    return f'''RUN_TRAINING = False  # Agent switches to True only after prerequisites and smoke run pass.
SMOKE = True  # Three epochs per task measure throughput; excluded from research results.
SEED = 42
MAX_SECONDS = 1800 if SMOKE else 3600  # Hard one-hour worker budget; notebook session has a separate cap.
subprocess.run(["nvidia-smi"], check=True)
if RUN_TRAINING:
    command = [sys.executable, "-m", "dentalselfie.run_pair", "--experiment", "{experiment}",
               "--config", str(ROOT / "src/configs/experiments.json"), "--data-root", str(DATA_ROOT),
               "--output", "/kaggle/working/experiments", "--seed", str(SEED),
               "--max-seconds", str(MAX_SECONDS)]
    if SMOKE:
        command.append("--smoke")
    subprocess.run(command, check=True)
else:
    print("Not trained. This notebook is a prepared entry point, with no fabricated outputs.")
'''


SUMMARY = '''import json
import pandas as pd
from IPython.display import display

output = Path("/kaggle/working/experiments")
rows = []
for path in sorted(output.glob("*.result.json")):
    item = json.loads(path.read_text())
    rows.append({"track": path.stem, "status": item["status"], "smoke_only": item["smoke_only"],
                 "epochs": item["epochs_completed"], "seconds": item["elapsed_seconds"],
                 **item["metrics"]})
if rows:
    summary = pd.DataFrame(rows)
    display(summary)
    summary.to_csv(output / "session_summary.csv", index=False)
    for path in sorted(output.glob("*/results.csv")):
        history = pd.read_csv(path)
        display(history)
        if len(history) > 1:
            steady_epoch = history["time"].diff().dropna().median()
            print(path.parent.name, "median post-first epoch seconds:", round(steady_epoch, 2),
                  "estimated 100-epoch hours:", round(steady_epoch * 100 / 3600, 2))
    display(summary.select_dtypes("number"))
else:
    print("No completed worker results yet; this is not an experiment result.")
'''


def main():
    definitions = [
        ("03_yolov8n_baseline", "E0: reproducible YOLOv8n baseline on two GPUs", [
            ("code", DATA_ROOT), ("code", training("E0")),
            ("markdown", "Exit gate: smoke logs first, then full validation metrics/checkpoints. Do not tune or select checkpoints using test.")]),
        ("04_yolo26n_comparison", "E1: modern detector under the same protocol", [
            ("code", DATA_ROOT), ("code", training("E1")),
            ("markdown", "Exit gate: compare against E0 using identical splits, input size and declared training budget. Do not imply that stock YOLO26 reproduces custom dental paper modules.")]),
    ]
    directory = ROOT / "notebooks"
    directory.mkdir(exist_ok=True)
    for name, title, cells in definitions:
        target = directory / f"{name}.ipynb"
        if target.exists():
            previous = nbf.read(target, as_version=4)
            if any(c.get("outputs") or c.get("execution_count") is not None
                   for c in previous.cells if c.cell_type == "code"):
                raise RuntimeError(f"Refusing to overwrite executed evidence: {target}")
        notebook = nbf.v4.new_notebook()
        notebook.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
                                 language_info={"name": "python"})
        notebook.cells = [nbf.v4.new_markdown_cell(f"# {name}\n\n{title}\n\nStatus: prepared; results require execution."),
                          nbf.v4.new_code_cell(BOOTSTRAP)]
        notebook.cells.extend(nbf.v4.new_code_cell(body) if kind == "code" else nbf.v4.new_markdown_cell(body)
                              for kind, body in cells)
        notebook.cells += [nbf.v4.new_markdown_cell("## Execution evidence and throughput\n\nSmoke runs use three epochs at the configured batch size, AMP and workers. They are excluded from final comparisons. Full runs use validation for checkpoint selection; test is never evaluated here. Save the whole experiments output to retain logs, configuration, dependency freeze, curves, metrics and resumable checkpoints."),
                           nbf.v4.new_code_cell(SUMMARY),
                           nbf.v4.new_markdown_cell("## Checkpoint and session protocol\n\nOne task runs on each T4. Screening is capped at 30 epochs with patience 8 and a one-hour per-worker wall-clock budget; a no-progress watchdog stops both workers if their logs remain unchanged for 10 minutes. Kaggle's session cap is two hours, including setup and output publication. After every epoch, resume.pt retains optimizer, EMA and scaler state before finalization can strip last.pt. Scheduler and process RNG states are recorded; data-loader worker/prefetch states are not serialized, so resume is not bitwise identical. This bounded screening is not a final convergence claim. Extend from preserved checkpoints only after reviewing the learning curves."),
                           nbf.v4.new_code_cell('import inspect\nfrom IPython.display import Code, display\nfrom dentalselfie.checkpoints import CheckpointGuard\nimport dentalselfie.train as worker\ndisplay(Code(inspect.getsource(worker), language="python"))\ndisplay(Code(inspect.getsource(CheckpointGuard), language="python"))')]
        nbf.validate(notebook)
        nbf.write(notebook, target)
        print(target.name)


if __name__ == "__main__":
    main()
