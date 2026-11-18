"""Single-GPU worker. Invoke through run_pair on a Kaggle T4 x2 session."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .acquire import digest
from .checkpoints import CheckpointGuard, restore_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["caries", "gum"], required=True)
    parser.add_argument("--experiment", choices=["E0", "E1", "E2"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--resume", type=Path, help="Explicit last.pt from the same run/protocol")
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version
    if not torch.cuda.is_available():
        raise RuntimeError("Training worker requires a Kaggle GPU, not local CPU fallback")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = config["experiments"][args.experiment]
    if not experiment["ready"]:
        raise ValueError(experiment["reason"])
    task_root = args.data_root / args.task
    lock = json.loads((args.data_root / "manifests" / f"{args.task}.lock.json").read_text(encoding="utf-8"))
    splits = args.data_root / "manifests" / f"{args.task}.splits.csv"
    if digest(splits) != lock["splits_sha256"]:
        raise ValueError("Split manifest does not match its lock")
    name = f"{args.experiment}_{args.task}_seed{args.seed}" + ("_smoke" if args.smoke else "")
    run = args.output / name
    if run.exists() and args.resume is None:
        raise FileExistsError(f"Run already exists; use another output directory or explicit resume: {run}")
    args.output.mkdir(parents=True, exist_ok=True)
    source_yaml = yaml.safe_load((task_root / "data.yaml").read_text(encoding="utf-8"))
    source_yaml["path"] = str(task_root.resolve())
    runtime_yaml = args.output / f"{name}.data.yaml"
    runtime_yaml.write_text(yaml.safe_dump(source_yaml), encoding="utf-8")
    kwargs = dict(config["common"])
    kwargs.update(experiment["overrides"])
    kwargs.update(data=str(runtime_yaml), device=0, seed=args.seed,
                  project=str(args.output), name=name, exist_ok=False)
    if args.smoke:
        kwargs.update(epochs=3, patience=3, plots=False)
    # Do not run test here: validation selects checkpoints; final test is a later gate.
    meta = {"task": args.task, "experiment": args.experiment, "seed": args.seed,
            "ultralytics": ultralytics_version, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "split_lock": lock, "config_sha256": digest(args.config), "train_kwargs": kwargs,
            "resume": str(args.resume) if args.resume else None, "smoke_only": args.smoke}
    model = YOLO(str(args.resume) if args.resume else experiment["model"])
    guard = CheckpointGuard(args.max_seconds)
    model.add_callback("on_model_save", guard.save)
    started = time.monotonic()
    if args.resume:
        previous_meta = args.resume.resolve().parent.parent.parent / f"{name}.run.json"
        # Source checkpoint must be accompanied by its original provenance.
        if not previous_meta.is_file():
            raise ValueError("Resume checkpoint has no matching run metadata")
        previous = json.loads(previous_meta.read_text(encoding="utf-8"))
        for key in ("task", "experiment", "seed", "config_sha256"):
            if previous[key] != meta[key]:
                raise ValueError(f"Resume provenance differs: {key}")
        if previous["split_lock"]["splits_sha256"] != lock["splits_sha256"]:
            raise ValueError("Resume split differs")
        (args.output / f"{name}.resume.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if previous["ultralytics"] != ultralytics_version:
            raise ValueError("Resume Ultralytics version differs")
        (args.output / f"{name}.run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        model.add_callback("on_train_start", lambda trainer: restore_state(args.resume, trainer))
        model.train(resume=True, device=0, data=str(runtime_yaml), save_dir=str(run.resolve()),
                    workers=kwargs["workers"], batch=kwargs["batch"])
    else:
        (args.output / f"{name}.run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (args.output / f"{name}.requirements.txt").write_text(freeze.stdout, encoding="utf-8")
        model.train(**kwargs)
    final = {"status": "budget_stopped" if guard.budget_stopped else "completed",
             "smoke_only": args.smoke, "elapsed_seconds": time.monotonic() - started,
             "epochs_completed": model.trainer.epoch + 1,
             "metrics": {key: float(value) for key, value in model.trainer.metrics.items()},
             "resume_checkpoint": str(run / "weights/resume.pt")}
    (args.output / f"{name}.result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
