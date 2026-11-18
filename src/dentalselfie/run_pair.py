"""Two bounded worker processes, one GPU per task, with persistent failure logs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["E0", "E1", "E2"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--no-progress-seconds", type=float, default=600,
                         help="Stop both workers if neither task log changes for this long")
    parser.add_argument("--resume-root", type=Path, help="Previous experiments output containing both task runs")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config["experiments"][args.experiment]["ready"]:
        raise ValueError("Experiment is still a research placeholder")
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Select a Kaggle GPU T4 x2 session; expected exactly two visible GPUs")
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)], flush=True)
    for task in ("caries", "gum"):
        if not (args.data_root / task / "data.yaml").is_file():
            raise FileNotFoundError(f"Prepared dataset missing: {task}")
    args.output.mkdir(parents=True, exist_ok=True)
    children = []
    logs = []
    try:
        for gpu, task in enumerate(("caries", "gum")):
            name = f"{args.experiment}_{task}_seed{args.seed}" + ("_smoke" if args.smoke else "")
            log = (args.output / f"{name}.log").open("a" if args.resume_root else "x", encoding="utf-8")
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                       WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
            command = [sys.executable, "-m", "dentalselfie.train", "--task", task,
                       "--experiment", args.experiment, "--config", str(args.config.resolve()),
                       "--data-root", str(args.data_root.resolve()), "--output", str(args.output.resolve()),
                       "--seed", str(args.seed), "--max-seconds", str(args.max_seconds)]
            if args.smoke:
                command.append("--smoke")
            if args.resume_root:
                checkpoint = args.resume_root / name / "weights/resume.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Resume checkpoint missing: {checkpoint}")
                command += ["--resume", str(checkpoint.resolve())]
            children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        last_notice = time.monotonic()
        last_progress = [time.monotonic() for _ in logs]
        log_mtimes = [Path(log.name).stat().st_mtime for log in logs]
        while any(process.poll() is None for process in children):
            failures = [p.returncode for p in children if p.poll() is not None and p.returncode != 0]
            if failures:
                for log in logs:
                    print(Path(log.name).name, Path(log.name).read_text(encoding="utf-8", errors="replace")[-6000:], flush=True)
                raise RuntimeError(f"A GPU worker failed ({failures}); inspect the task logs")
            current_mtimes = [Path(log.name).stat().st_mtime for log in logs]
            now = time.monotonic()
            for index, process in enumerate(children):
                if process.poll() is None and current_mtimes[index] != log_mtimes[index]:
                    last_progress[index] = now
                elif process.poll() is None and now - last_progress[index] >= args.no_progress_seconds:
                    task = ("caries", "gum")[index]
                    raise TimeoutError(f"No {task} worker-log progress for {args.no_progress_seconds:.0f}s; stopping both workers")
            log_mtimes = current_mtimes
            if time.monotonic() - last_notice >= 30:
                for log in logs:
                    text = Path(log.name).read_text(encoding="utf-8", errors="replace")
                    print(Path(log.name).name, text[-1400:], flush=True)
                last_notice = time.monotonic()
            time.sleep(1)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError("GPU worker failed; inspect logs")
        print("Both workers finished. Outputs:", args.output, flush=True)
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
