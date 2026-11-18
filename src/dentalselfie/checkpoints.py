"""Epoch-boundary recovery and a wall-clock guard for bounded Kaggle sessions."""
from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import time

from .acquire import digest


class CheckpointGuard:
    """Keep an unstripped checkpoint before Ultralytics final evaluation strips last.pt."""

    def __init__(self, max_seconds: float):
        if max_seconds <= 0:
            raise ValueError("Training time budget must be positive")
        self.deadline = time.monotonic() + max_seconds
        self.started = time.monotonic()
        self.budget_stopped = False

    def save(self, trainer):
        import numpy as np
        import torch

        directory = Path(trainer.last).parent
        temporary = directory / "resume.pt.part"
        shutil.copyfile(trainer.last, temporary)
        temporary.replace(directory / "resume.pt")
        state = {"epoch": trainer.epoch, "python": random.getstate(), "numpy": np.random.get_state(),
                 "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                 "scheduler": trainer.scheduler.state_dict()}
        state_path = directory / "resume_state.pt"
        torch.save(state, state_path.with_suffix(".part"))
        state_path.with_suffix(".part").replace(state_path)
        receipt = {"epoch": trainer.epoch, "checkpoint_sha256": digest(directory / "resume.pt"),
                   "state_sha256": digest(state_path), "elapsed_seconds": time.monotonic() - self.started,
                   "rng_limit": "Worker/prefetch RNG state is not serialized; resumption is not bitwise identical"}
        (directory / "resume_receipt.json").write_text(json.dumps(receipt, indent=2))
        if time.monotonic() >= self.deadline:
            self.budget_stopped = True
            trainer.stop = True
            print("Session budget reached; resumable epoch checkpoint saved", flush=True)


def restore_state(checkpoint: Path, trainer):
    """Restore only our verified, self-generated training state; never untrusted input."""
    import numpy as np
    import torch

    receipt = json.loads(checkpoint.with_name("resume_receipt.json").read_text())
    state_path = checkpoint.with_name("resume_state.pt")
    if digest(checkpoint) != receipt["checkpoint_sha256"] or digest(state_path) != receipt["state_sha256"]:
        raise ValueError("Resume checkpoint/state checksum mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    trainer.scheduler.load_state_dict(state["scheduler"])
