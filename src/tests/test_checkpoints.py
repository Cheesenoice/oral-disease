"""A bounded session preserves resumable checkpoints and detects altered state."""
import pickle
import sys
import time
from types import SimpleNamespace

import pytest

from dentalselfie.checkpoints import CheckpointGuard, restore_state


def test_guard_preserves_checkpoint_before_stop_and_restores_state(tmp_path, monkeypatch):
    restored = []
    torch = SimpleNamespace(
        save=lambda state, path: path.write_bytes(pickle.dumps(state)),
        load=lambda path, **kwargs: pickle.loads(path.read_bytes()),
        get_rng_state=lambda: b"cpu-rng", set_rng_state=lambda state: restored.append(state),
        cuda=SimpleNamespace(get_rng_state_all=lambda: [b"gpu-rng"],
                             set_rng_state_all=lambda state: restored.extend(state)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    last = tmp_path / "last.pt"
    last.write_bytes(b"unstripped optimizer state")
    trainer = SimpleNamespace(last=last, epoch=3, stop=False,
                              scheduler=SimpleNamespace(state_dict=lambda: {"last_epoch": 3},
                                                        load_state_dict=lambda state: restored.append(state)))
    guard = CheckpointGuard(1)
    guard.deadline = time.monotonic() - 1
    guard.save(trainer)
    assert guard.budget_stopped and trainer.stop
    last.write_bytes(b"stripped final weights")
    checkpoint = tmp_path / "resume.pt"
    assert checkpoint.read_bytes() == b"unstripped optimizer state"
    restore_state(checkpoint, trainer)
    assert restored == [b"cpu-rng", b"gpu-rng", {"last_epoch": 3}]
    checkpoint.write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        restore_state(checkpoint, trainer)


def test_guard_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        CheckpointGuard(0)
