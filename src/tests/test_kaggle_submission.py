"""GPU submission guards must reject active kernels and failed smoke notebooks."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load_submitter():
    path = Path(__file__).resolve().parents[1] / "scripts/submit_kaggle_run.py"
    spec = importlib.util.spec_from_file_location("submission_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("active", [True, False])
def test_no_duplicate_or_failed_smoke_submission(tmp_path, monkeypatch, active):
    module = load_submitter()
    config = tmp_path / "src/configs/kaggle.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"owner": "fixture"}))
    smoke = tmp_path / "docs/reports/training/E0_smoke_v2"
    smoke.mkdir(parents=True)
    (smoke / "collection_receipt.json").write_text(json.dumps({"state": "ERROR", "completed_workers": 2}))

    class FakeAPI:
        def authenticate(self):
            pass

        def kernels_status(self, kernel):
            return SimpleNamespace(to_dict=lambda: {"status": "RUNNING" if active else "COMPLETE"})

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "KaggleApi", FakeAPI)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps([{"resource": "GPU", "remaining": "30.00h"}])))
    monkeypatch.setattr(sys, "argv", ["submit", "--experiment", "E0", "--phase", "full", "--smoke-version", "2"])
    with pytest.raises(RuntimeError, match="active" if active else "successful two-task smoke"):
        module.main()
