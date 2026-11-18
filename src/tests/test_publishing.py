"""Publication checks use synthetic remote inventories, never real uploads."""
import json
from types import SimpleNamespace
import zipfile

import pytest

from dentalselfie.publishing import verify_remote_data


@pytest.mark.parametrize("wrong_size", [False, True])
def test_remote_inventory_pagination_and_mismatch(tmp_path, monkeypatch, wrong_size):
    from kaggle.api import kaggle_api_extended

    config = tmp_path / "src/configs/kaggle.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"owner": "fixture", "data_slug": "synthetic"}))
    stage = tmp_path / ".venv/work/kaggle/staging/data"
    stage.mkdir(parents=True)
    with zipfile.ZipFile(stage / "prepared.zip", "w") as archive:
        archive.writestr("caries/data.yaml", "abc")
        archive.writestr("gum/data.yaml", "defg")
    (tmp_path / "docs/reports/data_preparation").mkdir(parents=True)
    calls = []

    class FakeAPI:
        def authenticate(self):
            pass

        def dataset_list_files(self, dataset, page_token, page_size):
            calls.append(page_token)
            assert dataset == "fixture/synthetic"
            if page_token is None:
                return SimpleNamespace(files=[SimpleNamespace(name="prepared/caries/data.yaml", total_bytes=3)],
                                       next_page_token="second")
            return SimpleNamespace(files=[SimpleNamespace(name="prepared/gum/data.yaml",
                                                           total_bytes=5 if wrong_size else 4)],
                                   next_page_token=None)

    monkeypatch.setattr(kaggle_api_extended, "KaggleApi", FakeAPI)
    if wrong_size:
        with pytest.raises(RuntimeError, match="Remote inventory differs"):
            verify_remote_data(tmp_path)
    else:
        result = verify_remote_data(tmp_path)
        assert result["checked_files"] == 2 and result["mismatches"] == 0
    assert calls == [None, "second"]
    assert (tmp_path / "docs/reports/data_preparation/kaggle_remote_inventory.csv").is_file()
