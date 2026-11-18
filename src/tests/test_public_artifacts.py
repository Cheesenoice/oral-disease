import json
import zipfile
import pytest
from dentalselfie.public_artifacts import package_research, scan_text
from dentalselfie.roi import prepare_rois
from test_roi import fixture


def setup_roi(root):
    dataset, manifests, config = fixture(root)
    configs = root / "src/configs";configs.mkdir(parents=True)
    (configs / "roi.json").write_bytes(config.read_bytes())
    (configs / "kaggle.json").write_text(json.dumps({"private":False,"owner":"test","roi_slug":"roi"}))
    output = root / "data/processed/gum_roi_v1"
    prepare_rois(dataset, manifests, output, config, workers=1)
    return output


def test_allowlisted_roi_export_and_crc(tmp_path):
    setup_roi(tmp_path)
    stage = package_research(tmp_path, "roi")
    before = (stage / "roi.zip").read_bytes()
    assert package_research(tmp_path, "roi") == stage
    assert (stage / "roi.zip").read_bytes() == before
    with zipfile.ZipFile(stage / "roi.zip") as z:
        assert z.testzip() is None
        assert "content_manifest.json" in z.namelist()
        assert not any("test/" in name or "pilot/" in name for name in z.namelist())
        assert len(json.loads(z.read("content_manifest.json"))) == 8


def test_changed_roi_cannot_be_published(tmp_path):
    output = setup_roi(tmp_path)
    next((output / "train/local").glob("*.jpg")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        package_research(tmp_path, "roi")


def test_credential_header_stops_export_without_exposing_contents(tmp_path):
    p = tmp_path / "candidate.txt"
    p.write_bytes(b"-" * 5 + b"BEGIN PRIVATE KEY" + b"-" * 5)
    with pytest.raises(ValueError, match="publication stopped"):
        scan_text(p)
