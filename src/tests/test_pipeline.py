import csv
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from dentalselfie.acquire import digest
from dentalselfie.audit import safe_extract
from dentalselfie.data import assign_splits, convert_index, parse_yolo, prepare_splits
from dentalselfie.package import package_code, package_data


@pytest.mark.parametrize("line", ["0 0.5 1.7 0.2 0.3", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 -0.2 0.2",
                                  "0 0.5 0.5 0.2", "0 0.95 0.5 0.2 0.2", "-1 0.5 0.5 0.2 0.2"])
def test_invalid_box_rejected(line):
    with pytest.raises(ValueError):
        parse_yolo(line)


def test_explicit_empty_annotation_is_allowed():
    assert parse_yolo("") == []
    assert parse_yolo("0 0.5 0.5 0.2 0.3")[0][0] == 0


def test_connected_groups_and_duplicates_never_cross_splits():
    rows = [{"sample_id": str(i), "group_id": f"p{i // 2}", "sha256": f"hash{i}"} for i in range(30)]
    rows[2]["sha256"] = rows[0]["sha256"]  # Merge two existing patient groups transitively.
    result = assign_splits(rows)
    assert result == assign_splits(rows)
    assert len({result[i]["split"] for i in range(4)}) == 1
    assert {row["split"] for row in result} == {"train", "val", "test"}
    for field in ("sha256", "group_id"):
        for value in {row[field] for row in result}:
            assert len({row["split"] for row in result if row[field] == value}) == 1


@pytest.mark.parametrize("member", ["../escape.txt", "/absolute.txt", "C:/escape.txt", "a/../../escape.txt"])
def test_unsafe_archives_rejected(tmp_path, member):
    source = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(member, "data")
    with pytest.raises(ValueError):
        safe_extract(source, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_archive_never_overwrites_existing(tmp_path):
    target = tmp_path / "output"
    target.mkdir()
    (target / "existing.txt").write_text("original")
    source = tmp_path / "archive.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("existing.txt", "changed")
    with pytest.raises(ValueError):
        safe_extract(source, target)
    assert (target / "existing.txt").read_text() == "original"


def build_fixture(root, task):
    (root / "src/configs").mkdir(parents=True, exist_ok=True)
    mapping_path = root / "src/configs/label_mappings.json"
    mappings = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    mappings[task] = {"approved": True, "box_format": "yolo_cxcywh_normalized", "classes": {"0": 0},
                      "names": ["test_class"], "evidence": "synthetic test only"}
    mapping_path.write_text(json.dumps(mappings))
    source = root / "data/raw" / task
    source.mkdir(parents=True)
    rows = []
    for i in range(12):
        image = source / f"{i}.png"
        label = source / f"{i}.txt"
        Image.new("RGB", (20, 20), (i * 10, 20, 30)).save(image)
        label.write_text("0 0.5 0.5 0.2 0.2\n")
        rows.append({"image_path": image.relative_to(root).as_posix(), "label_path": label.relative_to(root).as_posix(),
                     "group_id": f"p{i}", "view": "front"})
    index = root / f"{task}.source.csv"
    with index.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return index


def normalize_and_split(root, task):
    manifest = convert_index(root, task, build_fixture(root, task))
    review = root / f"{task}.review.json"
    review.write_text(json.dumps({"reviewed": True, "manifest_sha256": digest(manifest),
                                 "grouping_evidence": "synthetic independent identities", "independence_claim": "fixture only"}))
    return prepare_splits(root, task, review)


def test_missing_label_is_not_a_negative(tmp_path):
    index = build_fixture(tmp_path, "caries")
    (tmp_path / "data/raw/caries/0.txt").unlink()
    with pytest.raises(ValueError, match="Missing"):
        convert_index(tmp_path, "caries", index)
    assert not (tmp_path / "data/manifests/caries.normalized.csv").exists()


def test_mapping_requires_verified_schema(tmp_path):
    index = build_fixture(tmp_path, "caries")
    path = tmp_path / "src/configs/label_mappings.json"
    mapping = json.loads(path.read_text())
    mapping["caries"]["approved"] = False
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="verified"):
        convert_index(tmp_path, "caries", index)


def test_end_to_end_package_and_tamper_detection(tmp_path):
    for task in ("caries", "gum"):
        normalize_and_split(tmp_path, task)
    (tmp_path / "src/configs/kaggle.json").write_text(json.dumps({"owner": "test", "data_slug": "fixture"}))
    (tmp_path / "src/configs/sources.json").write_text("{}")
    (tmp_path / "data/pilot").mkdir()
    (tmp_path / "data/pilot/private.jpg").write_text("must never upload")
    package = package_data(tmp_path)
    with zipfile.ZipFile(package / "prepared.zip") as archive:
        names = archive.namelist()
        assert "caries/data.yaml" in names and "gum/data.yaml" in names
        assert not any("pilot" in name or "raw" in name for name in names)
        safe_extract(package / "prepared.zip", tmp_path / "roundtrip")
    label = next((tmp_path / "data/processed/gum/dataset/labels").rglob("*.txt"))
    label.write_text("0 0.1 0.1 0.1 0.1")
    with pytest.raises(ValueError, match="differs"):
        package_data(tmp_path)


def test_code_package_allowlist_excludes_secrets_and_raw(tmp_path):
    (tmp_path / "src/configs").mkdir(parents=True)
    (tmp_path / "src/configs/kaggle.json").write_text(json.dumps({"owner": "test", "code_slug": "code"}))
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src/module.py").write_text("x = 1")
    (tmp_path / "kaggle.json").write_text("secret")
    (tmp_path / "data/pilot").mkdir(parents=True)
    (tmp_path / "data/pilot/private.jpg").write_text("private")
    with zipfile.ZipFile(package_code(tmp_path) / "project.zip") as archive:
        assert "src/module.py" in archive.namelist()
        assert "kaggle.json" not in archive.namelist()
        assert not any(name.startswith("data/") for name in archive.namelist())


def test_notebooks_valid_and_syntactically_valid():
    import nbformat
    root = Path(__file__).resolve().parents[2]
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 5
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                assert not any(output.output_type == "error" for output in cell.outputs)
                if path.name.startswith(("03_", "04_")):
                    assert not cell.outputs and cell.execution_count is None
                compile(cell.source, str(path), "exec")


def test_pair_failure_stops_other_worker_and_preserves_logs(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from dentalselfie import run_pair
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"experiments": {"E0": {"ready": True}}}))
    for task in ("caries", "gum"):
        directory = tmp_path / "data" / task
        directory.mkdir(parents=True)
        (directory / "data.yaml").write_text("names: [test]")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 2, get_device_name=lambda i: "Mock T4")))
    children, devices = [], []

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.returncode = 1 if not children else None
            self.terminated = False
            devices.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            children.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(run_pair.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(sys, "argv", ["run_pair", "--experiment", "E0", "--config", str(config),
                                    "--data-root", str(tmp_path / "data"), "--output", str(tmp_path / "out")])
    with pytest.raises(RuntimeError, match="worker failed"):
        run_pair.main()
    assert devices == ["0", "1"]
    assert children[1].terminated
    assert len(list((tmp_path / "out").glob("*.log"))) == 2
