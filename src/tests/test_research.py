"""Synthetic checks for source interpretation and interruption-safe preparation."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from dentalselfie.acquire import digest
from dentalselfie.research import audit_annotations, assign_groups, prepare_task, validate_prepared


def test_named_caries_annotations_and_whole_image_quarantine(tmp_path):
    image = tmp_path / "data/raw/caries/Dataset/Images/pilot/Frontal/example.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (100, 60), "white").save(image)
    label = tmp_path / "data/raw/caries/Dataset/Annotations/Labelme/pilot/Frontal/example.json"
    label.parent.mkdir(parents=True)
    info = {"task": "caries", "image_path": image.relative_to(tmp_path).as_posix(), "bytes": image.stat().st_size,
            "sha256": digest(image), "width": 100, "height": 60, "exif_orientation": 0,
            "decode_error": "", "phash": "0" * 16, "dhash": "0" * 16}
    records, _ = audit_annotations(tmp_path, pd.DataFrame([info]))
    assert records.iloc[0].reasons == ["missing_annotation_not_a_negative"]
    document = {"imageWidth": 100, "imageHeight": 60, "imagePath": "example.jpg",
                "shapes": [{"label": "D", "shape_type": "rectangle", "points": [[10, 10], [40, 40]]}]}
    label.write_text(json.dumps(document))
    records, _ = audit_annotations(tmp_path, pd.DataFrame([info]))
    assert records.iloc[0].eligible
    assert records.iloc[0].target_boxes[0][0] == 0
    document["shapes"].append({"label": "M", "shape_type": "rectangle", "points": [[50, 10], [70, 40]]})
    label.write_text(json.dumps(document))
    records, _ = audit_annotations(tmp_path, pd.DataFrame([info]))
    assert not records.iloc[0].eligible
    assert "unknown_caries_shape_or_class" in records.iloc[0].reasons


def test_conflicting_duplicate_annotations_are_not_arbitrarily_selected():
    common = {"task": "gum", "sha256": "identical", "eligible": True, "reasons": [], "group_hint": "same"}
    records = pd.DataFrame([dict(common, sample_id="a", target_boxes=[[0, 0.5, 0.5, 0.1, 0.1]]),
                            dict(common, sample_id="b", target_boxes=[[1, 0.5, 0.5, 0.1, 0.1]])])
    grouped = assign_groups(records, pd.DataFrame(columns=["sample_a", "sample_b"]))
    assert not grouped.eligible.any()
    assert grouped.group_id.nunique() == 1
    assert all("exact_duplicate_annotation_conflict" in reasons for reasons in grouped.reasons)


def test_preparation_recovers_after_interrupted_promotion(tmp_path, monkeypatch):
    config = tmp_path / "src/configs/label_mappings.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"caries": {"approved": True, "names": ["visible_caries"]}}))
    audit = tmp_path / "audit.jsonl"
    audit.write_text("synthetic audit fixture")
    rows = []
    for index in range(12):
        image = tmp_path / "data/raw" / f"{index}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 50), (index * 12, 30, 90)).save(image)
        label = image.with_suffix(".json")
        label.write_text("{}")
        rows.append({"task": "caries", "sample_id": f"sample{index:02}",
                     "image_path": image.relative_to(tmp_path).as_posix(),
                     "label_path": label.relative_to(tmp_path).as_posix(), "sha256": digest(image),
                     "group_id": f"case{index}", "group_basis": "synthetic", "view": "front",
                     "source_split": "unknown", "source_collection": "fixture", "width": 80, "height": 50,
                     "target_boxes": [[0, 0.5, 0.5, 0.2, 0.2]], "source_boxes": [{"source_class": "D"}]})
    selected = pd.DataFrame(rows)
    real_rename = Path.rename

    def blocked_rename(path, destination):
        if "data_preparation" in path.parts:
            raise PermissionError("synthetic Windows lock")
        return real_rename(path, destination)

    monkeypatch.setattr(Path, "rename", blocked_rename)
    monkeypatch.setattr("dentalselfie.research.time.sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        prepare_task(tmp_path, "caries", selected, audit, workers=1)
    monkeypatch.setattr(Path, "rename", real_rename)

    def should_not_encode_again(*args, **kwargs):
        raise AssertionError("Already complete staged images should be reused")

    monkeypatch.setattr(Image.Image, "save", should_not_encode_again)
    prepared = prepare_task(tmp_path, "caries", selected, audit, workers=1)
    pairs = pd.DataFrame(columns=["task", "sample_a", "sample_b"])
    checks = validate_prepared(tmp_path, prepared, pairs)
    assert len(prepared) == 12 and set(checks.validation) == {"passed"}
    reused = prepare_task(tmp_path, "caries", selected, audit, workers=1)
    assert prepared.sample_id.tolist() == reused.sample_id.tolist()
    first_label = tmp_path / prepared.iloc[0].label
    first_label.write_text("0 0.1 0.1 0.1 0.1\n")
    with pytest.raises(AssertionError):
        validate_prepared(tmp_path, prepared, pairs)
