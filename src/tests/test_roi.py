"""Meaningful geometry, provenance and split isolation checks for ROI preparation."""
import json
from pathlib import Path
import pandas as pd
import pytest
from PIL import Image
from dentalselfie.acquire import digest
from dentalselfie.roi import pixel_box, view_bounds, padded_view, prepare_rois


def test_pixel_geometry_and_invalid_boxes():
    assert pixel_box([.5, .5, .2, .4], 100, 50) == (40, 15, 60, 35)
    with pytest.raises(ValueError): pixel_box([0, .5, .2, .4], 100, 50)
    with pytest.raises(ValueError): pixel_box([.5, .5, float("nan"), .4], 100, 50)


def test_padding_is_explicit_and_keeps_original_pixels():
    image = Image.new("RGB", (10, 10), (255, 0, 0))
    crop, padding = padded_view(image, (-2, -2, 8, 8), 10, [128, 128, 128])
    assert padding == (2, 2, 0, 0)
    assert crop.getpixel((0, 0)) == (128, 128, 128)
    assert crop.getpixel((3, 3)) == (255, 0, 0)
    assert view_bounds((2, 2, 6, 6), 1.5) == (1, 1, 7, 7)


def fixture(root):
    dataset, manifests = root / "dataset", root / "manifests"
    manifests.mkdir()
    rows = []
    for split in ["train", "val"]:
        im, label = dataset / f"images/{split}/a{split}.jpg", dataset / f"labels/{split}/a{split}.txt"
        im.parent.mkdir(parents=True); label.parent.mkdir(parents=True)
        Image.new("RGB", (20, 10), "red").save(im)
        label.write_text("1 0.5 0.5 0.2 0.4\n")
        rows.append(dict(sample_id=f"a{split}", split=split, group_id=split, sha256=digest(im),
                         label_sha256=digest(label), width=20, height=10, label_count=1))
    # No test files exist: opening them would fail, so this proves they are not read.
    rows.append(dict(sample_id="held_out", split="test", group_id="test"))
    p = manifests / "gum.splits.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    (manifests / "gum.lock.json").write_text(json.dumps({"splits_sha256": digest(p)}))
    config = root / "roi.json"
    config.write_text(json.dumps(dict(protocol="test", splits=["train", "val"], size=16,
        context_multiplier=1.5, padding_rgb=[128]*3, jpeg_quality=95, jpeg_subsampling=0,
        classes=["none", "mild", "marked"])))
    return dataset, manifests, config


def test_derive_without_test_and_detect_crop_tampering(tmp_path):
    dataset, manifests, config = fixture(tmp_path)
    frame, receipt = prepare_rois(dataset, manifests, tmp_path / "out", config, workers=1)
    assert len(frame) == 2 and receipt["test_images_opened"] == 0
    again, _ = prepare_rois(dataset, manifests, tmp_path / "out", config)
    assert len(again) == 2
    (tmp_path / "out" / frame.iloc[0].local_path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        prepare_rois(dataset, manifests, tmp_path / "out", config)


def test_parent_hash_tampering_is_rejected(tmp_path):
    dataset, manifests, config = fixture(tmp_path)
    (dataset / "labels/train/atrain.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    with pytest.raises(ValueError, match="checksum"):
        prepare_rois(dataset, manifests, tmp_path / "out", config, workers=1)


def test_reuse_rechecks_original_parent(tmp_path):
    dataset, manifests, config = fixture(tmp_path)
    prepare_rois(dataset, manifests, tmp_path / "out", config, workers=1)
    (dataset / "labels/train/atrain.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    with pytest.raises(ValueError, match="Parent checksum"):
        prepare_rois(dataset, manifests, tmp_path / "out", config)


def test_group_leakage_is_rejected_even_if_split_hash_is_updated(tmp_path):
    dataset, manifests, config = fixture(tmp_path)
    p = manifests / "gum.splits.csv"
    frame = pd.read_csv(p)
    frame.loc[frame.split.isin(["train", "val"]), "group_id"] = "same_parent_group"
    frame.to_csv(p, index=False)
    (manifests / "gum.lock.json").write_text(json.dumps({"splits_sha256": digest(p)}))
    with pytest.raises(ValueError, match="groups cross"):
        prepare_rois(dataset, manifests, tmp_path / "out", config)
