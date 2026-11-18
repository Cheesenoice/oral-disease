import csv
import json

import pytest

from dentalselfie.acquire import digest
from dentalselfie.kaggle_data import stage_prepared_dataset


def test_local_staging_is_hash_checked_resumable_and_bound(tmp_path):
    source = tmp_path / "source"
    manifests = source / "manifests"
    manifests.mkdir(parents=True)
    for task in ("caries", "gum"):
        (source / task).mkdir()
        (source / task / "data.yaml").write_text("names: [fixture]")
        image = source / task / "images/train/example.jpg"
        label = source / task / "labels/train/example.txt"
        for p in (image, label):
            p.parent.mkdir(parents=True)
            p.write_bytes(b"synthetic fixture")
        split = manifests / f"{task}.splits.csv"
        with split.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=["sample_id", "split", "sha256", "label_sha256"])
            writer.writeheader()
            writer.writerow({"sample_id": "example", "split": "train", "sha256": digest(image),
                             "label_sha256": digest(label)})
        (manifests / f"{task}.lock.json").write_text(json.dumps({"splits_sha256": digest(split)}))
    target = stage_prepared_dataset(source, tmp_path / "local", workers=2)
    assert digest(target / "caries/images/train/example.jpg") == digest(source / "caries/images/train/example.jpg")
    (target / "caries/images/train/example.jpg").write_bytes(b"partial")
    stage_prepared_dataset(source, target, workers=2)
    assert digest(target / "caries/images/train/example.jpg") == digest(source / "caries/images/train/example.jpg")
    (source / "gum/images/train/example.jpg").write_bytes(b"tampered")
    (target / "gum/images/train/example.jpg").unlink()
    with pytest.raises(ValueError, match="checksum mismatch"):
        stage_prepared_dataset(source, target, workers=2)
    with pytest.raises(ValueError, match="separate"):
        stage_prepared_dataset(source, source / "nested")
