"""Stage a train/positive-only-validation subset of teeth_gap v1 for Kaggle.

The source test split is deliberately excluded. The validation images are kept
separate for a post-training positive-recall diagnostic only.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/teeth_gap_v1_screened2"
MANIFEST = ROOT / "docs/reports/alignment/teeth_gap_v1/manifest.csv"
OUT = ROOT / ".venv/work/kaggle/spacing_addon_train_v1"
EXPECTED_SOURCE_SHA = "091923e87bbcd4aa5cd1eb93aa6ab0ee72ff3b522eb451c603ebb04d95abba98"
EXPECTED_MANIFEST_SHA = "9c413bc2f2e811c22fb3e31f46f7ce584840fca75c5d6b99372ce17f1cb94036"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_bytes = MANIFEST.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == EXPECTED_MANIFEST_SHA
    records = pd.read_csv(MANIFEST, dtype={"output_image": str})
    assert set(records["split"].unique()) == {"train", "valid"}
    assert not records["source_split"].eq("test").any()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    counts = {}
    for split, expected_images, expected_boxes in (("train", 119, 216), ("valid", 22, 35)):
        rows = records.loc[records["split"].eq(split)].copy()
        assert (len(rows), int(rows["boxes"].sum())) == (expected_images, expected_boxes)
        image_dir = OUT / split / "images"
        label_dir = OUT / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for row in rows.itertuples(index=False):
            source_image = SOURCE / split / "images" / row.output_image
            source_label = SOURCE / split / "labels" / (Path(row.output_image).stem + ".txt")
            assert source_image.is_file() and source_label.is_file()
            assert sha256_file(source_image) == row.image_sha256
            assert sha256_file(source_label) == row.label_sha256
            target_image = image_dir / row.output_image
            target_label = label_dir / source_label.name
            shutil.copyfile(source_image, target_image)
            shutil.copyfile(source_label, target_label)
        counts[split] = {"images": len(rows), "groups": int(rows["group"].nunique()), "boxes": expected_boxes}

    (OUT / "manifest.csv").write_bytes(manifest_bytes)
    (OUT / "README.md").write_text(
        """# Teeth-gap supplemental data

Derived from Roboflow Universe project `random-dznie/teeth_gap`, version 1, declared CC BY 4.0.

The original `spaced` class is mapped to the project's single `gap_or_spacing` target. Only source
train and valid partitions are included: 119 train images/216 boxes and 22 positive-only validation
images/35 boxes. The source test partition is excluded.

Train images are used only for the add-on arm. The 22-image validation partition is reserved for
post-training positive-recall diagnostics; it has no verified negative images and must not be pooled
with the locked project mAP. Patient independence and clinical label accuracy are not established.

Attribution: Roboflow Universe, `teeth_gap` by `random-dznie`, v1, CC BY 4.0.
Source: https://universe.roboflow.com/random-dznie/teeth_gap/dataset/1
""",
        encoding="utf-8",
    )
    source_sha = hashlib.sha256((ROOT / "data/raw/alignment_external/teeth_gap_v1/original.zip").read_bytes()).hexdigest()
    assert source_sha == EXPECTED_SOURCE_SHA
    metadata = {
        "title": "DentalSelfie Teeth Gap Train Add-on 20261123",
        "id": "hngphc73/teeth-gap-train-addon-v1",
        "licenses": [{"name": "CC-BY-4.0"}],
        "subtitle": "Train-only external gap examples plus a separate positive-only diagnostic split",
        "description": (
            "Roboflow random-dznie/teeth_gap v1, declared CC BY 4.0. Source train: 119 images/216 boxes. "
            "Source valid: 22 positive-only images/35 boxes, reserved for recall diagnostics only. "
            "Source test is excluded. No patient independence or clinical label accuracy claim."
        ),
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    receipt = {
        "source_archive_sha256": source_sha,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "counts": counts,
        "source_test_included": False,
        "label_mapping": "source spaced (1) -> project gap_or_spacing (0)",
        "split_note": "source train and positive-only source valid remain separate; val never enters train",
    }
    (OUT / "package_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print(f"Staged: {OUT}")


if __name__ == "__main__":
    main()
