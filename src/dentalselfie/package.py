"""Create allowlisted Kaggle artifacts; never upload the workspace root."""
from __future__ import annotations

import json
import shutil
import zipfile
import csv
import hashlib
from pathlib import Path

from .acquire import digest


def package_code(root: Path) -> Path:
    target = root / ".venv/work/kaggle/staging/code"
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "src/configs/kaggle.json").read_text(encoding="utf-8"))
    allowed = ["src", "notebooks"]
    standalone = ["pyproject.toml", "requirements-kaggle.txt", "uv.lock", "README.md", "AGENTS.md", "KE_HOACH_DO_AN.md"]
    paths = [root / name for name in standalone]
    paths.extend((root / "docs").glob("*.md"))
    roi_reports = root / "docs/reports/roi_preparation"
    if roi_reports.exists():
        paths.extend(p for p in roi_reports.iterdir() if p.is_file())
    for directory in allowed:
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file()
                     and not any(part == "__pycache__" or part.endswith(".egg-info")
                                 or part == ".ipynb_checkpoints" for part in p.parts)
                     and not p.name.endswith(".pyc"))
    with zipfile.ZipFile(target / "project.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if path.is_file():
                # Local capture notebooks can contain personal images in cell outputs.
                # Keep the reserved notebook excluded even if its metadata is edited.
                if path.suffix == ".ipynb":
                    book = json.loads(path.read_text(encoding="utf-8"))
                    if (path.name.startswith("12_2_phone_photo_") or
                            book.get("metadata", {}).get("dentalselfie", {}).get("local_only", False)):
                        continue
                if path.name == "kaggle.json" and path.parent.name != "configs":
                    raise ValueError("Credential file must not be packaged")
                archive.write(path, path.relative_to(root).as_posix())
    metadata = {"title": "DentalSelfie Code", "id": f"{config['owner']}/{config['code_slug']}",
                "licenses": [{"name": "other"}]}
    (target / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (target / "README.md").write_text(
        "Project source and executed audit notebooks for public research datasets.\n"
        "Source photographs retain their original CC BY 4.0 attribution:\n"
        "https://doi.org/10.5281/zenodo.14827784\nhttps://doi.org/10.17632/3253gj88rr.1\n"
        "No personal pilot photographs or credentials are included.\n", encoding="utf-8")
    return target


def package_data(root: Path) -> Path:
    config = json.loads((root / "src/configs/kaggle.json").read_text(encoding="utf-8"))
    target = root / ".venv/work/kaggle/staging/data"
    members = []
    # Verify BOTH tasks and image/label hashes before creating any package.
    for task in ("caries", "gum"):
        lock = root / "data/manifests" / f"{task}.lock.json"
        splits = root / "data/manifests" / f"{task}.splits.csv"
        if not lock.is_file() or not splits.is_file():
            raise ValueError(f"{task}: audited, locked splits required before upload")
        record = json.loads(lock.read_text(encoding="utf-8"))
        if record["splits_sha256"] != digest(splits):
            raise ValueError(f"{task}: split changed after lock")
        if not (root / "data/processed" / task / "dataset/data.yaml").is_file():
            raise ValueError(f"{task}: materialized dataset missing")
        dataset = root / "data/processed" / task / "dataset"
        with splits.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                matches = list((dataset / "images" / row["split"]).glob(row["sample_id"] + ".*"))
                label = dataset / "labels" / row["split"] / f"{row['sample_id']}.txt"
                if len(matches) != 1 or digest(matches[0]) != row["sha256"] or digest(label) != row["label_sha256"]:
                    raise ValueError(f"{task}: prepared image/label differs from normalized manifest")
                members.extend([(matches[0], f"{task}/images/{row['split']}/{matches[0].name}"),
                                (label, f"{task}/labels/{row['split']}/{label.name}")])
        members.extend([(dataset / "data.yaml", f"{task}/data.yaml"),
                        (lock, f"manifests/{task}.lock.json"), (splits, f"manifests/{task}.splits.csv")])
    card = root / "docs/reports/data_preparation/DATA_CARD.md"
    if card.is_file():
        members.append((card, "DATA_CARD.md"))
    signature_rows = [(name, digest(source), source.stat().st_size) for source, name in members]
    signature_rows.append(("sources.json", digest(root / "src/configs/sources.json"), 0))
    signature = hashlib.sha256(json.dumps(signature_rows, sort_keys=True).encode()).hexdigest()
    receipt_path = target / "package_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["input_signature"] != signature:
            raise ValueError("Staged data belongs to another protocol; use an explicit new dataset version")
        if digest(target / "prepared.zip") != receipt["archive_sha256"]:
            raise ValueError("Staged archive checksum changed")
        return target
    target.mkdir(parents=True, exist_ok=True)
    partial = target / "prepared.zip.part"
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for source, name in members:
            archive.write(source, name)
        archive.write(root / "src/configs/sources.json", "sources.json")
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError("Prepared ZIP failed its CRC integrity check")
    partial.replace(target / "prepared.zip")
    shutil.copy2(root / "src/configs/sources.json", target / "sources.json")
    metadata = {"title": "DentalSelfie Data v1", "id": f"{config['owner']}/{config['data_slug']}",
                "licenses": [{"name": "CC-BY-4.0"}]}
    (target / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    description = (
        "Derived research datasets, CC BY 4.0. Original sources:\n"
        "https://doi.org/10.5281/zenodo.14827784\nhttps://doi.org/10.17632/3253gj88rr.1\n"
        "Changes: class remapping, format normalization and new group-aware splits.\n"
        "See sources.json and manifests for provenance. No DentalSelfie pilot photos included.\n"
    )
    if card.is_file():
        shutil.copy2(card, target / "DATA_CARD.md")
        description += "\n" + card.read_text(encoding="utf-8")
    (target / "README.md").write_text(description, encoding="utf-8")
    receipt_path.write_text(json.dumps({"input_signature": signature,
                                       "archive_sha256": digest(target / "prepared.zip"),
                                       "archive_bytes": (target / "prepared.zip").stat().st_size,
                                       "archive_members": len(members) + 1}, indent=2), encoding="utf-8")
    return target
