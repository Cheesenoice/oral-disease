"""Allowlisted public exports of derived ROIs and completed YOLO v4 evidence."""
from __future__ import annotations

import json
import hashlib
import re
import zipfile
from pathlib import Path

import pandas as pd

from .acquire import digest
from .research import write_json

SOURCES = "https://doi.org/10.5281/zenodo.14827784\nhttps://doi.org/10.17632/3253gj88rr.1\n"


def scan_text(path: Path):
    """Reject common credential signatures without printing candidate contents."""
    if path.suffix.lower() in {".md", ".py", ".txt", ".json", ".log", ".ipynb", ".yaml", ".toml", ".csv", ".ps1"}:
        data = path.read_bytes()
        patterns = [rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", rb"KGAT_[A-Za-z0-9_-]{24,}"]
        if any(re.search(pattern, data) for pattern in patterns):
            raise ValueError(f"Credential signature candidate; publication stopped: {path.name}")


def package_research(root: Path, kind: str) -> Path:
    config = json.loads((root / "src/configs/kaggle.json").read_text())
    if config.get("private", True):
        raise ValueError("This exporter requires the user's explicit public project policy")
    target = root / ".venv/work/kaggle/staging" / kind
    target.mkdir(parents=True, exist_ok=True)
    members = []
    if kind in ("roi", "raw_roi"):
        source = root / ("data/processed/gum_roi_v1" if kind == "roi" else "data/processed/gum_raw_roi_v1")
        receipt = json.loads((source / "receipt.json").read_text())
        if digest(source / "manifest.csv") != receipt["manifest_sha256"]:
            raise ValueError("ROI manifest checksum differs")
        frame = pd.read_csv(source / "manifest.csv")
        if set(frame.split) != {"train", "val"}:
            raise ValueError("Only train/validation ROIs belong in this export")
        for row in frame.to_dict("records"):
            for view in (("local", "context") if kind == "roi" else ("local",)):
                p = source / row[f"{view}_path"]
                if not p.resolve().is_relative_to(source.resolve()) or digest(p) != row[f"{view}_sha256"]:
                    raise ValueError("ROI crop path/hash differs")
                members.append((p, row[f"{view}_path"]))
        for name in ("manifest.csv", "receipt.json", "class_counts.csv"):
            members.append((source / name, name))
        members.append((root / "src/configs/roi.json", "roi_config.json"))
        title, license_name = "DentalSelfie Gingival ROI v1", "CC-BY-4.0"
        description = ("Derived local/context gingival crops: 10,889 ROIs and 21,778 views.\n"
                       "Train/validation only; parent splits and source class groups preserved.\n"
                       "224px RGB JPEG; context 1.5x; no new clinical labels.\n"
                       "Crops are correlated within source images and are not independent patients.\n"
                       "Source photographs/annotations: CC BY 4.0; retained source attribution:\n" + SOURCES)
        if kind == "raw_roi":
            title = "DentalSelfie Original Source Gingival ROI v1"
            description = ("10,889 local crops extracted directly from verified original high-resolution images before 224px resize.\n"
                           "Same crop IDs, class groups and train/validation splits as DentalSelfie ROI v1.\n"
                           "No context views or test images; no new clinical labels; source-parent correlated observations.\n"
                           "Original gingivitis data: Nguyen et al., A dental intraoral image dataset of gingivitis for image captioning, Data in Brief 57 (2024), 110960.\n"
                           "License and source attribution: CC BY 4.0, https://doi.org/10.17632/3253gj88rr.1\n")
    elif kind == "results":
        for experiment in ("E0", "E1"):
            name = f"{experiment}_full_v4"
            report = root / "docs/reports/training" / name
            receipt = json.loads((report / "collection_receipt.json").read_text())
            if receipt["state"] != "COMPLETE" or receipt["completed_workers"] != 2:
                raise ValueError("Only completed, collected v4 results may be published")
            for directory, prefix in ((root / "data/runs" / name, f"runs/{name}"), (report, f"reports/{name}")):
                members.extend((p, f"{prefix}/{p.relative_to(directory).as_posix()}") for p in directory.rglob("*") if p.is_file())
        roi_report = root / "docs/reports/roi_preparation"
        members.extend((p, f"reports/roi_preparation/{p.name}") for p in roi_report.iterdir() if p.is_file())
        for name in ("sources.json", "label_mappings.json", "experiments.json", "roi.json"):
            members.append((root / "src/configs" / name, f"configs/{name}"))
        members.append((root / "docs/reports/data_preparation/DATA_CARD.md", "DATA_CARD.md"))
        title, license_name = "DentalSelfie YOLO v4 Results", "other"
        description = ("Completed YOLOv8n/YOLO26n v4 evidence: four task workers, seed 42.\n"
                       "Includes checkpoints, curves, logs, run metadata and validation results.\n"
                       "Not a held-out-test evaluation or validated phone screening system.\n"
                       "Source images retain source attribution/licenses; model/software artifacts retain applicable upstream terms.\n"
                       "Ultralytics source: https://github.com/ultralytics/ultralytics\n" + SOURCES)
    else:
        raise ValueError("Unknown research export")
    names = [name for _, name in members]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive member")
    records = []
    for source, name in sorted(members, key=lambda item: item[1]):
        if any(part in {".venv", "pilot", ".kaggle", ".env"} for part in Path(name).parts):
            raise ValueError("Forbidden publication member")
        scan_text(source)
        records.append({"file": name, "bytes": source.stat().st_size, "sha256": digest(source)})
    archive = target / f"{kind}.zip"
    signature = hashlib.sha256(json.dumps({"files": records, "description": description}, sort_keys=True).encode()).hexdigest()
    cache = target / "package_receipt.json"
    if archive.exists() and cache.exists():
        previous = json.loads(cache.read_text())
        if digest(archive) != previous["sha256"]:
            raise ValueError("Staged public archive checksum differs")
        with zipfile.ZipFile(archive) as bundle:
            same_inputs = json.loads(bundle.read("content_manifest.json")) == records and bundle.read("README.md").decode() == description
        if same_inputs:
            previous["input_signature"] = signature
            write_json(cache, previous)
            return target
    with zipfile.ZipFile(archive.with_suffix(".zip.part"), "w", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for source, name in sorted(members, key=lambda item: item[1]):
            bundle.write(source, name)
        bundle.writestr("content_manifest.json", json.dumps(records, indent=2))
        bundle.writestr("README.md", description)
    with zipfile.ZipFile(archive.with_suffix(".zip.part")) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Export ZIP failed CRC verification")
    archive.with_suffix(".zip.part").replace(archive)
    write_json(target / "dataset-metadata.json", {"title": title, "id": f"{config['owner']}/{config[kind+'_slug']}", "licenses": [{"name": license_name}]})
    (target / "README.md").write_text(description, encoding="utf-8")
    write_json(target / "package_receipt.json", {"kind": kind, "members": len(records)+2, "input_signature": signature,
                "sha256": digest(archive), "bytes": archive.stat().st_size, "public": True})
    return target
