"""Stage immutable prepared inputs onto session-local storage with hash checks."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shutil
import time

from .acquire import digest


def stage_prepared_dataset(source: Path, destination: Path, workers: int = 8) -> Path:
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Staging source and destination must be separate directories")
    if not 1 <= workers <= 16:
        raise ValueError("Use between one and sixteen bounded copy workers")
    started = time.monotonic()
    members = []
    for task in ("caries", "gum"):
        lock_path = source / "manifests" / f"{task}.lock.json"
        split_path = source / "manifests" / f"{task}.splits.csv"
        lock = json.loads(lock_path.read_text())
        if digest(split_path) != lock["splits_sha256"]:
            raise ValueError(f"{task}: split lock mismatch")
        for path in (lock_path, split_path, source / task / "data.yaml"):
            members.append((path.relative_to(source), digest(path)))
        with split_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["split"] not in {"train", "val", "test"}:
                    raise ValueError("Unknown split")
                sample = row["sample_id"]
                if not sample or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in sample):
                    raise ValueError("Unsafe sample ID")
                members += [(Path(task) / "images" / row["split"] / f"{sample}.jpg", row["sha256"]),
                            (Path(task) / "labels" / row["split"] / f"{sample}.txt", row["label_sha256"])]
    for name in ("sources.json", "DATA_CARD.md"):
        if (source / name).is_file():
            members.append((Path(name), digest(source / name)))

    def copy_verified(member):
        relative, expected = member
        target = destination / relative
        if target.is_file() and digest(target) == expected:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        shutil.copyfile(source / relative, temporary)
        if digest(temporary) != expected:
            raise ValueError(f"Prepared input checksum mismatch: {relative}")
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(copy_verified, members):
            pass
    receipt = {"source": str(source), "files": len(members), "verified_sha256": True,
               "workers": workers, "seconds": time.monotonic() - started}
    (destination / "staging_receipt.json").write_text(json.dumps(receipt, indent=2))
    print("Verified session-local data staging:", receipt, flush=True)
    return destination
