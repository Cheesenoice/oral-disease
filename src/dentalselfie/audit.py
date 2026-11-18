"""Inventory real archives/directories without guessing the source annotation schema."""
from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


def audit_source(source: Path, output: Path) -> dict:
    if not source.exists():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            entries = [(item.filename, item.file_size) for item in archive.infolist() if not item.is_dir()]
            candidates = [name for name, size in entries if size < 100_000 and
                          PurePosixPath(name).suffix.lower() in {".txt", ".xml", ".json", ".yaml"}]
            # Spread examples over subfolders, not only the first class.
            chosen = {}
            for name in candidates:
                key = (str(PurePosixPath(name).parent), PurePosixPath(name).suffix.lower())
                chosen.setdefault(key, name)
            samples = {name: archive.read(name).decode("utf-8-sig", errors="replace")
                       for name in list(chosen.values())[:100]}
    else:
        entries = [(path.relative_to(source).as_posix(), path.stat().st_size)
                   for path in sorted(source.rglob("*")) if path.is_file()]
        samples = {}
        for name, size in entries:
            if len(samples) >= 100:
                break
            if size < 100_000 and Path(name).suffix.lower() in {".txt", ".xml", ".json", ".yaml"}:
                samples[name] = (source / name).read_text(encoding="utf-8-sig", errors="replace")
    extensions = Counter(PurePosixPath(name).suffix.lower() for name, _ in entries)
    result = {"source": str(source), "files": len(entries), "uncompressed_bytes": sum(s for _, s in entries),
              "extensions": dict(extensions), "annotation_mapping_verified": False}
    with (output / "inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "bytes"])
        writer.writerows(entries)
    (output / "annotation_samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def safe_extract(archive_path: Path, destination: Path, max_bytes: int = 25 * 2**30) -> None:
    """Reject traversal, symlinks and overwrites before writing any archive member."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if sum(m.file_size for m in members) > max_bytes:
            raise ValueError("Archive exceeds extraction size limit")
        targets = []
        seen = set()
        for member in members:
            portable = member.filename.replace("\\", "/")
            if PurePosixPath(portable).is_absolute() or ":" in portable:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            target = (destination / portable).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive traversal: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not accepted")
            if not member.is_dir():
                if target in seen or target.exists():
                    raise ValueError(f"Duplicate or existing extraction target: {target}")
                seen.add(target)
            targets.append((member, target))
        for member, target in targets:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("xb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
