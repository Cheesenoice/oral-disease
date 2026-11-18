from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="DentalSelfie project tools")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("download-caries")
    audit = sub.add_parser("audit")
    audit.add_argument("dataset", choices=["caries", "gum"])
    audit.add_argument("path", type=Path)
    extract = sub.add_parser("extract")
    extract.add_argument("archive", type=Path)
    extract.add_argument("destination", type=Path)
    convert = sub.add_parser("convert-yolo")
    convert.add_argument("task", choices=["caries", "gum"])
    convert.add_argument("index", type=Path)
    split = sub.add_parser("split")
    split.add_argument("task", choices=["caries", "gum"])
    split.add_argument("review", type=Path)
    split.add_argument("--seed", type=int, default=42)
    package = sub.add_parser("package")
    package.add_argument("kind", choices=["code", "data"])
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "src/configs/sources.json").is_file():
        parser.error("Run from the project root, or supply --root PATH before the command")
    if args.command == "doctor":
        report = {"python": platform.python_version(), "free_gb": round(shutil.disk_usage(root).free / 2**30, 1),
                  "kaggle_cli": bool(shutil.which("kaggle")), "git": bool(shutil.which("git")),
                  "adb": bool(shutil.which("adb")), "java": bool(shutil.which("java"))}
        print(json.dumps(report, indent=2))
    elif args.command == "download-caries":
        from .acquire import download_caries
        download_caries(root)
    elif args.command == "audit":
        from .audit import audit_source
        result = audit_source(args.path, root / "docs/reports/data_audit" / args.dataset)
        print(json.dumps(result, indent=2, ensure_ascii=True))
    elif args.command == "extract":
        from .audit import safe_extract
        destination = args.destination.resolve()
        if not destination.is_relative_to(root):
            parser.error("Extraction destination must stay within the project root")
        safe_extract(args.archive, destination)
        print(destination)
    elif args.command == "convert-yolo":
        from .data import convert_index
        print(convert_index(root, args.task, args.index))
    elif args.command == "split":
        from .data import prepare_splits
        print(prepare_splits(root, args.task, args.review, args.seed))
    elif args.command == "package":
        from .package import package_code, package_data
        print((package_code if args.kind == "code" else package_data)(root))


if __name__ == "__main__":
    main()
