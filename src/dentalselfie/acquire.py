"""Public-source acquisition with resumable downloads and immutable receipts."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_caries(root: Path) -> Path:
    source = json.loads((root / "src/configs/sources.json").read_text(encoding="utf-8"))["caries"]
    target = root / "data/raw/caries" / source["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source["size"] or digest(target, "md5") != source["md5"]:
            raise ValueError("Existing final archive fails checksum; preserve it for inspection.")
        print("Already downloaded and verified:", target, flush=True)
        return target
    metadata = requests.get(source["metadata_api"], timeout=60)
    metadata.raise_for_status()
    (target.parent / "source_metadata.json").write_text(
        json.dumps(metadata.json(), indent=2), encoding="utf-8"
    )
    partial = target.with_suffix(".zip.part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == source["size"]:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(source["url"], headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an unexpected resume range")
                elif response.status_code == 200:
                    offset = 0
                else:
                    raise ValueError(f"Unexpected download status {response.status_code}")
                mode = "ab" if offset else "wb"
                last_log = time.monotonic()
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_log >= 15:
                            print(f"caries: {offset / source['size']:.1%} ({offset:,} bytes)", flush=True)
                            last_log = time.monotonic()
            break
        except requests.RequestException as error:
            print(f"Network interruption {attempt + 1}/5: {type(error).__name__}; partial kept", flush=True)
            if attempt == 4:
                raise
            time.sleep(2)
    if partial.stat().st_size != source["size"] or digest(partial, "md5") != source["md5"]:
        raise ValueError("Incomplete download or MD5 mismatch; .part kept, not accepted as raw data")
    partial.replace(target)
    receipt = {"source": source, "sha256": digest(target), "verified_md5": True}
    (target.parent / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("Downloaded and checksum verified:", target, flush=True)
    return target
