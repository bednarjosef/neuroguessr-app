#!/usr/bin/env python3
"""Check that every file download_manifest.json promises exists on HF at the right size.

Exits non-zero on any missing or size-mismatched file — the guard between uploading assets
and publishing an APK whose baked-in manifest expects them.
"""
import json
import sys
from pathlib import Path

REPO_ID = "josefbednar/neuroguessr-app"
ASSETS = Path(__file__).resolve().parent.parent / "assets_build"


def main() -> None:
    from huggingface_hub import HfApi

    manifest = json.loads((ASSETS / "download_manifest.json").read_text())
    remote = {
        f.path: f.size
        for f in HfApi().list_repo_tree(REPO_ID, path_in_repo="assets", recursive=True)
        if hasattr(f, "size")
    }
    bad = [
        f"{e['path']}: remote {remote.get('assets/' + e['path'], 'MISSING')} != {e['bytes']}"
        for e in manifest["files"]
        if remote.get("assets/" + e["path"]) != e["bytes"]
    ]
    if bad:
        print("REMOTE MISMATCH:\n" + "\n".join(bad))
        sys.exit(1)
    print(f"remote verified: {len(manifest['files'])} files, "
          f"{manifest['total_bytes'] / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
