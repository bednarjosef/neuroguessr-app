#!/usr/bin/env python3
"""Publish the app's runtime assets to Hugging Face so the in-app downloader can fetch them.

Hashes every file the app needs (index/ + encoder_fp16.onnx), writes download_manifest.json
twice — once next to the assets for the upload, once into the Android app's assets/ so the
APK knows exactly which asset build it pairs with — then uploads the lot to REPO_ID.

Run from anywhere: paths are resolved relative to this file. Re-running is safe; HF skips
unchanged files (xet dedup) and the manifest is regenerated from what is on disk.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO_ID = "josefbednar/neuroguessr-app"
MOBILE = Path(__file__).resolve().parent.parent
ASSETS = MOBILE / "assets_build"
ANDROID_ASSETS = MOBILE / "android" / "app" / "src" / "main" / "assets"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/assets/"

# everything the app reads from its external files dir, relative to assets_build/
FILES = sorted(p.relative_to(ASSETS) for p in (ASSETS / "index").iterdir()) + [
    Path("model/encoder_fp16.onnx"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    entries = []
    for rel in FILES:
        p = ASSETS / rel
        print(f"hashing {rel} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)
        entries.append({"path": str(rel), "bytes": p.stat().st_size, "sha256": sha256(p)})
    manifest = {
        "version": 1,
        "tag": json.loads((ASSETS / "manifest.json").read_text())["tag"],
        "base_url": BASE_URL,
        "total_bytes": sum(e["bytes"] for e in entries),
        "files": entries,
    }
    text = json.dumps(manifest, indent=1)
    (ASSETS / "download_manifest.json").write_text(text)
    (ANDROID_ASSETS / "download_manifest.json").write_text(text)
    print(f"manifest: {len(entries)} files, {manifest['total_bytes'] / 1e9:.2f} GB")

    if "--no-upload" in sys.argv:
        return
    from huggingface_hub import HfApi

    # One file at a time, one commit each. Slower on paper than the parallel path, but this
    # uplink collapses when driven near saturation (parallel streams died every ~6 min and
    # lost most in-flight chunks each time), while a single stream ran stall-free — and a
    # commit per file makes every finished file durable. Prefer this on flaky links.
    if "--sequential" in sys.argv:
        api = HfApi()
        api.create_repo(REPO_ID, exist_ok=True)
        done = {f.path for f in api.list_repo_tree(REPO_ID, path_in_repo="assets", recursive=True)}
        for rel in FILES + [Path("download_manifest.json")]:
            if f"assets/{rel}" in done and rel != Path("download_manifest.json"):
                print(f"already committed: {rel}", flush=True)
                continue
            src = ASSETS / rel
            api.upload_file(path_or_fileobj=src, path_in_repo=f"assets/{rel}",
                            repo_id=REPO_ID, commit_message=f"assets: {rel}")
            print(f"uploaded {rel}", flush=True)
        for name in ["NOTICE.md", "DINOV3_LICENSE.md"]:
            api.upload_file(path_or_fileobj=MOBILE / name, path_in_repo=name,
                            repo_id=REPO_ID, commit_message="licence notice")
        print(f"done: https://huggingface.co/{REPO_ID}")
        return

    # Stage hardlinks in the exact repo layout and hand the tree to upload_large_folder,
    # which transfers several files in parallel — a single stream only reached ~40% of the
    # uplink. It is also resumable: committed files are skipped, xet dedups partial chunks.
    stage = ASSETS / ".hf_stage"
    to_stage = {Path("assets") / f: ASSETS / f for f in FILES}
    to_stage[Path("assets/download_manifest.json")] = ASSETS / "download_manifest.json"
    to_stage[Path("NOTICE.md")] = MOBILE / "NOTICE.md"
    to_stage[Path("DINOV3_LICENSE.md")] = MOBILE / "DINOV3_LICENSE.md"
    for rel, src in to_stage.items():
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        os.link(src, dst)

    api = HfApi()
    api.create_repo(REPO_ID, exist_ok=True)
    api.upload_large_folder(
        repo_id=REPO_ID, folder_path=stage, repo_type="model",
        num_workers=int(os.environ.get("AR_UPLOAD_WORKERS", "4")),
        print_report_every=30,
    )
    print(f"done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
