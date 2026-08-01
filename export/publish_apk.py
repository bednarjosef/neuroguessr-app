#!/usr/bin/env python3
"""Publish the signed APK + repo front page to the Hugging Face distribution repo.

Run after publish_assets.py. Expects the release APK to have been copied to
dist/neuroguessr-<version>.apk (assembleRelease leaves it in
android/app/build/outputs/apk/release/app-release.apk).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO_ID = "josefbednar/neuroguessr-app"
MOBILE = Path(__file__).resolve().parent.parent
VERSION = sys.argv[1] if len(sys.argv) > 1 else "0.2"


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO_ID, exist_ok=True)
    for src, dest in [
        (MOBILE / "dist" / f"neuroguessr-{VERSION}.apk", f"neuroguessr-{VERSION}.apk"),
        (MOBILE / "export" / "hf_readme.md", "README.md"),
        (MOBILE / "docs" / "screenshot.png", "screenshot.png"),
    ]:
        api.upload_file(
            path_or_fileobj=src, path_in_repo=dest, repo_id=REPO_ID,
            commit_message=f"app: {dest}",
        )
        print(f"uploaded {dest}", flush=True)
    print(f"done: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
