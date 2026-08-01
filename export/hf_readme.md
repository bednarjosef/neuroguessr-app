---
license: other
license_name: dinov3-license
license_link: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
---

# NeuroGuessr

Point it at a photograph and it tells you where on Earth it was taken. Everything runs on the
phone — the neural encoder, the 300,000-place index, and the map. After the one-time asset
download below, the app never touches the network.

<img src="screenshot.png" width="320" alt="NeuroGuessr showing a prediction in Serbia">

## Install (Android)

You need an **arm64 phone with Android 10 or newer** and about **4.5 GB free**.

1. On the phone, download and open
   **[neuroguessr-0.2.apk](https://huggingface.co/josefbednar/neuroguessr-app/resolve/main/neuroguessr-0.2.apk?download=true)**
   (40 MB). Android will ask you to allow installs from your browser — allow it, install,
   and you can revoke the permission afterwards.
2. Open NeuroGuessr and tap **Download 2.20 GB**. Do it on Wi-Fi and keep the app open;
   it takes a few minutes. If it gets interrupted, reopening the app resumes where it stopped.
3. That's it. Pick a photo or take one, and it lands on the map.

## What's in here

- `neuroguessr-0.2.apk` — the app itself (Kotlin + Compose + ONNX Runtime).
- `assets/model/encoder_fp16.onnx` — a DINOv3 ViT-L/16 backbone, contrastively fine-tuned for
  place recognition, exported at fp16 with the residual stream kept fp32.
- `assets/index/` — 299,518 places as int8-quantised descriptors in five learned spaces,
  cell-sorted for a gated search.
- `assets/download_manifest.json` — sizes and SHA-256 of every asset; the copy baked into the
  APK is what the in-app downloader verifies against.

Median error on the held-out validation set: **44.4 km**, with 43% of guesses within 25 km.

## Licences

The encoder is a derivative of `facebook/dinov3-vitl16-pretrain-lvd1689m` and is distributed
under the [DINOv3 licence](DINOV3_LICENSE.md), a copy of which ships in this repo. Third-party
notices are in [NOTICE.md](NOTICE.md). Source and asset pipeline:
the app is built from the NeuroGuessr mobile repo; assets regenerate from the research repo.
