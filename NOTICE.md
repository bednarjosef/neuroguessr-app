# Third-party components and licences

## DINOv3 (Meta Platforms)

The encoder is a fine-tuned derivative of `facebook/dinov3-vitl16-pretrain-lvd1689m`, covered by
the **DINOv3 Licence** (<https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md>).

Relevant terms, verified 2026-07-25:

- Grants a royalty-free, worldwide licence to use, reproduce, distribute, copy, create
  derivative works of, and modify the materials. Commercial use is permitted, and there is no
  monthly-active-user threshold or additional-commercial-terms clause.
- **If you distribute the materials or any derivative of them, you must distribute them under
  the terms of that same agreement and provide a copy of it.** Any release that includes the
  encoder weights must therefore ship the DINOv3 licence text alongside them.
- Prohibits use for military or warfare purposes, nuclear applications, espionage, or the
  development or use of guns or illegal weapons; requires compliance with ITAR and trade
  controls.
- Research published using the materials must acknowledge them.

Because it is a custom licence rather than a standard open-source one, get your own legal
reading before shipping commercially.

## Inter

Copyright © 2016 The Inter Project Authors (https://github.com/rsms/inter), with Reserved Font
Name "Inter". Licensed under the **SIL Open Font License 1.1** — freely bundleable in an
application, including commercially. Bundled in `android/app/src/main/res/font/` (Inter
Regular/Medium/SemiBold and Inter Display Medium/SemiBold).

## Natural Earth

The offline basemap (`world.bin`) is built from Natural Earth vector data, which is in the
**public domain**. Countries, coastline, lakes and populated places at 1:50m and 1:110m, plus
populated places at 1:10m. <https://www.naturalearthdata.com/>

## ONNX Runtime

MIT Licence, Microsoft. Consumed as the `com.microsoft.onnxruntime:onnxruntime-android`
artifact.

## AndroidX / Jetpack Compose / Kotlin

Apache License 2.0.

---

## The place index — not distributed

The retrieval index holds 1024-dimensional descriptors and coordinates derived from Street View
imagery. It contains no images, but it is a derivative database, and Google's terms speak to
derivative databases. It is therefore **not** included in this repository or in any release
attachment, and the app expects it to be provisioned separately by whoever builds it.

Anyone planning to distribute this app publicly should resolve that question first — it is a
licensing matter, not a technical one.
