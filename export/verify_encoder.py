#!/usr/bin/env python3
"""Does the exported encoder reproduce the cached C4 descriptors on real val images?

The index was embedded on GPU under bf16 autocast, so even a perfect fp32 export lands at
cos ~0.99 rather than 1.0 — that is the ceiling. Any quantisation is judged against the
fp32 export, because a descriptor that drifts from the index is a descriptor that retrieves
the wrong place, and the km cost is invisible until you measure it.
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

IMG = 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def preprocess(path):
    """Byte-for-byte the embed_full/embed_c2 path: PIL BICUBIC to 384, /255, ImageNet norm."""
    img = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
    x = np.asarray(img, np.uint8).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return (x - MEAN) / STD


def cat(ix, prefix):
    fs = sorted([f for f in os.listdir(ix) if f.startswith(prefix)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(ix, f)) for f in fs])


def norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(REPO, "mobile", "assets_build"))
    ap.add_argument("--testdata", default=os.path.join(REPO, "mobile", "testdata"))
    ap.add_argument("--index-dir", default=os.path.join(REPO, "run_c4_index"))
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()

    import onnxruntime as ort
    paths = sorted([f for f in os.listdir(a.testdata) if f.endswith(".jpg")])
    X = np.concatenate([preprocess(os.path.join(a.testdata, p)) for p in paths])
    n = len(paths)
    ref_cls = norm(cat(a.index_dir, "c2_cls_val_c4_r")[:n].astype(np.float32))
    ref_lg = cat(a.index_dir, "c2_logits_val_c4_r")[:n].astype(np.float32)
    print(f"{n} val images | reference descriptors from the frozen C4 index\n")

    mdir = os.path.join(a.assets, "model")
    models = a.models or [f for f in ("encoder_fp32.onnx", "encoder_fp16.onnx",
                                      "encoder_int8.onnx", "encoder_int8_pc.onnx")
                          if os.path.exists(os.path.join(mdir, f))]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    base = None
    for m in models:
        p = os.path.join(mdir, m)
        sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        xi = X.astype(np.float16) if inp.type == "tensor(float16)" else X
        t0 = time.time()
        outs = [sess.run(None, {inp.name: xi[i:i + 1]}) for i in range(n)]
        dt = (time.time() - t0) / n
        names = [o.name for o in sess.get_outputs()]
        got = {nm: np.concatenate([o[j] for o in outs]).astype(np.float32)
               for j, nm in enumerate(names)}
        cls = norm(got["cls"])
        cos_ref = float((cls * ref_cls).sum(1).mean())
        agree = float((got["logits"].argmax(1) == ref_lg.argmax(1)).mean())
        line = (f"{m:24s} {os.path.getsize(p)/1e6:6.0f} MB  {dt*1000:7.0f} ms/img  "
                f"cos-vs-index {cos_ref:.4f}  top-cell-agree {100*agree:5.1f}%")
        if base is None:
            base = got
        else:
            drift = " ".join(f"{k}:{float((norm(got[k])*norm(base[k])).sum(1).mean()):.4f}"
                             for k in ("cls", "q_geo10c", "q_cls_wh"))
            line += f"  vs-fp32[{drift}]"
        print(line, flush=True)


if __name__ == "__main__":
    main()
