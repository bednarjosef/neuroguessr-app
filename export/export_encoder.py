#!/usr/bin/env python3
"""Export the C4 encoder (DINOv3 ViT-L/16 + merged LoRA + geo heads) to ONNX for Android.

One graph, six outputs — everything the on-device retrieval needs from a single forward:
    cls      (1,1024)  L2-normalised backbone CLS  -> E7 consensus features
    logits   (1,5000)  combined classifier logits  -> the gate
    q_geo5c  (1,1024)  band head @ 5 km            -> scoring space
    q_geo10c (1,1024)  band head @ 10 km           -> scoring space
    q_geo25c (1,1024)  band head @ 25 km           -> scoring space
    q_cls_wh (1,1024)  PCA-whitened CLS            -> scoring space

Input is a normalised float32 NCHW tensor (ImageNet mean/std), 384x384 — the app does the
resize + normalisation so the graph stays free of image ops.

Verified against the eager model on random input, then dynamically quantised to int8.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from retrieval.embed_full import GeoModelEval  # noqa: E402
from retrieval.train_place_head import PlaceHead  # noqa: E402

IMG = 384
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


class MobileGeo(nn.Module):
    """Eager reference AND the ONNX export target — identical math either way."""

    def __init__(self, base: GeoModelEval, heads: dict, mu: torch.Tensor, W: torch.Tensor):
        super().__init__()
        self.base = base
        self.h5, self.h10, self.h25 = heads["geo5c"], heads["geo10c"], heads["geo25c"]
        self.register_buffer("mu", mu)
        self.register_buffer("W", W)

    def forward(self, pixel_values):
        feats = self.base.features(pixel_values)          # (B,1024) CLS after final norm
        _, logits, _ = self.base.heads(feats)             # (B,5000) combined gate logits
        cls = F.normalize(feats, dim=-1)
        q_wh = F.normalize((cls - self.mu) @ self.W, dim=-1)
        return cls, logits, self.h5(cls), self.h10(cls), self.h25(cls), q_wh


OUT_NAMES = ["cls", "logits", "q_geo5c", "q_geo10c", "q_geo25c", "q_cls_wh"]


def build(ckpt_path, params_dir):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base = GeoModelEval(ck)
    base.load_from_ckpt(ck, strict=False)
    base.eval()
    log(f"base model loaded (step {ck.get('step')}, best {ck.get('best_median', float('nan')):.2f} km)")

    # Merge LoRA into the frozen backbone so the phone runs plain linear layers. The merge is
    # in-place on the same submodules, but it drops the PEFT wrapper that _core()'s attribute
    # walk was written against — so pin _core to the object we resolved before unwrapping.
    core = base._core()
    assert hasattr(core, "embeddings"), type(core)
    merged = base.backbone.merge_and_unload()
    base.backbone = merged
    base._core = lambda: core
    log("LoRA merged into backbone")

    heads = {}
    for nm in ("geo5c", "geo10c", "geo25c"):
        z = np.load(os.path.join(params_dir, f"place_head_{nm}.npz"))
        h = PlaceHead(int(z["dim"]), int(z["hidden"])).eval()
        h.load_state_dict({k: torch.from_numpy(z[k]) for k in
                           ("ln.weight", "ln.bias", "fc1.weight", "fc1.bias",
                            "fc2.weight", "fc2.bias")})
        heads[nm] = h
    wz = np.load(os.path.join(params_dir, "whiten.npz"))
    model = MobileGeo(base, heads, torch.from_numpy(wz["mu"]), torch.from_numpy(wz["W"])).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "run_c4_index", "ckpt.pt"))
    ap.add_argument("--assets", default=os.path.join(REPO, "mobile", "assets_build"))
    ap.add_argument("--skip-quant", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    outdir = os.path.join(a.assets, "model")
    os.makedirs(outdir, exist_ok=True)
    model = build(a.ckpt, os.path.join(a.assets, "params"))

    x = torch.randn(1, 3, IMG, IMG)
    with torch.no_grad():
        ref = model(x)
    log(f"eager forward OK — cls|{ref[0].shape} logits|{ref[1].shape} "
        f"(cls norm {ref[0].norm():.4f}, top cell {int(ref[1].argmax())})")

    fp32 = os.path.join(outdir, "encoder_fp32.onnx")
    torch.onnx.export(
        model, (x,), fp32, input_names=["pixel_values"], output_names=OUT_NAMES,
        opset_version=17, do_constant_folding=True, dynamo=False)
    log(f"ONNX exported ({os.path.getsize(fp32)/1e6:.0f} MB)")

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(fp32, so, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"pixel_values": x.numpy()})
    print()
    ok = True
    for nm, g, r in zip(OUT_NAMES, got, ref):
        r = r.numpy()
        cos = float((g.ravel() @ r.ravel()) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-12))
        mad = float(np.abs(g - r).max())
        ok &= cos > 0.9999
        print(f"  {nm:10s} cos {cos:.6f}  max|d| {mad:.3e}")
    print()
    assert ok, "ONNX/eager mismatch — do not ship this graph"
    log("fp32 ONNX matches eager")

    if a.skip_quant:
        return
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8 = os.path.join(outdir, "encoder_int8.onnx")
    # MatMul only: quantising the patch-embedding Conv yields ConvInteger, which no mobile
    # execution provider implements, and it is 0.3% of the weights anyway.
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"],
                     extra_options={"MatMulConstBOnly": True})
    log(f"int8 ONNX written ({os.path.getsize(int8)/1e6:.0f} MB)")

    sess8 = ort.InferenceSession(int8, so, providers=["CPUExecutionProvider"])
    g8 = sess8.run(None, {"pixel_values": x.numpy()})
    print("\n  int8 vs fp32 (descriptor drift is what costs km):")
    for nm, g, r in zip(OUT_NAMES, g8, got):
        cos = float((g.ravel() @ r.ravel()) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-12))
        print(f"  {nm:10s} cos {cos:.6f}")
    log("DONE")


if __name__ == "__main__":
    main()
