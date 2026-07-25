#!/usr/bin/env python3
"""Export the C4 encoder to ONNX with **fp16 weights**, straight from PyTorch.

Two things had to be learned the hard way:

1. `onnxconverter_common.float16` cannot convert this graph. However the block list is drawn,
   the DINOv3 embeddings prologue (Conv -> Concat of cls/register tokens -> RoPE) ends up with
   one fp32 and one fp16 input on the same node and ORT refuses to load it.
2. A plain `model.half()` re-export **overflows**. DINOv3 ViT-L carries a massive-activation
   channel in the residual stream: measured peak |activation| is 1.57e5 on real images (fp16
   tops out at 65504), so every output comes back NaN. The model trained in bf16, which has
   fp32's exponent range — fp16 does not.

The fix is that the overflow is *only* in the residual stream. Probing every Linear/Conv on
real images: the largest value entering or leaving any of them is ~50. So this script keeps
the residual stream, the LayerNorms and the patch-embedding Conv in fp32 (a few MB of weights
between them) and swaps every `nn.Linear` — 99.5% of the parameters — for one holding fp16
weights that casts its activation down on the way in and back up on the way out. The result:
fp16 storage and fp16 GEMMs, fp32 accumulation path, no overflow anywhere.

    .venv/bin/python -u mobile/export/export_fp16.py
    .venv/bin/python -u mobile/export/verify_encoder.py
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_encoder import IMG, OUT_NAMES, build  # noqa: E402

T0 = time.time()
FP16_MAX = 65504.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


class HalfLinear(nn.Module):
    """nn.Linear with fp16 weights. Casts down at the input, back up at the output, so the
    residual stream it feeds stays fp32 and only the GEMM itself runs in half precision."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.weight = nn.Parameter(lin.weight.detach().half(), requires_grad=False)
        self.bias = (nn.Parameter(lin.bias.detach().half(), requires_grad=False)
                     if lin.bias is not None else None)

    def forward(self, x):
        return F.linear(x.to(torch.float16), self.weight, self.bias).to(torch.float32)


def halve_linears(model):
    """Swap every nn.Linear for a HalfLinear, in place. Returns (#swapped, #params moved)."""
    mods = dict(model.named_modules())
    n, p = 0, 0
    for name, m in list(mods.items()):
        if not isinstance(m, nn.Linear):
            continue
        parent = mods[name.rsplit(".", 1)[0]] if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], HalfLinear(m))
        n += 1
        p += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
    return n, p


def real_batch(testdata, k):
    """Same preprocessing as verify_encoder.py, so the range probe sees realistic inputs."""
    from PIL import Image
    ps = sorted(f for f in os.listdir(testdata) if f.endswith(".jpg"))[:k]
    xs = []
    for p in ps:
        im = Image.open(os.path.join(testdata, p)).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
        xs.append(np.asarray(im, np.uint8).transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    return torch.from_numpy((np.concatenate(xs) - MEAN) / STD).float(), ps


def probe_linear_range(model, x):
    """Peak |value| entering/leaving each Linear — anything above 65504 cannot go fp16."""
    worst = {"v": 0.0, "where": ""}
    over = []

    def hook(name):
        def f(_m, inp, out):
            ts = [t for t in (list(inp) + [out]) if torch.is_tensor(t) and t.is_floating_point]
            v = max((float(t.abs().max()) for t in ts if t.numel()), default=0.0)
            if v > worst["v"]:
                worst["v"], worst["where"] = v, name
            if v > FP16_MAX:
                over.append((name, v))
        return f

    hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules()
          if n and isinstance(m, nn.Linear)]
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return worst, over


def nrm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "run_c4_index", "ckpt.pt"))
    ap.add_argument("--assets", default=os.path.join(REPO, "mobile", "assets_build"))
    ap.add_argument("--testdata", default=os.path.join(REPO, "mobile", "testdata"))
    ap.add_argument("--nprobe", type=int, default=4)
    a = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    outdir = os.path.join(a.assets, "model")
    os.makedirs(outdir, exist_ok=True)
    fp16_path = os.path.join(outdir, "encoder_fp16.onnx")

    model = build(a.ckpt, os.path.join(a.assets, "params"))
    xr, names = real_batch(a.testdata, a.nprobe)
    log(f"probe batch {tuple(xr.shape)} from {names[0]}..{names[-1]}")

    worst, over = probe_linear_range(model, xr)
    log(f"fp32 peak |value| across all Linear in/out = {worst['v']:.1f} at '{worst['where']}' "
        f"({len(over)} of them exceed the fp16 max {FP16_MAX:.0f})")
    assert not over, f"cannot half these Linears: {over[:4]}"

    with torch.no_grad():
        ref32 = [t.float().numpy() for t in model(xr)]

    n, p = halve_linears(model)
    model.eval()
    log(f"{n} Linear layers -> fp16 weights ({p/1e6:.1f}M params, {p*2/1e6:.0f} MB); "
        "LayerNorm / patch Conv / whitening stay fp32")

    with torch.no_grad():
        eager16 = model(xr)
    bad = [nm for nm, t in zip(OUT_NAMES, eager16) if not torch.isfinite(t).all()]
    assert not bad, f"eager fp16 produced non-finite outputs in {bad}"
    log("eager fp16 finite; cos vs fp32 " + " ".join(
        f"{nm}:{float((nrm(t.float().numpy()) * nrm(r)).sum(-1).mean()):.5f}"
        for nm, t, r in zip(OUT_NAMES, eager16, ref32)))

    torch.onnx.export(
        model, (xr[:1],), fp16_path, input_names=["pixel_values"], output_names=OUT_NAMES,
        opset_version=17, do_constant_folding=True, dynamo=False)
    log(f"ONNX exported ({os.path.getsize(fp16_path)/1e6:.0f} MB)")

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(fp16_path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    xi = xr.numpy().astype(np.float16) if inp.type == "tensor(float16)" else xr.numpy()
    got = [np.concatenate(o).astype(np.float32) for o in
           zip(*[sess.run(None, {inp.name: xi[i:i + 1]}) for i in range(len(xi))])]

    print("\n  fp16 ONNX vs fp32 eager (descriptor drift is what costs km):")
    ok = True
    for nm, g, r in zip(OUT_NAMES, got, ref32):
        finite = bool(np.isfinite(g).all())
        cos = float((nrm(g) * nrm(r)).sum(-1).mean())
        ok &= finite and cos > 0.999
        print(f"  {nm:10s} cos {cos:.6f}  max|d| {np.abs(g-r).max():.3e}  finite {finite}")
    print()
    assert ok, "fp16 ONNX drifts too far from fp32 — do not ship this graph"
    log("DONE — now run mobile/export/verify_encoder.py")


if __name__ == "__main__":
    main()
