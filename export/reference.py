#!/usr/bin/env python3
"""Desktop reference implementation of the on-device pipeline.

Reads the same packed assets the phone reads and runs the same arithmetic, so any
discrepancy between this and `selftest.json` is a bug in the Kotlin port — not a mystery.
"""
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

EARTH = 6371.0088
IMG = 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, map(lambda v: np.asarray(v, np.float64), (a1, o1, a2, o2)))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
    x = np.asarray(img, np.uint8).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return (x - MEAN) / STD


class Reference:
    def __init__(self, assets, model=None):
        self.cfg = json.load(open(os.path.join(assets, "manifest.json")))
        e7 = np.load(os.path.join(assets, "params", "e7.npz"))
        self.w7, self.b7 = e7["w"], float(e7["b"])
        self.fmu, self.fsd = e7["feat_mu"], e7["feat_sd"]
        self.score_sd, self.lam = float(e7["score_sd"]), float(e7["lam"])

        ixd = os.path.join(assets, "index")
        N = self.cfg["n_rows"]
        self.N = N
        self.latlon = np.memmap(os.path.join(ixd, "latlon.f32"), np.float32, "r").reshape(N, 2)
        self.cell_id = np.memmap(os.path.join(ixd, "cell_id.u16"), np.uint16, "r")
        self.offs = np.fromfile(os.path.join(ixd, "cell_offsets.u32"), np.uint32)
        self.csls = np.memmap(os.path.join(ixd, "csls_r.f32"), np.float32, "r").reshape(N, 3)
        self.vec = {s: np.memmap(os.path.join(ixd, f"{s}.i8"), np.int8, "r").reshape(N, 1024)
                    for s in self.cfg["spaces"]}
        self.scl = {s: np.fromfile(os.path.join(ixd, f"{s}.scale.f32"), np.float32)
                    for s in self.cfg["spaces"]}

        model = model or os.path.join(assets, "model", "encoder_fp32.onnx")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.onames = [o.name for o in self.sess.get_outputs()]

    def encode(self, path):
        out = self.sess.run(None, {self.inp: preprocess(path)})
        d = dict(zip(self.onames, out))
        q = {"cls": d["cls"][0]}
        for k, v in d.items():
            if k.startswith("q_"):
                q[k[2:]] = v[0]
        return q, d["logits"][0]

    def locate(self, path):
        q, logits = self.encode(path)
        lg = logits.astype(np.float64)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        order = np.argsort(-p)
        csum = np.cumsum(p[order])
        m = min(int((csum < self.cfg["gate_mass"]).sum() + 1), self.cfg["gate_cap"])
        cells = order[:m]

        lo, hi = self.offs[cells], self.offs[cells + 1]
        ids = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi) if h > l])
        s = np.zeros(len(ids), np.float64)
        for sp, wt in self.cfg["blend"].items():
            d = self.vec[sp][ids].astype(np.float32) @ q[sp]
            s += wt * (d * self.scl[sp][ids]).astype(np.float64)
        s *= 2.0
        for j, sp in enumerate(self.cfg["csls_spaces"]):
            s -= self.cfg["blend"][sp] * self.csls[ids, j]
        s += self.cfg["prior_lam"] * np.log(np.clip(p[self.cell_id[ids]], 1e-12, None))

        K = min(self.cfg["top_k"], len(ids))
        top = np.argpartition(-s, K - 1)[:K]
        top = top[np.argsort(-s[top])]
        rows = ids[top]
        sK = s[top]
        lat = self.latlon[rows, 0].astype(np.float64)
        lon = self.latlon[rows, 1].astype(np.float64)

        bb = (self.vec["cls"][rows].astype(np.float32) @ q["cls"]) * self.scl["cls"][rows]
        v = self.vec["cls"][rows].astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        cc = v @ v.T
        dkk = hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])

        w = np.exp((sK - sK.max()) / 0.05)
        w /= w.sum() + 1e-12
        A = np.where(cc > 0.5, cc, 0)
        A = A / (A.sum(1, keepdims=True) + 1e-12)
        diffu = A @ w
        cons10 = ((dkk <= 10) * w[None, :]).sum(1)
        cons25 = ((dkk <= 25) * w[None, :]).sum(1)
        cons50 = ((dkk <= 50) * w[None, :]).sum(1)
        ccmean = (cc * w[None, :]).sum(1)
        ccmax3 = np.sort(cc, axis=1)[:, -4:-1].mean(1)
        margin = sK[0] - sK
        ent = float(-(w * np.log(np.clip(w, 1e-12, None))).sum())
        nvalid = K / self.cfg["top_k"]

        F = np.stack([sK, bb, np.log1p(np.arange(K)), margin, cons10, cons25, cons50,
                      diffu * 10, ccmean, ccmax3, np.full(K, ent), np.full(K, nvalid),
                      sK * cons25, bb * cons25, margin * cons25, diffu * cons25 * 10], 1)
        sc = ((F - self.fmu) / self.fsd) @ self.w7 + self.b7
        final = sK / self.score_sd + self.lam * sc
        j = int(final.argmax())
        return {"lat": float(lat[j]), "lon": float(lon[j]), "row": int(rows[j]),
                "gate_cells": int(m), "n_candidates": int(len(ids)),
                "top_rows": [int(r) for r in rows[np.argsort(-final)][:5]]}
