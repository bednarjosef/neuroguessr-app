#!/usr/bin/env python3
"""Pack the C4 retrieval bed into flat, mmap-friendly, int8 assets for the Android app.

Faithful to retrieval/local_evals/local_c4_eval.py (the run that produced the 34.10 km
champion): gate = classifier posterior mass .95 capped at 400 cells, scoring space blend
"4band+CSLS+wh" (geo5c/geo10c/geo25c .22 each + whitened CLS .34, CSLS 2s-r on the three
geo bands only), prior term 0.05*log p(cell), then E7 rerank over the top-100.

Everything is written in CELL-SORTED order so a gate of K cells is K contiguous ranges.

Outputs (default mobile/assets_build/index):
    latlon.f32          (N,2)    float32   candidate coordinates
    cell_id.u16         (N,)     uint16    cell of each row
    cell_offsets.u32    (C+1,)   uint32    row range of each cell
    <space>.i8          (N,1024) int8      symmetric per-row quantised descriptor
    <space>.scale.f32   (N,)     float32   per-row dequant scale
    csls_r.f32          (N,3)    float32   CSLS r() for geo5c/geo10c/geo25c
and params/{whiten.npz, place_head_*.npz} used by the encoder export + the app.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from retrieval.train_place_head import PlaceHead  # noqa: E402

T0 = time.time()
BANDS = (("geo5c", "geo5_cls_c4.pt"), ("geo10c", "geo10_cls_c4.pt"), ("geo25c", "geo25_cls_c4.pt"))
CSLS_SPACES = ("geo5c", "geo10c", "geo25c")
BLEND = {"geo5c": 0.22, "geo10c": 0.22, "geo25c": 0.22, "cls_wh": 0.34}
GATE_MASS, GATE_CAP, PRIOR_LAM = 0.95, 400, 0.05
WHITEN_SAMPLE, CSLS_QUERIES, CSLS_TOPK = 200_000, 5_000, 10
SEED = 1337


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def cat(ix, prefix):
    fs = sorted([f for f in os.listdir(ix) if f.startswith(prefix)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    assert fs, f"no {prefix}* in {ix}"
    return np.concatenate([np.load(os.path.join(ix, f)) for f in fs])


def quantise_rows(x, out_i8, out_scale, chunk=100_000):
    """Symmetric per-row int8. score = int_dot * scale_row * scale_query."""
    n = x.shape[0]
    for i in range(0, n, chunk):
        blk = x[i:i + chunk].float()
        s = blk.abs().amax(dim=1).clamp(min=1e-12) / 127.0
        out_i8[i:i + chunk] = torch.round(blk / s[:, None]).clamp(-127, 127).to(torch.int8).numpy()
        out_scale[i:i + chunk] = s.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=os.path.join(REPO, "run_c4_index"))
    ap.add_argument("--out", default=os.path.join(REPO, "mobile", "assets_build"))
    ap.add_argument("--views", type=int, default=4, choices=[1, 4],
                    help="4 = every image (1.2M rows); 1 = one view per location (300k rows)")
    ap.add_argument("--tag", default="c4")
    a = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    torch.manual_seed(SEED)
    ix = a.index_dir
    ixdir = os.path.join(a.out, "index")
    pdir = os.path.join(a.out, "params")
    os.makedirs(ixdir, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)

    # ---- coordinates + cell assignment (gate buckets) --------------------------------
    z = np.load(os.path.join(ix, "train_latlon.npz"))
    trlat, trlon = z["lat"].astype(np.float64), z["lon"].astype(np.float64)
    N_ALL = len(trlat)

    ck = torch.load(os.path.join(ix, "ckpt.pt"), map_location="cpu", weights_only=False)
    cent = F.normalize(ck["buffers"]["centroids"].float(), dim=-1)
    C = cent.shape[0]
    n_cells_cfg = int(ck["config"]["n_cells"])
    assert C == n_cells_cfg, (C, n_cells_cfg)
    del ck

    lar, lor = np.radians(trlat), np.radians(trlon)
    tru = torch.tensor(np.stack([np.cos(lar) * np.cos(lor), np.cos(lar) * np.sin(lor),
                                 np.sin(lar)], -1), dtype=torch.float32)
    cell_a = torch.empty(N_ALL, dtype=torch.long)
    for i in range(0, N_ALL, 200_000):
        cell_a[i:i + 200_000] = (tru[i:i + 200_000] @ cent.T).argmax(1)
    del tru
    log(f"cells assigned: N={N_ALL} C={C}")

    # ---- row selection (4 views = all rows; 1 view = first row of each location) ------
    if a.views == 4:
        keep = np.arange(N_ALL)
    else:
        # rows are grouped 4-per-location in parquet order; take view 0 of each
        assert N_ALL % 4 == 0, N_ALL
        keep = np.arange(0, N_ALL, 4)
    N = len(keep)

    # cell-sorted order, restricted to kept rows
    order = torch.argsort(cell_a[torch.from_numpy(keep)], stable=True).numpy()
    sel = keep[order]                       # original row ids in cell-sorted order
    cs = cell_a[torch.from_numpy(sel)].numpy().astype(np.int64)
    offsets = np.zeros(C + 1, np.uint32)
    offsets[1:] = np.cumsum(np.bincount(cs, minlength=C)).astype(np.uint32)

    np.stack([trlat[sel], trlon[sel]], 1).astype(np.float32).tofile(
        os.path.join(ixdir, "latlon.f32"))
    cs.astype(np.uint16).tofile(os.path.join(ixdir, "cell_id.u16"))
    offsets.tofile(os.path.join(ixdir, "cell_offsets.u32"))
    log(f"geometry written: {N} rows kept ({a.views} view(s)/location)")

    # ---- base CLS descriptors (normalised), in cell-sorted order ----------------------
    tr_cls = torch.from_numpy(cat(ix, f"c2_cls_train_{a.tag}_r"))
    assert tr_cls.shape[0] == N_ALL, (tr_cls.shape, N_ALL)
    for i in range(0, N_ALL, 200_000):
        tr_cls[i:i + 200_000] = F.normalize(tr_cls[i:i + 200_000].float(), dim=-1).half()
    tr_cls = tr_cls[torch.from_numpy(sel)].contiguous()
    log("cls normalised + reordered")

    def emit(name, mat):
        """Stream straight to a headerless flat file (mmap'd verbatim on Android)."""
        sc = np.empty(N, np.float32)
        chunk = 100_000
        with open(os.path.join(ixdir, f"{name}.i8"), "wb") as fh:
            for i in range(0, N, chunk):
                blk = mat[i:i + chunk].float()
                s = blk.abs().amax(dim=1).clamp(min=1e-12) / 127.0
                q = torch.round(blk / s[:, None]).clamp(-127, 127).to(torch.int8)
                fh.write(np.ascontiguousarray(q.numpy()).tobytes())
                sc[i:i + chunk] = s.numpy()
        sc.tofile(os.path.join(ixdir, f"{name}.scale.f32"))
        log(f"space {name} quantised -> {N*1024/1e9:.2f} GB")

    emit("cls", tr_cls)

    # ---- band heads -------------------------------------------------------------------
    csls_r = {}
    gen = torch.Generator().manual_seed(SEED)
    qidx = torch.randint(0, N, (CSLS_QUERIES,), generator=gen)

    def csls_of(mat):
        Q = mat[qidx].float()
        r = torch.empty(N, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, N, 50_000):
                sims = mat[i:i + 50_000].float() @ Q.T
                r[i:i + 50_000] = sims.topk(CSLS_TOPK, dim=1).values.mean(1)
                del sims
        return r

    csls_r["cls"] = csls_of(tr_cls)
    log(f"csls r() cls mean {csls_r['cls'].mean():.3f}")

    for nm, fname in BANDS:
        hck = torch.load(os.path.join(ix, fname), map_location="cpu", weights_only=False)
        head = PlaceHead(hck["dim"], hck["hidden"]).eval()
        head.load_state_dict(hck["state_dict"])
        np.savez(os.path.join(pdir, f"place_head_{nm}.npz"),
                 **{k: v.numpy() for k, v in head.state_dict().items()},
                 dim=np.int64(hck["dim"]), hidden=np.int64(hck["hidden"]))
        with torch.no_grad():
            out = torch.empty_like(tr_cls)
            for i in range(0, N, 100_000):
                out[i:i + 100_000] = head(tr_cls[i:i + 100_000].float()).half()
        log(f"band {nm} projected")
        csls_r[nm] = csls_of(out)
        log(f"csls r() {nm} mean {csls_r[nm].mean():.3f}")
        emit(nm, out)
        del out

    # ---- PCA whitening of CLS ----------------------------------------------------------
    idx = torch.randint(0, N, (min(WHITEN_SAMPLE, N),), generator=gen)
    X = tr_cls[idx].float()
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / X.shape[0]
    ev, Vv = torch.linalg.eigh(cov.double())
    W = (Vv / ev.clamp(min=1e-6).sqrt()).float()
    del X, Xc, cov, ev, Vv
    np.savez(os.path.join(pdir, "whiten.npz"), mu=mu.numpy(), W=W.numpy())
    out = torch.empty_like(tr_cls)
    for i in range(0, N, 200_000):
        out[i:i + 200_000] = F.normalize((tr_cls[i:i + 200_000].float() - mu) @ W, dim=-1).half()
    log("cls whitened")
    emit("cls_wh", out)
    del out, tr_cls

    np.stack([csls_r[s].numpy() for s in CSLS_SPACES], 1).astype(np.float32).tofile(
        os.path.join(ixdir, "csls_r.f32"))

    manifest = {
        "tag": a.tag, "n_rows": int(N), "n_cells": int(C), "dim": 1024,
        "views_per_location": a.views,
        "spaces": ["cls", "geo5c", "geo10c", "geo25c", "cls_wh"],
        "csls_spaces": list(CSLS_SPACES),
        "blend": BLEND, "gate_mass": GATE_MASS, "gate_cap": GATE_CAP,
        "prior_lam": PRIOR_LAM, "top_k": 100, "e7_lam": 2.0,
        "quant": "int8-symmetric-per-row",
        "bytes_index": int(N * 1024 * 5 + N * 4 * 5 + N * 8 + N * 2 + N * 12),
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"DONE — manifest written ({manifest['bytes_index']/1e9:.2f} GB of index)")


if __name__ == "__main__":
    main()
