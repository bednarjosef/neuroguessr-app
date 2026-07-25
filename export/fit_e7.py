#!/usr/bin/env python3
"""Score the packed mobile index on full val, and fit the E7 reranker that ships with the app.

This runs the EXACT arithmetic the phone will run — same int8 index, same gate, same blend,
same CSLS correction, same features — so the median it prints is the accuracy of the app,
not of the research bed. Anything the quantisation or the 1-view reduction costs shows up
here as km.

Reports 5-fold CV numbers (honest) and ships weights fit on all queries (best estimate).
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

EARTH = 6371.0088
K = 100
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, map(lambda v: np.asarray(v, np.float64), (a1, o1, a2, o2)))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def rep(tag, d):
    print(f"{tag:30s} median {np.median(d):7.2f}  mean {d.mean():6.0f}  "
          f"@1 {100*(d<=1).mean():5.2f}%  @25 {100*(d<=25).mean():5.2f}%  "
          f"@200 {100*(d<=200).mean():5.2f}%  GG {(5000*np.exp(-d/1492.7)).mean():5.0f}", flush=True)


def cat(ix, prefix):
    fs = sorted([f for f in os.listdir(ix) if f.startswith(prefix)],
                key=lambda f: int(f.rsplit("_r", 1)[1].split(".")[0]))
    return np.concatenate([np.load(os.path.join(ix, f)) for f in fs])


def soft(s, tau=0.05):
    w = np.exp((s - s.max(1, keepdims=True)) / tau) * (s > -1e8)
    return w / (w.sum(1, keepdims=True) + 1e-12)


def features(s_k, bb_k, valid, d_kk, cc):
    """The 16 E7 features. Kept verbatim from the C4 bed so the app inherits its validation."""
    n = s_k.shape[0]
    w = soft(s_k)
    A = np.where(cc > 0.5, cc, 0)
    A = A / (A.sum(2, keepdims=True) + 1e-12)
    diffu = np.einsum('nkj,nj->nk', A, w)
    cons10 = ((d_kk <= 10) * w[:, None, :]).sum(2)
    cons25 = ((d_kk <= 25) * w[:, None, :]).sum(2)
    cons50 = ((d_kk <= 50) * w[:, None, :]).sum(2)
    ccmean = (cc * w[:, None, :]).sum(2)
    ccmax3 = np.sort(cc, axis=2)[:, :, -4:-1].mean(2)
    rank_pos = np.tile(np.arange(K, dtype=np.float64), (n, 1))
    margin = s_k[:, :1] - s_k
    entf = -(w * np.log(np.clip(w, 1e-12, None))).sum(1, keepdims=True) * np.ones((1, K))
    nvalid = valid.sum(1, keepdims=True) * np.ones((1, K)) / K
    return np.stack([s_k, bb_k, np.log1p(rank_pos), margin, cons10, cons25, cons50,
                     diffu * 10, ccmean, ccmax3, entf, nvalid,
                     s_k * cons25, bb_k * cons25, margin * cons25, diffu * cons25 * 10], axis=2)


def fit_logistic(X, y, iters=400, lr=0.5):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ g) / len(y)
        b -= lr * g.mean()
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(REPO, "mobile", "assets_build"))
    ap.add_argument("--index-dir", default=os.path.join(REPO, "run_c4_index"))
    ap.add_argument("--limit", type=int, default=0, help="score only the first N queries")
    a = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    ixd = os.path.join(a.assets, "index")
    pdir = os.path.join(a.assets, "params")
    man = json.load(open(os.path.join(a.assets, "manifest.json")))
    N, C = man["n_rows"], man["n_cells"]
    blend, csls_spaces = man["blend"], man["csls_spaces"]

    # ---- packed index (exactly what the phone mmaps) ---------------------------------
    latlon = np.memmap(os.path.join(ixd, "latlon.f32"), np.float32, "r").reshape(N, 2)
    cell_id = np.memmap(os.path.join(ixd, "cell_id.u16"), np.uint16, "r")
    offs = np.fromfile(os.path.join(ixd, "cell_offsets.u32"), np.uint32)
    csls = np.memmap(os.path.join(ixd, "csls_r.f32"), np.float32, "r").reshape(N, 3)
    SPACES = ["cls", "geo5c", "geo10c", "geo25c", "cls_wh"]
    vec = {s: np.memmap(os.path.join(ixd, f"{s}.i8"), np.int8, "r").reshape(N, 1024) for s in SPACES}
    scl = {s: np.fromfile(os.path.join(ixd, f"{s}.scale.f32"), np.float32) for s in SPACES}
    log(f"index mapped: {N} rows, {C} cells, {man['views_per_location']} view(s)/location")

    # ---- query side: cached val descriptors pushed through the same heads -------------
    va = F.normalize(torch.from_numpy(cat(a.index_dir, "c2_cls_val_c4_r")).float(), dim=-1)
    q = {"cls": va.numpy()}
    for nm in ("geo5c", "geo10c", "geo25c"):
        z = np.load(os.path.join(pdir, f"place_head_{nm}.npz"))
        h = PlaceHead(int(z["dim"]), int(z["hidden"])).eval()
        h.load_state_dict({k: torch.from_numpy(z[k]) for k in
                           ("ln.weight", "ln.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")})
        with torch.no_grad():
            q[nm] = h(va).numpy()
    wz = np.load(os.path.join(pdir, "whiten.npz"))
    q["cls_wh"] = F.normalize((va - torch.from_numpy(wz["mu"])) @ torch.from_numpy(wz["W"]),
                              dim=-1).numpy()

    lg = torch.from_numpy(cat(a.index_dir, "c2_logits_val_c4_r")).float()
    p = F.softmax(lg, dim=-1)
    logp = p.clamp(1e-12).log().double().numpy()
    rank_order = torch.argsort(p, dim=1, descending=True).numpy()
    csum = torch.gather(p, 1, torch.from_numpy(rank_order)).cumsum(1)
    m_of = ((csum < man["gate_mass"]).sum(1) + 1).numpy()

    vz = np.load(os.path.join(a.index_dir, "val_meta.npz"))
    tlat, tlon = vz["lat"].astype(np.float64), vz["lon"].astype(np.float64)
    n = a.limit or len(tlat)
    log(f"queries ready: {n} (median gate {int(np.median(m_of))} cells)")

    # ---- gate + blend + CSLS, per query ----------------------------------------------
    ep_ids = np.zeros((n, K), np.int64)
    ep_s = np.full((n, K), -1e9, np.float64)
    ep_bb = np.zeros((n, K), np.float64)
    base_pred = np.zeros((n, 2))
    ncand = np.zeros(n)

    for i in range(n):
        m = min(int(m_of[i]), man["gate_cap"])
        cells = rank_order[i, :m]
        lo, hi = offs[cells], offs[cells + 1]
        ids = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi) if h > l])
        ncand[i] = len(ids)
        s = np.zeros(len(ids), np.float64)
        for sp, wt in blend.items():
            d = vec[sp][ids].astype(np.float32) @ q[sp][i]
            s += wt * (d * scl[sp][ids]).astype(np.float64)
        corr = np.zeros(len(ids), np.float64)
        for j, sp in enumerate(csls_spaces):
            corr += blend[sp] * csls[ids, j]
        s = 2 * s - corr
        s += man["prior_lam"] * logp[i, cell_id[ids]]
        k = min(K, len(ids))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        tid = ids[top]
        ep_ids[i, :k] = tid
        ep_s[i, :k] = s[top]
        ep_bb[i, :k] = (vec["cls"][tid].astype(np.float32) @ q["cls"][i]) * scl["cls"][tid]
        base_pred[i] = latlon[tid[0]]
        if i and i % 500 == 0:
            log(f"  scored {i}/{n} (cand~{ncand[:i].mean():.0f})")

    d_base = hav(base_pred[:, 0], base_pred[:, 1], tlat[:n], tlon[:n])
    print()
    rep("blend+CSLS+wh (no rerank)", d_base)

    # ---- E7 features -----------------------------------------------------------------
    la_k, lo_k = latlon[ep_ids, 0].astype(np.float64), latlon[ep_ids, 1].astype(np.float64)
    valid = ep_s > -1e8
    d_kk = np.zeros((n, K, K), np.float32)
    cc = np.zeros((n, K, K), np.float32)
    for i in range(n):
        d_kk[i] = hav(la_k[i][:, None], lo_k[i][:, None], la_k[i][None, :], lo_k[i][None, :])
        v = vec["cls"][ep_ids[i]].astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        cc[i] = v @ v.T
    Ff = features(ep_s, ep_bb, valid, d_kk, cc)
    fmu = Ff.reshape(-1, Ff.shape[2]).mean(0)
    fsd = Ff.reshape(-1, Ff.shape[2]).std(0) + 1e-9
    Fn = (Ff - fmu) / fsd
    d_all = hav(la_k, lo_k, tlat[:n, None], tlon[:n, None])
    y = (d_all <= 25.0).astype(np.float64)
    log("E7 features built")

    # honest: 5-fold CV
    folds = np.arange(n) % 5
    sc = np.zeros((n, K))
    for f in range(5):
        tr = folds != f
        Xtr = Fn[tr].reshape(-1, Fn.shape[2])
        mm = valid[tr].ravel()
        w, b = fit_logistic(Xtr[mm], y[tr].ravel()[mm])
        sc[~tr] = (Fn[~tr].reshape(-1, Fn.shape[2]) @ w + b).reshape((~tr).sum(), K)
    sc[~valid] = -1e9
    sd = ep_s[valid].std()
    arn = np.arange(n)
    best = None
    for lam in (0.5, 1.0, 2.0):
        j = (ep_s / sd + lam * sc).argmax(1)
        d = hav(la_k[arn, j], lo_k[arn, j], tlat[:n], tlon[:n])
        rep(f"E7 rerank lam{lam} (5-fold CV)", d)
        if best is None or np.median(d) < best[1]:
            best = (lam, np.median(d))

    # ship: fit on everything
    w, b = fit_logistic(Fn.reshape(-1, Fn.shape[2])[valid.ravel()], y.ravel()[valid.ravel()])
    np.savez(os.path.join(pdir, "e7.npz"), w=w, b=np.float64(b), feat_mu=fmu, feat_sd=fsd,
             score_sd=np.float64(sd), lam=np.float64(best[0]))
    man["e7_lam"] = float(best[0])
    man["score_sd"] = float(sd)
    json.dump(man, open(os.path.join(a.assets, "manifest.json"), "w"), indent=2)
    log(f"e7.npz written (lam={best[0]}, CV median {best[1]:.2f} km) — DONE")


if __name__ == "__main__":
    main()
