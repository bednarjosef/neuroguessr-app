#!/usr/bin/env python3
"""Fit the confidence number the app shows next to a prediction.

Target: P(the answer is within 25 km of the truth). 25 km is the app's natural notion of
"got it" — it is the threshold the reranker itself was trained against, and at that radius a
guess is useful rather than merely regional.

Every input is something the phone already computes while answering, so displaying the number
costs nothing at inference time. Reported with 5-fold cross-validation, because a confidence
that was fit on the same queries it is scored on would be exactly the kind of lie this feature
exists to prevent.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval.train_place_head import PlaceHead  # noqa: E402
from fit_e7 import cat, features, hav, soft, K  # noqa: E402

ASSETS = os.path.join(REPO, "mobile", "assets_build")
IX = os.path.join(REPO, "run_c4_index")
HIT_KM = 25.0

# Order is the contract with Retrieval.kt — do not reorder without changing both.
FEAT_NAMES = ["agree10", "agree25", "agree100", "top_sim", "margin", "entropy", "log_cand"]


def collect():
    torch.set_num_threads(os.cpu_count())
    man = json.load(open(os.path.join(ASSETS, "manifest.json")))
    e7 = np.load(os.path.join(ASSETS, "params", "e7.npz"))
    N = man["n_rows"]
    ixd = os.path.join(ASSETS, "index")
    latlon = np.memmap(os.path.join(ixd, "latlon.f32"), np.float32, "r").reshape(N, 2)
    cell_id = np.memmap(os.path.join(ixd, "cell_id.u16"), np.uint16, "r")
    offs = np.fromfile(os.path.join(ixd, "cell_offsets.u32"), np.uint32)
    csls = np.memmap(os.path.join(ixd, "csls_r.f32"), np.float32, "r").reshape(N, 3)
    vec = {s: np.memmap(os.path.join(ixd, f"{s}.i8"), np.int8, "r").reshape(N, 1024)
           for s in man["spaces"]}
    scl = {s: np.fromfile(os.path.join(ixd, f"{s}.scale.f32"), np.float32) for s in man["spaces"]}

    va = F.normalize(torch.from_numpy(cat(IX, "c2_cls_val_c4_r")).float(), dim=-1)
    q = {"cls": va.numpy()}
    pdir = os.path.join(ASSETS, "params")
    for nm in ("geo5c", "geo10c", "geo25c"):
        z = np.load(os.path.join(pdir, f"place_head_{nm}.npz"))
        h = PlaceHead(int(z["dim"]), int(z["hidden"])).eval()
        h.load_state_dict({k: torch.from_numpy(z[k]) for k in
                           ("ln.weight", "ln.bias", "fc1.weight", "fc1.bias",
                            "fc2.weight", "fc2.bias")})
        with torch.no_grad():
            q[nm] = h(va).numpy()
    wz = np.load(os.path.join(pdir, "whiten.npz"))
    q["cls_wh"] = F.normalize((va - torch.from_numpy(wz["mu"])) @ torch.from_numpy(wz["W"]),
                              dim=-1).numpy()

    lg = torch.from_numpy(cat(IX, "c2_logits_val_c4_r")).float()
    p = F.softmax(lg, dim=-1)
    logp = p.clamp(1e-12).log().double().numpy()
    order = torch.argsort(p, dim=1, descending=True).numpy()
    csum = torch.gather(p, 1, torch.from_numpy(order)).cumsum(1)
    m_of = ((csum < man["gate_mass"]).sum(1) + 1).numpy()
    vz = np.load(os.path.join(IX, "val_meta.npz"))
    tlat, tlon = vz["lat"].astype(np.float64), vz["lon"].astype(np.float64)
    n = len(tlat)

    ep_ids = np.zeros((n, K), np.int64)
    ep_s = np.full((n, K), -1e9, np.float64)
    ep_bb = np.zeros((n, K), np.float64)
    ncand = np.zeros(n)
    for i in range(n):
        m = min(int(m_of[i]), man["gate_cap"])
        cells = order[i, :m]
        lo, hi = offs[cells], offs[cells + 1]
        ids = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi) if h > l])
        ncand[i] = len(ids)
        s = np.zeros(len(ids), np.float64)
        for sp, wt in man["blend"].items():
            s += wt * ((vec[sp][ids].astype(np.float32) @ q[sp][i]) * scl[sp][ids]).astype(np.float64)
        s *= 2.0
        for j, sp in enumerate(man["csls_spaces"]):
            s -= man["blend"][sp] * csls[ids, j]
        s += man["prior_lam"] * logp[i, cell_id[ids]]
        k = min(K, len(ids))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        ep_ids[i, :k] = ids[top]; ep_s[i, :k] = s[top]
        ep_bb[i, :k] = (vec["cls"][ids[top]].astype(np.float32) @ q["cls"][i]) * scl["cls"][ids[top]]
        if i and i % 1000 == 0:
            print(f"  scored {i}/{n}", flush=True)

    la, lo_ = latlon[ep_ids, 0].astype(np.float64), latlon[ep_ids, 1].astype(np.float64)
    valid = ep_s > -1e8
    d_kk = np.zeros((n, K, K), np.float32)
    cc = np.zeros((n, K, K), np.float32)
    for i in range(n):
        d_kk[i] = hav(la[i][:, None], lo_[i][:, None], la[i][None, :], lo_[i][None, :])
        v = vec["cls"][ep_ids[i]].astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        cc[i] = v @ v.T
    Ff = features(ep_s, ep_bb, valid, d_kk, cc)
    Fn = (Ff - e7["feat_mu"]) / e7["feat_sd"]
    sc = Fn @ e7["w"] + float(e7["b"])
    sc[~valid] = -1e9
    final = ep_s / float(e7["score_sd"]) + float(e7["lam"]) * sc
    j = final.argmax(1)
    arn = np.arange(n)
    err = hav(la[arn, j], lo_[arn, j], tlat, tlon)

    w = soft(ep_s)
    fs = np.stack([
        ((d_kk[arn, j] <= 10.0) * w).sum(1),
        ((d_kk[arn, j] <= 25.0) * w).sum(1),
        ((d_kk[arn, j] <= 100.0) * w).sum(1),
        ep_bb[arn, j],
        np.sort(final, axis=1)[:, -1] - np.sort(final, axis=1)[:, -2],
        -(w * np.log(np.clip(w, 1e-12, None))).sum(1),
        np.log1p(ncand),
    ], axis=1)
    return fs, err


def fit_logistic(X, y, iters=900, lr=0.35, l2=1e-4):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-(X @ w + b)))
        g = pr - y
        w -= lr * ((X.T @ g) / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b


def main():
    fs, err = collect()
    n = len(err)
    y = (err <= HIT_KM).astype(np.float64)
    mu, sd = fs.mean(0), fs.std(0) + 1e-9
    X = (fs - mu) / sd

    folds = np.arange(n) % 5
    oof = np.zeros(n)
    for f in range(5):
        tr = folds != f
        w, b = fit_logistic(X[tr], y[tr])
        oof[~tr] = 1 / (1 + np.exp(-(X[~tr] @ w + b)))

    brier = float(((oof - y) ** 2).mean())
    o = np.argsort(oof)
    auc_pos, auc_neg = oof[y == 1], oof[y == 0]
    auc = float((auc_pos[:, None] > auc_neg[None, :]).mean())
    print(f"\nn={n} | base rate within {HIT_KM:.0f} km = {100*y.mean():.1f}%")
    print(f"cross-validated Brier {brier:.4f} (base {float(((y.mean()-y)**2).mean()):.4f}) | AUC {auc:.3f}\n")

    print("calibration — does a stated confidence mean what it says?")
    print(f"{'stated':>12}  {'actual':>7}  {'n':>5}  {'median err':>11}")
    edges = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
    for a, b_ in zip(edges[:-1], edges[1:]):
        m = (oof >= a) & (oof < b_)
        if m.sum() < 20:
            continue
        print(f"{100*a:5.0f}-{100*b_:3.0f}%  {100*y[m].mean():6.1f}%  {m.sum():5d}  "
              f"{np.median(err[m]):8.1f} km")

    print("\nwhat the user would see:")
    for lo, hi, lab in ((0.75, 1.01, "high"), (0.45, 0.75, "medium"), (0.0, 0.45, "low")):
        m = (oof >= lo) & (oof < hi)
        if m.sum() == 0:
            continue
        print(f"  {lab:6s} ({100*lo:.0f}%+): {m.sum():4d} queries, median {np.median(err[m]):6.1f} km, "
              f"@25 {100*(err[m] <= 25).mean():5.1f}%, >1000km {100*(err[m] > 1000).mean():4.1f}%")

    w, b = fit_logistic(X, y)
    np.savez(os.path.join(ASSETS, "params", "confidence.npz"),
             w=w, b=np.float64(b), mu=mu, sd=sd, hit_km=np.float64(HIT_KM))
    print("\nweights:")
    for nm, wi in zip(FEAT_NAMES, w):
        print(f"  {nm:10s} {wi:+.3f}")
    print("\nconfidence.npz written")


if __name__ == "__main__":
    main()
