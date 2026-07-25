#!/usr/bin/env python3
"""Is the app's own error predictable from signals it already has at inference time?

The app currently states a coordinate to five decimals with no indication of whether it is
worth believing. This asks whether cheap in-flight signals — how tightly the top candidates
agree geographically, how peaked the score distribution is, how strong the best match is —
separate the good answers from the bad ones. If they do, the app can show honest confidence
instead of implying uniform precision.
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


def main():
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
    for i in range(n):
        m = min(int(m_of[i]), man["gate_cap"])
        cells = order[i, :m]
        lo, hi = offs[cells], offs[cells + 1]
        ids = np.concatenate([np.arange(l, h) for l, h in zip(lo, hi) if h > l])
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
        if i and i % 750 == 0:
            print(f"  {i}/{n}", flush=True)

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

    # candidate signals available on the phone at no extra cost
    w = soft(ep_s)
    agree25 = ((d_kk[arn, j] <= 25) * w).sum(1)          # mass agreeing with the winner
    top_sim = ep_bb[arn, j]                              # raw similarity of the winner
    ent = -(w * np.log(np.clip(w, 1e-12, None))).sum(1)

    print(f"\nn={n}  median {np.median(err):.1f} km  @25 {100*(err<=25).mean():.1f}%\n")
    for name, sig, hi_is_good in (("geo agreement (mass<=25km)", agree25, True),
                                  ("winner similarity", top_sim, True),
                                  ("score entropy", ent, False)):
        o = np.argsort(-sig if hi_is_good else sig)
        print(f"{name}:")
        for lab, sl in (("best 25%", o[:n // 4]), ("mid 50%", o[n // 4:3 * n // 4]),
                        ("worst 25%", o[3 * n // 4:])):
            e = err[sl]
            print(f"   {lab:9s} median {np.median(e):7.1f} km   @25 {100*(e<=25).mean():5.1f}%"
                  f"   >1000km {100*(e>1000).mean():5.1f}%")
        print()
    np.savez(os.path.join(ASSETS, "params", "confidence_probe.npz"),
             err=err, agree25=agree25, top_sim=top_sim, ent=ent)


if __name__ == "__main__":
    main()
