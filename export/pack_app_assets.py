#!/usr/bin/env python3
"""Fold the manifest + fitted E7 weights into the single config.json the app reads.

The app must never carry its own copy of the recipe constants: they are written here from
the same artefacts the evaluation used, so an index rebuild can't silently desynchronise
from the weights that were fit against it.
"""
import argparse
import json
import os
import shutil

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(REPO, "mobile", "assets_build"))
    ap.add_argument("--android", default=os.path.join(REPO, "mobile", "android", "app",
                                                      "src", "main", "assets"))
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.assets, "manifest.json")))
    e7 = np.load(os.path.join(a.assets, "params", "e7.npz"))
    cfg = dict(man)
    cfg.update({
        "e7_w": e7["w"].tolist(),
        "e7_b": float(e7["b"]),
        "e7_feat_mu": e7["feat_mu"].tolist(),
        "e7_feat_sd": e7["feat_sd"].tolist(),
        "e7_lam": float(e7["lam"]),
        "score_sd": float(e7["score_sd"]),
    })
    cpath = os.path.join(a.assets, "params", "confidence.npz")
    if os.path.exists(cpath):
        c = np.load(cpath)
        cfg.update({
            "conf_w": c["w"].tolist(), "conf_b": float(c["b"]),
            "conf_mu": c["mu"].tolist(), "conf_sd": c["sd"].tolist(),
            "conf_hit_km": float(c["hit_km"]),
        })
    else:
        print("  (no confidence.npz — the app will hide the confidence readout)")

    os.makedirs(a.android, exist_ok=True)
    with open(os.path.join(a.android, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    mapdir = os.path.join(a.assets, "map")
    if os.path.isdir(mapdir):
        for f in os.listdir(mapdir):
            if f.endswith((".bin", ".txt")):
                shutil.copy(os.path.join(mapdir, f), os.path.join(a.android, f))
                print(f"  map asset: {f}")

    print(f"config.json written -> {a.android}")
    print(f"  {cfg['n_rows']} rows, {cfg['views_per_location']} view(s)/loc, "
          f"E7 lam {cfg['e7_lam']}, score_sd {cfg['score_sd']:.4f}")


if __name__ == "__main__":
    main()
