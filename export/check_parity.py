#!/usr/bin/env python3
"""Does the phone compute the same answer as the desktop reference?

Runs the reference pipeline on the same images the device ran in its self-test, and compares
predictions three ways: phone vs desktop (a port bug shows up here), and each against the
ground-truth coordinates (so a parity pass on a broken pipeline can't look like success).
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference import Reference, hav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(REPO, "mobile", "assets_build"))
    ap.add_argument("--testdata", default=os.path.join(REPO, "mobile", "testdata"))
    ap.add_argument("--phone", default=os.path.join(REPO, "mobile", "logs", "selftest.json"))
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    truth = np.load(os.path.join(a.testdata, "val_truth.npz"))
    tlat, tlon = truth["lat"], truth["lon"]
    phone = {r["file"]: r for r in json.load(open(a.phone))} if os.path.exists(a.phone) else {}
    if not phone:
        print(f"no phone results at {a.phone} — comparing desktop against ground truth only\n")

    ref = Reference(a.assets, a.model)
    files = sorted(f for f in os.listdir(a.testdata) if f.endswith(".jpg"))
    rows = []
    for f in files:
        i = int(f.split("_")[1].split(".")[0])
        d = ref.locate(os.path.join(a.testdata, f))
        rec = {"file": f, "d_desktop": float(hav(d["lat"], d["lon"], tlat[i], tlon[i]))}
        p = phone.get(f)
        if p:
            rec["d_phone"] = float(hav(p["lat"], p["lon"], tlat[i], tlon[i]))
            rec["gap_km"] = float(hav(d["lat"], d["lon"], p["lat"], p["lon"]))
            rec["same_row"] = (p.get("top_rows") or [None])[0] == d["top_rows"][0]
            rec["ms"] = p.get("ms_encode", 0) + p.get("ms_score", 0) + p.get("ms_rerank", 0)
        rows.append(rec)
        print(f"  {f}  desktop {rec['d_desktop']:9.1f} km" +
              (f" | phone {rec['d_phone']:9.1f} km | gap {rec['gap_km']:8.2f} km"
               f" | same_top1 {rec['same_row']}" if p else ""), flush=True)

    dd = np.array([r["d_desktop"] for r in rows])
    print(f"\ndesktop: median {np.median(dd):.2f} km | @25 {100*(dd<=25).mean():.1f}% (n={len(dd)})")
    if phone:
        dp = np.array([r["d_phone"] for r in rows if "d_phone" in r])
        gap = np.array([r["gap_km"] for r in rows if "gap_km" in r])
        same = sum(1 for r in rows if r.get("same_row"))
        ms = np.array([r["ms"] for r in rows if "ms" in r])
        print(f"phone:   median {np.median(dp):.2f} km | @25 {100*(dp<=25).mean():.1f}%")
        print(f"\nPARITY: identical top-1 on {same}/{len(gap)} images | "
              f"max gap {gap.max():.3f} km | mean gap {gap.mean():.3f} km")
        print(f"on-device latency: median {np.median(ms):.0f} ms/image")
        if same == len(gap):
            print("VERDICT: phone and desktop agree exactly.")
        elif gap.max() < 1.0:
            print("VERDICT: agree within 1 km — float ordering noise on near-tied candidates.")
        else:
            print("VERDICT: MISMATCH — the Kotlin port diverges from the reference.")


if __name__ == "__main__":
    main()
