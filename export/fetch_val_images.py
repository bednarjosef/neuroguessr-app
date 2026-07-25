#!/usr/bin/env python3
"""Pull the first N val images, byte-identically to how prepare.py stored them.

Row i here == row i of val_meta.npz == row i of cat(c2_cls_val_c4_r*), so these images are
ground truth for two things: the coordinates the app must predict, and the exact descriptors
the exported encoder has to reproduce.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from prepare import DATASET_NAME, SAVE_MAX_SIDE, _resize_for_storage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(REPO, "mobile", "testdata"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, split="val", streaming=True)
    rows = []
    for i, r in enumerate(ds):
        if i >= a.n:
            break
        img = r["image"] if "image" in r else r["jpg"]
        p = os.path.join(a.out, f"val_{i:04d}.jpg")
        _resize_for_storage(img).save(p, "JPEG", quality=90)
        rows.append((i, float(r["latitude"]), float(r["longitude"])))
        print(f"  {i} {rows[-1][1]:.5f},{rows[-1][2]:.5f} -> {os.path.basename(p)}", flush=True)

    idx = np.array([r[0] for r in rows])
    np.savez(os.path.join(a.out, "val_truth.npz"),
             idx=idx, lat=np.array([r[1] for r in rows]), lon=np.array([r[2] for r in rows]))

    # cross-check against the frozen val_meta the whole bed was scored on
    vz = np.load(os.path.join(REPO, "run_c4_index", "val_meta.npz"))
    d = np.abs(vz["lat"][idx] - np.array([r[1] for r in rows])).max()
    print(f"\nmax |lat - val_meta.lat| over {len(rows)} rows: {d:.8f} "
          f"({'ALIGNED' if d < 1e-6 else 'MISALIGNED — row order differs!'})")
    print(f"saved {len(rows)} images (long side <= {SAVE_MAX_SIDE}) to {a.out}")


if __name__ == "__main__":
    main()
