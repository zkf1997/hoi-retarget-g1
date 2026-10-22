#!/usr/bin/env python3
"""Create data[_ds]/umr_realhand/ from data[_ds]/umr/: the UMR input variant for a REAL-size object anchored to the hand.

UMR scales the human by s = 1.32 / height and (in `retarget_object_size=original` mode) keeps the object mesh at real size
but still multiplies the object position by s, so the hand ends up 0.15 x radius too close to the surface. Here the robot-side
trajectory keeps the human wrist -> object offset at REAL scale:  p' = a + (p - a) / s  (UMR multiplies by s afterwards, so
s p' = s a + (p - a)), where a is the wrist of the hand holding the object. Per frame the two hands are weighted by how close
their fingertips are to the object surface (1 within 2 cm, 0 beyond 10 cm, smoothed over 5 frames); with no hand near, the
object keeps the uniform scaling p' = p. The source-side trajectory (prop_<obj>.csv) is unchanged (symlink), the robot-side
one is written as retarget_prop_<obj>.csv (read by our UMR patch). Everything else is symlinked.
    BENCH_DATASET=humoto python prepare_realsize_hand_anchor.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
DATASET = os.environ.get("BENCH_DATASET", "omomo")
SUFFIX = "" if DATASET == "omomo" else f"_{DATASET}"
DATA, RESULTS = HERE / f"data{SUFFIX}", HERE / f"results{SUFFIX}"
HB = HERE.parent / "hand_bench" / ("data" if DATASET == "humoto" else f"data_{DATASET}") / "seqs"
SRC, DST = DATA / "umr", DATA / "umr_realhand"
TIPS = [4, 8, 12, 16, 20]


def on_surface_chunked(pq, pts, chunk=200):
    out = []
    for i in range(0, len(pts), chunk):
        out.append(pq.on_surface(pts[i:i + chunk])[1])
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    DST.mkdir(exist_ok=True)
    link = DST / "object_assets"
    if (SRC / "object_assets").exists() and not link.exists():
        link.symlink_to(Path("..") / "umr" / "object_assets")
    index = json.loads((DATA / "seqs_test.json").read_text())
    stats = []
    for e in index:
        seq = e["seq"]
        seq_dir = SRC / "test" / seq
        if not seq_dir.exists():
            continue
        src = np.load(DATA / "source" / f"{seq}.npz", allow_pickle=True)
        hb = np.load(HB / f"{seq}.npz", allow_pickle=True)
        obj = str(src["object"]); oi = [str(o) for o in hb["objects"]].index(obj)
        s = float(np.asarray(np.load(RESULTS / "umr_wuji" / f"{seq}.npz", allow_pickle=True)["smpl_scale"]).reshape(-1)[0])
        mesh = trimesh.load(DATA / "objects" / obj / f"{obj}.obj", force="mesh"); pq = trimesh.proximity.ProximityQuery(mesh)
        out = DST / "test" / seq; out.mkdir(parents=True, exist_ok=True)
        for f in seq_dir.iterdir():
            dst = out / f.name
            if not dst.exists():
                dst.symlink_to(Path("../../..") / "umr" / "test" / seq / f.name)
        prop = seq_dir / f"prop_{obj}.csv"
        rows = list(csv.DictReader(prop.open()))
        p = np.array([[float(r["px"]), float(r["py"]), float(r["pz"])] for r in rows])
        T = min(len(p), int(hb["n_frames"]), len(src["trans"]))
        op = hb["obj_pos"][:T, oi].astype(np.float64); oR = hb["obj_rot"][:T, oi].astype(np.float64)
        assert np.allclose(op, p[:T], atol=1e-4), f"{seq}: prop csv frame differs from the source object trajectory"
        w, a = {}, {}
        for side in ("left", "right"):
            kp = hb[f"hand_kp_{side}"][:T].astype(np.float64)
            loc = np.einsum("tji,tkj->tki", oR, kp[:, TIPS] - op[:, None]).reshape(-1, 3)
            d = on_surface_chunked(pq, loc).reshape(T, 5).min(1)
            wi = np.clip((0.10 - d) / 0.08, 0.0, 1.0)
            w[side] = uniform_filter1d(wi, size=5, mode="nearest") if T > 5 else wi
            a[side] = kp[:, 0]
        W = w["left"] + w["right"]
        norm = np.maximum(W, 1.0)
        p_new = p.copy()
        for side in w:
            p_new[:T] += (w[side] / norm)[:, None] * (p[:T] - a[side]) * (1.0 / s - 1.0)
        with (out / f"retarget_prop_{obj}.csv").open("w") as fh:
            fh.write("px,py,pz,qx,qy,qz,qw\n")
            for r, pp in zip(rows, p_new):
                fh.write(",".join([f"{pp[0]:.6f}", f"{pp[1]:.6f}", f"{pp[2]:.6f}", r["qx"], r["qy"], r["qz"], r["qw"]]) + "\n")
        shift = np.linalg.norm((p_new[:T] - p[:T]) * s, axis=1)  # robot-world shift of the object centre
        stats.append((seq, float(np.minimum(W, 1).mean()), float(shift.mean()), float(shift.max())))
    print(f"{len(stats)} sequences -> {DST}")
    print("mean anchored fraction %.2f, mean robot-world object shift %.1f cm (max of per-seq max %.1f cm)" % (
        np.mean([x[1] for x in stats]), 100 * np.mean([x[2] for x in stats]), 100 * max(x[3] for x in stats)))


if __name__ == "__main__":
    main()
