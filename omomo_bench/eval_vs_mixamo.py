#!/usr/bin/env python3
"""Tracking error of OmniRetarget result lanes against the ORIGINAL HUMOTO Mixamo joints (not the lane's own input), so that
source variants (SMPL-X fit vs. Mixamo joints) can be compared on equal footing.

    BENCH_DATASET=humoto MUJOCO_GL=egl python eval_vs_mixamo.py --index data_humoto/seqs_test_mx.json \
        --lane smplx=results_humoto/omni_brainco_umrscale --lane mx=results_humoto/omni_brainco_mx ...

Per lane, mean over frames and sequences of |robot link - s * Mixamo joint| for wrist (Hand), ankle (Foot), toe (ToeBase),
pelvis (Hips), knee (Leg), elbow (ForeArm), s = the lane's smpl_scale. Robot links: hand flange, ankle_roll_link,
ankle_roll_sphere_5_link (toe sphere), pelvis, knee_link, elbow_link.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as C  # noqa: E402  (omomo_bench/compare.py, must be imported before hand_bench is on the path)
sys.path.append(str(HERE.parent / "hand_bench"))
import humoto_io as H  # noqa: E402

LINKS = {"wrist": ("left_wrist_yaw_link", "right_wrist_yaw_link"), "ankle": ("left_ankle_roll_link", "right_ankle_roll_link"),
         "toe": ("left_ankle_roll_sphere_5_link", "right_ankle_roll_sphere_5_link"), "pelvis": ("pelvis", "pelvis"),
         "knee": ("left_knee_link", "right_knee_link"), "elbow": ("left_elbow_link", "right_elbow_link")}
MIX = {"wrist": ("LeftHand", "RightHand"), "ankle": ("LeftFoot", "RightFoot"), "toe": ("LeftToeBase", "RightToeBase"),
       "pelvis": ("Hips", "Hips"), "knee": ("LeftLeg", "RightLeg"), "elbow": ("LeftForeArm", "RightForeArm")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--lane", action="append", required=True, metavar="NAME=DIR")
    args = ap.parse_args()
    index = json.loads(args.index.read_text())
    lanes = {spec.split("=")[0]: HERE / spec.split("=", 1)[1] for spec in args.lane}
    err = {n: {k: [] for k in LINKS} for n in lanes}
    bi = {n: i for i, n in enumerate(H.MIXAMO_BODY)}
    for e in index:
        seq = H.load_sequence(e["seq"])
        bp = seq["body_pos"]
        for name, d in lanes.items():
            p = d / f"{e['seq']}_original.npz"
            if not p.exists():
                continue
            r = np.load(p, allow_pickle=True)
            s = float(r["smpl_scale"]); q = r["qpos"]
            m = C.omni_model(str(r["hand"]), str(r["scaling"]), C.omni_subject(r, e["subject"]), e["object"])
            dd = mujoco.MjData(m)
            T = min(len(q), len(bp))
            ids = {k: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in v] for k, v in LINKS.items()}
            acc = {k: [] for k in LINKS}
            for t in range(0, T, 2):
                dd.qpos[:] = q[t]; mujoco.mj_forward(m, dd)
                for k in LINKS:
                    for side in (0, 1):
                        b = ids[k][side]
                        pos = dd.xpos[b] + (dd.xmat[b].reshape(3, 3) @ C.WRIST_OFFSET if k == "wrist" else 0.0)
                        acc[k].append(np.linalg.norm(pos - s * bp[t, bi[MIX[k][side]]]))
            for k in LINKS:
                err[name][k].append(np.mean(acc[k]))
    print(f"{'lane':10s} n   " + " ".join(f"{k:>8s}" for k in LINKS) + "   (cm, vs scaled Mixamo joints)")
    for name in lanes:
        n = len(err[name]["wrist"])
        print(f"{name:10s} {n:<3d} " + " ".join(f"{100*np.mean(err[name][k]):8.2f}" for k in LINKS))


if __name__ == "__main__":
    main()
