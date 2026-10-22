#!/usr/bin/env python3
"""Wrist displacement while the human hand is on the object ("hand pushed to the other side" defect).
Per lane and side: frames where the human wrist is within `near` of the object surface; among them the fraction where the
robot wrist is farther than `gap` from the (scaled) human wrist; plus hand-object penetration from the full-collision model.
    BENCH_DATASET=omomo MUJOCO_GL=egl python wrist_gap_stats.py --lane omni_wuji=results/omni_wuji_umrscale:_original --lane umr_wuji=results/umr_wuji [--seqs ...] [--out csv]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, mujoco, trimesh
sys.path.insert(0, str(Path(__file__).resolve().parent)); import compare as C
from scipy.spatial.transform import Rotation as Rot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lane", action="append", required=True); ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--near", type=float, default=0.10); ap.add_argument("--gap", type=float, default=0.12); ap.add_argument("--out", default="")
    ap.add_argument("--hand-xml", default="", help="robot MJCF for UMR lanes (default compare.UMR_ROBOT_XML by npz hand key)")
    a = ap.parse_args()
    lanes = {x.split("=")[0]: (Path(x.split("=", 1)[1].split(":")[0]), x.split("=", 1)[1].partition(":")[2]) for x in a.lane}
    index = json.load(open(C.DATA / "seqs_test.json"))
    if a.seqs: index = [e for e in index if e["seq"] in set(a.seqs)]
    rows = []
    for e in index:
        seq = e["seq"]; src = np.load(C.DATA / "source" / f"{seq}.npz", allow_pickle=True); obj = str(src["object"]); J = src["joints55"]
        s = 1.32 / float(src["tpose_height"]); T = len(J)
        mesh = trimesh.load(C.DATA / "objects" / obj / f"{obj}.obj", force="mesh"); pq = trimesh.proximity.ProximityQuery(mesh)
        oR = Rot.from_quat(src["obj_quat_xyzw"]).as_matrix(); op = src["obj_pos"]
        hd = {k: C.on_surface_chunked(pq, np.einsum("tji,tj->ti", oR, J[:, C.SX[k]] - op))[1] for k in ("L_Wrist", "R_Wrist")}
        for lane, (rdir, suf) in lanes.items():
            p = rdir / f"{seq}{suf}.npz"
            if not p.exists(): continue
            r = np.load(p, allow_pickle=True); is_omni = "human_joints" in r
            if is_omni:
                m = C.omni_model(str(r["hand"]), str(r["scaling"]) if "scaling" in r else "omni", C.omni_subject(r, e["subject"]), obj); q = r["qpos"][:, :36]
                op_r, oq_r = r["qpos"][:, 36:39], r["qpos"][:, 39:43]
            else:
                m = C.augmented_model(Path(a.hand_xml) if a.hand_xml else C.umr_robot_xml(r), obj, s); q = r["qpos"]
                op_r, oq_r = op * s, src["obj_quat_xyzw"][:, [3, 0, 1, 2]]
            Tq = min(len(q), T); d = mujoco.MjData(m); bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
            ob = bn.index(f"{obj}_link"); is_obj = np.array([m.geom_bodyid[g] == ob for g in range(m.ngeom)])
            wb = {"L_Wrist": bn.index("left_wrist_yaw_link"), "R_Wrist": bn.index("right_wrist_yaw_link")}
            pref = {"L_Wrist": ("left_", "l_"), "R_Wrist": ("right_", "r_")}
            is_hand = {k: np.array([(bn[m.geom_bodyid[g]].startswith(pref[k][1]) or bn[m.geom_bodyid[g]].startswith(pref[k][0] + "wrist")) and (m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)]) for k in wb}
            nq = q.shape[1]; rw = {k: np.zeros((Tq, 3)) for k in wb}; pen = {k: np.zeros(Tq) for k in wb}
            for t in range(Tq):
                d.qpos[:nq] = q[t]
                if not is_omni:
                    a0 = int(m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_bodyid[j] == ob][0]]); d.qpos[a0:a0 + 3] = op_r[t]; d.qpos[a0 + 3:a0 + 7] = oq_r[t]
                mujoco.mj_forward(m, d); mujoco.mj_collision(m, d)
                for k, b in wb.items(): rw[k][t] = d.xpos[b] / s if not is_omni else d.xpos[b] / s
                for i in range(d.ncon):
                    cc = d.contact[i]
                    if cc.dist >= 0: continue
                    for k in wb:
                        if (is_obj[cc.geom1] and is_hand[k][cc.geom2]) or (is_obj[cc.geom2] and is_hand[k][cc.geom1]): pen[k][t] = max(pen[k][t], -cc.dist)
            for k in wb:
                hw = J[:Tq, C.SX[k]]; gap = np.linalg.norm(rw[k] - hw, axis=1); near = hd[k][:Tq] < a.near
                rows.append({"seq": seq, "lane": lane, "side": k[0], "near_frames": int(near.sum()), "gap_mean_cm": 100 * gap[near].mean() if near.any() else np.nan,
                             "gap_frac": (gap[near] > a.gap).mean() if near.any() else np.nan, "pen_mm": 1000 * pen[k][near].mean() if near.any() else np.nan, "pen_frac": (pen[k][near] > 0.005).mean() if near.any() else np.nan})
        C._model_cache.clear()
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"{'lane':22s} {'near frames':>11s} {'gap mean cm':>11s} {'frac>gap':>8s} {'pen mm':>7s} {'pen>5mm':>8s}")
    for lane in lanes:
        rr = [r for r in rows if r["lane"] == lane and r["near_frames"] > 0]
        if not rr: continue
        nf = sum(r["near_frames"] for r in rr); wm = lambda key: sum(r[key] * r["near_frames"] for r in rr) / nf
        print(f"{lane:22s} {nf:11d} {wm('gap_mean_cm'):11.1f} {wm('gap_frac'):8.2f} {wm('pen_mm'):7.2f} {wm('pen_frac'):8.2f}   (seqs with any hand frac>0.5: {sum(1 for r in rr if r['gap_frac'] > 0.5)})")


if __name__ == "__main__":
    main()
