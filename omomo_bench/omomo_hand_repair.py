#!/usr/bin/env python3
"""OMOMO finger repair (dataset without finger data, no AnyTeleop / v3 stage): take a body lane solved with finger collision
OFF (hand placed where the human wrist is, fingers possibly inside the object) and, per frame and hand, re-solve the 20 finger
joints with the FULL collision model: rest-pose prior + previous-frame term + hand/object non-penetration (slack), joint
limits (hand_pipeline.refine_fingers with no contact targets). Arm and body stay as solved. Output lane keeps the UMR npz layout.
    BENCH_DATASET=omomo MUJOCO_GL=egl python omomo_hand_repair.py --body-lane umr_wuji_nofingercol --out pipe_handrepair [--seqs ...]
"""
from __future__ import annotations
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import hand_pipeline as H
import compare as C
from scipy.spatial.transform import Rotation as Rot

# SMPL-X joints55 indices for the canonical hand frame (wrist, index MCP, middle MCP, pinky MCP) per side
SMPLX_HAND = {"left": (20, 25, 28, 31), "right": (21, 40, 43, 46)}


def human_hand_frames(J, side):
    """Per-frame canonical hand frame from SMPL-X joints (same convention as hand_bench.humoto_io.hand_frame)."""
    w, i, m, pk = SMPLX_HAND[side]
    x = J[:, m] - J[:, w]; x /= np.linalg.norm(x, axis=1, keepdims=True)
    z = J[:, i] - J[:, pk]; z -= (z * x).sum(1, keepdims=True) * x; z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = np.cross(z, x)
    return J[:, w], np.stack([x, y, z], axis=-1)


def run_sequence(args):
    seq, body_lane, out_dir, iters, w_prev, overwrite, coupled = args
    C._model_cache.clear()
    out = C.RESULTS / out_dir / f"{seq}.npz"
    if out.exists() and not overwrite:
        return {"seq": seq, "skipped": True}
    t0 = time.time()
    r = dict(np.load(C.RESULTS / body_lane / f"{seq}.npz", allow_pickle=True))
    src = np.load(C.DATA / "source" / f"{seq}.npz", allow_pickle=True)
    s = float(np.asarray(r["smpl_scale"]).reshape(-1)[0]); obj = str(src["object"])
    names = [str(x) for x in r["robot_joint_names"]]; q_in = r["qpos"].astype(np.float64); T = min(len(q_in), len(src["trans"]))
    rb = H.Robot(obj, s)  # full collision model (finger geoms collide)
    col = {n: 7 + i for i, n in enumerate(names)}
    def to_model(qr):
        q = np.zeros(rb.m.nq); q[:7] = qr[:7]
        for n, i in col.items(): q[rb.qadr[n]] = qr[i]
        return q
    def from_model(q):
        o = np.zeros(7 + len(names)); o[:7] = q[:7]
        for n, i in col.items(): o[i] = q[rb.qadr[n]]
        return o
    op_r = src["obj_pos"][:T] * s; oq_r = src["obj_quat_xyzw"][:T][:, [3, 0, 1, 2]]
    res = np.zeros((T, 7 + len(names))); prev = {}; prev2 = {}; stats = {"pairs": 0, "slack": 0.0}
    frames = {side: human_hand_frames(src["joints55"][:T].astype(np.float64), side) for side in ("left", "right")}
    for t in range(T):
        q = to_model(q_in[t]); q[rb.obj_qadr:rb.obj_qadr + 3] = op_r[t]; q[rb.obj_qadr + 3:rb.obj_qadr + 7] = oq_r[t]
        for side in ("left", "right"):
            S = rb.side[side]; qf = [rb.qadr[n] for n in S["fingers"]]; qa = [rb.qadr[n] for n in S["arm"]]
            prior = q[qf].copy()  # pinned rest pose from the body lane
            if coupled:
                # arm + fingers in one QP: wrist anchored to the (scaled) human wrist pose, rest-pose finger prior, UMR arm prior,
                # hand + forearm non-penetration; no contact targets (OMOMO has no finger data)
                p_h, R_h = frames[side]; p1 = prev.get(side)
                if p1 is not None: q[qa + qf] = p1
                q, sol, st = H.refine_coupled(rb, q, side, prior, q_in_arm := to_model(q_in[t])[qa], p1, prev2.get(side), p_h[t] * s, R_h[t],
                                              np.zeros((5, 3)), np.zeros(5), w_prev=w_prev, w_acc=20.0, w_tip=0.0, w_pose=50.0, w_rot=5.0, iters=iters)
                prev2[side] = p1; prev[side] = sol
            else:
                q, st = H.refine_fingers(rb, q, side, prior, prev.get(side, prior), np.zeros((5, 3)), np.zeros(5), w_prior=1.0, w_prev=w_prev, iters=iters)
                prev[side] = q[qf].copy()
            stats["pairs"] = max(stats["pairs"], st["pairs"]); stats["slack"] = max(stats["slack"], st["slack"])
        res[t] = from_model(q)
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {k: r[k] for k in r if k != "qpos"}
    np.savez_compressed(out, qpos=res.astype(np.float32), stage="handrepair_coupled" if coupled else "handrepair", body_lane=body_lane, **extra)
    return {"seq": seq, "T": T, "seconds": round(time.time() - t0, 1), "max_pairs": stats["pairs"], "max_slack_mm": round(1000 * stats["slack"], 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body-lane", default="umr_wuji_nofingercol"); ap.add_argument("--out", default="pipe_handrepair")
    ap.add_argument("--seqs", nargs="*"); ap.add_argument("--iters", type=int, default=6); ap.add_argument("--w-prev", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--coupled", action="store_true", help="also move the 7 arm joints (wrist anchored to the human wrist pose), hand + forearm non-penetration")
    a = ap.parse_args()
    index = json.loads((C.DATA / "seqs_test.json").read_text())
    seqs = [e["seq"] for e in index if (not a.seqs or e["seq"] in set(a.seqs)) and (C.RESULTS / a.body_lane / f"{e['seq']}.npz").exists()]
    print(f"[{C.DATASET}] {len(seqs)} sequences, body lane {a.body_lane} -> {a.out}")
    jobs = [(s, a.body_lane, a.out, a.iters, a.w_prev, a.overwrite, a.coupled) for s in seqs]
    if a.workers <= 1:
        for j in jobs: print(run_sequence(j), flush=True)
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(run_sequence, jobs): print(r, flush=True)


if __name__ == "__main__":
    main()
