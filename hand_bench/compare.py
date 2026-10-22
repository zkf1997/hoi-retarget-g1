#!/usr/bin/env python3
"""Metrics for the HUMOTO hand-retargeting benchmark -> results/metrics.csv + results/summary.md

    python compare.py --workers 16

Per (method, robot hand, sequence, side) the retargeted hand is placed at the method's flange pose (AnyTeleop: on the human
wrist; TopoRetarget: optimised) and evaluated against the hand-held object it interacts with:
  pen_depth_mm   mean over interaction frames of the deepest robot-hand/object penetration (MuJoCo mj_geomDistance, hand
                 collision geoms vs CoACD convex pieces); pen_frac = share of interaction frames with > 5 mm penetration
  contact_recall / contact_precision / contact_f1
                 human fingertip within `contact_thr` of the object surface  vs  robot fingertip within `contact_thr`
  tip_obj_dist_err_mm  |robot tip->object distance - human tip->object distance| (contact geometry preservation)
  tip_pos_err_mm  robot fingertip vs size-scaled human fingertip position in the hand frame (kinematic tracking)
  pinch_err_mm    thumb-index tip distance vs scaled human;  inter_tip_err_mm: all 10 tip pairs
  base_drift_mm / base_rot_deg   flange offset from the human wrist (0 for AnyTeleop by construction)
  q_jerk_mrad (mrad/frame^2), q_vel_mrad (mrad/frame), limit_sat (share of joint samples within 1e-3 rad of a limit), runtime_ms
Interaction frames: human hand within `interact_thr` (10 cm) of the object. Smoothness is over all frames.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation as R

import bench_common as bc
from hand_model import HUMAN_TIPS, HandModel, human_to_robot_scale

TIP_PAIRS = [(i, j) for a, i in enumerate(HUMAN_TIPS) for j in HUMAN_TIPS[a + 1:]]


class Scene:
    """hand (fixed at the origin of its base frame) + object convex pieces on a mocap body."""

    def __init__(self, model: HandModel, obj: str):
        spec = mujoco.MjSpec.from_file(str(model.urdf))
        body = spec.worldbody.add_body(name="obj", mocap=True)
        for i, piece in enumerate(bc.load_object_pieces(obj)):
            mesh = spec.add_mesh(name=f"obj{i}")
            mesh.file = str(piece)
            body.add_geom(name=f"obj{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"obj{i}", contype=0, conaffinity=0)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.model = model
        obj_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.obj_geoms = np.array([g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == obj_body])
        self.hand_geoms = np.array([g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] != obj_body and self.m.geom_contype[g] | self.m.geom_conaffinity[g]])
        self.qadr = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j): int(self.m.jnt_qposadr[j]) for j in range(self.m.njnt)}
        self.mocap_id = int(self.m.body_mocapid[obj_body])
        self.fromto = np.zeros(6)

    def min_distance(self, q_active: np.ndarray, obj_pos_b: np.ndarray, obj_rot_b: np.ndarray, distmax: float = 0.05) -> float:
        qfull = self.model.qpos_dict(q_active)
        for n, v in qfull.items():
            self.d.qpos[self.qadr[n]] = v
        self.d.mocap_pos[self.mocap_id] = obj_pos_b
        self.d.mocap_quat[self.mocap_id] = R.from_matrix(obj_rot_b).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(self.m, self.d)
        xp, rb = self.d.geom_xpos, self.m.geom_rbound
        best = distmax
        for go in self.obj_geoms:
            c = np.linalg.norm(xp[self.hand_geoms] - xp[go], axis=1) - rb[self.hand_geoms] - rb[go]
            for gh in self.hand_geoms[c < distmax]:
                dist = mujoco.mj_geomDistance(self.m, self.d, int(gh), int(go), distmax, self.fromto)
                if dist < best:
                    best = dist
        return best


def evaluate(method: str, hand: str, seq: str, side: str, interact_thr: float, contact_thr: float) -> dict | None:
    res = bc.load_result(method, hand, seq, side)
    if res is None:
        return None
    d = bc.load_seq(seq)
    k = bc.main_object(d, side)
    if k < 0:
        return None
    obj = d["objects"][k]
    T = int(res.get("n_frames_done", d["n_frames"]))
    model = HandModel(hand, side)
    scene = Scene(model, obj)
    q = res["q_active"][:T].astype(np.float64)
    fl_pos, fl_rot = bc.flange_world(d, side, res["base_pos_h"][:T], res["base_quat_h"][:T])
    base_pos = fl_pos - np.einsum("tij,j->ti", fl_rot, model.flange)  # flange frame == base frame orientation
    kp_h = bc.hand_frame_kp(d, side)[:T]
    scale = human_to_robot_scale(model, kp_h)
    Rq = R.from_quat(res["base_quat_h"][:T][:, [1, 2, 3, 0]])

    op, orot, osc = d["obj_pos"][:T, k].astype(np.float64), d["obj_rot"][:T, k].astype(np.float64), d["obj_scale"][:T, k].astype(np.float64)
    vis = bc.load_object_visual(obj)
    pq = trimesh.proximity.ProximityQuery(vis)
    interact = d[f"near_dist_{side}"][:T] < interact_thr

    def tip_obj_dist(pts_world: np.ndarray, t: int) -> np.ndarray:
        local = (pts_world - op[t]) @ orot[t] / osc[t]
        return pq.signed_distance(local) * -osc[t] if vis.is_watertight else np.linalg.norm(local - pq.on_surface(local)[0], axis=1) * osc[t]

    pen, human_c, robot_c, d_err, tip_err, pinch_err, inter_err = [], [], [], [], [], [], []
    for t in np.flatnonzero(interact):
        # object in the hand base frame -> penetration
        p_b = fl_rot[t].T @ (op[t] - base_pos[t])
        R_b = fl_rot[t].T @ orot[t]
        if abs(osc[t] - 1.0) > 1e-3:
            raise RuntimeError(f"{seq}: object scale {osc[t]} != 1 not supported by the collision scene")
        pen.append(max(0.0, -scene.min_distance(q[t], p_b, R_b)))
        # tips
        rk_h = Rq[t].apply(model.keypoints(q[t]) - model.flange) + res["base_pos_h"][t]  # robot kps in the human hand frame
        rk_w = rk_h @ d[f"hand_R_{side}"][t].T + d[f"hand_kp_{side}"][t, 0]
        hk_w = d[f"hand_kp_{side}"][t].astype(np.float64)
        dr, dh = tip_obj_dist(rk_w[HUMAN_TIPS], t), tip_obj_dist(hk_w[HUMAN_TIPS], t)
        human_c.append(dh < contact_thr)
        robot_c.append(dr < contact_thr)
        d_err.append(np.abs(np.clip(dr, 0, None) - np.clip(dh, 0, None)))
        hs = kp_h[t] * scale
        tip_err.append(np.linalg.norm(rk_h[HUMAN_TIPS] - hs[HUMAN_TIPS], axis=1))
        pinch_err.append(abs(np.linalg.norm(rk_h[4] - rk_h[8]) - np.linalg.norm(hs[4] - hs[8])))
        inter_err.append(np.mean([abs(np.linalg.norm(rk_h[i] - rk_h[j]) - np.linalg.norm(hs[i] - hs[j])) for i, j in TIP_PAIRS]))
    hc, rc = np.array(human_c, dtype=bool).reshape(-1, 5), np.array(robot_c, dtype=bool).reshape(-1, 5)
    tp = (hc & rc).sum()
    row = dict(method=method, hand=hand, seq=seq, side=side, object=obj, n_frames=T, n_interact=int(interact.sum()),
               human_contact_frac=float(hc.mean()) if len(hc) else np.nan,
               pen_depth_mm=1e3 * np.mean(pen) if pen else np.nan, pen_max_mm=1e3 * np.max(pen) if pen else np.nan,
               pen_frac=float(np.mean(np.array(pen) > 0.005)) if pen else np.nan,
               contact_recall=tp / max(hc.sum(), 1) if hc.sum() else np.nan,
               contact_precision=tp / max(rc.sum(), 1) if rc.sum() else np.nan,
               tip_obj_dist_err_mm=1e3 * np.mean(d_err) if d_err else np.nan,
               tip_pos_err_mm=1e3 * np.mean(tip_err) if tip_err else np.nan,
               pinch_err_mm=1e3 * np.mean(pinch_err) if pinch_err else np.nan,
               inter_tip_err_mm=1e3 * np.mean(inter_err) if inter_err else np.nan,
               base_drift_mm=1e3 * np.linalg.norm(res["base_pos_h"][:T], axis=1).mean(),
               base_rot_deg=float(np.degrees(np.abs(Rq.magnitude()).mean())),
               q_vel_mrad=1e3 * float(np.abs(np.diff(q, axis=0)).mean()) if T > 1 else np.nan,
               q_jerk_mrad=1e3 * float(np.abs(np.diff(q, 2, axis=0)).mean()) if T > 2 else np.nan,
               limit_sat=float(((q < model.limits[:, 0] + 1e-3) | (q > model.limits[:, 1] - 1e-3)).mean()),
               runtime_ms=1e3 * float(res["runtime_per_frame"]), scale=scale)
    r_, p_ = row["contact_recall"], row["contact_precision"]
    row["contact_f1"] = 2 * r_ * p_ / (r_ + p_) if (r_ + p_) and not np.isnan(r_ + p_) else np.nan
    return row


def _job(a):
    try:
        return evaluate(*a)
    except Exception as e:  # noqa: BLE001
        import traceback

        print(f"FAILED {a}: {e!r}\n{traceback.format_exc()}", flush=True)
        return None


SUMMARY_COLS = ["pen_depth_mm", "pen_frac", "contact_recall", "contact_precision", "contact_f1", "tip_obj_dist_err_mm",
                "tip_pos_err_mm", "pinch_err_mm", "inter_tip_err_mm", "base_drift_mm", "base_rot_deg", "q_vel_mrad", "q_jerk_mrad", "limit_sat", "runtime_ms"]
LOWER_BETTER = {c: c not in ("contact_recall", "contact_precision", "contact_f1") for c in SUMMARY_COLS}


def write_summary(df: pd.DataFrame, out: Path) -> None:
    lines = [f"# {bc.DATASET.upper()} hand retargeting: AnyTeleop vs TopoRetarget", "",
             f"{df.seq.nunique()} sequences, {df.object.nunique()} objects, sides: {sorted(df.side.unique())}. "
             "Mean over (sequence, side); interaction frames only for object metrics. Bold = best per column (within a robot hand).", ""]
    for hand, g in df.groupby("hand"):
        lines += [f"## {hand}", ""]
        agg = g.groupby("method")[SUMMARY_COLS].mean()
        n = g.groupby("method").size()
        cols = [c for c in SUMMARY_COLS if agg[c].notna().any()]
        lines.append("| method | n | " + " | ".join(cols) + " |")
        lines.append("|---|---|" + "---|" * len(cols))
        best = {c: (agg[c].idxmin() if LOWER_BETTER[c] else agg[c].idxmax()) for c in cols}
        for m in agg.index:
            cells = []
            for c in cols:
                v = agg.loc[m, c]
                s = f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
                cells.append(f"**{s}**" if best[c] == m else s)
            lines.append(f"| {m} | {n[m]} | " + " | ".join(cells) + " |")
        lines.append("")
        # per-object contact f1 / penetration
        po = g.pivot_table(index="object", columns="method", values=["contact_f1", "pen_depth_mm"], aggfunc="mean")
        if len(po):
            lines += [f"### {hand}: per object (contact_f1 / pen_depth_mm)", ""]
            methods = sorted(g.method.unique())
            lines.append("| object | n | " + " | ".join(methods) + " |")
            lines.append("|---|---|" + "---|" * len(methods))
            cnt = g[g.method == methods[0]].groupby("object").size()
            for o in po.index:
                cells = [f"{po.loc[o, ('contact_f1', m)]:.2f} / {po.loc[o, ('pen_depth_mm', m)]:.1f}" for m in methods]
                lines.append(f"| {o} | {cnt.get(o, 0)} | " + " | ".join(cells) + " |")
            lines.append("")
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(bc.METHODS))
    ap.add_argument("--hands", nargs="*", default=list(bc.HANDS))
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--interact-thr", type=float, default=0.10)
    ap.add_argument("--contact-thr", type=float, default=0.015)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(bc.RESULTS))
    a = ap.parse_args()
    seqs = a.seqs or bc.list_prepared()
    jobs = [(m, h, s, side, a.interact_thr, a.contact_thr) for m in a.methods for h in a.hands for s in seqs for side in bc.SIDES
            if bc.result_path(m, h, s, side).exists()]
    print(f"{len(jobs)} result files to evaluate")
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(_job, jobs, chunksize=2) if r is not None]
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    df.to_csv(out / "metrics.csv", index=False)
    write_summary(df, out / "summary.md")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
