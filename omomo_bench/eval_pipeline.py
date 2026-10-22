#!/usr/bin/env python3
"""Metrics for the whole-body + hand pipeline lanes (UMR rest hand -> +AnyTeleop fingers -> +refinement).

    BENCH_DATASET=humoto MUJOCO_GL=egl python eval_pipeline.py --workers 16            # -> results_humoto/pipe_metrics.csv, pipe_summary.md
    BENCH_DATASET=omomo  MUJOCO_GL=egl python eval_pipeline.py --lane umr_wuji=results/umr_wuji --no-hand-data

Body metrics come from compare.py (object / ground penetration over ALL robot geoms incl. fingers, foot slide, wrist tracking,
jerk). Hand metrics (per sequence, averaged over the two hands; distances in HUMAN units = robot world / smpl_scale):
  flange_pos_err_cm   robot flange (`{l,r}_mount`) vs human wrist keypoint          flange_rot_deg  wrist_yaw frame vs canonical hand frame
  tip_pos_err_cm      5 robot fingertips vs human fingertips (interaction frames)   tip_obj_dist_err_cm  | d_robot - d_human | fingertip-to-surface
  contact_recall / precision / f1   human fingertip within tau of the object  <->  robot fingertip within tau
  hand_pen_mm         deepest hand-geom / object penetration per frame, mean over interaction frames (robot world, mm)
  hand_pen_frac       fraction of interaction frames with hand penetration > 5 mm
  finger_jerk         mean |d^3 q/dt^3| of the 40 finger joints (rad/s^3)
Interaction frame = any human fingertip of that hand within --near (3 cm) of the object.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C  # noqa: E402

HERE = Path(__file__).resolve().parent
HB_DATA = HERE.parent / "hand_bench" / ("data" if C.DATASET == "humoto" else f"data_{C.DATASET}")
TIPS_H = [4, 8, 12, 16, 20]
TIP_BODIES = ["thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]
HAND_COLS = ["flange_pos_err_cm", "flange_rot_deg", "tip_pos_err_cm", "tip_obj_dist_err_cm", "contact_recall", "contact_precision",
             "contact_f1", "hand_pen_mm", "hand_pen_frac", "finger_jerk", "interaction_frac",
             "kp_contact_recall", "kp_contact_precision", "kp_contact_f1", "kp_obj_dist_err_cm",  # dense points (phalanx joints + palm), 3 cm
             "scene_pen_mm", "scene_pen_frac"]  # whole robot vs every object of the evaluation (furniture included with --multi)
KP_TAU = 0.03


def hand_metrics(seq: str, e: dict, lanes: dict, tau: float, near: float, fps: float = 30.0, multi: bool = False) -> dict[str, dict]:
    """Hand metrics per lane. Contacts, distances and penetration are measured against the primary object, or with
    multi=True against every interacted hand-held object of the sequence (data*/multi_objects.json): a point's distance
    is its distance to the nearest object."""
    import mujoco, trimesh
    from scipy.spatial.transform import Rotation as Rot
    from hand_pipeline import hand_kp_bodies, dense_points_from_kp
    src = np.load(C.DATA / "source" / f"{seq}.npz", allow_pickle=True)
    hb = np.load(HB_DATA / "seqs" / f"{seq}.npz", allow_pickle=True)
    obj = str(src["object"]); objs_hb = [str(o) for o in hb["objects"]]
    objs = C.multi_objects(seq, src) if multi else [obj]
    pqs = [trimesh.proximity.ProximityQuery(trimesh.load(C.DATA / "objects" / o / f"{o}.obj", force="mesh")) for o in objs]
    out = {}
    for lane, (rdir, suffix) in lanes.items():
        p = rdir / f"{seq}{suffix}.npz"
        if not p.exists():
            continue
        r = np.load(p, allow_pickle=True)
        s = float(np.asarray(r["smpl_scale"]).reshape(-1)[0])
        ms = float(np.asarray(r["object_mesh_scale"]).reshape(-1)[0]) if "object_mesh_scale" in r else s  # robot-world object mesh scale
        q = r["qpos"].astype(np.float64)
        names = [str(x) for x in r["robot_joint_names"]]
        T = min(len(q), int(hb["n_frames"]), len(src["trans"]))
        m = C.augmented_model(C.UMR_ROBOT_XML["wuji"], objs if len(objs) > 1 else obj, ms)
        d = mujoco.MjData(m)
        bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        obj_bodies = [bn.index(f"{o}_link") for o in objs]
        obj_qadr = [int(m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_bodyid[j] == b][0]]) for b in obj_bodies]
        is_obj = np.array([m.geom_bodyid[g] in obj_bodies for g in range(m.ngeom)])
        # per object: human-world pose (hand_bench, = source for the primary) and robot-world pose
        poses = []
        for k, o in enumerate(objs):
            i = objs_hb.index(o)
            oR_h = hb["obj_rot"][:T, i].astype(np.float64); op_h = hb["obj_pos"][:T, i].astype(np.float64)
            if k == 0:
                op_r = C.umr_object_pos(r, src, s, T); oR_r = Rot.from_quat(src["obj_quat_xyzw"][:T]).as_matrix()
            else:
                op_r = op_h * s; oR_r = oR_h
            poses.append((op_h, oR_h, op_r, oR_r, Rot.from_matrix(oR_r).as_quat()[:, [3, 0, 1, 2]]))

        def dist_h(pts):  # human-world points (T, K, 3) -> distance to the nearest object, human units
            return np.min([C.on_surface_chunked(pq, np.einsum("tji,tkj->tki", oR_h, pts - op_h[:, None]).reshape(-1, 3))[1].reshape(pts.shape[:2])
                           for pq, (op_h, oR_h, _, _, _) in zip(pqs, poses)], axis=0)

        def dist_r(pts):  # robot-world points -> distance to the nearest robot-world object, human units
            return np.min([C.on_surface_chunked(pq, (np.einsum("tji,tkj->tki", oR_r, pts - op_r[:, None]) / ms).reshape(-1, 3))[1].reshape(pts.shape[:2]) * (ms / s)
                           for pq, (_, _, op_r, oR_r, _) in zip(pqs, poses)], axis=0)

        nq_r = q.shape[1]
        # whole-robot collision geoms (everything that is neither an object nor the ground plane)
        is_robot = np.array([(not is_obj[g]) and m.geom_bodyid[g] != 0 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE and (m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
        scene_pen = np.zeros(T)
        per_side = []
        for side in ("left", "right"):
            c = side[0]
            tips_b = [bn.index(f"{c}_{t}") for t in TIP_BODIES]
            mount, wrist = bn.index(f"{c}_mount"), bn.index(f"{side}_wrist_yaw_link")
            is_hand = np.array([bn[m.geom_bodyid[g]].startswith(f"{c}_") and (m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
            kp = hb[f"hand_kp_{side}"][:T].astype(np.float64); Rh = hb[f"hand_R_{side}"][:T].astype(np.float64)
            kp_b = [bn.index(n) for n in hand_kp_bodies(c)]
            tips_r = np.zeros((T, 5, 3)); fl_p = np.zeros((T, 3)); fl_R = np.zeros((T, 3, 3)); pen = np.zeros(T); kp_r = np.zeros((T, 21, 3))
            for t in range(T):
                d.qpos[:nq_r] = q[t]
                for a, (_, _, op_r, _, oq_r) in zip(obj_qadr, poses):
                    d.qpos[a:a + 3] = op_r[t]; d.qpos[a + 3:a + 7] = oq_r[t]
                mujoco.mj_forward(m, d); mujoco.mj_collision(m, d)
                tips_r[t] = d.xpos[tips_b]; fl_p[t] = d.xpos[mount]; fl_R[t] = d.xmat[wrist].reshape(3, 3); kp_r[t] = d.xpos[kp_b]
                worst = 0.0; worst_s = 0.0
                for i in range(d.ncon):
                    cc = d.contact[i]
                    if cc.dist < 0 and ((is_obj[cc.geom1] and is_hand[cc.geom2]) or (is_obj[cc.geom2] and is_hand[cc.geom1])):
                        worst = max(worst, -cc.dist)
                    if side == "left" and cc.dist < 0 and ((is_obj[cc.geom1] and is_robot[cc.geom2]) or (is_obj[cc.geom2] and is_robot[cc.geom1])):
                        worst_s = max(worst_s, -cc.dist)
                pen[t] = worst
                if side == "left":
                    scene_pen[t] = worst_s
            d_h = dist_h(kp[:, TIPS_H]); d_r = dist_r(tips_r)
            inter = (d_h.min(1) < near)
            ch, cr = d_h < tau, d_r < tau
            tp = (ch & cr).sum(); rec = tp / max(ch.sum(), 1); prec = tp / max(cr.sum(), 1)
            f1 = 2 * rec * prec / max(rec + prec, 1e-9) if ch.sum() else np.nan
            # dense points (phalanx joint centres + palm): contact at KP_TAU in human units, same convention as the tips
            dd_h = dist_h(dense_points_from_kp(kp)); dd_r = dist_r(dense_points_from_kp(kp_r))
            kch, kcr = dd_h < KP_TAU, dd_r < KP_TAU
            ktp = (kch & kcr).sum(); krec = ktp / max(kch.sum(), 1); kprec = ktp / max(kcr.sum(), 1)
            kf1 = 2 * krec * kprec / max(krec + kprec, 1e-9) if kch.sum() else np.nan
            fl_err = np.linalg.norm(fl_p / s - kp[:, 0], axis=1)
            rot_err = np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("tij,tkj->tik", Rh, fl_R)).as_rotvec(), axis=1))
            fq = [i for i, n in enumerate(names) if n.startswith(f"{c}_")]
            qf = q[:T, 7:][:, fq]
            jerk = float(np.abs(np.diff(qf, n=3, axis=0) * fps ** 3).mean()) if T > 3 else np.nan
            per_side.append({
                "flange_pos_err_cm": 100 * fl_err.mean(), "flange_rot_deg": rot_err.mean(),
                "tip_pos_err_cm": 100 * np.linalg.norm(tips_r / s - kp[:, TIPS_H], axis=2)[inter].mean() if inter.any() else np.nan,
                "tip_obj_dist_err_cm": 100 * np.abs(d_r - d_h)[inter].mean() if inter.any() else np.nan,
                "contact_recall": rec if ch.sum() else np.nan, "contact_precision": prec if cr.sum() else np.nan, "contact_f1": f1,
                "hand_pen_mm": 1000 * pen[inter].mean() if inter.any() else 1000 * pen.mean(),
                "hand_pen_frac": (pen[inter] > 0.005).mean() if inter.any() else (pen > 0.005).mean(),
                "finger_jerk": jerk, "interaction_frac": inter.mean(),
                "kp_contact_recall": krec if kch.sum() else np.nan, "kp_contact_precision": kprec if kcr.sum() else np.nan, "kp_contact_f1": kf1,
                "kp_obj_dist_err_cm": 100 * np.abs(dd_r - dd_h)[inter].mean() if inter.any() else np.nan,
                "scene_pen_mm": 1000 * scene_pen.mean(), "scene_pen_frac": (scene_pen > 0.005).mean()})
        out[lane] = {k: float(np.nanmean([ps[k] for ps in per_side])) for k in HAND_COLS}
    return out


def eval_one(args):
    seq, e, lanes, tau, near, with_hand = args[:6]; multi = bool(args[6]) if len(args) > 6 else False
    import mujoco  # noqa
    C._model_cache.clear()  # models are per (object, scale): the cache would grow without bound across sequences
    C.LANES.clear(); C.LANES.update(lanes)
    runtimes = {m: {} for m in lanes}; fallbacks = {m: {} for m in lanes}
    try:
        rows = C.eval_sequence(e, mujoco.MjModel.from_xml_path(str(C.WORK / "models" / "g1" / "g1_29dof.xml")) if (C.WORK / "models" / "g1" / "g1_29dof.xml").exists()
                               else mujoco.MjModel.from_xml_path(str(HERE / "omni_workdir" / "models" / "g1" / "g1_29dof.xml")), runtimes, fallbacks)
    except Exception as ex:
        return {"seq": seq, "error": f"{type(ex).__name__}: {ex}"}
    hm = hand_metrics(seq, e, lanes, tau, near, multi=multi) if with_hand else {}
    for r in rows:
        r.update(hm.get(r["method"], {k: np.nan for k in HAND_COLS}))
    return {"seq": seq, "rows": rows}


def summarize(rows, lanes, with_hand):
    body = ["obj_pen_max", "obj_pen_frac", "ground_pen", "foot_slide", "hand_contact", "wrist_err", "ankle_err", "joint_jerk", "root_jerk"]
    cols = body + (HAND_COLS if with_hand else [])
    higher = {"hand_contact", "contact_recall", "contact_precision", "contact_f1", "kp_contact_recall", "kp_contact_precision", "kp_contact_f1"}
    methods = [m for m in lanes if any(r["method"] == m for r in rows)]
    common = set.intersection(*[{r["seq"] for r in rows if r["method"] == m} for m in methods])
    L = [f"# Whole-body + hand pipeline on {C.DATASET.upper()}", "", f"Lanes: {', '.join(methods)}; sequences with every lane: {len(common)}", "",
         "| metric | " + " | ".join(methods) + " | best |", "|---|" + "---|" * (len(methods) + 1)]
    for c in cols:
        vals = {m: np.nanmean([r[c] for r in rows if r["method"] == m and r["seq"] in common]) for m in methods}
        ok = [m for m in methods if not np.isnan(vals[m])]
        best = (max if c in higher else min)(ok, key=lambda m: vals[m]) if ok else "-"
        L.append(f"| {c}{' (higher better)' if c in higher else ''} | " + " | ".join(f"{vals[m]:.4f}" for m in methods) + f" | {best} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", action="append", default=None, metavar="NAME=DIR[:SUFFIX]")
    ap.add_argument("--index", type=Path, default=C.DATA / "seqs_test.json")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--near", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-hand-data", action="store_true", help="dataset without human finger keypoints (OMOMO): body metrics only")
    ap.add_argument("--out", default="")
    ap.add_argument("--multi", action="store_true", help="hand metrics against every interacted hand-held object (data*/multi_objects.json), not only the primary")
    ap.add_argument("--merge", nargs="*", default=None, metavar="CSV", help="skip evaluation: merge these metrics csvs (later files override earlier rows per (seq, method)) and write summary<out>")
    args = ap.parse_args()
    if args.merge is not None:
        merged = {}
        for c in args.merge:
            with open(c) as f:
                for r in csv.DictReader(f):
                    row = {k: (v if k in ("seq", "object", "subject", "method", "fallback") else float(v)) for k, v in r.items()}
                    merged[(row["seq"], row["method"])] = row
        rows = list(merged.values())
        lane_names = []
        for r in rows:
            if r["method"] not in lane_names:
                lane_names.append(r["method"])
        order = ["umr_wuji", "s2", "s3", "s3p"]
        lanes = {m: None for m in sorted(lane_names, key=lambda m: order.index(m) if m in order else 99)}
        with open(C.RESULTS / f"pipe_metrics{args.out}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        md = summarize(rows, lanes, not args.no_hand_data)
        (C.RESULTS / f"pipe_summary{args.out}.md").write_text(md); print(md); return
    specs = args.lane or ["umr_wuji=" + str(C.RESULTS / "umr_wuji"), "s2=" + str(C.RESULTS / "pipe_s2"), "s3=" + str(C.RESULTS / "pipe_s3"), "s3p=" + str(C.RESULTS / "pipe_s3p")]
    lanes = {}
    for spec in specs:
        name, rest = spec.split("=", 1); dd, _, suffix = rest.partition(":")
        lanes[name] = ((HERE / dd) if not Path(dd).is_absolute() else Path(dd), suffix)
    index = json.loads(args.index.read_text())
    if args.seqs:
        index = [e for e in index if e["seq"] in set(args.seqs)]
    jobs = [(e["seq"], e, lanes, args.tau, args.near, not args.no_hand_data, args.multi) for e in index]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(eval_one, jobs):
            if "error" in r:
                print(f"[{r['seq']}] ERROR {r['error']}"); continue
            for row in r["rows"]:
                print(f"[{row['seq'][:50]}] {row['method']:9s} objpen={1e3*row['obj_pen_max']:.1f}mm handpen={row.get('hand_pen_mm', float('nan')):.1f}mm "
                      f"f1={row.get('contact_f1', float('nan')):.2f} tip={row.get('tip_pos_err_cm', float('nan')):.1f}cm flange={row.get('flange_pos_err_cm', float('nan')):.1f}cm wrist={1e2*row['wrist_err']:.1f}cm", flush=True)
            rows += r["rows"]
    if not rows:
        raise SystemExit("no rows")
    with open(C.RESULTS / f"pipe_metrics{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = summarize(rows, lanes, not args.no_hand_data)
    (C.RESULTS / f"pipe_summary{args.out}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
