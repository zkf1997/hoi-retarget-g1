"""Collect finished SPIDER optimizations into the rollout format of wbt/viser_rollout.py, plus per-sequence metrics.

    ~/software/venvs/spider/bin/python omomo_bench/spider_bridge/collect.py [--out results/spider_hiphi]
        -> <out>/spider_rollout.npz, <out>/spider_per_motion.csv, <out>/summary.md

SPIDER runs at sim_dt 0.01; trajectories are downsampled to the 50 Hz reference. Errors per frame: root position, body /
finger joint angles, fingertip and wrist positions (MuJoCo FK of simulated vs reference qpos), object position / rotation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
PROC = HERE / "data" / "processed" / "{dataset}" / "g1_wuji" / "humanoid_object"
TIPS = [f"{c}_{f}_tip" for c in "lr" for f in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")]
WRISTS = ["left_wrist_yaw_link", "right_wrist_yaw_link"]


def quat_angle(a, b):
    d = np.abs(np.sum(a * b, -1)).clip(0, 1)
    return 2 * np.arccos(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "results_hiphi" / "spider")
    ap.add_argument("--dataset", default="hiphi", help="hiphi (stiff actuators), hiphi_real, hiphi_realff (see build_hiphi.py)")
    ap.add_argument("--max-frames", type=int, default=0, help="> 0: score only the first N 50 Hz frames (e.g. 250 for 5 s clips)")
    a = ap.parse_args()
    proc = Path(str(PROC).format(dataset=a.dataset))
    a.out.mkdir(parents=True, exist_ok=True)
    rows, R = [], {k: [] for k in ("motion", "root", "dof", "obj", "tstep", "active")}
    runs = sorted(p for p in proc.glob("*/0/trajectory_mjwp.npz"))
    T_max = 0
    data = []
    for p in runs:
        tag = p.parent.parent.name
        o, ref = np.load(p), np.load(p.parent / "trajectory_kinematic.npz")["qpos"]
        q = o["qpos"].reshape(-1, o["qpos"].shape[-1])[::2]  # 100 Hz -> 50 Hz
        if a.max_frames > 0:
            q = q[:a.max_frames]
        n = len(q)
        idx = np.minimum(np.arange(n) + 1, len(ref) - 1)
        info = json.loads((p.parent.parent / "task_info.json").read_text())
        start = int(info.get("start_frame", 0))  # clip offset into the full motion (build_hiphi.py --window)
        r = ref[idx]
        m = mujoco.MjModel.from_xml_path(str(p.parent.parent / "scene.xml"))
        d = mujoco.MjData(m)
        tip_ids = [m.body(b).id for b in TIPS]
        wr_ids = [m.body(b).id for b in WRISTS]

        def fk(qs):
            out = np.zeros((len(qs), len(tip_ids) + len(wr_ids), 3))
            for i, qq in enumerate(qs):
                d.qpos[:] = qq
                mujoco.mj_kinematics(m, d)
                out[i] = d.xpos[tip_ids + wr_ids]
            return out

        ps, pr = fk(q), fk(r)
        tip = np.linalg.norm(ps[:, :10] - pr[:, :10], axis=-1).mean(1)
        wrist = np.linalg.norm(ps[:, 10:] - pr[:, 10:], axis=-1).mean(1)
        obj = np.linalg.norm(q[:, 76:79] - r[:, 76:79], axis=1)
        obj_rot = quat_angle(q[:, 79:83], r[:, 79:83])
        root = np.linalg.norm(q[:, :3] - r[:, :3], axis=1)
        ok = (obj < 0.03) & (obj_rot < np.deg2rad(30))
        rows.append({
            "motion": tag, "start_s": start / 50, "frames": n, "frames_total": len(ref), "root_mm": 1000 * root.mean(),
            "body_joint_deg": np.rad2deg(np.abs(q[:, 7:36] - r[:, 7:36]).mean()), "finger_joint_deg": np.rad2deg(np.abs(q[:, 36:76] - r[:, 36:76]).mean()),
            "wrist_mm": 1000 * wrist.mean(), "fingertip_mm": 1000 * tip.mean(), "object_mm": 1000 * obj.mean(), "object_max_mm": 1000 * obj.max(),
            "object_rot_deg": np.rad2deg(obj_rot.mean()), "obj_ok_frac": ok.mean(), "hand_ok_frac": (ok & (tip < 0.06)).mean(),
            "fell": bool((q[:, 2] < 0.45).any()), "plan_iters": float(o["opt_steps"].mean()),
        })
        data.append((tag, q, idx + start))
        T_max = max(T_max, n)
    if not data:
        print("no finished SPIDER runs under", proc)
        return
    for tag, q, idx in data:
        n = len(q)
        pad = lambda x: np.concatenate([x, np.repeat(x[-1:], T_max - n, 0)]) if n < T_max else x
        R["motion"].append(tag)
        R["root"].append(pad(q[:, :7])); R["dof"].append(pad(q[:, 7:76])); R["obj"].append(pad(q[:, 76:83]))
        R["tstep"].append(pad(idx)); R["active"].append(np.arange(T_max) < n)
    dof_names = [str(x) for x in np.load(HERE.parent / "wbt" / "data" / "motions" / f"{data[0][0]}.npz")["joint_names"]]
    np.savez_compressed(a.out / "spider_rollout.npz", motion=np.array(R["motion"]), dof_names=np.array(dof_names), phase=0.0,
                        **{k: np.stack(R[k]) for k in ("root", "dof", "obj", "tstep", "active")})
    with open(a.out / "spider_per_motion.csv", "w", newline="") as f:
        # field names compatible with viser_rollout.py (completed / obj_ok_frac / *_pos_mean)
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["completed", "frames_tracked", "body_pos_mean", "wrist_pos_mean", "fingertip_pos_mean", "object_pos_mean"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "completed": str(r["frames"] >= min(r["frames_total"] - 100, a.max_frames or 10**9) and not r["fell"]), "frames_tracked": r["frames"],
                        "body_pos_mean": r["root_mm"], "wrist_pos_mean": r["wrist_mm"], "fingertip_pos_mean": r["fingertip_mm"], "object_pos_mean": r["object_mm"]})
    md = ["| sequence | frames | root mm | body joint ° | finger joint ° | wrist mm | fingertip mm | object mm (max) | object ° | object ok | + hand ok | fell |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['motion']} | {r['frames']}/{r['frames_total']} | {r['root_mm']:.0f} | {r['body_joint_deg']:.1f} | {r['finger_joint_deg']:.1f} | "
                  f"{r['wrist_mm']:.0f} | {r['fingertip_mm']:.0f} | {r['object_mm']:.0f} ({r['object_max_mm']:.0f}) | {r['object_rot_deg']:.0f} | "
                  f"{r['obj_ok_frac']:.0%} | {r['hand_ok_frac']:.0%} | {'yes' if r['fell'] else 'no'} |")
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
