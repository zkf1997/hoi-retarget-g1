"""SPIDER vs RL tracking on the 5 s interaction clips, scored identically.

Both are compared to the same reference (clip frame k = full motion frame start + k) with MuJoCo FK of the same robot:
root / wrist / fingertip position error, object position / rotation error; obj_ok = object within 3 cm and 30 deg,
hand_ok = obj_ok and mean fingertip error < 6 cm. RL rollouts stop at the bad-tracking termination; frames after it count
as not ok (SPIDER never terminates). Mean errors are over the frames each method actually produced.

    ~/software/venvs/spider/bin/python omomo_bench/spider_bridge/compare_w5.py --rl <model_XXXXX_w5_rollout.npz>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
PROC = HERE / "data" / "processed" / "{}" / "g1_wuji" / "humanoid_object"
CLIPS = HERE.parent / "wbt" / "data" / "motions_w5"
TIPS = [f"{c}_{f}_tip" for c in "lr" for f in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")]
WRISTS = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
N = 250  # 5 s at 50 Hz


def quat_angle(a, b):
    return 2 * np.arccos(np.abs(np.sum(a * b, -1)).clip(0, 1))


def score(m, q, r, n_valid):
    """q, r: (N, 83) simulated / reference qpos; frames >= n_valid are not produced (terminated)."""
    d = mujoco.MjData(m)
    ids = [m.body(b).id for b in TIPS + WRISTS]

    def fk(Q):
        out = np.zeros((len(Q), len(ids), 3))
        for i, x in enumerate(Q):
            d.qpos[:] = x
            mujoco.mj_kinematics(m, d)
            out[i] = d.xpos[ids]
        return out

    v = slice(0, n_valid)
    ps, pr = fk(q[v]), fk(r[v])
    tip = np.linalg.norm(ps[:, :10] - pr[:, :10], axis=-1).mean(1)
    wrist = np.linalg.norm(ps[:, 10:] - pr[:, 10:], axis=-1).mean(1)
    root = np.linalg.norm(q[v, :3] - r[v, :3], axis=1)
    obj = np.linalg.norm(q[v, 76:79] - r[v, 76:79], axis=1)
    rot = quat_angle(q[v, 79:83], r[v, 79:83])
    ok = np.zeros(N, bool)
    hok = np.zeros(N, bool)
    ok[v] = (obj < 0.03) & (rot < np.deg2rad(30))
    hok[v] = ok[v] & (tip < 0.06)
    fell = bool((q[v, 2] < 0.45).any())
    return {"tracked_s": n_valid / 50, "root_mm": 1000 * root.mean(), "wrist_mm": 1000 * wrist.mean(), "tip_mm": 1000 * tip.mean(),
            "obj_mm": 1000 * obj.mean(), "obj_ok": ok.mean(), "hand_ok": hok.mean(), "fell": fell,
            "full": n_valid >= N - 5 and not fell}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE.parent / "results_hiphi" / "compare_w5")
    ap.add_argument("--spider", default="hiphi_w5", help="SPIDER dataset: hiphi_w5 (stiff) or hiphi_w5_uniws (RL-matched physics)")
    a = ap.parse_args()
    proc = Path(str(PROC).format(a.spider))
    a.out.mkdir(parents=True, exist_ok=True)
    R = np.load(a.rl)
    rl_idx = {str(t): i for i, t in enumerate(R["motion"])}
    rl_dof = [str(x) for x in R["dof_names"]]
    rows = []
    for tdir in sorted(proc.iterdir()):
        tag = tdir.name
        sp = tdir / "0" / "trajectory_mjwp.npz"
        if not sp.exists() or tag not in rl_idx:
            continue
        m = mujoco.MjModel.from_xml_path(str(tdir / "scene.xml"))
        clip = np.load(CLIPS / f"{tag}.npz")
        ref = np.concatenate([clip["joint_pos"], clip["object_pos_w"], clip["object_quat_w"]], 1).astype(np.float64)
        names = [str(x) for x in clip["joint_names"]]
        # SPIDER: 100 Hz states, state i (after step i) ~ reference frame (i+1)/2
        o = np.load(sp)
        qs = o["qpos"].reshape(-1, m.nq)[::2][:N]
        rs = ref[np.minimum(np.arange(len(qs)) + 1, len(ref) - 1)]
        s = score(m, np.pad(qs, ((0, N - len(qs)), (0, 0)), mode="edge"), np.pad(rs, ((0, N - len(rs)), (0, 0)), mode="edge"), min(len(qs), N))
        # RL: rollout frames with reference index tstep; joints reordered to the motion order
        e = rl_idx[tag]
        perm = [rl_dof.index(n) for n in names]
        n_rl = min(int(R["active"][e].sum()), N)
        qr = np.concatenate([R["root"][e], R["dof"][e][:, perm], R["obj"][e]], 1)[:N]
        rr = ref[np.clip(R["tstep"][e][:N], 0, len(ref) - 1)]
        rl = score(m, qr, rr, max(n_rl, 1))
        rows.append({"motion": tag, **{f"spider_{k}": v for k, v in s.items()}, **{f"rl_{k}": v for k, v in rl.items()}})
    with open(a.out / "per_clip.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    keys = [("full", "clips run to the end (5 s, no fall)", "{:.0f}/%d" % len(rows), "sum"), ("tracked_s", "mean tracked time [s]", "{:.1f}", "mean"),
            ("obj_ok", "object on target (3 cm / 30 deg), share of 5 s", "{:.0%}", "mean"),
            ("hand_ok", "+ fingertips < 6 cm, share of 5 s", "{:.0%}", "mean"),
            ("root_mm", "root error [mm] (median over clips)", "{:.0f}", "median"), ("wrist_mm", "wrist error [mm] (median)", "{:.0f}", "median"),
            ("tip_mm", "fingertip error [mm] (median)", "{:.0f}", "median"), ("obj_mm", "object error [mm] (median)", "{:.0f}", "median")]
    md = [f"# SPIDER vs RL on {len(rows)} 5 s interaction clips", "", f"SPIDER dataset: `{a.spider}`; RL rollout: `{a.rl}`", "", "| metric | SPIDER | RL |", "|---|---|---|"]
    for k, label, fmt, agg in keys:
        f_ = {"sum": np.sum, "mean": np.mean, "median": np.median}[agg]
        md.append(f"| {label} | {fmt.format(f_([r['spider_' + k] for r in rows]))} | {fmt.format(f_([r['rl_' + k] for r in rows]))} |")
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
