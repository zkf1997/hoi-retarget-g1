#!/usr/bin/env python3
"""Wrist orientation error of result lanes vs the human canonical hand frame (x distal, y palm normal, z thumb side).
    BENCH_DATASET=humoto MUJOCO_GL=egl python wrist_orientation_stats.py --lane umr_wuji=results_humoto/umr_wuji [--seqs ...]
Per lane and side: mean / median / p90 angle, interacting vs free frames, mean rotvec bias in the hand frame, residual after
removing the bias, fraction of frames > 30 deg. Robot frame = <side>_wrist_yaw_link (the hand URDFs' flange convention)."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C
from scipy.spatial.transform import Rotation as Rot

HB = C.HERE.parent / "hand_bench" / ("data" if C.DATASET == "humoto" else f"data_{C.DATASET}") / "seqs"


SMPLX_TIPS = {"left": [27, 30, 33, 36, 39], "right": [42, 45, 48, 51, 54]}


def load_hand_bench(seq, src):
    """hand_bench sequence (human canonical hand frames, keypoints, object positions); OMOMO has none, so build the same
    fields from the SMPL-X joints of the source (flat hands: distal finger joints stand in for the tips)."""
    p = HB / f"{seq}.npz"
    if p.exists():
        return np.load(p, allow_pickle=True)
    from omomo_hand_repair import human_hand_frames
    J = src["joints55"].astype(np.float64); T = len(J); out = {"objects": np.array([str(src["object"])]), "n_frames": T,
                                                               "obj_pos": src["obj_pos"][:, None].astype(np.float64)}
    for sd in ("left", "right"):
        out[f"hand_R_{sd}"] = human_hand_frames(J, sd)[1]
        kp = np.zeros((T, 21, 3)); kp[:, [4, 8, 12, 16, 20]] = J[:, SMPLX_TIPS[sd]]; out[f"hand_kp_{sd}"] = kp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", action="append", required=True, metavar="NAME=DIR[:SUFFIX]")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="npz with per-frame rotvecs")
    a = ap.parse_args()
    lanes = {x.split("=")[0]: (Path(x.split("=", 1)[1].split(":")[0]), x.split("=", 1)[1].partition(":")[2]) for x in a.lane}
    index = json.load(open(C.DATA / "seqs_test.json"))
    if a.seqs:
        index = [e for e in index if e["seq"] in set(a.seqs)]
    stats = {ln: {sd: {"rv": [], "inter": [], "seq": []} for sd in ("left", "right")} for ln in lanes}
    for e in index:
        seq = e["seq"]; src = np.load(C.DATA / "source" / f"{seq}.npz", allow_pickle=True); hb = load_hand_bench(seq, src)
        obj = str(src["object"]); oi = [str(o) for o in hb["objects"]].index(obj)
        for ln, (rdir, suffix) in lanes.items():
            p = rdir / f"{seq}{suffix}.npz"
            if not p.exists():
                continue
            r = np.load(p, allow_pickle=True)
            if "human_joints" in r:  # OmniRetarget result: qpos (T, 43) = 7 root + 29 body + object, model = the MJCF the run used
                q = r["qpos"][:, :36].astype(np.float64); T = min(len(q), int(hb["n_frames"]), len(src["trans"]))
                mm = C.omni_model(str(r["hand"]) if "hand" in r else "none", str(r["scaling"]) if "scaling" in r else "omni", C.omni_subject(r, e["subject"]), obj)
            else:
                s = float(np.asarray(r["smpl_scale"]).reshape(-1)[0])
                ms = float(np.asarray(r["object_mesh_scale"]).reshape(-1)[0]) if "object_mesh_scale" in r else s
                q = r["qpos"].astype(np.float64); T = min(len(q), int(hb["n_frames"]), len(src["trans"]))
                mm = C.augmented_model(C.umr_robot_xml(r), obj, ms)  # was always the Wuji MJCF: BrainCo lanes got wrong joint order
            d = mujoco.MjData(mm)
            bn = [mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(mm.nbody)]
            wr = {sd: bn.index(f"{sd}_wrist_yaw_link") for sd in ("left", "right")}
            R_rob = {sd: np.zeros((T, 3, 3)) for sd in wr}
            for t in range(T):
                d.qpos[:q.shape[1]] = q[t]; mujoco.mj_kinematics(mm, d)
                for sd, b in wr.items(): R_rob[sd][t] = d.xmat[b].reshape(3, 3)
            for sd in wr:
                Rh = hb[f"hand_R_{sd}"][:T].astype(np.float64); kp = hb[f"hand_kp_{sd}"][:T]; op = hb["obj_pos"][:T, oi]
                inter = np.linalg.norm(kp[:, [4, 8, 12, 16, 20]] - op[:, None], axis=2).min(1) < 0.15
                rv = Rot.from_matrix(np.einsum("tji,tjk->tik", Rh, R_rob[sd])).as_rotvec()
                st = stats[ln][sd]; st["rv"].append(rv); st["inter"].append(inter); st["seq"].append(np.full(T, seq))
        C._model_cache.clear()
    print(f"{len(index)} sequences")
    for ln in lanes:
        both = []
        for sd in ("left", "right"):
            st = stats[ln][sd]
            if not st["rv"]:
                continue
            rv = np.concatenate(st["rv"]); ang = np.degrees(np.linalg.norm(rv, axis=1)); inter = np.concatenate(st["inter"]); both.append(ang)
            bias = rv.mean(0); resid = np.degrees(np.linalg.norm((Rot.from_rotvec(bias).inv() * Rot.from_rotvec(rv)).as_rotvec(), axis=1))
            print(f"{ln:16s} {sd:5s} mean {ang.mean():5.1f} median {np.median(ang):5.1f} p90 {np.percentile(ang, 90):5.1f} | interacting {ang[inter].mean():5.1f} free {ang[~inter].mean():5.1f} | "
                  f"bias [x,y,z] {np.degrees(bias).round(1)} (|{np.degrees(np.linalg.norm(bias)):.1f}|) residual {resid.mean():.1f} | >30deg {np.mean(ang > 30) * 100:.1f}%")
        if both:
            ang = np.concatenate(both); print(f"{ln:16s} both  mean {ang.mean():5.1f} median {np.median(ang):5.1f} <10deg {np.mean(ang < 10) * 100:.0f}% <20deg {np.mean(ang < 20) * 100:.0f}% >30deg {np.mean(ang > 30) * 100:.1f}%")
        seqs = np.concatenate(stats[ln]["right"]["seq"]); ang = np.degrees(np.linalg.norm(np.concatenate(stats[ln]["right"]["rv"]), axis=1))
        per = sorted({s_: ang[seqs == s_].mean() for s_ in np.unique(seqs)}.items(), key=lambda kv: -kv[1])
        print(f"{ln:16s} right worst: " + ", ".join(f"{s_[:34]} {v:.0f}" for s_, v in per[:4]))
    if a.out:
        np.savez(a.out, **{f"{ln}_{sd}_{k}": np.concatenate(stats[ln][sd][k]) for ln in lanes for sd in ("left", "right") for k in ("rv", "inter") if stats[ln][sd][k]})


if __name__ == "__main__":
    main()
