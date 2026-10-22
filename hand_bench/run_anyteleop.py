#!/usr/bin/env python3
"""AnyTeleop-style retargeting (dex-retargeting, Qin et al. 2023) on the prepared HUMOTO sequences.

    python run_anyteleop.py --hand brainco [--optimizer vector|dexpilot] --workers 16

`vector`   (method name `anyteleop`):          AnyTeleop's teleop optimiser: wrist->fingertip vectors, Huber loss, SLSQP, joint limits.
`dexpilot` (method name `anyteleop_dexpilot`): dex-retargeting's DexPilot variant (adds thumb-finger projection prior).
Human keypoints are expressed in the per-frame canonical hand frame; the wrist->tip vectors are compared with the robot
flange->tip vectors scaled by robot/human hand size (median wrist->middle-tip length of the sequence), i.e. the same
`scaling_factor` convention as the official configs. The robot flange stays on the human wrist (no base optimisation).
The BrainCo distal <mimic> joints are handled by dex-retargeting's MimicJointKinematicAdaptor. No low-pass filter (alpha = 1).
Output: results/anyteleop[_dexpilot]/<hand>/<seq>__<side>.npz
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_common as bc  # noqa: E402
from hand_model import HUMAN_TIPS, HandModel, flange_link_name, human_to_robot_scale, tip_link_names  # noqa: E402


def build(hand: str, side: str, optimizer: str, scale: float):
    from dex_retargeting.retargeting_config import RetargetingConfig

    model = HandModel(hand, side)
    flange, tips = flange_link_name(hand, side), tip_link_names(hand, side)
    cfg = dict(urdf_path=str(model.urdf), target_joint_names=model.active, scaling_factor=scale, low_pass_alpha=1.0)
    if optimizer == "vector":
        cfg.update(type="vector", target_origin_link_names=[flange] * 5, target_task_link_names=tips,
                   target_link_human_indices=np.array([[0] * 5, HUMAN_TIPS]))
    else:
        cfg.update(type="dexpilot", wrist_link_name=flange, finger_tip_link_names=tips)
    return model, RetargetingConfig(**cfg).build()


def run_one(hand: str, seq: str, side: str, optimizer: str, frames: int | None, overwrite: bool) -> str:
    method = "anyteleop" if optimizer == "vector" else "anyteleop_dexpilot"
    if not overwrite and bc.result_path(method, hand, seq, side).exists():
        return f"skip {seq} {side}"
    d = bc.load_seq(seq)
    kp_h = bc.hand_frame_kp(d, side)
    model0 = HandModel(hand, side)
    scale = human_to_robot_scale(model0, kp_h)
    model, ret = build(hand, side, optimizer, scale)
    idx = ret.optimizer.target_link_human_indices
    names = ret.joint_names  # pinocchio dof order
    sel = [names.index(n) for n in model.active]
    T = d["n_frames"] if frames is None else min(frames, d["n_frames"])
    q = np.zeros((T, model.n_active))
    t0 = time.perf_counter()
    for t in range(T):
        kp = kp_h[t]
        ref = kp[idx[1]] - kp[idx[0]]
        q[t] = ret.retarget(ref)[sel]
    dt = (time.perf_counter() - t0) / T
    q = model.clip(q)
    base_pos = np.zeros((T, 3))
    base_quat = np.tile([1.0, 0, 0, 0], (T, 1))
    bc.save_result(method, hand, seq, side, q_active=q, joint_names=model.active, base_pos_h=base_pos, base_quat_h=base_quat,
                   runtime_per_frame=dt, extra={"scale": np.float32(scale), "n_frames_done": T, "optimizer": optimizer})
    return f"{seq} {side}: {T} frames, {dt*1000:.1f} ms/frame, scale {scale:.3f}"


def _job(a):
    try:
        return run_one(*a)
    except Exception as e:  # noqa: BLE001
        import traceback

        return f"FAILED {a[1]} {a[2]}: {e!r}\n{traceback.format_exc()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", choices=bc.HANDS, required=True)
    ap.add_argument("--optimizer", choices=["vector", "dexpilot"], default="vector")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--sides", nargs="*", default=list(bc.SIDES))
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    seqs = a.seqs or bc.list_prepared()
    jobs = [(a.hand, s, side, a.optimizer, a.frames, a.overwrite) for s in seqs for side in a.sides]
    t0 = time.time()
    if a.workers <= 1:
        for j in jobs:
            print(_job(j), flush=True)
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(_job, jobs):
                print(r, flush=True)
    print(f"done {len(jobs)} jobs in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
