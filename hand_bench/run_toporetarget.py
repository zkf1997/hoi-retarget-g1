#!/usr/bin/env python3
"""TopoRetarget (third_party/toporetarget, unofficial implementation of Wu et al. 2026) on the prepared HUMOTO sequences.

    python run_toporetarget.py --hand brainco --workers 16
    python run_toporetarget.py --hand wuji --seqs <seq> --frames 30      # quick test

Per (sequence, side): human 21 keypoints and the hand-held object are expressed in the per-frame canonical hand frame
(wrist at the origin), then `TopoRetargetPipeline.retarget()` is run with a MuJoCo-backed RobotHand (hand_model.HandModel):
bone-direction initialisation + interaction-mesh Laplacian optimisation, base pose (flange) and active joints are optimised.
Deviations from the upstream code, all documented here:
  * the human keypoints are used unscaled (the Laplacian/base-pose optimisation absorbs the size difference);
  * joint limits are enforced as L-BFGS-B bounds (upstream calls scipy without bounds);
  * `_estimate_joints_from_keypoints` (hard-coded for 3 joints per finger) is replaced by an open-hand initial guess;
  * `compute_laplacian` is replaced by a numerically identical vectorised version (upstream loops in Python).
Output: results/toporetarget/<hand>/<seq>__<side>.npz  (see bench_common.save_result).
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
from scipy.optimize import minimize as _scipy_minimize
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "third_party" / "toporetarget"))

import bench_common as bc  # noqa: E402
from hand_model import HandModel  # noqa: E402
from toporetarget.core import retarget as tr  # noqa: E402

METHOD = "toporetarget"


# --- patches ----------------------------------------------------------------------------------------------------------
def _compute_laplacian_vec(vertices, edges, neighbors, weights):
    lap = np.zeros_like(vertices)
    if not edges:
        return lap
    e = np.asarray(edges)
    w = np.array([weights.get((int(i), int(j)), 0.0) for i, j in edges])
    np.add.at(lap, e[:, 0], w[:, None] * (vertices[e[:, 0]] - vertices[e[:, 1]]))
    return lap


_BOUNDS: list | None = None


def _bounded_minimize(fun, x0, *args, **kw):
    if _BOUNDS is not None and len(_BOUNDS) == len(x0) and "bounds" not in kw:
        kw["bounds"] = _BOUNDS
        x0 = np.array([min(max(x, lo if lo is not None else -np.inf), hi if hi is not None else np.inf) for x, (lo, hi) in zip(x0, _BOUNDS)])
    return _scipy_minimize(fun, x0, *args, **kw)


tr.compute_laplacian = _compute_laplacian_vec
tr.minimize = _bounded_minimize


class MujocoTopoHand(tr.RobotHand):
    """toporetarget RobotHand backed by hand_model.HandModel; base_pos is the flange (= keypoint 0)."""

    def __init__(self, model: HandModel):
        super().__init__(n_joints=model.n_active)
        self.model = model
        self.set_joint_limits(model.limits)

    def forward_kinematics(self, joints, base_pos, base_quat):
        kp = self.model.keypoints(np.asarray(joints, dtype=np.float64)) - self.model.flange
        q = np.asarray(base_quat, dtype=np.float64)
        n = np.linalg.norm(q)
        rot = R.from_quat([q[1], q[2], q[3], q[0]]) if n > 1e-9 else R.identity()
        return rot.apply(kp) + np.asarray(base_pos, dtype=np.float64)


class Pipeline(tr.TopoRetargetPipeline):
    def _estimate_joints_from_keypoints(self, keypoints):
        return self.robot.model.clip(np.zeros(self.robot.n_joints))


def run_one(hand: str, seq: str, side: str, frames: int | None, cfg: dict, overwrite: bool) -> str:
    global _BOUNDS
    if not overwrite and bc.result_path(METHOD, hand, seq, side).exists():
        return f"skip {seq} {side}"
    d = bc.load_seq(seq)
    k = bc.main_object(d, side)
    if k < 0:
        return f"no hand-held object for {seq} {side}"
    model = HandModel(hand, side)
    robot = MujocoTopoHand(model)
    _BOUNDS = [(float(lo), float(hi)) for lo, hi in model.limits] + [(None, None)] * 7

    kp_h = bc.hand_frame_kp(d, side)
    p, rot, sc = bc.object_in_hand_frame(d, side, k)
    mesh = bc.load_object_visual(d["objects"][k])
    mesh.vertices = mesh.vertices * float(np.median(sc))
    quat = R.from_matrix(rot).as_quat()[:, [3, 0, 1, 2]]  # wxyz as expected by toporetarget
    obj_pose = np.concatenate([p, quat], axis=1)
    T = d["n_frames"] if frames is None else min(frames, d["n_frames"])

    pipe = Pipeline(robot, tr.RetargetConfig(**cfg))
    t0 = time.perf_counter()
    joints, base = pipe.retarget(kp_h[:T], obj_pose[:T], mesh, verbose=False)
    dt = (time.perf_counter() - t0) / T
    joints = model.clip(joints)
    bq = base[:, 3:] / np.linalg.norm(base[:, 3:], axis=1, keepdims=True)
    bc.save_result(METHOD, hand, seq, side, q_active=joints, joint_names=model.active, base_pos_h=base[:, :3], base_quat_h=bq,
                   runtime_per_frame=dt, extra={"object": d["objects"][k], "n_frames_done": T, "config": np.array(str(cfg))})
    drift = np.linalg.norm(base[:, :3], axis=1)
    return f"{seq} {side}: {T} frames, {dt:.2f} s/frame, base drift mean {drift.mean()*100:.1f} cm"


def _job(a):
    try:
        return run_one(*a)
    except Exception as e:  # noqa: BLE001
        import traceback

        return f"FAILED {a[1]} {a[2]}: {e!r}\n{traceback.format_exc()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", choices=bc.HANDS, required=True)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--sides", nargs="*", default=list(bc.SIDES))
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n-obj-samples", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=200)
    a = ap.parse_args()
    cfg = {"n_obj_samples": a.n_obj_samples, "max_iter": a.max_iter}
    seqs = a.seqs or bc.list_prepared()
    jobs = [(a.hand, s, side, a.frames, cfg, a.overwrite) for s in seqs for side in a.sides]
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
