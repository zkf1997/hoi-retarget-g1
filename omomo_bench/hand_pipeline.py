#!/usr/bin/env python3
"""Decoupled whole-body + hand pipeline (cf. rew35860/holosoma feature/wuji-hand-humoto): UMR body -> AnyTeleop fingers -> refinement.

    BENCH_DATASET=humoto python hand_pipeline.py --stage s3p --workers 16        # results_humoto/pipe_s3p/<seq>.npz
    BENCH_DATASET=hiphi  python hand_pipeline.py --stage s2 s3 s3p --workers 16

Stages (each written as its own result lane, UMR npz layout: qpos [T, 7 + 69] in `robot_joint_names` order, smpl_scale, fps):
  umr_wuji (input)  UMR whole-body retarget with the Wuji fingers pinned at hand_rest_pose.json
  s2                + AnyTeleop finger joints (hand_bench/results*/anyteleop/wuji/<seq>__<side>.npz) substituted per frame
  s3                + finger refinement: per frame, per hand, a small QP over the 20 finger joints that stays close to the
                    AnyTeleop pose, pulls the fingertips the human had in contact onto the object surface, and enforces
                    hand/object non-penetration (linearised MuJoCo geom distances with a soft slack); arms untouched
  v2                coupled arm + finger QP per hand and frame (replaces s3 + s3p): flange pose target, contact fingertips
                    (hysteresis, size-remapped to the robot hand), AnyTeleop / UMR priors, previous-frame + acceleration
                    terms, hand AND forearm vs object non-penetration with slack
  v3                v2 + DexWeave-inspired changes: corrected hand-size factor (ratio / s) for the contact targets, interaction
                    weight alpha (5 cm, 4-frame persistence) gating the wrist position anchor and strengthening the UMR arm prior
                    when the hand is free, fingertip priority weights (thumb 1.45, index 1.25, others 0.78), arm temporal term on
                    the correction relative to the UMR init, 4 inner iterations
  s3p               + arm placement before the finger refinement: 7 arm joints per side are re-solved (Gauss-Newton pose step
                    inside a QP with joint limits and hand/object non-penetration, regularised to the UMR pose) so the flange
                    (`{l,r}_mount`) sits at the scaled human wrist with the canonical hand-frame orientation, i.e. where
                    AnyTeleop assumed the hand to be, without pushing the palm into the object
Human hand data: hand_bench/data*/seqs/<seq>.npz (21 keypoints per hand, hand_R canonical frame, Z-up metres). The robot
world is the UMR-scaled world (human x s, object mesh and pose x s). Contact targets: human fingertip within --tau of the
object surface -> nearest surface point, mapped into the robot world.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C  # noqa: E402  (BENCH_DATASET-aware paths + model builders)

HERE = Path(__file__).resolve().parent
HB_DATA = HERE.parent / "hand_bench" / ("data" if C.DATASET == "humoto" else f"data_{C.DATASET}")
HB_RES = HERE.parent / "hand_bench" / ("results" if C.DATASET == "humoto" else f"results_{C.DATASET}")
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
TIPS_H = [4, 8, 12, 16, 20]  # MediaPipe fingertip indices (thumb, index, middle, ring, pinky)
TIP_BODIES = ["thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]
FINGER_KEYS = ("thumb", "index", "middle", "ring", "pinky")


# ----------------------------------------------------------------------------- model helpers
class Robot:
    def __init__(self, obj, mesh_scale: float):
        import mujoco
        self.mj = mujoco
        self.objs = [obj] if isinstance(obj, str) else list(obj)  # primary first; every object is a free body in the model
        obj = self.objs[0]
        self.m = C.augmented_model(C.UMR_ROBOT_XML["wuji"], self.objs if len(self.objs) > 1 else obj, mesh_scale)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.jname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        self.qadr = {n: int(m.jnt_qposadr[j]) for j, n in enumerate(self.jname) if n}
        self.dadr = {n: int(m.jnt_dofadr[j]) for j, n in enumerate(self.jname) if n}
        self.jrange = {n: (float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])) if m.jnt_limited[j] else (-np.inf, np.inf)
                       for j, n in enumerate(self.jname) if n}
        self.obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{obj}_link")
        self.obj_qadr = int(m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_bodyid[j] == self.obj_body][0]])
        bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        self.body = {n: i for i, n in enumerate(bname)}
        self.obj_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{o}_link") for o in self.objs]
        self.obj_qadrs = [int(m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_bodyid[j] == b][0]]) for b in self.obj_bodies]
        # non-penetration / contacts consider every object of the sequence
        self.obj_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.obj_bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.side = {}
        for side in ("left", "right"):
            c = side[0]
            hand_bodies = [i for i, n in enumerate(bname) if n.startswith(f"{c}_")]
            self.side[side] = {
                "arm": [f"{side}_{a}_joint" for a in ARM],
                "fingers": [f"{c}_thumb_cmc_flex", f"{c}_thumb_cmc_abd", f"{c}_thumb_mcp", f"{c}_thumb_ip"] +
                           [f"{c}_{f}_{j}" for f in ("index_finger", "middle_finger", "ring_finger", "pinky") for j in ("mcp_flex", "mcp_abd", "pip", "dip")],
                "flange": self.body[f"{c}_mount"],
                "wrist": self.body[f"{side}_wrist_yaw_link"],
                "tips": [self.body[f"{c}_{t}"] for t in TIP_BODIES],
                "kp": [self.body[n] for n in hand_kp_bodies(c)],
                "hand_geoms": [g for g in range(m.ngeom) if m.geom_bodyid[g] in hand_bodies and (m.geom_contype[g] or m.geom_conaffinity[g])],
                # palm = mount + hand root body (no finger links): used by the arm-placement non-penetration constraint
                "palm_geoms": [g for g in range(m.ngeom) if bname[m.geom_bodyid[g]] in (f"{c}_mount", f"{c}_wrist") and (m.geom_contype[g] or m.geom_conaffinity[g])],
                # forearm (elbow .. wrist links): the arm moves these too, so the coupled solve keeps them out of the object as well
                "forearm_geoms": [g for g in range(m.ngeom) if bname[m.geom_bodyid[g]] in (f"{side}_elbow_link", f"{side}_wrist_roll_link", f"{side}_wrist_pitch_link", f"{side}_wrist_yaw_link")
                                  and (m.geom_contype[g] or m.geom_conaffinity[g])],
            }

        self.d.qpos[:] = 0; self.d.qpos[3] = 1; self.mj.mj_forward(self.m, self.d)
        for side in ("left", "right"):
            S = self.side[side]
            pw, Rw = self.d.xpos[S["wrist"]], self.d.xmat[S["wrist"]].reshape(3, 3)
            S["flange_offset"] = Rw.T @ (self.d.xpos[S["flange"]] - pw)  # mount origin in the wrist_yaw frame (~[0.0415, 0, 0])

    def set(self, q):
        self.d.qpos[:] = q
        self.mj.mj_forward(self.m, self.d)

    def body_pose(self, b):
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def jac_body(self, b):
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        self.mj.mj_jacBody(self.m, self.d, jp, jr, b)
        return jp, jr

    def jac_point(self, p, b):
        jp = np.zeros((3, self.m.nv))
        self.mj.mj_jac(self.m, self.d, jp, None, p, b)
        return jp

    def near_obj_geoms(self, gh, thresh):
        """Object geoms whose bounding sphere comes within `thresh` of geom gh's bounding sphere (conservative broad phase in
        front of mj_geomDistance; furniture with 30-400 convex pieces made the exhaustive pair loop the bottleneck)."""
        og = getattr(self, "_og", None)
        if og is None:
            og = self._og = np.asarray(self.obj_geoms, dtype=int)
        if len(og) == 0:
            return og
        dd = np.linalg.norm(self.d.geom_xpos[og] - self.d.geom_xpos[gh], axis=1) - self.m.geom_rbound[og] - self.m.geom_rbound[gh]
        return og[dd < thresh]

    def dense_points(self, side):
        """(len(DENSE_POINTS), 3) world positions after set(q)."""
        return dense_points_from_kp(self.d.xpos[self.side[side]["kp"]])

    def dense_jac(self, side, i):
        idx, coef = DENSE_POINTS[i]
        bodies = self.side[side]["kp"]
        return sum(c * self.jac_point(self.d.xpos[bodies[j]], bodies[j]) for j, c in zip(idx, coef))


def so3_log(R):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R).as_rotvec()


# ----------------------------------------------------------------------------- stage 3a: arm placement
def place_arm(rb: Robot, q, side, p_target, R_target, q_umr, w_pos=1.0, w_rot=0.3, w_reg=0.05, iters=8, lam=1e-3):
    """Target = canonical human hand frame (hand_model convention: frame = wrist_yaw_link, flange/keypoint 0 on the human wrist).
    p_target is the scaled human wrist keypoint, so the wrist_yaw target is p_target - R_target @ flange_offset."""
    S = rb.side[side]
    p_target = np.asarray(p_target) - R_target @ S["flange_offset"]
    qa = [rb.qadr[n] for n in S["arm"]]; da = [rb.dadr[n] for n in S["arm"]]
    lo = np.array([rb.jrange[n][0] for n in S["arm"]]); hi = np.array([rb.jrange[n][1] for n in S["arm"]])
    q = q.copy()
    for _ in range(iters):
        rb.set(q)
        p, R = rb.body_pose(S["wrist"])
        jp, jr = rb.jac_body(S["wrist"])
        e = np.concatenate([w_pos * (p_target - p), w_rot * so3_log(R_target @ R.T)])
        J = np.vstack([w_pos * jp[:, da], w_rot * jr[:, da]])
        H = J.T @ J + (lam + w_reg) * np.eye(7)
        g = J.T @ e + w_reg * (q_umr[qa] - q[qa])
        dq = np.linalg.solve(H, g)
        dq = np.clip(dq, -0.3, 0.3)
        q[qa] = np.clip(q[qa] + dq, lo, hi)
    rb.set(q)
    p, R = rb.body_pose(S["wrist"])
    return q, float(np.linalg.norm(p_target - p)), float(np.degrees(np.linalg.norm(so3_log(R_target @ R.T))))


def place_arm_qp(rb: Robot, q, side, p_target, R_target, q_umr, w_pos=1.0, w_rot=0.3, w_reg=1e-4, iters=15, lam=1e-3,
                 w_slack=1e4, margin=0.002, pair_thresh=0.02, max_dq=0.3, geom_set="palm_geoms"):
    """Arm placement as a QP: Gauss-Newton pose step towards the canonical hand frame, subject to joint limits and
    linearised hand/object non-penetration (all hand geoms incl. the palm, with a soft slack) so the palm does not get
    pushed into large objects (HiPHI boxes / buckets) by the wrist target alone."""
    import clarabel
    from scipy import sparse
    S = rb.side[side]
    p_target = np.asarray(p_target) - R_target @ S["flange_offset"]
    qa = np.array([rb.qadr[n] for n in S["arm"]]); da = np.array([rb.dadr[n] for n in S["arm"]])
    lo = np.array([rb.jrange[n][0] for n in S["arm"]]); hi = np.array([rb.jrange[n][1] for n in S["arm"]])
    q = q.copy(); n = 7
    fromto = np.zeros(6)
    for _ in range(iters):
        rb.set(q)
        p, R = rb.body_pose(S["wrist"])
        jp, jr = rb.jac_body(S["wrist"])
        e = np.concatenate([w_pos * (p_target - p), w_rot * so3_log(R_target @ R.T)])
        J = np.vstack([w_pos * jp[:, da], w_rot * jr[:, da]])
        P = J.T @ J + (lam + w_reg) * np.eye(n)
        g = J.T @ e + w_reg * (q_umr[qa] - q[qa])
        rows, rhs = [], []
        for gh in S[geom_set]:
            bh = int(rb.m.geom_bodyid[gh])
            for go in rb.near_obj_geoms(gh, pair_thresh):
                dist = rb.mj.mj_geomDistance(rb.m, rb.d, gh, go, pair_thresh, fromto)
                if dist >= pair_thresh:
                    continue
                nrm = fromto[3:] - fromto[:3]; nn = np.linalg.norm(nrm)
                if nn < 1e-9:
                    continue
                nrm = nrm / nn if dist >= 0 else -nrm / nn
                rows.append(-nrm @ rb.jac_point(fromto[:3], bh)[:, da]); rhs.append(dist - margin)
        k = len(rows); nx = n + k
        Pf = np.zeros((nx, nx)); Pf[:n, :n] = P
        if k:
            Pf[n:, n:] = w_slack * np.eye(k)
        qv = np.zeros(nx); qv[:n] = -g
        dlo = np.maximum(lo - q[qa], -max_dq); dhi = np.minimum(hi - q[qa], max_dq)
        A = [np.hstack([np.eye(n), np.zeros((n, k))]), np.hstack([-np.eye(n), np.zeros((n, k))])]; b = [dhi, -dlo]
        if k:
            A.append(np.hstack([-np.array(rows), -np.eye(k)])); b.append(np.array(rhs))
            A.append(np.hstack([np.zeros((k, n)), -np.eye(k)])); b.append(np.zeros(k))
        A = sparse.csc_matrix(np.vstack(A)); b = np.concatenate(b)
        st = clarabel.DefaultSettings(); st.verbose = False; st.tol_gap_abs = 1e-6; st.tol_gap_rel = 1e-6; st.tol_feas = 1e-7
        sol = clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(Pf), format="csc"), qv, A, b, [clarabel.NonnegativeConeT(len(b))], st).solve()
        if str(sol.status) not in ("Solved", "AlmostSolved"):
            break
        dq = np.asarray(sol.x)[:n]
        q[qa] = np.clip(q[qa] + dq, lo, hi)
        if np.linalg.norm(dq) < 1e-5:
            break
    rb.set(q)
    p, R = rb.body_pose(S["wrist"])
    return q, float(np.linalg.norm(p_target - p)), float(np.degrees(np.linalg.norm(so3_log(R_target @ R.T))))


# ----------------------------------------------------------------------------- stage 3b: finger refinement QP
def refine_fingers(rb: Robot, q, side, q_prior_f, q_prev_f, tip_targets, tip_mask, w_prior=1.0, w_prev=0.2, w_tip=200.0,
                   w_slack=1e4, margin=0.002, pair_thresh=0.02, max_dq=0.3, iters=2):
    """tip_mask may be boolean or a per-tip weight in [0, 1] (soft ramp near the contact threshold)."""
    import clarabel
    from scipy import sparse
    S = rb.side[side]
    names = S["fingers"]
    qf = np.array([rb.qadr[n] for n in names]); df = np.array([rb.dadr[n] for n in names])
    lo = np.array([rb.jrange[n][0] for n in names]); hi = np.array([rb.jrange[n][1] for n in names])
    n = len(names)
    q = q.copy()
    fromto = np.zeros(6)
    stats = {"pairs": 0, "slack": 0.0}
    for _ in range(iters):
        rb.set(q)
        qc = q[qf]
        # objective: prior / previous-frame / contact tips
        P = (w_prior + w_prev) * np.eye(n)
        g = w_prior * (q_prior_f - qc) + w_prev * (q_prev_f - qc)
        for k, b in enumerate(S["tips"]):
            wk = float(tip_mask[k])
            if wk <= 0:
                continue
            p, _ = rb.body_pose(b)
            Jt = rb.jac_point(p, b)[:, df]
            P += w_tip * wk * Jt.T @ Jt
            g += w_tip * wk * Jt.T @ (tip_targets[k] - p)
        # non-penetration pairs (hand geoms vs object pieces)
        rows, rhs = [], []
        for gh in S["hand_geoms"]:
            bh = int(rb.m.geom_bodyid[gh])
            for go in rb.near_obj_geoms(gh, pair_thresh):
                dist = rb.mj.mj_geomDistance(rb.m, rb.d, gh, go, pair_thresh, fromto)
                if dist >= pair_thresh:
                    continue
                p1, p2 = fromto[:3], fromto[3:]
                nrm = p2 - p1
                nn = np.linalg.norm(nrm)
                if nn < 1e-9:
                    continue
                nrm = nrm / nn if dist >= 0 else -nrm / nn  # direction from hand surface towards object surface; sign flips when penetrating
                Jh = rb.jac_point(p1, bh)[:, df]
                # moving the hand point along -nrm increases the distance: d(dist) ~= -nrm . Jh dq
                rows.append(-nrm @ Jh)
                rhs.append(dist - margin)
        k = len(rows)
        stats["pairs"] = max(stats["pairs"], k)
        nx = n + k
        Pf = np.zeros((nx, nx)); Pf[:n, :n] = P
        if k:
            Pf[n:, n:] = w_slack * np.eye(k)
        qv = np.zeros(nx); qv[:n] = -g
        dlo = np.maximum(lo - qc, -max_dq); dhi = np.minimum(hi - qc, max_dq)
        A = [np.hstack([np.eye(n), np.zeros((n, k))]), np.hstack([-np.eye(n), np.zeros((n, k))])]
        b = [dhi, -dlo]
        if k:
            # constraint: rows_i . dq >= -(dist_i - margin) - slack_i   <=>  -rows_i . dq - slack_i <= dist_i - margin
            A.append(np.hstack([-np.array(rows), -np.eye(k)])); b.append(np.array(rhs))
            A.append(np.hstack([np.zeros((k, n)), -np.eye(k)])); b.append(np.zeros(k))
        A = sparse.csc_matrix(np.vstack(A)); b = np.concatenate(b)
        st = clarabel.DefaultSettings(); st.verbose = False; st.tol_gap_abs = 1e-6; st.tol_gap_rel = 1e-6; st.tol_feas = 1e-7
        sol = clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(Pf), format="csc"), qv, A, b, [clarabel.NonnegativeConeT(len(b))], st).solve()
        if str(sol.status) not in ("Solved", "AlmostSolved"):
            break
        x = np.asarray(sol.x)
        q[qf] = np.clip(qc + x[:n], lo, hi)
        if k:
            stats["slack"] = max(stats["slack"], float(np.max(x[n:])))
    return q, stats


# ----------------------------------------------------------------------------- v2: coupled arm + finger QP
TIP_PRIORITY = np.array([1.45, 1.25, 0.78, 0.78, 0.78])  # thumb, index, middle, ring, pinky (DexWeave)


def hand_kp_bodies(c: str) -> list[str]:
    """Robot bodies matching the 21 MediaPipe hand keypoints (wrist, then MCP/PIP/DIP/tip per finger) on the Wuji hand:
    the same keypoint correspondence hand_bench uses for AnyTeleop (joint origins = joint centres)."""
    out = [f"{c}_mount", f"{c}_thumb_proximal", f"{c}_thumb_middle", f"{c}_thumb_distal", f"{c}_thumb_tip"]
    for f in ("index_finger", "middle_finger", "ring_finger", "pinky"):
        out += [f"{c}_{f}_proximal", f"{c}_{f}_middle", f"{c}_{f}_distal", f"{c}_{f}_tip"]
    return out


# Dense contact points (UMR-style surface/contact term, but with the keypoint correspondence): every point is a fixed
# linear combination of keypoints, so the same formula gives corresponding points on the human and on the robot.
# 15 phalanx joint centres (MCP/PIP/DIP of 4 fingers, CMC/MCP/IP of the thumb) + 4 synthetic palm points.
DENSE_POINTS: list[tuple[list[int], list[float]]] = [([i], [1.0]) for i in (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)] + [
    ([5, 9, 13, 17], [0.25, 0.25, 0.25, 0.25]),            # MCP row centre
    ([0, 5, 9, 13, 17], [0.5, 0.125, 0.125, 0.125, 0.125]),  # palm centre
    ([0, 5], [0.5, 0.5]),                                     # index side of the palm
    ([0, 17], [0.5, 0.5]),                                    # pinky side of the palm
]


def dense_points_from_kp(kp: np.ndarray) -> np.ndarray:
    """(..., 21, 3) keypoints -> (..., len(DENSE_POINTS), 3)."""
    return np.stack([sum(c * kp[..., i, :] for i, c in zip(idx, coef)) for idx, coef in DENSE_POINTS], axis=-2)


def refine_coupled(rb: Robot, q, side, q_prior_f, q_arm_ref, q_prev, q_prev2, p_target, R_target, tip_targets, tip_w,
                   w_pose=50.0, w_rot=5.0, w_prior=1.0, w_arm=0.5, w_prev=2.0, w_acc=20.0, w_tip=200.0, w_slack=1e4,
                   margin=0.002, pair_thresh=0.02, max_dq=0.3, iters=2,
                   alpha=None, q_arm_ref_prev=None, w_prev_arm=None, priority=False, q_arm_ref_prev2=None, w_acc_abs_arm=0.0,
                   dense_targets=None, dense_w=None, w_dense=0.0):
    """v3 extras: alpha in [0,1] (interaction weight) gates the wrist POSITION anchor (w_pose * max(alpha, 0.15)) and
    strengthens the UMR arm prior when the hand is free (w_arm * (1 + 4 (1 - alpha))); priority=True weights the fingertip
    terms by TIP_PRIORITY; w_prev_arm penalises the change of the arm CORRECTION relative to the UMR init
    ((q - q_ref) - (q_prev - q_ref_prev)) instead of the absolute joint change, so UMR's own arm motion is not damped."""
    """One QP over the 7 arm + 20 finger joints of one side (v2 of stages 3/3+):
    flange pose target (scaled human wrist, canonical hand frame) + contact fingertip targets + priors (AnyTeleop fingers,
    UMR arm) + causal temporal terms (previous frame, acceleration) + hand/forearm vs object non-penetration with slack.
    q_prev / q_prev2: previous solutions of the 27 dofs (None at the start)."""
    import clarabel
    from scipy import sparse
    S = rb.side[side]
    names = S["arm"] + S["fingers"]
    qi = np.array([rb.qadr[n] for n in names]); di = np.array([rb.dadr[n] for n in names])
    lo = np.array([rb.jrange[n][0] for n in names]); hi = np.array([rb.jrange[n][1] for n in names])
    n = len(names); na = 7
    q = q.copy()
    fromto = np.zeros(6)
    p_target = np.asarray(p_target) - R_target @ S["flange_offset"]
    stats = {"pairs": 0, "slack": 0.0}
    a = 1.0 if alpha is None else float(alpha)
    w_pose_t = w_pose * (max(a, 0.15) if alpha is not None else 1.0)
    w_arm_t = w_arm * ((1.0 + 4.0 * (1.0 - a)) if alpha is not None else 1.0)
    prio = TIP_PRIORITY if priority else np.ones(5)
    for _ in range(iters):
        rb.set(q)
        qc = q[qi]
        P = np.zeros((n, n)); g = np.zeros(n)
        # flange pose (arm dofs only)
        p, R = rb.body_pose(S["wrist"]); jp, jr = rb.jac_body(S["wrist"])
        Jp = np.zeros((6, n)); Jp[:3, :na] = w_pose_t * jp[:, di[:na]]; Jp[3:, :na] = w_rot * jr[:, di[:na]]
        e = np.concatenate([w_pose_t * (p_target - p), w_rot * so3_log(R_target @ R.T)])
        P += Jp.T @ Jp; g += Jp.T @ e
        # contact fingertips (arm + finger dofs)
        for k, b in enumerate(S["tips"]):
            wk = float(tip_w[k]) * float(prio[k])
            if wk <= 0:
                continue
            pt, _ = rb.body_pose(b)
            Jt = rb.jac_point(pt, b)[:, di]
            P += w_tip * wk * Jt.T @ Jt; g += w_tip * wk * Jt.T @ (tip_targets[k] - pt)
        # dense contact points (UMR-style contact map on phalanx joints + palm points; arm + finger dofs)
        if w_dense > 0 and dense_targets is not None:
            pts = rb.dense_points(side)
            for k in range(len(DENSE_POINTS)):
                wk = float(dense_w[k])
                if wk <= 0:
                    continue
                Jd = rb.dense_jac(side, k)[:, di]
                P += w_dense * wk * Jd.T @ Jd; g += w_dense * wk * Jd.T @ (dense_targets[k] - pts[k])
        # priors
        W = np.concatenate([np.full(na, w_arm_t), np.full(n - na, w_prior)])
        ref = np.concatenate([q_arm_ref, q_prior_f])
        P += np.diag(W); g += W * (ref - qc)
        # temporal (causal): previous frame + acceleration
        if q_prev is not None:
            if w_prev_arm is not None and q_arm_ref_prev is not None:
                # arm: smooth the correction (q - q_ref) over time; fingers: absolute previous-frame term
                Wt = np.concatenate([np.full(na, w_prev_arm), np.full(n - na, w_prev)])
                tgt = np.concatenate([q_arm_ref + (q_prev[:na] - q_arm_ref_prev), q_prev[na:]])
                P += np.diag(Wt); g += Wt * (tgt - qc)
                if q_prev2 is not None:
                    Wa = np.concatenate([np.full(na, w_acc), np.full(n - na, w_acc)])
                    acc_tgt = 2 * q_prev - q_prev2
                    if q_arm_ref_prev2 is not None:  # arm: constant-velocity extrapolation of the CORRECTION
                        d1 = q_prev[:na] - q_arm_ref_prev; d2 = q_prev2[:na] - q_arm_ref_prev2
                        acc_tgt = np.concatenate([q_arm_ref + 2 * d1 - d2, acc_tgt[na:]])
                    P += np.diag(Wa); g += Wa * (acc_tgt - qc)
                    if w_acc_abs_arm > 0:  # plus an ABSOLUTE acceleration term on the arm: damps UMR's own arm jitter
                        Wb = np.concatenate([np.full(na, w_acc_abs_arm), np.zeros(n - na)])
                        P += np.diag(Wb); g += Wb * (2 * q_prev - q_prev2 - qc)
            else:
                P += w_prev * np.eye(n); g += w_prev * (q_prev - qc)
                if q_prev2 is not None:
                    P += w_acc * np.eye(n); g += w_acc * (2 * q_prev - q_prev2 - qc)
        # non-penetration: hand + forearm geoms vs object
        rows, rhs = [], []
        for gh in S["hand_geoms"] + S["forearm_geoms"]:
            bh = int(rb.m.geom_bodyid[gh])
            for go in rb.near_obj_geoms(gh, pair_thresh):
                dist = rb.mj.mj_geomDistance(rb.m, rb.d, gh, go, pair_thresh, fromto)
                if dist >= pair_thresh:
                    continue
                nrm = fromto[3:] - fromto[:3]; nn = np.linalg.norm(nrm)
                if nn < 1e-9:
                    continue
                nrm = nrm / nn if dist >= 0 else -nrm / nn
                rows.append(-nrm @ rb.jac_point(fromto[:3], bh)[:, di]); rhs.append(dist - margin)
        k = len(rows); stats["pairs"] = max(stats["pairs"], k)
        nx = n + k
        Pf = np.zeros((nx, nx)); Pf[:n, :n] = P + 1e-6 * np.eye(n)
        if k:
            Pf[n:, n:] = w_slack * np.eye(k)
        qv = np.zeros(nx); qv[:n] = -g
        dlo = np.maximum(lo - qc, -max_dq); dhi = np.minimum(hi - qc, max_dq)
        A = [np.hstack([np.eye(n), np.zeros((n, k))]), np.hstack([-np.eye(n), np.zeros((n, k))])]; b = [dhi, -dlo]
        if k:
            A.append(np.hstack([-np.array(rows), -np.eye(k)])); b.append(np.array(rhs))
            A.append(np.hstack([np.zeros((k, n)), -np.eye(k)])); b.append(np.zeros(k))
        A = sparse.csc_matrix(np.vstack(A)); b = np.concatenate(b)
        st = clarabel.DefaultSettings(); st.verbose = False; st.tol_gap_abs = 1e-6; st.tol_gap_rel = 1e-6; st.tol_feas = 1e-7
        sol = clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(Pf), format="csc"), qv, A, b, [clarabel.NonnegativeConeT(len(b))], st).solve()
        if str(sol.status) not in ("Solved", "AlmostSolved"):
            break
        x = np.asarray(sol.x)
        q[qi] = np.clip(qc + x[:n], lo, hi)
        if k:
            stats["slack"] = max(stats["slack"], float(np.max(x[n:])))
    return q, q[qi].copy(), stats


def contact_hysteresis(dist: np.ndarray, tau: float, min_frames: int = 3) -> np.ndarray:
    """Per-tip contact weight with hysteresis: enters below tau, leaves above 2*tau, episodes shorter than min_frames dropped.
    Inside an episode the weight ramps linearly from 1 (d <= tau) to 0 (d >= 2*tau)."""
    T, K = dist.shape
    w = np.zeros((T, K))
    for k in range(K):
        active = False; start = 0
        for t in range(T):
            d = dist[t, k]
            if not active and d < tau:
                active, start = True, t
            elif active and d > 2 * tau:
                if t - start >= min_frames:
                    w[start:t, k] = np.clip((2 * tau - dist[start:t, k]) / tau, 0.0, 1.0)
                active = False
        if active and T - start >= min_frames:
            w[start:T, k] = np.clip((2 * tau - dist[start:T, k]) / tau, 0.0, 1.0)
    return w


def contact_targets_v2(src, hb, obj_index, side, s, tau, hand_ratio, ms=None, op_r=None):
    """v2 targets: the human fingertip offset from the wrist (canonical hand frame) is scaled by the robot/human hand size
    ratio, placed at the scaled human wrist, then projected onto the object surface; contact weights use hysteresis."""
    import trimesh
    from scipy.spatial.transform import Rotation as Rot
    T = min(len(src["trans"]), int(hb["n_frames"]))
    kp = hb[f"hand_kp_{side}"][:T].astype(np.float64); Rh = hb[f"hand_R_{side}"][:T].astype(np.float64)
    op = hb["obj_pos"][:T, obj_index].astype(np.float64); oR = hb["obj_rot"][:T, obj_index].astype(np.float64)
    mesh = trimesh.load(C.DATA / "objects" / str(src["object"]) / f"{src['object']}.obj", force="mesh")
    pq = trimesh.proximity.ProximityQuery(mesh)
    tips_h = kp[:, TIPS_H]
    loc_h = np.einsum("tji,tkj->tki", oR, tips_h - op[:, None])
    dist_h = C.on_surface_chunked(pq, loc_h.reshape(-1, 3))[1].reshape(T, 5)
    w = contact_hysteresis(dist_h, tau)
    # size-remapped tips (human units): wrist + ratio * (tip - wrist)
    remap = kp[:, :1] + hand_ratio * (tips_h - kp[:, :1])
    loc_r = np.einsum("tji,tkj->tki", oR, remap - op[:, None])
    closest = C.on_surface_chunked(pq, loc_r.reshape(-1, 3))[0].reshape(T, 5, 3)
    ms = s if ms is None else ms
    op_r = src["obj_pos"][:T].astype(np.float64) * s if op_r is None else np.asarray(op_r, dtype=np.float64)[:T]
    oR_r = Rot.from_quat(src["obj_quat_xyzw"][:T]).as_matrix()
    targets = np.einsum("tij,tkj->tki", oR_r, closest * ms) + op_r[:, None]
    return targets, w, dist_h


def interaction_alpha(dist_tips: np.ndarray, near: float = 0.05, persist: int = 4) -> np.ndarray:
    """DexWeave-style interaction weight per frame: 1 when the hand has been within `near` of the object for >= persist frames,
    0 when it has been away for >= persist frames, linear ramp in between (causal moving average of the gated indicator)."""
    ind = (dist_tips.min(1) < near).astype(np.float64)
    T = len(ind); out = np.zeros(T)
    for t in range(T):
        out[t] = ind[max(0, t - persist + 1):t + 1].mean()
    return out


def contact_targets_v3(src, hb, obj_index, side, s, tau, hand_ratio, remap="ratio", persist=4, ms=None, op_r=None):
    """v3 targets: like v2 but with the correct size factor. The robot hand is `hand_ratio` x the human hand in REAL metres
    while the robot world is the human world x s, so in human units the remap factor is hand_ratio / s (v2 used hand_ratio,
    i.e. targets for a hand s x too short). Also returns the interaction weight alpha(t).
    `ms` = object mesh scale in the robot world (s in the scaled world, 1.0 when the object keeps its real size): the human
    contact points (object frame, human-size mesh) are mapped onto the robot-world object by x ms."""
    import trimesh
    from scipy.spatial.transform import Rotation as Rot
    T = min(len(src["trans"]), int(hb["n_frames"]))
    kp = hb[f"hand_kp_{side}"][:T].astype(np.float64)
    op = hb["obj_pos"][:T, obj_index].astype(np.float64); oR = hb["obj_rot"][:T, obj_index].astype(np.float64)
    mesh = trimesh.load(C.DATA / "objects" / str(src["object"]) / f"{src['object']}.obj", force="mesh")
    pq = trimesh.proximity.ProximityQuery(mesh)
    tips_h = kp[:, TIPS_H]
    loc_h = np.einsum("tji,tkj->tki", oR, tips_h - op[:, None])
    dist_h = C.on_surface_chunked(pq, loc_h.reshape(-1, 3))[1].reshape(T, 5)
    w = contact_hysteresis(dist_h, tau)
    ms = s if ms is None else ms
    factor = (hand_ratio / ms) if remap == "ratio" else 1.0  # "none": human contact points, wrist anchor absorbs the size mismatch
    remapped = kp[:, :1] + factor * (tips_h - kp[:, :1])
    loc_r = np.einsum("tji,tkj->tki", oR, remapped - op[:, None])
    closest = C.on_surface_chunked(pq, loc_r.reshape(-1, 3))[0].reshape(T, 5, 3)
    op_r = src["obj_pos"][:T].astype(np.float64) * s if op_r is None else np.asarray(op_r, dtype=np.float64)[:T]
    oR_r = Rot.from_quat(src["obj_quat_xyzw"][:T]).as_matrix()
    targets = np.einsum("tij,tkj->tki", oR_r, closest * ms) + op_r[:, None]
    return targets, w, dist_h, interaction_alpha(dist_h, persist=persist)


def nearest_surface_multi(points_h, objects):
    """Nearest object surface over several objects. points_h: (T, K, 3) human-world points; objects: list of dicts with
    pq (ProximityQuery of the human-size mesh), op_h (T,3), oR_h (T,3,3), op_r (T,3), oR_r (T,3,3), ms (robot mesh scale).
    Returns closest point in the robot world (T,K,3), distance in human units (T,K), human-world offset point - closest (T,K,3),
    and the index of the chosen object (T,K)."""
    T, K, _ = points_h.shape
    best_d = np.full((T, K), np.inf); best_r = np.zeros((T, K, 3)); best_v = np.zeros((T, K, 3)); best_i = np.zeros((T, K), int)
    for oi, o in enumerate(objects):
        loc = np.einsum("tji,tkj->tki", o["oR_h"], points_h - o["op_h"][:, None])
        cl, d = C.on_surface_chunked(o["pq"], loc.reshape(-1, 3))
        cl = cl.reshape(T, K, 3); d = d.reshape(T, K)
        cl_h = np.einsum("tij,tkj->tki", o["oR_h"], cl) + o["op_h"][:, None]
        cl_r = np.einsum("tij,tkj->tki", o["oR_r"], cl * o["ms"]) + o["op_r"][:, None]
        m = d < best_d
        best_d[m] = d[m]; best_r[m] = cl_r[m]; best_v[m] = (points_h - cl_h)[m]; best_i[m] = oi
    return best_r, best_d, best_v, best_i


def contact_targets_multi(objects, hb, side, T, s, tau, hand_ratio, tau_dense=None, persist=12):
    """v3 tip targets (human contact points on the NEAREST object, no size remap) + interaction weight, and optionally the
    dense phalanx/palm targets (offset vector reproduced x hand_ratio), all over several objects."""
    kp = hb[f"hand_kp_{side}"][:T].astype(np.float64)
    tr, td, _, _ = nearest_surface_multi(kp[:, TIPS_H], objects)
    out = {"targets3": tr, "w3": contact_hysteresis(td, tau), "alpha3": interaction_alpha(td, persist=persist)}
    if tau_dense is not None:
        dr, dd, dv, _ = nearest_surface_multi(dense_points_from_kp(kp), objects)
        out["targetsD"] = dr + dv * (s * hand_ratio); out["wD"] = contact_hysteresis(dd, tau_dense)
    return out


# ----------------------------------------------------------------------------- per sequence
def contact_targets_dense(src, hb, obj_index, side, s, tau, hand_ratio, ms=None, op_r=None):
    """UMR-style contact map on the dense hand point set: for every dense human point within tau (hysteresis, exit 2 tau)
    of the object surface, the corresponding robot point must reproduce the human point's offset vector to its closest
    surface point. The offset is scaled by hand_ratio (robot joint centres sit deeper under the skin than the human's).
    Returns targets [T, D, 3] (robot world), weights [T, D], human distances [T, D]."""
    import trimesh
    from scipy.spatial.transform import Rotation as Rot
    T = min(len(src["trans"]), int(hb["n_frames"]))
    kp = hb[f"hand_kp_{side}"][:T].astype(np.float64)
    pts = dense_points_from_kp(kp)  # (T, D, 3) human units, world
    op = hb["obj_pos"][:T, obj_index].astype(np.float64); oR = hb["obj_rot"][:T, obj_index].astype(np.float64)
    loc = np.einsum("tji,tkj->tki", oR, pts - op[:, None])
    mesh = trimesh.load(C.DATA / "objects" / str(src["object"]) / f"{src['object']}.obj", force="mesh")
    pq = trimesh.proximity.ProximityQuery(mesh)
    D = pts.shape[1]
    closest, dist = C.on_surface_chunked(pq, loc.reshape(-1, 3))
    closest = closest.reshape(T, D, 3); dist = dist.reshape(T, D)
    w = contact_hysteresis(dist, tau)
    ms = s if ms is None else ms
    v = loc - closest  # human offset vector (object frame, human units)
    op_r = src["obj_pos"][:T].astype(np.float64) * s if op_r is None else np.asarray(op_r, dtype=np.float64)[:T]
    oR_r = Rot.from_quat(src["obj_quat_xyzw"][:T]).as_matrix()
    targets = np.einsum("tij,tkj->tki", oR_r, closest * ms + v * (s * hand_ratio)) + op_r[:, None]
    return targets, w, dist


def contact_targets(src, hb, obj_index, side, s, tau, ms=None, op_r=None):
    """Human fingertip contacts with the primary object -> robot-world target points. Returns (targets [T,5,3], mask [T,5], dist_h [T,5])."""
    import trimesh
    from scipy.spatial.transform import Rotation as Rot
    T = min(len(src["trans"]), int(hb["n_frames"]))
    kp = hb[f"hand_kp_{side}"][:T, TIPS_H].astype(np.float64)
    op = hb["obj_pos"][:T, obj_index].astype(np.float64); oR = hb["obj_rot"][:T, obj_index].astype(np.float64)
    local = np.einsum("tji,tkj->tki", oR, kp - op[:, None])  # object frame
    mesh = trimesh.load(C.DATA / "objects" / str(src["object"]) / f"{src['object']}.obj", force="mesh")
    pq = trimesh.proximity.ProximityQuery(mesh)
    closest, dist = C.on_surface_chunked(pq, local.reshape(-1, 3))
    closest = closest.reshape(T, 5, 3); dist = dist.reshape(T, 5)
    # soft contact weight: 1 inside tau, linear ramp to 0 at 2*tau (avoids on/off chatter of the tip term between frames)
    mask = np.clip((2 * tau - dist) / tau, 0.0, 1.0)
    # robot world: object pose x s, mesh x ms (= s in the scaled world, 1 for a real-size object)
    ms = s if ms is None else ms
    op_r = src["obj_pos"][:T].astype(np.float64) * s if op_r is None else np.asarray(op_r, dtype=np.float64)[:T]
    oR_r = Rot.from_quat(src["obj_quat_xyzw"][:T]).as_matrix()
    targets = np.einsum("tij,tkj->tki", oR_r, closest * ms) + op_r[:, None]
    return targets, mask, dist


PARAMS = {"w_prev": 2.0, "w_tip": 200.0, "w_reg": 1e-4, "place_iters": 15, "smooth": 5, "place_nonpen": 2,
          "body_lane": "umr_wuji", "out_suffix": "", "obj_mesh_scale": None}  # body_lane umr_wuji_real + out_suffix _real = real-size-object world


def run_sequence(args):
    seq, stages, tau, overwrite, params = args
    PARAMS.update(params)
    C._model_cache.clear()  # models are per (object, scale): the cache would grow without bound across sequences
    import mujoco  # noqa: F401
    outs = {st: C.RESULTS / f"pipe_{st}{PARAMS['out_suffix']}" / f"{seq}.npz" for st in stages}
    if not overwrite and all(p.exists() for p in outs.values()):
        return {"seq": seq, "skipped": True}
    t0 = time.time()
    r = dict(np.load(C.RESULTS / PARAMS["body_lane"] / f"{seq}.npz", allow_pickle=True))
    src = np.load(C.DATA / "source" / f"{seq}.npz", allow_pickle=True)
    hb = np.load(HB_DATA / "seqs" / f"{seq}.npz", allow_pickle=True)
    s = float(np.asarray(r["smpl_scale"]).reshape(-1)[0])
    ms = float(np.asarray(r["object_mesh_scale"]).reshape(-1)[0]) if "object_mesh_scale" in r else s  # robot-world object mesh scale
    if PARAMS.get("obj_mesh_scale") is not None:  # ablation: override the body lane's object size for the hand stages only (1.0 = real size)
        ms = float(PARAMS["obj_mesh_scale"]); r["object_mesh_scale"] = np.asarray(ms)
    names = [str(x) for x in r["robot_joint_names"]]
    q_umr = r["qpos"].astype(np.float64)
    T = min(len(q_umr), int(hb["n_frames"]), len(src["trans"]))
    obj = str(src["object"]); objs = [str(o) for o in hb["objects"]]; oi = objs.index(obj)
    objs_m = C.multi_objects(seq, src) if PARAMS.get("multi") else [obj]  # primary first
    rb = Robot(objs_m, ms)
    # map UMR qpos (root + joints by name) into the model's qpos layout (identical order expected, mapped by name anyway)
    nq_r = 7 + len(names)
    col = {n: 7 + i for i, n in enumerate(names)}
    def to_model(qr):
        q = np.zeros(rb.m.nq); q[:7] = qr[:7]
        for n, i in col.items():
            q[rb.qadr[n]] = qr[i]
        return q
    def from_model(q):
        out = np.zeros(nq_r); out[:7] = q[:7]
        for n, i in col.items():
            out[i] = q[rb.qadr[n]]
        return out
    from scipy.spatial.transform import Rotation as Rot
    op_r = C.umr_object_pos(r, src, s, T); oq_r = src["obj_quat_xyzw"][:T][:, [3, 0, 1, 2]]
    # robot-world poses of every object (primary from the body lane, extras = hand_bench trajectory x s)
    obj_pose = [(op_r, oq_r)]
    for o in objs_m[1:]:
        k = objs.index(o)
        obj_pose.append((hb["obj_pos"][:T, k].astype(np.float64) * s, Rot.from_matrix(hb["obj_rot"][:T, k]).as_quat()[:, [3, 0, 1, 2]]))
    multi = None
    if PARAMS.get("multi"):
        import trimesh
        multi = []
        for o, (pr, qr) in zip(objs_m, obj_pose):
            k = objs.index(o)
            mesh = trimesh.load(C.DATA / "objects" / o / f"{o}.obj", force="mesh")
            multi.append({"pq": trimesh.proximity.ProximityQuery(mesh), "op_h": hb["obj_pos"][:T, k].astype(np.float64),
                          "oR_h": hb["obj_rot"][:T, k].astype(np.float64), "op_r": pr, "oR_r": Rot.from_quat(qr[:, [1, 2, 3, 0]]).as_matrix(), "ms": ms})
    hands = {}
    for side in ("left", "right"):
        p = HB_RES / "anyteleop" / "wuji" / f"{seq}__{side}.npz"
        if not p.exists():
            continue
        a = np.load(p, allow_pickle=True)
        hands[side] = {"q": a["q_active"].astype(np.float64), "names": [str(x) for x in a["joint_names"]],
                       "ratio": float(a["scale"]) if "scale" in a else 1.0}
        hands[side]["targets"], hands[side]["mask"], hands[side]["dist_h"] = contact_targets(src, hb, oi, side, s, tau, ms, op_r)
        if "v2" in stages:
            hands[side]["targets2"], hands[side]["w2"], _ = contact_targets_v2(src, hb, oi, side, s, tau, hands[side]["ratio"], ms, op_r)
        if "v3" in stages and multi is not None:
            hands[side].update(contact_targets_multi(multi, hb, side, T, s, tau, hands[side]["ratio"],
                                                     PARAMS.get("tau_dense", 0.03) if PARAMS.get("w_dense", 0.0) > 0 else None, PARAMS.get("v3_persist", 12)))
        elif "v3" in stages:
            hands[side]["targets3"], hands[side]["w3"], _, hands[side]["alpha3"] = contact_targets_v3(src, hb, oi, side, s, tau, hands[side]["ratio"], PARAMS.get("v3_remap", "none"), PARAMS.get("v3_persist", 12), ms, op_r)
            if PARAMS.get("w_dense", 0.0) > 0:
                hands[side]["targetsD"], hands[side]["wD"], _ = contact_targets_dense(src, hb, oi, side, s, PARAMS.get("tau_dense", 0.03), hands[side]["ratio"], ms, op_r)
        hands[side]["flange_p"] = hb[f"hand_kp_{side}"][:T, 0].astype(np.float64) * s
        hands[side]["flange_R"] = hb[f"hand_R_{side}"][:T].astype(np.float64)
    res = {st: np.zeros((T, nq_r)) for st in stages}
    info = {"place_pos_err": [], "place_rot_err": [], "pairs": 0, "slack": 0.0}
    prev = {}

    def model_q(t):
        q = to_model(q_umr[t])
        for a, (pr, qr) in zip(rb.obj_qadrs, obj_pose):
            q[a:a + 3] = pr[t]; q[a + 3:a + 7] = qr[t]
        q2 = q.copy()  # s2: AnyTeleop fingers substituted
        for side, h in hands.items():
            tt = min(t, len(h["q"]) - 1)
            for n, v in zip(h["names"], h["q"][tt]):
                q2[rb.qadr[n]] = v
        return q, q2

    # pass 1 (s3p only): arm placement per frame, then temporal smoothing of the placed arm trajectory, so that the finger
    # refinement in pass 2 sees the final (smooth) arm and its non-penetration / contact terms stay consistent with it
    arm_traj = {}
    if "s3p" in res:
        arm_cols = {side: [rb.qadr[n] for n in rb.side[side]["arm"]] for side in hands}
        arm_traj = {side: np.zeros((T, 7)) for side in hands}
        for t in range(T):
            q, q2 = model_q(t)
            q3 = q2.copy()
            for side, h in hands.items():
                mode = PARAMS.get("place_nonpen", 2)
                if mode:
                    q3, pe, re_ = place_arm_qp(rb, q3, side, h["flange_p"][t], h["flange_R"][t], q, w_reg=PARAMS["w_reg"], iters=PARAMS["place_iters"],
                                               geom_set="hand_geoms" if mode == 1 else "palm_geoms")
                else:
                    q3, pe, re_ = place_arm(rb, q3, side, h["flange_p"][t], h["flange_R"][t], q, w_reg=PARAMS["w_reg"], iters=PARAMS["place_iters"])
                info["place_pos_err"].append(pe); info["place_rot_err"].append(re_)
                arm_traj[side][t] = q3[arm_cols[side]]
        if PARAMS["smooth"] and PARAMS["smooth"] > 1 and T > PARAMS["smooth"]:
            from scipy.ndimage import uniform_filter1d
            for side in arm_traj:
                arm_traj[side] = uniform_filter1d(arm_traj[side], size=int(PARAMS["smooth"]), axis=0, mode="nearest")

    v2_prev = {side: (None, None) for side in hands}
    v3_prev = {side: (None, None, None, None) for side in hands}  # (sol, sol_prev, umr arm at t-1, umr arm at t-2)
    for t in range(T):
        q, q2 = model_q(t)
        if "s2" in res:
            res["s2"][t] = from_model(q2)
        if "v3" in res:
            use_at = PARAMS.get("finger_prior", "anyteleop") == "anyteleop"  # ablation: finger prior / init = AnyTeleop pose or the UMR rest pose
            q3 = (q2 if use_at else q).copy()
            for side, h in hands.items():
                S = rb.side[side]
                qf = [rb.qadr[n] for n in S["fingers"]]; qa = [rb.qadr[n] for n in S["arm"]]
                prev1, prev2, arm_ref_prev, arm_ref_prev2 = v3_prev[side]
                if prev1 is not None:
                    q3[qa + qf] = prev1
                q3, sol, stt = refine_coupled(rb, q3, side, (q2 if use_at else q)[qf].copy(), q[qa].copy(), prev1, prev2, h["flange_p"][t], h["flange_R"][t],
                                              h["targets3"][t], h["w3"][t], w_prev=PARAMS["w_prev"], w_acc=PARAMS.get("w_acc", 20.0),
                                              w_tip=PARAMS["w_tip"], w_pose=PARAMS.get("w_pose", 50.0), w_rot=PARAMS.get("w_rot", 5.0),
                                              iters=PARAMS.get("v3_iters", 4), alpha=(float(h["alpha3"][t]) if PARAMS.get("v3_alpha", 1) else None), q_arm_ref_prev=arm_ref_prev,
                                              w_prev_arm=PARAMS.get("w_prev_arm", 40.0), priority=True, q_arm_ref_prev2=arm_ref_prev2,
                                              w_acc_abs_arm=PARAMS.get("w_acc_abs_arm", 60.0),
                                              dense_targets=(h["targetsD"][t] if "targetsD" in h else None), dense_w=(h["wD"][t] if "wD" in h else None), w_dense=PARAMS.get("w_dense", 0.0))
                v3_prev[side] = (sol, prev1, q[qa].copy(), arm_ref_prev)
                info["pairs"] = max(info["pairs"], stt["pairs"]); info["slack"] = max(info["slack"], stt["slack"])
            res["v3"][t] = from_model(q3)
        if "v2" in res:  # coupled arm + finger QP, causal temporal terms, hysteresis / size-remapped contact targets
            q3 = q2.copy()
            for side, h in hands.items():
                S = rb.side[side]
                qf = [rb.qadr[n] for n in S["fingers"]]; qa = [rb.qadr[n] for n in S["arm"]]
                prev1, prev2 = v2_prev[side]
                if prev1 is not None:  # warm start from the previous solution (arm + fingers)
                    q3[qa + qf] = prev1
                q3, sol, stt = refine_coupled(rb, q3, side, q2[qf].copy(), q[qa].copy(), prev1, prev2, h["flange_p"][t], h["flange_R"][t],
                                              h["targets2"][t], h["w2"][t], w_prev=PARAMS["w_prev"], w_acc=PARAMS.get("w_acc", 20.0),
                                              w_tip=PARAMS["w_tip"], w_pose=PARAMS.get("w_pose", 50.0), w_rot=PARAMS.get("w_rot", 5.0))
                v2_prev[side] = (sol, prev1)
                info["pairs"] = max(info["pairs"], stt["pairs"]); info["slack"] = max(info["slack"], stt["slack"])
            res["v2"][t] = from_model(q3)
        for st in ("s3", "s3p"):
            if st not in res:
                continue
            q3 = q2.copy()
            if st == "s3p":
                for side, h in hands.items():
                    q3[arm_cols[side]] = arm_traj[side][t]
                    if PARAMS.get("place_nonpen", 2):  # projection: smallest arm change that removes palm/object penetration re-introduced by the smoothing
                        q3, _, _ = place_arm_qp(rb, q3, side, h["flange_p"][t], h["flange_R"][t], q3, w_pos=0.0, w_rot=0.0, w_reg=1.0, iters=3, max_dq=0.05,
                                                geom_set="hand_geoms" if PARAMS.get("place_nonpen", 2) == 1 else "palm_geoms")
            for side, h in hands.items():
                S = rb.side[side]
                qf = [rb.qadr[n] for n in S["fingers"]]
                prior = q2[qf].copy()
                prev_f = prev.get((st, side), prior)
                q3, stt = refine_fingers(rb, q3, side, prior, prev_f, h["targets"][t], h["mask"][t], w_prev=PARAMS["w_prev"], w_tip=PARAMS["w_tip"])
                prev[(st, side)] = q3[qf].copy()
                info["pairs"] = max(info["pairs"], stt["pairs"]); info["slack"] = max(info["slack"], stt["slack"])
            res[st][t] = from_model(q3)
    if PARAMS["smooth"] and PARAMS["smooth"] > 1:
        from scipy.ndimage import uniform_filter1d
        fcols = [7 + i for i, n in enumerate(names) if n[:2] in ("l_", "r_")]
        for st in ("s3", "s3p"):
            if st in res and T > PARAMS["smooth"]:
                res[st][:, fcols] = uniform_filter1d(res[st][:, fcols], size=int(PARAMS["smooth"]), axis=0, mode="nearest")

    for st, arr in res.items():
        outs[st].parent.mkdir(parents=True, exist_ok=True)
        extra = {k: r[k] for k in r if k not in ("qpos",)}
        np.savez_compressed(outs[st], qpos=arr.astype(np.float32), stage=st, hands=np.array(sorted(hands)), tau=tau, params=json.dumps(PARAMS),
                            objects=np.array(objs_m), **extra)
    return {"seq": seq, "T": T, "hands": sorted(hands), "seconds": round(time.time() - t0, 1),
            "place_pos_err_cm": round(100 * float(np.mean(info["place_pos_err"])), 2) if info["place_pos_err"] else None,
            "place_rot_err_deg": round(float(np.mean(info["place_rot_err"])), 1) if info["place_rot_err"] else None,
            "max_pairs": info["pairs"], "max_slack_mm": round(1000 * info["slack"], 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["s2", "s3", "s3p"], choices=["s2", "s3", "s3p", "v2", "v3"])
    ap.add_argument("--v3-iters", type=int, default=4)
    ap.add_argument("--w-prev-arm", type=float, default=40.0, help="v3: weight on the change of the arm correction (q - q_umr) between frames")
    ap.add_argument("--w-acc-abs-arm", type=float, default=60.0, help="v3: absolute acceleration term on the arm joints (damps UMR arm jitter)")
    ap.add_argument("--v3-persist", type=int, default=12, help="v3: frames of the interaction-weight ramp")
    ap.add_argument("--v3-alpha", type=int, default=1, help="v3: 1 = gate the wrist anchor / arm prior by the interaction weight, 0 = off")
    ap.add_argument("--v3-remap", choices=["none", "ratio"], default="none", help="v3 contact targets: human contact points (none) or size-remapped by hand_ratio/s (ratio)")
    ap.add_argument("--w-acc", type=float, default=20.0, help="v2: acceleration regularisation weight")
    ap.add_argument("--w-pose", type=float, default=50.0, help="v2: flange position weight (per m^2)")
    ap.add_argument("--w-rot", type=float, default=5.0, help="v2: flange orientation weight (per rad^2)")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--index", type=Path, default=C.DATA / "seqs_test.json")
    ap.add_argument("--tau", type=float, default=0.02, help="human fingertip-to-object distance that counts as contact (m)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--w-prev", type=float, default=2.0, help="finger refinement: weight pulling towards the previous frame")
    ap.add_argument("--w-tip", type=float, default=200.0, help="finger refinement: contact fingertip target weight")
    ap.add_argument("--w-reg", type=float, default=1e-4, help="arm placement: regularisation towards the UMR arm pose")
    ap.add_argument("--place-iters", type=int, default=15)
    ap.add_argument("--smooth", type=int, default=5, help="moving-average window (frames) applied to the refined finger joints, 0 = off")
    ap.add_argument("--body-lane", default="umr_wuji", help="stage-1 body lane under results[_ds]/ (umr_wuji = scaled objects, umr_wuji_real = real-size objects)")
    ap.add_argument("--out-suffix", default="", help="suffix of the output lanes pipe_<stage><suffix> (e.g. _real)")
    ap.add_argument("--finger-prior", choices=["anyteleop", "rest"], default="anyteleop", help="v3 finger prior and initial guess: AnyTeleop solution (default) or the UMR rest pose (ablation: is stage 2 needed?)")
    ap.add_argument("--multi", action="store_true", help="v3: all interacted hand-held objects (data*/multi_objects.json) get contact targets and non-penetration, not only the primary")
    ap.add_argument("--w-dense", type=float, default=0.0, help="v3: weight of the UMR-style dense contact term on phalanx joints + palm points (0 = off)")
    ap.add_argument("--tau-dense", type=float, default=0.03, help="v3 dense term: human joint-centre-to-surface distance that counts as contact (m); exit at 2x")
    ap.add_argument("--obj-mesh-scale", type=float, default=None, help="override the object mesh scale seen by the hand stages (1.0 = real size while the body lane stays scaled); default: the body lane's object_mesh_scale")
    ap.add_argument("--place-nonpen", type=int, default=2, help="arm placement: 0 = plain pose IK, 1 = QP with non-penetration on all hand geoms, 2 = QP with non-penetration on the palm only (fingers are handled by the finger refinement)")
    args = ap.parse_args()
    params = {"w_prev": args.w_prev, "w_tip": args.w_tip, "w_reg": args.w_reg, "place_iters": args.place_iters, "smooth": args.smooth,
              "place_nonpen": args.place_nonpen, "w_acc": args.w_acc, "w_pose": args.w_pose, "w_rot": args.w_rot,
              "v3_iters": args.v3_iters, "w_prev_arm": args.w_prev_arm, "v3_remap": args.v3_remap, "v3_persist": args.v3_persist, "v3_alpha": args.v3_alpha, "w_acc_abs_arm": args.w_acc_abs_arm,
              "body_lane": args.body_lane, "out_suffix": args.out_suffix, "obj_mesh_scale": args.obj_mesh_scale, "finger_prior": args.finger_prior,
              "w_dense": args.w_dense, "tau_dense": args.tau_dense, "multi": args.multi}
    index = json.loads(args.index.read_text())
    seqs = [e["seq"] for e in index if (not args.seqs or e["seq"] in set(args.seqs)) and (C.RESULTS / args.body_lane / f"{e['seq']}.npz").exists()]
    print(f"[{C.DATASET}] {len(seqs)} sequences, stages {args.stage}, body lane {args.body_lane} -> pipe_<stage>{args.out_suffix}")
    jobs = [(s, args.stage, args.tau, args.overwrite, params) for s in seqs]
    summary = []
    if args.workers <= 1:
        for j in jobs:
            r = run_sequence(j); summary.append(r); print(r, flush=True)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(run_sequence, jobs):
                summary.append(r); print(r, flush=True)
    (C.RESULTS / f"pipe_summary{args.out_suffix}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
