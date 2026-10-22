"""Torque-matched warm start: carry a SPIDER result from one actuator model to another (e.g. stiff kp 500 / armature 1 ->
real G1/Wuji gains and armature).

The source rollout (q, qd, qdd, ctrl) is physically consistent under the source model. The joint torques that reproduce
the same motion under the target model differ only by the armature term, so

    tau_tgt = clip(kp_src (u_src - q) - kd_src qd) - (I_arm_src - I_arm_tgt) qdd
    u_tgt   = q + (tau_tgt + kd_tgt qd) / kp_tgt

becomes the nominal ctrl of the target dataset (written into its trajectory_kinematic.npz; the reference qpos is kept).
This carries over the grasp squeeze that the sampler cannot discover with low gains (~1.75 rad arm target offsets).

    ~/software/venvs/spider/bin/python omomo_bench/spider_bridge/warmstart.py --tags <tag> --src hiphi --dst hiphi_real --out hiphi_real_ws
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np

PROC = Path(__file__).resolve().parent / "data" / "processed"


def task_dir(ds: str, tag: str) -> Path:
    return PROC / ds / "g1_wuji" / "humanoid_object" / tag


def torque_matched_ctrl(m_src: mujoco.MjModel, m_tgt: mujoco.MjModel, qpos, qvel, ctrl, dt: float) -> np.ndarray:
    """qpos/qvel/ctrl: per sim step (qpos[t] is the state after applying ctrl[t])."""
    qad = m_src.jnt_qposadr[m_src.actuator_trnid[:, 0]]
    dad = m_src.jnt_dofadr[m_src.actuator_trnid[:, 0]]
    kp_s, kd_s, lim_s = m_src.actuator_gainprm[:, 0], -m_src.actuator_biasprm[:, 2], m_src.actuator_forcerange[:, 1]
    kp_t, kd_t = m_tgt.actuator_gainprm[:, 0], -m_tgt.actuator_biasprm[:, 2]
    q0 = np.concatenate([qpos[:1], qpos[:-1]])  # state at which ctrl[t] was applied
    v0 = np.concatenate([qvel[:1], qvel[:-1]])
    tau = np.clip(kp_s * (ctrl - q0[:, qad]) - kd_s * v0[:, dad], -lim_s, lim_s)
    qacc = (qvel - v0) / dt
    tau = tau - (m_src.dof_armature[dad] - m_tgt.dof_armature[dad]) * qacc[:, dad]
    return q0[:, qad] + (tau + kd_t * v0[:, dad]) / kp_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--src", default="hiphi", help="dataset with a finished SPIDER run (trajectory_mjwp.npz)")
    ap.add_argument("--dst", default="hiphi_real", help="dataset with the target actuator scene (build_hiphi.py --actuator real)")
    ap.add_argument("--out", default="hiphi_real_ws", help="new dataset: dst scene + warm-started nominal ctrl")
    a = ap.parse_args()
    for tag in a.tags:
        S, D, O = task_dir(a.src, tag), task_dir(a.dst, tag), task_dir(a.out, tag)
        m_s = mujoco.MjModel.from_xml_path(str(S / "scene.xml"))
        m_t = mujoco.MjModel.from_xml_path(str(D / "scene.xml"))
        r = np.load(S / "0" / "trajectory_mjwp.npz")
        nq, nv, nu = m_s.nq, m_s.nv, m_s.nu
        c = torque_matched_ctrl(m_s, m_t, r["qpos"].reshape(-1, nq), r["qvel"].reshape(-1, nv), r["ctrl"].reshape(-1, nu), m_s.opt.timestep)
        k = np.load(D / "0" / "trajectory_kinematic.npz")
        ctrl = k["ctrl"].copy()
        step = int(round(0.02 / m_s.opt.timestep))  # sim steps per reference frame
        n = min(len(ctrl), len(c[::step]))
        ctrl[:n] = c[::step][:n]
        shutil.rmtree(O, ignore_errors=True)
        (O / "0").mkdir(parents=True)
        for f in ("scene.xml", "task_info.json"):
            shutil.copy(D / f, O / f)
        np.savez(O / "0" / "trajectory_kinematic.npz", qpos=k["qpos"], qvel=k["qvel"], ctrl=ctrl)
        print(f"{tag}: warm-started {n}/{len(ctrl)} frames -> {O}")


if __name__ == "__main__":
    main()
