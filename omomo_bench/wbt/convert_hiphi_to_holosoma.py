#!/usr/bin/env python3
"""HiPHI retarget results (results_hiphi/pipe_final) -> holosoma whole-body-tracking motion files + per-sequence object URDFs.

Per sequence <seq> (index i in data_hiphi/seqs_test.json):
  wbt/data/motions/<III>_<seq>.npz   holosoma MotionLoader format at OUT_FPS:
        fps, joint_names (69), body_names (103 = all URDF links + 2 foot contact points),
        joint_pos (T, 7+69) [root xyz, root quat wxyz, joints], joint_vel (T, 6+69) [root lin vel, root ang vel (world), joint vel],
        body_pos_w (T,B,3), body_quat_w (T,B,4) wxyz, body_lin_vel_w, body_ang_vel_w (world frame),
        object_pos_w (T,3), object_quat_w (T,4) wxyz, object_lin_vel_w (T,3)
  wbt/assets/objects/<III>_<seq>/hiphi_object.urdf  the sequence's object (same file name in every folder: IsaacSim names the prim after it) at the sequence's scale s (visual = display mesh, collision = CoACD
        pieces, all <mesh scale="s s s">), mass = nominal mass x s^3
The index prefix fixes the file order (sorted) so that motion i <-> object asset i in the training scene.
Everything is in the scaled world the retargeting used: robot real size, human/object x s, ground z = 0.
    BENCH_DATASET=hiphi conda run -n umr python wbt/convert_hiphi_to_holosoma.py [--fps 50] [--seqs ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = Path(__file__).resolve().parent          # omomo_bench/wbt
BENCH = HERE.parent                              # omomo_bench
sys.path.insert(0, str(BENCH))
DS = os.environ.get("BENCH_DATASET", "hiphi")
DATA = BENCH / (f"data_{DS}" if DS != "omomo" else "data")
RESULTS = BENCH / (f"results_{DS}" if DS != "omomo" else "results")
URDF = HERE / "assets" / "robot" / "g1_29dof_wuji_hand2_wbt.urdf"
MJCF = BENCH / "robots" / "g1_29dof_wuji_hand2.xml"
FOOT_OFFSET = np.array([0.0, 0.0, -0.037])  # foot_contact_point below ankle_roll_link (holosoma g1_29dof.urdf)

# nominal real-size masses (kg); scaled by s^3 in the scaled world
NOMINAL_MASS = {"Box_A_1": 2.0, "Chair_Q_1": 5.0, "Ball_A_1": 0.4, "StepStool_D_1": 3.0, "Bucket_B_1": 1.5, "Clothrack_A_1": 4.0,
                "ScoccerBall_A_1": 0.45, "Trashbin_C_1": 2.0, "Mop_A_1": 1.0, "Bench_B_1": 8.0, "Bottle_A_1": 0.5, "Table_A_1": 12.0, "Table_C_1": 10.0}


def urdf_body_names(urdf: Path) -> list[str]:
    return re.findall(r'<link name="([^"]+)"', urdf.read_text())


def resample_qpos(q: np.ndarray, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    """q (T, 7+J): linear interpolation for xyz + joints, slerp for the root quaternion (wxyz)."""
    out = np.zeros((len(t_out), q.shape[1]))
    for c in list(range(3)) + list(range(7, q.shape[1])):
        out[:, c] = np.interp(t_out, t_in, q[:, c])
    rots = Rot.from_quat(q[:, [4, 5, 6, 3]])  # wxyz -> xyzw
    out[:, 3:7] = Slerp(t_in, rots)(np.clip(t_out, t_in[0], t_in[-1])).as_quat()[:, [3, 0, 1, 2]]
    return out


def resample_pose(pos: np.ndarray, quat_xyzw: np.ndarray, t_in, t_out):
    p = np.stack([np.interp(t_out, t_in, pos[:, c]) for c in range(3)], 1)
    q = Slerp(t_in, Rot.from_quat(quat_xyzw))(np.clip(t_out, t_in[0], t_in[-1])).as_quat()
    return p, q


def ang_vel_from_quats(q_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity by central differences of the rotation: w = log(R_{t+1} R_{t-1}^T) / (2 dt)."""
    R = Rot.from_quat(q_xyzw)
    T = len(q_xyzw); w = np.zeros((T, 3))
    if T > 2:
        w[1:-1] = (R[2:] * R[:-2].inv()).as_rotvec() / (2 * dt)
        w[0] = (R[1] * R[0].inv()).as_rotvec() / dt
        w[-1] = (R[-1] * R[-2].inv()).as_rotvec() / dt
    return w


def object_urdf(obj: str, s: float, out: Path, display_mesh: Path, pieces: list[Path], mass: float, extents: np.ndarray) -> None:
    ex = extents * s
    ixx, iyy, izz = (mass / 12.0 * (ex[1] ** 2 + ex[2] ** 2), mass / 12.0 * (ex[0] ** 2 + ex[2] ** 2), mass / 12.0 * (ex[0] ** 2 + ex[1] ** 2))
    sc = f"{s:.6f} {s:.6f} {s:.6f}"
    cols = "\n".join(f'    <collision name="c{i:02d}"><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{p}" scale="{sc}"/></geometry></collision>'
                     for i, p in enumerate(pieces))
    out.write_text(f'''<?xml version="1.0"?>
<robot name="hiphi_object">
  <link name="object_body">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{mass:.4f}"/><inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{display_mesh}" scale="{sc}"/></geometry><material name="obj"><color rgba="0.75 0.72 0.66 1"/></material></visual>
{cols}
  </link>
</robot>
''')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--lane", default="pipe_final")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    a = ap.parse_args()
    (a.out / "motions").mkdir(parents=True, exist_ok=True)
    (HERE / "assets" / "objects").mkdir(parents=True, exist_ok=True)

    body_names = urdf_body_names(URDF)
    m = mujoco.MjModel.from_xml_path(str(MJCF)); d = mujoco.MjData(m)
    mj_body = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(m.nbody)}
    mj_joint = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): int(m.jnt_qposadr[j]) for j in range(m.njnt)}
    assert all((bn in mj_body) or bn.endswith("foot_contact_point") for bn in body_names), "URDF/MJCF body mismatch"
    index = json.loads((DATA / "seqs_test.json").read_text())
    dt = 1.0 / a.fps
    manifest = []
    for i, e in enumerate(index):
        seq = e["seq"]
        if a.seqs and seq not in a.seqs:
            continue
        r = np.load(RESULTS / a.lane / f"{seq}.npz", allow_pickle=True)
        src = np.load(DATA / "source" / f"{seq}.npz", allow_pickle=True)
        s = float(np.asarray(r["smpl_scale"]).reshape(-1)[0]); fps_in = float(np.asarray(r["fps"]).reshape(-1)[0])
        names = [str(x) for x in r["robot_joint_names"]]
        q_in = r["qpos"].astype(np.float64); T_in = min(len(q_in), len(src["trans"]))
        q_in = q_in[:T_in]
        t_in = np.arange(T_in) / fps_in; t_out = np.arange(0.0, t_in[-1] + 1e-9, dt)
        q = resample_qpos(q_in, t_in, t_out); T = len(q)
        obj = str(src["object"])
        op, oq = resample_pose(src["obj_pos"][:T_in].astype(np.float64) * s, src["obj_quat_xyzw"][:T_in].astype(np.float64), t_in, t_out)
        # FK for every body
        B = len(body_names)
        bpos = np.zeros((T, B, 3)); bquat = np.zeros((T, B, 4))
        qadr = [mj_joint[n] for n in names]
        for t in range(T):
            d.qpos[:] = 0
            for adr, v in zip(qadr, q[t, 7:]):
                d.qpos[adr] = v
            mujoco.mj_kinematics(m, d)
            # the MJCF has no free joint: FK is in the pelvis frame; place it at the root pose
            R0 = Rot.from_quat(q[t, [4, 5, 6, 3]]).as_matrix(); p0 = q[t, :3]
            for k, bn in enumerate(body_names):
                if bn.endswith("foot_contact_point"):
                    side = bn.split("_")[0]; b = mj_body[f"{side}_ankle_roll_link"]
                    Rb = R0 @ d.xmat[b].reshape(3, 3); bpos[t, k] = p0 + R0 @ d.xpos[b] + Rb @ FOOT_OFFSET
                else:
                    b = mj_body[bn]; Rb = R0 @ d.xmat[b].reshape(3, 3); bpos[t, k] = p0 + R0 @ d.xpos[b]
                bquat[t, k] = Rot.from_matrix(Rb).as_quat()[[3, 0, 1, 2]]
        blin = np.gradient(bpos, dt, axis=0)
        bang = np.stack([ang_vel_from_quats(bquat[:, k][:, [1, 2, 3, 0]], dt) for k in range(B)], 1)
        root_lin = np.gradient(q[:, :3], dt, axis=0); root_ang = ang_vel_from_quats(q[:, [4, 5, 6, 3]], dt)
        jvel = np.gradient(q[:, 7:], dt, axis=0)
        joint_vel = np.concatenate([root_lin, root_ang, jvel], 1)
        tag = f"{i:03d}_{seq}"
        np.savez(a.out / "motions" / f"{tag}.npz", fps=np.array(a.fps), joint_names=np.array(names), body_names=np.array(body_names),
                 joint_pos=q.astype(np.float32), joint_vel=joint_vel.astype(np.float32),
                 body_pos_w=bpos.astype(np.float32), body_quat_w=bquat.astype(np.float32),
                 body_lin_vel_w=blin.astype(np.float32), body_ang_vel_w=bang.astype(np.float32),
                 object_pos_w=op.astype(np.float32), object_quat_w=oq[:, [3, 0, 1, 2]].astype(np.float32),
                 object_lin_vel_w=np.gradient(op, dt, axis=0).astype(np.float32),
                 object_name=np.array(obj), smpl_scale=np.array(s), source_seq=np.array(seq))
        # object asset at this sequence's scale
        od = DATA / "objects" / obj; meta = json.loads((od / "meta.json").read_text())
        pieces = sorted(od.glob(f"{obj}_coacd_*.obj"))
        disp = od / f"{obj}_display.obj" if (od / f"{obj}_display.obj").exists() else od / f"{obj}.obj"
        mass = NOMINAL_MASS.get(obj, 2.0) * s ** 3
        object_urdf(obj, s, HERE / "assets" / "objects" / tag / "hiphi_object.urdf", disp.resolve(), [p.resolve() for p in pieces], mass, np.array(meta["extents_m"]))
        manifest.append({"index": i, "seq": seq, "tag": tag, "object": obj, "smpl_scale": s, "frames": T, "mass_kg": mass})
        print(f"[{i:02d}] {seq}: {T_in} -> {T} frames @ {a.fps} fps, object {obj} x {s:.3f}, mass {mass:.2f} kg, {len(pieces)} pieces", flush=True)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(manifest)} motions -> {a.out / 'motions'}; objects -> {HERE / 'assets' / 'objects'}")


if __name__ == "__main__":
    main()
