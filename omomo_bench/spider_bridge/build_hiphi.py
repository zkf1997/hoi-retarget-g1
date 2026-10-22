"""Convert HiPHI retargets (G1 + Wuji Hand 2 + one object) into SPIDER's humanoid_object format.

For every sequence tag (``omomo_bench/wbt/data/motions/<tag>.npz``, 50 Hz) this writes

    <out>/processed/hiphi/g1_wuji/humanoid_object/<tag>/scene.xml        robot + object + floor (MuJoCo MJCF)
    <out>/processed/hiphi/g1_wuji/humanoid_object/<tag>/task_info.json   {"ref_dt": 0.02}
    <out>/processed/hiphi/g1_wuji/humanoid_object/<tag>/0/trajectory_kinematic.npz
        qpos (T, 83) = [root xyz, root quat wxyz, 69 joints, obj xyz, obj quat wxyz]; qvel (T, 81); ctrl (T, 69)

Scene: robot from the holosoma collision URDF (``wbt/assets/robot/g1_29dof_wuji_hand2_wbt_hscoll.urdf``: inertials fixed,
primitive body collision, mesh hand collision) + free root joint + one position actuator per joint (SPIDER's G1 style:
stiff body PD with armature, softer fingers, torques clipped to the URDF effort limits). Contacts use contype/conaffinity
bits instead of explicit pairs: robot geoms (feet, legs, arms, hands) collide with the object and the floor but not with
themselves; the object (CoACD pieces, scaled by the sequence scale s) collides with robot and floor.

    ~/software/venvs/spider/bin/python omomo_bench/spider_bridge/build_hiphi.py [--tags 009_... 052_...] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
WBT = HERE.parent / "wbt"
URDF = WBT / "assets" / "robot" / "g1_29dof_wuji_hand2_wbt_hscoll.urdf"
DEFAULT_OUT = HERE / "data"

BODY_KP, BODY_ARMATURE = 500.0, 1.0           # SPIDER unitree_g1 defaults
FINGER_KP, FINGER_ARMATURE = 5.0, 0.01
COLLIDE_BODY_RE = re.compile(r"(ankle_roll|knee|hip_roll|elbow|wrist|shoulder_yaw|torso|pelvis$)|^[lr]_")  # robot links with contacts


def _object_parts(tag: str):
    urdf = (WBT / "assets" / "objects" / tag / "hiphi_object.urdf").read_text()
    inertial = re.search(r"<inertial>.*?<mass value=\"([^\"]+)\"/>.*?<inertia ([^/]*)/>", urdf, re.S)
    mass = float(inertial.group(1))
    I = dict(re.findall(r'(\w+)="([^"]+)"', inertial.group(2)))
    visual = re.search(r"<visual>.*?<mesh filename=\"([^\"]+)\" scale=\"([^\"]+)\"", urdf, re.S)
    pieces = re.findall(r"<collision[^>]*>.*?<mesh filename=\"([^\"]+)\" scale=\"([^\"]+)\"", urdf, re.S)
    return mass, [float(I["ixx"]), float(I["iyy"]), float(I["izz"])], visual.groups(), pieces


REAL_GAINS = HERE / "holosoma_g1_wuji_gains.json"  # holosoma / real-robot PD gains, armature, effort (exported from hiphi_wbt.robot)


def _real_gain(name: str, table: dict) -> float:
    for k, v in table.items():
        if k in name:
            return float(v)
    raise KeyError(name)


def build_scene(tag: str, root0=None, actuator: str = "stiff", collision: str = "regex") -> mujoco.MjSpec:
    """collision: "regex" = SPIDER-style subset of links (COLLIDE_BODY_RE, visual meshes included);
    "urdf" = exactly the URDF collision geoms of every link, as in the holosoma/IsaacSim RL setup."""
    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.compiler.discardvisual = False
    for mesh in spec.meshes:  # relative URDF mesh paths -> absolute (scene.xml lives elsewhere)
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str((URDF.parent / mesh.file).resolve())
    spec.meshdir = ""
    spec.option.timestep = 0.01
    pelvis = spec.body("pelvis")
    pelvis.add_freejoint(name="floating_base_joint")
    # robot collision bits: collide with object (bit 2) and floor (bit 2), never with itself
    for g in spec.geoms:
        body = g.parent.name
        if (g.group == 0) if collision == "urdf" else COLLIDE_BODY_RE.search(body):  # URDF import: collision group 0, visual 1
            g.contype, g.conaffinity, g.condim = 2, 0, 3
            g.friction = [1.0, 0.02, 0.001]
        else:
            g.contype, g.conaffinity = 0, 0
    # actuators: one position actuator per hinge joint, in joint order
    urdf_text = URDF.read_text()
    effort = {m.group(1): float(re.search(r'effort="([^"]+)"', m.group(2)).group(1))
              for m in re.finditer(r'<joint name="([^"]+)" type="revolute">(.*?)</joint>', urdf_text, re.S)}
    real = json.loads(REAL_GAINS.read_text()) if actuator == "real" else None
    for j in spec.joints:
        if j.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        finger = j.name.startswith(("l_", "r_"))
        a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        if real is None:  # SPIDER's G1 style: very stiff, heavy armature
            j.armature = FINGER_ARMATURE if finger else BODY_ARMATURE
            a.set_to_position(kp=FINGER_KP if finger else BODY_KP, dampratio=1.0, inheritrange=1.0)
        else:  # real-robot gains / rotor inertia (holosoma G1 + Wuji); ctrl unlimited so feed-forward offsets fit
            j.armature = real["armature"][real["dof_names"].index(j.name)]
            a.set_to_position(kp=_real_gain(j.name, real["stiffness"]), kv=_real_gain(j.name, real["damping"]))
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-effort[j.name], effort[j.name]]
    # floor
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=3, conaffinity=3,
                            condim=3, friction=[1.0, 0.02, 0.001])
    # object: free body, CoACD pieces collide with robot (bit 2) and floor (bit 1)
    mass, diag, (vis_file, vis_scale), pieces = _object_parts(tag)
    obj = spec.worldbody.add_body(name="object", pos=[0, 0, 0])
    obj.add_freejoint(name="object_joint")
    obj.mass, obj.inertia, obj.explicitinertial = mass, diag, True
    obj.ipos, obj.iquat = [0, 0, 0], [1, 0, 0, 0]  # COM at the object origin (URDF inertial origin is 0 0 0)
    spec.add_mesh(name="object_visual", file=vis_file, scale=[float(v) for v in vis_scale.split()])
    obj.add_geom(name="object_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="object_visual", contype=0, conaffinity=0,
                 group=1, density=0, rgba=[0.75, 0.72, 0.66, 1])
    for k, (f, sc) in enumerate(pieces):
        spec.add_mesh(name=f"object_c{k:02d}", file=f, scale=[float(v) for v in sc.split()])
        obj.add_geom(name=f"object_c{k:02d}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"object_c{k:02d}", contype=1,
                     conaffinity=3, condim=3, group=3, density=0, friction=[1.0, 0.02, 0.001])
    # camera for SPIDER's video renderer: fixed position near the start, always looking at the pelvis
    r0 = np.zeros(3) if root0 is None else np.asarray(root0)
    spec.worldbody.add_camera(name="track", pos=(r0 + np.array([2.6, -2.6, 0.6])).tolist(),
                              mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="pelvis")
    # visualisation sites SPIDER's viewers like to see
    for side, link in (("left", "left_wrist_yaw_link"), ("right", "right_wrist_yaw_link")):
        spec.body(link).add_site(name=f"trace_{side}_hand", size=[0.01, 0, 0])
    return spec


def feedforward_ctrl(m: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, dt: float, cutoff_hz: float = 5.0,
                     max_offset: float = np.deg2rad(30)) -> tuple[np.ndarray, np.ndarray]:
    """Inverse dynamics of the reference robot motion (gravity + inertia + floor contacts; hand-object contacts excluded:
    the reference's small hand-object penetrations turn into huge soft-contact forces): joint torques tau, and the PD
    targets that produce them, ctrl = q_ref + (tau + kd*qd_ref) / kp, offset capped at max_offset. qacc by finite
    differences, low-pass filtered."""
    obj_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("object").id]
    saved = m.geom_contype[obj_geoms].copy(), m.geom_conaffinity[obj_geoms].copy()
    m.geom_contype[obj_geoms] = 0
    m.geom_conaffinity[obj_geoms] = 0
    from scipy.signal import butter, filtfilt
    qacc = np.gradient(qvel, dt, axis=0)
    b, a = butter(2, cutoff_hz / (0.5 / dt))
    qacc = filtfilt(b, a, qacc, axis=0)
    d = mujoco.MjData(m)
    nu = m.nu
    dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(nu)])
    qad = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(nu)])
    tau = np.zeros((len(qpos), nu))
    for t in range(len(qpos)):
        d.qpos[:], d.qvel[:], d.qacc[:] = qpos[t], qvel[t], qacc[t]
        mujoco.mj_inverse(m, d)
        tau[t] = d.qfrc_inverse[dof]
    lim = m.actuator_forcerange[:, 1]
    tau = np.clip(tau, -lim, lim)
    kp, kd = m.actuator_gainprm[:, 0], -m.actuator_biasprm[:, 2]
    offset = np.clip((tau + kd * qvel[:, dof]) / kp, -max_offset, max_offset)
    m.geom_contype[obj_geoms], m.geom_conaffinity[obj_geoms] = saved
    return qpos[:, qad] + offset, tau


def interaction_window(mot, seconds: float, lead_in: float = 1.0, fps: int = 50) -> int:
    """Start frame of a `seconds` clip: `lead_in` s before the (seconds - lead_in) s with the most object motion, or,
    when the object (almost) never moves (sit / step on a bench), the most pelvis-height motion."""
    w = int(round((seconds - lead_in) * fps))
    path = np.linalg.norm(np.diff(mot["object_pos_w"], axis=0), axis=1)
    if path.sum() < 0.1:
        path = np.abs(np.diff(mot["joint_pos"][:, 2]))
    cs = np.concatenate([[0.0], np.cumsum(path)])
    if len(cs) <= w:
        return 0
    t = int(np.argmax(cs[w:] - cs[:-w]))
    return max(0, t - int(round(lead_in * fps)))


def convert(tag: str, out: Path, actuator: str = "stiff", ff: str = "none", dataset: str = "hiphi", window: float = 0.0,
            pad: float = 1.0, collision: str = "regex") -> dict:
    """window > 0: keep only a `window` s interaction clip (+ `pad` s for SPIDER's planning horizon), see interaction_window."""
    task_dir = out / "processed" / dataset / "g1_wuji" / "humanoid_object" / tag
    (task_dir / "0").mkdir(parents=True, exist_ok=True)
    mot = dict(np.load(WBT / "data" / "motions" / f"{tag}.npz"))
    start = 0
    if window > 0:
        start = interaction_window(mot, window)
        n = int(round((window + pad) * 50))
        for k in ("joint_pos", "object_pos_w", "object_quat_w"):
            mot[k] = mot[k][start:start + n]
    spec = build_scene(tag, root0=mot["joint_pos"][0, :3], actuator=actuator, collision=collision)
    m = spec.compile()
    xml = spec.to_xml()
    (task_dir / "scene.xml").write_text(xml)
    (task_dir / "task_info.json").write_text(json.dumps({"ref_dt": 0.02, "start_frame": start}))
    names = [str(n) for n in mot["joint_names"]]
    jn = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    assert names == jn, "joint order mismatch between motion and scene"
    qpos = np.concatenate([mot["joint_pos"], mot["object_pos_w"], mot["object_quat_w"]], 1).astype(np.float64)
    assert qpos.shape[1] == m.nq, (qpos.shape, m.nq)
    qvel = np.zeros((len(qpos), m.nv))
    for t in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(m, qvel[t], 0.02, qpos[t], qpos[t + 1])
    qvel[-1] = qvel[-2]
    ctrl = qpos[:, 7:7 + 69]
    extra = {}
    if ff == "id":
        ctrl, tau = feedforward_ctrl(m, qpos, qvel, 0.02)
        extra = {"ff_offset_deg_p95": float(np.rad2deg(np.percentile(np.abs(ctrl - qpos[:, 7:76]), 95))), "tau_ff_absmax": float(np.abs(tau).max())}
    np.savez(task_dir / "0" / "trajectory_kinematic.npz", qpos=qpos, qvel=qvel, ctrl=ctrl)
    return {**extra, "tag": tag, "start_frame": start, "obj_z0": float(mot["object_pos_w"][0, 2]), "nq": m.nq, "nv": m.nv, "nu": m.nu, "ngeom": m.ngeom, "frames": len(qpos), "mass": float(m.body_subtreemass[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--actuator", choices=["stiff", "real"], default="stiff", help="stiff = SPIDER G1 (kp 500, armature 1); real = holosoma/real-robot gains")
    ap.add_argument("--ff", choices=["none", "id"], default="none", help="id = inverse-dynamics feed-forward folded into the nominal ctrl")
    ap.add_argument("--dataset", default="hiphi", help="dataset_name of the processed tree (e.g. hiphi_real)")
    ap.add_argument("--collision", choices=["regex", "urdf"], default="regex", help="urdf = all URDF collision geoms (matches the RL setup)")
    ap.add_argument("--window", type=float, default=0.0, help="> 0: cut a clip of this many seconds around the interaction")
    a = ap.parse_args()
    tags = a.tags or sorted(p.stem for p in (WBT / "data" / "motions").glob("*.npz"))
    for t in tags:
        print(convert(t, a.out, a.actuator, a.ff, a.dataset, a.window, collision=a.collision), flush=True)


if __name__ == "__main__":
    main()
