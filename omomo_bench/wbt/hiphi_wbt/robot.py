"""RobotConfig for Unitree G1 29-DoF + Wuji Hand 2 (69 actuated joints, 103 links incl. two foot contact points).

Everything that is not hand-specific is copied from holosoma's g1_29dof preset; joint limits / efforts / velocities come from
the URDF; body-joint armatures from the G1 preset; finger gains are chosen so that the PD torque saturates at about half the
joint range (finger effort limits are 0.2 - 0.6 Nm).
"""
from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

from holosoma.config_types.robot import RobotAssetConfig, RobotInitState
from holosoma.config_values import robot as _robot

HERE = Path(__file__).resolve().parent.parent  # omomo_bench/wbt
ASSET_ROOT = HERE / "assets" / "robot"
URDF = ASSET_ROOT / "g1_29dof_wuji_hand2_wbt.urdf"
MJCF = HERE.parent / "robots" / "g1_29dof_wuji_hand2.xml"  # joint order of the retarget results (= UMR MJCF order)

FINGER_RE = re.compile(r"^[lr]_")

# finger PD gains (Nm/rad, Nm s/rad); keys are matched as substrings of the joint name (holosoma convention)
_KP_SCALE = float(os.environ.get("HIPHI_WBT_FINGER_KP", "1.0"))  # multiplies the finger gains below (kd scales with sqrt)
FINGER_STIFFNESS = {k: v * _KP_SCALE for k, v in {"thumb_cmc": 1.0, "thumb_mcp": 0.6, "thumb_ip": 0.6, "finger_mcp": 0.6, "finger_pip": 0.6,
                                                   "finger_dip": 0.6, "pinky_mcp": 0.6, "pinky_pip": 0.6, "pinky_dip": 0.6}.items()}
FINGER_DAMPING = {k: 0.03 * _KP_SCALE ** 0.5 for k in FINGER_STIFFNESS}
FINGER_ARMATURE = 0.0002


FROZEN_HAND = os.environ.get("HIPHI_WBT_FROZEN_HAND", "0") == "1"  # diagnostic: finger joints fixed (29-DoF policy, hands open)
# collision variant of the body: "hscoll" = body collision shapes copied from holosoma's G1 URDF (cylinder feet instead of the
# four 5 mm spheres of the Unitree rev_1_0 URDF, sphere pelvis, cylinder torso); "" = original meshes/spheres
URDF_VARIANT = os.environ.get("HIPHI_WBT_URDF", "hscoll")


def _base_urdf() -> Path:
    return ASSET_ROOT / (URDF.stem + (f"_{URDF_VARIANT}" if URDF_VARIANT else "") + ".urdf")


def _frozen_hand_urdf(base: Path) -> Path:
    out = base.with_name(base.stem + "_frozenhand.urdf")
    if not out.exists() or out.stat().st_mtime < base.stat().st_mtime:
        text = base.read_text()

        def freeze(m: re.Match) -> str:
            if FINGER_RE.match(m.group(1)):
                body = re.sub(r"<(limit|axis|dynamics)[^>]*/>", "", m.group(2))
                return f'<joint name="{m.group(1)}" type="fixed">{body}</joint>'
            return m.group(0)

        out.write_text(re.sub(r'<joint name="([^"]+)" type="revolute">(.*?)</joint>', freeze, text, flags=re.S))
    return out


def _urdf_variant() -> str:
    """Profiling knob: HIPHI_WBT_NO_HAND_COLLISION=1 strips the collision geometry of every hand link ([lr]_*)."""
    base = _base_urdf()
    if FROZEN_HAND:
        return _frozen_hand_urdf(base).name
    if os.environ.get("HIPHI_WBT_NO_HAND_COLLISION", "0") != "1":
        return base.name
    out = ASSET_ROOT / (base.stem + "_nohandcol.urdf")
    text = base.read_text()

    def strip(m: re.Match) -> str:
        name, body = m.group(1), m.group(2)
        if FINGER_RE.match(name):
            body = re.sub(r"<collision>.*?</collision>", "", body, flags=re.S)
        return f'<link name="{name}">{body}</link>'

    out.write_text(re.sub(r'<link name="([^"]+)">(.*?)</link>', strip, text, flags=re.S))
    print(f"[hiphi_wbt.robot] profiling variant without hand collision: {out}")
    return out.name


def _urdf_joints(urdf_text: str) -> dict[str, dict]:
    out = {}
    for jm in re.finditer(r'<joint name="([^"]+)" type="(revolute|continuous|prismatic)">(.*?)</joint>', urdf_text, re.S):
        lim = re.search(r"<limit([^>]*)/>", jm.group(3))
        out[jm.group(1)] = {k: float(v) for k, v in re.findall(r'(\w+)="([^"]+)"', lim.group(1))} if lim else {}
    return out


def _mjcf_joint_order(mjcf_text: str) -> list[str]:
    return re.findall(r'<joint[^>]*name="([^"]+)"', mjcf_text)


_ARM_KP = float(os.environ.get("HIPHI_WBT_ARM_KP", "1.0"))  # multiplies the arm (shoulder/elbow/wrist) PD gains; kd scales with sqrt


def _arm_scaled(gains: dict, scale: float, kd: bool) -> dict:
    keys = ("shoulder", "elbow", "wrist")
    return {k: (v * (scale ** 0.5 if kd else scale) if any(a in k for a in keys) else v) for k, v in gains.items()}


def build_robot_config():
    g1 = _robot.g1_29dof
    urdf_text = (ASSET_ROOT / _urdf_variant()).read_text()
    joints = _urdf_joints(urdf_text)
    order = [j for j in _mjcf_joint_order(MJCF.read_text()) if j in joints]
    assert len(order) == (29 if FROZEN_HAND else 69) and set(order) == set(joints), (len(order), len(joints))
    body_names = re.findall(r'<link name="([^"]+)"', urdf_text)
    assert len(body_names) in (93, 103), len(body_names)  # 93 = variant without the 10 fingertip sensor-frame links
    # Merge fixed-joint links (IsaacLab merge_fixed_joints) EXCEPT the joints marked dont_collapse (foot contact points,
    # fingertips). Default ON since 2026-10-14: with all 103 links kept, the PhysX articulation (71 fixed-joint bodies, many
    # of a few grams) could not even hold the arms open-loop and the policy learned ~3x slower (see README 5.6).
    collapse = os.environ.get("HIPHI_WBT_COLLAPSE", "1") == "1"
    if collapse:
        merged = set()
        for jm in re.finditer(r'<joint name="[^"]+" type="fixed"([^>]*)>(.*?)</joint>', urdf_text, re.S):
            if 'dont_collapse="true"' not in jm.group(1):
                merged.add(re.search(r'<child link="([^"]+)"', jm.group(2)).group(1))
        body_names = [b for b in body_names if b not in merged]
        print(f"[hiphi_wbt.robot] collapse_fixed_joints: {len(body_names)} bodies survive")

    body_idx = {n: i for i, n in enumerate(g1.dof_names)}
    lower = [joints[j]["lower"] for j in order]
    upper = [joints[j]["upper"] for j in order]
    # body joints: effort / velocity limits from holosoma's G1 preset (our rev_1_0 URDF says 35 Nm for ankles / waist roll+pitch,
    # holosoma uses 50 Nm); fingers from the URDF
    g1_eff = dict(zip(g1.dof_names, g1.dof_effort_limit_list))
    g1_vel = dict(zip(g1.dof_names, g1.dof_vel_limit_list))
    vel = [g1_vel.get(j, joints[j]["velocity"]) for j in order]
    eff = [g1_eff.get(j, joints[j]["effort"]) for j in order]
    arm = [g1.dof_armature_list[body_idx[j]] if j in body_idx else FINGER_ARMATURE for j in order]
    fric = [0.0] * len(order)
    defaults = dict(g1.init_state.default_joint_angles)
    for j in order:
        defaults.setdefault(j, 0.0)  # fingers: zero (the action term tracks the reference finger pose as a residual anyway)
    fingers = [j for j in order if FINGER_RE.match(j)]

    return replace(
        g1,
        num_bodies=len(body_names),
        dof_obs_size=len(order),
        actions_dim={"1": len(order) - 40 + 12, "body": len(order) - 40}.get(os.environ.get("HIPHI_WBT_SYNERGY", "0"), len(order)) if not FROZEN_HAND else len(order),
        dof_names=order,
        upper_dof_names=list(g1.upper_dof_names) + fingers,
        dof_pos_lower_limit_list=lower,
        dof_pos_upper_limit_list=upper,
        dof_vel_limit_list=vel,
        dof_effort_limit_list=eff,
        dof_armature_list=arm,
        dof_joint_friction_list=fric,
        body_names=body_names,
        init_state=RobotInitState(pos=[0.0, 0.0, 0.76], rot=list(g1.init_state.rot), lin_vel=[0.0] * 3, ang_vel=[0.0] * 3,
                                  default_joint_angles=defaults),
        control=replace(
            g1.control,
            stiffness={**_arm_scaled(g1.control.stiffness, _ARM_KP, False), **FINGER_STIFFNESS},
            damping={**_arm_scaled(g1.control.damping, _ARM_KP, True), **FINGER_DAMPING},
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=RobotAssetConfig(
            asset_root=str(ASSET_ROOT),
            collapse_fixed_joints=collapse,
            replace_cylinder_with_capsule=True,
            flip_visual_attachments=False,
            armature=0.001,
            thickness=0.01,
            urdf_file=_urdf_variant(),
            usd_file=None,
            xml_file=str(MJCF),
            robot_type="g1_29dof_wuji",
            # OFF by default: the convex hulls of consecutive finger links overlap, so self-collision pushes the fingers
            # 15-40 deg away from the reference pose (replay test 2026-10-09); hand-object collisions are unaffected
            enable_self_collisions=os.environ.get("HIPHI_WBT_SELF_COLL", "0") == "1",
            default_dof_drive_mode=3,
            fix_base_link=False,
            link_physics=g1.asset.link_physics,
        ),
    )


g1_wuji = build_robot_config()

FINGER_JOINTS = [j for j in g1_wuji.dof_names if FINGER_RE.match(j)]
FINGERTIP_BODIES = [f"{c}_{t}" for c in ("l", "r") for t in ("thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip")]
WRIST_BODIES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
