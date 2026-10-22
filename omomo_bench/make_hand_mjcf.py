#!/usr/bin/env python3
"""UMR side: export the G1 + hand URDFs to MJCF and write the UMR robot configs.

    MUJOCO_GL=disable python make_hand_mjcf.py

Writes
  robots/g1_29dof_{brainco_revo2,wuji_hand2}.xml    MJCF from MuJoCo's URDF compiler. Mesh files are written as absolute
                                                    paths (UMR's prepare_robot_xml drops `meshdir` for files in
                                                    sub-directories). Wuji gets the 10 <exclude> body pairs per hand
                                                    that Wuji's own MJCF uses (inherent 1-2 mm overlaps at the knuckles).
  configs/umr_g1_{none,brainco,wuji}.json           UMR-native scaling (object scaled by smpl_scale, no ground align)
  configs/umr_g1_{none,brainco,wuji}_omniscale.json OmniRetarget-style scaling (real-size object, human ground-aligned)
Hand joints are pinned at hand_rest_pose.json via joint_limits [v-1e-4, v+1e-4] and used as T-pose values, so UMR
optimises only the 29 body DoF while the hand geometry (at rest) takes part in surface matching and collision.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import mujoco

from hand_rest import HANDS, hand_rest_pose

HERE = Path(__file__).resolve().parent
WUJI_MJCF = HERE.parent / "urdf" / "third_party" / "wuji-description" / "hand2" / "hand2_beta2" / "body" / "mjcf"
UMR_G1_XML = HERE.parent / "UMR" / "assets" / "g1" / "g1_29dof.xml"
PIN = 1e-4

BASE_ROBOT = {
    "point_cloud_center": "body:waist_yaw_link",
    "height": 1.32,
    "tpose_qpos": {
        "left_shoulder_roll_joint": 1.57079632679, "left_elbow_joint": 1.57079632679,
        "right_shoulder_roll_joint": -1.57079632679, "right_elbow_joint": 1.57079632679,
    },
    "joint_limits": {
        "left_knee_joint": [0.1, "inf"], "right_knee_joint": [0.1, "inf"],
        "left_wrist_pitch_joint": [-0.3, 0.8], "right_wrist_pitch_joint": [-0.3, 0.8],
        "left_elbow_joint": ["-inf", 1.57079632679], "right_elbow_joint": ["-inf", 1.57079632679],
        "left_shoulder_pitch_joint": [-3.0, 1.4], "right_shoulder_pitch_joint": [-3.0, 1.4],
    },
}


def wuji_excludes() -> str:
    tags = []
    for side in ("left", "right"):
        tags += re.findall(r"<exclude[^>]*/>", (WUJI_MJCF / f"{side}_with_mount.xml").read_text())
    return "  <contact>\n    " + "\n    ".join(tags) + "\n  </contact>\n"


def export(hand: str, urdf: Path) -> tuple[Path, dict[str, float]]:
    m = mujoco.MjModel.from_xml_path(str(urdf))
    out = HERE / "robots" / f"{urdf.stem}.xml"
    out.parent.mkdir(exist_ok=True)
    mujoco.mj_saveLastXML(str(out), m)
    txt = out.read_text()
    meshdir = urdf.parent.resolve()
    txt, n = re.subn(r'<compiler([^>]*?)\s*meshdir="[^"]*"', r"<compiler\1", txt, count=1)
    assert n == 1, "compiler meshdir not found"
    txt, n = re.subn(r'(<mesh [^>]*?file=")([^"/][^"]*)"', lambda mm: f'{mm.group(1)}{(meshdir / mm.group(2)).as_posix()}"', txt)
    assert n > 0
    if hand == "wuji":
        txt = txt.replace("</mujoco>", wuji_excludes() + "</mujoco>")
    out.write_text(txt)
    m2 = mujoco.MjModel.from_xml_path(str(out))
    names = [mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m2.njnt)]
    limits = {nm: tuple(m2.jnt_range[j]) for j, nm in enumerate(names) if m2.jnt_limited[j]}
    rest = hand_rest_pose(names, limits)
    print(f"{out.relative_to(HERE)}: nq={m2.nq} geoms={m2.ngeom} excludes={m2.nexclude} hand joints pinned={len(rest)}")
    return out, rest


def write_config(hand: str, xml: Path, rest: dict[str, float], omniscale: bool) -> None:
    robot = json.loads(json.dumps(BASE_ROBOT))
    robot["name"] = "unitree_g1" if hand == "none" else f"unitree_g1_{hand}"
    robot["xml"] = str(xml.resolve())
    for j, v in rest.items():
        robot["tpose_qpos"][j] = v
        robot["joint_limits"][j] = [v - PIN, v + PIN]
    cfg = {"robot": robot, "hsi_hoi": {"object": {"convert_y_up": False}}}
    if omniscale:
        cfg["solver"] = {"retarget_object_size": "original"}
        cfg["retarget"] = {"source_ground_align": "global_foot_joint", "mat_height": 0.1}
    p = HERE / "configs" / f"umr_g1_{hand}{'_omniscale' if omniscale else ''}.json"
    p.write_text(json.dumps(cfg, indent=1))
    print("  ->", p.relative_to(HERE))


def main() -> None:
    for hand, urdf in HANDS.items():
        xml, rest = (UMR_G1_XML, {}) if urdf is None else export(hand, urdf)
        for omniscale in (False, True):
            write_config(hand, xml, rest, omniscale)


if __name__ == "__main__":
    main()
