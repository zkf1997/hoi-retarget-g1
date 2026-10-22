"""Shared helpers: hand rest pose (hand_rest_pose.json) and the G1 hand models used by the benchmark."""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REST_JSON = HERE / "hand_rest_pose.json"

G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

HANDS = {
    "none": None,
    "brainco": HERE.parent / "urdf" / "g1_brainco_revo2" / "g1_29dof_brainco_revo2.urdf",
    "wuji": HERE.parent / "urdf" / "g1_wuji_hand2" / "g1_29dof_wuji_hand2.urdf",
}


def rest_value(joint: str, limits: tuple[float, float] | None = None, cfg: dict | None = None) -> float:
    cfg = cfg or json.loads(REST_JSON.read_text())
    v = float(cfg.get("default", 0.0))
    for rule in cfg.get("rules", []):
        if fnmatch.fnmatch(joint, rule["match"]):
            v = float(rule["value"])
    if limits is not None:
        v = float(min(max(v, limits[0]), limits[1]))
    return v


def hand_rest_pose(joint_names, limits: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    """rest values for every joint in `joint_names` that is not one of the 29 G1 body joints."""
    cfg = json.loads(REST_JSON.read_text())
    return {j: rest_value(j, (limits or {}).get(j), cfg) for j in joint_names if j not in G1_JOINTS}
