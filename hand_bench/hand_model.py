"""MuJoCo-backed kinematic model of the two G1 hands (BrainCo Revo2, Wuji Hand 2) shared by both retargeting methods and by the metrics.

Base frame = `{side}_wrist_yaw_link` (G1 wrist convention, see make_hand_urdfs.py). The 21 "robot keypoints" follow the
MediaPipe layout used by HUMOTO / TopoRetarget / dex-retargeting:
    0 wrist  -> flange link origin (BrainCo `*_base2_link`, Wuji `*_mount`; both at x = 0.0415 in the base frame)
    1-4  thumb  CMC, MCP, IP, TIP
    5-8  index  MCP, PIP, DIP, TIP  (BrainCo fingers have no DIP joint: DIP := midpoint of the distal link)
    9-12 middle, 13-16 ring, 17-20 pinky
Active joints: BrainCo 6 per hand (thumb metacarpal + proximal, 4 finger proximal; the 5 distal joints follow the URDF <mimic>,
1.0x thumb / 1.155x fingers as on the hardware), Wuji 20 per hand (all).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from make_hand_urdfs import hand_urdf_path

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
HUMAN_TIPS = [4, 8, 12, 16, 20]


def _kp_spec(hand: str, side: str) -> list:
    """21 entries; each is a body name or (body_a, body_b, alpha) = a + alpha (b - a)."""
    s = side
    if hand == "brainco":
        kp = [f"{s}_base2_link",
              f"{s}_thumb_metacarpal_link", f"{s}_thumb_proximal_link", f"{s}_thumb_distal_link", f"{s}_thumb_tip_link"]
        for f in FINGERS[1:]:
            kp += [f"{s}_{f}_proximal_link", f"{s}_{f}_distal_link", (f"{s}_{f}_distal_link", f"{s}_{f}_tip_link", 0.5), f"{s}_{f}_tip_link"]
        return kp
    if hand == "wuji":
        c = s[0]
        kp = [f"{c}_mount", f"{c}_thumb_proximal", f"{c}_thumb_middle", f"{c}_thumb_distal", f"{c}_thumb_tip"]
        for f in ("index_finger", "middle_finger", "ring_finger", "pinky"):
            kp += [f"{c}_{f}_proximal", f"{c}_{f}_middle", f"{c}_{f}_distal", f"{c}_{f}_tip"]
        return kp
    raise KeyError(hand)


def active_joint_names(hand: str, side: str) -> list[str]:
    if hand == "brainco":
        return [f"{side}_thumb_metacarpal_joint", f"{side}_thumb_proximal_joint"] + [f"{side}_{f}_proximal_joint" for f in FINGERS[1:]]
    c = side[0]
    names = [f"{c}_thumb_cmc_flex", f"{c}_thumb_cmc_abd", f"{c}_thumb_mcp", f"{c}_thumb_ip"]
    for f in ("index_finger", "middle_finger", "ring_finger", "pinky"):
        names += [f"{c}_{f}_mcp_flex", f"{c}_{f}_mcp_abd", f"{c}_{f}_pip", f"{c}_{f}_dip"]
    return names


def tip_link_names(hand: str, side: str) -> list[str]:
    spec = _kp_spec(hand, side)
    return [spec[i] for i in HUMAN_TIPS]


def flange_link_name(hand: str, side: str) -> str:
    return _kp_spec(hand, side)[0]


def parse_mimic(urdf: Path) -> dict[str, tuple[str, float, float]]:
    out = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        m = j.find("mimic")
        if m is not None:
            out[j.get("name")] = (m.get("joint"), float(m.get("multiplier", 1.0)), float(m.get("offset", 0.0)))
    return out


class HandModel:
    def __init__(self, hand: str, side: str):
        self.hand, self.side = hand, side
        self.urdf = hand_urdf_path(hand, side)
        self.model = mujoco.MjModel.from_xml_path(str(self.urdf))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        self.qadr = {n: int(m.jnt_qposadr[j]) for j, n in enumerate(self.joint_names)}
        self.active = active_joint_names(hand, side)
        self.active_qadr = np.array([self.qadr[n] for n in self.active])
        self.mimic = parse_mimic(self.urdf)
        self.mimic_rows = [(self.qadr[mj], self.qadr[src], k, b) for mj, (src, k, b) in self.mimic.items()]
        self.limits = np.array([m.jnt_range[self.joint_names.index(n)] for n in self.active])  # (n_active, 2)
        self.n_active = len(self.active)
        spec = _kp_spec(hand, side)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.kp_a = np.array([bid(e if isinstance(e, str) else e[0]) for e in spec])
        self.kp_b = np.array([bid(e if isinstance(e, str) else e[1]) for e in spec])
        self.kp_alpha = np.array([0.0 if isinstance(e, str) else e[2] for e in spec])
        self.tip_bodies = self.kp_a[HUMAN_TIPS]
        self.flange = self.keypoints(np.zeros(self.n_active))[0].copy()  # flange origin in base frame (fixed)
        kp0 = self.keypoints(np.zeros(self.n_active))
        self.size = float(chain_length(kp0))  # flange -> middle MCP -> PIP -> DIP -> tip (pose invariant)
        self.hand_geoms = np.arange(m.ngeom)

    # --- joints ----------------------------------------------------------------------------------------------------
    def full_qpos(self, q_active: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[self.active_qadr] = q_active
        for mj, src, k, b in self.mimic_rows:
            q[mj] = k * q[src] + b
        return q

    def clip(self, q_active: np.ndarray) -> np.ndarray:
        return np.clip(q_active, self.limits[:, 0], self.limits[:, 1])

    def qpos_dict(self, q_active: np.ndarray) -> dict[str, float]:
        q = self.full_qpos(q_active)
        return {n: float(q[self.qadr[n]]) for n in self.joint_names}

    # --- kinematics ------------------------------------------------------------------------------------------------
    def forward(self, q_active: np.ndarray) -> None:
        self.data.qpos[:] = self.full_qpos(q_active)
        mujoco.mj_kinematics(self.model, self.data)

    def keypoints(self, q_active: np.ndarray) -> np.ndarray:
        """(21,3) keypoints in the base (wrist_yaw) frame."""
        self.forward(q_active)
        xa, xb = self.data.xpos[self.kp_a], self.data.xpos[self.kp_b]
        return xa + self.kp_alpha[:, None] * (xb - xa)

    def keypoints_world(self, q_active: np.ndarray, base_pos: np.ndarray, base_rot: np.ndarray) -> np.ndarray:
        return self.keypoints(q_active) @ base_rot.T + base_pos

    def base_from_wrist(self, wrist_pos: np.ndarray, wrist_rot: np.ndarray) -> np.ndarray:
        """Base (wrist_yaw) origin such that the flange sits at `wrist_pos` with orientation `wrist_rot`."""
        return wrist_pos - wrist_rot @ self.flange

    def geom_poses(self, q_active: np.ndarray):
        self.forward(q_active)
        return self.data.geom_xpos.copy(), self.data.geom_xmat.copy().reshape(-1, 3, 3)


MIDDLE_CHAIN = [0, 9, 10, 11, 12]


def chain_length(kp: np.ndarray) -> np.ndarray:
    """summed middle-finger chain length wrist->MCP->PIP->DIP->tip; kp (...,21,3) -> (...)"""
    return np.linalg.norm(kp[..., MIDDLE_CHAIN[1:], :] - kp[..., MIDDLE_CHAIN[:-1], :], axis=-1).sum(-1)


def human_to_robot_scale(model: HandModel, human_kp_frame: np.ndarray) -> float:
    """robot / human hand size, both measured as the middle-finger chain length (independent of finger flexion)."""
    return model.size / float(np.median(chain_length(human_kp_frame)))


if __name__ == "__main__":
    for hand in ("brainco", "wuji"):
        for side in ("right", "left"):
            hm = HandModel(hand, side)
            kp = hm.keypoints(np.zeros(hm.n_active))
            print(f"{hand} {side}: nq={hm.model.nq} active={hm.n_active} mimic={len(hm.mimic)} flange={np.round(hm.flange, 4)} size={hm.size:.4f}")
            print("   limits lo", np.round(hm.limits[:, 0], 2), "hi", np.round(hm.limits[:, 1], 2))
            print("   kp (cm):", np.round(kp * 100, 1).tolist())
            qmid = hm.limits.mean(1)
            print("   closed-ish tips (cm):", np.round(hm.keypoints(qmid)[HUMAN_TIPS] * 100, 1).tolist())
