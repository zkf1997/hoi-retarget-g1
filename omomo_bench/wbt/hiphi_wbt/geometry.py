"""Hand-object geometry for the HiPHI tracking task: per-object SDF grids, hand keypoints, reference distances.

A single ``HandObjectGeometry`` instance is cached on the environment (``env._hiphi_geo``) and shared by the observation
terms (obs.py), the reward terms (rewards.py) and the metrics (command.py). Environment i uses object i % N, matching
sim_multiasset.IsaacSim and FixedMotionCommand.

Keypoints per hand (21): wrist + {proximal, middle, distal, tip} of the 5 fingers (link origins). The same points are
evaluated on the reference motion (motion body_pos_w + reference object pose, unsigned distance) once at setup so that the
contact-excess reward can compare the simulated hand-object distance with the demonstrated one frame by frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from holosoma.utils.rotations import quat_rotate, quat_rotate_inverse

FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
SEGMENTS = ("proximal", "middle", "distal", "tip")


WRIST_LINK = {"l": "left_wrist_yaw_link", "r": "right_wrist_yaw_link"}  # the hand base (l_wrist) is merged into this link


def hand_keypoint_bodies(side: str) -> list[str]:
    return [WRIST_LINK[side]] + [f"{side}_{f}_{s}" for f in FINGERS for s in SEGMENTS]


KEYPOINT_BODIES = hand_keypoint_bodies("l") + hand_keypoint_bodies("r")  # 42
TIP_BODIES = [f"{c}_{f}_tip" for c in ("l", "r") for f in FINGERS]  # 10
TIP_IDX_IN_KEYPOINTS = [KEYPOINT_BODIES.index(b) for b in TIP_BODIES]


class ObjectSDF:
    """Stack of per-object SDF grids with batched trilinear sampling (points given in the object frame)."""

    def __init__(self, sdf_dir: Path, device: str):
        files = sorted(Path(sdf_dir).glob("*.npz"))
        assert files, f"no SDF grids in {sdf_dir} (run build_object_sdf.py)"
        grids, origins, spacings = [], [], []
        for f in files:
            d = np.load(f)
            grids.append(torch.tensor(d["sdf"].astype(np.float32)))
            origins.append(torch.tensor(d["origin"]))
            spacings.append(torch.tensor(d["spacing"]))
        self.names = [f.stem for f in files]
        self.grid = torch.stack(grids).to(device)  # (N, R, R, R)
        self.res = self.grid.shape[1]
        self.origin = torch.stack(origins).to(device)  # (N, 3)
        self.spacing = torch.stack(spacings).to(device)  # (N, 3)
        self.device = device
        self.max_dist = float(self.grid.max())

    def sample(self, obj: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """obj (E,) long object index; pts (E, P, 3) in the object frame -> signed distance (E, P) [m]."""
        E, P, _ = pts.shape
        u = (pts - self.origin[obj][:, None]) / self.spacing[obj][:, None]  # voxel coords
        u = u.clamp(0.0, self.res - 1.001)
        i0 = u.floor().long()
        f = u - i0.float()
        R = self.res
        base = obj[:, None] * (R * R * R)
        flat = self.grid.view(-1)

        def g(dx, dy, dz):
            idx = base + (i0[..., 0] + dx) * (R * R) + (i0[..., 1] + dy) * R + (i0[..., 2] + dz)
            return flat[idx]

        fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
        c00 = g(0, 0, 0) * (1 - fx) + g(1, 0, 0) * fx
        c10 = g(0, 1, 0) * (1 - fx) + g(1, 1, 0) * fx
        c01 = g(0, 0, 1) * (1 - fx) + g(1, 0, 1) * fx
        c11 = g(0, 1, 1) * (1 - fx) + g(1, 1, 1) * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz

    def sample_with_normal(self, obj: torch.Tensor, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Signed distance (E,P) and unit gradient (E,P,3) in the object frame (central differences, h = voxel)."""
        h = self.spacing[obj][:, None]  # (E,1,3)
        d = self.sample(obj, pts)
        grad = []
        for k in range(3):
            e = torch.zeros_like(pts)
            e[..., k] = h[..., k]
            grad.append((self.sample(obj, pts + e) - self.sample(obj, pts - e)) / (2 * h[..., k]))
        n = torch.stack(grad, -1)
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return d, n


def world_to_object(pts_w: torch.Tensor, obj_pos_w: torch.Tensor, obj_quat_xyzw: torch.Tensor) -> torch.Tensor:
    """pts_w (E,P,3) -> object frame, given object pose (E,3), (E,4) xyzw."""
    E, P, _ = pts_w.shape
    rel = pts_w - obj_pos_w[:, None]
    q = obj_quat_xyzw[:, None].expand(E, P, 4).reshape(-1, 4)
    return quat_rotate_inverse(q, rel.reshape(-1, 3), w_last=True).view(E, P, 3)


def object_to_world_dir(v_obj: torch.Tensor, obj_quat_xyzw: torch.Tensor) -> torch.Tensor:
    E, P, _ = v_obj.shape
    q = obj_quat_xyzw[:, None].expand(E, P, 4).reshape(-1, 4)
    return quat_rotate(q, v_obj.reshape(-1, 3), w_last=True).view(E, P, 3)


class HandObjectGeometry:
    def __init__(self, env, sdf_dir: Path):
        self.env = env
        self.device = env.device
        self.sdf = ObjectSDF(sdf_dir, env.device)
        self.mc = env.command_manager.get_state("motion_command")
        n_obj = len(self.sdf.names)
        assert n_obj == self.mc.motion.num_motions, (n_obj, self.mc.motion.num_motions)
        self.env_obj = torch.arange(env.num_envs, device=env.device) % n_obj
        body_names = list(env.simulator.body_names)
        self.kp_idx = torch.tensor([body_names.index(b) for b in KEYPOINT_BODIES], device=env.device)
        self.tip_idx = torch.tensor([body_names.index(b) for b in TIP_BODIES], device=env.device)
        self.wrist_idx = torch.tensor([body_names.index(WRIST_LINK["l"]), body_names.index(WRIST_LINK["r"])], device=env.device)
        self.tip_in_kp = torch.tensor(TIP_IDX_IN_KEYPOINTS, device=env.device)
        self.ref_dist = self._reference_distances()  # (T_total, 42) unsigned, object-centric
        self._step_cache: dict = {}
        self._cache_key = None
        print(f"[HandObjectGeometry] {n_obj} SDF grids ({self.sdf.res}^3), {len(KEYPOINT_BODIES)} hand keypoints, "
              f"reference distances for {self.ref_dist.shape[0]} frames: tips in contact (<1.5 cm) "
              f"{(self.ref_dist[:, self.tip_in_kp] < 0.015).float().mean():.1%} of frame-tips")

    # ------------------------------------------------------------------ reference
    def _reference_distances(self) -> torch.Tensor:
        m = self.mc.motion
        # raw buffer = every body of the npz files (npz order); body names read from the first file
        from hiphi_wbt.presets import MOTION_DIR_ACTIVE

        first = sorted(Path(MOTION_DIR_ACTIVE).glob("*.npz"))[0]
        all_body_names = [str(b) for b in np.load(first)["body_names"]]
        raw = m._body_pos_w
        idx = torch.tensor([all_body_names.index(b) for b in KEYPOINT_BODIES], device=self.device)
        T = raw.shape[0]
        # object index per frame = motion index of that frame
        frame_motion = torch.searchsorted(m.motion_end_idx, torch.arange(T, device=self.device), right=True)
        out = torch.empty(T, len(KEYPOINT_BODIES), device=self.device)
        obj_pos = m.object_pos_w
        obj_quat = m.object_quat_w  # xyzw
        for s in range(0, T, 8192):
            e = min(T, s + 8192)
            pts = world_to_object(raw[s:e][:, idx], obj_pos[s:e], obj_quat[s:e])
            out[s:e] = self.sdf.sample(frame_motion[s:e], pts).abs()
        return out

    # ------------------------------------------------------------------ per-step (cached per sim step)
    def current(self) -> dict:
        key = (int(getattr(self.env.simulator, "_sim_step_counter", 0)), int(self.env.episode_length_buf.sum().item()))
        if key == self._cache_key:
            return self._step_cache
        sim = self.env.simulator
        obj_pos = self.mc.simulator_object_pos_w
        obj_quat = self.mc.simulator_object_quat_w  # xyzw
        kp_w = sim._rigid_body_pos[:, self.kp_idx]  # (E,42,3)
        pts = world_to_object(kp_w, obj_pos, obj_quat)
        d, n_obj = self.sdf.sample_with_normal(self.env_obj, pts)
        n_w = object_to_world_dir(n_obj, obj_quat)
        tip_force = sim.contact_forces[:, self.tip_idx].norm(dim=-1)  # (E,10)
        self._step_cache = {"dist": d, "normal_w": n_w, "tip_force": tip_force, "ref_dist": self.ref_dist[self.mc.time_steps]}
        self._cache_key = key
        return self._step_cache


def get_geometry(env) -> HandObjectGeometry:
    geo = getattr(env, "_hiphi_geo", None)
    if geo is None:
        from hiphi_wbt.presets import SDF_DIR  # noqa: WPS433 (lazy: presets import this module)

        geo = env._hiphi_geo = HandObjectGeometry(env, SDF_DIR)
    return geo
