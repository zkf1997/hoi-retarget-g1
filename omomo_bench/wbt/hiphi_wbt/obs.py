"""Extra observation terms: hand-object geometry, hand-local SDF samples (BPS-style), reference object pose, future reference.

All outputs are per-env flat vectors; frames: the robot's reference-body (torso) frame as in holosoma's obj_pos_b.
"""
from __future__ import annotations

import torch

from holosoma.managers.observation.terms.wbt import _get_motion_command_and_assert_type
from holosoma.utils.rotations import quat_rotate, quat_rotate_inverse, quaternion_to_matrix, subtract_frame_transforms

from hiphi_wbt.geometry import get_geometry


def _to_ref_frame_dir(env, mc, v_w: torch.Tensor) -> torch.Tensor:
    """Rotate world direction vectors (E,P,3) into the robot reference-body frame."""
    E, P, _ = v_w.shape
    q = mc.robot_ref_quat_w[:, None].expand(E, P, 4).reshape(-1, 4)
    return quat_rotate_inverse(q, v_w.reshape(-1, 3), w_last=True).view(E, P, 3)


def hand_object_geometry(env, sigma_max: float = 0.1, force_threshold: float = 1.0) -> torch.Tensor:
    """Per hand keypoint (42): clipped signed distance to the object surface and the surface normal (torso frame);
    per fingertip (10): binary contact flag from the contact sensor.  -> 42 + 126 + 10 = 178 dims."""
    mc = _get_motion_command_and_assert_type(env)
    g = get_geometry(env).current()
    d = g["dist"].clamp(-sigma_max, sigma_max)
    n = _to_ref_frame_dir(env, mc, g["normal_w"])
    contact = (g["tip_force"] > force_threshold).float()
    return torch.cat([d, n.reshape(env.num_envs, -1), contact], dim=1)


def hand_local_sdf(env, half: float = 0.08, n: int = 4, sigma_max: float = 0.15) -> torch.Tensor:
    """BPS-style local geometry: SDF sampled on an n^3 grid (±half m) attached to each wrist link. -> 2*n^3 dims."""
    geo = get_geometry(env)
    mc = geo.mc
    sim = env.simulator
    E = env.num_envs
    ax = torch.linspace(-half, half, n, device=env.device)
    grid = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)  # (n^3,3)
    P = grid.shape[0]
    wrist_pos = sim._rigid_body_pos[:, geo.wrist_idx]  # (E,2,3)
    wrist_quat = sim._rigid_body_rot[:, geo.wrist_idx]  # (E,2,4) xyzw
    pts_w = []
    for h in range(2):
        q = wrist_quat[:, h, None].expand(E, P, 4).reshape(-1, 4)
        pts_w.append(wrist_pos[:, h, None] + quat_rotate(q, grid[None].expand(E, P, 3).reshape(-1, 3), w_last=True).view(E, P, 3))
    pts_w = torch.cat(pts_w, 1)  # (E,2P,3)
    from hiphi_wbt.geometry import world_to_object

    pts_o = world_to_object(pts_w, mc.simulator_object_pos_w, mc.simulator_object_quat_w)
    d = geo.sdf.sample(geo.env_obj, pts_o).clamp(-sigma_max, sigma_max)
    return d


def ref_obj_pos_b(env) -> torch.Tensor:
    """Reference object position in the robot reference-body frame (the actor otherwise never sees the object target)."""
    mc = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(mc.robot_ref_pos_w, mc.robot_ref_quat_w, mc.object_pos_w, mc.object_quat_w)
    return pos.view(env.num_envs, -1)


def ref_obj_ori_b(env) -> torch.Tensor:
    mc = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(mc.robot_ref_pos_w, mc.robot_ref_quat_w, mc.object_pos_w, mc.object_quat_w)
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def ref_future(env, offsets: tuple[int, ...] = (5, 10)) -> torch.Tensor:
    """For each future offset k (control steps): reference joint positions at t+k minus the current robot joint positions,
    and the reference object position at t+k in the robot reference frame.  -> len(offsets) * (num_dof + 3) dims."""
    mc = _get_motion_command_and_assert_type(env)
    m = mc.motion
    end = m.motion_end_idx[mc.motion_ids] - 1
    out = []
    for k in offsets:
        t = torch.minimum(mc.time_steps + k, end)
        dq = m.joint_pos[t] - env.simulator.dof_pos
        obj_pos_w = m.object_pos_w[t] + env.simulator.scene.env_origins
        pos, _ = subtract_frame_transforms(mc.robot_ref_pos_w, mc.robot_ref_quat_w, obj_pos_w, m.object_quat_w[t])
        out += [dq, pos]
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------- body-only proprioception (finger joints excluded)
def _body_mask(env):
    m = getattr(env, "_body_dof_mask", None)
    if m is None:
        import re as _re

        m = env._body_dof_mask = torch.tensor([not _re.match(r"^[lr]_", n) for n in env.simulator.dof_names], device=env.device)
    return m


def body_dof_pos(env) -> torch.Tensor:
    return (env.simulator.dof_pos - env.default_dof_pos)[:, _body_mask(env)]


def body_dof_vel(env) -> torch.Tensor:
    return env.simulator.dof_vel[:, _body_mask(env)]


def body_actions(env) -> torch.Tensor:
    a = env.action_manager.action
    names = getattr(env, "_action_names", None)
    if names is None:  # one action per DoF
        return a[:, _body_mask(env)]
    import re as _re

    m = getattr(env, "_body_action_mask", None)
    if m is None:
        m = env._body_action_mask = torch.tensor([not _re.match(r"^[lr]_", n) for n in names], device=env.device)
    return a[:, m]


def finger_dof_pos(env) -> torch.Tensor:
    """Finger joints relative to the REFERENCE finger pose (what the residual acts on), instead of relative to the default pose."""
    mc = _get_motion_command_and_assert_type(env)
    m = ~_body_mask(env)
    return (env.simulator.dof_pos - mc.joint_pos)[:, m]
