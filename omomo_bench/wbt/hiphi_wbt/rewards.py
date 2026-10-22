"""Hand-focused tracking rewards for the whole-body tracking task.

All terms read the MotionCommand state ("motion_command"); body subsets are given by name and must be part of
``motion_config.body_names_to_track``.
"""
from __future__ import annotations

import re

import torch

from holosoma.managers.reward.terms.wbt import _get_motion_command_and_assert_type


def _subset_idx(motion_command, body_names: list[str]) -> torch.Tensor:
    tracked = list(motion_command.motion_cfg.body_names_to_track)
    idx = [tracked.index(b) for b in body_names]
    return torch.tensor(idx, dtype=torch.long, device=motion_command.device)


def body_subset_position_error_exp(env, sigma: float, body_names: list[str]) -> torch.Tensor:
    """exp(-mean_k ||p_ref_k - p_robot_k||^2 / sigma^2) over a subset of the tracked bodies (relative frame, like holosoma)."""
    mc = _get_motion_command_and_assert_type(env)
    cache = getattr(mc, "_subset_cache", None)
    if cache is None:
        cache = mc._subset_cache = {}
    key = tuple(body_names)
    if key not in cache:
        cache[key] = _subset_idx(mc, body_names)
    idx = cache[key]
    err = torch.sum(torch.square(mc.body_pos_relative_w[:, idx] - mc.robot_body_pos_w[:, idx]), dim=-1)
    return torch.exp(-err.mean(-1) / sigma**2)


def joint_subset_position_error_exp(env, sigma: float, joint_pattern: str) -> torch.Tensor:
    """exp(-mean_j (q_ref_j - q_j)^2 / sigma^2) over joints whose name matches ``joint_pattern``."""
    mc = _get_motion_command_and_assert_type(env)
    cache = getattr(mc, "_joint_mask_cache", None)
    if cache is None:
        cache = mc._joint_mask_cache = {}
    if joint_pattern not in cache:
        names = list(env.simulator.dof_names)
        cache[joint_pattern] = torch.tensor([bool(re.match(joint_pattern, n)) for n in names], device=mc.device)
    mask = cache[joint_pattern]
    if not bool(mask.any()):  # e.g. frozen-hand diagnostic: no matching joints
        return torch.ones(env.num_envs, device=mc.device)
    err = torch.square(mc.joint_pos[:, mask] - env.simulator.dof_pos[:, mask]).mean(-1)
    return torch.exp(-err / sigma**2)


# ---------------------------------------------------------------- contact-aware terms (need the SDF grids, see geometry.py)
def fingertip_contact_excess_exp(env, sigma: float = 0.02, ref_max: float = 0.05) -> torch.Tensor:
    """FlashDexRetarget-style interaction reward: for each fingertip, penalise only the EXCESS of the simulated signed
    distance to the object surface over the demonstrated (unsigned) distance,  r = mean_f exp(-max(0, d_f - d_f^ref)/sigma).
    Fingertips whose reference distance is above ``ref_max`` (hand clearly away from the object) get reward 1."""
    from hiphi_wbt.geometry import get_geometry

    geo = get_geometry(env)
    g = geo.current()
    d = g["dist"][:, geo.tip_in_kp]
    d_ref = g["ref_dist"][:, geo.tip_in_kp]
    excess = torch.relu(d - d_ref)
    r = torch.exp(-excess / sigma)
    r = torch.where(d_ref > ref_max, torch.ones_like(r), r)
    env._hiphi_tip_dist_err = (d - d_ref).abs().mean(-1)  # metric: |d - d_ref| over fingertips
    return r.mean(-1)


def hand_penetration_penalty(env, margin: float = 0.01) -> torch.Tensor:
    """Depth of hand keypoints inside the object beyond ``margin`` (sum over keypoints, metres) -> use a negative weight."""
    from hiphi_wbt.geometry import get_geometry

    g = get_geometry(env).current()
    return torch.relu(-g["dist"] - margin).sum(-1)


# ---------------------------------------------------------------- action-rate penalty restricted to a joint subset
def penalty_action_rate_subset(env, joint_pattern: str = r"^(?![lr]_).*", **_) -> torch.Tensor:
    """sum_j (a_t - a_{t-1})^2 over joints whose name matches ``joint_pattern`` (default: every non-finger joint).

    holosoma's penalty sums over ALL action dims: with 40 finger dims and unit exploration noise it is ~-0.28 per step and
    makes the expected per-step reward negative, so the policy learns to end episodes early (2026-10-13 diagnosis)."""
    cache = getattr(env, "_action_rate_masks", None)
    if cache is None:
        cache = env._action_rate_masks = {}
    if joint_pattern not in cache:
        names = list(getattr(env, "_action_names", None) or env.simulator.dof_names)  # synergy actions are named l_<group>
        cache[joint_pattern] = torch.tensor([bool(re.match(joint_pattern, n)) for n in names], device=env.device)
    mask = cache[joint_pattern]
    diff = env.action_manager.action[:, mask] - env.action_manager.prev_action[:, mask]
    return torch.sum(torch.square(diff), dim=1)
