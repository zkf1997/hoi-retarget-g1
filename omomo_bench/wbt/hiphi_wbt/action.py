"""Joint-position action term with reference-residual targets for a joint subset (the fingers).

For joints matching ``reference_joint_pattern`` the PD target is  q_ref(t) + action_scale * a  (q_ref = current reference
pose from the motion command) instead of  q_default + action_scale * a.  A zero action therefore reproduces the retargeted
finger pose exactly; the policy only learns corrections (closing a little more when the object is not quite grasped, etc.).
Set ``reference_joint_pattern`` to ".*" to use reference-residual actions for every joint.
"""
from __future__ import annotations

import re
from typing import Any

import torch

from holosoma.config_types.action import ActionTermCfg
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm


class ResidualReferenceJointPositionActionTerm(JointPositionActionTerm):
    def __init__(self, cfg: ActionTermCfg, env: Any):
        super().__init__(cfg, env)
        pattern = cfg.params.get("reference_joint_pattern", r"^[lr]_")
        names = list(env.dof_names)
        self._ref_mask = torch.tensor([bool(re.match(pattern, n)) for n in names], device=env.device)
        self._zero_ref_actions = bool(cfg.params.get("zero_reference_actions", False))  # ablation: pure reference playback
        # residual range of the reference-tracked joints: holosoma's action scale (0.25 * effort / kp = 0.25 rad for the fingers)
        # times this factor. Random exploration on 40 finger joints with the full 0.25 rad range stops the body from learning
        # (2026-10-13 diagnostics), so the default shrinks the finger residual to +-0.05 rad per unit action.
        self._ref_scale = float(cfg.params.get("reference_action_scale", 0.2))
        finger_mask = torch.tensor([bool(re.match(r"^[lr]_", n)) for n in names], device=env.device)
        self.action_scales[finger_mask] *= self._ref_scale  # in place: env.action_scales aliases this tensor; fingers only
        self._motion_command = None
        print(f"[ResidualReferenceJointPositionActionTerm] {int(self._ref_mask.sum())}/{len(names)} joints track the reference pose as a residual"
              f" (scale x{self._ref_scale}{', actions ignored' if self._zero_ref_actions else ''})")

    def _base_target(self) -> torch.Tensor:
        if self._motion_command is None:
            self._motion_command = self.env.command_manager.get_state("motion_command")
        ref = self._motion_command.joint_pos  # (num_envs, num_dof) reference joint positions at the current time step
        return torch.where(self._ref_mask[None, :], ref, self.env.default_dof_pos)

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        actions_scaled = actions * self.action_scales
        if self._zero_ref_actions:
            actions_scaled = torch.where(self._ref_mask[None, :], torch.zeros_like(actions_scaled), actions_scaled)  # all ref-tracked joints
        control_type = self.env.robot_config.control.control_type
        assert control_type == "P", "ResidualReferenceJointPositionActionTerm supports position control only"
        torques = (
            self._kp_scale * self.p_gains * (actions_scaled + self._base_target() - self.env.simulator.dof_pos)
            - self._kd_scale * self.d_gains * self.env.simulator.dof_vel
        )
        if self._randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques) * 2.0 - 1.0) * self._rfi_lim * self._rfi_lim_scale * self.env.torque_limits
        if self.env.robot_config.control.clip_torques:
            torques = torch.clip(torques, -self.env.torque_limits, self.env.torque_limits)
        return torques


# ---------------------------------------------------------------------------------------------------------------------
# Finger synergies: the policy outputs 29 body actions + 6 scalars per hand (thumb flexion, thumb abduction, index, middle,
# ring, pinky flexion); each scalar is broadcast to the flexion joints of that finger as a residual on the reference pose.
# Finger abduction joints of the four fingers follow the reference exactly. 69 -> 41 action dims: the 40 raw finger dims
# inflated PPO's KL / entropy and throttled the adaptive learning rate (2026-10-14 bisection, runs AA vs AB).
SYNERGIES = {
    "thumb_flex": ["thumb_cmc_flex", "thumb_mcp", "thumb_ip"],
    "thumb_abd": ["thumb_cmc_abd"],
    "index": ["index_finger_mcp_flex", "index_finger_pip", "index_finger_dip"],
    "middle": ["middle_finger_mcp_flex", "middle_finger_pip", "middle_finger_dip"],
    "ring": ["ring_finger_mcp_flex", "ring_finger_pip", "ring_finger_dip"],
    "pinky": ["pinky_mcp_flex", "pinky_pip", "pinky_dip"],
}


def active_synergies() -> dict:
    # HIPHI_WBT_SYNERGY=body -> no finger actions at all (fingers = reference playback, 29-dim policy)
    import os

    return {} if os.environ.get("HIPHI_WBT_SYNERGY", "0") == "body" else SYNERGIES


def synergy_action_names(dof_names: list[str]) -> list[str]:
    body = [n for n in dof_names if not re.match(r"^[lr]_", n)]
    return body + [f"{side}_{g}" for side in ("l", "r") for g in active_synergies()]


class SynergyResidualActionTerm(ResidualReferenceJointPositionActionTerm):
    def __init__(self, cfg: ActionTermCfg, env: Any):
        super().__init__(cfg, env)
        names = list(env.dof_names)
        self._syn_names = synergy_action_names(names)
        n_act, n_dof = len(self._syn_names), len(names)
        M = torch.zeros(n_act, n_dof, device=env.device)
        for i, n in enumerate(names):
            if not re.match(r"^[lr]_", n):
                M[self._syn_names.index(n), i] = 1.0
            else:
                side, rest = n[0], n[2:]
                for g, joints in active_synergies().items():
                    if rest in joints:
                        M[self._syn_names.index(f"{side}_{g}"), i] = 1.0
        self._expand = M  # (n_act, n_dof)
        env._action_names = self._syn_names  # for reward terms that mask actions by name
        print(f"[SynergyResidualActionTerm] {n_act} policy actions -> {n_dof} joint targets; finger abduction follows the reference")

    @property
    def action_dim(self) -> int:
        return self._expand.shape[0]

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions @ self._expand)
