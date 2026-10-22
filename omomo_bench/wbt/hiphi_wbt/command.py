"""MotionCommand with a fixed environment -> motion assignment (env i tracks motion i % N) and hand tracking metrics.

With one object asset per environment (sim_multiasset.IsaacSim, env i <- asset i % N) the motion of an environment
must stay fixed so that the simulated object is the one the motion was retargeted with. Time steps within the motion are
sampled uniformly (plus ``start_at_timestep_zero_prob``); evaluation starts every environment at its motion's first frame.
"""
from __future__ import annotations

import os
import re

import torch

# diagnostic: displace the object at every reset (metres, world xyz), e.g. HIPHI_WBT_OBJECT_OFFSET=3,0,0 puts it out of reach
_OBJ_OFFSET = [float(v) for v in os.environ.get("HIPHI_WBT_OBJECT_OFFSET", "0,0,0").split(",")]
_ROOT_Z_OFFSET = float(os.environ.get("HIPHI_WBT_ROOT_Z_OFFSET", "0"))  # diagnostic: lift the robot at reset
_ZERO_INIT_VEL = os.environ.get("HIPHI_WBT_ZERO_INIT_VEL", "0") == "1"  # diagnostic: reset with zero velocities

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_angle_axis, quat_conjugate, quat_error_magnitude, quat_from_euler_xyz, quat_mul


class FixedMotionCommand(MotionCommand):
    # bodies / joints for the extra metrics (must be in body_names_to_track / dof_names)
    METRIC_BODY_GROUPS = {
        "fingertip": [f"{c}_{t}" for c in ("l", "r") for t in ("thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip")],
        "wrist": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    }
    FINGER_JOINT_RE = r"^[lr]_"

    def setup(self) -> None:
        super().setup()
        tracked = list(self.motion_cfg.body_names_to_track)
        self._metric_idx = {k: torch.tensor([tracked.index(b) for b in v if b in tracked], dtype=torch.long, device=self.device)
                            for k, v in self.METRIC_BODY_GROUPS.items()}
        names = list(self._env.simulator.dof_names)
        self._finger_mask = torch.tensor([bool(re.match(self.FINGER_JOINT_RE, n)) for n in names], device=self.device)
        # optional object-assist curriculum (ResMimic / DexMachina style virtual controller on the object)
        self.assist = self.cfg.params.get("object_assist")  # {"kp","kd","kp_rot","kd_rot","len_full","len_zero"} or None
        if os.environ.get("HIPHI_WBT_NO_ASSIST", "0") == "1":  # evaluation: never push the object with the virtual controller
            self.assist = None
        self._assist_scale = 1.0
        if self.assist:
            print(f"[FixedMotionCommand] object assist enabled: {self.assist}")
        # per-motion adaptive start-time sampling: holosoma's global failure-bin sampler, restricted to each env's own clip
        self.adaptive = bool(self.cfg.params.get("adaptive_per_motion", False))
        if self.adaptive:
            from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler

            self._sampler = AdaptiveTimestepsSampler(self.motion.time_step_total, self.device, int(1 / self._env.dt))
            nb = self._sampler.num_bins
            # bin index -> global frame range; bins of motion m: those whose centre frame lies in [start_m, end_m)
            centres = (torch.arange(nb, device=self.device).float() + 0.5) / nb * self.motion.time_step_total
            self._bin_motion = torch.searchsorted(self.motion.motion_end_idx, centres.long(), right=True).clamp(0, self.motion.num_motions - 1)
            print(f"[FixedMotionCommand] per-motion adaptive start sampling: {nb} bins")
        n_env, n_mot = self.num_envs, self.motion.num_motions
        print(f"[FixedMotionCommand] {n_mot} motions, {n_env} envs -> {n_env // n_mot} full rounds"
              + (f" + {n_env % n_mot} extra envs" if n_env % n_mot else "") + "; env i tracks motion i % N")

    def reset(self, env_ids: torch.Tensor | None) -> None:
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        num_motions = self.motion.num_motions

        # fixed assignment: env i <- motion i % N  (matches the object asset assignment of sim_multiasset.IsaacSim)
        motion_ids = env_ids % num_motions
        self.motion_ids[env_ids] = motion_ids
        start_idx = self.motion.motion_start_idx[motion_ids]
        end_idx = self.motion.motion_end_idx[motion_ids]
        motion_len = end_idx - start_idx
        if self._env.is_evaluating:
            phase = torch.full((n,), float(os.environ.get("HIPHI_WBT_EVAL_PHASE", "0")), device=self.device)  # eval start fraction
            self.time_steps[env_ids] = start_idx + (phase * (motion_len - 1).float()).long()
        elif self.adaptive:
            terminated = self._env.termination_manager.terminated[env_ids]
            if torch.any(terminated):
                self._sampler.update_current_bin_failed_count(self.time_steps[env_ids][terminated])
            probs = self._sampler.sampling_probabilities  # (nb,)
            masked = probs[None, :] * (self._bin_motion[None, :] == motion_ids[:, None]).float()  # (n, nb)
            masked = masked + 1e-12 * (self._bin_motion[None, :] == motion_ids[:, None]).float()
            bins = torch.multinomial(masked, 1).squeeze(1)
            g = ((bins.float() + torch.rand(n, device=self.device)) / self._sampler.num_bins * self.motion.time_step_total).long()
            self.time_steps[env_ids] = g.clamp(start_idx, end_idx - 1)
        else:
            phase = torch.rand(n, device=self.device)
            self.time_steps[env_ids] = start_idx + (phase * (motion_len - 1).float()).long()
        prob = self.motion_cfg.start_at_timestep_zero_prob
        if prob >= 1.0:
            self.time_steps[env_ids] = start_idx
        elif prob > 0.0:
            subset = self.time_steps[env_ids]
            self.time_steps[env_ids] = torch.where(torch.rand_like(subset, dtype=torch.float32) < prob, start_idx, subset)
        last = self.time_steps[env_ids] >= end_idx - 1
        self.time_steps[env_ids] = torch.where(last, end_idx - 2, self.time_steps[env_ids])

        # --- the rest is holosoma's MotionCommand.reset (initial state from the motion + noise) ---
        root_pos = self.root_pos_w[env_ids].clone()
        root_rot = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()
        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()

        cfg = self.init_pose_cfg
        k = cfg.overall_noise_scale
        dof_pos_noise = cfg.dof_pos * k
        root_pos_noise = torch.tensor(cfg.root_pos, device=self.device) * k
        root_rot_noise_rpy = torch.tensor(cfg.root_rot, device=self.device) * k
        root_vel_noise = torch.tensor(cfg.root_lin_vel, device=self.device) * k
        root_ang_vel_noise_rpy = torch.tensor(cfg.root_ang_vel, device=self.device) * k

        target_dof_pos = dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        lim = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]
        target_dof_pos = torch.clip(target_dof_pos, lim[:, 0], lim[:, 1])
        target_root_pos = root_pos + (torch.rand(root_pos.shape, device=self.device) - 0.5) * 2 * root_pos_noise.unsqueeze(0)
        rpy = (torch.rand((n, 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        delta = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        target_root_rot = quat_mul(delta, root_rot, w_last=True)
        target_root_lin_vel = root_lin_vel + (torch.rand(root_lin_vel.shape, device=self.device) - 0.5) * 2 * root_vel_noise.unsqueeze(0)
        target_root_ang_vel = root_ang_vel + (torch.rand(root_ang_vel.shape, device=self.device) - 0.5) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)

        sim = self._env.simulator
        if _ROOT_Z_OFFSET:
            target_root_pos = target_root_pos + torch.tensor([0.0, 0.0, _ROOT_Z_OFFSET], device=self.device)
        if _ZERO_INIT_VEL:
            dof_vel = torch.zeros_like(dof_vel); target_root_lin_vel = torch.zeros_like(target_root_lin_vel); target_root_ang_vel = torch.zeros_like(target_root_ang_vel)
        sim.dof_pos[env_ids] = target_dof_pos
        sim.dof_vel[env_ids] = dof_vel
        sim.robot_root_states[env_ids, :3] = target_root_pos
        sim.robot_root_states[env_ids, 3:7] = target_root_rot
        sim.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        sim.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        if self.motion.has_object:
            obj_pos = self.object_pos_w[env_ids]
            obj_ori = self.object_quat_w[env_ids]
            obj_lin_vel = self.object_lin_vel_w[env_ids]
            obj_pos_noise = torch.tensor([cfg.object_pos], device=self.device) * k
            target_obj_pos = obj_pos + (torch.rand(obj_pos.shape, device=self.device) - 0.5) * 2 * obj_pos_noise
            if any(_OBJ_OFFSET):
                target_obj_pos = target_obj_pos + torch.tensor(_OBJ_OFFSET, device=self.device)
            states = torch.cat([target_obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1)
            sim.set_actor_states([self.object_name], env_ids, states)

    # ------------------------------------------------------------------ object assist
    def step(self) -> None:
        super().step()
        if self.adaptive and not self._env.is_evaluating:
            self._sampler.update_bin_failed_count()
        if self.assist and self.motion.has_object:
            self._apply_object_assist()

    def _apply_object_assist(self) -> None:
        """Virtual PD controller pulling the object towards its reference pose; gains scale with a curriculum signal
        (average episode length): full gains below len_full control steps, zero above len_zero."""
        a = self.assist
        env = self._env
        try:
            avg_len = float(env._get_average_episode_tracker().get_average())
        except Exception:  # noqa: BLE001
            avg_len = 0.0
        scale = float(min(1.0, max(0.0, (a["len_zero"] - avg_len) / max(1e-6, a["len_zero"] - a["len_full"]))))
        self._assist_scale = scale
        obj = env.simulator.scene.rigid_objects[self.object_name]
        if scale <= 0.0:
            obj.set_external_force_and_torque(torch.zeros(0, 3, device=self.device), torch.zeros(0, 3, device=self.device))
            return
        states = self._simulator_object_states()  # (E,13): pos, quat xyzw, lin vel, ang vel
        pos, quat, lin_vel, ang_vel = states[:, :3], states[:, 3:7], states[:, 7:10], states[:, 10:13]
        mass = obj.data.default_mass.to(self.device).view(-1, 1)  # (E,1)
        force = mass * (a["kp"] * (self.object_pos_w - pos) + a["kd"] * (self.object_lin_vel_w - lin_vel))
        q_err = quat_mul(self.object_quat_w, quat_conjugate(quat, w_last=True), w_last=True)
        angle, axis = quat_angle_axis(q_err, w_last=True)
        rot_vec = angle.unsqueeze(-1) * axis  # world-frame rotation from current to reference
        inertia = (mass * 0.02)  # crude: m r^2 with r ~ 14 cm, only sets the torque scale
        torque = inertia * (a["kp_rot"] * rot_vec - a["kd_rot"] * ang_vel)
        force = (scale * force).clamp(-a.get("max_force", 200.0), a.get("max_force", 200.0))
        torque = (scale * torque).clamp(-a.get("max_torque", 20.0), a.get("max_torque", 20.0))
        obj.set_external_force_and_torque(force.view(-1, 1, 3), torque.view(-1, 1, 3), is_global=True)
        self.metrics["assist/scale"] = torch.full((self.num_envs,), scale, device=self.device)
        self.metrics["assist/force"] = force.norm(dim=-1)

    def update_metrics(self):
        super().update_metrics()
        for name, idx in self._metric_idx.items():
            if idx.numel() == 0:
                continue
            d = torch.norm(self.body_pos_relative_w[:, idx] - self.robot_body_pos_w[:, idx], dim=-1)
            self.metrics[f"motion/error_{name}_pos"] = d.mean(-1)
            self.metrics[f"motion/error_{name}_pos_max"] = d.max(-1).values
        if self._finger_mask.any():
            self.metrics["motion/error_finger_joint_pos"] = (self.joint_pos[:, self._finger_mask] - self.robot_joint_pos[:, self._finger_mask]).abs().mean(-1)
        # which bad-tracking condition is active (fractions logged per iteration -> what terminates the episodes)
        z_err = torch.abs(self.ref_pos_w[:, -1] - self.robot_ref_pos_w[:, -1])
        self.metrics["term/bad_ref_z"] = (z_err > 0.5).float()
        self.metrics["term/bad_ref_ori"] = (quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w) > 0.8).float()
        tracked = list(self.motion_cfg.body_names_to_track)
        if not hasattr(self, "_term_body_idx"):
            names = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
            self._term_body_idx = torch.tensor([tracked.index(b) for b in names], device=self.device)
        d = torch.norm(self.body_pos_relative_w[:, self._term_body_idx] - self.robot_body_pos_w[:, self._term_body_idx], dim=-1)
        self.metrics["term/bad_ankle"] = (d[:, :2] > 0.25).any(-1).float()
        self.metrics["term/bad_wrist"] = (d[:, 2:] > 0.25).any(-1).float()
        if self.motion.has_object:
            self.metrics["term/bad_object_pos"] = (torch.norm(self.object_pos_w - self.simulator_object_pos_w, dim=-1) > 0.3).float()
            self.metrics["term/bad_object_ori"] = (quat_error_magnitude(self.object_quat_w, self.simulator_object_quat_w) > 1.0).float()
        if self.adaptive:
            self._sampler.get_stats()
            self.metrics["motion/adaptive_entropy"] = self._sampler.metrics["sampling_entropy"]
            self.metrics["motion/adaptive_top1_prob"] = self._sampler.metrics["sampling_top1_prob"]
        err = getattr(self._env, "_hiphi_tip_dist_err", None)
        if err is not None:
            self.metrics["motion/error_tip_obj_dist"] = err
        if self.motion.has_object:
            self.metrics["motion/error_object_pos"] = torch.norm(self.object_pos_w - self.simulator_object_pos_w, dim=-1)
            self.metrics["motion/error_object_rot"] = quat_error_magnitude(self.object_quat_w, self.simulator_object_quat_w)
