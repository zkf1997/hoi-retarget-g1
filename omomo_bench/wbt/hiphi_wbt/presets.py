"""Experiment presets: G1 + Wuji Hand 2 tracking the 60 HiPHI human-object interaction retargets (one object per sequence).

Registered into holosoma's registries when this file is passed with ``--import-file``; the module path ``hiphi_wbt.*`` must
be importable (PYTHONPATH=omomo_bench/wbt).

    exp:g1-wuji-hiphi-wbt            policy controls all 69 joints; finger targets are residuals on the reference finger pose
    exp:g1-wuji-hiphi-wbt-fixhand    same, but fingers are pure reference playback (no finger actions: residual scale 0)
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig
from holosoma.config_types.experiment import TrainingConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.randomization import RandomizationTermCfg
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_types.scene import RigidObjectConfig, SceneConfig
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values import simulator as _sim
from holosoma.config_values.experiment import EXPERIMENT_REGISTRY, g1_29dof_wbt
from holosoma.config_values.robot import ROBOT_REGISTRY
from holosoma.config_values.wbt.g1 import observation as _obs
from holosoma.config_values.wbt.g1 import randomization as _rand
from holosoma.config_values.wbt.g1 import reward as _rew

from hiphi_wbt.robot import FINGERTIP_BODIES, g1_wuji

WBT_DIR = Path(__file__).resolve().parent.parent  # omomo_bench/wbt
CLIPS = os.environ.get("HIPHI_WBT_CLIPS", "")  # e.g. "w5": 5 s interaction clips from make_clips.py (data/motions_w5)
MOTION_DIR = WBT_DIR / "data" / (f"motions_{CLIPS}" if CLIPS else "motions")
OBJECT_DIR = WBT_DIR / "assets" / "objects"
SDF_DIR_ALL = WBT_DIR / "data" / "sdf"


def _subset_dirs(tags: list[str]) -> tuple[Path, Path, Path]:
    """Symlink the selected motions / objects / SDF grids into data/subsets/<name>/ (sorted order is preserved)."""
    name = "+".join(t[:3] for t in tags) if len(tags) <= 6 else f"{len(tags)}seqs"
    root = WBT_DIR / "data" / "subsets" / (name + (f"_{CLIPS}" if CLIPS else ""))
    for sub, src, ext in (("motions", MOTION_DIR, ".npz"), ("objects", OBJECT_DIR, ""), ("sdf", SDF_DIR_ALL, ".npz")):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():
            old.unlink()
        for t in tags:  # objects are directories <tag>/hiphi_object.urdf (symlink the directory)
            assert (src / f"{t}{ext}").exists(), src / f"{t}{ext}"
            (d / f"{t}{ext}").symlink_to(src / f"{t}{ext}")
    return root / "motions", root / "objects", root / "sdf"


_MOTIONS = [t for t in os.environ.get("HIPHI_WBT_MOTIONS", "").split(",") if t]
if _MOTIONS:  # fast iteration on a subset of sequences
    MOTION_DIR_ACTIVE, OBJECT_DIR_ACTIVE, SDF_DIR = _subset_dirs(_MOTIONS)
    print(f"[hiphi_wbt] subset of {len(_MOTIONS)} motions: {_MOTIONS}")
else:
    MOTION_DIR_ACTIVE, OBJECT_DIR_ACTIVE, SDF_DIR = MOTION_DIR, OBJECT_DIR, SDF_DIR_ALL

# variant flags: HIPHI_WBT_VARIANT=geo,contact,future,bps,pen
VARIANT = {v for v in os.environ.get("HIPHI_WBT_VARIANT", "").split(",") if v}
_known = {"geo", "contact", "future", "bps", "pen", "vobj"}
assert VARIANT <= _known, f"unknown variant flags {VARIANT - _known}"
RUN_SUFFIX = ("_" + "-".join(sorted(VARIANT)) if VARIANT else "") + ("_frozenhand" if os.environ.get("HIPHI_WBT_FROZEN_HAND", "0") == "1" else "") + (f"_{os.environ['HIPHI_WBT_URDF']}" if os.environ.get("HIPHI_WBT_URDF") else "") + ("_fixhand" if os.environ.get("HIPHI_WBT_FIXHAND", "0") == "1" else "") + ("_noadapt" if os.environ.get("HIPHI_WBT_ADAPTIVE", "1") != "1" else "") + (f"_armkp{os.environ['HIPHI_WBT_ARM_KP']}" if os.environ.get("HIPHI_WBT_ARM_KP") else "") + ("_resall" if os.environ.get("HIPHI_WBT_RESIDUAL", "fingers") == "all" else "") + ({"1": "_syn", "body": "_synbody"}.get(os.environ.get("HIPHI_WBT_SYNERGY", "0"), "")) + (f"_bodyobs{os.environ['HIPHI_WBT_BODY_OBS']}" if os.environ.get("HIPHI_WBT_BODY_OBS", "0") != "0" else "") + ("_nohandcol" if os.environ.get("HIPHI_WBT_NO_HAND_COLLISION", "0") == "1" else "") + (f"_fkp{os.environ['HIPHI_WBT_FINGER_KP']}" if os.environ.get("HIPHI_WBT_FINGER_KP") else "") + ("_nocollapse" if os.environ.get("HIPHI_WBT_COLLAPSE", "1") != "1" else "") + ("_stockobj" if os.environ.get("HIPHI_WBT_STOCK_OBJECT") == "1" else "") + ("_nohandrew" if os.environ.get("HIPHI_WBT_NO_HAND_REWARDS") == "1" else "") + ("_stockobs" if os.environ.get("HIPHI_WBT_STOCK_OBS") == "1" else "") + (f"_fs{os.environ['HIPHI_WBT_FINGER_SCALE']}" if os.environ.get("HIPHI_WBT_FINGER_SCALE") else "") + (f"_{len(_MOTIONS)}seq" if _MOTIONS else "") + (f"_{CLIPS}" if CLIPS else "")

BODY_NAMES_TO_TRACK = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link",
    "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link", *FINGERTIP_BODIES,
]

# ---------------------------------------------------------------- command
init_pose = NoiseToInitialPoseConfig(overall_noise_scale=1.0, dof_pos=0.05, root_pos=[0.05, 0.05, 0.01], root_rot=[0.1, 0.1, 0.2],
                                     root_lin_vel=[0.5, 0.5, 0.2], root_ang_vel=[0.52, 0.52, 0.78], object_pos=[0.02, 0.02, 0.0])
motion_config = MotionConfig(
    motion_file="",
    motion_dir=str(MOTION_DIR_ACTIVE),
    body_names_to_track=BODY_NAMES_TO_TRACK,
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=False,  # per-env fixed motion; phases sampled uniformly
    start_at_timestep_zero_prob=0.1,
    noise_to_initial_pose=init_pose,
)
_cmd_params = {"motion_config": motion_config, "adaptive_per_motion": os.environ.get("HIPHI_WBT_ADAPTIVE", "1") == "1"}
if "vobj" in VARIANT:  # virtual object controller curriculum: full assist while episodes are short, off once they reach len_zero steps
    _cmd_params["object_assist"] = {"kp": 60.0, "kd": 8.0, "kp_rot": 40.0, "kd_rot": 4.0, "len_full": 50, "len_zero": 300,
                                    "max_force": 150.0, "max_torque": 15.0}
command = CommandManagerCfg(
    params={},
    setup_terms={"motion_command": CommandTermCfg(func="hiphi_wbt.command:FixedMotionCommand", params=_cmd_params)},
    reset_terms={"motion_command": CommandTermCfg(func="hiphi_wbt.command:FixedMotionCommand")},
    step_terms={"motion_command": CommandTermCfg(func="hiphi_wbt.command:FixedMotionCommand")},
)

# ---------------------------------------------------------------- action (finger targets = residual on the reference pose)
FINGER_SCALE = float(os.environ.get("HIPHI_WBT_FINGER_SCALE", "0.2"))  # finger residual range factor (1.0 = holosoma's 0.25 rad)
FIXHAND = os.environ.get("HIPHI_WBT_FIXHAND", "0") == "1"  # fingers = reference playback, policy finger outputs ignored
# which joints use reference + residual targets: "fingers" (holosoma body joints = default pose + action) or "all"
RESIDUAL = os.environ.get("HIPHI_WBT_RESIDUAL", "fingers")
_REF_PATTERN = r".*" if RESIDUAL == "all" else r"^[lr]_"
SYNERGY = os.environ.get("HIPHI_WBT_SYNERGY", "0") in ("1", "body")  # "1": 41-dim action space (6 finger synergies per hand); "body": 29-dim
action = ActionManagerCfg(terms={"joint_control": ActionTermCfg(
    func="hiphi_wbt.action:SynergyResidualActionTerm" if SYNERGY else "hiphi_wbt.action:ResidualReferenceJointPositionActionTerm",
    params={"reference_joint_pattern": _REF_PATTERN, "reference_action_scale": FINGER_SCALE, "zero_reference_actions": FIXHAND}, scale=1.0, clip=None)})

# ---------------------------------------------------------------- observation (actor also sees the object pose)
actor_terms = dict(_obs.actor_obs_shared.terms)
actor_terms.update({
    "obj_pos_b": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:obj_pos_b", scale=1.0, noise=0.01),
    "obj_ori_b": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:obj_ori_b", scale=1.0, noise=0.02),
    # reference object pose (holosoma's actor never sees where the object should go)
    "ref_obj_pos_b": ObsTermCfg(func="hiphi_wbt.obs:ref_obj_pos_b", scale=1.0),
    "ref_obj_ori_b": ObsTermCfg(func="hiphi_wbt.obs:ref_obj_ori_b", scale=1.0),
})
if "geo" in VARIANT:  # hand keypoint -> object surface distances + normals + fingertip contact flags (178)
    actor_terms["hand_object_geometry"] = ObsTermCfg(func="hiphi_wbt.obs:hand_object_geometry", params={"sigma_max": 0.1}, scale=1.0)
if "bps" in VARIANT:  # SDF on a 4^3 grid around each wrist (128)
    actor_terms["hand_local_sdf"] = ObsTermCfg(func="hiphi_wbt.obs:hand_local_sdf", params={"half": 0.08, "n": 4}, scale=1.0)
if "future" in VARIANT:  # reference joints / object position 0.1 s and 0.2 s ahead (2 * 72)
    actor_terms["ref_future"] = ObsTermCfg(func="hiphi_wbt.obs:ref_future", params={"offsets": (5, 10)}, scale=1.0)
if os.environ.get("HIPHI_WBT_STOCK_OBS", "0") == "1":  # bisection knob
    actor_terms = dict(_obs.actor_obs_shared.terms)
BODY_OBS = os.environ.get("HIPHI_WBT_BODY_OBS", "0")  # "1": proprioception of the 29 body joints only; "2": + finger pos relative to the reference
_critic_terms = dict(_obs.critic_obs_w_object_terms)
if BODY_OBS in ("1", "2"):
    for terms in (actor_terms, _critic_terms):
        terms["dof_pos"] = replace(terms["dof_pos"], func="hiphi_wbt.obs:body_dof_pos")
        terms["dof_vel"] = replace(terms["dof_vel"], func="hiphi_wbt.obs:body_dof_vel")
        terms["actions"] = replace(terms["actions"], func="hiphi_wbt.obs:body_actions")
        if BODY_OBS == "2":
            terms["finger_dof_pos"] = ObsTermCfg(func="hiphi_wbt.obs:finger_dof_pos", scale=1.0, noise=0.01)
observation = ObservationManagerCfg(groups={
    "actor_obs": replace(_obs.actor_obs_shared, terms=actor_terms),
    "critic_obs": ObsGroupCfg(concatenate=True, enable_noise=False, history_length=1, terms=_critic_terms),
})

# ---------------------------------------------------------------- reward (holosoma object WBT + hand terms)
hand_terms = {
    # holosoma's action-rate penalty summed over all 69 dims made the per-step reward negative (episodes got shorter with
    # training); body joints keep the stock weight, finger residuals get a 10x smaller one
    "action_rate_l2": RewardTermCfg(func="hiphi_wbt.rewards:penalty_action_rate_subset", params={"joint_pattern": r"^(?![lr]_).*"}, weight=-0.1),
    "finger_action_rate_l2": RewardTermCfg(func="hiphi_wbt.rewards:penalty_action_rate_subset", params={"joint_pattern": r"^[lr]_"}, weight=-0.01),
    "fingertip_position_error_exp": RewardTermCfg(func="hiphi_wbt.rewards:body_subset_position_error_exp",
                                                   params={"sigma": 0.08, "body_names": FINGERTIP_BODIES}, weight=1.0),
    "wrist_position_error_exp": RewardTermCfg(func="hiphi_wbt.rewards:body_subset_position_error_exp",
                                               params={"sigma": 0.1, "body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"]}, weight=0.5),
    "finger_joint_error_exp": RewardTermCfg(func="hiphi_wbt.rewards:joint_subset_position_error_exp",
                                             params={"sigma": 0.3, "joint_pattern": r"^[lr]_"}, weight=0.5),
    # contacts with the object through the hand links are desired: exclude every hand body ([lr]_...) from the penalty
    "undesired_contacts": RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
        params={"threshold": 1.0, "undesired_contacts_body_names": (
            "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
            "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?![lr]_).+$")},
        weight=-0.1),
}
if "contact" in VARIANT:  # penalise only the excess of the fingertip-object distance over the demonstrated one
    hand_terms["fingertip_contact_excess_exp"] = RewardTermCfg(func="hiphi_wbt.rewards:fingertip_contact_excess_exp",
                                                               params={"sigma": 0.02, "ref_max": 0.05}, weight=1.0)
if "pen" in VARIANT:
    hand_terms["hand_penetration"] = RewardTermCfg(func="hiphi_wbt.rewards:hand_penetration_penalty", params={"margin": 0.01}, weight=-5.0)
if os.environ.get("HIPHI_WBT_NO_HAND_REWARDS", "0") == "1":  # bisection knob
    hand_terms = {k: v for k, v in hand_terms.items() if k in ("action_rate_l2", "finger_action_rate_l2", "undesired_contacts")}
reward = RewardManagerCfg(terms={**_rew.g1_29dof_wbt_reward_w_object.terms, **hand_terms})

# ---------------------------------------------------------------- termination
termination = TerminationManagerCfg(terms={
    "timeout": TerminationTermCfg(func="holosoma.managers.termination.terms.common:timeout_exceeded", is_timeout=True),
    "bad_tracking": TerminationTermCfg(func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly", params={
        "bad_ref_pos_threshold": 0.5, "bad_ref_ori_threshold": 0.8, "bad_motion_body_pos_threshold": 0.25,
        "body_names_to_track": BODY_NAMES_TO_TRACK,
        "bad_motion_body_pos_body_names": ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"],
        "bad_object_pos_threshold": 0.3, "bad_object_ori_threshold": 1.0}),
})

# ---------------------------------------------------------------- randomization (object mass range milder than the box demo)
_rz = _rand.g1_29dof_wbt_randomization_w_object
randomization = replace(_rz, setup_terms={**_rz.setup_terms, "randomize_object_rigid_body_mass_startup": RandomizationTermCfg(
    func="holosoma.managers.randomization.terms.objects:randomize_object_rigid_body_mass_startup",
    params={"mass_distribution_params": [0.8, 1.5]})})

# ---------------------------------------------------------------- scene / simulator
if os.environ.get("HIPHI_WBT_SINGLE_OBJECT", "0") == "1":  # profiling knob: homogeneous scene (motion/object mismatch!)
    _obj_path, _replicate = str(sorted(OBJECT_DIR_ACTIVE.glob("*/hiphi_object.urdf"))[0]), True
else:
    _obj_path, _replicate = str(OBJECT_DIR_ACTIVE), False
scene = SceneConfig(replicate_physics=_replicate, env_spacing=float(os.environ.get("HIPHI_WBT_ENV_SPACING", "6.0")),
                    rigid_objects={"object": RigidObjectConfig(urdf_file=_obj_path, position=[0.0, 0.0, 0.5])})
if os.environ.get("HIPHI_WBT_STOCK_OBJECT", "0") == "1":  # bisection knob: holosoma's 0.1 kg large_box instead of our per-env objects
    scene = SceneConfig(replicate_physics=True, env_spacing=float(os.environ.get("HIPHI_WBT_ENV_SPACING", "6.0")),
                        rigid_objects={"object": RigidObjectConfig(urdf_file="holosoma/data/scene_objects/boxes/large_box.urdf", position=[0.0, 0.0, 0.5])})
simulator = replace(g1_29dof_wbt.simulator, _target_="hiphi_wbt.sim_multiasset.IsaacSim")

g1_wuji_hiphi_wbt = replace(
    g1_29dof_wbt,
    env_class="hiphi_wbt.env.HiPHIWBTManager",
    training=TrainingConfig(project="HiPHI_WBT", name="g1_wuji_hiphi_wbt" + RUN_SUFFIX, num_envs=2040),  # multiple of 60 motions
    robot=g1_wuji,
    simulator=simulator,
    scene=scene,
    command=command,
    action=action,
    observation=observation,
    reward=reward,
    termination=termination,
    randomization=randomization,
)

# ablation: fingers = pure reference playback (the policy's finger outputs are ignored, the arm still has to place the hand)
g1_wuji_hiphi_wbt_fixhand = replace(
    g1_wuji_hiphi_wbt,
    training=TrainingConfig(project="HiPHI_WBT", name="g1_wuji_hiphi_wbt_fixhand" + RUN_SUFFIX, num_envs=2040),
    action=ActionManagerCfg(terms={"joint_control": ActionTermCfg(
        func="hiphi_wbt.action:ResidualReferenceJointPositionActionTerm",
        params={"reference_joint_pattern": r"^[lr]_", "zero_reference_actions": True}, scale=1.0, clip=None)}),
)

# fast iteration preset: subset of sequences (HIPHI_WBT_MOTIONS), fewer envs, more frequent checkpoints
g1_wuji_hiphi_fast = replace(
    g1_wuji_hiphi_wbt,
    training=TrainingConfig(project="HiPHI_WBT_fast", name="fast" + RUN_SUFFIX, num_envs=1024),
)

ROBOT_REGISTRY.add("g1_wuji", g1_wuji)
EXPERIMENT_REGISTRY.add("g1_wuji_hiphi_fast", g1_wuji_hiphi_fast)
EXPERIMENT_REGISTRY.add("g1_wuji_hiphi_wbt", g1_wuji_hiphi_wbt)
EXPERIMENT_REGISTRY.add("g1_wuji_hiphi_wbt_fixhand", g1_wuji_hiphi_wbt_fixhand)
