"""Evaluate a trained HiPHI tracking policy: per-motion body / hand / object tracking errors and success rate.

Every environment i plays motion i % N from its first frame (FixedMotionCommand in evaluation mode) until the clip ends or
the policy is terminated by the bad-tracking check. Per frame we read the motion command's metrics (mean over the frame), so
each motion gets its mean (and max for the fingertips) tracking error over the frames it survived, plus whether it reached
the end of the clip.

    PYTHONPATH=omomo_bench/wbt .../hswbt/bin/python omomo_bench/wbt/eval_tracking.py \
        --checkpoint logs/HiPHI_WBT/<run>/model_XXXXX.pt [--num-envs 60] [--init-noise 0.0] [--out DIR]

Extra ``--a.b value`` arguments are forwarded to holosoma's config override parser (e.g. ``--training.headless False``).
Outputs (in --out, default <checkpoint dir>/eval_tracking/): <model>_per_motion.csv, <model>_summary.json, <model>.md
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # hiphi_wbt plugin

import hiphi_wbt.presets  # noqa: E402,F401  (registers robot / experiments, makes hiphi_wbt.* importable)

# metrics read from the motion command after every step: name -> (column, unit scale, unit label)
METRICS = {
    "motion/error_body_pos": ("body_pos", 1000.0, "mm"),
    "motion/error_ref_pos": ("root_pos", 1000.0, "mm"),
    "motion/error_ref_rot": ("root_rot", 57.29578, "deg"),
    "motion/error_joint_pos": ("joint_pos_norm", 1.0, "rad"),
    "motion/error_wrist_pos": ("wrist_pos", 1000.0, "mm"),
    "motion/error_fingertip_pos": ("fingertip_pos", 1000.0, "mm"),
    "motion/error_fingertip_pos_max": ("fingertip_pos_max", 1000.0, "mm"),
    "motion/error_finger_joint_pos": ("finger_joint", 57.29578, "deg"),
    "motion/error_object_pos": ("object_pos", 1000.0, "mm"),
    "motion/error_object_rot": ("object_rot", 57.29578, "deg"),
    "motion/error_tip_obj_dist": ("tip_obj_dist", 1000.0, "mm"),
}
# per-frame success criteria (FlashDexRetarget-style): object within 3 cm / 30 deg; "hand" additionally mean fingertip < 6 cm
OBJ_POS_OK, OBJ_ROT_OK, TIP_OK = 0.03, 0.5236, 0.06


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="model_XXXXX.pt (local path)")
    ap.add_argument("--num-envs", type=int, default=0, help="0 = one environment per motion")
    ap.add_argument("--init-noise", type=float, default=0.0, help="overall_noise_scale of the initial-pose noise (0 = deterministic)")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = longest motion + 10")
    ap.add_argument("--out", default="", help="output directory (default <checkpoint dir>/eval_tracking)")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--record", action="store_true", help="save the simulated robot / object trajectories (<stem>_rollout.npz) for viser_rollout.py")
    ap.add_argument("--randomize", action="store_true", help="keep the training startup randomization (object/robot friction, object mass/inertia, base CoM); default: nominal physics")
    ap.add_argument("--friction", type=float, default=-1.0, help=">= 0: set robot and object static/dynamic friction to this value, restitution 0 (unified with the SPIDER MuJoCo scene)")
    ap.add_argument("--push", action="store_true", help="keep the training push perturbations (default: off, nominal evaluation)")
    ap.add_argument("--phase", type=float, default=0.0, help="start fraction of each clip (0 = first frame; e.g. 0.5 = mid-clip, object already held)")
    ap.add_argument("--zero-actions", action="store_true", help="sanity check: ignore the policy, send zero actions (fingers = reference playback)")
    ap.add_argument("--dump-joints", type=int, default=0, help="with --dump-steps: also print the N worst joints / bodies")
    ap.add_argument("--finger-kp", type=float, default=0.0, help="diagnostic: multiply the finger PD gains (0 = keep the checkpoint's)")
    ap.add_argument("--self-coll", type=int, default=-1, help="diagnostic: 0/1 overrides robot self-collision (-1 = keep)")
    ap.add_argument("--fresh-robot", action="store_true", help="diagnostic: use the robot config built now from hiphi_wbt.robot (env vars) instead of the checkpoint's")
    ap.add_argument("--urdf", default="", help="diagnostic: robot URDF file name (in assets/robot) overriding the checkpoint's")
    ap.add_argument("--freeze", action="store_true", help="diagnostic: hold the reference at the clip's first frame (robot should just stand)")
    ap.add_argument("--finger-armature", type=float, default=0.0, help="diagnostic: armature of the finger joints (0 = keep)")
    ap.add_argument("--finger-kd", type=float, default=0.0, help="diagnostic: multiply the finger damping (0 = keep)")
    ap.add_argument("--trace-joints", default="", help="diagnostic: comma-separated joint names; print q / q_ref / vel / applied torque each step")
    ap.add_argument("--poke-ref", default="", help="diagnostic: 'joint=rad' added to that joint's REFERENCE for the whole clip (use with --replay --freeze)")
    ap.add_argument("--poke", default="", help="diagnostic: 'joint=value' added to the action of that joint every step (use with --zero-actions --freeze)")
    ap.add_argument("--replay", action="store_true", help="diagnostic: every joint tracks the reference pose open-loop (implies --zero-actions)")
    ap.add_argument("--dump-steps", type=int, default=0, help="print the mean metrics of the first N steps (frame-0 alignment check)")
    return ap.parse_known_args()


def main():
    args, extra = parse_args()
    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.utils.config_registry import parse_config
    from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_saved_experiment_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

    init_eval_logging()
    os.environ["HIPHI_WBT_EVAL_PHASE"] = str(args.phase)
    os.environ.setdefault("HIPHI_WBT_NO_ASSIST", "1")  # the object-assist curriculum force must never act during evaluation
    ckpt = Path(args.checkpoint).resolve()
    out_dir = Path(args.out) if args.out else ckpt.parent / "eval_tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt.stem + (f"_{args.tag}" if args.tag else "")

    saved_cfg, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(ckpt)))
    eval_cfg = saved_cfg.get_eval_config()
    cfg: ExperimentConfig = parse_config(ExperimentConfig, default=eval_cfg, args=["--training.headless", "True", *extra],
                                         description="Eval overrides")
    if os.environ.get("HIPHI_WBT_MOTIONS"):  # evaluate on a different subset than the checkpoint was trained on
        import hiphi_wbt.presets as P

        mc_term = cfg.command.setup_terms["motion_command"]
        mcfg0 = mc_term.params["motion_config"]
        mcfg0 = {**mcfg0, "motion_dir": str(P.MOTION_DIR_ACTIVE)} if isinstance(mcfg0, dict) else dataclasses.replace(mcfg0, motion_dir=str(P.MOTION_DIR_ACTIVE))
        cfg = dataclasses.replace(cfg, command=dataclasses.replace(cfg.command, setup_terms={**cfg.command.setup_terms,
                                  "motion_command": dataclasses.replace(mc_term, params={**mc_term.params, "motion_config": mcfg0})}))
        obj = cfg.scene.rigid_objects["object"]
        cfg = dataclasses.replace(cfg, scene=dataclasses.replace(cfg.scene, rigid_objects={**cfg.scene.rigid_objects,
                                  "object": dataclasses.replace(obj, urdf_file=str(P.OBJECT_DIR_ACTIVE))}))
        print(f"[eval] motion subset override: {P.MOTION_DIR_ACTIVE}")
    if not args.push:  # nominal evaluation: no random pushes (the training config applies them every 1-3 s)
        rnd = cfg.randomization
        upd = {}
        for f in dataclasses.fields(rnd):
            terms = getattr(rnd, f.name)
            if isinstance(terms, dict) and "push_randomizer_state" in terms:
                t = terms["push_randomizer_state"]
                upd[f.name] = {**terms, "push_randomizer_state": dataclasses.replace(t, params={**t.params, "enabled": False})}
        cfg = dataclasses.replace(cfg, randomization=dataclasses.replace(rnd, **upd))
        print(f"[eval] pushes disabled in {list(upd)}")
    if not args.randomize:  # nominal physics: drop the startup randomization of materials, masses, inertias and CoM
        rnd = cfg.randomization
        st = {k: v for k, v in rnd.setup_terms.items() if not (k.startswith("randomize_") and k.endswith("_startup"))}
        print(f"[eval] nominal physics: dropped {sorted(set(rnd.setup_terms) - set(st))}")
        if args.friction >= 0:  # degenerate material ranges -> every robot / object shape gets exactly this friction
            mu, rt = [args.friction, args.friction], [0.0, 0.0]
            t = rnd.setup_terms["randomize_robot_rigid_body_material_startup"]
            st["randomize_robot_rigid_body_material_startup"] = dataclasses.replace(t, params={
                **t.params, "static_friction_range": mu, "dynamic_friction_range": mu, "restitution_range": rt})
            t = rnd.setup_terms["randomize_object_rigid_body_material_startup"]
            mat = dict(t.params["material"])
            mat["isaacsim"] = {"static_friction": mu, "dynamic_friction": mu, "restitution": rt}
            st["randomize_object_rigid_body_material_startup"] = dataclasses.replace(t, params={**t.params, "material": mat})
            print(f"[eval] friction fixed to {args.friction} (robot + object), restitution 0")
        cfg = dataclasses.replace(cfg, randomization=dataclasses.replace(rnd, setup_terms=st))
    mcfg = cfg.command.setup_terms["motion_command"].params["motion_config"]  # dataclass or dict (loaded from yaml)
    motion_dir = (mcfg.get("motion_dir") if isinstance(mcfg, dict) else mcfg.motion_dir) or ""
    if motion_dir:
        motion_names = [p.stem for p in sorted(Path(os.path.expanduser(motion_dir)).glob("*.npz"))]  # MultiMotionLoader order
    else:  # stock holosoma config: single motion_file
        motion_names = [Path(mcfg["motion_file"] if isinstance(mcfg, dict) else mcfg.motion_file).stem]
    num_motions = len(motion_names)
    num_envs = args.num_envs or num_motions
    cfg = dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, num_envs=num_envs))
    if args.finger_kp > 0:
        import re as _re

        ctrl = cfg.robot.control
        stiff = {k: (v * args.finger_kp if not any(b in k for b in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist")) else v) for k, v in ctrl.stiffness.items()}
        damp = {k: (v * args.finger_kp ** 0.5 if k in stiff and stiff[k] != ctrl.stiffness[k] else v) for k, v in ctrl.damping.items()}
        cfg = dataclasses.replace(cfg, robot=dataclasses.replace(cfg.robot, control=dataclasses.replace(ctrl, stiffness=stiff, damping=damp)))
        print(f"[eval] finger gains x{args.finger_kp}: " + ", ".join(f"{k}={v:.2f}" for k, v in stiff.items() if stiff[k] != ctrl.stiffness[k]))
    if args.fresh_robot:
        from hiphi_wbt.robot import g1_wuji as _fresh

        cfg = dataclasses.replace(cfg, robot=_fresh)
        print(f"[eval] fresh robot config: {_fresh.asset.urdf_file}, {len(_fresh.body_names)} bodies, {len(_fresh.dof_names)} dofs")
    if args.urdf:
        cfg = dataclasses.replace(cfg, robot=dataclasses.replace(cfg.robot, asset=dataclasses.replace(cfg.robot.asset, urdf_file=args.urdf)))
        print(f"[eval] robot urdf = {args.urdf}")
    if args.finger_armature > 0:
        import re as _re2

        arm = [args.finger_armature if _re2.match(r"^[lr]_", n) else a for n, a in zip(cfg.robot.dof_names, cfg.robot.dof_armature_list)]
        cfg = dataclasses.replace(cfg, robot=dataclasses.replace(cfg.robot, dof_armature_list=arm))
        print(f"[eval] finger armature = {args.finger_armature}")
    if args.finger_kd > 0:
        ctrl = cfg.robot.control
        damp = {k: (v * args.finger_kd if not any(b in k for b in ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist")) else v) for k, v in ctrl.damping.items()}
        cfg = dataclasses.replace(cfg, robot=dataclasses.replace(cfg.robot, control=dataclasses.replace(ctrl, damping=damp)))
        print(f"[eval] finger damping x{args.finger_kd}")
    if args.self_coll >= 0:
        cfg = dataclasses.replace(cfg, robot=dataclasses.replace(cfg.robot, asset=dataclasses.replace(cfg.robot.asset, enable_self_collisions=bool(args.self_coll))))
        print(f"[eval] self collisions = {bool(args.self_coll)}")
    if args.replay or args.freeze:  # diagnostics: never terminate (the robot may fall; we watch the error growth)
        args.zero_actions = args.zero_actions or args.replay
        bt = cfg.termination.terms["bad_tracking"]
        bt = dataclasses.replace(bt, params={**bt.params, "bad_ref_pos_threshold": 1e6, "bad_ref_ori_threshold": 1e6,
                                             "bad_motion_body_pos_threshold": 1e6, "bad_object_pos_threshold": 1e6, "bad_object_ori_threshold": 1e6})
        cfg = dataclasses.replace(cfg, termination=dataclasses.replace(cfg.termination, terms={**cfg.termination.terms, "bad_tracking": bt}))
    if args.replay:
        args.zero_actions = True
        term = cfg.action.terms["joint_control"]
        term = dataclasses.replace(term, func="hiphi_wbt.action:ResidualReferenceJointPositionActionTerm",
                                   params={**term.params, "reference_joint_pattern": ".*", "zero_reference_actions": True})
        cfg = dataclasses.replace(cfg, action=dataclasses.replace(cfg.action, terms={**cfg.action.terms, "joint_control": term}))

    env, device, sim_app = setup_simulation_environment(cfg)
    algo = get_class(cfg.algo._target_)(device=device, env=env, config=cfg.algo.config, log_dir=str(out_dir), multi_gpu_cfg=None)
    algo.setup()
    algo.load(str(ckpt))
    algo._pre_evaluate_policy(reset_env=False)  # eval mode + env.set_is_evaluating()
    policy = algo.get_inference_policy()

    import torch

    mc = env.command_manager.get_state("motion_command")
    mc.init_pose_cfg = dataclasses.replace(mc.init_pose_cfg, overall_noise_scale=args.init_noise)
    if os.environ.get("EVAL_PRINT_MASSES"):
        robot = env.simulator._robot
        masses = robot.data.default_mass[0].cpu().numpy()
        names = list(robot.body_names)
        print(f"[masses] total {masses.sum():.2f} kg over {len(names)} sim bodies")
        for n, m in sorted(zip(names, masses), key=lambda t: -t[1])[:8]:
            print(f"   {n:28s} {m:8.4f} kg")
        for pat in ("left_wrist_yaw_link", "l_wrist", "l_mount", "l_pinky_proximal", "l_pinky_tip", "left_elbow_link", "left_shoulder_roll_link"):
            if pat in names:
                print(f"   {pat:28s} {masses[names.index(pat)]:8.4f} kg")
        inert = robot.root_physx_view.get_inertias()[0].cpu().numpy().reshape(len(names), 3, 3)
        for pat in ("l_pinky_tip", "l_pinky_proximal", "left_wrist_yaw_link"):
            if pat in names:
                print(f"   inertia diag {pat}: {inert[names.index(pat)].diagonal()}")
    if os.environ.get("EVAL_PRINT_MATERIALS"):  # PhysX per-shape [static friction, dynamic friction, restitution]
        for label, asset in (("robot", env.simulator._robot), ("object", env.simulator.scene.rigid_objects["object"])):
            mp = asset.root_physx_view.get_material_properties().reshape(-1, 3)
            print(f"[materials] {label}: static {mp[:, 0].min():.3f}-{mp[:, 0].max():.3f} dynamic {mp[:, 1].min():.3f}-{mp[:, 1].max():.3f} "
                  f"restitution {mp[:, 2].min():.3f}-{mp[:, 2].max():.3f} ({mp.shape[0]} shapes)")
    if os.environ.get("EVAL_PRINT_OBJECTS"):
        import json as _json

        man = {e["tag"]: e for e in _json.load(open(Path(__file__).resolve().parent / "data" / "manifest.json"))}
        obj = env.simulator.scene.rigid_objects["object"]
        om = obj.data.default_mass.view(-1).cpu().numpy()
        bad = 0
        for e in range(num_envs):
            tag = motion_names[int(mc.motion_ids[e])]
            exp = man[tag]["mass_kg"] if tag in man else float("nan")
            ok = abs(om[e] - exp) < 1e-3 * max(1.0, exp)
            bad += not ok
            if e < 8 or not ok:
                print(f"[objects] env {e:3d} motion {tag[:34]:34s} sim mass {om[e]:.4f} expected {exp:.4f} {'OK' if ok else 'MISMATCH'}")
        print(f"[objects] {bad}/{num_envs} envs have an object mass that does not match their motion")
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            for e in (0, 1, 5, 30, 59):
                if e >= num_envs:
                    continue
                prim = stage.GetPrimAtPath(f"/World/hiphi_objects/env_{e}/object")
                refs = []
                for pp in [prim] + list(prim.GetChildren()):
                    for ref in pp.GetMetadata("references").GetAddedOrExplicitItems() if pp.HasMetadata("references") else []:
                        refs.append(ref.assetPath)
                body = stage.GetPrimAtPath(f"/World/hiphi_objects/env_{e}/object/object_body")
                mass = body.GetAttribute("physics:mass").Get() if body.IsValid() and body.GetAttribute("physics:mass") else None
                stack = [str(sp.layer.identifier).split("/")[-2:] for sp in body.GetPrimStack()][:4] if body.IsValid() else []
                print(f"[objects] env {e}: children={[c.GetName() for c in prim.GetChildren()][:4]} refs={refs[:2]} body_mass_attr={mass} stack={stack}")
            tmpl = stage.GetPrimAtPath("/World/Template")
            print(f"[objects] /World/Template children: {[c.GetName() for c in tmpl.GetChildren()][:5] if tmpl.IsValid() else None}")
        except Exception as ex:  # noqa: BLE001
            print("[objects] usd inspection failed:", ex)
    if args.poke_ref and not args.poke_ref.endswith("!"):
        jn, val = args.poke_ref.split("=")
        mc.motion._joint_pos[:, list(env.simulator.dof_names).index(jn)] += float(val)
        print(f"[poke-ref] reference of {jn} shifted by {val} rad for the whole clip (before reset)")
    obs = env.reset_all()
    if args.poke_ref and args.poke_ref.endswith("!"):
        jn, val = args.poke_ref[:-1].split("=")
        mc.motion._joint_pos[:, list(env.simulator.dof_names).index(jn)] += float(val)
        print(f"[poke-ref] reference of {jn} shifted by {val} rad AFTER reset: the joint must now move towards it")
    if args.poke_ref and args.poke_ref.endswith("!"):  # trailing '!' = shift AFTER the reset, so the joint has to move towards the new target
        pass
    motion_ids = mc.motion_ids.clone()
    start_idx = mc.motion.motion_start_idx[motion_ids]
    end_idx = mc.motion.motion_end_idx[motion_ids]
    motion_len = (end_idx - start_idx).cpu()
    max_steps = args.max_steps or int(motion_len.max().item()) + 10

    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    failed_at = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    frames = torch.zeros(num_envs, device=device)
    sums = {k: torch.zeros(num_envs, device=device) for k in METRICS}
    ok_obj = torch.zeros(num_envs, device=device)
    ok_hand = torch.zeros(num_envs, device=device)
    maxs = {k: torch.zeros(num_envs, device=device) for k in METRICS}
    prev_t = mc.time_steps.clone()
    t0 = time.time()
    rec = {k: [] for k in ("root", "dof", "obj", "tstep", "active")} if args.record else None
    for step in range(max_steps):
        actor_obs = torch.cat([obs[k] for k in algo.actor_obs_keys], dim=1)
        actions = policy({"actor_obs": actor_obs})
        if args.zero_actions:
            actions = torch.zeros_like(actions)
        if args.poke_ref and step in (0, 5, 20):
            if step == 0:
                q_poke0 = mc.joint_pos.clone()  # the (shifted) reference; the robot was reset to it, so deviations show what the PD does
            dq = (env.simulator.dof_pos - mc.joint_pos)[0].abs()
            top = torch.topk(dq, 6)
            print(f"[poke-ref] step {step} largest |q - q_ref|: " + ", ".join(f"{list(env.simulator.dof_names)[i]} {float(dq[i]) * 57.3:.1f}deg" for i in top.indices.tolist()))
        if args.poke:
            jn, val = args.poke.split("=")
            actions[:, list(env.simulator.dof_names).index(jn)] += float(val)
            if step == 0:
                q_poke0 = env.simulator.dof_pos.clone()
                print(f"[poke] {jn} += {val}; sim joint order (first 12): {list(env.simulator._robot.joint_names[:12])}; dof_ids[:12]={list(env.simulator.dof_ids[:12])}")
            if step in (5, 20):
                dq = (env.simulator.dof_pos - q_poke0)[0].abs()
                top = torch.topk(dq, 6)
                print(f"[poke] step {step} largest joint moves: " + ", ".join(f"{list(env.simulator.dof_names)[i]} {float(dq[i]) * 57.3:.1f}deg" for i in top.indices.tolist()))
        obs, _, dones, extras = env.step({"actions": actions})
        if args.freeze:
            mc.time_steps[:] = start_idx  # next step() advances to frame 1 again: the reference stays at the clip start
        if args.trace_joints and step < args.dump_steps:
            names = list(env.simulator.dof_names)
            robot = env.simulator._robot
            print(f"        [trace] time_step {int(mc.time_steps[0])} start_idx {int(start_idx[0])} motion_id {int(mc.motion_ids[0])} episode_len {int(env.episode_length_buf[0])}"
                  f" total {mc.motion.time_step_total} nmot {mc.motion.num_motions} | dof39={names[39]} motion.joint_pos[0,39]={float(mc.motion.joint_pos[0, 39]) * 57.3:.1f}"
                  f" raw[0,39]={float(mc.motion._joint_pos[0, 39]) * 57.3:.1f} idx39={int(mc.motion._joint_indexes[39])} mc.joint_pos[0,39]={float(mc.joint_pos[0, 39]) * 57.3:.1f}")
            for jn in args.trace_joints.split(","):
                i = names.index(jn); isim = env.simulator.dof_ids[i]
                q = float(env.simulator.dof_pos[0, i]); qr = float(mc.joint_pos[0, i]); v = float(env.simulator.dof_vel[0, i])
                tau = float(robot.data.applied_torque[0, isim]); lim = env.simulator.dof_pos_limits[i]
                print(f"        {jn}: q {q * 57.3:7.1f} ref {qr * 57.3:7.1f} deg  vel {v:6.2f}  tau {tau:7.3f}  lim [{float(lim[0]) * 57.3:.0f},{float(lim[1]) * 57.3:.0f}]  kp {float(env.action_manager.get_term('joint_control').p_gains[i]) if hasattr(env.action_manager, 'get_term') else -1:.2f}")
        if step < args.dump_steps:
            sel = active if (active.any() and not (args.replay or args.freeze)) else torch.ones_like(active)
            flags = [k.split("/")[1] for k in mc.metrics if k.startswith("term/") and bool(mc.metrics[k][0] > 0.5)]
            bt = env.termination_manager._term_instances.get("bad_tracking") if hasattr(env.termination_manager, "_term_instances") else None
            if bt is not None:
                for name in ("bad_ref_pos", "bad_ref_ori", "bad_motion_body_pos", "bad_object_pos", "bad_object_ori"):
                    fn = getattr(bt, name, None)
                    try:
                        if fn is not None and bool(fn(mc)[0]):
                            flags.append(name + "*")
                    except Exception:  # noqa: BLE001
                        pass
            if flags:
                print(f"        term flags (env 0): {flags}")
            print(f"[eval] step {step}: " + "  ".join(f"{col} {float(mc.metrics[k][sel].mean()) * sc:.1f}{u}" for k, (col, sc, u) in METRICS.items() if k in mc.metrics))
            if args.dump_joints:
                dq = (env.simulator.dof_pos - mc.joint_pos).abs().mean(0)
                top = torch.topk(dq, args.dump_joints)
                names = list(env.simulator.dof_names)
                print("        worst joints: " + ", ".join(f"{names[i]} {float(dq[i]) * 57.3:.1f}deg" for i in top.indices.tolist()))
                tracked = list(mc.motion_cfg.body_names_to_track)
                db = torch.norm(mc.body_pos_relative_w - mc.robot_body_pos_w, dim=-1).mean(0)
                topb = torch.topk(db, min(6, db.numel()))
                print("        worst bodies: " + ", ".join(f"{tracked[i]} {float(db[i]) * 1000:.0f}mm" for i in topb.indices.tolist()))
        # metrics were computed in this step before the motion advanced / envs were reset
        for k in METRICS:
            v = mc.metrics.get(k)
            if v is None:
                continue
            v = torch.nan_to_num(v.detach().float(), nan=0.0)
            sums[k] += torch.where(active, v, torch.zeros_like(v))
            maxs[k] = torch.where(active, torch.maximum(maxs[k], v), maxs[k])
        if rec is not None:  # state after this step, env-local coordinates; quaternions converted xyzw -> wxyz
            org = env.simulator.scene.env_origins
            rs = env.simulator.robot_root_states
            rec["root"].append(torch.cat([rs[:, :3] - org, rs[:, [6, 3, 4, 5]]], 1).cpu().numpy().astype(np.float32))
            rec["dof"].append(env.simulator.dof_pos.cpu().numpy().astype(np.float32))
            op, oq = mc.simulator_object_pos_w - org, mc.simulator_object_quat_w
            rec["obj"].append(torch.cat([op, oq[:, [3, 0, 1, 2]]], 1).cpu().numpy().astype(np.float32))
            rec["tstep"].append((mc.time_steps - start_idx).cpu().numpy().astype(np.int32))
            rec["active"].append(active.cpu().numpy().copy())
        frames += active.float()
        if "motion/error_object_pos" in mc.metrics:
            o = (mc.metrics["motion/error_object_pos"] < OBJ_POS_OK) & (mc.metrics["motion/error_object_rot"] < OBJ_ROT_OK)
            ok_obj += (active & o).float()
            ok_hand += (active & o & (mc.metrics["motion/error_fingertip_pos"] < TIP_OK)).float()
        timeouts = extras.get("time_outs", torch.zeros_like(dones))
        fail_now = active & dones.bool() & ~timeouts.bool()
        failed_at = torch.where(fail_now, (prev_t - start_idx), failed_at)
        if step < args.dump_steps and bool(fail_now.any()):
            bt = env.termination_manager._term_instances.get("bad_tracking") if hasattr(env.termination_manager, "_term_instances") else None
            for e in fail_now.nonzero().flatten().tolist()[:8]:
                flags = [k.split("/")[1] for k in mc.metrics if k.startswith("term/") and bool(mc.metrics[k][e] > 0.5)]
                print(f"        [fail] step {step} env {e} motion {motion_names[int(motion_ids[e])][:30]} flags {flags}")
        done_now = active & (mc.time_steps < prev_t) & ~fail_now  # time step wrapped to the start -> clip finished
        completed |= done_now
        active &= ~(fail_now | done_now)
        prev_t = mc.time_steps.clone()
        if step % 500 == 0 or not active.any():
            print(f"[eval] step {step}/{max_steps}  active {int(active.sum())}  completed {int(completed.sum())}  failed {int((failed_at >= 0).sum())}  ({time.time() - t0:.0f}s)")
        if not active.any() and not (args.replay or args.freeze):
            break
    # motions still active at max_steps count as completed (tracked to the end of the loop)
    completed |= active

    frames_c = frames.clamp(min=1.0)
    rows = []
    for e in range(num_envs):
        m = int(motion_ids[e])
        row = {"env": e, "motion": motion_names[m] if m < len(motion_names) else str(m), "frames_total": int(motion_len[e]),
               "frames_tracked": int(frames[e]), "completed": bool(completed[e]), "failed_at_frame": int(failed_at[e])}
        remaining = (motion_len[e].to(device).float() * (1.0 - args.phase)).clamp(min=1.0)
        row["obj_ok_frac"] = float(ok_obj[e] / remaining)  # fraction of the remaining clip with the object on target
        row["hand_ok_frac"] = float(ok_hand[e] / remaining)
        for k, (col, scale, _) in METRICS.items():
            row[f"{col}_mean"] = float(sums[k][e] / frames_c[e] * scale)
            row[f"{col}_max"] = float(maxs[k][e] * scale)
        rows.append(row)

    def agg(col):
        vals = [r[col] for r in rows]
        return sum(vals) / len(vals)

    summary = {
        "checkpoint": str(ckpt), "num_envs": num_envs, "num_motions": num_motions, "init_noise": args.init_noise,
        "success_rate": sum(r["completed"] for r in rows) / len(rows),
        "frames_tracked_fraction": sum(r["frames_tracked"] for r in rows) / (sum(r["frames_total"] for r in rows) * (1.0 - args.phase)),
        "phase": args.phase,
        "obj_ok_frac": agg("obj_ok_frac"), "hand_ok_frac": agg("hand_ok_frac"),
        **{f"{col}_mean_{unit}": agg(f"{col}_mean") for _, (col, _, unit) in METRICS.items()},
        **{f"{col}_max_{unit}": agg(f"{col}_max") for _, (col, _, unit) in METRICS.items()},
    }
    if rec is not None:
        np.savez_compressed(out_dir / f"{stem}_rollout.npz", motion=np.array([motion_names[int(m)] for m in motion_ids.cpu()]),
                            dof_names=np.array(list(env.simulator.dof_names)), phase=args.phase,
                            **{k: np.stack(v, 1) for k, v in rec.items()})  # (E, T, ...)
        print(f"[eval] rollout -> {out_dir / (stem + '_rollout.npz')}")
    with open(out_dir / f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    import csv

    with open(out_dir / f"{stem}_per_motion.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = ["body_pos", "wrist_pos", "fingertip_pos", "fingertip_pos_max", "finger_joint", "tip_obj_dist", "object_pos", "object_rot"]
    units = {col: unit for _, (col, _, unit) in METRICS.items()}
    md = [f"# Tracking evaluation: `{ckpt.name}`", "",
          f"{num_envs} envs, {num_motions} motions, init noise {args.init_noise}. Success = clip played to the end without a bad-tracking termination.", "",
          f"- success rate: **{summary['success_rate']:.3f}**  (frames tracked: {summary['frames_tracked_fraction']:.3f})",
          f"- frames with object within 3 cm / 30 deg: **{summary['obj_ok_frac']:.3f}**; and fingertips within 6 cm: **{summary['hand_ok_frac']:.3f}**",
          *[f"- {c} mean: **{summary[f'{c}_mean_{units[c]}']:.1f} {units[c]}** (per-motion max avg {summary[f'{c}_max_{units[c]}']:.1f})" for c in cols],
          "", "| motion | frames | ok | fail@ | obj ok | hand ok | " + " | ".join(f"{c} ({units[c]})" for c in cols) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(cols)]
    for r in rows:
        md.append(f"| {r['motion']} | {r['frames_tracked']}/{r['frames_total']} | {'yes' if r['completed'] else 'no'} | {r['failed_at_frame'] if r['failed_at_frame'] >= 0 else '-'} | {r['obj_ok_frac']:.2f} | {r['hand_ok_frac']:.2f} | "
                  + " | ".join(f"{r[f'{c}_mean']:.1f}" for c in cols) + " |")
    (out_dir / f"{stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:12]))
    print(f"[eval] wrote {out_dir / (stem + '.md')}")
    if sim_app:
        close_simulation_app(sim_app)


if __name__ == "__main__":
    main()
