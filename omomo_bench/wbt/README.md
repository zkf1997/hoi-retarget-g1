# HiPHI multi-sequence human-object interaction tracking policy (G1 + Wuji Hand 2, holosoma / IsaacSim)

English version (2026-10-22).

Uses the 60 HiPHI retargeting results in `results_hiphi/pipe_final` (final pipeline: UMR body → AnyTeleop fingers → v3 joint refinement)
to train **one** policy on holosoma's whole-body tracking (WBT) framework that tracks all 60 sequences with objects at once, with the focus on hand tracking accuracy.

```
wbt/
├── convert_hiphi_to_holosoma.py   retargeting results → holosoma motion npz + per-sequence object URDF
├── hiphi_wbt/                     holosoma plugin (registered via --import-file)
│   ├── robot.py                   RobotConfig for the 69 DoF G1 + Wuji Hand 2 (103 bodies)
│   ├── sim_multiasset.py          IsaacSim subclass: one object asset per env (env i ← asset i % 60)
│   ├── command.py                 FixedMotionCommand: env i always tracks motion i % 60; hand metrics
│   ├── action.py                  joint-position action with finger targets = reference finger pose + residual
│   ├── rewards.py                 fingertip / wrist / finger-joint tracking rewards
│   └── presets.py                 experiment preset exp:g1-wuji-hiphi-wbt (and the -fixhand ablation)
├── eval_tracking.py               per-sequence evaluation: success rate, body / wrist / fingertip / finger-joint / object errors
├── plot_training.py               tensorboard curves → PNG
├── assets/robot/                  training URDF (with foot-sole contact-point links added); meshes → ../../urdf/g1_wuji_hand2/meshes
├── assets/objects/<III>_<seq>.urdf  per-sequence object URDF (generated, git-ignored)
└── data/motions/<III>_<seq>.npz   holosoma motion data (generated, git-ignored)
```

Environment: venv `/home/kaizhao/software/venvs/hswbt` (`--system-site-packages` on top of the conda `sonic` IsaacSim 5.1 / IsaacLab 0.54.3; holosoma installed in editable mode).
All commands run from the repository root; `PYTHONPATH=omomo_bench/wbt` makes `hiphi_wbt.*` importable.

## 1. Data conversion

```bash
BENCH_DATASET=hiphi conda run -n umr python omomo_bench/wbt/convert_hiphi_to_holosoma.py   # --fps 50 --lane pipe_final
```

- The 30 fps `qpos` (7 + 69) is linearly interpolated to 50 fps (IsaacSim 200 Hz physics, decimation 4); MuJoCo forward kinematics with `robots/g1_29dof_wuji_hand2.xml` gives the world poses of the 103 bodies, velocities by finite differences.
- Joint order = the MJCF order of the retargeting results; holosoma aligns by name (the "order of the joint_names ... does not match" message in the startup log is a normal notice).
- Object: the pose is the source data × the sequence's `smpl_scale` (the same scaled world the retargeting uses); the URDF uses the display mesh as visual and CoACD convex pieces as collision, meshes scaled by s, mass = nominal mass × s³ (`NOMINAL_MASS` given per object category).
- Result: 60 sequences, 182,922 frames (2942–3269 frames per sequence ≈ 60 s); `data/manifest.json` records index, object, scale and mass.

## 2. Task definition (`hiphi_wbt/presets.py`)

Built on holosoma's `g1_29dof_wbt` (WBT with object) with these replacements:

| Part | Approach |
|---|---|
| Robot | `g1_wuji`: 69 DoF, 103 bodies (`collapse_fixed_joints=False` keeps fixed links such as fingertips); body-joint gains / armature follow the G1 preset, finger kp 0.6 (thumb CMC 1.0), kd 0.03, armature 2e-4; self-collision enabled |
| Scene | One object asset per env: `sim_multiasset.IsaacSim` uses IsaacLab `MultiAssetSpawnerCfg(random_choice=False)`, env i gets the i % 60-th URDF; `replicate_physics=False`; `env_spacing=6 m` |
| Motion assignment | `FixedMotionCommand`: env i always tracks motion i % 60 (consistent with its object); start frames are sampled uniformly within the sequence (10% start from frame 0), evaluation always starts from frame 0 |
| Tracked bodies | holosoma's 14 bodies + 10 fingertips |
| Actions | `ResidualReferenceJointPositionActionTerm`: PD target of the finger joints (`^[lr]_`) = reference finger pose + 0.25·effort/kp·a, so zero action is the retargeted finger pose; body joints as in holosoma (default pose + scaled action) |
| Observations | actor: holosoma shared observations + object position and orientation in the robot frame; critic: holosoma's with-object critic observations |
| Rewards | all terms of holosoma's with-object WBT + fingertip position exp (σ 0.08 m, weight 1.0) + wrist position exp (σ 0.1, 0.5) + finger joint exp (σ 0.3 rad, 0.5); `undesired_contacts` excludes all hand links (hand-object contact is desired) |
| Termination | `BadTrackingZOnly` (including object position 0.3 m / orientation 1.0 rad thresholds); training episode 10 s |
| Randomization | holosoma's with-object randomization, object mass factor changed to [0.8, 1.5] |

## 3. Training / evaluation

```bash
PY=/home/kaizhao/software/venvs/hswbt/bin/python
# training (2040 env = 34 rounds × 60 sequences; logs in logs/HiPHI_WBT/<time>-g1_wuji_hiphi_wbt-locomotion/, with tensorboard and model_XXXXX.pt)
PYTHONPATH=omomo_bench/wbt $PY holosoma/src/holosoma/holosoma/train_agent.py --import-file omomo_bench/wbt/hiphi_wbt/presets.py \
    exp:g1-wuji-hiphi-wbt logger:disabled --training.num_envs 2040 --training.headless True \
    --algo.config.num_learning_iterations 20000 --algo.config.save_interval 500
# curves
$PY omomo_bench/wbt/plot_training.py logs/HiPHI_WBT/<run>
# per-sequence evaluation: 60 envs each play from frame 0 to the end (or until terminated); writes <run>/eval_tracking/model_XXXXX.{md,csv,_summary.json}
PYTHONPATH=omomo_bench/wbt $PY omomo_bench/wbt/eval_tracking.py --checkpoint logs/HiPHI_WBT/<run>/model_XXXXX.pt [--init-noise 0]
```

Evaluation metrics (per sequence, averaged over surviving frames): success rate (whole sequence played without triggering bad-tracking termination), fraction of tracked frames, body / wrist / fingertip (mean and per-frame worst fingertip) / finger-joint / object position and orientation errors.

## 4. Throughput (RTX 4090, 2040 env, 2026-10-09)

| Configuration | steps/s |
|---|---|
| Default (`env_spacing=0`, heterogeneous objects, hand collision + self-collision) | 3,000 |
| Self-collision off | 3,400 |
| Hand collision geometry removed | 4,500 |
| Same object + `replicate_physics=True` | 8,900 |
| **`env_spacing=6 m` (heterogeneous objects, hand collision + self-collision kept)** | **8,500** |
| No hand collision + self-collision off + same object | 14,300 |

With `env_spacing=0` the 2040 robots overlap in space and the PhysX broad phase has to handle a large number of cross-env candidate pairs (even though they are filtered in the end); spacing them 6 m apart restores the speed of the homogeneous scene; heterogeneous assets by themselves barely affect speed. The 54 hand collision bodies account for about half of the remaining overhead and are kept for training.
The environment variables `HIPHI_WBT_ENV_SPACING`, `HIPHI_WBT_SELF_COLL=0`, `HIPHI_WBT_NO_HAND_COLLISION=1`, `HIPHI_WBT_SINGLE_OBJECT=1` in `presets.py` / `robot.py` are only for this kind of profiling.

## 5. Contact-aware improvements (2026-10-09 afternoon, fast iteration in progress)

Following FlashDexRetarget / UniCross / DexRepNet++ / Weave (see the conversation log for details), switchable terms are added to the plugin, selected with the environment variable `HIPHI_WBT_VARIANT=geo,contact,future,bps,pen`:

| Flag | Content | Dimension / weight |
|---|---|---|
| (always on) | actor observation gains the reference object position / orientation in the torso frame (the stock holosoma actor cannot see the object target at all) | 3 + 6 |
| `geo` | signed distance (clipped to ±10 cm) and normal (torso frame) from 42 hand keypoints (wrist + proximal / middle / distal / tip link origins of each finger) to the object surface + contact flags of the 10 fingertips | 178 |
| `bps` | SDF sampled at 4³ grid points (±8 cm) in each wrist frame (dynamic BPS / local occupancy) | 128 |
| `future` | reference joint angles 0.1 s and 0.2 s ahead minus current joint angles + reference object position | 2 × 72 |
| `contact` | interaction reward $r=\tfrac1{10}\sum_f \exp(-\max(0,d_f-d_f^{ref})/0.02)$: only penalizes "farther than the reference"; signed distance on the robot side, unsigned on the reference side; fingertips with reference distance > 5 cm are excluded | weight 1.0 |
| `pen` | sum of the depths by which hand keypoints penetrate the object beyond 1 cm | weight −5 |
| `vobj` | object virtual-controller curriculum (ResMimic / DexMachina): PD external force / torque on the object towards the reference pose (kp 60, kd 8, scaled by mass, limited to 150 N / 15 Nm); the gain decays linearly to 0 as the mean episode length goes from 50 to 300 steps | the object does not drop early in training and the policy sees longer trajectories |

Geometry source: `build_object_sdf.py` computes a 96³ signed-distance grid for each object's scaled CoACD convex pieces (AABB + 15 cm margin, open3d minimum over pieces); `hiphi_wbt/geometry.py` samples it trilinearly on the GPU (60 grids, 212 MB); reference-side distances are precomputed at startup for all 182,922 frames. In the reference, 51.6% of the "frame-fingertip" distances are < 1.5 cm.

**Fast iteration**: `HIPHI_WBT_MOTIONS=<tag>,<tag>` symlinks the selected motions / objects / SDFs into `data/subsets/` and trains only on those sequences; preset `exp:g1-wuji-hiphi-fast` (1024 env, project `HiPHI_WBT_fast`, run name carries the flags). `wbt_logs/launch_fast.sh <window> '<flags>' [iters]` launches it in tmux in one step.

### 5.1 Diagnosis: reference replay (2026-10-09 evening)

`eval_tracking.py --replay` (all joints track the reference pose open-loop; `--dump-joints` prints the joints / bodies with the largest deviation) reveals:

- **Robot self-collision pushes the fingers apart**: with self-collision enabled, one step after starting from the reference pose the thumb IP, pinky MCP etc. already deviate by 10–16°, and 30–40° a few steps later; a 4× finger PD gain does not help. With self-collision off, the step-0 finger error drops from 3.5° to 0.6°. The cause is that the convex-hull collision bodies of adjacent phalanges overlap at the joint (PhysX only filters directly connected parent-child links). `robot.py` therefore defaults to `enable_self_collisions=False` (hand-object collision is unaffected); all previous runs (full 1402 iterations, fast A/B/C/D) carried this problem.
- In open-loop replay the torso starts to topple within 0.3 s (root position error 70 mm, 12° at step 13); as with holosoma, closed-loop policy control is needed.
- Hand mass 0.69 kg each (holosoma hemisphere hand 0.12 kg), wrist pitch / yaw torque limit 5 Nm, gravity torque about 0.6 Nm; not a bottleneck.

Fast experiments A/B/D were restarted around 20:30 as A2/B2/D2 (self-collision off); a stock holosoma `exp:g1-29dof-wbt-w-object` (1024 env, 400 iterations, `wbt_logs/stock_ref.log`) runs in parallel as a learning-speed reference.

### 5.2 Diagnosis: why episodes stay at only a dozen steps (2026-10-13)

After the fast experiments A2 / B2 / D2 (2 sequences, 1024 env, 3000 iterations) finish, the episode length is still only 16–22 steps (0.3–0.4 s), the policy noise has already dropped to 0.33, and evaluation from frame 0 falls within 1.2 s. Item-by-item investigation (the `term/bad_*` metrics record which termination condition fires at each step; the 0.25 m thresholds of the ankle and wrist each account for half):

| Trial | Episode length at 300 iterations |
|---|---|
| stock holosoma `g1-29dof-wbt-w-object` (1024 env) | 23 (28 at 500) |
| our 69 DoF (A2 / E; E already uses holosoma foot collision + 50 Nm ankle) | 7–8 and falling |
| ours with **finger joints fixed** (F, 29 DoF actions, `HIPHI_WBT_FROZEN_HAND=1`) | 17 and rising (20 at 400) |

Conclusion: it is not the feet (the rev_1_0 URDF foot is 4 small 5 mm spheres, replaced by holosoma's cylindrical foot, `HIPHI_WBT_URDF=hscoll` is the default), not the ankle torque (35 → 50 Nm), not self-collision (already off), but the **40 finger action dimensions**: the finger residual range is holosoma's 0.25 rad per action unit, and with initial noise 1.0 the random ±14° finger actions at every step keep the body from learning. Exploration noise in the fingers was suspected first, but G (`HIPHI_WBT_FIXHAND=1`, the policy's finger output is ignored) and H (finger residual scaled to 0.2×) likewise shrink to 7–8 steps after 200 iterations, so it is not the actions themselves.

**The real cause is holosoma's `action_rate_l2` penalty (weight −0.1), which sums over all action dimensions**: with unit exploration noise E[(Δa)²] ≈ 2 per dimension per step, 69 dimensions are 2.8× more than 29, which turns the initial return from the stock −0.4 into −2.4 and makes the expected per-step reward negative, so the policy learns to fall as fast as possible to end the episode (the reason the episode length decreases with training). F has no finger action dimensions and is therefore fine. Fix: `action_rate_l2` now sums only over body joints (`hiphi_wbt.rewards:penalty_action_rate_subset`, weight −0.1), with a separate finger term at weight −0.01; the initial return is back to −0.7. The two switches `reference_action_scale=0.2` and `HIPHI_WBT_FIXHAND` are kept. Experiments I (baseline) and J (`geo,contact`) are running with the corrected reward.

### 5.3 Results (I / J / F, 1500 iterations) and adaptive sampling (2026-10-13 evening)

| Run | Episode length at 1500 | Notes |
|---|---|---|
| F fingers frozen (29 DoF) | 30 | body error 0.13 m (the open fingers count as tracked bodies) |
| I baseline (corrected reward) | 17 | evaluation from frame 0 falls within 1.3 s |
| J `geo,contact` | 18 | no difference from I; fingertip-object distance error 39 mm |
| stock holosoma (reference) | 56 (at 1200) | but its clips are only 6.5 s, ours are 62 s |

I / J evaluation still has 0 successes; the finger-joint error grows with training (the residual is used for something else), and wrist / elbow / knee already deviate by 20–30° before the fall.

Another difference from the stock setup is **adaptive start-frame sampling** (`use_adaptive_timesteps_sampler=True`, BeyondMimic style: start points sampled from a 1 s-binned histogram of failure positions), which we kept off because envs are bound to motions. `FixedMotionCommand` now has a per-motion restricted version (`adaptive_per_motion`, on by default; the global bin probabilities are masked by each env's own clip and renormalized; `HIPHI_WBT_ADAPTIVE=0` turns it off). Experiments K (baseline), L (`geo,contact`), N (only `002_rock`, a sequence with almost no walking, to see whether standing manipulation can be learned) are running.

### 5.4 Open-loop replay diagnosis (2026-10-13 late night)

New diagnostic switches in `eval_tracking.py`: `--freeze` (reference stays at frame 0), `--replay` (all joints track the reference open-loop with PD), `HIPHI_WBT_OBJECT_OFFSET` (moves the object away), `HIPHI_WBT_ZERO_INIT_VEL`, `--trace-joints` (prints q / reference / velocity / applied torque per step), `--poke-ref`, `--urdf` / `--fresh-robot` / `--finger-kp` / `--finger-kd` / `--finger-armature` / `--self-coll`. During evaluation `HIPHI_WBT_MOTIONS` now overrides the motion directory stored in the checkpoint (the earlier "002 single-sequence" evaluation actually ran the 000 from the checkpoint).

Conclusions:
- The stock holosoma G1 also falls within 0.4 s open-loop (frozen reference, PD hold): these gains are real-robot-level "soft" gains that rely on closed-loop policy compensation, so not standing open-loop is normal.
- Our robot has one more problem: **the shoulder gains (kp 14) cannot hold the 0.92 kg hand**. In these two HiPHI sequences the arms abduct 73°, the static gravity torque on shoulder roll is about 9–10 Nm, and open-loop the shoulder roll sags 50° within 0.4 s, after which the wrist error exceeds 0.25 m and triggers termination. The policy has to compensate with a constant bias of about 1.5 action units, which slows learning. New `HIPHI_WBT_ARM_KP` (shoulder / elbow / wrist kp multiplier, kd scaled by sqrt); experiments P (×3 baseline), Q (×3 + `geo,contact`) compared against K (×1).
- The finger abduction joints (pinky / ring / middle abd) drift towards the negative limit at 3–5 rad/s open-loop, and the PD torque saturating at 0.2 Nm cannot stop it; finger kp ×4, kd ×10, armature ×50, removing hand collision and moving the object away do not change this; cause unknown (suspected PhysX joint friction scaling with constraint force, or the axis in the USD conversion); for now it does not affect the conclusions on body learning.
- When replaying the F (fingers fixed) checkpoint, the old `_frozenhand.urdf` had been deleted by the regeneration logic and the `frozen2` evaluation failed; diagnostic use only.

### 5.5 Further investigation of learning speed (2026-10-14 early morning)

| Run | Setting | At 300 | Notes |
|---|---|---|---|
| K | baseline + adaptive start sampling | 13.8 | 16 at 900 |
| P / Q | shoulder / elbow / wrist kp ×3 (body still default pose + action) | worse | the stiff arm pulls the hand towards the default pose, more wrist terminations, stopped |
| R | whole-body "reference + residual" actions (`HIPHI_WBT_RESIDUAL=all`) | 14.8 | |
| S | R + shoulder / elbow / wrist kp ×3 | 15.3 | |
| stock_on_hiphi | stock holosoma 29 DoF G1 (sphere hands, stock box) directly tracking our 002 npz (`stock_on_hiphi.py`) | running | separates "hard data" from "our robot / hand is slow" |

Policy K (1000 iterations) from frame 0 of 002: root position error < 4 cm within 30 steps, but elbow / wrist gradually deviate 25–38°, fingertip error 100–200 mm, torso tilts after step 35, falls at step 64.

### 5.6 Bisection: data or robot (2026-10-14 early morning)

`stock_on_hiphi.py` lets the **stock** holosoma 29 DoF G1 (sphere hands, stock box, stock reward / observation / termination) directly track our 002 npz (the loader picks bodies / joints by name; the extra finger columns in the 69-joint file are ignored):

| Run (all on single sequence 002, 1024 env) | At 300 | Later |
|---|---|---|
| stock G1 + our 002 data (`exp:stock-on-hiphi`) | **30** | 44 at 450, **92** at 650 |
| stock task config + our 69 DoF robot (`exp:stock-on-hiphi-wuji`) | 9 | no growth |
| stock task config + our robot, fingers frozen (29 DoF, hand mass kept) | 11 | no growth |
| our task config K / R / S | 14–15 | 17 at 1500 |

Conclusion: **the data is fine, the task configuration is not the main cause, the problem is in the robot model itself** (it does not learn even with the fingers frozen to 29 DoF). Remaining differences: hand mass 0.92 kg vs 0.12 kg, 103 bodies (fixed joints not merged), the URDF itself. Two further bisections are running: the stock G1 with its sphere-hand mass changed to 0.92 kg (`exp:stock-on-hiphi-heavyhand`), and our robot with fingers frozen + fixed joints merged into 32 bodies (`HIPHI_WBT_COLLAPSE=1`).

### 5.7 Root cause: empty links default to 1 kg in IsaacSim (2026-10-14 morning)

The bisection continues: the stock G1 with the sphere hand set to 0.92 kg learns at the same speed (29 steps at 300), so it is not the hand mass; our robot with fingers frozen + fixed joints merged into 32 bodies recovers to 23 steps / 300 (42 at 500), so the problem is in the links that get merged away. Printing the simulated masses with `EVAL_PRINT_MASSES=1`: **total mass 44.4 kg (URDF 34.4 kg); the 10 fingertip links `<link name="l_pinky_tip" />` have no `<inertial>`, and IsaacSim gives each 1 kg and inertia 1e-4**; the 4 empty links imu / d435 / mid360 also get 1 kg each. So 5 kg hangs at the end of each hand: the finger PD (0.2 Nm limit) cannot hold a 1 kg fingertip, hence the "drift"; the shoulder cannot hold 5 kg, hence the 50° sag within 0.4 s open-loop; of course the policy cannot learn. The finger-frozen / merged-fixed-joint experiments merely moved these masses to the wrist.

Fix: all 14 empty links in the three URDFs get an inertial of 5 g / 1e-7. After the fix the simulated total mass is 34.46 kg, in open-loop replay the pinky abduction error is < 1°, and the shoulder roll only sags under gravity to 37° and then bounces back. `collapse_fixed_joints=True` is kept at the same time (82 bodies: fingertips and foot-sole contact points are `dont_collapse`). Experiments V (baseline) and W (`geo,contact`) are rerun with the fixed model.

All previous training results (I–U, the full 1402-iteration run) carried these 14 kg of wrong mass and are void.

### 5.8 Comparison after the fix (2026-10-14 afternoon)

| Run (single sequence 002, 1024 env) | At 200 | At 300 | At 400 |
|---|---|---|---|
| stock task config + fixed robot (fingers frozen, 29 DoF, `stockWF2`) | 13 | 22 | 32 (50 at 550) |
| our config X (69 DoF) | 16.5 | 16 | 17.7, no growth afterwards |
| X + stock box (Y) | 17.5 | 18 | 19.6 |
| X − hand reward terms (Z) | 14 | 16.5 | 14 |
| our preset, all switches set to stock behavior, 69-dim actions (AA) | 8 | 14.6 | |
| AA but 29-dim actions (fingers frozen, AB) | running | | |

With the robot model fixed, the stock task configuration learns normally; our task configuration stalls at 16–20 steps after 300 iterations on the same sequence. Object, hand rewards, observations and adaptive sampling are not the cause (Y / Z / AA). The remaining suspect is the **69-dim action space itself**: the 40 finger dimensions, even when ignored (fixhand), enter the PPO Gaussian policy, the entropy and the KL; when the KL is high, the adaptive learning rate is pushed back to 1e-5 (in tensorboard AA's actor LR jumps between 1e-5 and 7.6e-5 while the stock run stays at 7.6e-5), and the noise std decays more slowly (0.79 vs 0.57). AB is used to verify this.

### 5.9 Further bisection (2026-10-14 afternoon – evening)

| Run (single sequence 002) | 200 | 300 | 400 | Notes |
|---|---|---|---|---|
| AB: AA's switches + fingers frozen (29-dim) | 14.5 | 30 | 44.5 | **as fast as stock, even faster** |
| AC: X + `desired_kl` 0.02 | 17.9 | 15 | | no effect |
| AE: X + finger synergy actions (41-dim, `HIPHI_WBT_SYNERGY=1`) | 15.6 | 16.2 | | no effect → not the action dimensionality |
| AH: AA + hand collision geometry removed | 8.9 | | | same as AA → not finger-object contact |
| AI: AA + finger kp ×0.1 | 9.1 | | | same |
| AJ: AA + observations with body joints only (`HIPHI_WBT_BODY_OBS=1`) | running | | | whether the 40 finger joint positions / velocities / actions are observation noise |
| AK: X + synergy actions + body observations + fingers relative to reference | running | | | |

The only difference between AA and AB is whether the finger joints "PD-track the reference (action ignored)" or are "fixed joints"; the gap between them (19 vs 44) shows the problem comes from having the finger joints at all. Further split:

| Run (AA's switches + …, 002) | 300 | 400 | 550 |
|---|---|---|---|
| AA: 69-dim actions | 14.6 | 19.4 | |
| AJ: AA + fingers removed from observations | same as AA | | | 
| AL: 41-dim (synergy) | 17.9 | 25 | 30 |
| AM: 29-dim (fingers only replay the reference, `HIPHI_WBT_SYNERGY=body`) | 20.4 | 27.7 | 36 |
| AB: 29-dim, fingers as fixed joints | 30 | 44.5 | 57 |

Conclusion: **two factors add up**. (1) Policy action dimensionality: 69 → 41 → 29 dims gets progressively faster (the ineffective finger dimensions raise the variance of the KL / importance ratio); (2) the fingers closing on the box per the reference makes balance harder by itself (AM vs AB). AN (AM + hand collision removed) 20.8 at 500 and AO (AM + our 1.1 kg box) 21.6 at 500 are no better than AM (without hand collision the box loses its support and it gets harder), so the remaining gap is the task of "holding the box with the fingers per the reference" being harder in itself, plus ±5-step noise of single-seed experiments; no longer a bug.

**Current default configuration**: mass-fixed 82-body model, `HIPHI_WBT_SYNERGY=1` (41 dims: 29 body + 6 synergy residuals per hand), adaptive start sampling, `geo,contact`. AF (baseline) / AG (`geo,contact`) / AP (`geo,contact,vobj`) run 2000 iterations on two sequences for model selection, then back to the full 60 sequences.

### 5.10 The real difficulty in evaluation (2026-10-14 evening)

AG (synergy actions + `geo,contact`, two sequences, 1000 iterations), closed-loop evaluation from frame 0: body error 43 mm, wrist 18 mm, fingertip 37 mm are all fine, but **both sequences are terminated by `bad_object_pos` at step 64 at the same time**: in both the box is on the ground, the left / right hands reach the box around step 32, and in steps 35–64 the 1.1 kg box is lifted at 1.3 m/s; the policy follows the body but does not pick up the box (the fingertip-object distance jumps from 42 mm to 148 mm, the box stays on the ground, error exceeds 0.3 m). So the success rate from frame 0 measures the single hardest moment, "quickly lift the box off the ground".

Countermeasures: object virtual-controller curriculum (`vobj`, AP running), longer training, possibly finger gains; add evaluation metrics that start mid-sequence (object already held).

### 5.11 Second root cause: all 60 envs actually use env_0's box (2026-10-15)

Evaluation of the full-run policy at 12500 iterations: 3/60 successes; the 5 clothes-rack sequences are all terminated by `bad_object_pos` at step 12. `EVAL_PRINT_OBJECTS=1` shows **the object mass in 59/60 envs is 1.1021 kg (env_0's Box_A)**. Cause: holosoma spawns the objects first and then calls `clone_environments(copy_from_source=False)`; env_i inherits env_0's entire subtree live, so each env contains both its own object and env_0's object, and since the path pattern of the IsaacLab rigid-body view is taken from env_0's body name, all envs treat env_0's box as "the object" while the real object becomes an uncontrolled extra rigid body. All previous multi-object training (including the full train2 run) is affected; the two-sequence fast experiments (000 and 002 are both Box_A) are not.

Tried: deep-copy cloning (PhysX GPU view crashes), deactivating the inherited copies (the view pattern still only matches env_0), unifying body names (mass and other attributes are still overridden by env_0's stronger opinion). Final fix (`sim_multiasset.py`): objects are placed outside the env subtree at `/World/hiphi_objects/env_i/object` (identity transform; holosoma already writes poses in world coordinates + env origin) and spawned after cloning; the object URDF becomes `assets/objects/<tag>/hiphi_object.urdf` and the body link is uniformly named `object_body`. Verification: 0/60 mass mismatches, the clothes rack no longer fails at step 12; the earliest failure is now the soccer ball (`bad_object_ori`; the orientation of a rolling ball is meaningless anyway).

Full training restarted with the fixed scene (`geo,contact,vobj`, synergy actions), 4.9 s/iteration, directory `logs/HiPHI_WBT/20261015_125909-g1_wuji_hiphi_wbt_contact-geo-vobj_syn-locomotion`, tmux `wbt:train4`.

### 5.12 Object assist force leaking into evaluation (2026-10-20)

The virtual PD force of the `vobj` curriculum decays with the "mean training episode length" (fully on at 50 steps, off at 300). In the full run the episode length stays at 40–50 steps, so **the assist force is at full strength throughout training** (`assist/scale` ≈ 1.0) and the policy never has to move the object by itself; at evaluation the checkpoint's configuration also carries `object_assist`, so the object is pulled towards the reference pose even when not touched. `eval_tracking.py` now defaults to `HIPHI_WBT_NO_ASSIST=1`.

`model_30499` without the assist force (from frame 0): 4/60 completed (6/60 with assist), object correct 5.3% (7.8%), body 55 mm and fingertip 61 mm essentially unchanged. The only sequence where both body and object truly track is 052 (bench, object 15 mm, 95%); the object-correct rate of 050 drops from 53% to 6%.

Conclusion: the decay condition of vobj is wrong and should decay by iteration count; retraining with the fixed objects and without vobj (or with per-iteration decay) is needed.

## 6. Status

- 2026-10-09 full training of `exp:g1-wuji-hiphi-wbt` (2040 env, about 5.8 s/iteration, run directory `logs/HiPHI_WBT/20261009_124245-g1_wuji_hiphi_wbt-locomotion`) was stopped manually after 1402 iterations and replaced by fast iteration; reward −2.4 → 0.55, episode length 11 → 14 steps, checkpoint `model_01000.pt`.
- 2026-10-14 21:48 full training restarted: `exp:g1-wuji-hiphi-wbt`, `HIPHI_WBT_SYNERGY=1 HIPHI_WBT_VARIANT=geo,contact`, 2040 env, 30000 iterations, 7.6 s/iteration, tmux `wbt:train2`, log `wbt_logs/train_full2.log`, directory `logs/HiPHI_WBT/20261014_194823-g1_wuji_hiphi_wbt_contact-geo_syn-locomotion`. AP (two sequences, +`vobj`) runs in parallel to decide whether to add the curriculum.
- 2026-10-09 15:05 three parallel fast experiments (2 sequences: `000_Bringing-lug_0311`, `002_Cause_to_move_in_place-rock_0108`, 1024 env, 3000 iterations, about 11 s/iteration each with all three running): A baseline, B `geo,contact`, C `geo,contact,future,bps,pen` (no difference among the three at 500 iterations, episodes only 11 steps; C stopped at 18:50 and replaced by D `geo,contact,vobj`), directories `logs/HiPHI_WBT_fast/20261009_1505*`, `_1506*`, logs `wbt_logs/fast_fast{A,B,C}.log`.
- Alignment check (`eval_tracking.py --zero-actions --dump-steps 12`): step-0 body error 8.5 mm, fingertip 7.4 mm, finger joint 4.2°, object 11.6 mm, showing that motion data, object assets, env origins and robot initialization are consistent; the robot falling afterwards under zero actions is normal.
- `model_00500` (iteration 500) per-sequence evaluation success rate 0 (all terminated by bad-tracking within 0.5 s), the expected result this early in training; holosoma's WBT usually needs tens of thousands of iterations. Later checkpoints are evaluated one by one as `model_XXXXX.pt`.
- `exp:g1-wuji-hiphi-wbt-fixhand` (fingers pure replay, policy does not control the fingers) is pending as an ablation (running it alongside the main training halves the speed of both; wait for the main training to finish or for a free GPU).
