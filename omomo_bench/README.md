# omomo_bench — Whole-body retargeting of human-object interaction motion to G1 + dexterous hands

Retargets three human-object interaction datasets, OMOMO / HUMOTO / HiPHI (SMPL-X body + fingers + object 6DoF trajectory, 30 fps), to the Unitree G1 29-DoF plus the Wuji Hand 2 (or BrainCo Revo2) five-finger hand, outputs all robot joint trajectories and the object trajectory, and provides two sets of metrics and visualizations, for the body and for the hands. The directory contains two kinds of things:

- **Comparison of two upstream methods for the body stage**: OmniRetarget (`holosoma_retargeting`) vs UMR, fingers fixed at the rest pose. Conclusion: UMR penetrates less, is faster and has a more stable root; OmniRetarget tracks more accurately but with hands 55–73% of the sequences need a fallback.
- **The final decoupled pipeline** (chosen 2026-10-07): UMR body → AnyTeleop fingers → v3 coupled refinement (12 iterations); all 100 HUMOTO sequences and 60 HiPHI sequences have been run, outputs in `results_*/pipe_final`.

Below, we first describe what each step of the final pipeline does, then how to run it, the metrics, the results, and finally the ablations behind all design decisions.

---


## 1. Directory layout

```
omomo_bench/
├── env_setup.sh                creates the conda environments umr / hsretargeting
├── prepare_omomo.py            OMOMO pickle → UMR / OmniRetarget inputs + object meshes (CoACD) + source npz        [umr]
├── prepare_humoto.py           HUMOTO (SMPL-X fitted via hiphi2smplx) → data_humoto/                                [umr]
├── prepare_hiphi.py            HiPHI (SMPL-X from hiphi2smplx + object trajectories) → data_hiphi/                  [umr]
├── prepare_humoto_mixamo.py    Variant inputs for OmniRetarget using Mixamo joints directly (mx/mxh/mxtoe/mxpel/mxleg) [umr]
├── prepare_scaling_variants.py UMR inputs with OmniRetarget-style object z anchoring (data/umr_omniscale)             [umr]
├── prepare_realsize_hand_anchor.py UMR inputs with real-size objects anchored at the wrist (data_*/umr_realhand)      [umr]
├── run_omni.py / omni_entry.py Batch OmniRetarget (hand URDF, UMR-style scaling, fallback chain, upstream patches)    [hsretargeting]
├── run_umr.py                  Batch UMR (--hand, --scaling umr|omni|real|realhand, --variant)                       [umr]
├── hand_pipeline.py            Stages 2/3: AnyTeleop finger replacement + s3 / s3p / v2 / v3 refinement → results_*/pipe_<stage>   [umr]
├── eval_pipeline.py            Body + hand metrics → results_*/pipe_metrics*.csv, pipe_summary*.md                   [umr]
├── compare.py                  Body metrics + model/object utility functions (reused by eval_pipeline / visualization) [umr|hsretargeting]
├── wrist_orientation_stats.py  Body-stage wrist orientation error statistics                                          [umr]
├── viser_compare.py            viser side-by-side playback (--pipeline preset = pipeline stages)                      [hsretargeting]
├── render_compare.py           Static PNG grid (columns = source + lanes, rows = frames), MuJoCo EGL offscreen rendering [umr|hsretargeting]
├── run_final_pipeline.sh       One-shot: UMR body → hand stages → evaluation (per dataset)
├── run_variant_pipeline.sh     Same, with a different UMR body variant
├── scaling_conflict_figure.py  Body/hand size conflict illustration → results/figures/scaling_conflict.png
├── make_hand_mjcf.py           G1+hand URDF from ../urdf/ → robots/g1_29dof_<hand>.xml + configs/umr_g1_<hand>*.json
├── hand_rest_pose.json / hand_rest.py   Finger rest pose (fnmatch rules for both hands)
├── configs/umr_g1_wuji*.json   UMR robot config and variants (_v2 _real _free _flat _contact0 _contact4 ...)
├── data[_humoto|_hiphi]/       (git-ignored) converted data: source/ umr/ omni/ objects/ seqs_test.json
├── results[_humoto|_hiphi]/    (git-ignored) per-lane npz, logs, metrics, figures
└── omni_workdir[_humoto]/      (git-ignored) working directory required by holosoma
```

Environments: `umr` (UMR, hand_pipeline, eval, rendering), `hsretargeting` (holosoma, mujoco pinned to 3.3.5, viser), `handbench` (AnyTeleop in hand_bench). All scripts switch the data directory via `BENCH_DATASET=omomo|humoto|hiphi`; MuJoCo offscreen rendering and collision detection require `MUJOCO_GL=egl`.

---

## 2. What the final pipeline does

### 2.0 Inputs and scale conventions

Input per sequence (`data_*/source/<seq>.npz`): SMPL-X parameters (16-dim betas, 63-dim body, 90-dim fingers), 55 joints, the pose trajectory of the main object, T-pose height; object meshes in `data_*/objects/<obj>/` (visual mesh + CoACD convex pieces). The 21 hand keypoints and the AnyTeleop results come from `../hand_bench/`.

**Scaled world**: UMR requires the human to be scaled to the robot height, $s = 1.32 / \text{T-pose height}$ (HUMOTO 0.85, OMOMO about 0.77, HiPHI 0.72–0.85). Human vertices, joints, object positions and object meshes are all multiplied by $s$; the robot stays at real size. All stages, metrics and visualizations live in this world; the result npz stores `smpl_scale` and `object_mesh_scale` (defaults to $s$). This means the robot hand (middle-finger chain length 20.9 cm, 15% longer than the human 18.1 cm) has to grasp an object shrunk to 0.77–0.85×, so the hand/object size ratio is 1.35–1.5× that of the human. Several alternatives to this conflict are in Section 6; the conclusion is that the scaled world is still best.

### 2.1 Stage 1: UMR body retargeting (`run_umr.py --hand wuji`)

UMR retargets with **point-cloud surface correspondences**. Offline, between two sampled point clouds, the SMPL-X T-pose (the sequence's betas, fingers straight) and the robot T-pose (`tpose_qpos` in `configs`, fingers at the rest pose), a template-residual autoencoder learns 4096 slots with a shared ordering; each slot is bound at one end to a face of the SMPL-X mesh and at the other end to a surface point of some robot geom, and moves with the respective pose. Online, each frame solves a Gauss-Newton-style QP whose variables are the 7-dim root pose plus the 29 body joints (the 40 finger joints are pinned to the rest pose of `hand_rest_pose.json` via `joint_limits = [v−1e−4, v+1e−4]`), with residual terms:

| Term | Slots it acts on | Residual | Weight |
|---|---|---|---|
| Surface position | 456 sampled per body segment (each hand 15 / 80, torso 50 / 714) | $p^{robot}_i(q) - s\,p^{human}_i$ | hand 10, torso 20, head 1 |
| Surface normal | same as above | $n^{robot}_i(q) - \hat n_i$, target normal with a fixed correction by the human-robot normal angle in T-pose | hand 1, legs/torso 5, forearm 0 |
| Object contact | the nearest 128 of all 4096 within ≤ 10 cm of the object | $(p^{robot}_i - o_{k(i)}) - v_i$, where $v_i$ is the offset from the human-side slot to the nearest object sample point (zeroed below 5 mm) | 5 × strength (0.5–1), multiplied directly into the row, effective quadratic weight 6–25 |
| Ground contact / self-contact / temporal smoothing | | | defaults |

Constraints: joint limits, ground non-penetration, robot-object non-penetration (`mj_geomDistance` linearized, with slack, slack cost 5000). Output `results_*/umr_wuji/<seq>.npz`: `qpos (T, 7+69)` in `robot_joint_names` order, `smpl_scale`, `ground_z`, `hand`, `object_mesh_scale`.

Correspondences are reused per SMPL-X template: HUMOTO one template (shared by all 100 sequences), OMOMO per subject, HiPHI one template per sequence (betas fitted per sequence at conversion), so HiPHI must use `--no-slot-reuse`. About 40 s per sequence (HUMOTO) to 3 min (long HiPHI sequences).

The wrist pose given by the body stage can only serve as an initial value: the wrist position is systematically pulled back by about 4 cm (the robot hand is longer than the human hand, and surface matching aligns the fingertips, which pushes the wrist back), the wrist orientation error averages 10°, and when carrying bowls, plates or vases with both hands the whole hand flips by 40–50° (the least-squares compromise of the contact term with pinned fingers, see 6.5).

### 2.2 Stage 2: AnyTeleop fingers (`../hand_bench/run_anyteleop.py`, `hand_pipeline.py --stage s2`)

The fingers do not depend on the body stage. For each sequence and each hand, the human's 21 keypoints are taken in the **canonical hand frame** (x = wrist→middle MCP, z = orthogonalized direction pinky MCP→index MCP, y = palm normal; consistent with the wrist frame convention of the URDFs of both G1 hands), and the vector optimizer of dex-retargeting solves the 20 finger joints frame by frame: the target is the 5 wrist→fingertip vectors multiplied by `scaling_factor`, ratio = robot middle-finger chain length / human middle-finger chain length (Wuji ≈ 1.15), with joint limits and temporal smoothing (`low_pass_alpha=1.0`, i.e. no filtering). This step was already verified in the hand benchmark to outperform TopoRetarget (contact F1 0.74 vs 0.61).

`pipe_s2` simply writes the 40 AnyTeleop finger joint values frame by frame into the UMR qpos; the arms do not move. On its own it is unusable (the fingertip targets ignore the object, penetration 6.5 mm), but it provides a global hand shape that "already wraps the object", which is the precondition for the later refinement to converge (without it F1 drops from 0.74 to 0.48, see 6.3).

### 2.3 Stage 3: v3 coupled refinement (`hand_pipeline.py --stage v3 --v3-iters 12`)

Each frame and each hand solves a 27-variable QP (7 arm joints + 20 finger joints), linearized from the AnyTeleop hand shape (first frame) or the previous frame's solution (other frames), with 12 inner iterations per frame. The 7 arm DoFs are all G1 joints from the shoulder to the flange (shoulder 3, elbow 1, wrist 3); beyond the 6-dim wrist pose target, one redundant DoF is left for the priors and non-penetration; the waist does not participate, to avoid changing the body stage's torso.

| Term | Content | Weight |
|---|---|---|
| Wrist position anchor | flange `{l,r}_mount` to the scaled human wrist keypoint | $50 \cdot \max(\alpha, 0.15)$ |
| Wrist orientation | `wrist_yaw_link` to the human canonical hand frame | 5 |
| Contact fingertip targets | fingertips whose human fingertip is < 2 cm from the object surface (hysteresis: enter 2 cm, exit 4 cm, at least 3 frames); target = human contact point projected onto the object surface and mapped into the robot world; thumb / index / others priority 1.45 / 1.25 / 0.78 | 200 |
| Finger prior | AnyTeleop hand shape | 1 |
| Arm prior | UMR arm solution | $0.5 \cdot (1 + 4(1-\alpha))$ |
| Temporal terms | fingers: previous frame 2, acceleration 20; arm: velocity of the correction relative to the UMR solution 40, absolute acceleration 60 | |
| Non-penetration | all hand + forearm geoms against the object's CoACD pieces, pairs within 2 cm linearized, 2 mm margin, with slack | slack 10⁴ |
| Joint limits | hard constraints | |

$\alpha(t)$ is the contact degree: a 12-frame causal moving average of the indicator "fingertip < 5 cm from the object". In contact, the wrist anchor has full weight and the arm prior is weak; out of contact, the wrist anchor drops to 15% and the arm prior is 5× stronger. Contact targets are **not** remapped by hand size (ablation shows it is useless, see 6.4); the hand size difference is absorbed by the relaxed wrist anchor. Solver: Clarabel. Output `results_*/pipe_final/<seq>.npz` (symlink to `pipe_v3_it12`), npz layout identical to stage 1, plus `stage`, `hands`, `params`.

12 iterations vs 4 give a small but consistent gain (HUMOTO F1 0.739 → 0.745, penetration 0.77 → 0.71 mm; HiPHI unchanged), at the cost of a 3× slower hand stage (HUMOTO 100 sequences with 8 processes take about 9 minutes).

### 2.4 Evaluation and visualization

`eval_pipeline.py` computes the metrics of Section 4 for each lane with MuJoCo forward kinematics + collision detection; `viser_compare.py --pipeline` plays the source human and all stages side by side; `render_compare.py --pipeline --focus right_hand` produces static grid figures.

---

## 3. How to run

```bash
# Data preparation (see Section 7)
export BENCH_DATASET=humoto            # or hiphi
# Stage 1 + 2/3 + evaluation, one shot
bash run_final_pipeline.sh humoto 3                    # HUMOTO: UMR with 3 processes
bash run_final_pipeline.sh hiphi  3 --no-slot-reuse    # HiPHI: retrain correspondence per sequence
```

Step by step:

```bash
conda run -n umr python run_umr.py --hand wuji --workers 3 [--no-slot-reuse]                         # -> results_*/umr_wuji
# AnyTeleop fingers: in hand_bench, BENCH_DATASET=<ds> python run_anyteleop.py --hand wuji        # -> hand_bench/results*/anyteleop/wuji
MUJOCO_GL=egl conda run -n umr python hand_pipeline.py --stage s2 v3 --v3-iters 12 --out-suffix _it12 --workers 8   # -> pipe_s2_it12, pipe_v3_it12
MUJOCO_GL=egl conda run -n umr python eval_pipeline.py --out _it12 --lane v3_it12=results_$BENCH_DATASET/pipe_v3_it12 --workers 8
conda run -n hsretargeting python viser_compare.py --final --port 8090      # source | stage 1 | 2 | final (--pipeline shows all intermediate versions)
MUJOCO_GL=egl conda run -n umr python render_compare.py --pipeline --seqs <seq> --focus right_hand --distance 0.7 --tag _right
```

Common switches: `hand_pipeline.py --stage s2 s3 s3p v2 v3` (the refinement versions, see 6.2), `--body-lane` (use another body-stage lane), `--out-suffix`, `--finger-prior rest`, `--v3-remap ratio`, `--obj-mesh-scale 1.0`; `run_umr.py --scaling umr|omni|real|realhand`, `--variant v2|free|flat|contact0|contact4|...` (corresponding to `configs/umr_g1_wuji_<variant>.json`); `eval_pipeline.py --lane NAME=DIR[:SUFFIX]` (repeatable), `--out` (mandatory, otherwise the default csv is overwritten), `--merge a.csv b.csv --out X` (merge into a comparison table). Run long jobs in tmux.

Long-running viser instances: HUMOTO 8090, HiPHI 8091, OMOMO 8092 (OmniRetarget vs UMR). The comparison of the scaled world against real-size anchoring is started manually with `--lane` when needed.

---

## 4. Metrics

Body metrics (`compare.py`, computed in the scaled world): `obj_pen_max` mean over frames of the deepest robot-object penetration, `obj_pen_frac` fraction of frames with penetration > 5 mm, `ground_pen`, `foot_slide` (robot ankle xy displacement during human foot stance phases), `hand_contact` (fraction of frames with the human wrist < 10 cm from the object in which the robot hand is also < 10 cm), `wrist_err / ankle_err` (against the scaled SMPL-X keypoints), `joint_jerk / root_jerk` (mean absolute third-order difference × fps³).

Hand metrics (`eval_pipeline.py`, distances converted to human units = robot world / s, computed per hand and then averaged): `flange_pos_err_cm` / `flange_rot_deg` flange error relative to the human wrist keypoint and the canonical hand frame; `tip_pos_err_cm` distance of robot fingertips to human fingertips in interaction frames (tracking metric); `tip_obj_dist_err_cm` difference between the robot fingertip-to-object-surface distance and the human's (grasp metric); `contact_recall / precision / f1`, with a 2 cm contact threshold for both human and robot fingertips; `hand_pen_mm / hand_pen_frac` hand-object penetration (interaction frames only); `finger_jerk`.

---

## 5. Results

### 5.1 Final pipeline (HUMOTO 100 sequences / HiPHI 60 sequences, 2026-10-07)

> **Update 2026-10-13**: the HUMOTO final is now v3 on the finger-collision-off body (`pipe_v3_nfcfull`, see 6.10): contact F1 0.799, hand-object penetration 0.89 mm / 4.6 % of frames, flange 2.42 cm / 4.0 deg, joint jerk 167 / root jerk 16 (`results_humoto/pipe_summary_nfcfull.md`). The HUMOTO column below is the earlier `pipe_v3_it12`; the HiPHI column is still the final.

| Metric | HUMOTO stage 1 | HUMOTO stage 2 | **HUMOTO final** | HiPHI stage 1 | HiPHI stage 2 | **HiPHI final** |
|---|---|---|---|---|---|---|
| Contact F1 (recall / precision) | 0.26 | 0.46 | **0.745 (0.71 / 0.86)** | 0.17 | 0.29 | **0.577 (0.53 / 0.76)** |
| Fingertip-object distance error (cm) | 4.43 | 2.44 | **1.30** | 5.22 | 3.70 | **1.74** |
| Fingertip position error (cm) | 9.5 | 5.0 | **3.25** | 13.5 | 8.7 | **5.66** |
| Hand-object penetration (mm) / frames >5 mm | 2.6 / 0.18 | 6.5 / 0.48 | **0.71 / 0.044** | 3.1 / 0.18 | 12.4 / 0.64 | **0.66 / 0.016** |
| Whole-body object penetration (mm) / frames >5 mm | 4.5 / 0.32 | 9.9 / 0.70 | **1.7 / 0.11** | 5.7 / 0.44 | 15.6 / 0.75 | **3.1 / 0.24** |
| Flange position (cm) / orientation (°) | 4.25 / 9.9 | 4.25 / 9.9 | **2.54 / 4.0** | 3.91 / 14.1 | 3.91 / 14.1 | **3.04 / 6.0** |
| Wrist tracking error (cm) | 3.7 | 3.7 | **2.2** | 3.0 | 3.0 | **2.3** |
| Body joint jerk / root jerk | 258 / 19.4 | 258 / 19.4 | **191 / 19.4** | 193 / 10.3 | 193 / 10.3 | **134 / 10.3** |
| Finger jerk | 0.6 | 53 | 65 | 0.5 | 199 | 88 |

Full table in `results_*/pipe_summary_chosen.md`, per sequence in `pipe_metrics_it12.csv`. HiPHI objects are mostly large (boxes, chairs, buckets, benches, mops), handled with both hands, so the contact F1 is naturally lower. Figures: `results_*/figures/*_final_*.png` (columns = source | stage 1 | stage 2 | final).

### 5.2 Evolution of the refinement versions (HUMOTO 100 sequences, 4 iterations)

| Metric | s2 (+AnyTeleop) | s3 finger QP | s3p v1 +arm placement | v2 coupled QP | v3 |
|---|---|---|---|---|---|
| Contact F1 | 0.46 | 0.63 | 0.73 | 0.73 | **0.74** |
| Fingertip-object distance error (cm) | 2.44 | 1.58 | **1.23** | 1.33 | 1.30 |
| Hand-object penetration (mm) / frames >5 mm | 6.5 / 0.48 | 2.9 / 0.20 | 2.2 / 0.14 | 0.86 / 0.054 | **0.77 / 0.048** |
| Whole-body object penetration (mm) | 9.9 | 5.2 | 3.9 | 2.0 | **1.8** |
| Flange position (cm) / orientation (°) | 4.25 / 9.9 | 4.25 / 9.9 | **1.67 / 1.0** | 2.05 / 3.8 | 2.56 / 4.0 |
| Body joint jerk | 258 | 258 | **182** | 207 | 191 |
| Finger jerk | 53 | 144 | 113 | **64** | 66 |

Same set on HiPHI: F1 0.29 / 0.48 / 0.52 / 0.55 / **0.58**, hand-object penetration 12.4 / 4.9 / 7.0 / 1.2 / **0.65** mm, flange 3.9 / 3.9 / **2.1** / 2.5 / 3.0 cm, joint jerk 193 / 193 / 145 / 163 / **135**. The gain from v1 → v2 is in penetration (non-penetration constraints and contact targets enter the same QP, the wrist anchor yields); v3 further reduces penetration, raises F1 and lowers jerk; v1 has the best wrist tracking because it explicitly IKs to the human wrist frame, while v2/v3 relax the wrist anchor on purpose.

### 5.3 Comparison of the two body-stage methods (fingers at rest pose)

**Summary document: `reports/body_stage_comparison_2026-10-17.md`** (UMR vs OmniRetarget, CoACD vs convex hull, the pushed-off mechanism, the finger collision switch, penetration after running all stages, recommended configuration). OMOMO 602 sequences, HUMOTO 100 sequences, HiPHI 60 sequences (per-sequence height), uniform UMR scaling, all lanes evaluated with the same CoACD object geometry. OmniRetarget uses CoACD collision (our default, see 6.9), rerun after the baked-hand fix of 2026-10-15. Full tables, BrainCo and the fallback breakdown are in `reports/umr_vs_omni_3ds.md` (`results*/summary_3ds.md`, `wrist_gap_3ds.log`, `wrist_orientation_3ds.log`).

| Metric (G1 + Wuji) | OMOMO Omni | OMOMO UMR | HUMOTO Omni | HUMOTO UMR | HiPHI Omni | HiPHI UMR |
|---|---|---|---|---|---|---|
| Object penetration (mm) / frames >5 mm | 10.1 / 42% | **5.4** / 43% | 10.4 / 60% | **4.5 / 32%** | 13.5 / 47% | **5.7 / 44%** |
| Sequences needing fallback | 359 / 602 | 0 | 71 / 100 | 0 | 32 / 60 | 0 |
| Hand-object contact preservation | **0.97** | 0.95 | **1.00** | 0.87 | 0.96 | 0.96 |
| Wrist position error (cm) | **2.2** | 3.5 | **1.6** | 3.7 | **2.9** | 3.0 |
| Wrist orientation error (°) / frames >30° | 32 / 43% | **13 / 6%** | 39 / 59% | **10 / 5%** | 47 / 67% | **14 / 3%** |
| Sequences with the hand pushed to the other side of the object (per hand) | **10** | 62 | **0** | 16 | 6 | **5** |
| Ground penetration (mm) | 4.3 | **0.2** | 4.2 | **0.05** | 2.4 | **0.2** |
| Foot slide (mm/frame) | **1.0** | 2.6 | **1.0** | 2.8 | **0.9** | 1.9 |
| Ankle error (cm) | **4.3** | 5.2 | 4.6 | **3.6** | 5.1 | **4.7** |
| Joint jerk / root jerk | **201** / 56 | 273 / **14** | **238** / 62 | 258 / **19** | **140** / 35 | 193 / **10** |
| Time per frame (s, measured with 20 parallel processes) | 2.9 | **0.25** | 1.1 | **0.15** | 2.3 | **0.10** |

OmniRetarget's penetration comes almost entirely from fallbacks: sequences where the hard constraints are feasible penetrate < 2 mm, but under CoACD 53–71% of the sequences are infeasible (the fixed rest-pose fingers cannot wrap handles and edges), and after falling back to relaxed tolerances / fingers excluded / object constraints off they average 12–28 mm, 97% of its total penetration. UMR's object constraint has slack and never fails; penetration is stable at 4.5–5.7 mm. OmniRetarget has more accurate wrist positions, better contact preservation and almost never pushes the hand to the other side of the object (it has wrist keypoints and an interaction mesh; UMR has no wrist position term, see 6.10); UMR's wrist orientation is 3–4× better (OmniRetarget has no orientation term, the hand rotates freely about the forearm), ground penetration near 0, root jerk 3–4× lower, 10–20× faster, but 2–3× more foot slide and higher joint jerk. BrainCo (OMOMO / HUMOTO) gives the same ranking. OmniRetarget hard-codes 29 DoF; with hands it can only bake the hand meshes into fixed geoms (`run_omni.py --hand`). The earlier experiment "OmniRetarget on HUMOTO is better with Mixamo joints directly than via SMPL-X" (`results_humoto/omni_brainco_mx*`) was run before the baked-hand fix and has not been rerun.

---

## 6. Design decisions and ablations

Every item was verified on HUMOTO; the 20-sequence subset = the 5 worst two-handed bowl-carrying sequences + 15 random ones (`results_humoto/wrist_subset.json`).

### 6.1 Scaled objects or real size

In the scaled world the hand/object size ratio is 1.35× that of the human (`results/figures/scaling_conflict.png`). Two real-size conventions were tried, on all of HUMOTO:

| After v3 | Scaled world (adopted) | Real size, object center × s (`--scaling real`) | Real size, anchored at the wrist (`--scaling realhand`) |
|---|---|---|---|
| Contact F1 | 0.74 | 0.71 | **0.76** |
| Hand-object penetration (mm) | **0.77** | 1.05 | 0.79 |
| Whole-body object penetration (mm) | **1.8** | 2.9 | 2.6 |
| Flange position (cm) | **2.56** | 2.94 | 2.68 |
| Body joint / root jerk | **191 / 19** | 212 / 23 | 213 / 23 |

The first is worse because the whole human is × s while the object radius is not scaled, so the hand ends up 0.15 × radius closer to the surface. The second (`prepare_realsize_hand_anchor.py`: the robot-side object trajectory becomes $p' = a + (p-a)/s$, with $a$ the human wrist, the two hands weighted by fingertip-to-surface distance; a UMR patch makes the source-side contact follow the original trajectory and the robot side the new one, and the result npz gains `obj_pos_robot`) matches and slightly exceeds the scaled world on the hand metrics, but the UMR body stage compromises between "body following the × s human" and "hand points landing on a larger object", so jerk and forearm penetration get worse. Getting both would require changing the UMR body terms, not done. HiPHI's large floor objects would additionally need z anchoring.

### 6.2 The three refinement versions

v1 (s3p): first a Gauss-Newton IK places the 7 arm joints at the human wrist pose (palm-object non-penetration QP), then a 20-variable finger QP is solved, followed by a 5-frame moving average. v2: 27-variable coupled QP, contact decision with hysteresis, causal temporal terms, hand and forearm non-penetration. v3: following DexWeave (arXiv:2609.34724), adds the contact-degree-gated wrist anchor, fingertip priorities, the correction temporal term plus an absolute acceleration term, and uses the human's contact point itself as the contact target. DexWeave's hand shape initialization is a 729-entry codebook retrieval rather than AnyTeleop, its coupled refinement runs 40 iterations, the scene is not scaled by default, and there is no physics validation.

### 6.3 Stage 2 cannot be skipped

`--finger-prior rest` (v3's finger prior and initial value replaced by the rest pose): F1 0.74 → 0.48, fingertip error 3.3 → 6.3 cm, all 100 sequences get worse. The per-frame linearized QP with non-penetration constraints, starting from an open hand, closes 90° along a straight path that passes through the object, and is blocked outside. What is needed is a global hand shape initialization; the source can be swapped (codebook retrieval would also work), but it cannot be omitted.

### 6.4 Hand-size remapping is useless

v2 multiplied the human fingertip offset relative to the wrist by 1.15 before projecting onto the object surface (and in the scaled world this factor should be ratio/s = 1.35). `--v3-remap ratio` verified on the full set: F1 0.741 vs 0.739, penetration 0.84 vs 0.77 mm, per-sequence F1 win rate 49%, no difference on small objects either. The target points are all projected onto the surface in the end; spreading them out and then projecting barely changes the landing point, and the size difference is already absorbed by the wrist anchor. v3 does not remap by default.

### 6.5 Body-stage wrist orientation and the various UMR variants

Wrist orientation error (`wrist_orientation_stats.py`, all HUMOTO frames): mean 9.9°, median 6.8°, 4.8% of frames > 30°, bias only 3.5–5.8°, so not a frame convention problem; interaction frames 12°, the worst two-handed bowl-carrying sequences 40–54°. The cause is the contact term: it uses the 128 slots nearest to the object out of all 4096, with an effective weight up to 25, while the surface term has only 15 slots on the hand with weight 10; with the fingers pinned and the hand 15% longer, the cheapest solution that brings more hand surface points onto the bowl rim is to flip the whole hand. Variants on the subset:

| UMR variant | Wrist orientation / frames >30° | Wrist position (cm) | Whole-body penetration (mm) | Joint / root jerk | After v3: F1 / penetration (mm) / joint jerk |
|---|---|---|---|---|---|
| Baseline | 12.9° / 12% | 4.6 | 4.3 | 341 / 27 | **0.77** / 0.97 / 250 |
| `contact4` contact term ×4 | 14.6° / 14% | **3.9** | 9.6 | 391 / 37 | — |
| `contact0` contact term off | 7.0° / 1% | 4.9 | **2.3** | 307 / 24 | — |
| `v2` hand surface slots 15→40, position 10→30, normal 1→5 | 6.4° / 0% | 4.9 | 7.2 | 299 / 28 | 0.76 / 0.83 / 231 |
| `free` fingers not pinned | 9.9° / 2.6% | **3.7** | 2.5 | **261 / 24** | 0.76 / 1.07 / 217 |
| `flat` correspondence template with a straight robot hand | 14.1° / 9.9% | 4.6 | 4.0 | 317 / 27 | 0.75 / 0.48 / 233 |

The jerk reduction of `free` is entirely in the shoulder/elbow/wrist (410 → 256) and the waist, the legs are unchanged: the per-frame changing contact pairings on the hand previously could only be absorbed by the arm, and with free fingers they are absorbed by the fingers (finger jerk 184), but UMR's own fingers are unusable (15 slots cannot distinguish the five fingers, F1 0.28). `v2` on the full set: contact F1 0.739 → 0.723 (HUMOTO), unchanged but more penetration (HiPHI), rejected; full-set verification of `free` (`run_variant_pipeline.sh <ds> free`, followed by AnyTeleop + v3 with 12 iterations, `results_*/pipe_summary_freefinal.md`): HUMOTO contact F1 0.741 vs 0.745, hand-object penetration 0.94 vs 0.71 mm (worse), joint jerk 168 vs 191 (96% of sequences better); HiPHI F1 0.580 vs 0.577, penetration 0.63 vs 0.66 mm, joint jerk 117 vs 134 (98% better), root jerk 9.3 vs 10.3. That is, a smoother body is a stable gain; the hands pay a penetration cost on HUMOTO and are unchanged on HiPHI. The default remains the baseline body; switch to `free` when only body smoothness matters or for large-object tasks like HiPHI. The body stage cannot be tuned to get correct orientation, correct position and no penetration all at once; v3 finally corrects the orientation to 4°.

### 6.6 UMR-style dense contact term added to the third stage (2026-10-07)

v3 originally had contact targets only for the 5 fingertips. Following UMR's contact map, but with correspondences from the AnyTeleop keypoint mapping instead of learned slots: take the 15 knuckle joint centers (MCP/PIP/DIP of the four fingers, CMC/MCP/IP of the thumb) plus 4 palm points formed as linear combinations of keypoints; whenever a human-side point is < 3 cm from the object surface (hysteresis, exit at 6 cm, at least 3 frames), the corresponding point on the robot must reproduce the offset vector from the human-side point to the nearest surface point, with the offset enlarged by the robot/human hand size ratio (robot joint centers lie deeper under the skin). Implementation: `hand_pipeline.py --w-dense W --tau-dense 0.03` (`DENSE_POINTS`, `contact_targets_dense`, `Robot.dense_points/dense_jac`); `eval_pipeline.py` gains `kp_contact_recall/precision/f1` (same set of points, 3 cm) and `kp_obj_dist_err_cm`.

| HUMOTO 100 sequences | final (fingertips only) | + dense term w=50 | + dense term w=150 |
|---|---|---|---|
| Fingertip contact F1 (recall / precision) | 0.745 (0.71 / 0.86) | 0.764 (0.73 / 0.86) | **0.781 (0.76 / 0.88)** |
| Knuckle+palm contact F1 (recall / precision) | 0.528 (0.43 / 0.84) | 0.564 (0.46 / 0.86) | **0.604 (0.50 / 0.87)** |
| Fingertip-object distance error (cm) | 1.30 | 1.18 | **1.07** |
| Knuckle+palm-object distance error (cm) | 1.71 | 1.62 | **1.50** |
| Fingertip position error (cm) | 3.25 | 2.96 | **2.65** |
| Hand-object penetration (mm) / frames >5 mm | **0.71 / 0.044** | 0.83 / 0.048 | 0.89 / 0.052 |
| Whole-body object penetration, fraction of frames >5 mm | **0.106** | 0.118 | 0.120 |
| Flange position (cm) / orientation (°) | **2.54** / 3.99 | 2.67 / 4.01 | 2.80 / **3.84** |
| Joint jerk / finger jerk | 191 / **65** | 191 / 72 | 192 / 81 |

Per sequence: w=150 vs final has a fingertip F1 win rate of 78%, knuckle+palm F1 89%, fingertip-object distance 91%; penetration is better in only 27%, flange 21%. That is, contact quality improves consistently and substantially (F1 +0.036, equal to the sum of all improvements from v1 to v3), at the cost of 0.2 mm more hand-object penetration, 0.26 cm more flange error and 25% more finger jerk, with the trend growing with the weight.
| HiPHI 60 sequences | final (fingertips only) | + dense term w=50 | + dense term w=150 |
|---|---|---|---|
| Fingertip contact F1 (recall / precision) | 0.577 (0.53 / 0.76) | **0.595 (0.55 / 0.78)** | 0.587 (0.54 / 0.76) |
| Knuckle+palm contact F1 (recall / precision) | 0.449 (0.33 / 0.84) | 0.503 (0.38 / 0.86) | **0.564 (0.44 / 0.88)** |
| Fingertip-object distance error (cm) | 1.74 | 1.61 | **1.52** |
| Knuckle+palm-object distance error (cm) | 2.33 | 2.17 | **2.06** |
| Hand-object penetration (mm) / frames >5 mm | **0.66 / 0.016** | 0.85 / 0.042 | 1.17 / 0.063 |
| Whole-body object penetration, fraction of frames >5 mm | **0.236** | 0.272 | 0.305 |
| Flange position (cm) / orientation (°) | **3.04** / 5.99 | 3.19 / **5.84** | 3.52 / 5.85 |
| Joint jerk / finger jerk | 134 / **88** | 134 / 96 | 134 / 105 |

Conclusion: the dense contact term does fill in the knuckle and palm contacts; knuckle+palm F1 rises by 0.04–0.08 on both datasets, fingertip F1 follows with +0.02–0.04, and fingertip position error drops by 0.3–0.9 cm; the cost is that penetration and flange error get monotonically worse, more so with higher weight, and HiPHI's large objects are more sensitive to this (at w=150 hand-object penetration 0.66 → 1.17 mm, whole-body penetration frame fraction 0.24 → 0.31). w=50 is a compromise acceptable on both datasets: contact F1 +0.02, knuckle+palm F1 +0.04–0.05, hand-object penetration +0.15 mm (still within 1 mm), flange +0.15 cm. Per-sequence win rates of w=50 vs final on HiPHI: fingertip F1 67%, knuckle+palm F1 98%, fingertip-object distance 75%; hand-object penetration better in only 13%, flange 32%. `pipe_final` has not yet been switched to the version with the dense term; the lanes are in `results_*/pipe_v3_dense{50,150}`, tables in `pipe_summary_dense.md`.


### 6.7 Multiple objects (HUMOTO, step one: handheld objects, 2026-10-08)

Of the 100 HUMOTO sequences, 41 have multiple objects and in 36 the hands touch more than one. Previously only the main object (the one with the most contact frames) took part in retargeting; the other objects had neither contact targets nor non-penetration constraints in the hand stage. Step one handles only handheld objects (`handheld` flag, meshes and CoACD already available):

- `prepare_humoto_multi.py`: besides the main object, any handheld object that either hand is within 2 cm of for more than 10 frames is selected; assets are copied into `data_humoto/objects/`, and `data_humoto/multi_objects.json` is written (35 sequences gain 56 objects, at most 5).
- `compare.augmented_model` accepts a list of objects, one free body each; `hand_pipeline.py --multi`: v3's fingertip targets, contact degree α and dense term all take **the nearest object** for each point in each frame, and all objects enter the non-penetration constraints; the result npz gains an `objects` key. The robot-world pose of the extra objects = HUMOTO trajectory × s.
- `eval_pipeline.py --multi`: hand metrics are computed over all objects (distance of a point to the nearest object, penetration against any object); both lanes use the same evaluation. viser and `render_compare.py` draw all objects in the result.

| 35 multi-object sequences (evaluated over all objects) | original final (main object only) | multi-object |
|---|---|---|
| Fingertip contact F1 (recall / precision) | 0.679 (0.60 / 0.87) | **0.767 (0.71 / 0.90)** |
| Knuckle+palm contact F1 | 0.550 | **0.578** |
| Fingertip-object distance error (cm) | 1.69 | **1.30** |
| Hand-object penetration (mm) / frames >5 mm | 4.25 / 0.29 | **0.79 / 0.034** |
| Flange position (cm) / orientation (°) | 2.01 / **3.7** | **1.86** / 4.9 |
| Joint jerk / finger jerk | **71 / 55** | 82 / 85 |

Per-sequence win rates: contact F1 86%, hand-object penetration 97%, flange position 71%. The other 65 single-object sequences are bit-identical to the original final. The cost is higher finger jerk and joint jerk: when the nearest object switches between two objects the target jumps (e.g. the hand moving from the basket handle to the bowl). HUMOTO's `pipe_final` now points to `pipe_v3_multi`; numbers over all 100 sequences: F1 0.740 → 0.771, hand-object penetration 2.05 → 0.85 mm (evaluated over all objects). (Step-one numbers; steps two and three below.)

#### Step two: furniture enters the hand stage; step three: multi-object UMR body stage (2026-10-08)

- **Furniture assets**: tables, chairs, shelves, side tables, carts, trash cans and floor lamps take their meshes from each sequence's GLB (`humoto_io.load_object_mesh`, scale 1), with CoACD via `hand_bench/prepare_humoto.export_object` (threshold 0.04, ≤32 pieces). For thin-walled parts CoACD fills the cavities (spurious fill 7–10% of the bounding box; a hand reaching into a shelf or a bucket is blocked), so shelves and side tables use a "surface patch" decomposition instead: faces subdivided to ≤5 cm, then k-means on face centers into 96–128 clusters, each cluster's convex hull taken (flat clusters thickened by 5 mm along the normal); spurious fill drops to 2.6–2.7%; the trash can uses 400 surface patches; the cart keeps 64 CoACD pieces (5.9%). `prepare_humoto_multi.py` now also puts all furniture in the sequence into `multi_objects.json` (41 sequences have extra objects, 109 in total; the handheld-only list of step one is in `multi_objects_handheld.json`). The hand stage gained a bounding-sphere prefilter (`Robot.near_obj_geoms`); otherwise 400-piece furniture makes the pairwise distance queries the bottleneck.
- **Multi-object UMR** (`prepare_umr_multi.py` → `data_humoto/umr_multi/`, each extra object written as `<obj>.xml` + `prop_<obj>.csv`, material names renamed per object to avoid clashes when merging; `run_umr.py --scaling multi` + `configs/umr_g1_wuji_multi.json`): UMR is patched (`solver.object_multi`): all objects are sampled with 1024 surface points each and concatenated into one point set, so the contact term's nearest neighbor naturally picks the nearest object; the penetration model has one free body per object, with each pose written in every frame. The caps on contact points and penetration pairs are raised from 128 to 512, otherwise chairs and tables crowd out the hand-cup constraints. UMR is run only on the 41 multi-object sequences; the other 59 have inputs identical to the baseline and reuse the baseline body directly.
- **Evaluation** gains `scene_pen_mm / scene_pen_frac`: the per-frame deepest penetration of the whole robot (excluding world geometry) against all objects (the existing `obj_pen_*` only looks at the main object).
- A bug was fixed along the way: `run_umr.py` previously recorded `object_mesh_scale` as s only for `--scaling umr`, and 1.0 for all other variants; now only the real-size variants (`real`/`realhand`/`omni`) record 1.0. Earlier `--variant` experiments all used `--scaling umr` and are unaffected.

| 41 multi-object sequences, evaluated over all objects | main object only | step one, handheld objects | **step two + furniture (final)** | step three + multi-object UMR |
|---|---|---|---|---|
| Fingertip contact F1 | 0.694 | 0.769 | **0.770** | 0.757 |
| Knuckle+palm contact F1 | 0.552 | **0.575** | 0.569 | 0.561 |
| Hand-object penetration (mm) / frames >5 mm | 5.05 / 0.31 | 2.14 / 0.11 | 1.18 / 0.056 | **0.61 / 0.040** |
| Whole-body-scene penetration (mm) / frames >5 mm | 9.92 / 0.57 | 5.61 / 0.26 | 4.18 / **0.19** | **3.06** / 0.20 |
| Flange position (cm) / orientation (°) | 1.99 / **3.8** | **1.86** / 4.9 | 2.01 / 5.3 | 2.08 / 5.5 |
| Joint jerk / root jerk | **82 / 5.6** | 91 / 5.6 | 95 / 5.6 | 111 / 7.1 |
| Finger jerk | **59** | 85 | 96 | 96 |

The UMR body stage by itself (without the hand stage): whole-body-scene penetration 12.6 → 6.7 mm, hand-object penetration 7.3 → 3.5 mm, but joint jerk 106 → 147 and wrist error 2.6 → 3.2 cm. Per sequence: step two vs step one, hand-object penetration better in 73%, scene penetration in 80%, F1 unchanged; step three vs step two, hand-object penetration better in 61%, scene penetration in 56%, F1 49%, joint jerk better in only 12%. So step two is a stable gain and is set as the HUMOTO final (`results_humoto/pipe_final -> pipe_v3_multi2`); step three halves penetration again at the cost of joint jerk +16% and root jerk +27%, and is kept as an option in `pipe_v3_multi3` / `umr_wuji_multi`, to be used when penetration matters more for physics simulation. The jerk presumably comes from UMR's contact/penetration active set switching more often per frame with multiple objects. All tables in `results_humoto/pipe_summary_multi23.md`, figures in `results_humoto/figures/*_multi23.png`.

### 6.9 OmniRetarget object collision: CoACD convex pieces vs the native whole mesh (2026-10-09)

The object in holosoma's bundled example is a box, whose collision uses the whole mesh as a single geom, which MuJoCo treats as its convex hull. Our `run_omni.py` changes this to one geom per CoACD convex piece (shared with UMR). The ablation `run_omni.py --native-collision` (model name suffixed `_native`, does not overwrite the original results) was run on the 6 least convex OMOMO objects (large and small tables, plastic box, trash can, wooden chair, white chair), 3 sequences each, Wuji hand, UMR scaling, with both groups evaluated on the CoACD geometry (`compare.py --out _native`). The volume the whole-mesh convex hull fills in beyond CoACD: tables 84%, plastic box 70% of the bounding box, trash can and wooden chair 35%, white chair 21%.

After the baked-hand fix (2026-10-15) both collision variants were rerun on all three datasets in full (lanes `omni_wuji_umrscale` / `omni_wuji_umrscale_native`, `results*/summary_3ds.md`), all evaluated on the CoACD geometry:

| G1 + Wuji | OMOMO CoACD | OMOMO convex hull | HUMOTO CoACD | HUMOTO convex hull | HiPHI CoACD | HiPHI convex hull |
|---|---|---|---|---|---|---|
| Object penetration (mm) / frames >5 mm | 10.1 / 42% | **2.1 / 9%** | 10.4 / 60% | **4.1 / 21%** | 13.5 / 47% | **10.5 / 33%** |
| Sequences needing fallback | 359 | **39** | 71 | **26** | 32 | **22** |
| Ground penetration (mm) | 4.3 | **1.6** | 4.2 | **2.0** | **2.4** | 3.8 |
| Hand-object contact preservation | **0.97** | 0.83 | **1.00** | 0.98 | 0.96 | **0.97** |
| Wrist position error (cm) | **2.2** | 3.5 | **1.6** | 2.1 | **2.9** | 3.4 |
| Wrist orientation error (°) | **32** | 37 | **39** | 41 | **47** | 49 |
| Sequences with the hand pushed off (per hand) | **10** | 59 | 0 | 0 | 6 | **4** |
| Joint / root jerk | **201 / 56** | 232 / 61 | **238 / 62** | 244 / 64 | **140 / 35** | 146 / 36 |
| Time per frame (s) | 2.9 | **1.1** | 1.1 | **0.7** | 2.3 | **1.2** |

The convex hull makes the hard constraints far more often feasible (OMOMO fallbacks 359 → 39), hence very low penetration and more than twice the speed, but it effectively swaps in a different object: concavities are filled in and the hand cannot get in. On OMOMO contact preservation drops 0.97 → 0.83 (floor lamp 1.00 → 0.03, clothes rack 0.98 → 0.46), wrist error 2.2 → 3.5 cm, sequences with the hand blocked outside the true surface 10 → 59, orientation and jerk also worse. HUMOTO's objects are small and nearly convex, so the two differ little; HiPHI's large furniture still needs 22 fallbacks with the convex hull and penetration only drops to 10.5 mm, and the bench and clothes rack penetrate more with the convex hull. For human-object interaction retargeting what matters is that the hand truly touches the object, so the main comparison keeps CoACD and the convex hull is reported alongside as the "holosoma default configuration" control column. Figure `results/figures/sub17_woodchair_000_native.png` (re-rendered after the fix): with the convex hull the hand hovers outside the chair back.

### 6.10 OMOMO: fixed fingers push the hand to the other side of the object (2026-10-10)

In `sub16_whitechair_002` (both hands gripping the top bar of the chair back), both hands of UMR's Wuji lane are pushed 13–14 cm away from the human wrist, to the other side of the chair back, for the entire time the human holds the chair back (before the baked-hand fix OmniRetarget showed the same phenomenon; after the rerun its wrist offset is 5 cm with no pushed-off frames, see Section 8). The cause is not the source data: the human wrist is 4 cm from the chair and the flattened SMPL-X fingers do not pass through the chair back either. The cause is **the fixed rest-pose fingers + the robot-object hard constraint**: the top bar of the chair back is only 2 cm thick, the human holds the bar between thumb and four fingers, the robot fingers are pinned into a slightly bent "hook", the bar has to land inside the hook, the linearized non-penetration constraint cannot reach that position from outside, and the solver can only move the whole hand to the other side of the bar (the hook goes over it from that side, or the palm rests against the far side). UMR's hand also penetrates 6 mm at the same time. `wrist_gap_stats.py` statistics (fraction of frames, among those where the human wrist is < 10 cm from the object, in which the robot wrist is > 12 cm from the human wrist; 535 sequences with contact):

| lane | mean wrist offset (cm) | fraction of frames > 12 cm | sequences with one hand pushed off for more than half the duration |
|---|---|---|---|
| omni_brainco | 5.5 | 0.01 | 6 |
| omni_wuji | 5.6 | 0.02 | 10 |
| umr_brainco | 5.9 | 0.04 | 23 |
| umr_wuji | 7.4 | 0.09 | 62 |

(OmniRetarget is the rerun after the baked-hand fix of 2026-10-15, `results/wrist_gap_3ds.csv`; sequence counts are per hand.)

Wuji is worse than BrainCo (longer hand, deeper hook). UMR is much worse than OmniRetarget (62 vs 10): OmniRetarget has a wrist keypoint cost and an interaction mesh, so moving the hand to the other side is expensive, and when the hard constraints are infeasible it walks the fallback chain and relaxes the constraints (of the 57 sequences where UMR is pushed off, OmniRetarget used a fallback in 44, and on those its object penetration is 14.3 mm vs UMR's 7.2 mm), i.e. it trades penetration for wrist position; UMR has no wrist position term, its surface / contact terms are nearly equivalent on both sides of the bar, once the per-frame linearized non-penetration with slack has pushed the hand across the surface no term pulls it back, and it never relaxes collisions. List and visualization of the pushed-off sequences: `reports/pushed_sequences_umr_wuji.md`. They concentrate on thin bars and thin plates: chairs, tables, box edges, the vacuum cleaner pole. A more precise geometric analysis (penetration of the rest-pose hand placed at the human wrist pose, a translation sweep along the hand axis, the approach during frames 8–20) is in `reports/whitechair_002_wrong_side.md`: it is the palm block that presses directly on the bar, the fixed fingers block the way out, there is no collision-free position within ±16 cm along the hand axis; the fingers hit the chair back 0.3 s before the human wrist does, and from that moment the hard constraint slides the hand along the chair back to the other side. **For datasets with a hand stage (HUMOTO / HiPHI)**: the fixed fingers of the body stage are only placeholders, so excluding them from the object hard constraints is reasonable; fingers and penetration are handled by AnyTeleop + v3 (see the full-set verification below). **Not applicable to OMOMO**: OMOMO has no finger data and no subsequent hand stage, the body-stage output is the final result, and finger collision must be kept, otherwise no stage can repair fingers that penetrate the object; the two variants below serve on OMOMO only as ablations to locate the cause, not as a fix. OmniRetarget uses `--fingers-free` (palm and body remain constrained), UMR uses `configs/umr_g1_wuji_nofingercol.json` (`robots/g1_29dof_wuji_hand2_nofingercol.xml`, finger geoms do not collide, `run_umr.py --variant nofingercol`). On that sequence the wrist offset of both methods drops from 13–14 cm to 5 cm and the frames > 12 cm go to zero; evaluated with the full collision model, UMR's hand-object penetration drops from 6.4 mm to 1.9 mm (the fingers are no longer pushed into the object), OmniRetarget's is 0. The cost is that fingers may pass through the object (0 frames > 5 mm in this sequence), which is handled by the hand stage only on datasets that continue with AnyTeleop + v3; OMOMO's official results keep finger collision unchanged.

**The fix for OMOMO (keeping finger collision)**: letting UMR solve the fingers as free DoFs (`--variant free`, collision as usual) does not help (wrist offset 12.6 cm, 76% of the frames still pushed off), because UMR has no term that would open the fingers to let the bar in. What works is two steps: body stage with finger collision off (the hand lands at the human wrist) → `omomo_hand_repair.py` re-solves frame by frame with the full collision model. Two re-solves were tried:

- Fingers only (default): per frame and per hand the 20 finger joints, rest-pose prior + previous-frame term + hand/object non-penetration with slack + joint limits (`refine_fingers` with the contact targets removed), arms unchanged. Full set (`results/run_omomo_repair_full.sh` → `pipe_handrepair`, `wrist_gap_omomo_repair.csv`): wrist offset 6.0 cm, but penetration under finger collision 6.48 mm, worse than the original lane (5.79): with the arm pinned the palm block is already pressed into the box / trash can, and moving the fingers alone cannot push it out.
- Arm + fingers coupled (`--coupled`, **adopted**): per frame and per hand the 7 arm + 20 finger joints are solved together, wrist position / orientation anchored to the human hand's SMPL-X canonical hand frame (`human_hand_frames`), rest-pose finger prior, previous-frame + acceleration terms, hand and forearm non-penetration against the object with slack, no contact targets (`refine_coupled`, w_tip=0, w_pose=50, w_rot=5, 8 iterations, about 6 s per sequence). Full set (`results/run_omomo_repair_coupled.sh` → `pipe_handrepair_coupled`, `wrist_gap_omomo_coupled.csv`; body metrics `summary_handrepair_coupled.md` generated by `compare.py`, to be added after the run finishes):

| OMOMO 535 sequences with contact (same statistics) | umr_wuji (original final) | two-step (coupled) |
|---|---|---|
| Mean wrist offset (cm) | 7.4 | **6.7** |
| Fraction of pushed-off frames (> 12 cm) | 0.09 | **0.01** |
| Sequences with one hand pushed off for more than half the duration | 62 | **5** |
| Hand-object penetration (mm) / frames > 5 mm | 5.79 / 0.44 | **1.23 / 0.06** |

Per-object penetration drops across the board (small box 4.4 → 0.1, large box 5.1 → 0.4, tables 7.1–7.3 → 0.5–0.6, chairs 5.5–6.6 → 1.1–2.7, trash can 6.2 → 2.7 mm); pushed-off frames remain only for the wooden chair 0.05 and the white chair 0.03. The remaining 5 sequences (`sub16_whitechair_009/037`, `sub17_woodchair_001/004/046`) are cases where the finger-collision-off body stage itself already pressed the hand into the chair back, and the coupled re-solve can only pull one hand back. White chair 002: wrist offset 5.5 cm, pushed-off frames 0, penetration 0.5–1.1 mm. This is now the OMOMO final (`results/pipe_final -> pipe_handrepair_coupled`; `run_final_pipeline.sh omomo` follows this chain: `BODY_VARIANT=nofingercol` → `omomo_hand_repair.py --coupled`). Visualization: 8095 (original lane / finger collision off / two-step final / OmniRetarget).

Does this leave a large initial penetration for the subsequent hand stage? On the HUMOTO 20-sequence subset we tried "body stage with finger collision off → AnyTeleop → v3 with 12 iterations" (lanes `umr_wuji_nofingercol` / `pipe_v3_nfcsub`, evaluated with the full collision model):

| 20-sequence subset | body stage (finger collision) | body stage (collision off) | v3 (body with finger collision) | v3 (body with collision off) |
|---|---|---|---|---|
| Hand-object penetration (mm) / frames >5 mm | 2.8 / 0.22 | 5.9 / 0.40 (deepest 16 mm) | 0.89 / 0.060 | **0.73 / 0.049** |
| Flange position (cm) / orientation (°) | 5.4 / 12.9 | 3.5 / 6.0 | 2.92 / 3.81 | **2.87 / 3.44** |
| Fingertip contact F1 | 0.27 | 0.33 | 0.773 | **0.784** |
| Whole-body object penetration (mm) | 4.3 | 8.2 | 1.8 | **1.4** |
| Joint jerk | 341 | 225 | 251 | **207** |

The body stage's initial penetration does double, but v3 removes all of it, including the deepest 16 mm; and because the wrist pose is more accurate (flange 5.4 → 3.5 cm, orientation 12.9° → 6.0°), the final result is better on every metric, with a per-sequence F1 win rate of 80%. Reason: the third stage's non-penetration is a per-frame linearized constraint, so a few centimeters of penetration are pushed out in one or two iterations; what truly cannot be repaired is the ten-odd-centimeter error of a wrist pushed to the other side of the object, and finger collision off avoids exactly that. Full-set verification (2026-10-13, `run_nfc_full.sh <ds>`: UMR `--variant nofingercol` → AnyTeleop → v3 with 12 iterations, HUMOTO with multiple objects, evaluated with the full collision model and all objects, lanes `umr_wuji_nofingercol` / `pipe_v3_nfcfull`, table `pipe_summary_nfcfull.md`):

| | HUMOTO final (finger collision) | HUMOTO finger collision off | HiPHI final (finger collision) | HiPHI finger collision off |
|---|---|---|---|---|
| Fingertip contact F1 (recall / precision) | 0.771 (0.71 / 0.91) | **0.799 (0.74 / 0.92)** | **0.577** (0.53 / 0.76) | 0.572 (0.52 / 0.75) |
| Knuckle+palm contact F1 | 0.553 | **0.560** | 0.449 | 0.450 |
| Fingertip-object distance error (cm) | 1.33 | **1.19** | 1.74 | **1.73** |
| Hand-object penetration (mm) / frames >5 mm | 1.00 / 0.059 | **0.89 / 0.046** | **0.66 / 0.016** | 0.81 / 0.031 |
| Whole-body-scene penetration (mm) / frames >5 mm | 2.95 / 0.163 | **2.77 / 0.139** | **3.06 / 0.236** | 3.38 / 0.267 |
| Flange position (cm) / orientation (°) | 2.54 / 4.58 | **2.42 / 3.97** | 3.04 / 5.99 | **3.01 / 5.87** |
| Joint jerk / root jerk | 197 / 19.4 | **167 / 16.2** | 134 / 10.3 | **113 / 8.7** |
| Body-stage hand-object penetration (mm) | 4.7 | 8.2 | 3.1 | 13.0 |

The comparison of the body stage itself on the three datasets (including the four-way comparison against OmniRetarget CoACD / convex hull on OMOMO, wrist orientation, number of pushed-off sequences) is in `reports/umr_finger_collision_3ds.md` (2026-10-17): finger collision off lowers the wrist position error by 0.8–1.0 cm, the wrist orientation by 3–4°, the pushed-off sequences 57 → 5 / 11 → 0 / 4 → 2, and joint jerk by a third, at the cost of 8–17 mm of body-stage finger penetration, so a hand stage must follow. On HUMOTO it is better across the board (per sequence: F1 69%, flange 73%, joint jerk 94% of sequences better) and is now the HUMOTO final (`results_humoto/pipe_final -> pipe_v3_nfcfull`, `run_final_pipeline.sh` defaults to `BODY_VARIANT=nofingercol` for HUMOTO). On HiPHI it is a trade-off: clearly smoother (joint jerk −16%, 97% of sequences), slightly better wrist pose, but contact F1 −0.005, hand-object penetration 0.66 → 0.81 mm, scene penetration +0.3 mm. HiPHI's objects are large; with finger collision off in the body stage the whole hand gets pressed into boxes and chairs (body-stage penetration 13 mm, deeper than HUMOTO's 8 mm), and v3 cannot push all of it out. HiPHI keeps the original final; use `BODY_VARIANT=nofingercol` when smoother trajectories are needed.

### 6.11 Miscellaneous

- 12 iterations vs 4: small consistent improvement on HUMOTO, unchanged on HiPHI, adopted.
- Raising UMR's hand surface weights only lowers the wrist orientation from 10° to 6°; the systematic 4 cm wrist pull-back stays.
- The Wuji hand is mounted with the official `_with_mount` (palm normal −y, same convention as Dex3 / Inspire); the rew35860/holosoma fork uses Wuji Hand 1 with the palm rotated by 90°.
- Joint limits: hard constraints in both OmniRetarget and UMR; also in v3's QP.
- UMR's Clarabel occasionally reports `InsufficientProgress`: the default 1e-8 tolerance over-iterates and breaks down numerically; relaxing to 1e-6/1e-7 and accepting the iterate when it is feasible suffices.

---

## 7. Data preparation

- **OMOMO** (`prepare_omomo.py`): SMPL-X parameters and `obj_com_pos / obj_rot` from the pickle; objects use the centered mesh + the median `obj_scale` per object type, CoACD decomposed; height from the zero-pose template bounding box (sub16 1.723 m, sub17 1.737 m). Test set 602 sequences, no finger data, the pipeline stops at stage 1.
- **HUMOTO** (Mixamo GLB): `../hand_bench/humoto_to_bvh.py` exports BVH → `hiphi2smplx --skeleton-map mixamo` fits SMPL-X (16-dim betas fitted only once, `betas_humoto.npy`, height 1.548 m, the other sequences use the fixed shape + body/finger IK) → `prepare_humoto.py` converts to Z-up and selects the handheld object with the most contact frames as the main object. 100 sequences, 32 256 frames, 41 object types.
- **HiPHI**: from the 7025 sequences of the HOI subset, 5 sequences per each of 12 object categories (ranked by the fraction of frames with a fingertip within 3 cm of the object, `../hand_bench/data_hiphi/selection.json`), 60 sequences in total with 109 788 frames, from 13 actors; SMPL-X fitted by `hiphi2smplx --fit-shape-hands` with 16-dim betas **per sequence** (not shared per actor), 90 Hz subsampled by 3. `prepare_hiphi.py` writes `data_hiphi/`. Scaling up is cheap (all 7025 sequences in 3 minutes); the pipeline takes about 3.5 hours per 100 sequences.
- Hand keypoints, object contact labels and AnyTeleop results are all in `../hand_bench/data*/seqs/`, `results*/anyteleop/`.

---

## 7.1 Changes to upstream code

- **UMR** (submodule `../UMR`, commit pointer unchanged): the changes to `scripts/retarget_smpl_to_humanoid_surface_vector.py` are kept as a patch in `patches/umr_multi_object_retarget_prop.patch`: an independent robot-side object trajectory `retarget_prop_<obj>.csv` (real size anchored at the hand, 6.1) and multi-object loading `solver.object_multi` (6.7). After cloning run `git -C ../UMR apply ../omomo_bench/patches/umr_multi_object_retarget_prop.patch`; without the patch the default flow runs as usual, only `--scaling realhand|multi` needs it.
- **holosoma** (submodule): unchanged; all modifications are runtime patches in `omni_entry.py`.
- **hiphi2smplx** (outside the repo, github.com/noitom-robotics/hiphi2smplx @ 65e5131): the local changes are exported as `patches/hiphi2smplx_mixamo_hands_hoi.patch` (just `git apply`), including the Mixamo skeleton mapping (`--skeleton-map mixamo`, `MIXAMO_MAP`), `load_betas` keeping all 16 dims, joint shape + finger fitting (`shape_fitting.py`, `--fit-shape-hands`), finger IK (`hand_ik.py`), HiPHI HOI refinement (`hoi_refine.py`, `hoi_cli.py`).

## 8. Known issues / notes

- **OmniRetarget baked-hand bug (found 2026-10-14, all OmniRetarget lanes affected, all rerun on 2026-10-15/16)**: `run_omni.py bake_hands` writes the hand meshes at the finger rest pose as fixed geoms into `wrist_yaw_link`, using MuJoCo's compiled `geom_xpos/xmat`. The compiled geom coordinates already include MuJoCo's centering of the mesh (`mesh_pos/mesh_quat`), and on reloading the raw STL is centered again, so the translation/rotation is applied twice: for Wuji each hand mesh piece is off by 1.35 cm on average, 3.7 cm at most, orientation off by up to 171°; for BrainCo 1.41 cm on average, 4.4 cm at most, 122°. Palm and fingers scatter into fragments (`reports/figures/omni_hand_bake_bug.png`, left two columns old bake, right two columns fixed). Affected: `results/omni_{wuji,brainco}_umrscale`, `results_humoto/omni_*`, the CoACD vs native whole-mesh ablation (6.9), the `--fingers-free` experiment, all UMR vs OmniRetarget comparison tables (the omni rows of 5.3, 6.5, 6.10) and the scripts that evaluate/render OmniRetarget lanes with `compare.omni_model`; the hand-object collisions and the fallback chain during OmniRetarget solving (thin bar "gripped between thumb and four fingers") also happened on this misplaced hand. UMR lanes, hand_pipeline and the finals of the three datasets do not use the baked model and are unaffected. The fix is in `bake_hands` (compensating by `mesh_pos/mesh_quat`; after the fix the difference to the articulated URDF is < 0.001 mm; the output carries a `BAKE_MARK` comment, and `--setup` automatically rebuilds templates and per-object xml files lacking that mark). **Fully rerun on 2026-10-15/16** (OMOMO 602 sequences for each of the two hands, HUMOTO 100 sequences for each of the two hands, HiPHI Wuji 60 sequences, the convex hull ablation on the three datasets; old results kept as `*_oldbake`, old tables as `summary_oldbake.md` etc.); the numbers in 5.3 / 6.9 / 6.10 and the figures and videos of the white chair report have been updated. Before the fix OmniRetarget was also pushed to the other side of the chair back on the white chair sequence; after the fix this disappears.
- mujoco ≥ 3.14 leaves the holosoma_retargeting joints motionless (pybind enum comparison issue); `hsretargeting` pins `mujoco==3.3.5`.
- holosoma upstream's `no-activate-obj-non-penetration` is a no-op; `omni_entry.py` does the real filtering; `task_name.split("_")[0]` looks up the height dictionary, non-OMOMO names fall back to `--subject`.
- The `robot_xml` in UMR result npz files is a per-sequence temporary file and gets overwritten by the run for the other hand; all scripts rebuild the model from the `hand` key; object poses are not stored and are rebuilt from source × `smpl_scale` (the `realhand` lane uses `obj_pos_robot`).
- `eval_pipeline.py` without `--out` overwrites the default `pipe_metrics.csv`.
- In the evaluation, trimesh's `on_surface` on a 90k-face mesh eats 60 GB of memory; now chunked; the model cache is cleared per sequence, BLAS threads pinned to 1.
- `pkill -f` / `pgrep -f` in the same command line match their own shell; use the bracket pattern.
- viser used to crash when switching sequences (vertices cleared during loading while the playback thread was still rendering); fixed with a lock and deferred handle swap.
