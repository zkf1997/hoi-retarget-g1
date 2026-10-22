# hand_bench — AnyTeleop vs TopoRetarget on HUMOTO / HiPHI (G1 + BrainCo Revo2 / G1 + Wuji Hand 2)

> Since 2026-10-09 this directory is renamed from `humoto_bench` to `hand_bench`: it is the hand retargeting benchmark, and the HUMOTO and HiPHI datasets share the same scripts (`BENCH_DATASET`); the old name only reflected the first dataset that was integrated.
Retargets the **hand motion** of HUMOTO (human-object interaction, Mixamo skeleton with 21 hand keypoints + object 6DoF trajectories, 30 fps) with
**AnyTeleop** (the vector optimizer of `dex-retargeting`, plus a DexPilot variant) and **TopoRetarget** (`third_party/toporetarget`,
an unofficial implementation of Wu et al. 2026) to the two dexterous hands of G1, outputs unified metrics and plays the results side by side in viser. Both methods are **hand** retargeting:
the robot hand floats at the human wrist pose (TopoRetarget additionally optimizes the hand base pose); the torso is not retargeted.

```
hand_bench/
├── humoto_io.py          HUMOTO GLB parsing: Mixamo 65-joint FK, 21 keypoints per hand (MediaPipe order), object poses, GLB-embedded object meshes
├── make_hand_urdfs.py    cut the hand subtree out of the two G1 URDFs in urdf/ → urdf/g1_*/hand_{left,right}.urdf (root = {side}_wrist_yaw_link)
├── hand_model.py         MuJoCo hand model: active joints / mimic, 21 robot keypoints, FK, joint limits, hand size
├── prepare_humoto.py     sequence selection + data/seqs/<seq>.npz + handheld object meshes with CoACD convex decomposition      [handbench env]
├── run_anyteleop.py      dex-retargeting (vector = AnyTeleop; --optimizer dexpilot)                                            [handbench env]
├── run_toporetarget.py   TopoRetarget pipeline (bone-direction init + interaction-mesh Laplacian optimization)                 [handbench env]
├── compare.py            metrics → results/metrics.csv, results/summary.md                                                   [handbench env]
├── viser_compare.py      viser three lanes: HUMOTO human+objects | method A robot hands | method B robot hands                 [handbench env]
├── bench_common.py       shared I/O (hand frame, object poses, result read/write)
├── data/     (git-ignored)  seqs/*.npz, objects/<obj>/{visual.obj, coacd_*.obj}, selection.json, handheld_objects.json
└── results/  (git-ignored)  anyteleop/, anyteleop_dexpilot/, toporetarget/ containing <hand>/<seq>__<side>.npz; metrics.csv; summary.md
```

Environment: `conda create -n handbench python=3.11 && pip install mujoco==3.3.5 dex-retargeting torch(cpu) trimesh viser coacd pyyaml pandas`
(`dex-retargeting 0.5.0` brings pinocchio/nlopt). `third_party/toporetarget` is `git clone https://github.com/0iui0/toporetarget` (commit 8c7fbcb, git-ignored).

## One-shot workflow

```bash
conda activate handbench; export MUJOCO_GL=disable
python make_hand_urdfs.py                                  # 4 hand-only URDFs
python prepare_humoto.py --max-seqs 100 --per-object 3     # 102 sequences (at most 3 per handheld object, ranked by contact-frame fraction), 52 objects, ~11 min
python run_anyteleop.py --hand brainco --workers 16;  python run_anyteleop.py --hand brainco --optimizer dexpilot --workers 16
python run_anyteleop.py --hand wuji    --workers 16;  python run_anyteleop.py --hand wuji    --optimizer dexpilot --workers 16
python run_toporetarget.py --hand brainco --workers 22;  python run_toporetarget.py --hand wuji --workers 22     # ~10–25 min each
python compare.py --workers 16                             # → results/summary.md
python viser_compare.py --hand brainco --methods anyteleop toporetarget      # http://localhost:8080
```

## Data (`humoto_io.py`, `prepare_humoto.py`)

- HUMOTO has one GLB per sequence: a Mixamo skeleton under `Armature` (scale 0.01, bone lengths in cm); the hand bones `Thumb1..4 / Index1..4 / …` are exactly the
  MediaPipe 21-point layout (0 wrist + 5 fingers × 4); each object is a root node with per-frame TRS animation (meters). All 735 sequences share the same skeleton.
  Constant channels have only 2 keyframes; everything is resampled onto the 30 fps grid. glTF Y-up → Z-up.
- **Object meshes come from the GLB-embedded copy** (`humoto_objects_0805/*.obj` and the GLB node axis conventions disagree; the embedded mesh is unambiguous). The mesh of a given object is identical across sequences.
- **Handheld objects**: longest bounding-box edge ≤ 0.55 m (tables, chairs, shelves, sinks etc. are excluded). For each hand and frame, the nearest handheld object and the minimum keypoint-to-surface distance are recorded.
- Sequence selection: sequences containing handheld objects are ranked by "fraction of frames in which either hand is < 2 cm from the object", at most 3 per object type, 102 sequences in total, 32 597 frames, 46 object types actually grasped.
- Collision bodies: CoACD (threshold 0.05, ≤ 16 pieces); the original mesh is used for visualization.

## Coordinate conventions

- **Canonical hand frame of the human hand** (`humoto_io.hand_frame`): origin at the wrist joint; x = wrist → middle MCP; z = pinky MCP → index MCP (thumb side, orthogonalized); y = z × x.
  This matches the G1 wrist convention shared by the two hands in `urdf/`: +x distal, +z thumb side, palm facing +y (right hand) / −y (left hand), so human keypoints transformed into this frame can be fed directly to the robot hand.
- **Robot 21 keypoints** (`hand_model.py`): 0 = flange (BrainCo `*_base2_link` / Wuji `*_mount`, both at x = 0.0415); thumb CMC/MCP/IP/TIP and
  four-finger MCP/PIP/DIP/TIP are taken from the corresponding joint body origins and the `*_tip` links; BrainCo fingers have no DIP joint, DIP := midpoint of the distal phalanx.
- The flange is placed at the human wrist joint (wrist → middle MCP: human 9.3 cm, robot flange → MCP 10.3 cm). **Hand size ratio** = robot/human middle-finger chain length (wrist → MCP → PIP → DIP → tip, independent of curl):
  BrainCo ≈ 1.00, Wuji ≈ 1.15.
- **Active DoFs**: BrainCo 6 per hand (thumb metacarpal + proximal, four-finger proximal; the 5 distal joints follow via URDF `<mimic>`, thumb 1.0×, four fingers 1.155×, matching the real hardware);
  Wuji 20 per hand (all).

## The two methods

| | AnyTeleop (`run_anyteleop.py`) | TopoRetarget (`run_toporetarget.py`) |
|---|---|---|
| Implementation | `dex-retargeting 0.5.0` `VectorOptimizer` (`--optimizer dexpilot` gives `DexPilotOptimizer`) | `third_party/toporetarget` (unofficial reproduction) `TopoRetargetPipeline` |
| Input | 5 human wrist → fingertip vectors × size ratio | 21 human points (unscaled) + object mesh + object pose (all in the canonical hand frame) |
| Objective | Huber(robot flange → fingertip vector − target), + regularization toward the previous frame; SLSQP, joint limits as constraints; mimic handled by `MimicJointKinematicAdaptor` | Stage 1: bone-direction energy + temporal smoothness (L-BFGS-B); stage 2: interaction-mesh Laplacian energy (Delaunay on 21 points + 50 object surface samples) + bone direction + temporal smoothness + soft penetration penalty |
| Optimization variables | active joints | active joints + hand base 6DoF (position + quaternion) |
| Low-pass filter | off (alpha=1); the official teleop config uses 0.2 | none |
| Runtime | ≈ 1–2 ms/frame | ≈ 0.15–0.3 s/frame (single core, without parallel contention) |

Changes to the upstream toporetarget code (all implemented inside `run_toporetarget.py` as subclasses / monkey patches; the submodule is untouched):

1. `RobotHand` is implemented with `hand_model.HandModel` (MuJoCo FK); `base_pos` is the flange (keypoint 0);
2. joint limits are passed as `bounds` to L-BFGS-B (upstream calls `scipy.optimize.minimize` **without** limits);
3. `_estimate_joints_from_keypoints` upstream hard-codes 3 joints per finger (out of range for the 6-DoF BrainCo); replaced by an open-hand initial value;
4. `compute_laplacian` replaced by a numerically equivalent vectorized version (upstream is a Python double loop; ~5× faster).

Known limitations (of this unofficial implementation): the penetration penalty only acts on "keypoint–object sample distance < 2 mm" and does not constrain finger **geometry** penetration;
the base pose starts from the identity quaternion and the energy is insensitive to global rotation, so it often converges to local minima with large wrist rotation (see `base_rot_deg`); object surface samples are random per frame (results are not fully reproducible).

## Metrics (`compare.py`)

Computed on "interaction frames" (human hand < 10 cm from the object), with the robot hand placed at each method's own flange pose:

| Metric | Meaning |
|---|---|
| `pen_depth_mm` / `pen_frac` | mean deepest penetration between the hand collision geometry and the object CoACD convex pieces (`mj_geomDistance`) / fraction of frames with penetration > 5 mm |
| `contact_recall` / `precision` / `f1` | a human fingertip < 1.5 cm from the object surface counts as contact; same threshold for robot fingertips; counted per finger per frame |
| `tip_obj_dist_err_mm` | \|robot fingertip-object distance − human fingertip-object distance\| |
| `tip_pos_err_mm` | robot fingertip vs human fingertip scaled by the size ratio (hand frame); `pinch_err_mm` thumb-index fingertip distance error; `inter_tip_err_mm` distance error over the 10 fingertip pairs |
| `base_drift_mm` / `base_rot_deg` | flange translation / rotation relative to the human wrist (always 0 for AnyTeleop) |
| `q_vel_mrad` / `q_jerk_mrad` / `limit_sat` | joint velocity, second difference (mrad/frame), fraction of joint samples at a limit |
| `runtime_ms` | solve time per frame (TopoRetarget measured with 22 parallel processes, inflated) |

## Results

102 sequences (`data/selection.json`, ≤ 3 per handheld object) × left/right hands = 204 (sequence, hand) samples, 46 object types; run completed 2026-09-25.
Full table (including per-object contact_f1 / penetration) in `results/summary.md`; per-sample values in `results/metrics.csv`.

**G1 + BrainCo Revo2 (6 active DoFs + 5 mimic per hand)**

| method | n | pen_depth_mm | pen_frac | contact_recall | contact_precision | contact_f1 | tip_obj_dist_err_mm | tip_pos_err_mm | pinch_err_mm | inter_tip_err_mm | base_drift_mm | base_rot_deg | q_vel_mrad | q_jerk_mrad | limit_sat | runtime_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anyteleop | 204 | **9.089** | **0.713** | **0.864** | **0.883** | **0.856** | **5.947** | **15.9** | **12.9** | **8.840** | **0.000** | **0.000** | 3.329 | 1.690 | 0.060 | **1.515** |
| anyteleop_dexpilot | 204 | 9.380 | 0.715 | 0.815 | 0.873 | 0.821 | 6.573 | 17.4 | 14.6 | 9.529 | 0.000 | 0.000 | **3.177** | 1.967 | **0.058** | 1.744 |
| toporetarget | 204 | 11.3 | 0.748 | 0.617 | 0.831 | 0.702 | 11.9 | 27.6 | 23.0 | 13.3 | 31.3 | 40.3 | 3.240 | **1.522** | 0.155 | 367.9 |

**G1 + Wuji Hand 2 (20 DoFs per hand)**

| method | n | pen_depth_mm | pen_frac | contact_recall | contact_precision | contact_f1 | tip_obj_dist_err_mm | tip_pos_err_mm | pinch_err_mm | inter_tip_err_mm | base_drift_mm | base_rot_deg | q_vel_mrad | q_jerk_mrad | limit_sat | runtime_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anyteleop | 204 | **6.162** | **0.466** | **0.653** | **0.932** | **0.741** | **8.140** | **1.059** | **0.455** | **1.007** | **0.000** | **0.000** | 3.233 | 1.687 | **0.021** | **1.716** |
| anyteleop_dexpilot | 204 | 6.647 | 0.493 | 0.633 | 0.921 | 0.729 | 8.164 | 2.422 | 5.697 | 2.841 | 0.000 | 0.000 | 3.427 | 2.247 | 0.027 | 2.957 |
| toporetarget | 204 | 12.9 | 0.761 | 0.482 | 0.811 | 0.613 | 18.9 | 41.9 | 38.9 | 21.6 | 56.9 | 64.3 | **2.862** | **1.175** | 0.078 | 601.8 |

Reading:
- **AnyTeleop (vector) has the lowest penetration, the best contact preservation and the lowest fingertip tracking error on both hands**, at 1–2 ms per frame. The DexPilot variant is close; its thumb projection prior slightly increases the pinch error on Wuji.
- Wuji's 20 DoFs reproduce the scaled human fingertips almost exactly (`tip_pos_err` 1 mm), yet its contact recall is lower than BrainCo's: the Wuji hand is 15% larger than the human hand, and with the flange fixed at the human wrist the fingertips move outward proportionally, further from the object; BrainCo's size ratio is ≈ 1, so its fingertips land where the human's do.
- TopoRetarget here is an **unofficial reproduction** and is weaker than AnyTeleop: the base 6DoF often converges to local minima with large wrist rotation (mean 40°/64°, translation 3–6 cm), the penetration penalty constrains only keypoint–sample distances rather than the finger geometry, and the interaction-mesh Laplacian term is not enough to pull the hand back to the object surface. Its only advantage is temporal smoothness (`q_jerk`). These numbers therefore reflect this implementation on HUMOTO and cannot be taken directly as the upper bound of the paper's method.
- A mean penetration of 6–13 mm is high for both methods; it comes mainly from two-handed holding of large objects (woven_basket, wash_tub, step_stool: the human fingers themselves are flush with / sunk into the object surface; HUMOTO's hand data already has ~1 cm-scale penetration into the objects), which is a source-data limitation.

## HiPHI (`BENCH_DATASET=hiphi`)

The same scripts also run on the HOI subset of HiPHI (`hiphi_io.py` + `prepare_hiphi.py`; `bench_common.py` switches to `data_hiphi/`, `results_hiphi/` via the environment variable `BENCH_DATASET=hiphi`;
`run_*.py` / `compare.py` / `viser_compare.py` are unchanged):

```bash
export BENCH_DATASET=hiphi MUJOCO_GL=disable
python prepare_hiphi.py --per-category 5 --workers 20      # parse all 7025 HOI raw motions (~3 min), take the 5 sequences with the highest contact-frame fraction per object category
python run_anyteleop.py --hand brainco ...; python run_toporetarget.py --hand brainco ...   # as above
python compare.py --workers 20; python viser_compare.py --hand brainco
```

- Data: `/home/kaizhao/projects/HiPHI/data/<frame>/<lu>/<motion_id>/`, 55-joint BVH (Y-up, cm, 90 Hz, ZXY Euler; the position channels are the local translations),
  3 joints per finger + End Site fingertip → MediaPipe 21 points directly; `object_tracks/<obj>.csv` (meters, XYZW) + `object_meshes/<mesh_id>.obj` (cm).
  Only non-mirrored raw motions are used; 90 Hz is decimated by 3 to 30 Hz. 7 motions whose skeleton lacks `Spine2` are skipped.
- Objects are furniture / container scale (ball 24 cm, boxes 30–70 cm, tables and chairs 50–90 cm, bottle 34 cm, mop 137 cm), exactly one per sequence, all treated as interaction objects;
  the BVH keypoints lie inside the hand, so a 3 cm contact threshold is used for ranking. 60 sequences selected, 109788 frames (30 Hz), 13 object meshes;
  category distribution {'ball': 5, 'bench': 5, 'bottle': 5, 'box': 5, 'bucket': 5, 'chair': 5, 'clothrack': 5, 'mop': 5, 'soccerball': 5, 'stepstool': 5, 'table': 5, 'trashbin': 5}; median contact-frame fraction 0.93.
- Hand size ratio (middle-finger chain length): BrainCo ≈ 1.00–1.05, Wuji ≈ 1.15–1.21 (the HiPHI actors' hands are slightly smaller than the HUMOTO character's).

### HiPHI results

60 sequences (5 for each of 12 object categories) × left/right hands = 120 samples, 13 object meshes, 1830 frames per sequence on average (30 Hz), interaction frames 86%, human fingertips in contact 50% of the time.
Full table in `results_hiphi/summary.md`, per-sample values in `results_hiphi/metrics.csv`.

**G1 + BrainCo Revo2**

| method | n | pen_depth_mm | pen_frac | contact_recall | contact_precision | contact_f1 | tip_obj_dist_err_mm | tip_pos_err_mm | pinch_err_mm | inter_tip_err_mm | base_drift_mm | base_rot_deg | q_vel_mrad | q_jerk_mrad | limit_sat | runtime_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anyteleop | 120 | **18.3** | 0.876 | **0.819** | 0.669 | **0.694** | **9.610** | **26.8** | **11.9** | 20.2 | **0.000** | **0.000** | 6.118 | 4.954 | **0.058** | **0.913** |
| anyteleop_dexpilot | 120 | 18.5 | **0.875** | 0.808 | 0.650 | 0.673 | 10.1 | 28.1 | 12.8 | **19.7** | 0.000 | 0.000 | 6.071 | 5.373 | 0.063 | 1.263 |
| toporetarget | 120 | 23.9 | 0.877 | 0.560 | **0.694** | 0.580 | 14.4 | 35.6 | 29.7 | 21.0 | 47.1 | 55.2 | **5.157** | **2.897** | 0.344 | 148.7 |

**G1 + Wuji Hand 2**

| method | n | pen_depth_mm | pen_frac | contact_recall | contact_precision | contact_f1 | tip_obj_dist_err_mm | tip_pos_err_mm | pinch_err_mm | inter_tip_err_mm | base_drift_mm | base_rot_deg | q_vel_mrad | q_jerk_mrad | limit_sat | runtime_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anyteleop | 120 | 23.4 | 0.924 | **0.734** | **0.811** | **0.743** | **7.139** | **2.773** | **0.674** | **2.316** | **0.000** | **0.000** | 6.018 | 4.866 | **0.033** | **1.465** |
| anyteleop_dexpilot | 120 | **23.4** | 0.924 | 0.725 | 0.799 | 0.732 | 7.270 | 3.717 | 4.750 | 3.360 | 0.000 | 0.000 | 6.856 | 6.008 | 0.034 | 2.362 |
| toporetarget | 120 | 24.2 | **0.883** | 0.564 | 0.718 | 0.610 | 12.6 | 34.9 | 26.3 | 19.4 | 51.3 | 51.0 | **5.798** | **2.858** | 0.042 | 532.5 |

Reading:
- The conclusions match HUMOTO: **AnyTeleop (vector) has the best contact preservation (F1 0.69 / 0.74), the lowest fingertip-object distance error, the most accurate fingertip tracking and is the fastest**; the DexPilot variant is nearly identical.
  TopoRetarget wins only on temporal smoothness; its contact F1 is lower by 0.1–0.13, the base still drifts by 5 cm / 50°, and on BrainCo 34% of the joint samples sit at a limit.
- Penetration is more than twice that of HUMOTO (18–24 mm vs 6–13 mm) and similar across the three methods: HiPHI objects are furniture / container scale, the human pushes, pulls, hugs and lifts boxes, chair backs and steps with the whole palm,
  the BVH hand keypoints lie inside the palm and the object meshes are coarse scans / models, so the source data itself has a 1–3 cm overlap between palm and object; hand retargeting cannot correct this.
  For HiPHI the more discriminative metrics are contact preservation and fingertip-object distance error, not penetration depth.
- Per object: AnyTeleop reaches contact F1 0.8–0.9 on balls, the step stool and the trash bin; the clothes rack (thin bar) and the mop (thin handle) are hard for all methods (F1 0.3–0.5),
  because BrainCo has 6 DoFs and Wuji's phalanges are thick, making the closure of a human hand around a thin bar hard to reproduce. TopoRetarget matches or slightly beats AnyTeleop on Box_A_1 / Table_A_1 / StepStool (large flat surfaces)
  and is clearly worse on balls (F1 0.5 vs 0.8–0.9, penetration 33–38 mm).
- Wuji's fingertip tracking error is only 2.8 mm (20 DoFs reproduce the scaled human hand exactly), but its penetration is 5 mm larger than BrainCo's: the hand is 15–20% larger, and with the flange fixed at the human wrist the whole hand sinks deeper into the object.

## viser (`viser_compare.py`)

Three lanes laid out along +y: HUMOTO human skeleton (black) + hand keypoints (red) + all objects | method A's two robot hands + objects | method B.
GUI: sequence, robot hand (brainco/wuji), play/pause, speed, frame, lane spacing, lane visibility. For remote use: `ssh -L 8080:localhost:8080`.
