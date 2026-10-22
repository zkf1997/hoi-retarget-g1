# Final retargeting results: archive layout and storage format

Whole-body retargeting of human-object interaction motion (OMOMO, HUMOTO, HiPHI) to the Unitree G1 (29 DoF) with the
Wuji Hand 2 (20 DoF per hand). Produced by `omomo_bench/` on branch `handover-2026-10` of github.com/zkf1997/retarget;
pipeline and numbers in `omomo_bench/HANDOVER.md`. Packed by `omomo_bench/pack_release.py`.

## In one paragraph

Each dataset folder has an index (`seqs_test.json`), one **source** npz per sequence (SMPL-X parameters + 55 joints + object
pose, real size, Z-up, 30 fps) and the object meshes; each `results*/pipe_final/<seq>.npz` holds the robot motion as a
`(T, 76)` MuJoCo `qpos` (root position, root quaternion wxyz, 69 joints named in `robot_joint_names`) in a world where the
human and the objects are scaled by `smpl_scale` and the robot is at its real size. Put the object at
`source obj_pos * smpl_scale` with the source orientation and its mesh scaled by `object_mesh_scale`. Map joints by name.
A complete 12-line reader is in "Result npz". Not included: SMPL-X model files (license), intermediate lanes, raw datasets.

## Layout

```
retarget_final_results_<date>/
├── RESULTS_README.md               this file (named so that extracting over a clone does not overwrite the repo README)
├── manifest.json                   creation date, git commit, per-dataset counts and the result lane each pipe_final was copied from
├── urdf/g1_wuji_hand2/             G1 + Wuji Hand 2 URDF and meshes (viewer visuals; relative mesh paths)
└── omomo_bench/
    ├── data/          OMOMO  (602 sequences, 15 objects, subjects sub16 / sub17)
    ├── data_humoto/   HUMOTO (100 sequences, 41 hand-held objects + furniture)
    ├── data_hiphi/    HiPHI  (60 sequences, 13 objects)
    │   ├── seqs_test.json              sequence index: [{"seq", "object", "subject", "n_frames", ...}]
    │   ├── multi_objects.json          HUMOTO only: {seq: [object names]}, primary (hand-held) object first
    │   ├── multi_objects_handheld.json HUMOTO only: same, hand-held objects only (no furniture)
    │   ├── seqs_test_perseq.json       HiPHI only: index with subject = seq (per-sequence body height)
    │   ├── source/<seq>.npz            human + object input (see "Source npz")
    │   └── objects/<obj>/              object meshes (see "Object folder")
    ├── results/pipe_final/<seq>.npz           OMOMO final robot motions  (= lane pipe_handrepair_coupled)
    ├── results_humoto/pipe_final/<seq>.npz    HUMOTO final               (= lane pipe_v3_nfcfull)
    ├── results_hiphi/pipe_final/<seq>.npz     HiPHI final                (= lane pipe_v3_it12)
    └── results*/final_metrics/                metrics of these lanes (csv per sequence, md summary)
```

Extracting the archive over a clone of the repository (`tar -xzf ... --strip-components=1 -C <clone>`) puts every file where
the scripts expect it; `results*/pipe_final` are real directories here (symlinks in the working tree). The archive contains
only the final lanes: viewer presets such as `--final` or lane names like `umr_wuji` / `pipe_s2*` refer to intermediate
lanes that are not shipped (the viewer prints `lane ... skipped` for them); use `--lane final=results*/pipe_final`.

## Conventions

- Units: metres, radians, seconds. World frame Z-up. 30 fps (HiPHI was 90 Hz, subsampled by 3).
- Quaternions: `obj_quat_xyzw` in source files is (x, y, z, w); everything in result `qpos` follows MuJoCo (w, x, y, z).
- **Scaled world.** The retargeting runs in a world where the human and the objects are scaled by
  `s = smpl_scale = 1.32 / human_T-pose_height` (0.72–0.85) so that the human has the robot's height; the robot is at its real
  size. Robot root positions are in that scaled world. The object is at `source obj_pos * s` with the source orientation and
  its mesh scaled by `object_mesh_scale` (= `s` for every final lane). To put everything back to real scale divide all
  positions by `s` (joint angles are unchanged); the robot then appears `1/s` times its real size relative to the objects.
- Ground: z = 0 in both worlds (`ground_z` is 0 for every final sequence).

## Source npz (`data*/source/<seq>.npz`, `np.load(..., allow_pickle=True)`)

| key | shape / type | meaning |
|---|---|---|
| `seq_name`, `object`, `subject`, `gender` | str | sequence id, primary object folder name, subject id (OMOMO sub16/sub17, else dataset name), SMPL-X gender |
| `fps` | int | 30 |
| `betas` | (16,) | SMPL-X shape (HUMOTO: one shape for all sequences; HiPHI: fitted per sequence) |
| `root_orient`, `pose_body`, `pose_hand`, `trans` | (T,3) (T,63) (T,90) (T,3) | SMPL-X parameters (axis-angle); `pose_hand` = left 45 + right 45, present for HUMOTO and HiPHI (OMOMO: flat hands, key absent) |
| `joints55` | (T,55,3) | SMPL-X joints in the world (0 pelvis, 20/21 left/right wrist, 25–39 left fingers, 40–54 right fingers, 3 joints per finger, order index-middle-pinky-ring-thumb) |
| `tpose_height` | float | human T-pose height in metres; `s = 1.32 / tpose_height` |
| `obj_pos`, `obj_quat_xyzw` | (T,3) (T,4) | primary object pose (mesh local frame -> world), real size |
| `obj_scale`, `obj_scale_raw` | float, (T,) | bookkeeping from the converter (OMOMO: the raw-unit -> metre factor of `meta.json`, already applied to `<obj>.obj`; HUMOTO / HiPHI: 1.0). Do **not** apply it: `<obj>.obj` is already in metres |
| `extra_objects`, `extra_obj_pos`, `extra_obj_quat_xyzw` | (K,) str, (T,K,3), (T,K,4) | HUMOTO only: other scene objects (furniture, second hand-held objects) and their poses |
| `category`, `text` | str | HiPHI only: object category and the HiPHI text description (HiPHI `extra_objects` is empty) |

## Object folder (`data*/objects/<obj>/`)

| file | content |
|---|---|
| `<obj>.obj` | visual mesh, real size, in the object's local frame (centred: `meta.json` gives the original centre and scale) |
| `<obj>_display.obj` | the same decimated to <= 5 000 faces (viewers) |
| `<obj>_coacd_NN.obj` | CoACD convex pieces used for collision (threshold 0.03) |
| `meta.json` | `object`, `scale` (raw -> metres), `center_raw`, `n_pieces`, `extents_m`; OMOMO adds `coacd_threshold`, HUMOTO / HiPHI add `source` (free text from the converter; the HiPHI string still says HUMOTO because the two share the converter) |

HUMOTO furniture pieces not in `objects/` were read from the HUMOTO GLB at run time (`hand_bench/humoto_io.py`); the
viewer falls back to that reader when a name is missing.

## Result npz (`results*/pipe_final/<seq>.npz`)

| key | shape / type | meaning |
|---|---|---|
| `qpos` | (T, 76) float32 | per-frame MuJoCo qpos of `robots/g1_29dof_wuji_hand2.xml` + free root: `[0:3]` root position (scaled world), `[3:7]` root quaternion (w,x,y,z), `[7:76]` 69 joint angles in `robot_joint_names` order |
| `robot_joint_names` | (69,) str | joint order: 12 leg joints, 3 waist, left arm 7, **left fingers 20**, right arm 7, **right fingers 20** (interleaved by kinematic tree; see list below) |
| `fps`, `frame_ids` | (1,) float, (T,) int | 30; source frame index of each row (0..T-1 for every final lane) |
| `smpl_scale` | (1,) float | `s` (see Conventions) |
| `object_mesh_scale` | float | scale of the object mesh in the robot world (= `s`) |
| `ground_z` | (1,) float | ground height used (0) |
| `hand`, `scaling` | str | `wuji`, `umr` (scaling convention) |
| `stage` | str | `v3` (HUMOTO, HiPHI: UMR body -> AnyTeleop fingers -> coupled arm+finger refinement, 12 iterations) or `handrepair_coupled` (OMOMO: UMR body with finger collision off -> coupled arm+finger repair with full collision) |
| `body_lane` | str | OMOMO only: body-stage lane the repair started from (`umr_wuji_nofingercol`); for HUMOTO / HiPHI it is inside `params` |
| `params`, `tau` | str (JSON), float | HUMOTO / HiPHI: hand-stage parameters (incl. `body_lane`, `v3_iters`, `multi`) and the contact distance threshold (0.02 m) |
| `objects` | (K,) str | HUMOTO: objects used in the hand stage, primary first (`multi: true` in `params`) |
| `hands` | (2,) str | `left`, `right` |
| `robot_xml`, `source_data`, `source_object_dir`, `noitom_*`, `zero_source_finger_pose`, `robot_name`, `source_format`, `source_sequence_key` | str / misc | bookkeeping written by UMR; the paths are absolute paths of the machine that produced the files and are not used by any script (models are rebuilt from the `hand` key) |

Joint names (`robot_joint_names`), in order:
```
left_hip_pitch_joint left_hip_roll_joint left_hip_yaw_joint left_knee_joint left_ankle_pitch_joint left_ankle_roll_joint
right_hip_pitch_joint right_hip_roll_joint right_hip_yaw_joint right_knee_joint right_ankle_pitch_joint right_ankle_roll_joint
waist_yaw_joint waist_roll_joint waist_pitch_joint
left_shoulder_pitch_joint left_shoulder_roll_joint left_shoulder_yaw_joint left_elbow_joint left_wrist_roll_joint left_wrist_pitch_joint left_wrist_yaw_joint
l_thumb_cmc_flex l_thumb_cmc_abd l_thumb_mcp l_thumb_ip  l_index_finger_mcp_flex l_index_finger_mcp_abd l_index_finger_pip l_index_finger_dip
l_middle_finger_mcp_flex l_middle_finger_mcp_abd l_middle_finger_pip l_middle_finger_dip  l_ring_finger_mcp_flex l_ring_finger_mcp_abd l_ring_finger_pip l_ring_finger_dip
l_pinky_mcp_flex l_pinky_mcp_abd l_pinky_pip l_pinky_dip
right_shoulder_pitch_joint ... right_wrist_yaw_joint   (7, same pattern)
r_thumb_cmc_flex ... r_pinky_dip                       (20, same pattern)
```
Always map by name (`robot_joint_names`) rather than by index: the plain-G1 order used by holosoma / IsaacLab puts the
29 body joints first and has no finger joints.

Minimal reader (MuJoCo forward kinematics with the repository's MJCF):
```python
import numpy as np, mujoco
r = np.load("results_humoto/pipe_final/<seq>.npz", allow_pickle=True)
src = np.load("data_humoto/source/<seq>.npz", allow_pickle=True)
m = mujoco.MjModel.from_xml_path("robots/g1_29dof_wuji_hand2.xml")        # 69 joints, no free root
d = mujoco.MjData(m)
names = [str(x) for x in r["robot_joint_names"]]
adr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): m.jnt_qposadr[j] for j in range(m.njnt)}
s = float(np.asarray(r["smpl_scale"]).reshape(-1)[0])          # stored as shape (1,)
for t in range(len(r["qpos"])):
    q = r["qpos"][t]
    for n, v in zip(names, q[7:]): d.qpos[adr[n]] = v                        # joints by name
    mujoco.mj_kinematics(m, d)                                                # link poses in the root frame; root pose = q[:3], q[3:7] (wxyz)
    obj_pos = src["obj_pos"][t] * s; obj_quat_xyzw = src["obj_quat_xyzw"][t]  # object in the robot world, mesh scaled by r["object_mesh_scale"]
```
`omomo_bench/compare.py: augmented_model()` builds the same model with a free root and the object attached (used by every
evaluation script).

## Viewing

From a clone of the branch with the archive extracted over it (conda env `hsretargeting`, see `HANDOVER.md`):
```bash
cd omomo_bench
BENCH_DATASET=humoto python viser_compare.py --lane final=results_humoto/pipe_final --all-objects --port 8090
BENCH_DATASET=hiphi  python viser_compare.py --lane final=results_hiphi/pipe_final --port 8091
BENCH_DATASET=omomo  python viser_compare.py --lane final=results/pipe_final --port 8092
MUJOCO_GL=egl BENCH_DATASET=humoto python render_compare.py --seqs <seq> --lane final=results_humoto/pipe_final --frames 5   # PNG grid
```
The viewer draws the source human as an SMPL-X mesh when the SMPL-X model files are available (`$SMPLX_DIR/smplx/SMPLX_*.npz`
or `UMR/smpl/`, license-gated, not included), otherwise as joints. Robot visuals come from `urdf/g1_wuji_hand2`, forward
kinematics from `robots/g1_29dof_wuji_hand2.xml` (after `bash omomo_bench/relocate.sh` if the clone is not at
`/home/kaizhao/projects/retarget`).

## Metrics (`results*/final_metrics/`)

HUMOTO: `pipe_metrics_nfcfull.csv` has one row per sequence and lane; this archive's lane is `final_nfc` (`final` is the same
v3 stage on the full-collision body, `umr_body` / `umr_body_nfc` the two body stages). HiPHI: `pipe_metrics_it12.csv`, lane
`v3_it12` = this archive's lane. `wrist_gap_final.csv` / `.log`: wrist displacement and hand penetration of the body stage vs
the final lane on contact frames. Columns
include `contact_f1`, `tip_obj_dist_err_cm`, `hand_pen_mm`, `hand_pen_frac`, `obj_pen_max` (m), `scene_pen_mm`,
`flange_pos_err_cm`, `flange_rot_deg`, `joint_jerk`, `root_jerk`, `finger_jerk`; `pipe_summary_*.md` is the mean table.
OMOMO: `summary_handrepair_coupled.md` (body metrics, lane `coupled`) and `wrist_gap_omomo_coupled.csv` (per hand: wrist
displacement from the human wrist while the human hand is on the object, hand-object penetration).
