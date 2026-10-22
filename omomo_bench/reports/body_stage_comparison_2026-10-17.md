# Body-stage comparison summary: UMR vs OmniRetarget, CoACD vs convex hull, finger-collision switch (2026-10-17)

This document collects all body-stage comparisons rerun and re-evaluated on OMOMO (602 sequences), HUMOTO (100 sequences) and HiPHI (60 sequences) after the OmniRetarget baked-hand bug fix of 2026-10-14, together with the effect of the finger-collision switch on the full pipeline. All lanes use the same CoACD object geometry and are evaluated with the full collision model; the robot is G1 + Wuji Hand 2 (OMOMO / HUMOTO also have BrainCo, same ranking). Data files: `results*/summary_3ds.md`, `summary_nfc.md`, `wrist_gap_{3ds,nfc}.log`, `wrist_orientation_{3ds,nfc}.log`, `pipe_summary_nfcfull.md`, `results/summary_handrepair_coupled.md`, `summary_repairfc.md`; scripts `results/run_umr_vs_omni_3ds.sh`, `run_nfc_compare.sh`, `run_repair_fullcol.sh`.

## 0. Prerequisite: the OmniRetarget baked-hand bug

OmniRetarget hard-codes 29 joints, so the hand can only be attached as fixed geoms under `wrist_yaw_link`. Our `run_omni.py bake_hands` wrote the MuJoCo-compiled geom poses (which already include the mesh re-centering `mesh_pos/mesh_quat`) back into the XML; on reload the original STLs were re-centered a second time: each hand mesh piece is off by 1.4 cm on average, 3.7–4.4 cm at most, with orientation off by up to 171°, and the hand appears as scattered pieces (`reports/figures/omni_hand_bake_bug.png`). Fixed on 2026-10-14 (compensated by `mesh_pos/mesh_quat`; error vs the articulated URDF < 0.001 mm; output carries `BAKE_MARK`); all OmniRetarget lanes rerun; old results kept as `*_oldbake`. UMR, hand_pipeline and the finals of the three datasets do not use the baked model and are unaffected. Before the fix OmniRetarget also pushed the hand to the other side of the chair back in the white-chair sequence; after the fix this disappears.

## 1. Setup

| | Description |
|---|---|
| Scaling | Uniform UMR-style: human and objects scaled by 1.32 / T-pose height; HiPHI per-sequence height (`data_hiphi/seqs_test_perseq.json`) |
| UMR | Full MJCF with finger joints, fingers pinned at the rest pose; object non-penetration as a hard constraint with slack (slack cost 5000); contact term weight 5, threshold 10 cm, at most 128 slots |
| OmniRetarget | 15 keypoints + interaction mesh, hard non-penetration; fallback chain when infeasible default → pen_tol_0.02 → fingers_free → fingers_free_pen_tol_0.02 → no_obj_nonpen |
| Collision geometry | CoACD convex pieces (default) or holosoma's default whole-mesh convex hull (`--native-collision`) |
| Evaluation | `compare.py` (penetration, hand-contact preservation, wrist/ankle error, jerk), `wrist_gap_stats.py` (fraction of frames, among those where the human hand is within 10 cm of the object, in which the robot wrist is > 12 cm from the human wrist; a sequence where either hand exceeds half of its frames counts as a "pushed-off sequence", counted per hand), `wrist_orientation_stats.py` (relative to the canonical hand frame of the human hand) |

## 2. UMR vs OmniRetarget (body stage)

| Metric | OMOMO Omni | OMOMO UMR | HUMOTO Omni | HUMOTO UMR | HiPHI Omni | HiPHI UMR |
|---|---|---|---|---|---|---|
| Object penetration (mm) / frames > 5 mm | 10.1 / 42% | **5.4** / 43% | 10.4 / 60% | **4.5 / 32%** | 13.5 / 47% | **5.7 / 44%** |
| Sequences needing fallback | 359 / 602 | 0 | 71 / 100 | 0 | 32 / 60 | 0 |
| Hand-object contact preservation | **0.97** | 0.95 | **1.00** | 0.87 | 0.96 | 0.96 |
| Wrist position error (cm) | **2.2** | 3.5 | **1.6** | 3.7 | **2.9** | 3.0 |
| Wrist orientation error (°) / frames > 30° | 32 / 43% | **13 / 6%** | 39 / 59% | **10 / 5%** | 47 / 67% | **14 / 3%** |
| Sequences with hand pushed off | **10** | 62 | **0** | 16 | 6 | **5** |
| Ground penetration (mm) | 4.3 | **0.2** | 4.2 | **0.05** | 2.4 | **0.2** |
| Foot slide (mm/frame) | **1.0** | 2.6 | **1.0** | 2.8 | **0.9** | 1.9 |
| Joint jerk / root jerk | **201** / 56 | 273 / **14** | **238** / 62 | 258 / **19** | **140** / 35 | 193 / **10** |
| Time per frame (s) | 2.9 | **0.25** | 1.1 | **0.15** | 2.3 | **0.10** |

OmniRetarget penetration split by fallback mode (OMOMO): default hard constraint 243 sequences 0.7 mm; pen_tol 103 sequences 12.6 mm; fingers_free 145 sequences 15.2 mm; both combined 51 sequences 19.1 mm; object constraint off 60 sequences 24.6 mm. 97% of the total penetration comes from fallback sequences (also 97% on HUMOTO and HiPHI).

Conclusion: OmniRetarget has more accurate wrist position, better contact preservation, almost never pushes the hand to the other side of the object, and barely penetrates when the hard constraint is feasible; but it lacks a wrist orientation term (the hand rotates freely about the forearm), penetrates heavily after fallback, has high ground penetration and root jerk, and is 10–20x slower. UMR never fails, has stable penetration, 3–4x better wrist orientation, and clean ground and root, but its wrist position is 1 cm worse, with more pushed-off sequences and more foot slide.

## 3. OmniRetarget: CoACD convex pieces vs whole-mesh convex hull

| G1 + Wuji | OMOMO CoACD | OMOMO convex hull | HUMOTO CoACD | HUMOTO convex hull | HiPHI CoACD | HiPHI convex hull |
|---|---|---|---|---|---|---|
| Object penetration (mm) / frames > 5 mm | 10.1 / 42% | **2.1 / 9%** | 10.4 / 60% | **4.1 / 21%** | 13.5 / 47% | **10.5 / 33%** |
| Sequences needing fallback | 359 | **39** | 71 | **26** | 32 | **22** |
| Hand-object contact preservation | **0.97** | 0.83 | **1.00** | 0.98 | 0.96 | **0.97** |
| Wrist position error (cm) | **2.2** | 3.5 | **1.6** | 2.1 | **2.9** | 3.4 |
| Wrist orientation error (°) | **32** | 37 | **39** | 41 | **47** | 49 |
| Sequences with hand pushed off | **10** | 59 | 0 | 0 | 6 | **4** |
| Joint / root jerk | **201 / 56** | 232 / 61 | **238 / 62** | 244 / 64 | **140 / 35** | 146 / 36 |
| Time per frame (s) | 2.9 | **1.1** | 1.1 | **0.7** | 2.3 | **1.2** |

The convex hull makes the hard constraint easy to satisfy: low penetration, few fallbacks, twice as fast; but it swaps in a larger object: concavities are filled and the hand cannot get in. On OMOMO contact preservation drops 0.97 → 0.83 (floor lamp 1.00 → 0.03, clothes stand 0.98 → 0.46), wrist error 2.2 → 3.5 cm, and sequences held off outside the real surface 10 → 59. HUMOTO's small items are nearly convex, so the difference is small; HiPHI's large objects still need 22 fallbacks with the convex hull, and the stool and clothes rack penetrate more. **Decision: the main comparison uses CoACD; the convex hull serves as the reference column for the "holosoma default configuration".**

## 4. "Hand pushed to the other side of the object"

The trigger condition is the same for both methods: the fingers are fixed in a slightly curled rest pose, the robot hand is 34% longer than the scaled human hand, forming a hook; when the human grasps a thin bar, a thin edge or a bowl rim, the palm block presses on the object and the curled fingers block the way back. It happens often with UMR (OMOMO 57 sequences, HUMOTO 11, HiPHI 4, counted per sequence) and rarely with OmniRetarget (10 / 0 / 6), because:

1. OmniRetarget has a wrist keypoint cost and the interaction mesh, so moving the hand to the other side is expensive; UMR has no wrist position term, its contact term is defined as "the vector from a robot slot to its paired object point equals the human-side vector", and the pull-back force comes only from a few slots on the hand with weight 5.
2. When infeasible, OmniRetarget relaxes the constraint instead of moving the hand: of the 57 sequences where UMR is pushed off, OmniRetarget used a fallback in 44 (its penetration there 14.3 mm vs 7.2 mm for UMR), i.e. it trades penetration for wrist position; UMR always keeps full collision.
3. UMR's non-penetration is a per-frame linearized hard constraint: the fingers touch the object about 0.3 s before the human wrist does, and the constraint pushes the hand along the surface to the other side; the way back would pass through the object, the linearized constraint cannot see the detour, and the iteration stops at a constrained stationary point "pressing on the object from the other side" (not a minimum of the contact term but a KKT point).

Distribution: OMOMO concentrates on large box 11, plastic box 8, small table 6, wooden chair 6, vacuum 5, white chair 5 (thin bars, edges); HUMOTO is almost entirely thin-walled containers such as vases, bowls and plates; HiPHI's flat-palm pressing on large objects is rare. Longer hands are worse: UMR with BrainCo 23 sequences, with Wuji 57. List and visualization: `reports/pushed_sequences_umr_wuji.md`, viser 8096 / 8097 / 8098. Detailed geometric analysis: `reports/whitechair_002_wrong_side.md`.

Effect on later stages (pushed-off sequences only): the 11 HUMOTO sequences after v3 have flange error 8.1 → 2.4 cm, pushed-off frames go to zero, F1 0.82; what remains is jerk (284 vs 216 for the finger-collision-off body); the 4 HiPHI sequences still have 6 cm flange error in the final and 15% of frames pushed off, large objects cannot be fully repaired; OMOMO has no later stage and must solve it in the body stage.

## 5. UMR body stage: finger collision on / off

`--variant nofingercol`: the 50 finger geoms do not collide; the palm block, mount and forearm collide as usual. Four-way comparison on OMOMO:

| Metric | Omni + convex hull | Omni + CoACD | UMR full collision | UMR finger collision off |
|---|---|---|---|---|
| Object penetration, whole body (mm) / frames > 5 mm | **2.1 / 0.09** | 10.1 / 0.42 | 5.4 / 0.43 | 14.2 / 0.67 |
| Hand penetration, contact frames (mm) | 0.05 | 0.05 | 5.8 | 17.0 |
| Sequences needing fallback | 39 | 359 | 0 | 0 |
| Hand-object contact preservation | 0.83 | 0.97 | 0.95 | **1.00** |
| Wrist position error (cm) | 3.5 | **2.2** | 3.5 | 2.5 |
| Wrist orientation error (°) / frames > 30° | 37 / 49% | 32 / 43% | 13 / 6% | **8 / 0.3%** |
| Sequences with hand pushed off | 59 | 10 | 57 | **5** |
| Ground penetration (mm) | 1.6 | 4.3 | **0.2** | **0.2** |
| Joint jerk / root jerk | 232 / 61 | 201 / 56 | 273 / 14 | **181 / 10** |
| Time per frame (s) | 1.1 | 2.9 | 0.25 | **0.07** |

HUMOTO / HiPHI (full collision → finger collision off): wrist position 3.7 → 2.7 / 3.0 → 2.2 cm, wrist orientation 9.9 → 6.8 / 14.1 → 11.6°, pushed off 11 → 0 / 4 → 2 sequences, joint jerk 258 → 170 / 193 → 130, hand penetration 3.2 → 8.1 / 3.4 → 12.8 mm. Per sequence: wrist error is smaller in 91–94% of sequences, joint jerk lower in 94–100%, penetration larger in 91–98%.

## 6. Penetration after running all stages

| Full pipeline | Body stage with finger collision | Body stage without finger collision | Diff |
|---|---|---|---|
| HUMOTO (AnyTeleop + v3) hand-object penetration (mm) / frames > 5 mm | 1.00 / 5.9% | 0.89 / 4.6% | −0.11 |
| Whole-body-object / scene penetration (mm) | 1.7 / 2.95 | 1.5 / 2.77 | −0.2 / −0.18 |
| HiPHI (AnyTeleop + v3) hand-object penetration (mm) / frames > 5 mm | 0.66 / 1.6% | 0.81 / 3.1% | **+0.15** |
| Whole-body-object / scene penetration (mm) | 3.1 / 3.06 | 3.4 / 3.38 | **+0.3 / +0.32** |
| OMOMO (coupled repair) hand-object penetration, contact frames (mm) / frames > 5 mm | 1.24 / 6% | 1.23 / 6% | 0 |
| Whole-body-object penetration (mm) / frames > 5 mm | 1.4 / 6.7% | 1.3 / 6.8% | 0 |

Other final metrics: HUMOTO F1 0.771 → 0.799, flange 2.54 / 4.6° → 2.42 / 4.0°, joint jerk 197 → 167; HiPHI F1 0.577 → 0.572, flange 3.04 / 6.0° → 3.01 / 5.9°, jerk 134 → 113; OMOMO wrist error 2.35 → 2.26 cm, jerk 201 → 189, pushed off 6 → 5 sequences. Per sequence, the HiPHI +0.15 mm has a median of 0 and concentrates in two or three large-object sequences (max +5.7 mm).

## 7. Conclusions and recommended configuration

- **Use UMR for the first stage.** The third stage re-solves the arms and fingers and anchors the wrist, so OmniRetarget's lead in wrist position and contact gets redone; where it lags (root jerk, ground penetration, body-object penetration, instability from fallback) the third stage cannot reach. UMR is also 10–20x faster, supports multiple objects, and its model format matches the later stages.
- **Compare OmniRetarget with CoACD**; the convex hull serves only as reference.
- **Turn finger collision off in the body stage, keep palm and forearm collision**, provided a later stage with full collision actually solves the fingers. HUMOTO (v3) and OMOMO (coupled repair) already adopt this and final penetration does not increase; HiPHI keeps finger collision (turning it off on large objects adds 0.15–0.3 mm penetration in exchange for a 16% jerk reduction; can be changed when smoothness matters more).
- Directions not to take: UMR solving the fingers itself while keeping collision (`free`, pushed-off unchanged); increasing the contact weight in the body stage (`contact4`, penetration and jerk rise together).

## 8. Related files

- Reports: `umr_vs_omni_3ds.md` (English tables for the three datasets), `umr_finger_collision_3ds.md`, `pushed_sequences_umr_wuji.md`, `whitechair_002_wrong_side.md`
- Figures: `reports/figures/omni_hand_bake_bug.png`, `sub16_whitechair_002_report_{right,left}.{png,mp4}`, `sub16_whitechair_002_analysis.png`, `sub17_woodchair_000_native.png`
- Visualization: 8090 / 8091 final pipeline (HUMOTO / HiPHI), 8095 five sequences including the white chair (OMOMO, four lanes), 8096 / 8097 / 8098 pushed-off sequences
- Lanes: `omni_wuji_umrscale`, `omni_wuji_umrscale_native`, `umr_wuji`, `umr_wuji_nofingercol`, `pipe_handrepair_coupled` (OMOMO final), `pipe_handrepair_coupled_fullcol`, `pipe_v3_nfcfull` (HUMOTO final), `pipe_v3_it12` (HiPHI final)
