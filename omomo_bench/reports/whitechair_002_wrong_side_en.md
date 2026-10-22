# OMOMO `sub16_whitechair_002`: analysis of the robot hand landing on the other side of the chair back (2026-10-13)

English translation (2026-10-20) of whitechair_002_wrong_side.md; the Chinese file is the original.

Sequence: the person places both hands from above onto the top rail of the chair back and grasps it (rail diameter about 2 cm, the chair back is a thin mesh), then lifts the chair. 131 frames; the human wrists are within 10 cm of the chair in 109 frames; while grasping, the wrist is about 4 cm from the rail surface. In the original results (UMR Wuji, OmniRetarget Wuji) both hands land on the other side of the chair back for the whole sequence, with the wrist 13–20 cm from the human wrist.

## 1. Symptom (right hand)

| lane | Mean wrist offset during grasp | Frames with offset > 12 cm | Hand-chair penetration (full collision model) |
|---|---|---|---|
| UMR Wuji (original) | 13.9 cm | 97% | 6.4 mm |
| OmniRetarget Wuji (original) | 13.0 cm | 93% | 0 |
| UMR Wuji, finger collision off | 4.9 cm | 0% | 1.9 mm |
| OmniRetarget Wuji, fingers excluded from constraints | 5.1 cm | 0% | 0 |
| UMR BrainCo (original, reference) | 4.6 cm | 0% | 1.6 mm |

Material: `figures/sub16_whitechair_002_report_right.png` / `_left.png` (6-frame close-ups, columns = source | UMR original | UMR fixed | Omni original | Omni fixed); videos `figures/sub16_whitechair_002_report_right.mp4` / `_left.mp4` (all 131 frames, same five columns); `figures/sub16_whitechair_002_approach_right.png` (the approach during frames 8–20); `figures/sub16_whitechair_002_analysis.png` (curves).

## 2. Ruling out source-data problems

- The human wrist is 3.6–4.6 cm from the rail surface and never enters the chair.
- OMOMO has no finger data and the SMPL-X hand is flat; none of the 15 flat finger joints falls into any collision convex piece of the chair during the grasp.
- Of the two BrainCo lanes, the UMR one is normal on this sequence, so the problem is related to the hand geometry, not to the chair or the trajectory itself.

## 3. Hand geometry: the robot hand is 34% longer than the scaled human hand

Right hand, wrist to middle-finger distal joint: human 17.3 cm, which times the scale s = 0.766 gives 13.2 cm in the robot world; Wuji rest pose 17.8 cm, 19.7 cm to the fingertip. The fingers are pinned in the slightly curled rest pose, so the whole hand is a fixed "hook" with an opening (palm to fingertips) of about 8 cm, much wider than the top rail, so in principle it could accommodate the rail.

## 4. Where it actually gets stuck

Place the rest-pose Wuji hand as-is at the human wrist position and orientation (canonical hand frame) and compute the penetration against the chair's convex pieces with MuJoCo:

- **Palm block** (the segment from the flange to the finger bases, 7–14 cm) penetrates > 5 mm in 95 of the 109 frames, mean 10–15 mm. When the human grasps the rail the palm wraps around it, and the rail sits exactly inside the robot's palm block.
- **Fingers** penetrate > 5 mm in only 5 frames. So "longer fingertips hit the chair back" is not the direct cause; at the target pose it is the palm that penetrates.

Translate the whole hand back and forth along the hand axis and check where it can be placed without collision (frame 65, right panel of `analysis.png`):

| Finger handling | Collision-free translation |
|---|---|
| Fingers pinned, colliding | **no** position within ±16 cm |
| Fingers straightened, colliding | only one, 8 cm backward |
| Fingers ignored (palm only) | ≥ 6 cm backward, or 8–12 cm forward |

Meaning: the palm only needs to move back 6 cm to clear the rail, but after moving back the rail falls into the region enclosed by the curled fingers and the fixed fingers hit it; moving forward, both the palm block and the fingers press on the rail. No position along the hand axis keeps palm and fingers off the rail at the same time. The finger length and curl turn the human "palm wrapped around the rail" position into one that is infeasible for the robot, consistent with the user's judgement, except that what collides directly is the palm, while the fingers block the way out.

## 5. How it gets pushed across over time (`approach_right.png`, video frames 8–20)

- The human hand descends from above, the palm lands on the top rail, the fingers hang down on the other side of the rail.
- The robot fingers are longer than the human's and point downward; at frames 8–11, when the human wrist is still 16–18 cm from the chair, the robot fingertips already touch the chair back. The hard constraint blocks the descent here.
- The human keeps moving down and forward; the wrist anchor and surface correspondence terms pull the robot hand to follow, but the hand cannot pass through the rail; the per-frame linearized solver can only slide along the constraint surface and routes the hand under the rail to the other side of the chair back. By frame 13 the wrist deviation is already 13–19 cm, and the rest of the sequence stays in this local optimum because returning to the correct side would require passing through the rail.
- Both methods share the mechanism. UMR has no wrist position term and deviates further (13.9 cm, with the fingers pressed 6 mm into the chair); OmniRetarget has a wrist keypoint term, but the hard constraint takes precedence, still 13 cm.

## 6. Fix and verification

OMOMO has no finger data; the rest-pose fingers are only placeholders, and letting them take part in the hard constraint puts the cart before the horse.

- OmniRetarget: `run_omni.py ... -- --fingers-free`, finger geoms excluded from the object constraint, palm and body as usual.
- UMR: `run_umr.py --variant nofingercol`, the finger geoms of `robots/g1_29dof_wuji_hand2_nofingercol.xml` do not collide.

Results in the table of Section 1: the wrist offset returns to 5 cm and pushed-off frames go to zero; evaluated with the full collision model, UMR's hand-chair penetration drops from 6.4 mm to 1.9 mm, OmniRetarget's is 0. In the videos and close-ups the fixed hand rests on the top rail with the fingers hanging to the other side, matching the human.

The side effect is that fingers may pass through thin objects (no frame > 5 mm in this sequence). On HUMOTO and HiPHI the fingers are re-solved by the subsequent AnyTeleop + v3, which handles non-penetration; on a 20-sequence HUMOTO subset, "finger collision off in the body stage → v3" was verified to be better across the board (contact F1 0.773 → 0.784, hand-object penetration 0.89 → 0.73 mm, flange 2.92 → 2.87 cm).

## 7. Scope

`wrist_gap_stats.py` statistics over all of OMOMO (robot wrist offset > 12 cm in frames where the human wrist is < 10 cm from the object): sequences in which one hand is pushed off for more than half the frames: umr_wuji 62 / 535, umr_brainco 23, omni_wuji 10, omni_brainco 6 (OmniRetarget numbers are from the rerun after the baked-hand fix; before the fix 19 / 11); concentrated on thin bars and thin plates such as chairs, table edges, box edges and vacuum handles. Wuji is worse than BrainCo because the hand is longer and the hook deeper. After the hand-geometry fix, OmniRetarget no longer pushes the hand to the other side of the chair back in this sequence (wrist offset 13 → 5 cm, see Section 9), so the OmniRetarget-column symptoms in Sections 1–6 of this report come mainly from the misplaced hand; the UMR conclusions are unchanged.

## 8. Fix for OMOMO (finger collision must be kept)

For OMOMO the body stage is the final output, so finger collision cannot be turned off. Letting UMR solve the fingers as free DoFs while keeping collision (`--variant free`) does not help (12.6 cm / 76%). Two steps are adopted: a body stage with finger collision off (`run_umr.py --variant nofingercol`, the hand lands at the human wrist), then `omomo_hand_repair.py --coupled` jointly re-solves the 7 arm joints + 20 finger joints per frame with the full collision model: wrist position / orientation anchored to the canonical hand frame of the human hand, rest-pose finger prior, previous-frame + acceleration terms, hand and forearm non-penetration against the object with slack, no contact target, 8 iterations. The version that re-solves only the fingers with the arm fixed (`pipe_handrepair`) also fixes the wrist position (6.0 cm), but when the palm block presses on a box / trash bin it cannot push itself out, and its penetration of 6.48 mm is worse than the original, so it was dropped.

This sequence (coupled): wrist offset left 5.5 / right 5.6 cm, pushed-off frames 0, mean penetration with finger collision 0.5 / 1.1 mm, frames > 5 mm 2% / 0. All 535 sequences with contact (`results/wrist_gap_omomo_coupled.csv`): mean wrist offset 7.4 → 6.7 cm, pushed-off frame fraction 0.09 → 0.01, affected sequences 62 → 5, hand-object penetration 5.79 → 1.23 mm, frames > 5 mm 0.44 → 0.06. Body metrics (`results/summary_handrepair_coupled.md`, 602 sequences): object penetration 5.4 → 1.3 mm, penetrating-frame fraction 0.43 → 0.07, wrist error 3.45 → 2.26 cm, joint jerk 273 → 189, root jerk 14.2 → 10.3. The remaining 5 sequences (white chair 009/037, wooden chair 001/004/046) are cases where the collision-free body stage itself presses the hand into the chair back; the coupled re-solve can pull back only one hand. Results are in `results/pipe_handrepair_coupled` (= `results/pipe_final`), visualization 8095.

## 9. OmniRetarget baked-hand bug and rerun (2026-10-14/15)

In the initial version of this report the hands in the two OmniRetarget columns were misplaced: `run_omni.py bake_hands` wrote the MuJoCo-compiled geom poses (which already include the mesh re-centering `mesh_pos/mesh_quat`) back into the XML, and on reload they were re-centered again, each hand mesh piece off by 1–4 cm and rotated by up to 171°. After the fix all OmniRetarget lanes were rerun; the figures, videos and curves (`sub16_whitechair_002_report_{right,left}.{png,mp4}`, `_analysis.png`, `_approach_right.png`) have been regenerated from the new results; the old results are kept in `*_oldbake`.

After the rerun, the OmniRetarget Wuji wrist offset on this sequence is left 12.7 → 5.0, right 13.3 → 5.3 cm, pushed-off frames 0.90–0.96 → 0; BrainCo left 15.1 → 5.0 cm. Full set (535 sequences with contact): pushed-off sequences omni_wuji 19 → 10, omni_brainco 11 → 6; the two UMR lanes unchanged (62 / 23).
