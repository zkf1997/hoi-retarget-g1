# UMR vs OmniRetarget on OMOMO / HUMOTO / HiPHI (2026-10-16, after the hand-bake fix)

Body stage only (fingers at rest pose), UMR-style scaling, every lane evaluated on the same CoACD object pieces.
OMOMO 602 seqs, HUMOTO 100, HiPHI 60 (per-sequence heights; Wuji only). Sources: `*/summary_3ds.md`, `*/metrics_3ds_part?.csv`,
`*/wrist_gap_3ds.log`, `*/wrist_orientation_3ds.log`; driver `results/run_umr_vs_omni_3ds.sh`, `results/run_3ds_fixup.sh`.
Omni = OmniRetarget with CoACD collision (our default); Omni-hull = holosoma default (whole mesh as one geom -> convex hull).

## G1 + Wuji

| metric | OMOMO Omni | OMOMO Omni-hull | OMOMO UMR | HUMOTO Omni | HUMOTO Omni-hull | HUMOTO UMR | HiPHI Omni | HiPHI Omni-hull | HiPHI UMR |
|---|---|---|---|---|---|---|---|---|---|
| object penetration (mm) | 10.1 | **2.1** | 5.4 | 10.4 | **4.1** | 4.5 | 13.5 | 10.5 | **5.7** |
| frames > 5 mm | 0.42 | **0.09** | 0.43 | 0.60 | **0.21** | 0.32 | 0.47 | **0.33** | 0.44 |
| seqs needing a fallback | 359 | 39 | 0 | 71 | 26 | 0 | 32 | 22 | 0 |
| hand-contact preservation | **0.97** | 0.83 | 0.95 | **1.00** | 0.98 | 0.87 | 0.96 | **0.97** | 0.96 |
| wrist position error (cm) | **2.2** | 3.5 | 3.5 | **1.6** | 2.1 | 3.7 | **2.9** | 3.4 | 3.0 |
| wrist orientation error (deg, mean) | 32 | 37 | **13** | 39 | 41 | **10** | 47 | 49 | **14** |
| frames > 30 deg | 43% | 49% | **6%** | 59% | 60% | **5%** | 67% | 70% | **3%** |
| hand pushed off (seqs, any hand > 50% frames) | **10** | 59 | 62 | **0** | **0** | 16 | 6 | **4** | 5 |
| ground penetration (mm) | 4.3 | 1.6 | **0.2** | 4.2 | 2.0 | **0.05** | 2.4 | 3.8 | **0.2** |
| foot slide (mm) | **1.0** | 1.1 | 2.6 | **1.0** | **1.0** | 2.8 | **0.9** | 0.9 | 1.9 |
| ankle error (cm) | 4.3 | **4.3** | 5.2 | 4.6 | 4.7 | **3.6** | 5.1 | 5.4 | **4.7** |
| joint jerk | **201** | 232 | 273 | **238** | 244 | 258 | **140** | 146 | 193 |
| root jerk | 56 | 61 | **14** | 62 | 64 | **19** | 35 | 36 | **10** |
| runtime (s / frame) | 2.9 | 1.1 | **0.25** | 1.1 | 0.7 | **0.15** | 2.3 | 1.2 | **0.10** |

BrainCo (OMOMO / HUMOTO) follows the same pattern: Omni pen 9.7 / 10.7 mm, UMR 5.3 / 5.4; contact 0.98 / 1.00 vs 0.97 / 0.90; wrist 2.1 / 1.5 vs 3.4 / 3.0 cm;
orientation 32 / 39 deg vs 12 / 9 deg; pushed-off seqs 6 / 0 vs 23 / 14.

## Where OmniRetarget's penetration comes from (Wuji, mm, by fallback mode)

| | OMOMO Omni | OMOMO Omni-hull | HUMOTO Omni | HUMOTO Omni-hull | HiPHI Omni | HiPHI Omni-hull |
|---|---|---|---|---|---|---|
| default (hard non-penetration) | 243 seqs, 0.7 | 563 seqs, 1.3 | 29, 0.9 | 74, 1.2 | 28, 0.9 | 38, 1.8 |
| pen_tol_0.02 | 103, 12.6 | 23, 11.3 | 48, 14.3 | 16, 12.8 | - | 7, 16.0 |
| fingers_free | 145, 15.2 | 7, 9.4 | 9, 9.6 | 6, 10.4 | 10, 20.5 | 2, 23.1 |
| fingers_free_pen_tol_0.02 | 51, 19.1 | 1, 6.8 | 8, 18.9 | 3, 13.2 | 4, 20.0 | 1, 19.8 |
| no_obj_nonpen | 60, 24.6 | 8, 21.2 | 6, 15.0 | 1, 14.7 | 18, 27.8 | 12, 31.9 |
| share of total penetration from fallback seqs | 97% | 60% | 97% | 79% | 97% | 89% |

When the hard constraint is feasible OmniRetarget is nearly penetration-free (< 2 mm); its average penetration is almost entirely the fallback
sequences. With CoACD the constraint is infeasible on 60% (OMOMO) / 71% (HUMOTO) / 53% (HiPHI) of the sequences: rest-pose fingers plus many small
convex pieces leave no feasible pose when the hand wraps a handle or rim.

## CoACD or convex hull for OmniRetarget

Keep CoACD as the main OmniRetarget lane, report the hull as "holosoma default".
- The hull makes the hard constraint feasible far more often (fallbacks 359 -> 39 on OMOMO), hence the low penetration, but it is a different
  object: concavities are filled, so the hands stay outside the hull. OMOMO contact drops 0.97 -> 0.83 (floor lamp 1.00 -> 0.03, clothes stand
  0.98 -> 0.46), wrist error 2.2 -> 3.5 cm, pushed-off seqs 10 -> 59 (the hand is held off the true surface), orientation and joint jerk worse.
- On HUMOTO (small hand-held objects, mostly convex) the two differ little; on HiPHI (large furniture) the hull still needs 22 fallbacks and only
  lowers penetration to 10.5 mm, with worse wrist error; StepStool / Clothrack penetration gets much worse with the hull.
- So the hull trades interaction fidelity for a feasibility the evaluation then rewards. For a HOI retargeting comparison the CoACD lane is the
  fair one (same geometry as the evaluation and as UMR's surface terms).

## Conclusions

1. Wrist / hand placement: OmniRetarget tracks the wrist position better (2.2 / 1.6 / 2.9 cm vs UMR 3.5 / 3.7 / 3.0) and keeps contact, and it no
   longer pushes hands through thin rails after the fix (pushed-off seqs 10 vs UMR 62 on OMOMO, 0 vs 16 on HUMOTO). UMR's "hand on the far side"
   defect is the fixed-finger + slack non-penetration problem analysed in `whitechair_002_wrong_side.md`; the two-stage repair fixes it for OMOMO.
2. Wrist orientation: UMR is 3-4x better (10-14 deg vs 32-47 deg; > 30 deg in 3-6% vs 43-67% of frames). OmniRetarget has no orientation term
   (15 keypoints), so the hand rolls freely about the forearm.
3. Penetration: UMR 4.5-5.7 mm everywhere and never fails; OmniRetarget is < 2 mm when its hard constraint is feasible but falls back on 53-71% of
   sequences, ending at 10-13.5 mm on average.
4. Body quality: UMR has near-zero ground penetration (0.05-0.2 vs 2.4-4.3 mm) and 3-4x lower root jerk, but 2-3x more foot slide; joint jerk is
   lower for OmniRetarget (201 / 238 / 140 vs 273 / 258 / 193).
5. Speed: UMR 10-20x faster (0.1-0.25 vs 1.1-2.9 s per frame).
6. Hands (BrainCo vs Wuji): same ranking; Wuji is worse for UMR on OMOMO (62 vs 23 pushed-off seqs) because its fingers are longer.

Notes: OMOMO orientation uses the flat SMPL-X hand frame (no finger data). `wrist_orientation_stats.py` evaluated UMR BrainCo with the Wuji MJCF
before 2026-10-16 (right hand 97 deg was that bug; now 9 deg). Final pipelines (UMR body + hand stage) are separate: see README section 5.
