# UMR body stage: finger collision on vs off (palm / forearm collision kept), three datasets (2026-10-17)

Lanes: `umr_wuji` (full hand collision, default) vs `umr_wuji_nofingercol` (`run_umr.py --variant nofingercol`: the 50 finger geoms of
`robots/g1_29dof_wuji_hand2_nofingercol.xml` have contype/conaffinity 0; palm block, mount and forearm still collide). OMOMO also lists
OmniRetarget with CoACD pieces and with holosoma's convex hull. All lanes evaluated on the same CoACD object pieces with the FULL collision
model, so finger penetration of the finger-collision-off lane is counted. Files: `*/summary_nfc.md`, `*/metrics_nfc_all.csv`,
`*/wrist_gap_nfc.{csv,log}`, `*/wrist_orientation_nfc.log`; driver `results/run_nfc_compare.sh`.

## OMOMO (602 seqs, G1 + Wuji)

| metric | Omni + convex hull | Omni + CoACD | UMR full collision | UMR fingers off |
|---|---|---|---|---|
| object penetration, all links (mm) / frames > 5 mm | **2.1 / 0.09** | 10.1 / 0.42 | 5.4 / 0.43 | 14.2 / 0.67 |
| hand penetration on contact frames (mm) / > 5 mm | 0.05 / 0 | 0.05 / 0 | 5.8 / 0.44 | 17.0 / 0.83 |
| seqs needing a fallback | 39 | 359 | 0 | 0 |
| hand-contact preservation | 0.83 | 0.97 | 0.95 | **1.00** |
| wrist position error (cm) | 3.5 | **2.2** | 3.5 | 2.5 |
| wrist orientation error (deg) / frames > 30 deg | 37 / 49% | 32 / 43% | 13 / 6% | **8 / 0.3%** |
| hand pushed across the object (seqs) | 59 | 10 | 57 | **5** |
| ground penetration (mm) | 1.6 | 4.3 | **0.2** | **0.2** |
| foot slide (mm / frame) | 1.1 | **1.0** | 2.6 | 2.4 |
| joint jerk / root jerk | 232 / 61 | 201 / 56 | 273 / 14 | **181 / 10** |
| runtime (s / frame) | 1.1 | 2.9 | 0.25 | **0.07** |

Per sequence, fingers-off beats full collision on wrist error in 94% and on joint jerk in 94% of the sequences; object penetration is worse in 91%.
After the coupled arm+finger repair with full collision (`pipe_handrepair_coupled`, OMOMO final): object penetration 1.3 mm / 0.07, hand contact 0.985,
wrist error 2.3 cm, joint jerk 189, root jerk 10.3, pushed-off 5 seqs (`summary_handrepair_coupled.md`, `wrist_gap_omomo_coupled.csv`).

## HUMOTO (100) and HiPHI (60): UMR body stage

| metric | HUMOTO full | HUMOTO fingers off | HiPHI full | HiPHI fingers off |
|---|---|---|---|---|
| object penetration, all links (mm) / frames > 5 mm | **4.5 / 0.32** | 9.8 / 0.64 | **5.7 / 0.44** | 16.0 / 0.79 |
| hand penetration on contact frames (mm) | **3.2** | 8.1 | **3.4** | 12.8 |
| hand-contact preservation | 0.87 | **0.97** | 0.96 | **0.99** |
| wrist position error (cm) | 3.7 | **2.7** | 3.0 | **2.2** |
| wrist orientation error (deg) / frames > 30 deg | 9.9 / 4.8% | **6.8 / 0.2%** | 14.1 / 3.3% | **11.6 / 0.3%** |
| hand pushed across the object (seqs) | 11 | **0** | 4 | 2 |
| joint jerk / root jerk | 258 / 19 | **170 / 16** | 193 / 10 | **130 / 9** |
| per-seq wins of fingers-off: wrist / jerk | 91% / 100% | | 88% / 98% | |

Final pipeline (AnyTeleop + v3, 12 iterations) on each body (`pipe_summary_nfcfull.md`):

| final | HUMOTO on full | HUMOTO on fingers off | HiPHI on full | HiPHI on fingers off |
|---|---|---|---|---|
| fingertip contact F1 | 0.771 | **0.799** | **0.577** | 0.572 |
| hand penetration (mm) / > 5 mm | 1.00 / 0.059 | **0.89 / 0.046** | **0.66 / 0.016** | 0.81 / 0.031 |
| whole-body object penetration (mm) | 1.7 | **1.5** | **3.1** | 3.4 |
| flange position (cm) / rotation (deg) | 2.54 / 4.6 | **2.42 / 4.0** | 3.04 / 6.0 | **3.01 / 5.9** |
| joint jerk / root jerk | 197 / 19 | **167 / 16** | 134 / 10 | **113 / 9** |

## Reading

- Rest-pose fingers carry no information in the body stage (wrong length, wrong pose); letting them collide only moves the hand: wrist error
  3.5 -> 2.5 / 3.7 -> 2.7 / 3.0 -> 2.2 cm, orientation 13 -> 8 / 10 -> 7 / 14 -> 12 deg, pushed-off 57 -> 5 / 11 -> 0 / 4 -> 2 seqs, joint jerk
  -30% on every dataset, root jerk also lower (the arm no longer drags the torso).
- The price is finger penetration in the body stage (hand penetration 17 / 8 / 13 mm), so the lane is only usable with a stage that owns finger
  collision: v3 on HUMOTO removes it completely (final hand penetration 0.89 mm) and the whole final is better; the coupled repair on OMOMO brings
  it to 1.2 mm. On HiPHI's large objects the hand is pressed into boxes / buckets (12.8 mm) and v3 does not remove all of it (0.66 -> 0.81 mm),
  contact F1 is flat, smoothness is better: a trade-off, default kept at full collision there.
- Against OmniRetarget on OMOMO: fingers-off UMR matches Omni + CoACD on wrist position (2.5 vs 2.2 cm) and pushed-off seqs (5 vs 10), beats it on
  orientation (8 vs 32 deg), ground penetration, root jerk and speed, and after the repair also on penetration (1.3 vs 10.1 mm). Omni + hull has the
  lowest raw penetration because the hull is a bigger object: contact 0.83, 59 pushed-off seqs, orientation 37 deg.
