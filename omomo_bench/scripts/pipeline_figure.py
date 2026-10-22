#!/usr/bin/env python3
"""Block diagram of the final retargeting pipeline -> reports/figures/pipeline_overview.png (matplotlib only).
    python scripts/pipeline_figure.py [--out PATH]"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

_ap = argparse.ArgumentParser(description=__doc__); _ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "reports" / "figures" / "pipeline_overview.png"))
OUT = Path(_ap.parse_args().out)
INK, MUTED, SURF = "#1b1b1b", "#5b5b5b", "#fcfcfb"
FILL = {"data": "#e8eef8", "body": "#dcebdc", "hand": "#f6e7d2", "final": "#e9dff2", "eval": "#eeeeee"}
EDGE = {"data": "#2a78d6", "body": "#1baf7a", "hand": "#eb6834", "final": "#7a4fb0", "eval": "#888888"}

fig, ax = plt.subplots(figsize=(19, 7.6), dpi=170, facecolor=SURF)
ax.set_xlim(0, 19); ax.set_ylim(0, 7.6); ax.axis("off"); ax.set_facecolor(SURF)

def box(x, y, w, h, title, body, kind, fs=8.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + 0.14, y + h - 0.22, title, fontsize=fs + 1, fontweight="bold", color=INK, va="top", ha="left")
    ax.text(x + 0.14, y + h - 0.58, body, fontsize=fs, color=INK, va="top", ha="left", linespacing=1.35)

def arrow(x0, y0, x1, y1, label=None, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.3, color=color, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.13, label, fontsize=8, color=MUTED, ha="center", va="bottom")

# inputs
box(0.3, 4.3, 4.3, 2.9, "Source (per sequence)",
    "SMPL-X body (+ fingers for HUMOTO / HiPHI),\n55 joints, object 6-DoF trajectory, 30 fps,\nobject mesh + CoACD convex pieces\n\nScaled world: human and objects x s,\ns = 1.32 / T-pose height; robot at real size", "data")
box(5.1, 4.3, 4.3, 2.9, "Stage 1  UMR body  (run_umr.py)",
    "surface-slot correspondence + contact term,\nslack non-penetration (object, ground),\nfingers pinned at a rest pose\nfinger collision OFF (OMOMO, HUMOTO),\nON for HiPHI (large objects)\nout: 76-dim qpos -> results*/umr_wuji*", "body")
box(9.9, 5.5, 4.3, 1.7, "Stage 2  AnyTeleop fingers",
    "hand_bench/run_anyteleop.py: human finger\nkeypoints -> 20 joints per hand at the\nstage-1 wrist pose   (HUMOTO, HiPHI)", "hand")
box(14.7, 4.3, 4.0, 2.9, "Stage 3  Coupled refinement v3",
    "hand_pipeline.py --stage s2 v3, 12 iterations\nper arm: 7 arm + 20 finger joints, one QP\nwrist anchor (relaxed by contact),\nwrist orientation = canonical hand frame,\nfingertip targets on the object surface,\nhand + forearm non-penetration (slack)\n-> results*/pipe_final", "final")
box(9.9, 2.6, 8.8, 1.6, "OMOMO (no finger data): coupled arm + finger repair  (omomo_hand_repair.py --coupled)",
    "per frame and hand: the same 27 joints re-solved with FULL collision, wrist anchored to the SMPL-X hand frame,\nrest-pose finger prior, no contact targets, 8 iterations   -> results/pipe_final", "hand")
box(5.1, 0.3, 13.6, 1.5, "Evaluation and viewers",
    "eval_pipeline.py: contact F1, fingertip-object distance, hand / body / scene penetration, flange pose, jerk      compare.py: body metrics\n"
    "wrist_gap_stats.py, wrist_orientation_stats.py      viser_compare.py: web viewer      render_compare.py: PNG grids / mp4", "eval")

arrow(4.6, 5.75, 5.1, 5.75)
arrow(9.4, 6.3, 9.9, 6.3, "HUMOTO, HiPHI")
arrow(14.2, 6.3, 14.7, 6.3)
arrow(9.4, 4.8, 9.9, 3.9, "OMOMO")
arrow(16.7, 4.3, 16.7, 1.8)
arrow(12.0, 2.6, 12.0, 1.8)
ax.text(0.3, 1.0, "Final lanes\nHUMOTO  pipe_v3_nfcfull\nHiPHI   pipe_v3_it12\nOMOMO   pipe_handrepair_coupled", fontsize=8.8, color=INK, va="center", ha="left", family="monospace")
fig.suptitle("Whole-body HOI retargeting to G1 + Wuji Hand 2: final pipeline (2026-10)", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT, facecolor=SURF); print("wrote", OUT)
