"""Plot training curves from a holosoma run directory (tensorboard events) -> <run>/training_curves.png (+ CSV).

    python omomo_bench/wbt/plot_training.py logs/HiPHI_WBT/<run> [more runs ...] [--out PNG]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PANELS = [  # (title, [tags], unit scale, ylabel)
    ("reward", ["Train/mean_reward"], 1.0, ""),
    ("episode length", ["Train/mean_episode_length"], 1.0, "steps"),
    ("body pos error", ["Env/motion/error_body_pos"], 1000.0, "mm"),
    ("wrist pos error", ["Env/motion/error_wrist_pos"], 1000.0, "mm"),
    ("fingertip pos error", ["Env/motion/error_fingertip_pos", "Env/motion/error_fingertip_pos_max"], 1000.0, "mm"),
    ("finger joint error", ["Env/motion/error_finger_joint_pos"], 57.29578, "deg"),
    ("object pos error", ["Env/motion/error_object_pos"], 1000.0, "mm"),
    ("object rot error", ["Env/motion/error_object_rot"], 57.29578, "deg"),
    ("fingertip-object dist error", ["Env/motion/error_tip_obj_dist"], 1000.0, "mm"),
    ("undesired contacts (raw)", ["Env/raw_rew_undesired_contacts"], 1.0, ""),
]


def load(run: Path) -> pd.DataFrame:
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags()["scalars"]
    frames = []
    for t in tags:
        ev = acc.Scalars(t)
        frames.append(pd.DataFrame({"step": [e.step for e in ev], t: [e.value for e in ev]}).set_index("step"))
    df = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smooth", type=int, default=20, help="rolling-mean window (iterations)")
    a = ap.parse_args()
    dfs = {r.name: load(r) for r in a.runs}
    for name, df in dfs.items():
        if df.empty:
            print(f"{name}: no scalars yet"); continue
        df.to_csv(a.runs[0].parent / f"{name}_scalars.csv")
        print(f"{name}: {len(df)} iterations, tags: {len(df.columns)}")
    tags_all = set().union(*[set(d.columns) for d in dfs.values()])
    panels = [p for p in PANELS if any(t in tags_all for t in p[1])]
    if not panels:
        print("no known tags found; available:", sorted(tags_all)[:40]); sys.exit(1)
    n = len(panels); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    for ax, (title, tags, scale, unit) in zip(axes.flat, panels):
        for name, df in dfs.items():
            for t in tags:
                if t in df.columns:
                    s = df[t].dropna() * scale
                    s = s.rolling(a.smooth, min_periods=1).mean()
                    lab = name if len(tags) == 1 else f"{name} {t.split('/')[-1].replace('error_', '')}"
                    ax.plot(s.index, s.values, lw=1.2, label=lab)
        ax.set_title(title); ax.set_xlabel("iteration"); ax.set_ylabel(unit); ax.grid(alpha=0.3)
        if len(dfs) > 1 or len(tags) > 1:
            ax.legend(fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    out = a.out or (a.runs[0] / "training_curves.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
