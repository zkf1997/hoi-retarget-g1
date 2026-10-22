"""Cut the WBT motions to short interaction clips (same windows as the SPIDER clips, spider_bridge/build_hiphi.py --window).

    ~/software/venvs/spider/bin/python omomo_bench/wbt/make_clips.py [--seconds 5]
        -> omomo_bench/wbt/data/motions_w5/<tag>.npz   (train / eval with HIPHI_WBT_CLIPS=w5)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spider_bridge"))
from build_hiphi import interaction_window  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()
    src = HERE / "data" / "motions"
    dst = HERE / "data" / f"motions_w{a.seconds:g}"
    dst.mkdir(parents=True, exist_ok=True)
    starts = {}
    for p in sorted(src.glob("*.npz")):
        d = dict(np.load(p))
        T = d["joint_pos"].shape[0]
        s = interaction_window(d, a.seconds)
        n = int(round(a.seconds * int(d["fps"])))
        out = {k: (v[s:s + n] if v.ndim and v.shape[0] == T else v) for k, v in d.items()}
        np.savez(dst / p.name, **out)
        starts[p.stem] = s
    (dst / "clip_starts.json").write_text(json.dumps(starts, indent=1))
    print(f"wrote {len(starts)} clips to {dst}")


if __name__ == "__main__":
    main()
