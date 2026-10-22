#!/usr/bin/env python3
"""Create data/umr_omniscale/ from data/umr/: the UMR input variant that reproduces OmniRetarget's object placement.

UMR multiplies object positions by smpl_scale s uniformly. OmniRetarget scales xy by s but anchors z at the frame-0
height: z_omni = z0 + s (z - z0). Feeding z' = z - z0 + z0 / s makes UMR's s * z' equal z_omni exactly.
Everything else (poses, betas, object XML) is symlinked. s per subject is read from an existing results/umr_none/*.npz
(UMR's own smpl_scale; falls back to 1.32 / T-pose height from data/omni/height_dict.pkl).
"""
from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SRC, DST = DATA / "umr", DATA / "umr_omniscale"


def subject_scale(sub: str) -> float:
    for d in ("umr_none", "umr"):
        for p in sorted((HERE / "results" / d).glob(f"{sub}_*.npz")) if (HERE / "results" / d).exists() else []:
            return float(np.asarray(np.load(p, allow_pickle=True)["smpl_scale"]).reshape(-1)[0])
    return 1.32 / pickle.load(open(DATA / "omni" / "height_dict.pkl", "rb"))[sub]


def main() -> None:
    DST.mkdir(exist_ok=True)
    link = DST / "object_assets"
    if not link.exists():
        link.symlink_to(Path("..") / "umr" / "object_assets")
    scales = {}
    n = 0
    for seq_dir in sorted((SRC / "test").iterdir()):
        seq = seq_dir.name
        sub = seq.split("_")[0]
        s = scales.setdefault(sub, subject_scale(sub))
        out = DST / "test" / seq
        out.mkdir(parents=True, exist_ok=True)
        for f in seq_dir.iterdir():
            if f.name.startswith("prop_"):
                rows = list(csv.DictReader(f.open()))
                z = np.array([float(r["pz"]) for r in rows])
                z_new = z - z[0] + z[0] / s
                with (out / f.name).open("w") as fh:
                    fh.write("px,py,pz,qx,qy,qz,qw\n")
                    for r, zz in zip(rows, z_new):
                        fh.write(",".join([r["px"], r["py"], f"{zz:.6f}", r["qx"], r["qy"], r["qz"], r["qw"]]) + "\n")
            else:
                dst = out / f.name
                if not dst.exists():
                    dst.symlink_to(Path("../../..") / "umr" / "test" / seq / f.name)
        n += 1
    print(f"{n} sequences -> {DST}; subject scales {scales}")


if __name__ == "__main__":
    main()
