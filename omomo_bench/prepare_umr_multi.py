#!/usr/bin/env python3
"""UMR input variant with every object of the sequence (multi-object step 3): data_humoto/umr_multi/test/<seq>/.

Everything of data_humoto/umr/test/<seq>/ is symlinked (primary object xml + prop csv, body motion). For every extra object
of data_humoto/multi_objects.json (hand-held objects + furniture, prepare_humoto_multi.py) an <obj>.xml (same format as
the primary, CoACD / surface-patch pieces from data_humoto/objects) and a prop_<obj>.csv (source extra_obj_pos /
extra_obj_quat_xyzw, same frames and units as the primary's csv) are written. Our UMR patch loads all of them when
solver.object_multi is true (configs/umr_g1_wuji_multi.json).
    BENCH_DATASET=humoto python prepare_umr_multi.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prepare_omomo import object_mjcf

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_humoto"
SRC, DST = DATA / "umr", DATA / "umr_multi"


def main() -> None:
    multi = json.loads((DATA / "multi_objects.json").read_text())
    (DST / "object_assets").mkdir(parents=True, exist_ok=True)
    n_seq = n_obj = 0
    for seq, objs in multi.items():
        sd = SRC / "test" / seq
        if not sd.exists():
            continue
        out = DST / "test" / seq; out.mkdir(parents=True, exist_ok=True)
        for f in sd.iterdir():
            if not (out / f.name).exists():
                (out / f.name).symlink_to(Path("../../..") / "umr" / "test" / seq / f.name)
        src = np.load(DATA / "source" / f"{seq}.npz", allow_pickle=True)
        T = len(src["trans"])
        extra = [str(x) for x in src["extra_objects"]] if "extra_objects" in src else []
        for o in objs[1:]:
            link = DST / "object_assets" / o
            if not link.exists():
                link.symlink_to(Path("../..") / "objects" / o)
            meta = json.loads((DATA / "objects" / o / "meta.json").read_text())
            # material names must be unique once several object MJCFs are merged into one model
            (out / f"{o}.xml").write_text(object_mjcf(o, meta, meshdir=f"../../object_assets/{o}").replace("\"obj_visual\"", f"\"{o}_visual_mat\""))
            k = extra.index(o)
            pos, quat = src["extra_obj_pos"][:T, k], src["extra_obj_quat_xyzw"][:T, k]
            with (out / f"prop_{o}.csv").open("w") as fh:
                fh.write("px,py,pz,qx,qy,qz,qw\n")
                for p, q in zip(pos, quat):
                    fh.write(",".join(f"{v:.6f}" for v in (*p, *q)) + "\n")
            n_obj += 1
        n_seq += 1
    print(f"{n_seq} sequences -> {DST} ({n_obj} extra objects written)")


if __name__ == "__main__":
    main()
