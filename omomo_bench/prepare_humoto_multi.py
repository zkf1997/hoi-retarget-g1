#!/usr/bin/env python3
"""Multi-object HUMOTO (hand stage, step 1): select every HAND-HELD object a hand actually touches and prepare its assets.

Per sequence the object list is the primary object (data_humoto/source <seq>.npz `object`) followed by every other
hand-held object (hand_bench `handheld` flag) that is within 2 cm of a hand for >= MIN_FRAMES frames (hand_bench
`near_obj_* / near_dist_*`). Furniture (tables, shelves, chairs: handheld = False) is not included yet. Assets are copied
from hand_bench/data/objects (visual mesh + CoACD pieces) into data_humoto/objects/<obj>/ exactly like the primary.
Writes data_humoto/multi_objects.json: {seq: [primary, extra, ...]}.
    BENCH_DATASET=humoto python prepare_humoto_multi.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HB = HERE.parent / "hand_bench" / "data"
OUT = HERE / "data_humoto"
MIN_FRAMES = 10
FURNITURE = True  # step 2: include furniture (hand_bench/data/objects/<name> exported from the GLB, see README 6.7)


def prepare_object(obj: str) -> bool:
    src, dst = HB / "objects" / obj, OUT / "objects" / obj
    if (dst / "meta.json").exists():
        return True
    pieces = sorted(src.glob("coacd_*.obj"))
    if not (src / "visual.obj").exists() or not pieces:
        return False
    import trimesh
    dst.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(src / "visual.obj", force="mesh")
    mesh.export(dst / f"{obj}.obj")
    for i, p in enumerate(pieces):
        shutil.copy(p, dst / f"{obj}_coacd_{i:02d}.obj")
    (dst / "meta.json").write_text(json.dumps({"object": obj, "scale": 1.0, "center_raw": [0, 0, 0], "n_pieces": len(pieces),
                                               "extents_m": mesh.extents.tolist(), "source": "HUMOTO GLB node mesh (hand_bench/data/objects)"}, indent=1))
    return True


def main() -> None:
    index = json.loads((OUT / "seqs_test.json").read_text())
    out, hand_out, n_multi, n_extra = {}, {}, 0, 0
    for e in index:
        seq = e["seq"]
        src = np.load(OUT / "source" / f"{seq}.npz", allow_pickle=True)
        hb = np.load(HB / "seqs" / f"{seq}.npz", allow_pickle=True)
        T = min(len(src["trans"]), int(hb["n_frames"]))
        objs = [str(o) for o in hb["objects"]]
        primary = str(src["object"])
        sel = [primary]
        for i, o in enumerate(objs):
            if o == primary or not bool(hb["handheld"][i]):
                continue
            n = sum(int(((hb[f"near_obj_{sd}"][:T] == i) & (hb[f"near_dist_{sd}"][:T] < 0.02)).sum()) for sd in ("left", "right"))
            if n >= MIN_FRAMES and prepare_object(o):
                sel.append(o)
        hand_only = list(sel)
        if FURNITURE:  # step 2: every piece of furniture in the scene (static or moved), for hand/forearm and body non-penetration + contacts
            for i, o in enumerate(objs):
                if o not in sel and not bool(hb["handheld"][i]) and prepare_object(o):
                    sel.append(o)
        hand_out[seq] = hand_only
        out[seq] = sel
        if len(sel) > 1:
            n_multi += 1; n_extra += len(sel) - 1
    (OUT / "multi_objects.json").write_text(json.dumps(out, indent=1))
    (OUT / "multi_objects_handheld.json").write_text(json.dumps(hand_out, indent=1))  # step-1 list (hand-held objects only)
    print(f"{len(out)} sequences, {n_multi} with extra objects ({n_extra} extra hand-held objects + furniture) -> {OUT / 'multi_objects.json'}")


if __name__ == "__main__":
    main()
