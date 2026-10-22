#!/usr/bin/env python3
"""HiPHI (HOI subset, originals only) -> benchmark inputs, same format as prepare_humoto.py.

    BENCH_DATASET=hiphi python prepare_hiphi.py --per-category 5 --workers 20
    BENCH_DATASET=hiphi python prepare_hiphi.py --seqs Adjusting-adjust_0001 Bringing-carry_0001

Every HOI motion has exactly one tracked object (ball/box/bucket/chair/table/...), which is used as the interaction object.
BVH is strided 90 -> 30 Hz. Selection: per object category the --per-category originals with the largest fraction of frames in
which a hand is within --contact-dist of the object. Writes data_hiphi/seqs/<motion_id>.npz, data_hiphi/objects/<mesh_id>/.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("BENCH_DATASET", "hiphi")

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import bench_common as bc  # noqa: E402
import hiphi_io as HP  # noqa: E402
import prepare_humoto as P  # noqa: E402

assert bc.DATASET == "hiphi", "run with BENCH_DATASET=hiphi"


def _job(args):
    seq, contact_dist = args
    try:
        meta = HP.load_meta(seq)
        handheld = {o["mesh_id"]: True for o in meta["objects"]}
        out, stats = P.process(seq, handheld, contact_dist, load_sequence=HP.load_sequence, load_mesh=lambda s, o: HP.load_object_mesh(o))
        stats["category"] = meta["objects"][0]["object_category"]
        stats["frame_lu"] = meta["frame_lu"]
        stats["text"] = meta["text_annotation"]
    except Exception as e:  # noqa: BLE001
        return seq, None, {"seq": seq, "error": repr(e)}
    return seq, out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=None, help="debug: only parse the first N candidates")
    ap.add_argument("--contact-dist", type=float, default=0.03)
    ap.add_argument("--coacd-threshold", type=float, default=0.05)
    ap.add_argument("--coacd-max-hulls", type=int, default=16)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    bc.SEQ_DIR.mkdir(parents=True, exist_ok=True)

    md = HP.list_motions()
    cands = a.seqs or md.motion_id.tolist()
    if a.max_candidates:
        cands = cands[: a.max_candidates]
    print(f"parsing {len(cands)} HOI motions ...", flush=True)
    results = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (seq, out, stats) in enumerate(ex.map(_job, [(s, a.contact_dist) for s in cands], chunksize=4)):
            results[seq] = (out, stats)
            if out is None:
                print("  FAILED", seq, stats["error"], flush=True)
            if i % 500 == 0:
                print(f"  {i}/{len(cands)}", flush=True)
    ok = {s: v for s, v in results.items() if v[0] is not None}
    if a.seqs:
        selected = list(ok)
    else:
        by_cat: dict[str, list] = {}
        for s, (out, st) in ok.items():
            by_cat.setdefault(st["category"], []).append((st["contact_frac"], s))
        selected = [s for lst in by_cat.values() for _, s in sorted(lst, reverse=True)[: a.per_category]]
    total = 0
    for s in selected:
        np.savez_compressed(bc.SEQ_DIR / f"{s}.npz", **ok[s][0])
        total += ok[s][0]["n_frames"]
    (bc.DATA / "selection.json").write_text(json.dumps({"selected": selected, "stats": {s: ok[s][1] for s in selected}, "args": vars(a)}, indent=1))
    (bc.DATA / "candidates.json").write_text(json.dumps({s: st for s, (_, st) in results.items()}, indent=1))
    print(f"selected {len(selected)} motions, {total} frames @30 Hz -> {bc.SEQ_DIR}")
    objs = sorted({o for s in selected for o in ok[s][1]["handheld"]})
    print(f"exporting {len(objs)} objects (visual + CoACD) ...", flush=True)
    for o in objs:
        P.export_object(None, o, a.coacd_threshold, a.coacd_max_hulls, load_mesh=lambda s, oo: HP.load_object_mesh(oo))


if __name__ == "__main__":
    main()
