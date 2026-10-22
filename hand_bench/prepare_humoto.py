#!/usr/bin/env python3
"""HUMOTO -> benchmark inputs.

    python prepare_humoto.py --max-seqs 60 --per-object 2          # default selection
    python prepare_humoto.py --seqs carry_hold_mug_with_left_hand_below-797 ...

For every selected sequence writes data/seqs/<seq>.npz:
    hand_kp_{left,right} (T,21,3) world Z-up metres, MediaPipe order      hand_R_{left,right} (T,3,3) canonical hand frame (humoto_io.hand_frame)
    body_pos (T,25,3)  objects [K names]  obj_pos (T,K,3) obj_rot (T,K,3,3) obj_scale (T,K)  handheld (K,) bool
    near_obj_{side} (T,) index of the closest hand-held object (-1: none)   near_dist_{side} (T,) min keypoint->object-surface distance
and for every hand-held object that occurs: data/objects/<obj>/visual.obj (GLB mesh, node-local frame) + coacd_*.obj convex pieces.
Selection: sequences that contain at least one hand-held object (mesh extent <= --handheld-max-extent), ranked by the fraction of
frames in which a hand is within --contact-dist of it; up to --per-object per object, --max-seqs in total.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import bench_common as bc
import humoto_io as H

HERE = Path(__file__).resolve().parent
DATA, SEQ_DIR, OBJ_DIR = bc.DATA, bc.SEQ_DIR, bc.OBJ_DIR


def handheld_objects(max_extent: float) -> dict[str, bool]:
    """object name -> hand-held? (mesh extent from the first sequence that contains it)."""
    first = {}
    for s in H.list_sequences():
        for o in H.load_meta(s)["objects"]:
            first.setdefault(o, s)
    out = {}
    for o, s in first.items():
        try:
            out[o] = float(H.load_object_mesh(s, o).extents.max()) <= max_extent
        except Exception:
            out[o] = False
    return out


def _surface_points(mesh: trimesh.Trimesh, n: int = 1500) -> np.ndarray:
    pts = mesh.vertices
    if len(pts) > n:
        pts = pts[np.random.RandomState(0).choice(len(pts), n, replace=False)]
    return np.vstack([pts, mesh.sample(n, seed=0)]) if len(mesh.faces) else pts


def process(seq: str, handheld: dict[str, bool], contact_dist: float, load_sequence=H.load_sequence, load_mesh=H.load_object_mesh) -> dict:
    d = load_sequence(seq)
    T = d["n_frames"]
    objs = d["objects"]
    hh = np.array([handheld.get(o, False) for o in objs], dtype=bool)
    pts = {o: _surface_points(load_mesh(seq, o)) for o, h in zip(objs, hh) if h}
    out = {"seq": seq, "n_frames": T, "fps": d["fps"], "body_pos": d["body_pos"].astype(np.float32),
           "objects": np.array(objs), "obj_pos": d["obj_pos"].astype(np.float32), "obj_rot": d["obj_rot"].astype(np.float32),
           "obj_scale": d["obj_scale"].astype(np.float32), "handheld": hh}
    stats = {"seq": seq, "n_frames": T, "objects": objs, "handheld": [o for o, h in zip(objs, hh) if h]}
    for side in ("left", "right"):
        kp = d["hand_kp"][side]
        out[f"hand_kp_{side}"] = kp.astype(np.float32)
        out[f"hand_R_{side}"] = H.hand_frame(kp).astype(np.float32)
        near = -np.ones(T, dtype=np.int64)
        dist = np.full(T, np.inf, dtype=np.float32)
        for k, o in enumerate(objs):
            if not hh[k]:
                continue
            P = pts[o]
            tree = cKDTree(P)
            # bring the hand into the object frame instead of transforming the surface points every frame
            local = np.einsum("tji,tkj->tki", d["obj_rot"][:, k], kp - d["obj_pos"][:, None, k]) / d["obj_scale"][:, k, None, None]
            dm = tree.query(local.reshape(-1, 3), k=1)[0].reshape(T, -1).min(1) * d["obj_scale"][:, k]
            better = dm < dist
            dist[better], near[better] = dm[better], k
        out[f"near_obj_{side}"], out[f"near_dist_{side}"] = near, dist
        stats[f"contact_frac_{side}"] = float((dist < contact_dist).mean())
    stats["contact_frac"] = float(((out["near_dist_left"] < contact_dist) | (out["near_dist_right"] < contact_dist)).mean())
    return out, stats


def _job(args):
    seq, handheld, contact_dist = args
    try:
        out, stats = process(seq, handheld, contact_dist)
    except Exception as e:  # noqa: BLE001
        return seq, None, {"seq": seq, "error": repr(e)}
    return seq, out, stats


def export_object(seq: str, obj: str, threshold: float, max_hulls: int, load_mesh=H.load_object_mesh) -> None:
    import coacd

    coacd.set_log_level("error")

    od = OBJ_DIR / obj
    if (od / "pieces.json").exists():
        return
    od.mkdir(parents=True, exist_ok=True)
    m = load_mesh(seq, obj)
    m.export(od / "visual.obj")
    cm = coacd.Mesh(np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64))
    parts = coacd.run_coacd(cm, threshold=threshold, max_convex_hull=max_hulls, preprocess_mode="auto", seed=0)
    names = []
    for i, (v, f) in enumerate(parts):
        hull = trimesh.Trimesh(v, f, process=False).convex_hull
        hull.export(od / f"coacd_{i:02d}.obj")
        names.append(f"coacd_{i:02d}.obj")
    (od / "pieces.json").write_text(json.dumps({"visual": "visual.obj", "pieces": names, "extent": m.extents.tolist(),
                                               "watertight": bool(m.is_watertight), "source_seq": seq}, indent=1))
    print(f"  object {obj}: {len(m.faces)} faces -> {len(names)} convex pieces")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="every sequence with a hand-held object")
    ap.add_argument("--max-seqs", type=int, default=60)
    ap.add_argument("--per-object", type=int, default=2)
    ap.add_argument("--handheld-max-extent", type=float, default=0.55)
    ap.add_argument("--contact-dist", type=float, default=0.02)
    ap.add_argument("--coacd-threshold", type=float, default=0.05)
    ap.add_argument("--coacd-max-hulls", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    handheld = handheld_objects(a.handheld_max_extent)
    (DATA / "handheld_objects.json").write_text(json.dumps(handheld, indent=1, sort_keys=True))
    print(f"{sum(handheld.values())}/{len(handheld)} object types are hand-held (extent <= {a.handheld_max_extent} m)")

    if a.seqs:
        cands = a.seqs
    else:
        cands = [s for s in H.list_sequences() if any(handheld.get(o, False) for o in H.load_meta(s)["objects"])]
    print(f"parsing {len(cands)} candidate sequences ...")
    results = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for seq, out, stats in ex.map(_job, [(s, handheld, a.contact_dist) for s in cands], chunksize=4):
            results[seq] = (out, stats)
            if out is None:
                print("  FAILED", seq, stats["error"])

    ok = {s: v for s, v in results.items() if v[0] is not None}
    if a.seqs or a.all:
        selected = list(ok)
    else:  # balanced pick: per hand-held object, the sequences with the largest contact fraction
        by_obj: dict[str, list] = {}
        for s, (out, st) in ok.items():
            for o in st["handheld"]:
                by_obj.setdefault(o, []).append((st["contact_frac"], s))
        picked: dict[str, float] = {}
        for o, lst in sorted(by_obj.items(), key=lambda kv: -len(kv[1])):
            for frac, s in sorted(lst, reverse=True)[: a.per_object]:
                picked[s] = max(picked.get(s, 0.0), frac)
        selected = [s for s, _ in sorted(picked.items(), key=lambda kv: -kv[1])[: a.max_seqs]]

    total = 0
    for s in selected:
        out, st = ok[s]
        np.savez_compressed(SEQ_DIR / f"{s}.npz", **out)
        total += out["n_frames"]
    (DATA / "selection.json").write_text(json.dumps({"selected": selected, "stats": {s: ok[s][1] for s in selected},
                                                     "args": vars(a)}, indent=1))
    print(f"selected {len(selected)} sequences, {total} frames -> {SEQ_DIR}")

    objs = sorted({o for s in selected for o in ok[s][1]["handheld"]})
    print(f"exporting {len(objs)} hand-held objects (visual + CoACD) ...")
    for o in objs:
        src = next(s for s in selected if o in ok[s][1]["handheld"])
        export_object(src, o, a.coacd_threshold, a.coacd_max_hulls)


if __name__ == "__main__":
    main()
