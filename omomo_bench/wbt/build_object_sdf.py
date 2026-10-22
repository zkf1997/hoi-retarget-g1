"""Precompute a signed-distance grid for every object URDF in assets/objects/<tag>/hiphi_object.urdf (scaled collision pieces, object frame).

Each object -> data/sdf/<tag>.npz with
    sdf (N,N,N) float16  signed distance [m], negative inside (inside any CoACD piece); indexed [ix, iy, iz]
    origin (3,) float32  world coordinates (object frame) of voxel (0,0,0)
    spacing (3,) float32 voxel size per axis
The grid covers the AABB of the scaled collision mesh plus `--margin` on every side; distances outside the grid are
clamped by the sampler (hiphi_wbt.geometry). Sign: open3d RaycastingScene per convex piece, min over pieces.

    .../hswbt/bin/python omomo_bench/wbt/build_object_sdf.py [--res 96] [--margin 0.15]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

HERE = Path(__file__).resolve().parent
OBJ_DIR = HERE / "assets" / "objects"
OUT_DIR = HERE / "data" / "sdf"


def pieces_from_urdf(urdf: Path) -> list[trimesh.Trimesh]:
    text = urdf.read_text()
    out = []
    for m in re.finditer(r'<collision[^>]*>.*?<mesh filename="([^"]+)"(?: scale="([^"]+)")?.*?</collision>', text, re.S):
        mesh = trimesh.load(m.group(1), force="mesh")
        if m.group(2):
            mesh.apply_scale([float(v) for v in m.group(2).split()])
        out.append(mesh)
    assert out, urdf
    return out


def signed_distance_min(pieces: list[trimesh.Trimesh], pts: np.ndarray) -> np.ndarray:
    best = None
    q = o3d.core.Tensor(pts.astype(np.float32))
    for p in pieces:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(p.vertices, np.float32)),
                                                        o3d.core.Tensor(np.asarray(p.faces, np.uint32))))
        d = scene.compute_signed_distance(q).numpy()
        best = d if best is None else np.minimum(best, d)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for urdf in sorted(OBJ_DIR.glob("*/hiphi_object.urdf")):
        tag = urdf.parent.name
        if a.only and tag not in a.only:
            continue
        pieces = pieces_from_urdf(urdf)
        allv = np.concatenate([p.vertices for p in pieces])
        lo, hi = allv.min(0) - a.margin, allv.max(0) + a.margin
        spacing = (hi - lo) / (a.res - 1)
        axes = [lo[i] + spacing[i] * np.arange(a.res) for i in range(3)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        sdf = signed_distance_min(pieces, g).reshape(a.res, a.res, a.res)
        np.savez_compressed(OUT_DIR / f"{tag}.npz", sdf=sdf.astype(np.float16), origin=lo.astype(np.float32), spacing=spacing.astype(np.float32))
        print(f"{tag}: {len(pieces)} pieces, extent {np.round(hi - lo, 3)}, spacing {np.round(spacing * 1000, 1)} mm, sdf [{sdf.min():.3f}, {sdf.max():.3f}]", flush=True)


if __name__ == "__main__":
    main()
