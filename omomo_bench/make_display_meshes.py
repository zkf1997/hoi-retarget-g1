#!/usr/bin/env python3
"""Light display meshes for the viewers: data*/objects/<obj>/<obj>_display.obj with <= MAX_FACES faces (vertex clustering on a
grid that grows until the face budget is met). Only viser uses them; collision, contacts and metrics use the full meshes.
    BENCH_DATASET=humoto python make_display_meshes.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
DS = os.environ.get("BENCH_DATASET", "omomo")
DATA = HERE / ("data" if DS == "omomo" else f"data_{DS}")
MAX_FACES = 5000


def cluster(m: trimesh.Trimesh, cell: float) -> trimesh.Trimesh:
    key = np.floor(m.vertices / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    v = np.zeros((inv.max() + 1, 3)); np.add.at(v, inv, m.vertices); v /= np.bincount(inv)[:, None]
    f = inv[m.faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1]
    f = inv[m.faces][(lambda ff: (ff[:, 0] != ff[:, 1]) & (ff[:, 1] != ff[:, 2]) & (ff[:, 0] != ff[:, 2]))(inv[m.faces])]
    out = trimesh.Trimesh(v, f, process=False)
    out.remove_unreferenced_vertices()
    return out


def main() -> None:
    for od in sorted((DATA / "objects").iterdir()):
        src = od / f"{od.name}.obj"
        if not src.exists():
            continue
        m = trimesh.load(src, force="mesh")
        dst = od / f"{od.name}_display.obj"
        if len(m.faces) <= MAX_FACES:
            m.export(dst); continue
        cell = max(m.extents) / 400
        d = cluster(m, cell)
        while len(d.faces) > MAX_FACES:
            cell *= 1.3; d = cluster(m, cell)
        d.export(dst)
        print(f"{od.name}: {len(m.faces)} -> {len(d.faces)} faces (cell {1000 * cell:.1f} mm)")


if __name__ == "__main__":
    main()
