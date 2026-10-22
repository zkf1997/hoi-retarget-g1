#!/usr/bin/env python3
"""Headless orthographic preview of a URDF (no OpenGL needed): MuJoCo for FK, matplotlib scatter of mesh vertices.

Usage: MUJOCO_GL=disable python preview_urdf.py robot.urdf out.png [--title "..."]
Top row: whole robot (front / side / top). Bottom row: zoom on the hands (red = hand bodies).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

BODY_KEYS = ("pelvis", "torso", "hip", "knee", "ankle", "shoulder", "elbow", "_wrist_", "waist", "head", "logo", "imu", "world")  # "_wrist_" matches G1 wrist links but not Wuji "r_wrist"


def scene_points(model, data, max_per_geom=2500, seed=0):
    rng = np.random.default_rng(seed)
    pts, is_hand = [], []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        v = model.mesh_vert[v0:v0 + nv]
        if len(v) > max_per_geom:
            v = v[rng.choice(len(v), max_per_geom, replace=False)]
        R = data.geom_xmat[g].reshape(3, 3)
        w = v @ R.T + data.geom_xpos[g]
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        hand = not any(k in body for k in BODY_KEYS)
        pts.append(w)
        is_hand.append(np.full(len(w), hand))
    return np.vstack(pts), np.concatenate(is_hand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.urdf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    P, H = scene_points(model, data)
    cols = np.where(H, "tab:red", "0.55")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(args.title or args.urdf.name)
    planes = [(1, 2, "front (y-z)"), (0, 2, "side (x-z)"), (0, 1, "top (x-y)")]
    for ax, (i, j, lab) in zip(axes[0], planes):
        ax.scatter(P[:, i], P[:, j], s=0.2, c=cols, linewidths=0)
        ax.set_aspect("equal"); ax.set_title(lab); ax.grid(True, lw=0.3)
    hand = P[H]
    lo, hi = hand.min(0) - 0.04, hand.max(0) + 0.04
    m = np.all((P > lo) & (P < hi), axis=1)
    for ax, (i, j, lab) in zip(axes[1], planes):
        ax.scatter(P[m, i], P[m, j], s=0.6, c=cols[m], linewidths=0)
        ax.set_aspect("equal"); ax.set_title("hands " + lab); ax.grid(True, lw=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
