#!/usr/bin/env python3
"""HUMOTO GLB (Mixamo rig) -> HiPHI-style BVH, so hiphi2smplx can fit SMPL-X to it.

    python humoto_to_bvh.py --out data/bvh [--seqs ...]            # default: the sequences in data/selection.json

BVH conventions (match the HiPHI BVH files hiphi2smplx was written for):
  * glTF world frame, Y-up, centimetres (hiphi2smplx loads with unit_scale=0.01)
  * joint names = Mixamo names without the `mixamorig:` prefix (Hips, Spine, Spine1, Spine2, Neck, Head, LeftShoulder,
    LeftArm, LeftForeArm, LeftHand, LeftHandThumb1..3, ..., LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase); finger *4,
    HeadTop_End and *Toe_End become End Sites
  * every joint has 6 channels (Xposition Yposition Zposition Zrotation Xrotation Yrotation); positions are the exact
    per-frame parent-relative offsets, so the BVH forward kinematics reproduces the GLB joint positions exactly
  * rotations are expressed RELATIVE TO THE BIND POSE: R_rel(t) = R_world(t) R_bind^T, so a joint in its rest
    orientation has identity rotation (like HiPHI's BVH and SMPL-X, whose rest poses are T-poses). Mixamo's own
    local frames (bone along +y) would otherwise bias hiphi2smplx's orientation / posture reference terms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

import humoto_io as H

HERE = Path(__file__).resolve().parent
PREFIX = "mixamorig:"
END_SITES = {"HeadTop_End", "LeftToe_End", "RightToe_End"} | {f"{s}Hand{f}4" for s in ("Left", "Right") for f in H.FINGERS}


def glb_world(g: H._GLB) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Per-frame world matrices (T, N, 4, 4) in the glTF (Y-up, metre) frame, parents, bind world matrices from the skin."""
    j, names = g.j, g.names
    N = len(names)
    parent = -np.ones(N, dtype=int)
    for i, nd in enumerate(j["nodes"]):
        for c in nd.get("children", []):
            parent[c] = i
    T0 = np.array([nd.get("translation", [0, 0, 0]) for nd in j["nodes"]], dtype=np.float64)
    Q0 = np.array([nd.get("rotation", [0, 0, 0, 1]) for nd in j["nodes"]], dtype=np.float64)
    S0 = np.array([nd.get("scale", [1, 1, 1]) for nd in j["nodes"]], dtype=np.float64)
    chans = []
    for a in j["animations"]:
        for ch in a["channels"]:
            smp = a["samplers"][ch["sampler"]]
            chans.append((ch["target"]["node"], ch["target"]["path"], g.acc(smp["input"])[:, 0], g.acc(smp["output"])))
    times = max((c[2] for c in chans), key=len)
    T = len(times)
    Tt, Qt, St = np.repeat(T0[None], T, 0), np.repeat(Q0[None], T, 0), np.repeat(S0[None], T, 0)
    for n, path, tin, out in chans:
        if len(tin) != T:
            out = np.stack([np.interp(times, tin, out[:, k]) for k in range(out.shape[1])], axis=1)
            if path == "rotation":
                out /= np.linalg.norm(out, axis=1, keepdims=True)
        {"translation": Tt, "rotation": Qt, "scale": St}[path][:, n] = out
    local = np.stack([H._trs(Tt[:, n], Qt[:, n], St[:, n]) for n in range(N)], axis=1)
    world = np.empty_like(local)
    depth = np.zeros(N, dtype=int)
    for n in range(N):
        d, p = 0, parent[n]
        while p >= 0:
            d, p = d + 1, parent[p]
        depth[n] = d
    for n in np.argsort(depth, kind="stable"):
        world[:, n] = local[:, n] if parent[n] < 0 else world[:, parent[n]] @ local[:, n]
    skin = j["skins"][0]
    ibm = g.acc(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    bind = np.zeros((N, 4, 4)); bind[:] = np.eye(4)
    for k, node in enumerate(skin["joints"]):
        bind[node] = np.linalg.inv(ibm[k])
    return world, parent, bind, names


def rot_only(M: np.ndarray) -> np.ndarray:
    R = M[..., :3, :3]
    return R / np.linalg.norm(R, axis=-2, keepdims=True)


def write_bvh(seq: str, out: Path, fps: float = H.FPS) -> dict:
    g = H._GLB(H.SEQ_ROOT / seq / f"{seq}.glb")
    world, parent, bind, names = glb_world(g)
    idx = {n: i for i, n in enumerate(names)}
    T = world.shape[0]
    hips = idx[PREFIX + "Hips"]
    # joints in hierarchy order (parents first), restricted to the Mixamo subtree under Hips
    order = []
    def visit(n):
        order.append(n)
        for c, p in enumerate(parent):
            if p == n and names[c].startswith(PREFIX):
                visit(c)
    visit(hips)
    short = {n: names[n][len(PREFIX):] for n in order}
    joints = [n for n in order if short[n] not in END_SITES]
    Rrel = np.einsum("tnij,nkj->tnik", rot_only(world), rot_only(bind))  # R_world R_bind^T  (T, N, 3, 3)
    pos = world[..., :3, 3] * 100.0  # cm
    pbind = bind[:, :3, 3] * 100.0

    lines = ["HIERARCHY"]
    channels = "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"
    def emit(n, depth):
        ind = "    " * depth
        kids = [c for c in order if parent[c] == n]
        lines.append(f"{ind}{'ROOT' if depth == 0 else 'JOINT'} {short[n]}")
        lines.append(ind + "{")
        off = np.zeros(3) if depth == 0 else pbind[n] - pbind[parent[n]]
        lines.append(f"{ind}    OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        lines.append(f"{ind}    {channels}")
        ends = [c for c in kids if short[c] in END_SITES]
        for c in kids:
            if short[c] not in END_SITES:
                emit(c, depth + 1)
        if ends:  # one End Site per joint (BVH allows a single one); fingers/toes/head have exactly one
            e = ends[0]
            off = pbind[e] - pbind[n]
            lines.append(f"{ind}    End Site")
            lines.append(f"{ind}    {{")
            lines.append(f"{ind}        OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
            lines.append(f"{ind}    }}")
        lines.append(ind + "}")
    emit(hips, 0)

    rows = np.zeros((T, 6 * len(joints)))
    for k, n in enumerate(joints):
        if n == hips:
            t = pos[:, n]
            R = Rrel[:, n]
        else:
            p = parent[n]
            Rp_T = Rrel[:, p].transpose(0, 2, 1)
            t = np.einsum("tij,tj->ti", Rp_T, pos[:, n] - pos[:, p])
            R = Rp_T @ Rrel[:, n]
        eul = Rot.from_matrix(R).as_euler("ZXY", degrees=True)
        rows[:, 6 * k:6 * k + 3] = t
        rows[:, 6 * k + 3:6 * k + 6] = eul
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]
    lines += [" ".join(f"{v:.6f}" for v in row) for row in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    # reference world positions (metres, glTF Y-up) for validation
    return {"n_frames": T, "joints": [short[n] for n in joints], "pos_m": pos[:, joints] / 100.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "data" / "bvh")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--selection", type=Path, default=HERE / "data" / "selection.json")
    args = ap.parse_args()
    seqs = args.seqs or json.loads(args.selection.read_text())["selected"]
    for k, s in enumerate(seqs):
        p = args.out / f"{s}.bvh"
        if p.exists():
            continue
        info = write_bvh(s, p)
        np.save(args.out / f"{s}.ref_pos.npy", info["pos_m"].astype(np.float32))
        print(f"[{k + 1}/{len(seqs)}] {s}: {info['n_frames']} frames, {len(info['joints'])} joints")


if __name__ == "__main__":
    main()
