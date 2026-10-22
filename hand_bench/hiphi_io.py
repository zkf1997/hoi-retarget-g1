"""HiPHI reader (HOI subset): 55-joint BVH (Y-up, cm, 90 Hz, ZXY Euler) + object_tracks/*.csv (m, XYZW) + object_meshes/*.obj (cm).

Produces the same dictionary layout as humoto_io.load_sequence (Z-up metres, 30 fps by default via frame striding):
body_pos (T,25,3) in humoto_io.MIXAMO_BODY order, hand_kp[side] (T,21,3) MediaPipe order (wrist, 5 x (J1,J2,J3,End Site)),
objects / obj_pos / obj_rot / obj_scale. Only the original (non-mirrored) motions are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from humoto_io import MIXAMO_BODY, YUP2ZUP

HIPHI_ROOT = Path("/home/kaizhao/projects/HiPHI")
DATA_ROOT = HIPHI_ROOT / "data"
MESH_ROOT = HIPHI_ROOT / "object_meshes"
META_CSV = HIPHI_ROOT / "metadata" / "hiphi_metadata.csv"
SRC_FPS = 90.0
STRIDE = 3  # 90 Hz -> 30 Hz
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# HiPHI BVH names that stand in for Mixamo ones (end sites are named "<parent>_End" by the parser)
BODY_ALIAS = {"HeadTop_End": "Head_End", "LeftToe_End": "LeftToeBase_End", "RightToe_End": "RightToeBase_End"}


def motion_dir(motion_id: str) -> Path:
    frame_lu = motion_id.replace("__mirror", "").rsplit("_", 1)[0]  # frames/LUs may themselves contain underscores
    frame, lu = frame_lu.split("-", 1)
    return DATA_ROOT / frame / lu / motion_id


def list_motions(hoi_only: bool = True, originals_only: bool = True) -> pd.DataFrame:
    md = pd.read_csv(META_CSV)
    if hoi_only:
        md = md[md.is_hoi == True]  # noqa: E712
    if originals_only:
        md = md[md.mirrored == False]  # noqa: E712
    return md.reset_index(drop=True)


def load_meta(motion_id: str) -> dict:
    return json.loads((motion_dir(motion_id) / "metadata.json").read_text())


# --- BVH -------------------------------------------------------------------------------------------------------------
def parse_bvh(path: Path):
    """returns names, parents, offsets (cm), channel specs, motion (F, nchan), frame_time"""
    txt = path.read_text().split("\n")
    names, parents, offsets, channels = [], [], [], []
    stack, i = [], 0
    while not txt[i].strip().startswith("MOTION"):
        tok = txt[i].split()
        i += 1
        if not tok:
            continue
        if tok[0] in ("ROOT", "JOINT"):
            names.append(tok[1]); parents.append(stack[-1] if stack else -1); offsets.append(None); channels.append([])
            stack.append(len(names) - 1)
        elif tok[0] == "End":
            names.append(f"{names[stack[-1]]}_End"); parents.append(stack[-1]); offsets.append(None); channels.append([])
            stack.append(len(names) - 1)
        elif tok[0] == "OFFSET":
            offsets[stack[-1]] = np.array([float(x) for x in tok[1:4]])
        elif tok[0] == "CHANNELS":
            channels[stack[-1]] = tok[2:2 + int(tok[1])]
        elif tok[0] == "}":
            stack.pop()
    n_frames = int(txt[i + 1].split()[1])
    frame_time = float(txt[i + 2].split()[2])
    motion = np.loadtxt(txt[i + 3:i + 3 + n_frames], dtype=np.float64)
    return names, np.array(parents), np.stack(offsets), channels, motion, frame_time


def bvh_fk(names, parents, offsets, channels, motion) -> np.ndarray:
    """world positions (F, J, 3) in the BVH frame (Y-up, cm). Rotation channels are applied in the declared order."""
    F, J = motion.shape[0], len(names)
    pos = np.zeros((F, J, 3))
    rot = np.zeros((F, J, 3, 3))
    c = 0
    for j in range(J):
        ch = channels[j]
        t = np.repeat(offsets[j][None], F, 0)
        Rl = np.repeat(np.eye(3)[None], F, 0)
        if ch:
            vals = motion[:, c:c + len(ch)]
            c += len(ch)
            rot_axes, rot_vals = "", []
            for k, name in enumerate(ch):
                if name.endswith("position"):  # HiPHI writes the full local translation into the position channels
                    t[:, "XYZ".index(name[0])] = vals[:, k]
                else:
                    rot_axes += name[0]; rot_vals.append(vals[:, k])
            if rot_axes:
                Rl = R.from_euler(rot_axes, np.stack(rot_vals, 1), degrees=True).as_matrix()
        p = parents[j]
        if p < 0:
            pos[:, j], rot[:, j] = t, Rl
        else:
            pos[:, j] = pos[:, p] + np.einsum("fij,fj->fi", rot[:, p], t)
            rot[:, j] = rot[:, p] @ Rl
    return pos, rot


def load_object_mesh(mesh_id: str):
    """object-local mesh in metres (the trajectory quaternion/translation apply directly; no axis conversion of the mesh)."""
    import trimesh

    m = trimesh.load(MESH_ROOT / f"{mesh_id}.obj", force="mesh", process=False)
    m.vertices = m.vertices * 0.01
    m.merge_vertices()
    return m


def load_sequence(motion_id: str, stride: int = STRIDE) -> dict:
    d = motion_dir(motion_id)
    meta = load_meta(motion_id)
    names, parents, offsets, channels, motion, ft = parse_bvh(d / "motion_actor.bvh")
    motion = motion[::stride]
    pos_cm, rot = bvh_fk(names, parents, offsets, channels, motion)
    pos = (pos_cm * 0.01) @ YUP2ZUP.T
    idx = {n: i for i, n in enumerate(names)}
    T = len(motion)
    out = {"seq": motion_id, "n_frames": T, "fps": 1.0 / (ft * stride), "meta": meta}
    out["body_pos"] = np.stack([pos[:, idx[BODY_ALIAS.get(b, b)]] for b in MIXAMO_BODY], axis=1)
    out["hand_kp"], out["hand_rot"] = {}, {}
    for side in ("Left", "Right"):
        kp_names = [f"{side}Hand"] + [f"{side}Hand{f}{i}" if i < 4 else f"{side}Hand{f}3_End" for f in FINGERS for i in (1, 2, 3, 4)]
        out["hand_kp"][side.lower()] = np.stack([pos[:, idx[n]] for n in kp_names], axis=1)
        out["hand_rot"][side.lower()] = YUP2ZUP @ rot[:, idx[f"{side}Hand"]]
    objs, op, orot = [], [], []
    for o in meta["objects"]:
        tr = pd.read_csv(d / o["trajectory_path"]).iloc[::stride]
        if len(tr) != T:
            raise ValueError(f"{motion_id}: track {len(tr)} rows vs {T} frames")
        objs.append(o["mesh_id"])
        op.append(tr[["px", "py", "pz"]].to_numpy() @ YUP2ZUP.T)
        orot.append(YUP2ZUP @ R.from_quat(tr[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix())
    out["objects"], out["objects_missing"] = objs, []
    out["obj_pos"] = np.stack(op, 1) if objs else np.zeros((T, 0, 3))
    out["obj_rot"] = np.stack(orot, 1) if objs else np.zeros((T, 0, 3, 3))
    out["obj_scale"] = np.ones((T, len(objs)))
    return out


if __name__ == "__main__":
    import sys

    mid = sys.argv[1] if len(sys.argv) > 1 else "Adjusting-adjust_0001"
    d = load_sequence(mid)
    kp = d["hand_kp"]["right"]
    L = lambda a, b: np.linalg.norm(kp[:, a] - kp[:, b], axis=-1).mean() * 100
    print(mid, d["n_frames"], "frames @", round(d["fps"], 2), "fps; objects", d["objects"])
    print("feet z min %.3f | hips z %.2f | head %.2f" % (d["body_pos"][:, [17, 22], 2].min(), d["body_pos"][0, 0, 2], d["body_pos"][0, 5, 2]))
    print("right hand cm: wrist-midMCP %.1f MCP-PIP %.1f PIP-DIP %.1f DIP-tip %.1f thumb CMC-MCP %.1f idx-pinky MCP %.1f" % (L(0, 9), L(9, 10), L(10, 11), L(11, 12), L(1, 2), L(5, 17)))
    m = load_object_mesh(d["objects"][0])
    dmin = []
    for t in range(0, d["n_frames"], max(1, d["n_frames"] // 20)):
        V = m.vertices @ d["obj_rot"][t, 0].T + d["obj_pos"][t, 0]
        dmin.append(min(np.linalg.norm(d["hand_kp"][s][t][:, None] - V[None], axis=-1).min() for s in ("left", "right")))
    print("min hand->object dist over sampled frames (cm):", np.round(np.array(dmin) * 100, 1))
