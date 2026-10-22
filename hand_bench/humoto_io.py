"""HUMOTO reader: GLB (Mixamo rig + object nodes, Y-up, 30 fps) -> world-space joints / 21-keypoint hands / object poses (Z-up, metres).

HUMOTO (humoto_0805/<seq>/<seq>.glb + .yaml, humoto_objects_0805/<obj>/<obj>.obj):
  * one Mixamo skeleton (65 joints, `mixamorig:*`) under an `Armature` node scaled 0.01 (bone translations are in cm);
    hands have Thumb1..4 / Index1..4 / ... which is exactly the MediaPipe 21-keypoint layout (wrist + 5 x 4).
  * every object is a root node with an animated TRS (metres); the .obj file is the mesh in that node's local frame.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

HUMOTO_ROOT = Path("/home/kaizhao/projects/humoto")
SEQ_ROOT = HUMOTO_ROOT / "humoto_0805"
OBJ_ROOT = HUMOTO_ROOT / "humoto_objects_0805"
FPS = 30.0

# glTF is Y-up; we work Z-up: x -> x, y -> z, z -> -y
YUP2ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# MediaPipe 21: 0 wrist, then per finger 4 points (CMC/MCP, MCP/PIP, IP/DIP, TIP)
HAND_KP_NODES = {
    side: [f"mixamorig:{side}Hand"] + [f"mixamorig:{side}Hand{f}{i}" for f in FINGERS for i in (1, 2, 3, 4)]
    for side in ("Left", "Right")
}
MIXAMO_BODY = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "HeadTop_End",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToe_End",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToe_End",
]
MIXAMO_BODY_PARENT = {
    "Spine": "Hips", "Spine1": "Spine", "Spine2": "Spine1", "Neck": "Spine2", "Head": "Neck", "HeadTop_End": "Head",
    "LeftShoulder": "Spine2", "LeftArm": "LeftShoulder", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm",
    "RightShoulder": "Spine2", "RightArm": "RightShoulder", "RightForeArm": "RightArm", "RightHand": "RightForeArm",
    "LeftUpLeg": "Hips", "LeftLeg": "LeftUpLeg", "LeftFoot": "LeftLeg", "LeftToeBase": "LeftFoot", "LeftToe_End": "LeftToeBase",
    "RightUpLeg": "Hips", "RightLeg": "RightUpLeg", "RightFoot": "RightLeg", "RightToeBase": "RightFoot", "RightToe_End": "RightToeBase",
}
HAND_KP_PARENT = [0] + [0 if i % 4 == 0 else i for i in range(0, 20)]  # index of parent for kps 1..20
HAND_KP_PARENT = [-1] + [0 if (k - 1) % 4 == 0 else k - 1 for k in range(1, 21)]


def list_sequences() -> list[str]:
    return sorted(p.name for p in SEQ_ROOT.iterdir() if p.is_dir())


def load_meta(seq: str) -> dict:
    return yaml.safe_load((SEQ_ROOT / seq / f"{seq}.yaml").read_text())


def object_meta(obj: str) -> dict:
    p = OBJ_ROOT / obj / f"{obj}.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def object_mesh_path(obj: str) -> Path:
    return OBJ_ROOT / obj / f"{obj}.obj"


def load_object_mesh(seq: str, obj: str):
    """Object mesh in its node-local frame, taken from the sequence GLB (the .obj files use a different axis convention
    than the GLB nodes, the embedded mesh is unambiguous). Apply (obj_rot * obj_scale, obj_pos) from `load_sequence`."""
    import trimesh

    g = _GLB(SEQ_ROOT / seq / f"{seq}.glb")
    node = g.j["nodes"][g.names.index(obj)]
    prims = g.j["meshes"][node["mesh"]]["primitives"]
    parts = []
    for pr in prims:
        v = g.acc(pr["attributes"]["POSITION"])
        f = g.acc(pr["indices"]).astype(np.int64).reshape(-1, 3)
        parts.append(trimesh.Trimesh(v, f, process=False))
    m = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    m.merge_vertices()
    return m


class _GLB:
    def __init__(self, path: Path):
        b = path.read_bytes()
        n = struct.unpack("<I", b[12:16])[0]
        self.j = json.loads(b[20:20 + n])
        off = 20 + n
        m = struct.unpack("<I", b[off:off + 4])[0]
        self.bin = b[off + 8:off + 8 + m]
        self.names = [nd.get("name", f"node{i}") for i, nd in enumerate(self.j["nodes"])]

    def acc(self, i: int) -> np.ndarray:
        a = self.j["accessors"][i]
        bv = self.j["bufferViews"][a["bufferView"]]
        ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[a["componentType"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(self.bin, dtype=dt, count=a["count"] * ncomp, offset=start).reshape(a["count"], ncomp).astype(np.float64)


def _trs(t, q, s):
    M = np.tile(np.eye(4), (len(t), 1, 1))
    M[:, :3, :3] = R.from_quat(q).as_matrix() * s[:, None, :]
    M[:, :3, 3] = t
    return M


def load_sequence(seq: str) -> dict:
    """Returns dict with
        n_frames, fps,
        body_pos   (T, 25, 3)   Mixamo body joints (MIXAMO_BODY order), Z-up metres
        hand_kp    {"left"/"right": (T, 21, 3)}   MediaPipe-ordered keypoints, world Z-up metres
        hand_rot   {"left"/"right": (T, 3, 3)}    world rotation of the Mixamo hand node (Y along the palm)
        objects    [names]; obj_pos (T, K, 3); obj_rot (T, K, 3, 3); obj_scale (T, K)  (node TRS, Z-up)
    """
    g = _GLB(SEQ_ROOT / seq / f"{seq}.glb")
    j, names = g.j, g.names
    N = len(names)
    parent = -np.ones(N, dtype=int)
    for i, nd in enumerate(j["nodes"]):
        for c in nd.get("children", []):
            parent[c] = i

    # per-node static TRS
    T0 = np.array([nd.get("translation", [0, 0, 0]) for nd in j["nodes"]], dtype=np.float64)
    Q0 = np.array([nd.get("rotation", [0, 0, 0, 1]) for nd in j["nodes"]], dtype=np.float64)
    S0 = np.array([nd.get("scale", [1, 1, 1]) for nd in j["nodes"]], dtype=np.float64)

    # animation channels: HUMOTO samples on a 30 fps grid; constant channels may only have 2 keys -> resample
    chans = []
    for a in j["animations"]:
        for ch in a["channels"]:
            smp = a["samplers"][ch["sampler"]]
            chans.append((ch["target"]["node"], ch["target"]["path"], g.acc(smp["input"])[:, 0], g.acc(smp["output"])))
    times = max((c[2] for c in chans), key=len)
    n_frames = len(times)
    anim = {}
    for n, path, tin, out in chans:
        if len(tin) != n_frames:
            out = np.stack([np.interp(times, tin, out[:, k]) for k in range(out.shape[1])], axis=1)
            if path == "rotation":
                out /= np.linalg.norm(out, axis=1, keepdims=True)
        anim[(n, path)] = out
    T = n_frames
    Tt = np.repeat(T0[None], T, 0)
    Qt = np.repeat(Q0[None], T, 0)
    St = np.repeat(S0[None], T, 0)
    for (n, path), v in anim.items():
        {"translation": Tt, "rotation": Qt, "scale": St}[path][:, n] = v

    # FK (nodes are not guaranteed topologically sorted -> iterate depth order)
    local = np.stack([_trs(Tt[:, n], Qt[:, n], St[:, n]) for n in range(N)], axis=1)  # (T, N, 4, 4)
    world = np.empty_like(local)
    depth = np.zeros(N, dtype=int)
    for n in range(N):
        d, p = 0, parent[n]
        while p >= 0:
            d, p = d + 1, parent[p]
        depth[n] = d
    for n in np.argsort(depth, kind="stable"):
        world[:, n] = local[:, n] if parent[n] < 0 else world[:, parent[n]] @ local[:, n]

    C = np.eye(4)
    C[:3, :3] = YUP2ZUP
    world = C[None, None] @ world

    idx = {nm: i for i, nm in enumerate(names)}
    out = {"seq": seq, "n_frames": T, "fps": FPS, "times": times}
    out["body_pos"] = np.stack([world[:, idx[f"mixamorig:{b}"], :3, 3] for b in MIXAMO_BODY], axis=1)
    out["hand_kp"], out["hand_rot"] = {}, {}
    for side in ("Left", "Right"):
        ids = [idx[nm] for nm in HAND_KP_NODES[side]]
        out["hand_kp"][side.lower()] = np.stack([world[:, i, :3, 3] for i in ids], axis=1)
        Rw = world[:, ids[0], :3, :3]
        out["hand_rot"][side.lower()] = Rw / np.linalg.norm(Rw, axis=1, keepdims=True)  # strip the 0.01 armature scale

    meta = load_meta(seq)
    objs = [o for o in meta["objects"] if o in idx]
    missing = [o for o in meta["objects"] if o not in idx]
    out["objects"], out["objects_missing"] = objs, missing
    if objs:
        W = np.stack([world[:, idx[o]] for o in objs], axis=1)  # (T, K, 4, 4)
        sc = np.linalg.norm(W[..., :3, :3], axis=-2)  # (T, K, 3) column norms
        out["obj_pos"] = W[..., :3, 3]
        out["obj_rot"] = W[..., :3, :3] / sc[..., None, :]
        out["obj_scale"] = sc.mean(-1)
    else:
        out["obj_pos"] = np.zeros((T, 0, 3)); out["obj_rot"] = np.zeros((T, 0, 3, 3)); out["obj_scale"] = np.zeros((T, 0))
    out["meta"] = meta
    return out


def hand_frame(kp: np.ndarray) -> np.ndarray:
    """Canonical hand frame from 21 keypoints (works for a batch (T,21,3) or a single (21,3)).
    x: wrist -> middle MCP (distal), z: pinky MCP -> index MCP (thumb side, orthogonalised), y = z x x.
    This matches the G1 wrist convention used by both G1 hand URDFs: +x distal, +z thumb side, palm facing +y (right) / -y (left).
    Returns rotation matrices whose columns are (x, y, z) expressed in the input frame."""
    single = kp.ndim == 2
    kp = kp[None] if single else kp
    x = kp[:, 9] - kp[:, 0]
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    z = kp[:, 5] - kp[:, 17]
    z -= (z * x).sum(-1, keepdims=True) * x
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    Rm = np.stack([x, y, z], axis=-1)
    return Rm[0] if single else Rm


def to_hand_frame(kp: np.ndarray, Rm: np.ndarray | None = None) -> np.ndarray:
    """(T,21,3) world keypoints -> wrist-centred canonical hand frame."""
    Rm = hand_frame(kp) if Rm is None else Rm
    return np.einsum("tji,tkj->tki", Rm, kp - kp[:, :1])
