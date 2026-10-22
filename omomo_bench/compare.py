#!/usr/bin/env python3
"""Compare OmniRetarget vs UMR results on OMOMO: per-sequence metrics + summary tables.

Run in the `hsretargeting` conda env (mujoco, trimesh, numpy):
    python compare.py                    # default lanes: omni_{brainco,wuji} (UMR-style scaling) vs umr_{brainco,wuji}
    python compare.py --seqs sub16_largebox_000
    python compare.py --lane omni=results/omni:_original --lane umr=results/umr --out _nohand   # legacy body-only lanes

Both methods output qpos[T, 36+] = root pos, root quat wxyz, 29 G1 joints (MJCF order) at 30 fps, in their own
"scaled world" (human uniformly scaled by 1.32 / T-pose height). Metrics (all in metres / radians, per sequence):
  obj_pen_max      mean over frames of the deepest robot-object penetration (0 = none)
  obj_pen_frac     fraction of frames with robot-object penetration > 5 mm
  ground_pen       mean over frames of the deepest robot-ground penetration
  foot_slide       mean per-frame xy travel of the ankle links while the human foot is in stance (toe xy speed < 1 cm/frame)
  hand_contact     fraction of frames where a human wrist is within 10 cm of the object and the robot hand link is too
  wrist_err/ankle_err   mean distance between robot hand/ankle links and the scaled human wrist/ankle joints
  joint_jerk       mean |d^3 q / dt^3| over the 29 body joints (smoothness)
  limit_viol       fraction of (frame, joint) beyond the MJCF joint range
  runtime_s        wall-clock seconds of the retargeting run (from run_summary.json), and per frame
Writes results/metrics.csv, results/summary.md  (BENCH_DATASET=humoto -> data_humoto/, results_humoto/)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
# Dataset switch: BENCH_DATASET=omomo (default) | humoto  ->  data[_humoto]/, results[_humoto]/, omni_workdir[_humoto]/
DATASET = os.environ.get("BENCH_DATASET", "omomo")
SUFFIX = "" if DATASET == "omomo" else f"_{DATASET}"
DATA = HERE / f"data{SUFFIX}"
RESULTS = HERE / f"results{SUFFIX}"
WORK = HERE / f"omni_workdir{SUFFIX}"
FPS = 30.0
# result lanes: name -> (dir, filename suffix). Extend with --lane name=dir[:suffix]  (e.g. umr_brainco=results/umr_brainco)
LANES: dict[str, tuple[Path, str]] = {
    "omni_brainco": (RESULTS / "omni_brainco_umrscale", "_original"), "umr_brainco": (RESULTS / "umr_brainco", ""),
    "omni_wuji": (RESULTS / "omni_wuji_umrscale", "_original"), "umr_wuji": (RESULTS / "umr_wuji", ""),
}  # default: both methods x both hands, UMR-style scaling everywhere (override with --lane)
G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def body_qpos36(r) -> np.ndarray:
    """root(7) + the 29 body joints in MJCF g1_29dof order, remapped by name when the result carries robot_joint_names
    (UMR with a hand-equipped robot stores joints in kinematic-tree order and appends the finger DoFs)."""
    q = r["qpos"]
    if "robot_joint_names" not in r:
        return q[:, :36]
    names = [str(x) for x in r["robot_joint_names"]]
    idx = [7 + names.index(j) for j in G1_JOINTS]
    return np.concatenate([q[:, :7], q[:, idx]], axis=1)

# SMPL-X joint ids
SX = {"L_Wrist": 20, "R_Wrist": 21, "L_Ankle": 7, "R_Ankle": 8, "L_Toe": 10, "R_Toe": 11, "Pelvis": 0}
# InterMimic 52-joint ids (holosoma `human_joints`)
IM = {"L_Wrist": 17, "R_Wrist": 36, "L_Ankle": 3, "R_Ankle": 7, "L_Toe": 4, "R_Toe": 8, "Pelvis": 0}
ROBOT_LINKS = {"L_Wrist": "left_wrist_yaw_link", "R_Wrist": "right_wrist_yaw_link",
               "L_Ankle": "left_ankle_roll_link", "R_Ankle": "right_ankle_roll_link", "Pelvis": "pelvis"}
WRIST_OFFSET = np.array([0.0415, 0.0, 0.0])  # hand flange point in wrist_yaw_link frame (= rubber_hand_link origin), same for all models


# ----------------------------------------------------------------------------- models
_model_cache: dict[str, mujoco.MjModel] = {}


def robot_object_model(obj: str, mesh_scale: float) -> mujoco.MjModel:
    """omni_workdir/models/g1/g1_29dof_w_<obj>.xml, with the object meshes scaled (UMR scales objects by smpl_scale)."""
    key = f"{obj}:{mesh_scale:.5f}"
    if key in _model_cache:
        return _model_cache[key]
    xml_path = WORK / "models" / "g1" / f"g1_29dof_w_{obj}.xml"
    text = xml_path.read_text()
    if abs(mesh_scale - 1.0) > 1e-6:
        s = f'scale="{mesh_scale:.6f} {mesh_scale:.6f} {mesh_scale:.6f}"'
        text = re.sub(rf'(<mesh name="{obj}_[^"]*"[^>]*?)scale="1 1 1"', rf"\1{s}", text)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=xml_path.parent, delete=False) as f:
        f.write(text)
        tmp = Path(f.name)
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    _model_cache[key] = m
    return m


def omni_subject(r, default: str) -> str:
    """Subject (= height_dict key, selects the per-subject scaled MJCF) of an OmniRetarget result: the `subject` key written
    by run_omni.py, else the height_dict entry whose 1.32/height matches the stored smpl_scale, else the index subject."""
    if "subject" in r:
        return str(r["subject"])
    hp = DATA / "omni" / "height_dict.pkl"
    if "smpl_scale" in r and hp.exists():
        import pickle
        heights = pickle.load(open(hp, "rb"))
        s = float(np.asarray(r["smpl_scale"]).reshape(-1)[0])
        best = min(heights, key=lambda k: abs(1.32 / heights[k] - s))
        if abs(1.32 / heights[best] - s) < 1e-4:
            return best
    return default


def omni_model(hand: str, scaling: str, subject: str, obj: str) -> mujoco.MjModel:
    """The exact MJCF an OmniRetarget run used (hand geoms baked, object mesh pre-scaled for the umr-scaling variants)."""
    stem = "g1_29dof" + ("" if hand == "none" else f"_{hand}") + (f"_umrscale_{subject}" if scaling == "umr" else "")
    xml_path = WORK / "models" / "g1" / f"{stem}_w_{obj}.xml"
    key = str(xml_path)
    if key not in _model_cache:
        _model_cache[key] = mujoco.MjModel.from_xml_path(key)
    return _model_cache[key]


UMR_ROBOT_XML = {"brainco": HERE / "robots" / "g1_29dof_brainco_revo2.xml", "wuji": HERE / "robots" / "g1_29dof_wuji_hand2.xml",
                 "none": HERE.parent / "UMR" / "assets" / "g1" / "g1_29dof.xml"}


def umr_robot_xml(r) -> Path:
    """Robot MJCF of a UMR result. UMR writes robots/<seq>.floating_mjcf.xml next to the robot xml and the BrainCo / Wuji
    runs overwrite each other's copy, so the `robot_xml` stored in the npz is unreliable; rebuild from the `hand` key."""
    return UMR_ROBOT_XML[str(r["hand"]) if "hand" in r else "none"]


def multi_objects(seq: str, src) -> list[str]:
    """Objects of a sequence for the multi-object hand stage: data*/multi_objects.json (prepare_humoto_multi.py), primary first."""
    p = DATA / "multi_objects.json"
    if p.exists():
        d = json.loads(p.read_text())
        if seq in d:
            return d[seq]
    return [str(src["object"])]


def umr_object_pos(r, src, s: float, T: int | None = None) -> np.ndarray:
    """Robot-world object positions of a UMR-style result: the `obj_pos_robot` key (hand-anchored real-size variant,
    run_umr.py --scaling realhand) if present, else the source trajectory x smpl_scale."""
    p = np.asarray(r["obj_pos_robot"], dtype=np.float64) if "obj_pos_robot" in r else src["obj_pos"].astype(np.float64) * s
    return p if T is None else p[:T]


def augmented_model(robot_xml: Path, obj, mesh_scale: float, for_render: bool = False) -> mujoco.MjModel:
    """<robot MJCF> (+ freejoint root the way UMR adds it, if missing) + ground plane + object body (CoACD pieces, scaled).
    for_render adds a light and a checker floor (the URDF-derived MJCFs have neither)."""
    import xml.etree.ElementTree as ET
    objs = [obj] if isinstance(obj, str) else list(obj)  # several objects: one free body each, qpos in list order after the robot
    key = f"{robot_xml}:{'+'.join(objs)}:{mesh_scale:.5f}:{for_render}"
    if key in _model_cache:
        return _model_cache[key]
    root = ET.parse(robot_xml).getroot()
    if root.find(".//freejoint") is None and not any(j.get("type") == "free" for j in root.iter("joint")):
        wb0 = root.find("worldbody")
        kids = [c for c in list(wb0) if c.tag == "body"]
        assert len(kids) == 1, f"expected one root body in {robot_xml}"
        wb0.remove(kids[0])
        fr = ET.SubElement(wb0, "body", name="float_root", pos="0 0 0")
        ET.SubElement(fr, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-6 1e-6 1e-6")
        ET.SubElement(fr, "freejoint", name="float_root_freejoint")
        fr.append(kids[0])
    asset = root.find("asset") or ET.SubElement(root, "asset")
    wb = root.find("worldbody")
    sc = f"{mesh_scale:.6f} {mesh_scale:.6f} {mesh_scale:.6f}"
    for obj in objs:
        meta = json.loads((DATA / "objects" / obj / "meta.json").read_text())
        body = ET.SubElement(wb, "body", name=f"{obj}_link")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="0.01 0.01 0.01")
        for i in range(meta["n_pieces"]):
            ET.SubElement(asset, "mesh", name=f"{obj}_c{i:02d}", file=str((DATA / "objects" / obj / f"{obj}_coacd_{i:02d}.obj").resolve()), scale=sc)
            ET.SubElement(body, "geom", name=f"{obj}_c{i:02d}", type="mesh", mesh=f"{obj}_c{i:02d}", contype="1", conaffinity="1")
    if mujoco.mj_name2id(mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), mujoco.mjtObj.mjOBJ_GEOM, "ground") < 0:
        ET.SubElement(wb, "geom", name="ground", type="plane", size="20 20 0.1", pos="0 0 0", contype="1", conaffinity="1")
    if for_render:
        ET.SubElement(asset, "texture", type="2d", name="grid", builtin="checker", rgb1="0.25 0.3 0.4", rgb2="0.35 0.45 0.6", width="300", height="300")
        ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="8 8", texuniform="true", reflectance="0.1")
        for g in wb.iter("geom"):
            if g.get("name") == "ground":
                g.set("material", "grid")
        ET.SubElement(wb, "light", pos="0 0 4", dir="0 0 -1", directional="true")
        vis = root.find("visual") or ET.SubElement(root, "visual")
        ET.SubElement(vis, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    _model_cache[key] = m
    return m


def _pq_mesh(pq):
    return getattr(pq, "mesh", None) if getattr(pq, "mesh", None) is not None else pq._mesh  # trimesh 4 vs 5


def on_surface_chunked(pq, pts: np.ndarray, chunk: int = 200, far: float = 0.15):
    """Closest surface point + distance, (closest [N,3], distance [N]). trimesh's closest_point searches every triangle within
    the nearest-VERTEX distance of a query point, so points far from a high-poly mesh (HiPHI Bottle_A_1, 91k faces) cost
    O(faces) each and blew the evaluation up to hours (and 60 GB without chunking). Points farther than `far` from the
    nearest vertex (KD-tree, cached on the query object) therefore get the nearest vertex instead: an upper bound on the
    distance that is exact to within one edge length (mm), and every metric / target only uses such points through
    thresholds of 2-6 cm or as "no interaction". Points within `far` are exact, computed in chunks."""
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    closest = np.zeros_like(pts); dist = np.zeros(len(pts))
    if len(pts) == 0:
        return closest, dist
    kd = getattr(pq, "_vertex_kdtree", None)
    if kd is None:
        kd = cKDTree(np.asarray(_pq_mesh(pq).vertices)); pq._vertex_kdtree = kd
    dv, iv = kd.query(pts)
    near = dv < far
    closest[~near] = np.asarray(_pq_mesh(pq).vertices)[iv[~near]]; dist[~near] = dv[~near]
    idx = np.where(near)[0]
    for i in range(0, len(idx), chunk):
        sel = idx[i:i + chunk]
        c, d_, _ = pq.on_surface(pts[sel])
        closest[sel] = c; dist[sel] = d_
    return closest, dist


def body_id(m, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise KeyError(f"body {name} not in model; bodies: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, j) for j in range(m.nbody)]}")
    return i


# ----------------------------------------------------------------------------- per-frame evaluation
def evaluate(m: mujoco.MjModel, qpos_robot: np.ndarray, obj_pos: np.ndarray, obj_quat_wxyz: np.ndarray, obj_body: str):
    """Runs FK + collision per frame. qpos_robot may carry any number of robot DoFs (body only, or body + fingers);
    the object freejoint is the last 7 qpos entries. Returns dict of per-frame arrays."""
    d = mujoco.MjData(m)
    T = len(qpos_robot)
    nq_r = qpos_robot.shape[1]
    assert m.nq == nq_r + 7, f"model nq={m.nq} vs robot qpos {nq_r} + 7"
    obj_bid = body_id(m, obj_body)
    ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    is_obj = np.array([m.geom_bodyid[g] == obj_bid for g in range(m.ngeom)])
    is_robot = np.array([(m.geom_contype[g] or m.geom_conaffinity[g]) and not is_obj[g] and g != ground for g in range(m.ngeom)])
    links = {k: body_id(m, v) for k, v in ROBOT_LINKS.items()}
    out = {k: np.zeros((T, 3)) for k in links}
    obj_pen = np.zeros(T); gnd_pen = np.zeros(T)
    for t in range(T):
        d.qpos[:nq_r] = qpos_robot[t]
        d.qpos[nq_r:nq_r + 3] = obj_pos[t]; d.qpos[nq_r + 3:nq_r + 7] = obj_quat_wxyz[t]
        mujoco.mj_forward(m, d)
        mujoco.mj_collision(m, d)
        for k, b in links.items():
            out[k][t] = d.xpos[b] + (d.xmat[b].reshape(3, 3) @ WRIST_OFFSET if "Wrist" in k else 0.0)
        worst_obj = 0.0; worst_gnd = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            g1, g2 = c.geom1, c.geom2
            if (is_obj[g1] and is_robot[g2]) or (is_obj[g2] and is_robot[g1]):
                worst_obj = max(worst_obj, -c.dist)
            if (g1 == ground and is_robot[g2]) or (g2 == ground and is_robot[g1]):
                worst_gnd = max(worst_gnd, -c.dist)
        obj_pen[t] = worst_obj; gnd_pen[t] = worst_gnd
    return {"links": out, "obj_pen": obj_pen, "ground_pen": gnd_pen}


def stance_mask(toe_xyz: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    v = np.linalg.norm(np.diff(toe_xyz[:, :2], axis=0), axis=1)
    mask = np.zeros(len(toe_xyz), bool); mask[1:] = v < thresh
    return mask


def foot_slide(ankle_xyz: np.ndarray, stance: np.ndarray) -> float:
    step = np.linalg.norm(np.diff(ankle_xyz[:, :2], axis=0), axis=1)
    s = stance[1:]
    return float(step[s].mean()) if s.any() else 0.0


def jerk(q: np.ndarray, dt: float) -> float:
    if len(q) < 4:
        return float("nan")
    return float(np.abs(np.diff(q, n=3, axis=0) / dt**3).mean())


def limit_violation(m_plain: mujoco.MjModel, qj: np.ndarray) -> float:
    lo, hi = [], []
    for j in range(m_plain.njnt):
        if m_plain.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            lo.append(m_plain.jnt_range[j, 0]); hi.append(m_plain.jnt_range[j, 1])
    lo, hi = np.array(lo), np.array(hi)
    viol = (qj < lo - 1e-4) | (qj > hi + 1e-4)
    return float(viol.mean())


def human_object_contact(joints: np.ndarray, obj_mesh: trimesh.Trimesh, obj_pos, obj_quat_xyzw, thresh=0.10):
    """per-frame: is either wrist within `thresh` of the object surface (source world)."""
    from scipy.spatial.transform import Rotation as Rot
    T = len(joints)
    near = np.zeros(T, bool)
    pq = trimesh.proximity.ProximityQuery(obj_mesh)
    for t in range(T):
        R = Rot.from_quat(obj_quat_xyzw[t]).as_matrix()
        w = joints[t, [SX["L_Wrist"], SX["R_Wrist"]]]
        local = (w - obj_pos[t]) @ R  # world -> object frame
        dist = on_surface_chunked(pq, local)[1]  # unsigned (see robot_object_contact)
        near[t] = (dist < thresh).any()
    return near


def robot_object_contact(hands: dict, obj_mesh: trimesh.Trimesh, scale: float, obj_pos, obj_quat_wxyz, thresh=0.10):
    from scipy.spatial.transform import Rotation as Rot
    m = obj_mesh.copy(); m.apply_scale(scale)
    pq = trimesh.proximity.ProximityQuery(m)
    T = len(obj_pos)
    near = np.zeros(T, bool)
    for t in range(T):
        R = Rot.from_quat(obj_quat_wxyz[t][[1, 2, 3, 0]]).as_matrix()
        w = np.stack([hands["L_Wrist"][t], hands["R_Wrist"][t]])
        local = (w - obj_pos[t]) @ R
        near[t] = (on_surface_chunked(pq, local)[1] < thresh).any()  # unsigned surface distance; signed_distance was the HiPHI bottleneck
    return near


# ----------------------------------------------------------------------------- per sequence
def load_fallback(method: str) -> dict[str, str]:
    p = LANES[method][0] / "run_summary.json"
    if not p.exists():
        return {}
    return {r["seq"]: r.get("fallback") or "" for r in json.loads(p.read_text()) if r.get("ok")}


def load_runtime(method: str) -> dict[str, float]:
    p = LANES[method][0] / "run_summary.json"
    if not p.exists():
        return {}
    return {r["seq"]: r["seconds"] for r in json.loads(p.read_text()) if r.get("ok")}


def eval_sequence(e: dict, m_plain: mujoco.MjModel, runtimes: dict, fallbacks: dict) -> list[dict]:
    seq, obj = e["seq"], e["object"]
    src = np.load(DATA / "source" / f"{seq}.npz", allow_pickle=True)
    joints = src["joints55"]
    obj_mesh = trimesh.load(DATA / "objects" / obj / f"{obj}.obj", force="mesh")
    human_near = human_object_contact(joints, obj_mesh, src["obj_pos"], src["obj_quat_xyzw"])
    rows = []
    for method, (rdir, suffix) in LANES.items():
        p = rdir / f"{seq}{suffix}.npz"
        if not p.exists():
            continue
        r = np.load(p, allow_pickle=True)
        q = body_qpos36(r)
        T = min(len(q), len(joints))
        q = q[:T]
        is_omni = "human_joints" in r
        if is_omni and seq not in fallbacks.get(method, {}) and "fallback" in r:  # run_summary.json missing (run interrupted)
            fallbacks.setdefault(method, {})[seq] = str(r["fallback"])
        if is_omni:
            scale = 1.32 / float(src["tpose_height"])
            obj_pos, obj_quat = r["qpos"][:T, 36:39], r["qpos"][:T, 39:43]
            mesh_scale = float(r["object_mesh_scale"]) if "object_mesh_scale" in r else 1.0  # umr-style scaling: s, native: 1
            hj = r["human_joints"][:T]
            human_w = {k: hj[:, IM[k]] for k in ("L_Wrist", "R_Wrist", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe")}
            ground_z = 0.0
        else:
            scale = float(np.asarray(r["smpl_scale"]).reshape(-1)[0])
            obj_pos = umr_object_pos(r, src, scale, T)
            obj_quat = src["obj_quat_xyzw"][:T][:, [3, 0, 1, 2]]
            mesh_scale = float(np.asarray(r["object_mesh_scale"]).reshape(-1)[0]) if "object_mesh_scale" in r else scale  # umr: s, real: 1
            human_w = {k: joints[:T, SX[k]] * scale for k in ("L_Wrist", "R_Wrist", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe")}
            ground_z = float(np.asarray(r["ground_z"]).reshape(-1)[0]) if "ground_z" in r else 0.0
        if is_omni:  # evaluate with the exact MJCF the run used (hand collision geoms included)
            m = omni_model(str(r["hand"]) if "hand" in r else "none", str(r["scaling"]) if "scaling" in r else "omni",
                           omni_subject(r, e["subject"]), obj)
            ev = evaluate(m, q[:, :36], obj_pos, obj_quat, f"{obj}_link")
        elif "robot_xml" not in r:
            m = robot_object_model(obj, mesh_scale)
            ev = evaluate(m, q[:, :36], obj_pos, obj_quat, f"{obj}_link")
        else:  # UMR: evaluate with the robot model it optimized (body-only G1 or G1 + fingers), see umr_robot_xml
            m = augmented_model(umr_robot_xml(r), obj, mesh_scale)
            ev = evaluate(m, r["qpos"][:T], obj_pos, obj_quat, f"{obj}_link")
        L = ev["links"]
        stance_l, stance_r = stance_mask(human_w["L_Toe"]), stance_mask(human_w["R_Toe"])
        robot_near = robot_object_contact({"L_Wrist": L["L_Wrist"], "R_Wrist": L["R_Wrist"]}, obj_mesh, mesh_scale, obj_pos, obj_quat)
        hn = human_near[:T]
        rows.append({
            "seq": seq, "object": obj, "subject": e["subject"], "method": method, "frames": T, "smpl_scale": round(scale, 4),
            "obj_pen_max": ev["obj_pen"].mean(), "obj_pen_frac": (ev["obj_pen"] > 0.005).mean(),
            "ground_pen": ev["ground_pen"].mean(),
            "foot_slide": 0.5 * (foot_slide(L["L_Ankle"], stance_l) + foot_slide(L["R_Ankle"], stance_r)),
            "hand_contact": float((robot_near & hn).sum() / max(hn.sum(), 1)) if hn.any() else float("nan"),
            "human_contact_frac": hn.mean(),
            "wrist_err": 0.5 * (np.linalg.norm(L["L_Wrist"] - human_w["L_Wrist"], axis=1).mean() + np.linalg.norm(L["R_Wrist"] - human_w["R_Wrist"], axis=1).mean()),
            "ankle_err": 0.5 * (np.linalg.norm(L["L_Ankle"] - human_w["L_Ankle"], axis=1).mean() + np.linalg.norm(L["R_Ankle"] - human_w["R_Ankle"], axis=1).mean()),
            "joint_jerk": jerk(q[:, 7:36], 1 / FPS), "root_jerk": jerk(q[:, :3], 1 / FPS),
            "limit_viol": limit_violation(m_plain, q[:, 7:36]),
            "runtime_s": runtimes[method].get(seq, float("nan")),
            "fallback": fallbacks.get(method, {}).get(seq, ""),
        })
        rows[-1]["runtime_per_frame"] = rows[-1]["runtime_s"] / T
    return rows


def summarize(rows: list[dict]) -> str:
    metrics = ["obj_pen_max", "obj_pen_frac", "ground_pen", "foot_slide", "hand_contact", "wrist_err", "ankle_err",
               "joint_jerk", "root_jerk", "limit_viol", "runtime_per_frame"]
    higher = {"hand_contact"}
    methods = [m for m in LANES if any(r["method"] == m for r in rows)]
    n_index = len({r["seq"] for r in rows})
    lines = [f"# Body-stage metrics on {DATASET.upper()}: " + " vs ".join(methods), ""]
    common = set.intersection(*[{r["seq"] for r in rows if r["method"] == m} for m in methods]) if methods else set()
    lines.append("Sequences evaluated: " + ", ".join(f"{m}: {sum(r['method']==m for r in rows)}" for m in methods) +
                 f"; with results from all methods: {len(common)}")
    for m in methods:
        fb = [r for r in rows if r["method"] == m and r.get("fallback") not in ("", "default", None)]
        if fb:
            lines.append(f"- {m}: {len(fb)} sequences needed a fallback (" +
                         ", ".join(f"{k}: {v}" for k, v in sorted(__import__('collections').Counter(r['fallback'] for r in fb).items())) + ")")
    lines += ["", f"## Mean over the {len(common)} sequences solved by every method (lower is better unless noted)", "",
              "| metric | " + " | ".join(methods) + " | best |", "|---|" + "---|" * (len(methods) + 1)]
    for mtr in metrics:
        vals = {}
        for m in methods:
            v = np.array([r[mtr] for r in rows if r["method"] == m and r["seq"] in common], float)
            vals[m] = np.nanmean(v) if len(v) else float("nan")
        ok = [m for m in methods if not np.isnan(vals[m])]
        best = (max if mtr in higher else min)(ok, key=lambda m: vals[m]) if ok else "-"
        lines.append(f"| {mtr}{' (higher better)' if mtr in higher else ''} | " + " | ".join(f"{vals[m]:.4f}" for m in methods) + f" | {best} |")
    for hand in ("brainco", "wuji"):
        pair = [m for m in methods if m.endswith("_" + hand)]
        both = set.intersection(*[{r["seq"] for r in rows if r["method"] == m} for m in pair]) if len(pair) >= 2 else set()
        if len(pair) < 2:
            continue
        lines += ["", f"## G1 + {hand}: mean over the {len(both)} sequences solved by {' and '.join(pair)}", "",
                  "| metric | " + " | ".join(pair) + " | best |", "|---|" + "---|" * (len(pair) + 1)]
        for mtr in metrics:
            vals = {m: np.nanmean(np.array([r[mtr] for r in rows if r["method"] == m and r["seq"] in both], float)) if both else float("nan") for m in pair}
            ok = [m for m in pair if not np.isnan(vals[m])]
            best = (max if mtr in higher else min)(ok, key=lambda m: vals[m]) if ok else "-"
            lines.append(f"| {mtr}{' (higher better)' if mtr in higher else ''} | " + " | ".join(f"{vals[m]:.4f}" for m in pair) + f" | {best} |")
    lines += ["", "## Per object (mean over that method's solved sequences): object penetration [mm] / hand-contact preservation / wrist error [cm]", "",
              "| object | n | " + " | ".join(f"{m} pen | {m} contact | {m} wrist" for m in methods) + " |",
              "|---|---|" + "---|" * (3 * len(methods))]
    for obj in sorted({r["object"] for r in rows}):
        def agg(method, mtr):
            v = np.array([r[mtr] for r in rows if r["method"] == method and r["object"] == obj], float)
            return np.nanmean(v) if len(v) else float("nan")
        n = len({r["seq"] for r in rows if r["object"] == obj})
        cells = " | ".join(f"{1e3*agg(m,'obj_pen_max'):.1f} | {agg(m,'hand_contact'):.2f} | {1e2*agg(m,'wrist_err'):.1f}" for m in methods)
        lines.append(f"| {obj} | {n} | {cells} |")
    lines += ["", "## Success rate per method (sequences in index: %d)" % n_index, "", "| method | solved | fallback used |", "|---|---|---|"]
    for m in methods:
        solved = sum(r["method"] == m for r in rows)
        fb = sum(1 for r in rows if r["method"] == m and r.get("fallback") not in ("", "default", None))
        lines.append(f"| {m} | {solved} | {fb} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=DATA / "seqs_test.json")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--lane", action="append", default=None, metavar="NAME=DIR[:SUFFIX]",
                    help="result lane (repeatable); given lanes REPLACE the defaults, e.g. --lane omni=results/omni:_original --lane umr=results/umr")
    ap.add_argument("--out", default="", help="suffix for results/metrics<out>.csv and results/summary<out>.md")
    ap.add_argument("--merge", nargs="*", default=None, metavar="CSV", help="skip evaluation: merge these per-chunk metrics csvs and write summary<out>")
    args = ap.parse_args()
    if args.merge is not None:
        merged = {}  # (seq, method) -> row; later csvs override earlier ones (re-evaluated sequences)
        for c in args.merge:
            with open(c) as f:
                for r in csv.DictReader(f):
                    row = {k: (v if k in ("seq", "object", "subject", "method", "fallback") else float(v)) for k, v in r.items()}
                    merged[(row["seq"], row["method"])] = row
        rows = list(merged.values())
        if args.lane:
            LANES.clear()
            for spec in args.lane:
                name, rest = spec.split("=", 1); d, _, suffix = rest.partition(":"); LANES[name] = (HERE / d, suffix)
        with open(RESULTS / f"metrics{args.out}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        (RESULTS / f"summary{args.out}.md").write_text(summarize(rows))
        print(summarize(rows)); return
    if args.lane:
        LANES.clear()
    index = json.loads(args.index.read_text())
    if args.seqs:
        index = [e for e in index if e["seq"] in set(args.seqs)]
    m_plain = mujoco.MjModel.from_xml_path(str(WORK / "models" / "g1" / "g1_29dof.xml"))
    for spec in args.lane or []:
        name, rest = spec.split("=", 1)
        d, _, suffix = rest.partition(":")
        LANES[name] = ((HERE / d) if not Path(d).is_absolute() else Path(d), suffix)
    runtimes = {m: load_runtime(m) for m in LANES}
    fallbacks = {m: load_fallback(m) for m in LANES}
    rows = []
    for e in index:
        try:
            rs = eval_sequence(e, m_plain, runtimes, fallbacks)
        except Exception as ex:  # keep going, report
            print(f"[{e['seq']}] ERROR {type(ex).__name__}: {ex}"); continue
        for r in rs:
            print(f"[{r['seq']}] {r['method']:5s} pen={1e3*r['obj_pen_max']:.1f}mm ({100*r['obj_pen_frac']:.0f}% frames) ground={1e3*r['ground_pen']:.1f}mm "
                  f"slide={1e3*r['foot_slide']:.2f}mm contact={r['hand_contact']:.2f} wrist={1e2*r['wrist_err']:.1f}cm ankle={1e2*r['ankle_err']:.1f}cm "
                  f"jerk={r['joint_jerk']:.0f} lim={100*r['limit_viol']:.1f}% t={r['runtime_s']:.0f}s")
        rows += rs
    if not rows:
        raise SystemExit("no results")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"metrics{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (RESULTS / f"summary{args.out}.md").write_text(summarize(rows))
    print(f"\nwrote {RESULTS/f'metrics{args.out}.csv'} and {RESULTS/f'summary{args.out}.md'}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
