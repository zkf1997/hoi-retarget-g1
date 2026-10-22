#!/usr/bin/env python3
"""Side-by-side viser viewer: OMOMO human + object  |  OmniRetarget G1  |  UMR G1   (hands fixed at a rest pose).

Run in the `hsretargeting` conda env (viser, yourdfpy, trimesh, torch, smplx):
    python viser_compare.py                       # default lanes: omni_brainco | umr_brainco | omni_wuji | umr_wuji (UMR scaling)
    python viser_compare.py --seqs sub16_largebox_000 --port 8081
    python viser_compare.py --lane omni_brainco=results/omni_brainco_umrscale:_original --lane umr_brainco=results/umr_brainco
    python viser_compare.py --lane omni=results/omni:_original --lane umr=results/umr --hand none   # legacy body-only results
    BENCH_DATASET=humoto python viser_compare.py --pipeline      # stage lanes: UMR body | +AnyTeleop fingers | +finger refinement | +arm placement
Each lane picks its URDF (BrainCo / Wuji) from the `hand` key stored in the result npz; --hand is only the fallback.
BENCH_DATASET=humoto switches to data_humoto/ and results_humoto/ (HUMOTO lanes).

Lanes (offset along +y, spacing adjustable in the GUI):
  0  source: SMPL-X mesh (if the body model is available, else joints) + object, original OMOMO world
  1  OmniRetarget: results/omni/<seq>_original.npz  qpos[0:36] robot, qpos[36:43] object (pos, quat wxyz), its own scaled world
  2  UMR:          results/umr/<seq>.npz            qpos[0:36] robot; object = source pose * smpl_scale, mesh scaled by smpl_scale
Body joints come from the retargeting result. Hand joints are held at hand_rest_pose.json unless the result itself
contains them (UMR run with a hand-equipped robot, mapped by `robot_joint_names`).
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as vtf
import yourdfpy
from scipy.spatial.transform import Rotation as Rot
import mujoco

# MuJoCo copies of the URDFs (make_hand_mjcf.py) used only for fast per-frame FK in the viewer; yourdfpy's scene-graph
# FK cost ~25 ms per robot per frame, MuJoCo ~0.1 ms.
def mjcf_for(hand: str) -> Path:  # HERE is defined further down
    return {"brainco": HERE / "robots" / "g1_29dof_brainco_revo2.xml", "wuji": HERE / "robots" / "g1_29dof_wuji_hand2.xml",
            "none": HERE.parent / "UMR" / "assets" / "g1" / "g1_29dof.xml"}[hand]


def fk_model(xml: Path) -> mujoco.MjModel:
    """Robot MJCF with a free root joint (the URDF-derived MJCFs have none), the way compare.augmented_model adds it."""
    import xml.etree.ElementTree as ET
    root = ET.parse(xml).getroot()
    if root.find(".//freejoint") is None and not any(j.get("type") == "free" for j in root.iter("joint")):
        wb = root.find("worldbody")
        kids = [c for c in list(wb) if c.tag == "body"]
        wb.remove(kids[0])
        fr = ET.SubElement(wb, "body", name="float_root", pos="0 0 0")
        ET.SubElement(fr, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-6 1e-6 1e-6")
        ET.SubElement(fr, "freejoint", name="float_root_freejoint")
        fr.append(kids[0])
    for mesh in root.iter("mesh"):  # mesh paths are relative to the xml's directory; loading from a string loses that
        f = mesh.get("file")
        if f and not Path(f).is_absolute():
            mesh.set("file", str((xml.parent / f).resolve()))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

HERE = Path(__file__).resolve().parent
# Dataset switch: BENCH_DATASET=omomo (default) | humoto  ->  data[_humoto]/, results[_humoto]/, omni_workdir[_humoto]/
HUMAN_FACES = 4000  # decimated source-human mesh (full SMPL-X: 20 908 faces)
DATASET = os.environ.get("BENCH_DATASET", "omomo")
SUFFIX = "" if DATASET == "omomo" else f"_{DATASET}"
DATA = HERE / f"data{SUFFIX}"
RESULTS = HERE / f"results{SUFFIX}"
URDFS = {
    "brainco": HERE.parent / "urdf" / "g1_brainco_revo2" / "g1_29dof_brainco_revo2.urdf",
    "wuji": HERE.parent / "urdf" / "g1_wuji_hand2" / "g1_29dof_wuji_hand2.urdf",
    "none": HERE.parent / "urdf" / "third_party" / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf",
}
# SMPL-X body models (license-gated, not in git): $SMPLX_DIR/smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz, else UMR/smpl/ (symlinks
# made by env_setup.sh). Without them the source human is drawn as joints only.
SMPLX_DIR = Path(os.environ.get("SMPLX_DIR", "/media/kaizhao/hdd/dataset/models_smplx_v1_1/models"))
if not (SMPLX_DIR / "smplx").exists() and (HERE.parent / "UMR" / "smpl" / "SMPLX_NEUTRAL.npz").exists():
    SMPLX_DIR = HERE.parent / "UMR"  # smplx.create(model_path, model_type="smplx") looks for <model_path>/smplx/ or the files directly


# qpos[7:36] order shared by holosoma_retargeting and UMR (MuJoCo g1_29dof.xml joint order)
G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LANE_COLORS = {"omni": (70, 130, 230), "umr": (230, 120, 60)}


# ----------------------------------------------------------------------------- hand rest pose
def hand_rest_pose(urdf: yourdfpy.URDF, cfg_path: Path) -> dict[str, float]:
    cfg = json.loads(cfg_path.read_text())
    out = {}
    for j in urdf.actuated_joint_names:
        if j in G1_JOINTS:
            continue
        v = cfg.get("default", 0.0)
        for rule in cfg.get("rules", []):
            if fnmatch.fnmatch(j, rule["match"]):
                v = rule["value"]
        lim = urdf.joint_map[j].limit
        if lim is not None and lim.lower is not None and lim.upper is not None:
            v = float(np.clip(v, lim.lower, lim.upper))
        out[j] = v
    return out


# ----------------------------------------------------------------------------- robot lane
class RobotLane:
    """One posed G1 in the viser scene: per-link mesh handles under a lane frame."""

    def __init__(self, server: viser.ViserServer, name: str, urdf_path: Path, rest: Path, color, lane_y: float):
        self.server, self.name = server, name
        self.urdf = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=True,
                                       build_collision_scene_graph=False, load_collision_meshes=False)
        self.rest = hand_rest_pose(self.urdf, rest)
        self.frame = server.scene.add_frame(f"/{name}", position=(0.0, lane_y, 0.0), show_axes=False)
        self.label = server.scene.add_label(f"/{name}/label", text=name, position=(0.0, 0.0, 1.9))
        self.handles: list[tuple[str, np.ndarray, object]] = []  # (link, T_link_geom, handle)
        sg = self.urdf.scene.graph
        for node in sg.nodes_geometry:
            parent, geom_name = sg[node]  # transform (unused), geometry key
            geom = self.urdf.scene.geometry[geom_name]
            if not isinstance(geom, trimesh.Trimesh):
                continue
            link = self._link_of(node)
            T_link_geom = self.urdf.scene.graph.get(node, frame_from=link)[0]
            mesh = geom.copy()
            mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=np.tile(np.array([*color, 255], dtype=np.uint8), (len(mesh.faces), 1)))
            h = server.scene.add_mesh_trimesh(f"/{name}/{node}", mesh)
            self.handles.append((link, T_link_geom, h))
        self.object = None
        self.visible = True
        # MuJoCo FK: joint -> qpos address, link -> body id, geom offsets stacked for one batched transform per frame
        hand = next((k for k, v in URDFS.items() if Path(v).resolve() == Path(urdf_path).resolve()), "none")
        self.mj = fk_model(mjcf_for(hand)); self.mjd = mujoco.MjData(self.mj)
        self.qadr = {mujoco.mj_id2name(self.mj, mujoco.mjtObj.mjOBJ_JOINT, j): int(self.mj.jnt_qposadr[j]) for j in range(self.mj.njnt)
                     if self.mj.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE}
        self.bid = np.array([mujoco.mj_name2id(self.mj, mujoco.mjtObj.mjOBJ_BODY, link) for link, _, _ in self.handles])
        assert (self.bid >= 0).all(), f"URDF links without MJCF body: {[l for (l, _, _), b in zip(self.handles, self.bid) if b < 0]}"
        self.T_lg = np.stack([T for _, T, _ in self.handles])  # (n, 4, 4)
        self.q0 = np.zeros(self.mj.nq); self.q0[3] = 1.0
        for j, v in self.rest.items():
            if j in self.qadr:
                self.q0[self.qadr[j]] = v

    def _link_of(self, node: str) -> str:
        sg = self.urdf.scene.graph
        while node not in self.urdf.link_map:
            node = sg.transforms.parents[node]
        return node

    def set_extras(self, extras: list, scale: float = 1.0):
        for h in getattr(self, "extra_handles", []):
            h.remove()
        self.extra_handles, self.extra_data = [], extras
        for k, ex in enumerate(extras):
            self.extra_handles.append(add_object_mesh(self.server, f"/{self.name}/extra_{k}", ex["mesh"], (160, 190, 175), scale))

    def set_object(self, mesh: trimesh.Trimesh | None, scale: float = 1.0):
        if self.object is not None:
            self.object.remove(); self.object = None
        if mesh is None:
            return
        self.object = add_object_mesh(self.server, f"/{self.name}/object", mesh, (190, 180, 160), scale)

    def pose(self, qpos: np.ndarray, obj_pos=None, obj_quat_wxyz=None, joint_names=None):
        names = list(joint_names) if joint_names is not None else G1_JOINTS
        q = self.q0.copy()  # hands at rest unless the result itself carries hand joints
        q[:7] = qpos[:7]  # root xyz + quaternion wxyz, same convention as MuJoCo's free joint
        for j, v in zip(names, qpos[7:7 + len(names)]):
            a = self.qadr.get(j)
            if a is not None:
                q[a] = v
        self.mjd.qpos[:] = q
        mujoco.mj_kinematics(self.mj, self.mjd)
        T = np.tile(np.eye(4), (len(self.bid), 1, 1))
        T[:, :3, :3] = self.mjd.xmat[self.bid].reshape(-1, 3, 3); T[:, :3, 3] = self.mjd.xpos[self.bid]
        T = T @ self.T_lg
        quat_xyzw = Rot.from_matrix(T[:, :3, :3]).as_quat()
        with self.server.atomic():
            for (_, _, h), p, qx in zip(self.handles, T[:, :3, 3], quat_xyzw):
                h.position = p
                h.wxyz = (qx[3], qx[0], qx[1], qx[2])
            if self.object is not None and obj_pos is not None:
                self.object.position = obj_pos
                self.object.wxyz = obj_quat_wxyz
            t = getattr(self, "_t", None)
            for h, ex in zip(getattr(self, "extra_handles", []), getattr(self, "extra_data", [])):
                if t is not None:
                    i = min(t, len(ex["pos"]) - 1); h.position = ex["pos"][i]; h.wxyz = ex["quat"][i]

    def set_visible(self, v: bool):
        self.frame.visible = v
        self.visible = v


# ----------------------------------------------------------------------------- source lane (human)
class HumanLane:
    def __init__(self, server: viser.ViserServer, lane_y: float, use_smplx: bool):
        self.server = server
        self.frame = server.scene.add_frame("/human", position=(0.0, lane_y, 0.0), show_axes=False)
        self.label = server.scene.add_label("/human/label", text=f"{DATASET.upper()} source (human)", position=(0.0, 0.0, 1.9))
        self.mesh_handle = None
        self.joints_handle = None
        self.object = None
        self.extras, self.extra_pos, self.extra_quat = [], [], []
        self.verts = None
        self.model = None
        self.world = (1.0, 0.0)  # (scale, ground_z): source shown in the robot world = (p - [0,0,ground_z]) * scale
        if use_smplx:
            try:
                import smplx, torch  # noqa
                self.torch, self.smplx = torch, smplx
                self.model = lambda gender, T: smplx.create(str(SMPLX_DIR), model_type="smplx", gender=gender, num_betas=16,
                                                            use_pca=False, flat_hand_mean=True, batch_size=T, ext="npz")
                self.model_path_neutral = next((p for p in (HERE.parent / "UMR" / "smpl" / "SMPLX_NEUTRAL.npz", SMPLX_DIR / "smplx" / "SMPLX_NEUTRAL.npz") if p.exists()),
                                               HERE.parent / "UMR" / "smpl" / "SMPLX_NEUTRAL.npz")  # same lookup as render_compare.py
            except Exception as e:  # pragma: no cover
                print("[human] smplx unavailable, showing joints only:", e)

    def load(self, src: dict, obj_mesh: trimesh.Trimesh):
        joints = src["joints55"]
        verts, faces = None, None
        if self.model is not None:
            try:
                torch = self.torch
                T = len(joints)
                with torch.no_grad():
                    gender = str(src["gender"])
                    m = self.model(gender, T) if gender != "neutral" else self.smplx.create(str(self.model_path_neutral), model_type="smplx", gender="neutral", num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=T, ext="npz")
                    hands = {}
                    if "pose_hand" in src:  # HUMOTO fits carry finger poses (OMOMO: flat hands)
                        ph = np.asarray(src["pose_hand"], dtype=np.float32)
                        hands = {"left_hand_pose": torch.as_tensor(ph[:, :45]), "right_hand_pose": torch.as_tensor(ph[:, 45:])}
                    out = m(betas=torch.as_tensor(np.repeat(src["betas"][None], T, 0), dtype=torch.float32),
                            global_orient=torch.as_tensor(src["root_orient"], dtype=torch.float32),
                            body_pose=torch.as_tensor(src["pose_body"], dtype=torch.float32),
                            transl=torch.as_tensor(src["trans"], dtype=torch.float32), **hands)
                    verts = out.vertices.numpy(); faces = m.faces.astype(np.int32)
            except Exception as ex:  # SMPL-X files missing (license-gated) or unreadable: draw joints instead
                if not getattr(self, '_warned', False):
                    print(f'[human] SMPL-X unavailable ({type(ex).__name__}: {str(ex)[:120]}); showing joints only', flush=True); self._warned = True
                verts, faces = None, None
        for h in (self.mesh_handle, self.joints_handle, self.object):
            if h is not None:
                h.remove()
        self.mesh_handle = self.joints_handle = None
        self.joints, self.verts, self.faces = joints, verts, faces
        if self.verts is not None:
            # the full SMPL-X mesh is 10 475 vertices = 126 KB per frame at 30 fps, too much for a tunnelled browser; show a
            # decimated mesh whose vertices are a subset of the original (nearest original vertex of the decimated one), so the
            # per-frame update is a cheap index gather of the current vertices
            try:
                from scipy.spatial import cKDTree
                dec = trimesh.Trimesh(self.verts[0], self.faces, process=False).simplify_quadric_decimation(face_count=HUMAN_FACES)
                self.vidx = cKDTree(self.verts[0]).query(np.asarray(dec.vertices))[1].astype(np.int64)
                self.faces = np.asarray(dec.faces, dtype=np.int32); self.verts = np.ascontiguousarray(self.verts[:, self.vidx], dtype=np.float32)
            except Exception as ex:  # fast_simplification missing etc.: full mesh
                print(f"[human] decimation unavailable ({type(ex).__name__}: {ex}); full mesh", flush=True)
            self.mesh_handle = self.server.scene.add_mesh_simple("/human/body", self.verts[0], self.faces, color=(200, 160, 140))
        else:
            self.joints_handle = self.server.scene.add_point_cloud("/human/joints", self.joints[0], colors=(200, 60, 60), point_size=0.03)
        self.object = add_object_mesh(self.server, "/human/object", obj_mesh, (190, 180, 160))
        self.obj_pos, self.obj_quat = src["obj_pos"], src["obj_quat_xyzw"][:, [3, 0, 1, 2]]
        # other scene objects (HUMOTO furniture etc.): shown in the source lane only, not retargeted
        for h in self.extras:
            h.remove()
        self.extras, self.extra_pos, self.extra_quat = [], [], []
        for k, name in enumerate([str(x) for x in src["extra_objects"]] if "extra_objects" in src else []):
            mesh = load_extra_object_mesh(name)
            if mesh is None:
                continue
            self.extras.append(add_object_mesh(self.server, f"/human/extra_{k}", mesh, (150, 150, 160)))
            self.extra_pos.append(src["extra_obj_pos"][:, k]); self.extra_quat.append(src["extra_obj_quat_xyzw"][:, k][:, [3, 0, 1, 2]])

    def set_world(self, scale: float, ground_z: float):
        self.world = (float(scale), float(ground_z))
        if self.object is not None:
            self.object.scale = self.world[0]
        for h in self.extras:
            h.scale = self.world[0]

    def _w(self, p):
        s, gz = self.world
        return (p - np.array([0.0, 0.0, gz])) * s if s != 1.0 or gz != 0.0 else p

    def pose(self, t: int):
        if self.mesh_handle is not None and self.verts is not None and self.frame.visible:  # hidden human: no vertex traffic
            self.mesh_handle.vertices = self._w(self.verts[min(t, len(self.verts) - 1)])
        elif self.joints_handle is not None:
            self.joints_handle.points = self._w(self.joints[min(t, len(self.joints) - 1)])
        if self.object is None:
            return
        t = min(t, len(self.obj_pos) - 1)
        self.object.position = self._w(self.obj_pos[t])
        self.object.wxyz = self.obj_quat[t]
        for h, p, q in zip(self.extras, self.extra_pos, self.extra_quat):
            h.position = self._w(p[t]); h.wxyz = q[t]

    def set_visible(self, v: bool):
        self.frame.visible = v


def display_mesh(name: str):
    """Object mesh for display: the decimated <obj>_display.obj (make_display_meshes.py, <= 5k faces) if present, else the
    full visual mesh. Full meshes (up to 90k faces, several per lane) stalled the browser when switching sequences."""
    p = DATA / "objects" / name / f"{name}_display.obj"
    if p.exists():
        return trimesh.load(p, force="mesh")
    p = DATA / "objects" / name / f"{name}.obj"
    return trimesh.load(p, force="mesh") if p.exists() else None


def add_object_mesh(server, path: str, mesh, rgb, scale: float = 1.0):
    """Single-colour mesh upload (add_mesh_simple): no per-face colour array, far smaller than add_mesh_trimesh."""
    return server.scene.add_mesh_simple(path, np.asarray(mesh.vertices, dtype=np.float32) * scale, np.asarray(mesh.faces, dtype=np.uint32), color=rgb)


def load_extra_object_mesh(name: str):
    """Visual mesh of a non-retargeted scene object: data*/objects/<name>/<name>.obj or the HUMOTO export in hand_bench."""
    for p in (DATA / "objects" / name / f"{name}_display.obj", DATA / "objects" / name / f"{name}.obj", HERE.parent / "hand_bench" / "data" / "objects" / name / "visual.obj"):
        if p.exists():
            return trimesh.load(p, force="mesh")
    try:  # objects that hand_bench did not export (furniture): read the GLB node mesh directly
        import sys
        sys.path.insert(0, str(HERE.parent / "hand_bench"))
        import humoto_io as H
        return H.load_object_mesh(_current_seq[0], name)
    except Exception:
        return None


_current_seq = [""]
ALL_OBJECTS = [False]


# ----------------------------------------------------------------------------- data
LANES: list[dict] = []          # filled from --lane in main(); each {"name", "dir", "suffix"}
DEFAULT_LANES = [f"omni_brainco={RESULTS}/omni_brainco_umrscale:_original", f"umr_brainco={RESULTS}/umr_brainco",
                 f"omni_wuji={RESULTS}/omni_wuji_umrscale:_original", f"umr_wuji={RESULTS}/umr_wuji"]
# decoupled pipeline (hand_pipeline.py): one lane per stage, in order
PIPELINE_LANES = [f"1_umr_body={RESULTS}/umr_wuji", f"2_anyteleop_fingers={RESULTS}/pipe_s2",
                  f"3_finger_refine={RESULTS}/pipe_s3", f"3p_arm_place={RESULTS}/pipe_s3p", f"v2_coupled={RESULTS}/pipe_v2",
                  f"v3_dexweave={RESULTS}/pipe_v3", f"final={RESULTS}/pipe_final"]
FINAL_LANES = [f"1_umr_body={RESULTS}/umr_wuji", f"2_anyteleop_fingers={RESULTS}/pipe_s2", f"final={RESULTS}/pipe_final"]
PIPELINE_LABELS = {"1_umr_body": "stage 1: UMR body, rest hand", "2_anyteleop_fingers": "stage 2: + AnyTeleop fingers",
                   "3_finger_refine": "stage 3: + finger refinement", "3p_arm_place": "stage 3+: + arm placement (v1)",
                   "v2_coupled": "v2: coupled arm + finger QP", "v3_dexweave": "v3: + interaction gating / priorities",
                   "final": "final: v3 with 12 iterations"}


def lane_hand(spec: dict, default: str) -> str:
    """Hand model of a result lane: the `hand` key written by run_omni.py / run_umr.py into every npz."""
    for p in sorted(spec["dir"].glob(f"*{spec['suffix']}.npz"))[:3]:
        r = np.load(p, allow_pickle=True)
        if "hand" in r:
            return str(r["hand"])
    return default
LANE_PALETTE = [(70, 130, 230), (230, 120, 60), (60, 180, 90), (170, 90, 200)]


def parse_lane(spec: str) -> dict:
    """'omni=results/omni:_original' -> {name, dir, suffix}; suffix defaults to '' (file <seq><suffix>.npz)."""
    name, rest = spec.split("=", 1)
    d, _, suffix = rest.partition(":")
    return {"name": name, "dir": (HERE / d) if not Path(d).is_absolute() else Path(d), "suffix": suffix}


def result_path(spec: dict, seq: str) -> Path:
    return spec["dir"] / f"{seq}{spec['suffix']}.npz"


def available_sequences(index_path: Path) -> list[dict]:
    idx = json.loads(index_path.read_text())
    out = []
    for e in idx:
        e = dict(e)
        e["have"] = [spec["name"] for spec in LANES if result_path(spec, e["seq"]).exists()]
        if e["have"]:
            out.append(e)
    return out


def load_sequence(e: dict) -> dict:
    _current_seq[0] = e["seq"]
    src = dict(np.load(DATA / "source" / f"{e['seq']}.npz", allow_pickle=True))
    obj = str(src["object"])
    obj_mesh = display_mesh(obj)
    d = {"src": src, "obj_mesh": obj_mesh, "T": len(src["trans"]), "fps": float(src["fps"]),
         "s": 1.32 / float(src["tpose_height"]) if "tpose_height" in src else 1.0, "ground_z": 0.0}  # robot world = human x s
    for spec in LANES:
        p = result_path(spec, e["seq"])
        if not p.exists():
            continue
        r = np.load(p, allow_pickle=True)
        q = r["qpos"]
        names = [str(x) for x in r["robot_joint_names"]] if "robot_joint_names" in r else None
        if "smpl_scale" in r:
            d["s"] = float(np.asarray(r["smpl_scale"]).reshape(-1)[0])
        if "ground_z" in r:
            d["ground_z"] = float(np.asarray(r["ground_z"]).reshape(-1)[0])
        if "human_joints" in r:  # holosoma / OmniRetarget: object pose lives in qpos[36:43]; mesh real size (native) or x s (umr scaling)
            d[spec["name"]] = {"qpos": q, "joint_names": None, "scale": float(r["object_mesh_scale"]) if "object_mesh_scale" in r else 1.0,
                               "obj_pos": q[:, 36:39] if q.shape[1] >= 43 else None,
                               "obj_quat": q[:, 39:43] if q.shape[1] >= 43 else None}
        else:                    # UMR: object = source pose * smpl_scale, mesh scaled by smpl_scale
            s_ = float(np.asarray(r["smpl_scale"]).reshape(-1)[0]) if "smpl_scale" in r else 1.0
            ms_ = float(np.asarray(r["object_mesh_scale"]).reshape(-1)[0]) if "object_mesh_scale" in r else s_  # real-size lanes: 1
            d[spec["name"]] = {"qpos": q, "joint_names": names, "scale": ms_,
                               "obj_pos": (np.asarray(r["obj_pos_robot"]) if "obj_pos_robot" in r else src["obj_pos"] * s_),
                               "obj_quat": src["obj_quat_xyzw"][:, [3, 0, 1, 2]], "extras": []}
            extra_names = [str(x) for x in src["extra_objects"]] if "extra_objects" in src else []
            objs_r = [str(x) for x in r["objects"]] if "objects" in r else [str(src["object"])]
            if ALL_OBJECTS[0] and (DATA / "multi_objects.json").exists():  # --all-objects: every object in every lane
                objs_r = json.loads((DATA / "multi_objects.json").read_text()).get(e["seq"], objs_r)
            for o in objs_r[1:]:  # multi-object hand stage (hand_pipeline --multi)
                if o in extra_names and (DATA / "objects" / o / f"{o}.obj").exists():
                    k = extra_names.index(o)
                    d[spec["name"]]["extras"].append({"name": o, "mesh": display_mesh(o),
                                                      "pos": src["extra_obj_pos"][:, k] * s_, "quat": src["extra_obj_quat_xyzw"][:, k][:, [3, 0, 1, 2]]})
    return d


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=DATA / "seqs_test.json")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--hand", choices=list(URDFS), default="brainco", help="URDF for lanes whose results carry no `hand` key")
    ap.add_argument("--rest", type=Path, default=HERE / "hand_rest_pose.json")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--lane-spacing", type=float, default=2.5)
    ap.add_argument("--no-smplx", action="store_true", help="show human joints instead of the SMPL-X mesh")
    ap.add_argument("--lane", action="append", default=None, metavar="NAME=DIR[:SUFFIX]",
                    help="robot lane from a results dir (repeatable). Default: omni=results/omni:_original  umr=results/umr")
    ap.add_argument("--pipeline", action="store_true", help="lanes = every hand_pipeline.py stage / variant side by side")
    ap.add_argument("--all-objects", action="store_true", help="draw every object of data*/multi_objects.json in every robot lane")
    ap.add_argument("--final", action="store_true", help="lanes = the stages of the chosen pipeline only: UMR body | + AnyTeleop fingers | final (v3, 12 iterations)")
    args = ap.parse_args()
    ALL_OBJECTS[0] = args.all_objects
    LANES.extend(parse_lane(x) for x in (args.lane or (FINAL_LANES if args.final else PIPELINE_LANES if args.pipeline else DEFAULT_LANES)))
    for spec in list(LANES):
        if not spec["dir"].exists():
            print(f"[viser] lane '{spec['name']}' skipped: {spec['dir']} does not exist", flush=True); LANES.remove(spec)
    if not LANES:
        raise SystemExit("no result lane found; pass --lane name=dir[:suffix] (the result archive ships results*/pipe_final only)")

    seqs = available_sequences(args.index)
    if args.seqs:
        seqs = [e for e in seqs if e["seq"] in set(args.seqs)]
    if not seqs:
        raise SystemExit("no sequences with results found")
    names = [e["seq"] for e in seqs]

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=12, height=12, position=(0, args.lane_spacing, 0))
    human = HumanLane(server, 0.0, use_smplx=not args.no_smplx)
    lanes = {}
    for i, spec in enumerate(LANES):
        hand = lane_hand(spec, args.hand)
        lanes[spec["name"]] = RobotLane(server, spec["name"], URDFS[hand], args.rest,
                                        LANE_PALETTE[i % len(LANE_PALETTE)], (i + 1) * args.lane_spacing)
        lanes[spec["name"]].label.text = f"{PIPELINE_LABELS.get(spec['name'], spec['name'])}  [{hand} URDF]"

    state = {"t": 0, "playing": True, "data": None}
    # load() (dropdown callback) vs render() (playback loop) raced and crashed on sequence switch -> one lock. It must be
    # re-entrant: viser runs a slider's on_update synchronously in the thread that sets .value, so the playback loop (holding
    # the lock) setting frame.value re-enters the frame callback; a plain Lock deadlocked when paused / switching sequences.
    lock = threading.RLock()
    state["programmatic"] = False  # True while the server itself moves the frame slider (callback must not re-render)
    with server.gui.add_folder("Sequence"):
        dd = server.gui.add_dropdown("sequence", names, initial_value=names[0])
        info = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        play = server.gui.add_checkbox("play", True)
        speed = server.gui.add_slider("speed", 0.1, 2.0, 0.1, 1.0)
        frame = server.gui.add_slider("frame", 0, 1, 1, 0)
    with server.gui.add_folder("Display"):
        show_h = server.gui.add_checkbox("source human", True)
        src_world = server.gui.add_checkbox("source in robot world (x smpl_scale)", False)
        lane_checkboxes = [server.gui.add_checkbox(spec["name"], True) for spec in LANES]
        spacing = server.gui.add_slider("lane spacing", 0.0, 5.0, 0.1, args.lane_spacing)

    def load(name: str):
        e = next(x for x in seqs if x["seq"] == name)
        d = load_sequence(e)
        human.load(d["src"], d["obj_mesh"])
        human.set_world(d["s"] if src_world.value else 1.0, d["ground_z"] if src_world.value else 0.0)
        for k, lane in lanes.items():
            if k in d:
                lane.set_object(d["obj_mesh"], d[k]["scale"]); lane.set_extras(d[k].get("extras", []), d[k]["scale"]); lane.set_visible(True)
            else:
                lane.set_object(None); lane.set_extras([]); lane.set_visible(False)
        T = min([d["T"]] + [len(d[k]["qpos"]) for k in lanes if k in d])
        state.update(t=0, data=d, T=T)
        state["programmatic"] = True
        try:
            frame.max = T - 1; frame.value = 0
        finally:
            state["programmatic"] = False
        have = ", ".join(f"{k} (scale {d[k]['scale']:.3f}, {d[k]['qpos'].shape[1]} qpos)" for k in lanes if k in d)
        info.content = f"**{name}**  object: `{d['src']['object']}`  frames: {T}  fps: {d['fps']:.0f}  smpl_scale: {d['s']:.3f}  \n{have}"

    def render(t: int):
        d = state["data"]
        t = min(t, state["T"] - 1)
        human.pose(t)
        for k, lane in lanes.items():
            if k in d and lane.visible:
                r = d[k]; lane._t = t
                lane.pose(r["qpos"][t], None if r["obj_pos"] is None else r["obj_pos"][t], None if r["obj_quat"] is None else r["obj_quat"][t],
                          r.get("joint_names"))

    @dd.on_update
    def _(_):
        with lock:
            try:
                load(dd.value); render(0)
            except Exception as ex:  # keep the server alive; report in the GUI
                info.content = f"**{dd.value}**: failed to load ({type(ex).__name__}: {ex})"
                print(f"[viser] load {dd.value} failed: {type(ex).__name__}: {ex}", flush=True)

    @frame.on_update
    def _(_):
        if state["programmatic"] or play.value:
            return
        with lock:
            state["t"] = int(frame.value); render(state["t"])

    @spacing.on_update
    def _(_):
        for i, lane in enumerate(lanes.values()):
            lane.frame.position = (0.0, (i + 1) * spacing.value, 0.0)

    show_h.on_update(lambda _: human.set_visible(show_h.value))

    @src_world.on_update
    def _(_):
        with lock:
            d = state["data"]
            human.set_world(d["s"] if src_world.value else 1.0, d["ground_z"] if src_world.value else 0.0)
            render(state["t"])
    for cb, lane in zip(lane_checkboxes, lanes.values()):
        cb.on_update(lambda _, cb=cb, lane=lane: lane.set_visible(cb.value))

    load(names[0]); render(0)
    print(f"viser ready: http://localhost:{args.port}   sequences: {len(names)}")
    clock = time.time(); carry = 0.0  # real-time pacing: advance by elapsed wall time, dropping frames if rendering is slow
    beat = time.time(); nrender = 0
    while True:
        if time.time() - beat > 10.0:  # heartbeat in the log: is the loop advancing, how many clients are connected
            print(f"[viser] heartbeat t={state['t']}/{state.get('T')} play={play.value} renders/10s={nrender} clients={len(server.get_clients())} seq={dd.value}", flush=True)
            beat = time.time(); nrender = 0
        with lock:
            d = state["data"]
            if play.value:
                nrender += 1
                now = time.time(); carry += (now - clock) * d["fps"] * speed.value; clock = now
                step = max(1, int(carry)); carry -= step
                state["t"] = (state["t"] + step) % state["T"]
                state["programmatic"] = True
                try:
                    frame.value = state["t"]
                finally:
                    state["programmatic"] = False
                try:
                    render(state["t"])
                except Exception as ex:  # never let a bad frame take the server down
                    print(f"[viser] render failed: {type(ex).__name__}: {ex}", flush=True)
            fps = d["fps"]
        if not play.value:
            clock = time.time(); carry = 0.0
        time.sleep(max(0.002, 1.0 / (fps * speed.value) - (time.time() - clock)))


if __name__ == "__main__":
    main()
