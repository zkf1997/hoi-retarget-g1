"""Shared I/O for the HUMOTO hand-retargeting benchmark (inputs from prepare_humoto.py, outputs of run_*.py)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import os

HERE = Path(__file__).resolve().parent
DATASET = os.environ.get("BENCH_DATASET", "humoto")  # humoto | hiphi  -> data[_hiphi]/, results[_hiphi]/
_SUF = "" if DATASET == "humoto" else f"_{DATASET}"
DATA = HERE / f"data{_SUF}"
SEQ_DIR = DATA / "seqs"
OBJ_DIR = DATA / "objects"
RESULTS = HERE / f"results{_SUF}"
HANDS = ("brainco", "wuji")
SIDES = ("left", "right")
METHODS = ("anyteleop", "anyteleop_dexpilot", "toporetarget")


def list_prepared() -> list[str]:
    return sorted(p.stem for p in SEQ_DIR.glob("*.npz"))


def load_seq(seq: str) -> dict:
    d = dict(np.load(SEQ_DIR / f"{seq}.npz", allow_pickle=True))
    d["objects"] = [str(o) for o in d["objects"]]
    d["n_frames"] = int(d["n_frames"])
    return d


def hand_frame_kp(d: dict, side: str) -> np.ndarray:
    """(T,21,3) human keypoints in the canonical wrist-centred hand frame."""
    kp, Rm = d[f"hand_kp_{side}"].astype(np.float64), d[f"hand_R_{side}"].astype(np.float64)
    return np.einsum("tji,tkj->tki", Rm, kp - kp[:, :1])


def main_object(d: dict, side: str) -> int:
    """index of the hand-held object this hand is closest to most often (-1 if never near one)."""
    near = d[f"near_obj_{side}"]
    near = near[near >= 0]
    if len(near) == 0:
        return -1
    return int(np.bincount(near).argmax())


def object_in_hand_frame(d: dict, side: str, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """object k pose expressed in the per-frame hand frame: pos (T,3), rot (T,3,3), scale (T,)"""
    kp, Rm = d[f"hand_kp_{side}"].astype(np.float64), d[f"hand_R_{side}"].astype(np.float64)
    p = np.einsum("tji,tj->ti", Rm, d["obj_pos"][:, k].astype(np.float64) - kp[:, 0])
    rot = np.einsum("tji,tjk->tik", Rm, d["obj_rot"][:, k].astype(np.float64))
    return p, rot, d["obj_scale"][:, k].astype(np.float64)


def load_object_visual(obj: str):
    import trimesh

    return trimesh.load(OBJ_DIR / obj / "visual.obj", force="mesh", process=False)


def load_object_pieces(obj: str) -> list[Path]:
    info = json.loads((OBJ_DIR / obj / "pieces.json").read_text())
    return [OBJ_DIR / obj / p for p in info["pieces"]]


def result_path(method: str, hand: str, seq: str, side: str) -> Path:
    return RESULTS / method / hand / f"{seq}__{side}.npz"


def save_result(method: str, hand: str, seq: str, side: str, *, q_active: np.ndarray, joint_names: list[str],
                base_pos_h: np.ndarray, base_quat_h: np.ndarray, runtime_per_frame: float, extra: dict | None = None) -> Path:
    """base_pos_h / base_quat_h (wxyz): pose of the robot *flange* (keypoint 0) in the human canonical hand frame.
    AnyTeleop keeps the flange on the human wrist (zeros / identity); TopoRetarget optimises it."""
    p = result_path(method, hand, seq, side)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, q_active=q_active.astype(np.float32), joint_names=np.array(joint_names),
                        base_pos_h=base_pos_h.astype(np.float32), base_quat_h=base_quat_h.astype(np.float32),
                        runtime_per_frame=np.float32(runtime_per_frame), **(extra or {}))
    return p


def load_result(method: str, hand: str, seq: str, side: str) -> dict | None:
    p = result_path(method, hand, seq, side)
    if not p.exists():
        return None
    d = dict(np.load(p, allow_pickle=True))
    d["joint_names"] = [str(n) for n in d["joint_names"]]
    return d


def flange_world(d: dict, side: str, base_pos_h: np.ndarray, base_quat_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """flange pose in world from the hand-frame pose: pos (T,3), rot (T,3,3)."""
    kp, Rm = d[f"hand_kp_{side}"].astype(np.float64), d[f"hand_R_{side}"].astype(np.float64)
    q = np.asarray(base_quat_h, dtype=np.float64)
    Rb = R.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    pos = kp[:, 0] + np.einsum("tij,tj->ti", Rm, np.asarray(base_pos_h, dtype=np.float64))
    return pos, Rm @ Rb
