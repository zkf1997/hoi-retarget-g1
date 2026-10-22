#!/usr/bin/env python3
"""Entry point for holosoma_retargeting's robot_retarget.py with an alternative preprocessing scaling scheme.

    python omni_entry.py [--scaling umr] [--fingers-free] [--subject NAME] -- <robot_retarget.py args>

Non-penetration constraint filtering (holosoma stores `activate_obj_non_penetration` but never reads it, so the
`--retargeter.no-activate-obj-non-penetration` flag is a no-op upstream; and a hand-equipped robot with fixed, curled
fingers cannot satisfy a hard finger/object constraint once the human wraps the hand around a thin handle - the pole
ends up between thumb and fingers and the linearised constraints become infeasible):
  --fingers-free                               drop finger-geom / object pairs, keep palm and body / object pairs
  --retargeter.no-activate-obj-non-penetration drop every robot / object pair (ground pairs are always kept)
Both are implemented by wrapping InteractionMeshRetargeter._update_jacobians_and_phis_from_q; run_omni.py uses them
as graded fallbacks and records which one was needed.

--scaling umr  reproduces UMR's scheme inside OmniRetarget (nothing in the submodule is modified; two functions of the
               robot_retarget module are replaced at import time):
  * preprocess_motion_data: no ground alignment; human joints x s; object xyz x s (native: z-align human to the lowest
    toe, object xy x s and object z anchored at its frame-0 height)
  * load_object_data: the object mesh is s-scaled on BOTH the demo and the robot side (native: robot side real size)
The robot MJCF with an s-scaled object mesh is selected by run_omni.py via --robot-config.robot-urdf-file.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import trimesh

RT = Path(__file__).resolve().parent.parent / "holosoma" / "src" / "holosoma_retargeting" / "holosoma_retargeting"


def load_robot_retarget():
    spec = importlib.util.spec_from_file_location("robot_retarget", RT / "examples" / "robot_retarget.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_umr_scaling(rr) -> None:
    from holosoma_retargeting.src.utils import extract_object_first_moving_frame

    def preprocess_motion_data(human_joints, retargeter, foot_names, scale=0.714, mat_height=0.1, object_poses=None):
        human_joints = human_joints * scale
        if object_poses is not None:
            object_poses[:, -3:] = object_poses[:, -3:] * scale
            return human_joints, object_poses, extract_object_first_moving_frame(object_poses)
        return human_joints

    def load_object_data(object_file, smpl_scale=0.714, bounding_box_oriented=False, sample_count=50, seed=42,
                         surface_weights=None, use_face_normals=False):
        obj_mesh = trimesh.load(object_file, force="mesh")
        points, _ = trimesh.sample.sample_surface_even(obj_mesh, sample_count, seed=seed)
        points = np.asarray(points) * smpl_scale
        return points, points.copy()

    rr.preprocess_motion_data = preprocess_motion_data
    rr.load_object_data = load_object_data


FINGER_KEYS = ("thumb", "index", "middle", "ring", "pinky")


def patch_nonpen_filter(fingers_free: bool) -> None:
    """Honour activate_obj_non_penetration and optionally exempt the finger geoms from the object constraint."""
    import mujoco
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter as R

    orig = R._update_jacobians_and_phis_from_q

    def finger_geoms(self) -> set[int]:
        m = self.robot_model
        out = set()
        for g in range(m.ngeom):
            if "_hand_" not in self._geom_names[g] or m.geom_dataid[g] < 0:
                continue
            mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])) or ""
            if any(k in mesh for k in FINGER_KEYS):
                out.add(g)
        return out

    def patched(self, q):
        Js, phis = orig(self, q)
        if not hasattr(self, "_finger_geoms"):
            self._finger_geoms = finger_geoms(self)
            print(f"[nonpen-filter] object pairs {'dropped' if not self.activate_obj_non_penetration else 'kept'}, "
                  f"fingers_free={fingers_free} ({len(self._finger_geoms)} finger geoms)")
        keep = {}
        for (g1, g2), phi in phis.items():
            n1, n2 = self._geom_names[g1], self._geom_names[g2]
            if self.object_name in n1 or self.object_name in n2:
                if not self.activate_obj_non_penetration:
                    continue
                if fingers_free and (g1 in self._finger_geoms or g2 in self._finger_geoms):
                    continue
            keep[(g1, g2)] = phi
        return {k: Js[k] for k in keep}, keep

    R._update_jacobians_and_phis_from_q = patched


def patch_subject(rr, subject: str) -> None:
    """holosoma derives the height_dict key from task_name.split('_')[0]; HUMOTO sequence names do not start with the
    subject, so fall back to the subject given by run_omni.py (--subject)."""
    import pickle

    def calculate_scale_factor(task_name, robot_height):
        with open("demo_data/height_dict.pkl", "rb") as f:
            height_dict = pickle.load(f)
        key = task_name.split("_")[0]
        human_height = height_dict.get(key, height_dict.get(subject))
        if human_height is None:
            raise KeyError(f"neither {key!r} nor subject {subject!r} in demo_data/height_dict.pkl")
        return robot_height / human_height

    rr.calculate_scale_factor = calculate_scale_factor


def main() -> None:
    argv = sys.argv[1:]
    scaling = "omni"
    if "--scaling" in argv:
        i = argv.index("--scaling"); scaling = argv[i + 1]; del argv[i:i + 2]
    subject = None
    if "--subject" in argv:
        i = argv.index("--subject"); subject = argv[i + 1]; del argv[i:i + 2]
    fingers_free = "--fingers-free" in argv
    if fingers_free:
        argv.remove("--fingers-free")
    if argv and argv[0] == "--":
        argv = argv[1:]
    rr = load_robot_retarget()
    if scaling == "umr":
        patch_umr_scaling(rr)
    elif scaling != "omni":
        raise SystemExit(f"unknown scaling {scaling}")
    patch_nonpen_filter(fingers_free)
    if subject:
        patch_subject(rr, subject)
    import tyro
    cfg = tyro.cli(rr.RetargetingConfig, args=argv)
    rr.main(cfg)


if __name__ == "__main__":
    main()
