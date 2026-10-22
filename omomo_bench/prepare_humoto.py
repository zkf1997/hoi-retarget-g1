#!/usr/bin/env python3
"""Convert HUMOTO sequences (SMPL-X fitted by hiphi2smplx from the Mixamo rig, see hand_bench/humoto_to_bvh.py) into the
OmniRetarget / UMR input layouts under omomo_bench/data_humoto/ (same layout as prepare_omomo.py -> data/).

Run in the `umr` conda env:
    python prepare_humoto.py                       # all sequences in hand_bench/data/selection.json with a SMPL-X fit
    python prepare_humoto.py --seqs <name> ...

Inputs
  hand_bench/data/smplx/<seq>.npz        SMPL-X (neutral, 16 betas, 165-dim poses incl. fingers) in the glTF Y-up frame
  hand_bench/data/seqs/<seq>.npz         objects, poses (Z-up), per-frame closest hand-held object (hand_bench/prepare_humoto.py)
  hand_bench/data/objects/<obj>/         visual.obj (GLB mesh, node frame) + coacd_NN.obj
Primary object = the hand-held object with the most frames in which a hand is within 2 cm of it; every other object
(furniture) is ignored by both retargeters but kept in source/<seq>.npz (`extra_objects`) for the viewer.
Frame conversion: SMPL-X params are rotated to Z-up (x->x, y->z, z->-y) so that human and objects share HUMOTO's Z-up
world with the floor at z = 0.  Subject name: `humoto` (one Mixamo skeleton for the whole dataset -> one SMPL-X shape).
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_omomo import FPS, load_smplx, write_omni_pt, write_umr_seq  # noqa: E402

HERE = Path(__file__).resolve().parent
HB = HERE.parent / "hand_bench" / "data"
OUT = HERE / "data_humoto"
SUBJECT = "humoto"
YUP2ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def primary_object(d) -> str | None:
    objs = [str(o) for o in d["objects"]]
    best, best_n = None, 0
    for i, o in enumerate(objs):
        if not bool(d["handheld"][i]):
            continue
        n = 0
        for side in ("left", "right"):
            n += int(((d[f"near_obj_{side}"] == i) & (d[f"near_dist_{side}"] < 0.02)).sum())
        if n > best_n:
            best, best_n = o, n
    return best


def prepare_object(obj: str) -> dict:
    src = HB / "objects" / obj
    dst = OUT / "objects" / obj
    meta_p = dst / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())
    dst.mkdir(parents=True, exist_ok=True)
    import trimesh
    mesh = trimesh.load(src / "visual.obj", force="mesh")
    mesh.export(dst / f"{obj}.obj")
    pieces = sorted(src.glob("coacd_*.obj"))
    for i, p in enumerate(pieces):
        shutil.copy(p, dst / f"{obj}_coacd_{i:02d}.obj")
    meta = {"object": obj, "scale": 1.0, "center_raw": [0, 0, 0], "n_pieces": len(pieces),
            "extents_m": mesh.extents.tolist(), "source": "HUMOTO GLB node mesh (hand_bench/data/objects)"}
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"  [object] {obj}: extents={np.round(mesh.extents, 3)} pieces={len(pieces)}")
    return meta


def to_zup(poses: np.ndarray, trans: np.ndarray, betas: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rotate SMPL-X params from Y-up to Z-up; returns (poses, trans, joints55 [T,55,3] Z-up, tpose_height)."""
    T = len(trans)
    model = load_smplx("neutral", 1).to(device)
    b = torch.as_tensor(betas[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        out0 = model(betas=b)
        j0 = out0.joints[0, 0].cpu().numpy()
        v = out0.vertices[0].cpu().numpy()
    height = float(v[:, 1].max() - v[:, 1].min())
    Rc = Rot.from_matrix(YUP2ZUP)
    poses = poses.copy()
    poses[:, :3] = (Rc * Rot.from_rotvec(poses[:, :3])).as_rotvec()
    trans = (YUP2ZUP @ (trans + j0).T).T - j0
    modelT = load_smplx("neutral", T).to(device)
    with torch.no_grad():
        out = modelT(betas=b.repeat(T, 1),
                     global_orient=torch.as_tensor(poses[:, :3], dtype=torch.float32, device=device),
                     body_pose=torch.as_tensor(poses[:, 3:66], dtype=torch.float32, device=device),
                     left_hand_pose=torch.as_tensor(poses[:, 75:120], dtype=torch.float32, device=device),
                     right_hand_pose=torch.as_tensor(poses[:, 120:165], dtype=torch.float32, device=device),
                     transl=torch.as_tensor(trans, dtype=torch.float32, device=device))
        joints = out.joints[:, :55].cpu().numpy()
    return poses.astype(np.float32), trans.astype(np.float32), joints, height


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    seqs = args.seqs or json.loads((HB / "selection.json").read_text())["selected"]
    for d in ("objects", "source", "omni", "umr/test", "umr/object_assets"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    index, heights = [], {}
    for k, seq in enumerate(seqs):
        fit_p, hb_p = HB / "smplx" / f"{seq}.npz", HB / "seqs" / f"{seq}.npz"
        if not fit_p.exists():
            print(f"[{k+1}/{len(seqs)}] {seq}: no SMPL-X fit, skip"); continue
        d = np.load(hb_p, allow_pickle=True)
        obj = primary_object(d)
        if obj is None:
            print(f"[{k+1}/{len(seqs)}] {seq}: no hand-held object in contact, skip"); continue
        meta = prepare_object(obj)
        link = OUT / "umr" / "object_assets" / obj
        if not link.exists():
            link.symlink_to(Path("../../objects") / obj)
        objs = [str(o) for o in d["objects"]]
        oi = objs.index(obj)
        sc = d["obj_scale"][:, oi]
        assert np.allclose(sc, 1.0, atol=0.02), f"{seq}: object {obj} has node scale {sc.mean():.3f}"
        pos = d["obj_pos"][:, oi].astype(np.float64)
        quat = Rot.from_matrix(d["obj_rot"][:, oi].astype(np.float64)).as_quat()  # xyzw

        fit = np.load(fit_p)
        poses, trans, joints, height = to_zup(fit["poses"].astype(np.float64), fit["trans"].astype(np.float64),
                                              fit["betas"].astype(np.float64), args.device)
        T = min(len(trans), len(pos))
        poses, trans, joints, pos, quat = poses[:T], trans[:T], joints[:T], pos[:T], quat[:T]
        heights[SUBJECT] = height
        extra = [o for o in objs if o != obj]
        np.savez_compressed(OUT / "source" / f"{seq}.npz",
                            seq_name=seq, object=obj, subject=SUBJECT, fps=FPS, gender="neutral",
                            betas=fit["betas"].astype(np.float32), root_orient=poses[:, :3], pose_body=poses[:, 3:66],
                            pose_hand=poses[:, 75:165], trans=trans, joints55=joints.astype(np.float32),
                            obj_pos=pos.astype(np.float32), obj_quat_xyzw=quat.astype(np.float32),
                            obj_scale=1.0, obj_scale_raw=sc.astype(np.float32), tpose_height=height,
                            extra_objects=np.array(extra), extra_obj_pos=d["obj_pos"][:T, [objs.index(o) for o in extra]].astype(np.float32),
                            extra_obj_quat_xyzw=(Rot.from_matrix(d["obj_rot"][:T, [objs.index(o) for o in extra]].reshape(-1, 3, 3)).as_quat()
                                                 .reshape(T, len(extra), 4).astype(np.float32) if extra else np.zeros((T, 0, 4), np.float32)))
        write_omni_pt(OUT / "omni" / f"{seq}.pt", joints, poses[:, :3], pos, quat)
        s = {"trans": trans, "root_orient": poses[:, :3], "pose_body": poses[:, 3:66], "betas": fit["betas"], "gender": "neutral"}
        write_umr_seq(OUT / "umr" / "test" / seq, s, obj, meta, pos, quat)
        index.append({"seq": seq, "object": obj, "subject": SUBJECT, "n_frames": int(T)})
        print(f"[{k+1}/{len(seqs)}] {seq}: T={T} obj={obj} (+{len(extra)} other objects) height={height:.3f}")

    pickle.dump(heights, open(OUT / "omni" / "height_dict.pkl", "wb"))
    (OUT / "seqs_test.json").write_text(json.dumps(index, indent=1))
    print(f"done: {len(index)} sequences, {len({e['object'] for e in index})} objects -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
