#!/usr/bin/env python3
"""HiPHI HOI motions (SMPL-X from hiphi2smplx + object tracks) -> omomo_bench/data_hiphi/ (UMR input layout, same as
prepare_omomo.py / prepare_humoto.py).  Run in the `umr` conda env:

    python prepare_hiphi.py                 # the 60 motions of hand_bench/data_hiphi/selection.json
    python prepare_hiphi.py --seqs Bringing-lug_0311

Inputs
  ~/projects/HiPHI/smplx/<frame>/<lu>/<motion>.npz      SMPL-X stage-1 fit (neutral, 16 betas, 165 poses incl. fingers), 90 Hz, BVH Y-up frame
  hand_bench/data_hiphi/seqs/<motion>.npz             object pose (Z-up, m, 30 Hz = BVH strided by 3) + hand keypoints
  hand_bench/data_hiphi/objects/<mesh_id>/            visual.obj (metres, object frame) + coacd_NN.obj
Frames: SMPL-X strided by 3 (90 -> 30 Hz) and truncated to the hand-bench frame count, so body, hands and object are aligned.
Subject label `hiphi` (shape differs per motion; run UMR with --no-slot-reuse so the correspondence is trained per motion).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_omomo import FPS, write_omni_pt, write_umr_seq  # noqa: E402
from prepare_humoto import to_zup  # noqa: E402
import prepare_humoto as PH  # noqa: E402

HERE = Path(__file__).resolve().parent
HB = HERE.parent / "hand_bench" / "data_hiphi"
SMPLX_ROOT = Path("/home/kaizhao/projects/HiPHI/smplx")
OUT = HERE / "data_hiphi"
SUBJECT = "hiphi"
STRIDE = 3


def smplx_path(motion_id: str) -> Path:
    frame = motion_id.split("-")[0]
    lu = motion_id.rsplit("_", 1)[0].split("-", 1)[1]
    return SMPLX_ROOT / frame / lu / f"{motion_id}.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sel = json.loads((HB / "selection.json").read_text())
    seqs = args.seqs or sel["selected"]
    stats = sel.get("stats", {})
    PH.HB, PH.OUT = HB, OUT  # prepare_object copies from hand_bench/data_hiphi/objects into data_hiphi/objects
    for d in ("objects", "source", "omni", "umr/test", "umr/object_assets"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    index, heights = [], {}
    for k, seq in enumerate(seqs):
        fp = smplx_path(seq)
        if not fp.exists():
            print(f"[{k+1}/{len(seqs)}] {seq}: no SMPL-X fit at {fp}, skip"); continue
        d = np.load(HB / "seqs" / f"{seq}.npz", allow_pickle=True)
        objs = [str(o) for o in d["objects"]]
        obj = objs[0]
        meta = PH.prepare_object(obj)
        link = OUT / "umr" / "object_assets" / obj
        if not link.exists():
            link.symlink_to(Path("../../objects") / obj)
        fit = np.load(fp)
        poses, trans, joints, height = to_zup(fit["poses"][::STRIDE].astype(np.float64), fit["trans"][::STRIDE].astype(np.float64),
                                              fit["betas"].astype(np.float64), args.device)
        T = min(len(trans), int(d["n_frames"]))
        poses, trans, joints = poses[:T], trans[:T], joints[:T]
        pos = d["obj_pos"][:T, 0].astype(np.float64)
        quat = Rot.from_matrix(d["obj_rot"][:T, 0].astype(np.float64)).as_quat()
        heights[SUBJECT] = height  # informational only (differs per motion; UMR computes its own scale per sequence)
        np.savez_compressed(OUT / "source" / f"{seq}.npz",
                            seq_name=seq, object=obj, subject=SUBJECT, fps=FPS, gender="neutral",
                            betas=fit["betas"].astype(np.float32), root_orient=poses[:, :3], pose_body=poses[:, 3:66],
                            pose_hand=poses[:, 75:165], trans=trans, joints55=joints.astype(np.float32),
                            obj_pos=pos.astype(np.float32), obj_quat_xyzw=quat.astype(np.float32),
                            obj_scale=1.0, obj_scale_raw=d["obj_scale"][:T, 0].astype(np.float32), tpose_height=height,
                            extra_objects=np.array([]), extra_obj_pos=np.zeros((T, 0, 3), np.float32), extra_obj_quat_xyzw=np.zeros((T, 0, 4), np.float32),
                            category=str(stats.get(seq, {}).get("category", "")), text=str(stats.get(seq, {}).get("text", "")))
        write_omni_pt(OUT / "omni" / f"{seq}.pt", joints, poses[:, :3], pos, quat)
        s = {"trans": trans, "root_orient": poses[:, :3], "pose_body": poses[:, 3:66], "betas": fit["betas"], "gender": "neutral"}
        write_umr_seq(OUT / "umr" / "test" / seq, s, obj, meta, pos, quat)
        index.append({"seq": seq, "object": obj, "subject": SUBJECT, "n_frames": int(T), "category": stats.get(seq, {}).get("category", "")})
        print(f"[{k+1}/{len(seqs)}] {seq}: T={T} obj={obj} height={height:.3f}")
    pickle.dump(heights, open(OUT / "omni" / "height_dict.pkl", "wb"))
    (OUT / "seqs_test.json").write_text(json.dumps(index, indent=1))
    print(f"done: {len(index)} sequences, {len({e['object'] for e in index})} objects -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
