#!/usr/bin/env python3
"""Source-variant experiment: feed OmniRetarget the Mixamo joints of HUMOTO directly instead of the fitted SMPL-X joints.

    python prepare_humoto_mixamo.py --variant mx --subset 5        # every 5th sequence of data_humoto/seqs_test.json

Writes data_humoto/omni_<variant>/<seq>.pt (InterMimic (T,591) tensor, same object pose as data_humoto/omni/) and
data_humoto/seqs_test_<variant>.json whose `subject` is `humoto<variant>`; the subject's height is added to
data_humoto/omni/height_dict.pkl (holosoma's scale = 1.32 / height).
Variants (all map the 15 keypoints OmniRetarget uses; the other 37 slots get the nearest Mixamo joint):
  mx     Pelvis=Hips, Hip=UpLeg, Knee=Leg, Ankle=Foot, Toe=ToeBase, Shoulder=Arm, Elbow=ForeArm, Wrist=Hand; height = bind-pose mesh height (1.574 m)
  mxh    same joints, height = SMPL-X template height of the fit (1.548 m) -> isolates the joint-definition effect from the scale effect
  mxtoe  as mx but Toe = Toe_End (toe tip) instead of ToeBase
  mxpel  as mx but Pelvis = midpoint of the two UpLeg joints (SMPL-X-like pelvis, 2 cm lower / further back than Mixamo Hips)
  mxleg  as mxh but hips and knees taken from the fitted SMPL-X joints (hybrid: tests whether the 4 cm hip-definition gap drives the wrist error)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hand_bench"))
import humoto_io as H  # noqa: E402

OUT = HERE / "data_humoto"
MESH_HEIGHT = 1.574   # bind-pose y-extent of the HUMOTO body mesh (GLB `Mesh`)
SMPLX_HEIGHT = 1.548  # T-pose height of the fitted SMPL-X shape (data_humoto/omni/height_dict.pkl['humoto'])
SMPLH_DEMO_JOINTS = ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine", "Chest", "Neck", "Head",
                     "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist"] + [f"L_{f}{i}" for f in ("Index", "Middle", "Pinky", "Ring", "Thumb") for i in (1, 2, 3)] + \
                    ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist"] + [f"R_{f}{i}" for f in ("Index", "Middle", "Pinky", "Ring", "Thumb") for i in (1, 2, 3)]
assert len(SMPLH_DEMO_JOINTS) == 52
BODY = {"Pelvis": "Hips", "Torso": "Spine", "Spine": "Spine1", "Chest": "Spine2", "Neck": "Neck", "Head": "Head"}
for s, S in (("L", "Left"), ("R", "Right")):
    BODY.update({f"{s}_Hip": f"{S}UpLeg", f"{s}_Knee": f"{S}Leg", f"{s}_Ankle": f"{S}Foot", f"{s}_Toe": f"{S}ToeBase",
                 f"{s}_Thorax": f"{S}Shoulder", f"{s}_Shoulder": f"{S}Arm", f"{s}_Elbow": f"{S}ForeArm", f"{s}_Wrist": f"{S}Hand"})
KP = {"Thumb": 1, "Index": 5, "Middle": 9, "Ring": 13, "Pinky": 17}  # MediaPipe layout in humoto_io.hand_kp


SMPLX_LEG = {"L_Hip": 1, "R_Hip": 2, "L_Knee": 4, "R_Knee": 5}  # SMPL-X joint ids used by the mxleg hybrid


def joints52(seq: dict, variant: str, smplx_joints: np.ndarray | None = None) -> np.ndarray:
    bp = {n: seq["body_pos"][:, i] for i, n in enumerate(H.MIXAMO_BODY)}
    T = seq["n_frames"]
    out = np.zeros((T, 52, 3), dtype=np.float32)
    for k, name in enumerate(SMPLH_DEMO_JOINTS):
        if variant == "mxleg" and name in SMPLX_LEG:
            n = min(T, len(smplx_joints))
            out[:n, k] = smplx_joints[:n, SMPLX_LEG[name]]
        elif name in BODY:
            src = BODY[name]
            if variant == "mxtoe" and name.endswith("_Toe"):
                src = src.replace("ToeBase", "Toe_End")
            p = bp[src]
            if variant == "mxpel" and name == "Pelvis":
                p = 0.5 * (bp["LeftUpLeg"] + bp["RightUpLeg"])
            out[:, k] = p
        else:  # fingers: L_Index2 -> left hand kp Index + 1
            side = "left" if name.startswith("L_") else "right"
            finger, i = name[2:-1], int(name[-1])
            out[:, k] = seq["hand_kp"][side][:, KP[finger] + i - 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["mx", "mxh", "mxtoe", "mxpel", "mxleg"], required=True)
    ap.add_argument("--subset", type=int, default=5, help="take every k-th sequence of seqs_test.json (0 = all)")
    args = ap.parse_args()
    index = json.loads((OUT / "seqs_test.json").read_text())
    if args.subset:
        index = index[::args.subset]
    subject = f"humoto{args.variant}"
    odir = OUT / f"omni_{args.variant}"
    odir.mkdir(exist_ok=True)
    hp = OUT / "omni" / "height_dict.pkl"
    heights = pickle.load(open(hp, "rb"))
    heights[subject] = SMPLX_HEIGHT if args.variant in ("mxh", "mxleg") else MESH_HEIGHT
    pickle.dump(heights, open(hp, "wb"))
    new_index = []
    for k, e in enumerate(index):
        s = np.load(OUT / "source" / f"{e['seq']}.npz", allow_pickle=True)
        seq = H.load_sequence(e["seq"])
        J = joints52(seq, args.variant, s["joints55"])
        T = min(len(J), len(s["trans"]))
        data = np.zeros((T, 591), dtype=np.float32)
        data[:, 0:3] = J[:T, 0]
        data[:, 3:7] = [0, 0, 0, 1]
        data[:, 162:318] = J[:T].reshape(T, -1)
        data[:, 318:321] = s["obj_pos"][:T]
        data[:, 321:325] = s["obj_quat_xyzw"][:T]
        torch.save(torch.from_numpy(data), odir / f"{e['seq']}.pt")
        new_index.append({**e, "subject": subject, "n_frames": int(T)})
        print(f"[{k+1}/{len(index)}] {e['seq']}: T={T}")
    (OUT / f"seqs_test_{args.variant}.json").write_text(json.dumps(new_index, indent=1))
    print(f"{args.variant}: {len(new_index)} sequences, subject {subject} height {heights[subject]:.3f} -> {odir}")


if __name__ == "__main__":
    main()
