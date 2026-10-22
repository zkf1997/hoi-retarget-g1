#!/usr/bin/env python3
"""Load a composed G1+hand URDF in MuJoCo and sanity-check the hand mounting.

Checks
  * the file loads (mesh paths resolve), joint/body counts, total mass
  * fingertip positions expressed in each wrist_yaw_link frame at q=0:
      distal (+x) should be ~0.15-0.25 m, thumb toward +z, palm/thumb toward the body (+y right, -y left)
  * self-collision contacts at q=0 and with all hand joints at mid-range (hand vs. arm/forearm)

Usage:  MUJOCO_GL=disable python validate_urdf.py path/to/robot.urdf [--json out.json]
Needs:  mujoco (e.g. conda env `isaac` or `gmr`)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def body_pos_in(model, data, body: str, ref: str):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ref)
    if bid < 0 or rid < 0:
        return None
    p = data.xpos[bid] - data.xpos[rid]
    R = data.xmat[rid].reshape(3, 3)
    return R.T @ p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.urdf))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    body_j = [n for n in jnames if not any(k in n for k in ("thumb", "index", "middle", "ring", "pinky", "finger"))]
    hand_j = [n for n in jnames if n not in body_j]
    total_mass = float(model.body_mass.sum())
    report = {
        "file": str(args.urdf),
        "nq": int(model.nq), "nv": int(model.nv), "nbody": int(model.nbody), "ngeom": int(model.ngeom),
        "body_joints": len(body_j), "hand_joints": len(hand_j), "total_mass_kg": round(total_mass, 3),
        "hand_joint_names": hand_j,
    }
    print(f"{args.urdf.name}: nq={model.nq} bodies={model.nbody} geoms={model.ngeom} "
          f"body joints={len(body_j)} hand joints={len(hand_j)} mass={total_mass:.3f} kg")

    # fingertip / tip bodies (Revo2: *_tip_link ; Wuji: {l,r}_*_tip)
    tips = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    tips = [t for t in tips if t and (t.endswith("_tip_link") or t.endswith("_tip"))]
    report["tips_in_wrist_frame"] = {}
    print("\nfingertips in wrist_yaw_link frame at q=0  (x distal, y toward body: + right / - left, z up=thumb side)")
    for side in ("left", "right"):
        ref = f"{side}_wrist_yaw_link"
        for t in tips:
            if t.startswith(side) or t.startswith(side[0] + "_"):
                v = body_pos_in(model, data, t, ref)
                report["tips_in_wrist_frame"][t] = [round(float(x), 4) for x in v]
                print(f"  {t:34s} {np.round(v, 4)}")

    # self-collision at q=0 and at mid-range hand pose
    def contacts(label):
        mujoco.mj_forward(model, data)
        pairs = set()
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
            if b1 != b2 and c.dist < -1e-4:
                pairs.add(tuple(sorted((b1, b2))))
        print(f"\npenetrating body pairs ({label}): {len(pairs)}")
        for p in sorted(pairs):
            print("  ", p)
        return sorted(pairs)

    report["contacts_q0"] = contacts("q=0")
    for n in hand_j:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        lo, hi = model.jnt_range[j]
        if model.jnt_limited[j]:
            data.qpos[model.jnt_qposadr[j]] = 0.5 * (lo + hi)
    report["contacts_hand_mid"] = contacts("hand joints at mid-range")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
