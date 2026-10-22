#!/usr/bin/env python3
"""Compose Unitree G1 (29 DoF) URDFs with third-party dexterous hands.

Outputs (self-contained, relative mesh paths, loadable by MuJoCo / yourdfpy / Isaac Lab):
  urdf/g1_brainco_revo2/g1_29dof_brainco_revo2.urdf   G1 + BrainCo Revo2 (5 fingers, 6 motors/hand, 11 revolute joints/hand)
  urdf/g1_wuji_hand2/g1_29dof_wuji_hand2.urdf         G1 + Wuji Hand 2 beta2 (5 fingers, 20 DoF/hand)

Sources (see urdf/scripts/fetch_third_party.sh for pinned commits):
  G1 body        unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf   (same body as UMR / holosoma g1_29dof.urdf)
  Revo2 hand     BrainCoTech/revo2_description  urdf/revo2_{left,right}_hand.urdf
  Revo2 mount    unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf
                 (official Unitree assembly: adapter plate `*_base2_link` at wrist +0.0415 m, hand base at +0.0591 m,
                  rpy = (+pi/2, 0, +pi/2) right / (-pi/2, 0, -pi/2) left)
  Wuji Hand 2    wuji-technology/wuji-description hand2/hand2_beta2/body/urdf/{left,right}_with_mount.urdf
                 (with-mount variant: root link `{l,r}_mount` is the arm-flange mount, flange face at z=0, hand along -z)

G1 wrist convention (wrist_yaw_link frame, both arms, zero pose = forearm forward):
  +x distal (finger direction), +z thumb side (up), palm normal toward the body (+y right hand, -y left hand).
  Mounting flange face is the plane x = 0.0415 (Unitree attaches rubber hand / Dex3 / Inspire there).
"""
from __future__ import annotations

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
URDF_ROOT = HERE.parent
TP = URDF_ROOT / "third_party"

G1_DIR = TP / "unitree_ros" / "robots" / "g1_description"
G1_BRAINCO_DIR = TP / "unitree_ros" / "robots" / "g1_with_brainco_hand"
REVO2_DIR = TP / "revo2_description"
WUJI2_DIR = TP / "wuji-description" / "hand2" / "hand2_beta2" / "body"

PI_2 = math.pi / 2
G1_FLANGE_X = 0.0415  # wrist_yaw_link -> hand flange face


# ----------------------------------------------------------------------------- helpers
def fmt(v: float) -> str:
    return f"{v:.10g}"


def load(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def links(robot: ET.Element):
    return robot.findall("link")


def joints(robot: ET.Element):
    return robot.findall("joint")


def remove_g1_rubber_hands(g1: ET.Element) -> None:
    """Drop Unitree's default rubber hands so the wrist flange is free."""
    for j in list(joints(g1)):
        if j.get("name") in ("left_hand_palm_joint", "right_hand_palm_joint"):
            g1.remove(j)
    for l in list(links(g1)):
        if l.get("name") in ("left_rubber_hand", "right_rubber_hand"):
            g1.remove(l)


def copy_meshes(robot: ET.Element, src_dir: Path, out_meshes: Path, sub: str, strip_prefixes=("package://",)) -> None:
    """Rewrite every <mesh filename> to meshes/<sub>/<basename> and copy the file there."""
    dst_dir = out_meshes / sub
    dst_dir.mkdir(parents=True, exist_ok=True)
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename")
        name = Path(fn).name
        # resolve source file: try exact relative path first, then by basename search
        cand = None
        rel = fn
        for p in strip_prefixes:
            if rel.startswith(p):
                rel = rel[len(p):]
                rel = rel.split("/", 1)[1] if "/" in rel else rel  # drop ros package name
        for c in (src_dir / rel, src_dir / "meshes" / name):
            if c.is_file():
                cand = c
                break
        if cand is None:
            hits = list(src_dir.rglob(name))
            if not hits:
                raise FileNotFoundError(f"mesh {fn} not found under {src_dir}")
            cand = hits[0]
        dst = dst_dir / name
        if not dst.exists():
            shutil.copy2(cand, dst)
        mesh.set("filename", f"meshes/{sub}/{name}")


def fixed_joint(name: str, parent: str, child: str, xyz, rpy) -> ET.Element:
    j = ET.Element("joint", name=name, type="fixed")
    ET.SubElement(j, "origin", xyz=" ".join(map(fmt, xyz)), rpy=" ".join(map(fmt, rpy)))
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    return j


def append_hand(g1: ET.Element, hand: ET.Element, attach: ET.Element) -> None:
    """Append all links/joints of `hand` (a full <robot>) plus the attach joint into g1."""
    for el in list(hand):
        if el.tag in ("link", "joint", "material"):
            if el.tag == "material" and el.get("name") in {m.get("name") for m in g1.findall("material")}:
                continue
            g1.append(el)
    g1.append(attach)


def tidy_materials(robot: ET.Element) -> None:
    """Empty material names (BrainCo/Wuji export style) confuse some loaders -> give them names."""
    n = 0
    for mat in robot.iter("material"):
        if not mat.get("name"):
            n += 1
            mat.set("name", f"hand_mat_{n}")


def write(robot: ET.Element, out: Path) -> None:
    ET.indent(robot, space="  ")
    out.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(robot)
    tree.write(out, encoding="utf-8", xml_declaration=True)


def add_mujoco_hint(robot: ET.Element) -> None:
    """MuJoCo reads this block when loading a URDF; keeps visual meshes and fixes tiny inertias."""
    # Unitree's URDFs already carry <mujoco><compiler meshdir="meshes"/></mujoco>; override it.
    mj = robot.find("mujoco")
    if mj is not None:
        robot.remove(mj)
    mj = ET.Element("mujoco")
    # strippath: MuJoCo's URDF loader strips mesh directories by default, which breaks meshes/<sub>/ paths
    ET.SubElement(mj, "compiler", meshdir=".", strippath="false", balanceinertia="true",
                  discardvisual="false", fusestatic="false")
    robot.insert(0, mj)


def summary(robot: ET.Element) -> str:
    rev = [j for j in joints(robot) if j.get("type") in ("revolute", "continuous", "prismatic")]
    mass = sum(float(m.get("value")) for m in robot.iter("mass"))
    return f"links={len(links(robot))} joints={len(joints(robot))} actuated={len(rev)} mass={mass:.3f} kg"


# ----------------------------------------------------------------------------- G1 + BrainCo Revo2
def add_revo2_mimic(hand: ET.Element, side: str) -> None:
    """Revo2 distal phalanges are linkage-driven by the proximal joint (from Unitree's official file):
    thumb distal = 1.0 * thumb proximal ; finger distal = 1.155 * finger proximal."""
    ratio = {"thumb": 1.0, "index": 1.155, "middle": 1.155, "ring": 1.155, "pinky": 1.155}
    for j in joints(hand):
        n = j.get("name")
        for finger, k in ratio.items():
            if n == f"{side}_{finger}_distal_joint":
                ET.SubElement(j, "mimic", joint=f"{side}_{finger}_proximal_joint", multiplier=fmt(k), offset="0")


def build_g1_brainco(base_urdf: Path, out_dir: Path, name: str) -> Path:
    g1 = load(base_urdf)
    g1.set("name", name)
    remove_g1_rubber_hands(g1)
    out_meshes = out_dir / "meshes"
    copy_meshes(g1, G1_DIR, out_meshes, "g1")

    # adapter plates (Unitree part between G1 flange and Revo2 base)
    uni = load(G1_BRAINCO_DIR / "g1_29dof_mode_15_brainco_hand.urdf")
    for side in ("left", "right"):
        plate = next(l for l in links(uni) if l.get("name") == f"{side}_base2_link")
        plate_robot = ET.Element("robot")
        plate_robot.append(plate)
        copy_meshes(plate_robot, G1_BRAINCO_DIR, out_meshes, "brainco_adapter")
        g1.append(plate)
        g1.append(fixed_joint(f"{side}_hand_adapter_joint", f"{side}_wrist_yaw_link", f"{side}_base2_link",
                              (G1_FLANGE_X, 0, 0), (0, 0, 0)))

    # hands (BrainCo's own description, latest kinematics)
    mount = {
        "right": ((0.0591, 0, 0), (PI_2, 0, PI_2)),
        "left": ((0.0591, 0, 0), (-PI_2, 0, -PI_2)),
    }
    for side in ("left", "right"):
        hand = load(REVO2_DIR / "urdf" / f"revo2_{side}_hand.urdf")
        copy_meshes(hand, REVO2_DIR, out_meshes, f"revo2_{side}")
        add_revo2_mimic(hand, side)
        xyz, rpy = mount[side]
        append_hand(g1, hand, fixed_joint(f"{side}_hand_mount_joint", f"{side}_wrist_yaw_link", f"{side}_base_link", xyz, rpy))

    tidy_materials(g1)
    add_mujoco_hint(g1)
    out = out_dir / f"{name}.urdf"
    write(g1, out)
    print(f"[brainco] {out.relative_to(URDF_ROOT)}  {summary(g1)}")
    return out


# ----------------------------------------------------------------------------- G1 + Wuji Hand 2
def build_g1_wuji(base_urdf: Path, out_dir: Path, name: str) -> Path:
    g1 = load(base_urdf)
    g1.set("name", name)
    remove_g1_rubber_hands(g1)
    out_meshes = out_dir / "meshes"
    copy_meshes(g1, G1_DIR, out_meshes, "g1")

    # Wuji mount frame: flange face at z=0 (mesh z in [-0.0418, 0]), fingers along -z, palm +y (right) / -y (left),
    # index side +x. Map to G1 wrist: -z_w -> +x_g, y_w -> y_g, x_w -> +z_g  ==>  R = Ry(-pi/2), same for both hands.
    rpy = (0, -PI_2, 0)
    for side, p in (("left", "l"), ("right", "r")):
        hand = load(WUJI2_DIR / "urdf" / f"{side}_with_mount.urdf")
        copy_meshes(hand, WUJI2_DIR, out_meshes, f"wuji_hand2_{side}")
        append_hand(g1, hand, fixed_joint(f"{side}_hand_mount_joint", f"{side}_wrist_yaw_link", f"{p}_mount",
                                          (G1_FLANGE_X, 0, 0), rpy))

    tidy_materials(g1)
    add_mujoco_hint(g1)
    out = out_dir / f"{name}.urdf"
    write(g1, out)
    print(f"[wuji]    {out.relative_to(URDF_ROOT)}  {summary(g1)}")
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="g1_29dof_rev_1_0",
                    help="G1 body URDF stem in unitree_ros/robots/g1_description (e.g. g1_29dof_rev_1_0, g1_29dof_mode_15)")
    ap.add_argument("--only", choices=["brainco", "wuji"], default=None)
    args = ap.parse_args()

    base = G1_DIR / f"{args.base}.urdf"
    if not base.is_file():
        raise SystemExit(f"base URDF not found: {base}")
    suffix = "" if args.base == "g1_29dof_rev_1_0" else "_" + args.base.replace("g1_29dof_", "")

    if args.only in (None, "brainco"):
        build_g1_brainco(base, URDF_ROOT / "g1_brainco_revo2", f"g1_29dof_brainco_revo2{suffix}")
    if args.only in (None, "wuji"):
        build_g1_wuji(base, URDF_ROOT / "g1_wuji_hand2", f"g1_29dof_wuji_hand2{suffix}")


if __name__ == "__main__":
    main()
