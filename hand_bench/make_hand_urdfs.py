#!/usr/bin/env python3
"""Cut the hand sub-trees out of the two G1 URDFs and write stand-alone hand URDFs.

    python make_hand_urdfs.py

Writes urdf/g1_brainco_revo2/hand_{left,right}.urdf and urdf/g1_wuji_hand2/hand_{left,right}.urdf.
The root link is `{side}_wrist_yaw_link` (kept as a mass-less root), so the hand base frame is the G1 wrist frame:
+x distal, +z thumb side, palm facing +y (right hand) / -y (left hand); the flange (BrainCo adapter `*_base2_link`,
Wuji `*_mount`) sits at x = 0.0415. Mesh paths stay relative, so the files must live next to the G1 URDF.
<mimic> tags (BrainCo distal joints) are kept: dex-retargeting and our TopoRetarget wrapper both honour them, MuJoCo ignores them.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
URDFS = {
    "brainco": HERE.parent / "urdf" / "g1_brainco_revo2" / "g1_29dof_brainco_revo2.urdf",
    "wuji": HERE.parent / "urdf" / "g1_wuji_hand2" / "g1_29dof_wuji_hand2.urdf",
}


def hand_urdf_path(hand: str, side: str) -> Path:
    return URDFS[hand].parent / f"hand_{side}.urdf"


def extract(urdf: Path, side: str, out: Path) -> None:
    tree = ET.parse(urdf)
    root = tree.getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = root.findall("joint")
    children = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)
    base = f"{side}_wrist_yaw_link"
    keep_links, keep_joints, stack = [base], [], [base]
    while stack:
        l = stack.pop()
        for j in children.get(l, []):
            keep_joints.append(j)
            c = j.find("child").get("link")
            keep_links.append(c)
            stack.append(c)
    new = ET.Element("robot", name=f"{urdf.stem}_{side}_hand")
    for tag in ("mujoco",):
        for e in root.findall(tag):
            new.append(copy.deepcopy(e))
    root_link = ET.SubElement(new, "link", name=base)  # bare root (no inertia -> MuJoCo/pinocchio treat it as fixed world frame)
    for l in keep_links[1:]:
        new.append(copy.deepcopy(links[l]))
    for j in keep_joints:
        new.append(copy.deepcopy(j))
    ET.indent(new, space="  ")
    out.write_text(ET.tostring(new, encoding="unicode"))
    n_rev = sum(1 for j in keep_joints if j.get("type") == "revolute")
    n_mimic = sum(1 for j in keep_joints if j.find("mimic") is not None)
    print(f"{out.relative_to(HERE.parent)}: {len(keep_links)} links, {len(keep_joints)} joints ({n_rev} revolute, {n_mimic} mimic)")


def main() -> None:
    for hand, urdf in URDFS.items():
        for side in ("left", "right"):
            extract(urdf, side, hand_urdf_path(hand, side))


if __name__ == "__main__":
    main()
