#!/usr/bin/env python3
"""Run OmniRetarget (holosoma_retargeting) on prepared OMOMO sequences (object_interaction task, G1).

Run in the `hsretargeting` conda env:
    python run_omni.py --setup                                   # build omni_workdir/ only
    python run_omni.py --hand brainco --workers 8                # G1 + BrainCo Revo2 (hands at rest), OmniRetarget-native scaling
    python run_omni.py --hand wuji --scaling umr --workers 8     # G1 + Wuji Hand 2, UMR-style scaling
    python run_omni.py --seqs sub16_largebox_000 --debug         # one sequence with viser

--hand    none | brainco | wuji   hand meshes are baked at hand_rest_pose.json into the wrist bodies of holosoma's own
                                  g1_29dof MJCF (OmniRetarget hard-codes 29 DoF), replacing the rubber hands; they act
                                  as collision geometry for the robot-object non-penetration constraint.
--scaling omni | umr              omni = native: human z-aligned to ground then x s, object mesh real size, object xy x s,
                                         object z anchored at its frame-0 height
                                  umr  = UMR-style: no ground alignment, human x s, object mesh AND xyz x s
                                         (implemented by monkey-patching preprocess_motion_data / load_object_data in
                                         omni_entry.py and per-subject robot MJCFs with scaled object meshes)
Set BENCH_DATASET=humoto to use data_humoto/, omni_workdir_humoto/, results_humoto/ (prepare_humoto.py).
Results: results/<tag>/<seq>_original.npz  (tag default: omni_<hand>[_umrscale]); qpos [T,43] = root pos, root quat wxyz,
29 joints, object pos + quat wxyz; extra keys hand / scaling / object_mesh_scale / fallback added by this runner.
All runs go through omni_entry.py (fallback chain default -> pen_tol_0.02 -> fingers_free -> fingers_free_pen_tol_0.02 -> no_obj_nonpen).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from hand_rest import HANDS

HERE = Path(__file__).resolve().parent
RT = HERE.parent / "holosoma" / "src" / "holosoma_retargeting" / "holosoma_retargeting"
# Dataset switch: BENCH_DATASET=omomo (default) | humoto  ->  data[_humoto]/, results[_humoto]/, omni_workdir[_humoto]/
DATASET = os.environ.get("BENCH_DATASET", "omomo")
SUFFIX = "" if DATASET == "omomo" else f"_{DATASET}"
DATA = HERE / f"data{SUFFIX}"
WORK = HERE / f"omni_workdir{SUFFIX}"
RESULTS = HERE / f"results{SUFFIX}"
PY = Path(os.environ.get("HS_PYTHON", "/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python"))
ROBOT_HEIGHT = 1.32


# ----------------------------------------------------------------------------- filesystem helpers
def symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def write_file(path: Path, text: str) -> None:
    """Write a real file; never write through a symlink into the holosoma submodule."""
    if path.is_symlink():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def heights() -> dict[str, float]:
    return pickle.load(open(DATA / "omni" / "height_dict.pkl", "rb"))


def robot_stem(hand: str, scaling: str, subject: str | None = None) -> str:
    s = "g1_29dof" + ("" if hand == "none" else f"_{hand}")
    if scaling == "umr":
        s += f"_umrscale_{subject}"
    return s


# ----------------------------------------------------------------------------- model generation
def object_urdf(obj: str) -> str:
    return f"""<?xml version="1.0" ?>
<robot name="{obj}">
  <link name="{obj}_link">
    <inertial><mass value="1.0"/><origin xyz="0 0 0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><origin rpy="0 0 0" xyz="0 0 0"/><geometry><mesh filename="{obj}.obj" scale="1.0 1.0 1.0"/></geometry>
      <material name="mat"><color rgba="0.7 0.8 0.9 0.7"/></material></visual>
    <collision name="{obj}"><origin rpy="0 0 0" xyz="0 0 0"/><geometry><mesh filename="{obj}.obj" scale="1.0 1.0 1.0"/></geometry></collision>
  </link>
</robot>
"""


BAKE_MARK = "<!-- hand bake v2: mesh refpos/refquat compensated (2026-10-14); models without this line have a mangled hand -->"


def bake_hands(template: str, hand: str) -> str:
    """Attach the hand collision meshes (posed at hand_rest_pose.json) as fixed geoms of {left,right}_wrist_yaw_link
    and drop the rubber-hand geoms. Poses come from MuJoCo FK of the composed URDF."""
    import mujoco
    from scipy.spatial.transform import Rotation as Rot
    from hand_rest import hand_rest_pose

    urdf = HANDS[hand]
    m = mujoco.MjModel.from_xml_path(str(urdf))
    d = mujoco.MjData(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    limits = {n: tuple(m.jnt_range[j]) for j, n in enumerate(names) if m.jnt_limited[j]}
    for j, v in hand_rest_pose(names, limits).items():
        d.qpos[m.jnt_qposadr[names.index(j)]] = v
    mujoco.mj_forward(m, d)

    def subtree(root: int) -> set[int]:
        out = set()
        for b in range(m.nbody):
            x = b
            while x > 0 and x != root:
                x = m.body_parentid[x]
            if x == root and b != root:
                out.add(b)
        return out

    mesh_dir = urdf.parent
    assets, geoms = [], {"left": [], "right": []}
    for side in ("left", "right"):
        wb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        bodies = subtree(wb)
        Rw, pw = d.xmat[wb].reshape(3, 3), d.xpos[wb]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] not in bodies or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g])
            files = [p for p in mesh_dir.rglob(f"{mesh_name}.*") if p.suffix.lower() in (".stl", ".obj")]
            if not files:
                raise FileNotFoundError(f"mesh file for {mesh_name} under {mesh_dir}")
            asset = f"hand_{hand}_{mesh_name}"
            line = f'    <mesh name="{asset}" file="{files[0].resolve().as_posix()}"/>'
            if line not in assets:
                assets.append(line)
            # geom_xpos/xmat are the COMPILED geom frame, which already includes MuJoCo's mesh re-centring
            # (mesh_pos/mesh_quat: the raw STL is moved to its inertial frame at load time). The XML geom frame that
            # reloads to the same compiled frame is compiled o ref^-1, otherwise the re-centring is applied twice and the
            # hand pieces land 1-4 cm off and rotated (bug present in every OmniRetarget lane before 2026-10-14).
            mid = m.geom_dataid[g]
            R_ref = Rot.from_quat(m.mesh_quat[mid][[1, 2, 3, 0]]).as_matrix()
            R_rel = Rw.T @ d.geom_xmat[g].reshape(3, 3)
            R_xml = R_rel @ R_ref.T
            p_rel = Rw.T @ (d.geom_xpos[g] - pw) - R_xml @ m.mesh_pos[mid]
            q = Rot.from_matrix(R_xml).as_quat()  # xyzw
            geoms[side].append(
                f'                            <geom name="{side}_hand_{len(geoms[side]):02d}" type="mesh" mesh="{asset}" '
                f'pos="{p_rel[0]:.6f} {p_rel[1]:.6f} {p_rel[2]:.6f}" quat="{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}" '
                f'rgba="0.85 0.72 0.6 1"/>')
    out = template.replace("  </asset>", "\n".join(assets) + "\n  </asset>", 1)
    for side in ("left", "right"):
        anchor = f'<geom name="{side}_wrist_yaw_link" type="mesh" rgba="0.7 0.7 0.7 1" mesh="{side}_wrist_yaw_link"/>'
        assert anchor in out, f"anchor geom for {side} wrist not found in template"
        out = out.replace(anchor, anchor + "\n" + "\n".join(geoms[side]), 1)
        # remove the two rubber-hand geoms but keep the (massless-ish) body: it is the wrist keypoint target
        body_re = re.compile(rf'(<body name="{side}_rubber_hand_link"[^>]*>)(.*?)(</body>)', re.S)
        mo = body_re.search(out)
        assert mo, f"{side}_rubber_hand_link body not found"
        inner = re.sub(r"\s*<geom[^>]*mesh=\"%s_rubber_hand_link\"[^>]*/>" % side, "", mo.group(2), flags=re.S)
        inner = re.sub(r"\s*<geom[^>]*mesh=\"%s_rubber_hand_link\"[^>]*>.*?/>" % side, "", inner, flags=re.S)
        out = out[: mo.start()] + mo.group(1) + inner + mo.group(3) + out[mo.end():]
    print(f"[bake] {hand}: {len(assets)} hand meshes, {len(geoms['left'])}+{len(geoms['right'])} collision geoms")
    return out.replace("</mujoco>", BAKE_MARK + "\n</mujoco>", 1)


NATIVE = [False]  # --native-collision: holosoma's own convention, the whole visual mesh as ONE collision geom (MuJoCo uses its convex hull)


def robot_object_xml(template: str, obj: str, n_pieces: int, mesh_scale: float = 1.0) -> str:
    """Swap the largebox asset/body of g1_29dof_w_largebox.xml for the OMOMO object (CoACD collision pieces)."""
    sc = f'scale="{mesh_scale:.6f} {mesh_scale:.6f} {mesh_scale:.6f}"'
    assets = [f'    <mesh name="{obj}_visual" file="../../{obj}/{obj}.obj" {sc}/>']
    geoms = [f'        <geom name="{obj}_visual_geom" type="mesh" mesh="{obj}_visual" contype="0" conaffinity="0" group="1" rgba="0.7 0.8 0.9 0.5"/>']
    if NATIVE[0]:
        geoms.append(f'        <geom name="{obj}_hull" type="mesh" mesh="{obj}_visual" contype="1" conaffinity="1" '
                     f'rgba="0.7 0.8 0.9 0.3" friction="0.9 0.5 0.5" solref="0.02 1" solimp="0.9 0.95 0.001"/>')
        n_pieces = 0
    for i in range(n_pieces):
        assets.append(f'    <mesh name="{obj}_c{i:02d}" file="../../{obj}/{obj}_coacd_{i:02d}.obj" {sc}/>')
        geoms.append(f'        <geom name="{obj}_c{i:02d}" type="mesh" mesh="{obj}_c{i:02d}" contype="1" conaffinity="1" '
                     f'rgba="0.7 0.8 0.9 0.3" friction="0.9 0.5 0.5" solref="0.02 1" solimp="0.9 0.95 0.001"/>')
    body = (f'    <body name="{obj}_link">\n        <freejoint/>\n'
            f'        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>\n' + "\n".join(geoms) + "\n    </body>")
    out, n1 = re.subn(r'\s*<mesh name="largebox_mesh"[^>]*/>', "\n" + "\n".join(assets), template, count=1)
    out, n2 = re.subn(r'<body name="largebox_link">.*?</body>', body, out, count=1, flags=re.S)
    if n1 != 1 or n2 != 1:
        raise RuntimeError("could not patch g1_29dof_w_largebox.xml (asset %d, body %d)" % (n1, n2))
    return out


def setup_workdir(hand: str, scaling: str) -> None:
    WORK.mkdir(exist_ok=True)
    for sub in (RT / "models").iterdir():          # mirror RT/models with symlinks (never touch the submodule)
        dst = WORK / "models" / sub.name
        if sub.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for f in sub.iterdir():
                symlink(f, dst / f.name)
        else:
            symlink(sub, dst)
    symlink(DATA / "omni" / "height_dict.pkl", WORK / "demo_data" / "height_dict.pkl")

    template = (RT / "models" / "g1" / "g1_29dof_w_largebox.xml").read_text()
    if hand != "none":
        cache = WORK / "models" / "g1" / f"g1_29dof_{hand}_w_largebox.template.xml"
        if not cache.exists() or BAKE_MARK not in cache.read_text():  # regenerate models baked before the refpos fix
            write_file(cache, bake_hands(template, hand))
        template = cache.read_text()
    variants = [(robot_stem(hand, scaling), 1.0)] if scaling == "omni" else \
               [(robot_stem(hand, scaling, sub), ROBOT_HEIGHT / h) for sub, h in heights().items()]
    if NATIVE[0]:
        variants = [(stem + "_native", sc) for stem, sc in variants]
    for stem, _ in variants:
        symlink(WORK / "models" / "g1" / "g1_29dof.urdf", WORK / "models" / "g1" / f"{stem}.urdf")  # viser only

    metas = [json.loads(p.read_text()) for p in sorted((DATA / "objects").glob("*/meta.json"))]
    for meta in metas:
        obj = meta["object"]
        odir = WORK / "models" / obj
        odir.mkdir(exist_ok=True)
        symlink(meta["_path"] if "_path" in meta else DATA / "objects" / obj / f"{obj}.obj", odir / f"{obj}.obj")
        for i in range(meta["n_pieces"]):
            symlink(DATA / "objects" / obj / f"{obj}_coacd_{i:02d}.obj", odir / f"{obj}_coacd_{i:02d}.obj")
        write_file(odir / f"{obj}.urdf", object_urdf(obj))
        write_file(WORK / "models" / "templates" / f"{obj}.urdf.jinja", object_urdf(obj))
        for stem, mesh_scale in variants:
            xml = WORK / "models" / "g1" / f"{stem}_w_{obj}.xml"
            if not xml.exists() or (hand != "none" and BAKE_MARK not in xml.read_text()):
                write_file(xml, robot_object_xml(template, obj, meta["n_pieces"], mesh_scale))
    print(f"[setup] hand={hand} scaling={scaling}: {len(metas)} objects x {len(variants)} robot variants ready in {WORK}")


# ----------------------------------------------------------------------------- running
def annotate(npz: Path, **extra) -> None:
    d = dict(np.load(npz, allow_pickle=True))
    d.update({k: np.asarray(v) for k, v in extra.items()})
    np.savez(npz, **d)


def diverged(npz: Path, max_dist: float = 2.0) -> bool:
    d = np.load(npz, allow_pickle=True)
    q, hj = d["qpos"], d["human_joints"]
    T = min(len(q), len(hj))
    return bool(np.linalg.norm(q[:T, :3] - hj[:T, 0], axis=1).max() > max_dist) or not np.isfinite(q).all()


def run_one(seq: str, obj: str, hand: str, scaling: str, results: Path, debug: bool = False, extra: list[str] | None = None,
            subject: str | None = None, omni_dir: str = "omni") -> dict:
    results.mkdir(parents=True, exist_ok=True)
    logs = results / "logs"; logs.mkdir(exist_ok=True)
    out_npz = results / f"{seq}_original.npz"
    sub = subject or seq.split("_")[0]
    stem = robot_stem(hand, scaling, sub) + ("_native" if NATIVE[0] else "")
    cmd = [str(PY), str(HERE / "omni_entry.py"), "--scaling", scaling, "--subject", sub, "--",
           "--data_path", str(DATA / omni_dir), "--task-type", "object_interaction", "--task-name", seq,
           "--data_format", "smplh", "--task-config.object-name", obj, "--save_dir", str(results),
           "--robot-config.robot-urdf-file", f"models/g1/{stem}.urdf"]
    if debug:
        cmd += ["--retargeter.debug", "--retargeter.visualize"]
    cmd += extra or []
    # The SQP reports `infeasible` when hard constraints conflict: deep initial robot-object penetration vs. joint
    # limits, or (with hands) a thin handle caught between the fixed thumb and fingers. Graded retries, recorded as
    # `fallback` in the npz / run_summary.json: looser penetration tolerance -> finger geoms exempt from the object
    # constraint (palm + body still constrained) -> no robot-object constraint at all (see omni_entry.py).
    tol = ["--retargeter.penetration-tolerance", "0.02"]
    fallbacks = [("default", []), ("pen_tol_0.02", tol)]
    if hand != "none":
        fallbacks += [("fingers_free", ["--fingers-free"]), ("fingers_free_pen_tol_0.02", ["--fingers-free", *tol])]
    fallbacks.append(("no_obj_nonpen", ["--retargeter.no-activate-obj-non-penetration"]))
    t0 = time.time()
    used = None
    rc = -1
    with open(logs / f"{seq}.log", "w") as lf:
        for name, fb in fallbacks:
            lf.write(f"### attempt: {name}\n" + " ".join(cmd + fb) + "\n\n"); lf.flush()
            rc = subprocess.call(cmd + fb, cwd=WORK, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"})
            if rc == 0 and out_npz.exists():
                if diverged(out_npz):  # the first-frame solve has no step bound and occasionally blows up (root 100s of m away)
                    lf.write(f"### attempt {name}: solve diverged (robot root far from human pelvis), trying next fallback\n")
                    out_npz.unlink()
                    continue
                used = name
                break
    dt = time.time() - t0
    ok = used is not None
    if ok:
        s = ROBOT_HEIGHT / heights()[sub]
        annotate(out_npz, hand=hand, scaling=scaling, object_mesh_scale=(s if scaling == "umr" else 1.0), smpl_scale=s, fallback=used, subject=sub)
    return {"seq": seq, "object": obj, "ok": ok, "rc": rc, "seconds": round(dt, 1), "fallback": used,
            "out": str(out_npz) if ok else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", action="store_true", help="(re)build omni_workdir and exit")
    ap.add_argument("--hand", choices=list(HANDS), default="none")
    ap.add_argument("--scaling", choices=["omni", "umr"], default="omni")
    ap.add_argument("--tag", default=None, help="results/<tag>; default omni_<hand>[_umrscale]")
    ap.add_argument("--index", type=Path, default=DATA / "seqs_test.json")
    ap.add_argument("--omni-dir", default="omni", help="sub-directory of data*/ holding the InterMimic .pt files (source-variant experiments)")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--debug", action="store_true", help="viser visualization + debug (single sequence)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--native-collision", action="store_true", help="ablation: object collision = the whole visual mesh as one geom (holosoma default; MuJoCo takes its convex hull) instead of CoACD pieces")
    ap.add_argument("extra", nargs="*", help="extra args passed to robot_retarget.py (after --)")
    args = ap.parse_args()

    NATIVE[0] = args.native_collision
    setup_workdir(args.hand, args.scaling)
    if args.setup:
        return
    tag = args.tag or f"omni_{args.hand}" + ("_umrscale" if args.scaling == "umr" else "")
    results = RESULTS / tag
    index = json.loads(args.index.read_text())
    if args.seqs:
        index = [e for e in index if e["seq"] in set(args.seqs)]
    if not args.force:
        index = [e for e in index if not (results / f"{e['seq']}_original.npz").exists()]
    print(f"[{tag}] running {len(index)} sequences with {args.workers} workers")
    summary = []
    if args.debug or args.workers <= 1:
        for e in index:
            r = run_one(e["seq"], e["object"], args.hand, args.scaling, results, args.debug, args.extra, e.get("subject"), args.omni_dir); summary.append(r); print(r, flush=True)
    else:
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(run_one, e["seq"], e["object"], args.hand, args.scaling, results, False, args.extra, e.get("subject"), args.omni_dir): e for e in index}
            for f in as_completed(futs):
                r = f.result(); summary.append(r); print(r, flush=True)
    results.mkdir(parents=True, exist_ok=True)
    sp = results / "run_summary.json"
    prev = {r["seq"]: r for r in json.loads(sp.read_text())} if sp.exists() else {}
    prev.update({r["seq"]: r for r in summary})
    sp.write_text(json.dumps(list(prev.values()), indent=1))
    print(f"done: {sum(r['ok'] for r in summary)}/{len(summary)} ok")


if __name__ == "__main__":
    sys.exit(main())
