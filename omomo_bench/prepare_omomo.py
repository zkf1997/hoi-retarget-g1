#!/usr/bin/env python3
"""Convert OMOMO (SMPL-X + object) sequences into the input layouts of OmniRetarget and UMR.

Run in the `umr` conda env (needs torch, smplx, trimesh, coacd, joblib):
    python prepare_omomo.py --split test --limit 20            # pilot
    python prepare_omomo.py --split test                       # whole test split (602 seqs)

Writes under omomo_bench/data/:
  objects/<obj>/<obj>.obj                 centred visual mesh in metres:  s * (V - mean(V)),  s = canonical OMOMO obj_scale
  objects/<obj>/<obj>_coacd_NN.obj        CoACD convex pieces (collision)
  objects/<obj>/meta.json                 scale, n pieces
  source/<seq>.npz                        SMPL-X params + 55 world joints + object pose (for metrics / viser)
  omni/<seq>.pt                           InterMimic-style (T,591) tensor for holosoma_retargeting --data_format smplh
  omni/height_dict.pkl                    {subN: T-pose height}  (holosoma reads demo_data/height_dict.pkl from cwd)
  umr/test/<seq>/{poses,transl,betas,gender,model_type,mocap_framerate,output_up}.npy + <obj>.xml + prop_<obj>.csv
  umr/object_assets/<obj>/ -> ../../objects/<obj>   (symlink so the sequence XMLs can use a fixed relative meshdir)
  seqs_<split>.json                       list of prepared sequences with object / subject / n_frames

Conventions
  * OMOMO object pose: world verts = s * R @ V + t ; obj_com_pos = t + s R mean(V)  (verified numerically)
    -> centred mesh + pose (obj_com_pos, R) reproduces the same world surface.
  * InterMimic .pt columns actually read by holosoma: [162:318] 52 joints (SMPLH_DEMO_JOINTS order), [318:325] obj xyz + quat xyzw.
    We also fill [0:3] pelvis, [3:7] root quat xyzw; everything else is zero.
  * Hands: SMPL-X hand pose = 0 with flat_hand_mean=True (flat open hands), matching UMR's zero_source_finger_pose.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
OMOMO_DATA = Path("/home/kaizhao/projects/omomo_release/data")
SMPLX_DIR = Path("/media/kaizhao/hdd/dataset/models_smplx_v1_1/models")  # contains smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz
FPS = 30

# SMPL-X joint indices (0-54) -> InterMimic 52-joint order used by holosoma (SMPLH_DEMO_JOINTS)
SMPLX_TO_INTERMIMIC = (
    [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15]          # pelvis, L leg(+toe), R leg(+toe), torso/spine/chest, neck, head
    + [13, 16, 18, 20] + list(range(25, 40))                 # L collar, shoulder, elbow, wrist, 15 left fingers (index,middle,pinky,ring,thumb)
    + [14, 17, 19, 21] + list(range(40, 55))                 # right side
)
assert len(SMPLX_TO_INTERMIMIC) == 52


# ----------------------------------------------------------------------------- SMPL-X
def load_smplx(gender: str, batch_size: int):
    import smplx
    return smplx.create(str(SMPLX_DIR), model_type="smplx", gender=gender, num_betas=16, use_pca=False,
                        flat_hand_mean=True, batch_size=batch_size, ext="npz")


@torch.no_grad()
def smplx_fk(seq: dict, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (joints55 [T,55,3] world, tpose_height)."""
    T = len(seq["trans"])
    gender = str(seq["gender"])
    model = load_smplx(gender, T).to(device)
    betas = torch.as_tensor(np.repeat(seq["betas"], T, 0), dtype=torch.float32, device=device)
    out = model(betas=betas,
                global_orient=torch.as_tensor(seq["root_orient"], dtype=torch.float32, device=device),
                body_pose=torch.as_tensor(seq["pose_body"], dtype=torch.float32, device=device),
                transl=torch.as_tensor(seq["trans"], dtype=torch.float32, device=device))
    joints = out.joints[:, :55].cpu().numpy()
    # T-pose height: y-extent of the zero-pose template (SMPL-X canonical frame is y-up), same definition as UMR
    m1 = load_smplx(gender, 1).to(device)
    v = m1(betas=betas[:1]).vertices[0].cpu().numpy()
    height = float(v[:, 1].max() - v[:, 1].min())
    return joints, height


# ----------------------------------------------------------------------------- objects
def canonical_scales(seqs: list[dict]) -> dict[str, float]:
    per_obj: dict[str, list[float]] = {}
    for s in seqs:
        sc = s["obj_scale"]
        sc = sc[sc > 1e-4]
        if len(sc):
            per_obj.setdefault(obj_of(s), []).append(float(np.median(sc)))
    return {k: float(np.median(v)) for k, v in per_obj.items()}


def obj_of(seq: dict) -> str:
    return seq["seq_name"].split("_")[1]


def prepare_object(obj: str, scale: float, out_dir: Path, coacd_threshold: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_p = out_dir / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text())
    mesh = trimesh.load(OMOMO_DATA / "captured_objects" / f"{obj}_cleaned_simplified.obj", force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    center = V.mean(0)
    Vc = scale * (V - center)
    vis = trimesh.Trimesh(Vc, mesh.faces, process=False)
    vis.export(out_dir / f"{obj}.obj")

    import coacd
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(Vc.astype(np.float64), np.asarray(mesh.faces, dtype=np.int64)),
                            threshold=coacd_threshold, max_convex_hull=32, mcts_iterations=150, resolution=2000)
    for i, (pv, pf) in enumerate(parts):
        trimesh.Trimesh(pv, pf, process=False).export(out_dir / f"{obj}_coacd_{i:02d}.obj")
    meta = {"object": obj, "scale": scale, "center_raw": center.tolist(), "n_pieces": len(parts),
            "extents_m": vis.extents.tolist(), "coacd_threshold": coacd_threshold}
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"  [object] {obj}: scale={scale:.4f} extents={np.round(vis.extents, 3)} coacd pieces={len(parts)}")
    return meta


def object_pose(seq: dict) -> tuple[np.ndarray, np.ndarray]:
    """(pos [T,3] = obj_com_pos, quat_xyzw [T,4])"""
    pos = seq["obj_com_pos"].astype(np.float64)
    quat = Rot.from_matrix(seq["obj_rot"].astype(np.float64)).as_quat()  # xyzw
    return pos, quat


# ----------------------------------------------------------------------------- writers
def write_omni_pt(path: Path, joints55: np.ndarray, root_orient: np.ndarray, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
    T = len(joints55)
    data = np.zeros((T, 591), dtype=np.float32)
    data[:, 0:3] = joints55[:, 0]
    data[:, 3:7] = Rot.from_rotvec(root_orient).as_quat()
    data[:, 162:318] = joints55[:, SMPLX_TO_INTERMIMIC].reshape(T, -1)
    data[:, 318:321] = pos
    data[:, 321:325] = quat_xyzw
    torch.save(torch.from_numpy(data), path)


def object_mjcf(obj: str, meta: dict, meshdir: str) -> str:
    assets = [f'    <mesh name="{obj}_visual" file="{obj}.obj"/>']
    geoms = [f'      <geom name="{obj}_visual_geom" type="mesh" mesh="{obj}_visual" material="obj_visual" contype="0" conaffinity="0" group="2"/>']
    for i in range(meta["n_pieces"]):
        assets.append(f'    <mesh name="{obj}_c{i:02d}" file="{obj}_coacd_{i:02d}.obj"/>')
        geoms.append(f'      <geom name="{obj}_c{i:02d}" type="mesh" mesh="{obj}_c{i:02d}" contype="1" conaffinity="1" group="3" '
                     f'rgba="0.7 0.8 0.9 0.4" friction="0.9 0.5 0.5"/>')
    nl = "\n"
    return f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="{obj}">
  <compiler angle="radian" autolimits="true" meshdir="{meshdir}"/>
  <asset>
    <material name="obj_visual" rgba="0.75 0.72 0.66 1"/>
{nl.join(assets)}
  </asset>
  <worldbody>
    <body name="{obj}_link" pos="0 0 0">
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
{nl.join(geoms)}
    </body>
  </worldbody>
</mujoco>
"""


def write_umr_seq(seq_dir: Path, seq: dict, obj: str, meta: dict, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    T = len(seq["trans"])
    poses = np.zeros((T, 165), dtype=np.float32)
    poses[:, 0:3] = seq["root_orient"]
    poses[:, 3:66] = seq["pose_body"]
    np.save(seq_dir / "poses.npy", poses)
    np.save(seq_dir / "transl.npy", seq["trans"].astype(np.float32))
    np.save(seq_dir / "betas.npy", seq["betas"].reshape(-1).astype(np.float32))
    np.save(seq_dir / "gender.npy", np.array(str(seq["gender"])))
    np.save(seq_dir / "model_type.npy", np.array("smplx"))
    np.save(seq_dir / "mocap_framerate.npy", np.array(float(FPS)))
    np.save(seq_dir / "output_up.npy", np.array("z"))
    (seq_dir / f"{obj}.xml").write_text(object_mjcf(obj, meta, meshdir=f"../../object_assets/{obj}"))
    with open(seq_dir / f"prop_{obj}.csv", "w") as f:
        f.write("px,py,pz,qx,qy,qz,qw\n")
        for p, q in zip(pos, quat_xyzw):
            f.write(",".join(f"{x:.6f}" for x in (*p, *q)) + "\n")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--limit", type=int, default=0, help="max sequences (0 = all)")
    ap.add_argument("--per-object", type=int, default=0, help="take at most K sequences per object (0 = all)")
    ap.add_argument("--seqs", nargs="*", default=None, help="explicit seq names")
    ap.add_argument("--coacd-threshold", type=float, default=0.03)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    raw = joblib.load(OMOMO_DATA / f"{args.split}_diffusion_manip_seq_joints24.p")
    all_seqs = sorted(raw.values(), key=lambda s: s["seq_name"])
    scales = canonical_scales(all_seqs)

    seqs = all_seqs
    if args.seqs:
        want = set(args.seqs)
        seqs = [s for s in seqs if s["seq_name"] in want]
    if args.per_object:
        cnt: dict[str, int] = {}
        picked = []
        for s in seqs:
            o = obj_of(s)
            if cnt.get(o, 0) < args.per_object:
                picked.append(s); cnt[o] = cnt.get(o, 0) + 1
        seqs = picked
    if args.limit:
        seqs = seqs[: args.limit]
    print(f"{len(seqs)} sequences from {args.split} split; objects: {sorted({obj_of(s) for s in seqs})}")

    out = args.out
    for d in ("objects", "source", "omni", f"umr/{args.split}", "umr/object_assets"):
        (out / d).mkdir(parents=True, exist_ok=True)

    metas: dict[str, dict] = {}
    for obj in sorted({obj_of(s) for s in seqs}):
        metas[obj] = prepare_object(obj, scales[obj], out / "objects" / obj, args.coacd_threshold)
        link = out / "umr" / "object_assets" / obj
        if not link.exists():
            link.symlink_to(Path("../../objects") / obj)

    heights_p = out / "omni" / "height_dict.pkl"
    heights: dict[str, float] = pickle.load(open(heights_p, "rb")) if heights_p.exists() else {}
    index = []
    for k, s in enumerate(seqs):
        name, obj = s["seq_name"], obj_of(s)
        sub = name.split("_")[0]
        if (out / "source" / f"{name}.npz").exists() and (out / "omni" / f"{name}.pt").exists() \
                and (out / "umr" / args.split / name / "poses.npy").exists():
            print(f"[{k+1}/{len(seqs)}] {name}: exists, skip")
        else:
            joints, height = smplx_fk(s, args.device)
            heights[sub] = height
            pos, quat = object_pose(s)
            np.savez_compressed(out / "source" / f"{name}.npz",
                                seq_name=name, object=obj, subject=sub, fps=FPS, gender=str(s["gender"]),
                                betas=s["betas"].reshape(-1), root_orient=s["root_orient"], pose_body=s["pose_body"].astype(np.float32),
                                trans=s["trans"], joints55=joints.astype(np.float32),
                                obj_pos=pos.astype(np.float32), obj_quat_xyzw=quat.astype(np.float32),
                                obj_scale=scales[obj], obj_scale_raw=s["obj_scale"], tpose_height=height)
            write_omni_pt(out / "omni" / f"{name}.pt", joints, s["root_orient"], pos, quat)
            write_umr_seq(out / "umr" / args.split / name, s, obj, metas[obj], pos, quat)
            print(f"[{k+1}/{len(seqs)}] {name}: T={len(joints)} height={height:.3f} obj={obj}")
        index.append({"seq": name, "object": obj, "subject": sub, "n_frames": int(len(s["trans"]))})

    pickle.dump(heights, open(heights_p, "wb"))
    (out / f"seqs_{args.split}.json").write_text(json.dumps(index, indent=1))
    print("heights:", {k: round(v, 3) for k, v in heights.items()})
    print("done ->", out)


if __name__ == "__main__":
    sys.exit(main())
