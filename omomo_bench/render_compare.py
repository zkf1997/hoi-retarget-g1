#!/usr/bin/env python3
"""Static side-by-side renders (PNG grid) of the source human and every result lane, for reports.

Run in the `hsretargeting` conda env (mujoco EGL, smplx, torch, PIL):
    MUJOCO_GL=egl python render_compare.py --seqs sub16_largebox_000 --frames 5       # -> results/figures/<seq>.png
    BENCH_DATASET=humoto MUJOCO_GL=egl python render_compare.py --seqs <seq> --lane omni_brainco=results_humoto/omni_brainco_umrscale:_original ...
    BENCH_DATASET=humoto MUJOCO_GL=egl python render_compare.py --pipeline --seqs <seq> --focus right_hand --distance 0.7   # 4 stages side by side

Columns = OMOMO/HUMOTO source (SMPL-X mesh + object, real size) | result lanes (robot MJCF the run used + object, in the
lane's own scaled world); rows = frames spread over the sequence. The camera follows the pelvis / robot root of each lane.
Lanes default to compare.py's LANES (omni_brainco, umr_brainco, omni_wuji, umr_wuji).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C  # noqa: E402  (honours BENCH_DATASET)

HERE = Path(__file__).resolve().parent
W, H = 420, 480  # default MuJoCo offscreen framebuffer is 640x480
# SMPL-X body models (license-gated): $SMPLX_DIR/smplx/SMPLX_*.npz, else UMR/smpl/ (env_setup.sh symlinks). Without them the
# source column shows the joints as spheres.
SMPLX_DIR = Path(os.environ.get("SMPLX_DIR", "/media/kaizhao/hdd/dataset/models_smplx_v1_1/models"))
if not (SMPLX_DIR / "smplx").exists() and (HERE.parent / "UMR" / "smpl" / "SMPLX_NEUTRAL.npz").exists():
    SMPLX_DIR = HERE.parent / "UMR"
SMPLX_NEUTRAL = next((p for p in (HERE.parent / "UMR" / "smpl" / "SMPLX_NEUTRAL.npz", SMPLX_DIR / "smplx" / "SMPLX_NEUTRAL.npz") if p.exists()), None)


def camera(lookat: np.ndarray, azimuth: float, distance: float, elevation: float = -12.0) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def render(model: mujoco.MjModel, qpos: np.ndarray, lookat, azimuth, distance) -> np.ndarray:
    d = mujoco.MjData(model)
    d.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, d)
    r = mujoco.Renderer(model, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0; opt.geomgroup[0] = 1; opt.geomgroup[1] = 1; opt.geomgroup[2] = 1
    r.update_scene(d, camera(lookat, azimuth, distance), opt)
    img = r.render().copy()
    r.close()
    return img


def joint_spheres(src: dict, frames: list[int], radius: float = 0.025):
    """Fallback source human when the SMPL-X files are missing: one small icosphere per SMPL-X joint (55 per frame)."""
    import trimesh
    ball = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    V, F = [], []
    for f in frames:
        verts, faces = [], []
        for k, j in enumerate(np.asarray(src["joints55"][f], np.float32)):
            faces.append(ball.faces + len(ball.vertices) * k); verts.append(ball.vertices + j)
        V.append(np.concatenate(verts)); F.append(np.concatenate(faces))
    return np.stack(V), F[0].astype(np.int32)


def smplx_vertices(src: dict, frames: list[int]):
    try:
        import smplx, torch
        gender = str(src["gender"])
        if gender == "neutral" and SMPLX_NEUTRAL is None:
            raise FileNotFoundError("SMPLX_NEUTRAL.npz")
    except Exception as ex:
        print(f"[render] SMPL-X unavailable ({type(ex).__name__}: {ex}); source column shows joints", flush=True)
        return joint_spheres(src, frames)
    T = len(frames)
    path = str(SMPLX_NEUTRAL) if gender == "neutral" else str(SMPLX_DIR)
    try:
        m = smplx.create(path, model_type="smplx", gender=gender, num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=T, ext="npz")
    except Exception as ex:
        print(f"[render] SMPL-X model for gender={gender} unavailable ({type(ex).__name__}); source column shows joints", flush=True)
        return joint_spheres(src, frames)
    kw = {}
    if "pose_hand" in src:
        ph = np.asarray(src["pose_hand"], np.float32)[frames]
        kw = {"left_hand_pose": torch.as_tensor(ph[:, :45]), "right_hand_pose": torch.as_tensor(ph[:, 45:])}
    with torch.no_grad():
        out = m(betas=torch.as_tensor(np.repeat(np.asarray(src["betas"], np.float32)[None], T, 0)),
                global_orient=torch.as_tensor(np.asarray(src["root_orient"], np.float32)[frames]),
                body_pose=torch.as_tensor(np.asarray(src["pose_body"], np.float32)[frames]),
                transl=torch.as_tensor(np.asarray(src["trans"], np.float32)[frames]), **kw)
    return out.vertices.numpy(), m.faces.astype(np.int32)


def human_scene_xml(obj_mesh_path: Path, body_obj: Path, extra: list[Path]) -> str:
    assets = [f'<mesh name="body" file="{body_obj.as_posix()}"/>', f'<mesh name="obj" file="{obj_mesh_path.as_posix()}"/>']
    geoms = ['<geom name="body" type="mesh" mesh="body" rgba="0.80 0.62 0.52 1" contype="0" conaffinity="0"/>',
             '<body name="obj_link"><freejoint/><geom type="mesh" mesh="obj" rgba="0.72 0.78 0.9 1" contype="0" conaffinity="0"/></body>']
    for k, p in enumerate(extra):
        assets.append(f'<mesh name="extra{k}" file="{p.as_posix()}"/>')
        geoms.append(f'<body name="extra{k}_link"><freejoint/><geom type="mesh" mesh="extra{k}" rgba="0.6 0.6 0.65 1" contype="0" conaffinity="0"/></body>')
    return f"""<mujoco><compiler angle="radian"/>
<visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/><quality shadowsize="2048"/></visual>
<asset><texture type="2d" name="grid" builtin="checker" rgb1="0.25 0.3 0.4" rgb2="0.35 0.45 0.6" width="300" height="300"/>
<material name="grid" texture="grid" texrepeat="8 8" texuniform="true" reflectance="0.1"/>{"".join(assets)}</asset>
<worldbody><light pos="0 0 4" dir="0 0 -1" directional="true"/><geom name="ground" type="plane" size="10 10 0.1" material="grid"/>{"".join(geoms)}</worldbody></mujoco>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--video", action="store_true", help="render every --stride-th frame and write <seq><tag>.mp4 (columns side by side) instead of a PNG grid")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--lane", action="append", default=None, metavar="NAME=DIR[:SUFFIX]")
    ap.add_argument("--azimuth", type=float, default=150.0)
    ap.add_argument("--distance", type=float, default=3.2)
    ap.add_argument("--out", type=Path, default=None, help="output dir (default results*/figures)")
    ap.add_argument("--focus", choices=["root", "left_hand", "right_hand"], default="root",
                    help="camera target: pelvis/root or a hand (robot wrist_yaw_link / human wrist joint); use with a small --distance")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--all-objects", action="store_true", help="draw every object of data*/multi_objects.json in every result lane")
    ap.add_argument("--pipeline", action="store_true", help="lanes = the four hand_pipeline.py stages side by side")
    args = ap.parse_args()
    LABELS = {"source": "source (human)"}
    if args.pipeline and not args.lane:
        C.LANES.clear()
        C.LANES.update({"1_umr_body": (C.RESULTS / "umr_wuji", ""), "2_anyteleop_fingers": (C.RESULTS / "pipe_s2", ""),
                        "3p_arm_place": (C.RESULTS / "pipe_s3p", ""),
                        "v2_coupled": (C.RESULTS / "pipe_v2", ""), "v3_dexweave": (C.RESULTS / "pipe_v3", ""), "final": (C.RESULTS / "pipe_final", "")})
        C.LANES.update({k: v for k, v in list(C.LANES.items()) if v[0].exists()}); [C.LANES.pop(k) for k, v in list(C.LANES.items()) if not v[0].exists()]
        LABELS.update({"1_umr_body": "stage 1: UMR body, rest hand", "2_anyteleop_fingers": "stage 2: + AnyTeleop fingers",
                       "3_finger_refine": "stage 3: + finger refinement", "3p_arm_place": "stage 3+: + arm placement (v1)",
                       "v2_coupled": "v2: coupled arm + finger QP", "v3_dexweave": "v3: + interaction gating, priorities, correction smoothing", "final": "final: v3, 12 iterations"})
    HUMAN_FOCUS = {"root": 0, "left_hand": 20, "right_hand": 21}  # SMPL-X joint ids
    if args.lane:
        C.LANES.clear()
        for spec in args.lane:
            name, rest = spec.split("=", 1); d, _, suffix = rest.partition(":"); C.LANES[name] = ((HERE / d) if not Path(d).is_absolute() else Path(d), suffix)
    out_dir = args.out or (C.RESULTS / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {e["seq"]: e for e in json.loads((C.DATA / "seqs_test.json").read_text())}
    font = ImageFont.load_default()

    for seq in args.seqs:
        e = index[seq]; obj = e["object"]
        src = dict(np.load(C.DATA / "source" / f"{seq}.npz", allow_pickle=True))
        T = len(src["trans"])
        frames = list(range(0, T, args.stride)) if args.video else np.linspace(0, T - 1, args.frames).round().astype(int).tolist()
        cols: dict[str, list[np.ndarray]] = {}
        # --- source human
        verts, faces = smplx_vertices(src, frames)
        extra_names = [str(x) for x in src["extra_objects"]] if "extra_objects" in src else []
        extra_paths = []
        for n in extra_names:
            for p in (C.DATA / "objects" / n / f"{n}.obj", HERE.parent / "hand_bench" / "data" / "objects" / n / "visual.obj"):
                if p.exists():
                    extra_paths.append(p); break
            else:
                try:
                    sys.path.insert(0, str(HERE.parent / "hand_bench")); import humoto_io as HIO, trimesh
                    tmp = Path(tempfile.mkdtemp()) / f"{n}.obj"; HIO.load_object_mesh(seq, n).export(tmp); extra_paths.append(tmp)
                except Exception:
                    extra_paths.append(None)
        imgs = []
        with tempfile.TemporaryDirectory() as td:
            import trimesh
            for k, f in enumerate(frames):
                body = Path(td) / f"body_{k}.obj"
                trimesh.Trimesh(verts[k], faces, process=False).export(body)
                valid = [p for p in extra_paths if p is not None]
                m = mujoco.MjModel.from_xml_string(human_scene_xml(C.DATA / "objects" / obj / f"{obj}.obj", body, valid))
                q = np.concatenate([src["obj_pos"][f], src["obj_quat_xyzw"][f][[3, 0, 1, 2]]])
                for j, (n, p) in enumerate(zip(extra_names, extra_paths)):
                    if p is None:
                        continue
                    q = np.concatenate([q, src["extra_obj_pos"][f, j], src["extra_obj_quat_xyzw"][f, j][[3, 0, 1, 2]]])
                look = src["joints55"][f, HUMAN_FOCUS[args.focus]] - ([0, 0, 0.15] if args.focus == "root" else 0.0)
                imgs.append(render(m, q, look, args.azimuth, args.distance))
        cols["source"] = imgs
        # --- result lanes
        for name, (rdir, suffix) in C.LANES.items():
            p = rdir / f"{seq}{suffix}.npz"
            if not p.exists():
                continue
            r = np.load(p, allow_pickle=True)
            if "human_joints" in r:
                m = C.omni_model(str(r["hand"]) if "hand" in r else "none", str(r["scaling"]) if "scaling" in r else "omni", e["subject"], obj)
                qall = r["qpos"]
                scale = float(r["smpl_scale"]) if "smpl_scale" in r else 1.0
            else:
                scale = float(np.asarray(r["smpl_scale"]).reshape(-1)[0])
                ms = float(np.asarray(r["object_mesh_scale"]).reshape(-1)[0]) if "object_mesh_scale" in r else scale
                objs_r = [str(x) for x in r["objects"]] if "objects" in r else [obj]  # multi-object hand stage: all objects, primary first
                if args.all_objects:  # draw every object of the sequence in every lane (shows what a primary-only lane passes through)
                    objs_r = C.multi_objects(seq, src)
                extra_names = [str(x) for x in src["extra_objects"]] if "extra_objects" in src else []
                objs_r = [objs_r[0]] + [o for o in objs_r[1:] if o in extra_names]
                m = C.augmented_model(C.umr_robot_xml(r), objs_r if len(objs_r) > 1 else obj, ms, for_render=True)
                Tq = len(r["qpos"])
                op = C.umr_object_pos(r, src, scale); oq = src["obj_quat_xyzw"][:, [3, 0, 1, 2]]
                parts = [r["qpos"], op[:Tq], oq[:Tq]]
                for o in objs_r[1:]:
                    k = extra_names.index(o)
                    parts += [src["extra_obj_pos"][:Tq, k] * scale, src["extra_obj_quat_xyzw"][:Tq, k][:, [3, 0, 1, 2]]]
                qall = np.concatenate(parts, axis=1)
            imgs = []
            dd = mujoco.MjData(m)
            wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{args.focus.split('_')[0]}_wrist_yaw_link") if args.focus != "root" else -1
            for f in frames:
                f = min(f, len(qall) - 1)
                if wrist >= 0:
                    dd.qpos[:len(qall[f])] = qall[f]; mujoco.mj_forward(m, dd); look = dd.xpos[wrist].copy()
                else:
                    look = qall[f, :3] - [0, 0, 0.15]
                imgs.append(render(m, qall[f], look, args.azimuth, args.distance * scale))
            cols[name] = imgs
        names = list(cols)
        pad, top = 6, 26
        if args.video:
            import imageio
            outp = out_dir / f"{seq}{args.tag}.mp4"
            with imageio.get_writer(outp, fps=args.fps / args.stride, codec="libx264", quality=8, macro_block_size=1) as wr:
                for ri, f in enumerate(frames):
                    row = Image.new("RGB", (len(names) * (W + pad), top + H), (255, 255, 255)); dr = ImageDraw.Draw(row)
                    for ci, n in enumerate(names):
                        dr.text((ci * (W + pad) + 8, 6), LABELS.get(n, n), fill=(0, 0, 0), font=font)
                        row.paste(Image.fromarray(cols[n][ri]), (ci * (W + pad), top))
                    dr.text((8, top + 8), f"frame {f}/{T}", fill=(255, 255, 255), font=font)
                    wr.append_data(np.asarray(row))
            print("wrote", outp, len(frames), "frames"); continue
        # --- grid
        grid = Image.new("RGB", (len(names) * (W + pad), top + len(frames) * (H + pad)), (255, 255, 255))
        dr = ImageDraw.Draw(grid)
        for ci, n in enumerate(names):
            dr.text((ci * (W + pad) + 8, 6), LABELS.get(n, n), fill=(0, 0, 0), font=font)
            for ri, img in enumerate(cols[n]):
                grid.paste(Image.fromarray(img), (ci * (W + pad), top + ri * (H + pad)))
        for ri, f in enumerate(frames):
            dr.text((8, top + ri * (H + pad) + 8), f"frame {f}/{T}", fill=(255, 255, 255), font=font)
        outp = out_dir / f"{seq}{args.tag}.png"
        grid.save(outp)
        print("wrote", outp, grid.size)


if __name__ == "__main__":
    main()
