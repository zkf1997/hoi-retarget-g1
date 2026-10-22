"""Viser side-by-side of SPIDER vs RL tracking on the 5 s interaction clips (reference in the middle).

    BENCH_DATASET=hiphi ~/software/miniconda3/envs/hsretargeting/bin/python omomo_bench/wbt/viser_compare_w5.py \
        --rl logs/HiPHI_WBT/<run>/eval_tracking/model_05999_w5uni_rollout.npz [--spider hiphi_w5_uniws] [--port 8101]

Lanes: SPIDER (orange, y = 0) | reference (grey, y = 1.4 m) | RL policy (blue, y = 2.8 m). All three show the same clip
frame (reference frame k of data/motions_w5). The RL lane freezes at its bad-tracking termination. Labels above each
robot show the live object error; the sequence dropdown is sorted by SPIDER's object-on-target share and shows both
methods' scores (from compare_w5.py's per_clip.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("BENCH_DATASET", "hiphi")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import viser  # noqa: E402

import viser_compare as V  # noqa: E402  (RobotLane, display_mesh, URDFS)

FPS, N = 50, 250
PROC = HERE.parent / "spider_bridge" / "data" / "processed" / "{}" / "g1_wuji" / "humanoid_object"
LANES = {"spider": 0.0, "ref": 1.4, "rl": 2.8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", required=True, type=Path, help="eval_tracking.py --record rollout on the w5 clips")
    ap.add_argument("--spider", default="hiphi_w5_uniws", help="SPIDER dataset (hiphi_w5_uniws = RL-matched physics, hiphi_w5 = stiff)")
    ap.add_argument("--csv", type=Path, default=HERE.parent / "results_hiphi" / "compare_w5_uni" / "per_clip.csv")
    ap.add_argument("--port", type=int, default=8101)
    a = ap.parse_args()

    manifest = {e["tag"]: e for e in json.load(open(HERE / "data" / "manifest.json"))}
    R = np.load(a.rl)
    rl_idx = {str(t): i for i, t in enumerate(R["motion"])}
    rl_dof = [str(x) for x in R["dof_names"]]
    stats = {r["motion"]: r for r in csv.DictReader(open(a.csv))} if a.csv.exists() else {}
    proc = Path(str(PROC).format(a.spider))
    tags = [t.name for t in sorted(proc.iterdir()) if (t / "0" / "trajectory_mjwp.npz").exists() and t.name in rl_idx]
    sp_ok = lambda t: float(stats.get(t, {}).get("spider_obj_ok", 0) or 0)
    rl_ok = lambda t: float(stats.get(t, {}).get("rl_obj_ok", 0) or 0)
    tags.sort(key=lambda t: (-sp_ok(t), t))
    label = lambda t: f"{t}  [{manifest[t]['object']}]  SPIDER {sp_ok(t):.0%} | RL {rl_ok(t):.0%}"

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=10, height=10, cell_size=0.5, plane="xy", position=(0.0, 1.4, 0.0))
    rest = HERE.parent / "hand_rest_pose.json"
    lanes = {"spider": V.RobotLane(server, "spider", V.URDFS["wuji"], rest, (230, 140, 60), LANES["spider"]),
             "ref": V.RobotLane(server, "reference", V.URDFS["wuji"], rest, (170, 170, 170), LANES["ref"]),
             "rl": V.RobotLane(server, "rl", V.URDFS["wuji"], rest, (70, 130, 230), LANES["rl"])}
    title = server.scene.add_label("/title", text="", position=(0.0, 1.4, 2.4))

    gui_seq = server.gui.add_dropdown("sequence", options=[label(t) for t in tags], initial_value=label(tags[0]))
    gui_t = server.gui.add_slider("frame", min=0, max=N - 1, step=1, initial_value=0)
    gui_play = server.gui.add_checkbox("play", initial_value=True)
    gui_speed = server.gui.add_slider("speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
    gui_info = server.gui.add_markdown("")
    lock = threading.RLock()
    cur = {"programmatic": False}

    def load(tag: str):
        with lock:
            info = manifest[tag]
            clip = np.load(HERE / "data" / "motions_w5" / f"{tag}.npz")
            o = np.load(proc / tag / "0" / "trajectory_mjwp.npz")
            e = rl_idx[tag]
            cur.update(tag=tag, clip=clip, names=[str(x) for x in clip["joint_names"]],
                       sp=o["qpos"].reshape(-1, o["qpos"].shape[-1])[::2][:N],  # state i ~ reference frame i + 1
                       e=e, rl_n=int(R["active"][e].sum()), rl_t0=int(R["tstep"][e][0]))
            mesh = V.display_mesh(info["object"])
            for lane in lanes.values():
                lane.set_object(mesh, info["smpl_scale"])
            st = stats.get(tag, {})
            f = lambda k, fmt="{:.0f}": fmt.format(float(st[k])) if k in st else "-"
            rl_end = cur["rl_t0"] + cur["rl_n"]
            title.text = f"{tag}  |  {info['object']}"
            gui_info.content = (
                f"**{tag}** — {info['object']} ({info['mass_kg']:.2f} kg)  \n"
                f"orange = SPIDER (`{a.spider}`), grey = reference, blue = RL  \n\n"
                "| | SPIDER | RL |\n|---|---|---|\n"
                f"| tracked | {f('spider_tracked_s', '{:.1f}')} s{' (fell)' if st.get('spider_fell') == 'True' else ''} | "
                f"{f('rl_tracked_s', '{:.1f}')} s{' (terminated at %.1f s)' % (rl_end / FPS) if rl_end < N - 5 else ''} |\n"
                f"| object on target | {f('spider_obj_ok', '{:.0%}')} | {f('rl_obj_ok', '{:.0%}')} |\n"
                f"| + fingertips | {f('spider_hand_ok', '{:.0%}')} | {f('rl_hand_ok', '{:.0%}')} |\n"
                f"| root err mm | {f('spider_root_mm')} | {f('rl_root_mm')} |\n"
                f"| fingertip err mm | {f('spider_tip_mm')} | {f('rl_tip_mm')} |\n"
                f"| object err mm | {f('spider_obj_mm')} | {f('rl_obj_mm')} |\n")
            cur["programmatic"] = True
            gui_t.value = 0
            cur["programmatic"] = False
            render(0)

    def render(k: int):
        with lock:
            clip, names = cur["clip"], cur["names"]
            k = min(k, len(clip["joint_pos"]) - 1)
            up = np.array([0.0, 0.0, 0.8])
            # reference
            ref_q, ref_o = clip["joint_pos"][k], clip["object_pos_w"][k]
            lanes["ref"].pose(ref_q, ref_o, clip["object_quat_w"][k], joint_names=names)
            lanes["ref"].label.text = "reference"
            lanes["ref"].label.position = tuple(ref_q[:3] + up)
            # SPIDER
            sp = cur["sp"][min(max(k - 1, 0), len(cur["sp"]) - 1)]
            lanes["spider"].pose(sp[:76], sp[76:79], sp[79:83], joint_names=names)
            err = 1000 * np.linalg.norm(sp[76:79] - ref_o)
            lanes["spider"].label.text = f"SPIDER  object err {err:.0f} mm"
            lanes["spider"].label.position = tuple(sp[:3] + up)
            # RL: rollout index t has reference index tstep[t] = rl_t0 + t; freeze after termination
            e = cur["e"]
            t = int(np.clip(k - cur["rl_t0"], 0, cur["rl_n"] - 1))
            q = np.concatenate([R["root"][e, t], R["dof"][e, t]])
            ob = R["obj"][e, t]
            lanes["rl"].pose(q, ob[:3], ob[3:], joint_names=rl_dof)
            done = k - cur["rl_t0"] >= cur["rl_n"]
            ref_t = clip["object_pos_w"][min(cur["rl_t0"] + t, len(clip["object_pos_w"]) - 1)]
            lanes["rl"].label.text = (f"RL  terminated at {(cur['rl_t0'] + cur['rl_n']) / FPS:.1f} s" if done
                                      else f"RL  object err {1000 * np.linalg.norm(ob[:3] - ref_t):.0f} mm")
            lanes["rl"].label.position = tuple(q[:3] + up)

    @gui_seq.on_update
    def _(_):
        load(tags[[label(t) for t in tags].index(gui_seq.value)])

    @gui_t.on_update
    def _(_):
        if not cur["programmatic"]:
            render(int(gui_t.value))

    load(tags[0])
    print(f"SPIDER vs RL viewer on http://localhost:{a.port}  ({len(tags)} clips)", flush=True)
    while True:
        t0 = time.time()
        if gui_play.value:
            with lock:
                nxt = (int(gui_t.value) + 1) % N
                cur["programmatic"] = True
                gui_t.value = nxt
                cur["programmatic"] = False
            render(nxt)
        time.sleep(max(0.0, 1.0 / (FPS * gui_speed.value) - (time.time() - t0)))


if __name__ == "__main__":
    main()
