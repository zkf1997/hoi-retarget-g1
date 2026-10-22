"""Viser viewer for tracking-policy evaluation rollouts (eval_tracking.py --record).

Per sequence: the SIMULATED robot + object (policy rollout, coloured) next to / on top of the REFERENCE robot + object
(retargeted motion, grey). After the policy is terminated the simulated lane freezes at its last frame.

    BENCH_DATASET=hiphi conda run -n umr python omomo_bench/wbt/viser_rollout.py \
        --rollout logs/HiPHI_WBT/<run>/eval_tracking/model_XXXXX_p0_rollout.npz [--port 8095]
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

FPS = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--name", default="policy", help="label of the simulated lane, e.g. 'RL policy' or 'SPIDER'")
    a = ap.parse_args()

    R = np.load(a.rollout, allow_pickle=False)
    motions = [str(m) for m in R["motion"]]
    dof_names = [str(n) for n in R["dof_names"]]
    root, dof, obj, tstep, active = R["root"], R["dof"], R["obj"], R["tstep"], R["active"]  # (E, T, ...)
    manifest = {e["tag"]: e for e in json.load(open(HERE / "data" / "manifest.json"))}
    csv_path = Path(str(a.rollout).replace("_rollout.npz", "_per_motion.csv"))
    if not csv_path.exists():
        cands = list(a.rollout.parent.glob("*_per_motion.csv"))
        csv_path = cands[0] if cands else csv_path
    stats = {r["motion"]: r for r in csv.DictReader(open(csv_path))} if csv_path.exists() else {}

    # order: longest-tracked first
    n_active = active.sum(1)
    order = list(np.argsort(-n_active))

    def label(e: int) -> str:
        m = motions[e]
        st = stats.get(m, {})
        ok = "DONE" if st.get("completed") == "True" else f"fail@{int(n_active[e])}"
        return f"{m}  [{manifest[m]['object']}]  {ok}  objOK {float(st.get('obj_ok_frac', 0) or 0):.2f}"

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=8, height=8, cell_size=0.5, plane="xy")
    sim = V.RobotLane(server, "simulated", V.URDFS["wuji"], HERE.parent / "hand_rest_pose.json", (70, 130, 230), 0.0)
    ref = V.RobotLane(server, "reference", V.URDFS["wuji"], HERE.parent / "hand_rest_pose.json", (170, 170, 170), 0.0)
    for lane in (sim, ref):
        lane.label.visible = False
    # 3D labels that follow each robot (above the pelvis)
    sim_tag = server.scene.add_label("/labels/sim", text=a.name, position=(0.0, 0.0, 1.5))
    ref_tag = server.scene.add_label("/labels/ref", text="reference", position=(0.0, 1.2, 1.5))
    seq_tag = server.scene.add_label("/labels/seq", text="", position=(0.0, 0.6, 1.9))

    gui_seq = server.gui.add_dropdown("sequence", options=[label(e) for e in order], initial_value=label(order[0]))
    gui_overlay = server.gui.add_checkbox("overlay sim on reference", initial_value=False)
    gui_t = server.gui.add_slider("frame", min=0, max=root.shape[1] - 1, step=1, initial_value=0)
    gui_play = server.gui.add_checkbox("play", initial_value=True)
    gui_speed = server.gui.add_slider("speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
    gui_info = server.gui.add_markdown("")
    lock = threading.RLock()
    cur = {"e": int(order[0]), "ref": None, "programmatic": False}

    def load(e: int):
        with lock:
            m = motions[e]
            info = manifest[m]
            mot = np.load(HERE / "data" / "motions" / f"{m}.npz")
            cur.update(e=e, ref=mot)
            mesh = V.display_mesh(info["object"])
            sim.set_object(mesh, info["smpl_scale"])
            ref.set_object(mesh, info["smpl_scale"])
            n = int(n_active[e])
            gui_t.max = max(n - 1, 1)
            cur["programmatic"] = True
            gui_t.value = 0
            cur["programmatic"] = False
            st = stats.get(m, {})
            status = "completed" if st.get("completed") == "True" else f"terminated at {n / FPS:.1f} s"
            seq_tag.text = f"{m}  |  {info['object']}  |  {status}  |  object OK {float(st.get('obj_ok_frac', 0) or 0):.0%}"
            sim_tag.text = f"{a.name} (sim)"
            fmt = lambda k, u: f"{float(st[k]):.0f} {u}" if k in st else "-"
            gui_info.content = (f"**{m}** — {info['object']} (scale {info['smpl_scale']:.2f}, {info['mass_kg']:.2f} kg)  \n"
                                f"tracked {n}/{int(st.get('frames_total', n))} frames ({n / FPS:.1f} s), "
                                f"{'completed' if st.get('completed') == 'True' else 'terminated'}  \n"
                                f"mean error: body {fmt('body_pos_mean', 'mm')}, wrist {fmt('wrist_pos_mean', 'mm')}, "
                                f"fingertip {fmt('fingertip_pos_mean', 'mm')}, object {fmt('object_pos_mean', 'mm')}  \n"
                                f"object on target (3 cm/30°): {float(st.get('obj_ok_frac', 0)):.0%}, + fingertips (6 cm): {float(st.get('hand_ok_frac', 0)):.0%}  \n"
                                f"blue = {a.name} (simulated), grey = reference")
            render(0)

    def render(t: int):
        with lock:
            e, mot = cur["e"], cur["ref"]
            n = int(n_active[e])
            t = min(t, n - 1)
            ts = int(np.clip(tstep[e, t], 0, mot["joint_pos"].shape[0] - 1))
            off = np.zeros(3) if gui_overlay.value else np.array([0.0, 1.2, 0.0])
            q_sim = np.concatenate([root[e, t], dof[e, t]])
            sim.pose(q_sim, obj[e, t, :3], obj[e, t, 3:], joint_names=dof_names)
            q_ref = mot["joint_pos"][ts].copy()
            q_ref[:3] += off
            ref.pose(q_ref, mot["object_pos_w"][ts] + off, mot["object_quat_w"][ts], joint_names=[str(j) for j in mot["joint_names"]])
            up = np.array([0.0, 0.0, 0.75])
            sim_tag.position = tuple(root[e, t, :3] + up)
            ref_tag.position = tuple(q_ref[:3] + up)
            seq_tag.position = tuple((root[e, t, :3] + q_ref[:3]) / 2 + np.array([0.0, 0.0, 1.15]))

    @gui_seq.on_update
    def _(_):
        e = order[[label(i) for i in order].index(gui_seq.value)]
        load(int(e))

    @gui_t.on_update
    def _(_):
        if not cur["programmatic"]:
            render(int(gui_t.value))

    @gui_overlay.on_update
    def _(_):
        render(int(gui_t.value))

    load(int(order[0]))
    print(f"viser rollout viewer on http://localhost:{a.port}  ({len(motions)} sequences)")
    while True:
        t0 = time.time()
        if gui_play.value:
            with lock:
                nxt = int(gui_t.value) + 1
                if nxt > gui_t.max:
                    nxt = 0
                cur["programmatic"] = True
                gui_t.value = nxt
                cur["programmatic"] = False
            render(nxt)
        time.sleep(max(0.0, 1.0 / (FPS * gui_speed.value) - (time.time() - t0)))


if __name__ == "__main__":
    main()
