#!/usr/bin/env python3
"""viser side-by-side playback: HUMOTO human + objects | robot hands (method A) | robot hands (method B).

    python viser_compare.py --hand brainco --methods anyteleop toporetarget --port 8080
    ssh -L 8080:localhost:8080 <host>   ->   http://localhost:8080

Lanes are offset along +y. Robot hands float at the method's flange pose (AnyTeleop: on the human wrist; TopoRetarget: its
optimised base pose); the body is not retargeted. GUI: sequence, robot hand, play/pause, speed, frame, lane spacing/visibility.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R

import bench_common as bc
import humoto_io as H
from hand_model import HandModel

BODY_EDGES = [(H.MIXAMO_BODY.index(c), H.MIXAMO_BODY.index(p)) for c, p in H.MIXAMO_BODY_PARENT.items()]
HAND_EDGES = [(k, H.HAND_KP_PARENT[k]) for k in range(1, 21)]
COLORS = {"anyteleop": (90, 160, 255), "anyteleop_dexpilot": (120, 200, 160), "toporetarget": (255, 150, 80)}


def quat_mul(a, b):  # wxyz
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


class HandVis:
    """one robot hand in one lane: a viser frame per MuJoCo body with the visual meshes attached."""

    def __init__(self, server: viser.ViserServer, prefix: str, model: HandModel, color):
        m = model.model
        self.model, self.m = model, m
        self.nodes = []
        vis = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
        if not vis:
            vis = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.bodies = sorted({int(m.geom_bodyid[g]) for g in vis})
        for b in self.bodies:
            f = server.scene.add_frame(f"{prefix}/b{b}", show_axes=False)
            self.nodes.append(f)
            for g in vis:
                if m.geom_bodyid[g] != b:
                    continue
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                fc = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                server.scene.add_mesh_simple(f"{prefix}/b{b}/g{g}", v, fc, color=color, wxyz=m.geom_quat[g], position=m.geom_pos[g])

    def precompute(self, q_active: np.ndarray, base_pos: np.ndarray, base_rot: np.ndarray):
        """world pose of every rendered body for all frames -> (T,nb,3), (T,nb,4)"""
        T = len(q_active)
        pos, quat = np.zeros((T, len(self.bodies), 3)), np.zeros((T, len(self.bodies), 4))
        bq = R.from_matrix(base_rot).as_quat()[:, [3, 0, 1, 2]]
        for t in range(T):
            self.model.forward(q_active[t])
            xp, xq = self.model.data.xpos[self.bodies], self.model.data.xquat[self.bodies]
            pos[t] = base_pos[t] + xp @ base_rot[t].T
            quat[t] = quat_mul(np.repeat(bq[t][None], len(self.bodies), 0), xq)
        return pos, quat

    def set(self, pos, quat, offset):
        for i, n in enumerate(self.nodes):
            n.position, n.wxyz = pos[i] + offset, quat[i]


class Viewer:
    def __init__(self, server: viser.ViserServer, methods: list[str], hand: str):
        self.server, self.methods, self.hand = server, methods, hand
        self.handles = []
        self.seq = None

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def load(self, seq: str, hand: str):
        self.clear()
        self.hand = hand
        d = bc.load_seq(seq)
        self.d, self.seq, self.T = d, seq, d["n_frames"]
        s = self.server.scene
        # lane 0: human
        self.body_lines = s.add_line_segments("/human/body", np.zeros((len(BODY_EDGES), 2, 3)), colors=(40, 40, 40), line_width=3)
        self.hand_lines = {side: s.add_line_segments(f"/human/hand_{side}", np.zeros((20, 2, 3)), colors=(200, 40, 40), line_width=2) for side in bc.SIDES}
        self.handles += [self.body_lines, *self.hand_lines.values()]
        # objects, one copy per lane
        self.obj_nodes = []  # (lane, k, handle)
        meshes = {o: (bc.load_object_visual(o) if (bc.OBJ_DIR / o / "visual.obj").exists() else H.load_object_mesh(seq, o)) for o in d["objects"]}
        for lane in range(1 + len(self.methods)):
            for k, o in enumerate(d["objects"]):
                col = (120, 120, 120) if d["handheld"][k] else (190, 190, 190)
                h = s.add_mesh_simple(f"/lane{lane}/obj_{o}", meshes[o].vertices * float(d["obj_scale"][0, k]), meshes[o].faces, color=col, opacity=0.85 if d["handheld"][k] else 0.4)
                self.obj_nodes.append((lane, k, h))
                self.handles.append(h)
        # robot hands
        self.robot = []  # (lane, HandVis, pos, quat)
        self.missing = []
        for li, method in enumerate(self.methods, start=1):
            for side in bc.SIDES:
                res = bc.load_result(method, hand, seq, side)
                if res is None:
                    self.missing.append(f"{method}/{side}")
                    continue
                model = HandModel(hand, side)
                hv = HandVis(self.server, f"/lane{li}/{side}", model, COLORS.get(method, (150, 150, 150)))
                self.handles += hv.nodes
                Tm = int(res.get("n_frames_done", self.T))
                fl_pos, fl_rot = bc.flange_world(d, side, res["base_pos_h"][:Tm], res["base_quat_h"][:Tm])
                base_pos = fl_pos - np.einsum("tij,j->ti", fl_rot, model.flange)
                pos, quat = hv.precompute(res["q_active"][:Tm].astype(np.float64), base_pos, fl_rot)
                self.robot.append((li, hv, pos, quat))
        self.labels = [s.add_label(f"/lane{li}/label", text=m) for li, m in enumerate(["HUMOTO"] + self.methods)]
        self.handles += self.labels

    def show(self, t: int, spacing: float, visible: list[bool]):
        d = self.d
        t = min(t, self.T - 1)
        with self.server.atomic():
            bp = d["body_pos"][t]
            self.body_lines.points = np.stack([bp[[a for a, _ in BODY_EDGES]], bp[[b for _, b in BODY_EDGES]]], axis=1)
            self.body_lines.visible = visible[0]
            for side in bc.SIDES:
                kp = d[f"hand_kp_{side}"][t]
                self.hand_lines[side].points = np.stack([kp[[a for a, _ in HAND_EDGES]], kp[[b for _, b in HAND_EDGES]]], axis=1)
                self.hand_lines[side].visible = visible[0]
            for lane, k, h in self.obj_nodes:
                off = np.array([0.0, spacing * lane, 0.0])
                h.position = d["obj_pos"][t, k] + off
                h.wxyz = R.from_matrix(d["obj_rot"][t, k]).as_quat()[[3, 0, 1, 2]]
                h.visible = visible[lane]
            for lane, hv, pos, quat in self.robot:
                tt = min(t, len(pos) - 1)
                hv.set(pos[tt], quat[tt], np.array([0.0, spacing * lane, 0.0]))
                for n in hv.nodes:
                    n.visible = visible[lane]
            for li, lab in enumerate(self.labels):
                lab.position = np.array([0.0, spacing * li, 2.0]) + (d["body_pos"][t, 0] * [1, 0, 0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", choices=bc.HANDS, default="brainco")
    ap.add_argument("--methods", nargs="*", default=["anyteleop", "toporetarget"])
    ap.add_argument("--seq", default=None)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--check", action="store_true", help="build the first sequence and exit")
    a = ap.parse_args()
    seqs = bc.list_prepared()
    server = viser.ViserServer(port=a.port)
    server.scene.add_grid("/grid", width=8, height=8, plane="xy")
    view = Viewer(server, a.methods, a.hand)
    n_lanes = 1 + len(a.methods)

    seq_dd = server.gui.add_dropdown("sequence", seqs, initial_value=a.seq or seqs[0])
    hand_dd = server.gui.add_dropdown("robot hand", list(bc.HANDS), initial_value=a.hand)
    play = server.gui.add_checkbox("play", True)
    speed = server.gui.add_slider("speed", 0.1, 3.0, 0.1, 1.0)
    frame = server.gui.add_slider("frame", 0, 1, 1, 0)
    spacing = server.gui.add_slider("lane spacing (m)", 0.5, 5.0, 0.1, 2.0)
    vis = [server.gui.add_checkbox(name, True) for name in ["HUMOTO"] + a.methods]
    status = server.gui.add_text("status", "", disabled=True)

    def reload(_=None):
        view.load(seq_dd.value, hand_dd.value)
        frame.max = view.T - 1
        frame.value = 0
        status.value = f"{view.T} frames" + (f"; missing {view.missing}" if view.missing else "")

    seq_dd.on_update(reload)
    hand_dd.on_update(reload)
    reload()
    if a.check:
        view.show(0, spacing.value, [True] * n_lanes)
        print("check ok:", seq_dd.value, view.T, "frames; missing:", view.missing)
        return
    t, last = 0.0, time.time()
    while True:
        now = time.time()
        if play.value:
            t = (t + (now - last) * H.FPS * speed.value) % view.T
            frame.value = int(t)
        else:
            t = frame.value
        last = now
        view.show(int(t), spacing.value, [v.value for v in vis])
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()
