#!/bin/bash
# White-chair (sub16_whitechair_002) analysis: per-frame wrist gap series, hand-axis shift sweep, rest-hand penetration at the human wrist pose, and the 3-panel plot.
# Recovered from the 2026-10-10 session; re-run after the OmniRetarget hand-bake fix (2026-10-14). Steps: series -> sweep -> tp -> plot.
cd /home/kaizhao/projects/retarget/omomo_bench

# ---------------- series
export BENCH_DATASET=omomo MUJOCO_GL=egl OMP_NUM_THREADS=1 && timeout 580 /home/kaizhao/software/miniconda3/envs/umr/bin/python - <<'EOF' 2>&1 | grep -v Warning
import numpy as np, mujoco, sys, json, trimesh
sys.path.insert(0, "."); import compare as C
from scipy.spatial.transform import Rotation as Rot
seq = "sub16_whitechair_002"; e = next(x for x in json.load(open(C.DATA / "seqs_test.json")) if x["seq"] == seq)
src = np.load(C.DATA / "source" / f"{seq}.npz", allow_pickle=True); obj = str(src["object"]); J = src["joints55"]; s = 1.32 / float(src["tpose_height"]); T = len(J)
mesh = trimesh.load(C.DATA / "objects" / obj / f"{obj}.obj", force="mesh"); pq = trimesh.proximity.ProximityQuery(mesh)
oR = Rot.from_quat(src["obj_quat_xyzw"]).as_matrix(); op = src["obj_pos"]
lanes = {"omni": (C.RESULTS / "omni_wuji_umrscale", "_original", None), "omni_fix": (C.RESULTS / "omni_wuji_fingersfree_test", "_original", None),
         "umr": (C.RESULTS / "umr_wuji", "", C.UMR_ROBOT_XML["wuji"]), "umr_fix": (C.RESULTS / "umr_wuji_nofingercol", "", C.UMR_ROBOT_XML["wuji"])}
out = {"human_wrist_chair_R": pq.on_surface(np.einsum("tji,tj->ti", oR, J[:, 21] - op))[1], "human_wrist_chair_L": pq.on_surface(np.einsum("tji,tj->ti", oR, J[:, 20] - op))[1]}
for lane, (rdir, suf, xml) in lanes.items():
    r = np.load(rdir / f"{seq}{suf}.npz", allow_pickle=True); is_omni = xml is None
    m = C.omni_model("wuji", "umr", C.omni_subject(r, e["subject"]), obj) if is_omni else C.augmented_model(xml, obj, s)
    q = r["qpos"][:, :36] if is_omni else r["qpos"]; Tq = min(len(q), T); d = mujoco.MjData(m)
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]; ob = bn.index(f"{obj}_link")
    is_obj = np.array([m.geom_bodyid[g] == ob for g in range(m.ngeom)])
    for side, k in (("R", "right"), ("L", "left")):
        wb = bn.index(f"{k}_wrist_yaw_link"); is_hand = np.array([bn[m.geom_bodyid[g]].startswith((f"{k[0]}_", f"{k}_wrist")) and (m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
        gap = np.zeros(Tq); pen = np.zeros(Tq); dist = np.zeros(Tq)
        for t in range(Tq):
            d.qpos[:q.shape[1]] = q[t]
            if is_omni: d.qpos[36:43] = r["qpos"][t, 36:43]
            else:
                a0 = int(m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_bodyid[j] == ob][0]]); d.qpos[a0:a0+3] = op[t] * s; d.qpos[a0+3:a0+7] = src["obj_quat_xyzw"][t][[3, 0, 1, 2]]
            mujoco.mj_forward(m, d); mujoco.mj_collision(m, d)
            gap[t] = np.linalg.norm(d.xpos[wb] / s - J[t, C.SX[f"{side}_Wrist"]])
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist < 0 and ((is_obj[c.geom1] and is_hand[c.geom2]) or (is_obj[c.geom2] and is_hand[c.geom1])): pen[t] = max(pen[t], -c.dist)
            dist[t] = pq.on_surface(np.einsum("ji,j->i", oR[t], d.xpos[wb] / s - op[t])[None])[1][0]
        out[f"{lane}_gap_{side}"] = gap; out[f"{lane}_pen_{side}"] = pen; out[f"{lane}_wristchair_{side}"] = dist
    C._model_cache.clear()
np.savez("results/figures/whitechair_series.npz", **out)
for side in ("R", "L"):
    print(side, "frames 8..20 wrist gap (cm):")
    for lane in lanes: print(f"   {lane:9s}", " ".join(f"{100*out[f'{lane}_gap_{side}'][t]:4.0f}" for t in range(8, 21)), " | human wrist->chair:", " ".join(f"{100*out[f'human_wrist_chair_{side}'][t]:4.0f}" for t in range(8, 21)))
EOF

# ---------------- sweep
export BENCH_DATASET=omomo MUJOCO_GL=egl OMP_NUM_THREADS=1 && timeout 580 /home/kaizhao/software/miniconda3/envs/umr/bin/python - <<'EOF' 2>&1 | grep -v Warning
import numpy as np, mujoco, sys, json, trimesh, tempfile
import xml.etree.ElementTree as ET
sys.path.insert(0, "."); import compare as C
sys.path.insert(0, "../hand_bench"); from humoto_io import hand_frame
from scipy.spatial.transform import Rotation as Rot
seq = "sub16_whitechair_002"; e = next(x for x in json.load(open(C.DATA / "seqs_test.json")) if x["seq"] == seq)
src = np.load(C.DATA / "source" / f"{seq}.npz", allow_pickle=True); obj = str(src["object"]); J = src["joints55"]; s = 1.32 / float(src["tpose_height"]); T = len(J)
meta = json.loads((C.DATA / "objects" / obj / "meta.json").read_text()); oq = src["obj_quat_xyzw"][:, [3, 0, 1, 2]]; op = src["obj_pos"]
urdf = str(C.HERE.parent / "urdf" / "g1_wuji_hand2" / "hand_right.urdf"); hm = mujoco.MjModel.from_xml_path(urdf)
with tempfile.TemporaryDirectory() as td:
    xml = f"{td}/hand.xml"; mujoco.mj_saveLastXML(xml, hm); root = ET.parse(xml).getroot()
    comp = root.find("compiler")
    if comp is not None and comp.get("meshdir") and not comp.get("meshdir").startswith("/"): comp.set("meshdir", str((C.HERE.parent / "urdf" / "g1_wuji_hand2" / comp.get("meshdir")).resolve()))
    wb = root.find("worldbody"); kids = [c for c in list(wb) if c.tag == "body"]
    fr = ET.SubElement(wb, "body", name="hand_root"); ET.SubElement(fr, "freejoint", name="hand_free")
    for k in kids: wb.remove(k); fr.append(k)
    asset = root.find("asset") or ET.SubElement(root, "asset"); sc = f"{s:.6f} {s:.6f} {s:.6f}"
    ob = ET.SubElement(wb, "body", name=f"{obj}_link"); ET.SubElement(ob, "freejoint")
    for i in range(meta["n_pieces"]):
        ET.SubElement(asset, "mesh", name=f"{obj}_c{i:02d}", file=str((C.DATA / "objects" / obj / f"{obj}_coacd_{i:02d}.obj").resolve()), scale=sc)
        ET.SubElement(ob, "geom", name=f"{obj}_c{i:02d}", type="mesh", mesh=f"{obj}_c{i:02d}", contype="1", conaffinity="1")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
d = mujoco.MjData(m); bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
ob_id = bn.index(f"{obj}_link"); obj_qadr = int(m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_bodyid[j] == ob_id][0]])
is_obj = np.array([m.geom_bodyid[g] == ob_id for g in range(m.ngeom)]); is_finger = np.array([bn[m.geom_bodyid[g]].startswith(("r_thumb", "r_index", "r_middle", "r_ring", "r_pinky")) for g in range(m.ngeom)])
m.geom_contype[~is_obj] = 1; m.geom_conaffinity[~is_obj] = 1
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
def rest_q(curl):
    q = np.zeros(m.nq); q[3] = 1
    for j, n in enumerate(jn):
        if not n or n == "hand_free" or n.startswith(obj) or not curl: continue
        a = int(m.jnt_qposadr[j]); sh = n[2:]; q[a] = {"thumb_cmc_flex": .2, "thumb_mcp": .2, "thumb_ip": .2}.get(sh, .3 if sh.endswith("mcp_flex") else .4 if sh.endswith("pip") else .2 if sh.endswith("dip") else 0.0)
    return q
off = np.array([0.0415, 0, 0])
def pen(t, shift, curl, fingers_on=True):
    kp21 = np.zeros((21, 3)); kp21[0] = J[t, 21]; kp21[9] = J[t, 43]; kp21[5] = J[t, 40]; kp21[17] = J[t, 46]; R = hand_frame(kp21)
    q = rest_q(curl); q[:3] = J[t, 21] * s - R @ off + R @ shift; q[3:7] = Rot.from_matrix(R).as_quat()[[3, 0, 1, 2]]; q[obj_qadr:obj_qadr+3] = op[t] * s; q[obj_qadr+3:obj_qadr+7] = oq[t]
    d.qpos[:] = q; mujoco.mj_forward(m, d); mujoco.mj_collision(m, d); pf = pp = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= 0: continue
        g = c.geom1 if is_obj[c.geom2] else c.geom2 if is_obj[c.geom1] else -1
        if g < 0: continue
        if is_finger[g]: pf = max(pf, -c.dist)
        else: pp = max(pp, -c.dist)
    return pf, pp
# sweep: shift the whole hand along its own x (distal) axis and along y (palm normal), curled vs straight fingers, frames 32/65/98
shifts = np.arange(-0.16, 0.161, 0.01)
out = {}
for t in (32, 65, 98):
    for axis, name in ((0, "x_distal"), (1, "y_palm")):
        for curl in (True, False):
            rows = []
            for sh in shifts:
                v = np.zeros(3); v[axis] = sh; pf, pp = pen(t, v, curl); rows.append((sh, pf, pp))
            out[f"t{t}_{name}_{'curl' if curl else 'straight'}"] = np.array(rows)
np.savez("results/figures/whitechair_shift_sweep.npz", shifts=shifts, **out)
for t in (32, 65, 98):
    for name in ("x_distal", "y_palm"):
        for curl in ("curl", "straight"):
            r = out[f"t{t}_{name}_{curl}"]; free = (r[:, 1] < 0.002) & (r[:, 2] < 0.002); free_palm_only = r[:, 2] < 0.002
            fw = [f"{100*sh:+.0f}" for sh, f in zip(shifts, free) if f]
            print(f"frame {t} shift along {name:8s} fingers {curl:8s}: collision-free shifts (cm): {' '.join(fw) if fw else 'none'} | ignoring fingers: {' '.join(f'{100*sh:+.0f}' for sh, f in zip(shifts, free_palm_only) if f)}")
EOF
tail -1 results/figures/video_right.log; tail -1 results/figures/video_left.log 2>/dev/null; ls results/figures/*report*.mp4 2>/dev/null

# ---------------- tp
export BENCH_DATASET=omomo MUJOCO_GL=egl OMP_NUM_THREADS=1 && timeout 580 /home/kaizhao/software/miniconda3/envs/umr/bin/python - <<'EOF' 2>&1 | grep -v Warning
import numpy as np, mujoco, sys, json, trimesh, tempfile
import xml.etree.ElementTree as ET
sys.path.insert(0, "."); import compare as C
sys.path.insert(0, "../hand_bench"); from humoto_io import hand_frame
from scipy.spatial.transform import Rotation as Rot
seq = "sub16_whitechair_002"; e = next(x for x in json.load(open(C.DATA / "seqs_test.json")) if x["seq"] == seq)
src = np.load(C.DATA / "source" / f"{seq}.npz", allow_pickle=True); obj = str(src["object"]); J = src["joints55"]; s = 1.32 / float(src["tpose_height"]); T = len(J)
meta = json.loads((C.DATA / "objects" / obj / "meta.json").read_text())
oq = src["obj_quat_xyzw"][:, [3, 0, 1, 2]]; op = src["obj_pos"]
# ---- hand-only model (Wuji right) + chair pieces, hand root free: penetration of the REST-POSE hand placed at the human wrist pose
urdf = str(C.HERE.parent / "urdf" / "g1_wuji_hand2" / "hand_right.urdf")
hm = mujoco.MjModel.from_xml_path(urdf)
with tempfile.TemporaryDirectory() as td:
    xml = f"{td}/hand.xml"; mujoco.mj_saveLastXML(xml, hm)
    root = ET.parse(xml).getroot()
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if f and not f.startswith("/"): mesh.set("file", str((C.HERE.parent / "urdf" / "g1_wuji_hand2" / f).resolve()) if not (C.HERE.parent / "urdf" / "g1_wuji_hand2" / "meshes" / f).exists() else str((C.HERE.parent / "urdf" / "g1_wuji_hand2" / "meshes" / f).resolve()))
    comp = root.find("compiler"); 
    if comp is not None and comp.get("meshdir"): comp.set("meshdir", str((C.HERE.parent / "urdf" / "g1_wuji_hand2" / comp.get("meshdir")).resolve()) if not comp.get("meshdir").startswith("/") else comp.get("meshdir"))
    wb = root.find("worldbody"); kids = [c for c in list(wb) if c.tag == "body"]
    fr = ET.SubElement(wb, "body", name="hand_root"); ET.SubElement(fr, "freejoint", name="hand_free")
    for k in kids: wb.remove(k); fr.append(k)
    asset = root.find("asset") or ET.SubElement(root, "asset"); sc = f"{s:.6f} {s:.6f} {s:.6f}"
    ob = ET.SubElement(wb, "body", name=f"{obj}_link"); ET.SubElement(ob, "freejoint")
    for i in range(meta["n_pieces"]):
        ET.SubElement(asset, "mesh", name=f"{obj}_c{i:02d}", file=str((C.DATA / "objects" / obj / f"{obj}_coacd_{i:02d}.obj").resolve()), scale=sc)
        ET.SubElement(ob, "geom", name=f"{obj}_c{i:02d}", type="mesh", mesh=f"{obj}_c{i:02d}", contype="1", conaffinity="1")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
d = mujoco.MjData(m)
bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
ob_id = bn.index(f"{obj}_link"); obj_qadr = int(m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_bodyid[j] == ob_id][0]])
is_obj = np.array([m.geom_bodyid[g] == ob_id for g in range(m.ngeom)])
finger_pref = ("r_thumb", "r_index", "r_middle", "r_ring", "r_pinky")
is_finger = np.array([bn[m.geom_bodyid[g]].startswith(finger_pref) for g in range(m.ngeom)])
is_palm = np.array([(not is_obj[g]) and (not is_finger[g]) and (m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
m.geom_contype[~is_obj] = 1; m.geom_conaffinity[~is_obj] = 1  # make sure hand geoms collide for the distance test
# rest pose joint values (hand_rest_pose.json rules for Wuji)
jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
def rest_q(curl=True):
    q = np.zeros(m.nq); q[3] = 1; 
    for j, n in enumerate(jn):
        if n in ("hand_free", None) or n.startswith(f"{obj}"): continue
        if not curl: continue
        a = int(m.jnt_qposadr[j]); sh = n[2:] if n[:2] in ("r_", "l_") else n
        q[a] = {"thumb_cmc_flex": .2, "thumb_mcp": .2, "thumb_ip": .2}.get(sh, .3 if sh.endswith("mcp_flex") else .4 if sh.endswith("pip") else .2 if sh.endswith("dip") else 0.0)
    return q
# the hand root in the hand-only URDF: find offset of r_mount from root so that the flange lands on the human wrist
mujoco.mj_kinematics(m, d); root_id = bn.index("hand_root"); mount_id = bn.index("r_mount")
d.qpos[:] = rest_q(True); mujoco.mj_kinematics(m, d); off = d.xmat[root_id].reshape(3,3).T @ (d.xpos[mount_id] - d.xpos[root_id]); print("mount offset from hand root (m):", off.round(4))
def pen_at(t, curl=True, straight_test=False):
    kp21 = np.zeros((21, 3)); kp21[0] = J[t, 21]; kp21[9] = J[t, 43]; kp21[5] = J[t, 40]; kp21[17] = J[t, 46]
    R = hand_frame(kp21); p = J[t, 21] * s - R @ off
    q = rest_q(curl); q[:3] = p; q[3:7] = Rot.from_matrix(R).as_quat()[[3, 0, 1, 2]]; q[obj_qadr:obj_qadr+3] = op[t] * s; q[obj_qadr+3:obj_qadr+7] = oq[t]
    d.qpos[:] = q; mujoco.mj_forward(m, d); mujoco.mj_collision(m, d)
    pf = pp = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist >= 0: continue
        g = c.geom1 if is_obj[c.geom2] else c.geom2 if is_obj[c.geom1] else -1
        if g < 0: continue
        if is_finger[g]: pf = max(pf, -c.dist)
        elif is_palm[g]: pp = max(pp, -c.dist)
    return pf, pp
res = np.array([[t, *pen_at(t, True), *pen_at(t, False)] for t in range(T)])  # t, finger pen (rest), palm pen (rest), finger pen (straight), palm pen (straight)
mesh = trimesh.load(C.DATA / "objects" / obj / f"{obj}.obj", force="mesh"); pq = trimesh.proximity.ProximityQuery(mesh)
oRm = Rot.from_quat(src["obj_quat_xyzw"]).as_matrix()
dh = pq.on_surface(np.einsum("tji,tj->ti", oRm, J[:, 21] - op))[1]
near = dh < 0.10
print(f"right hand, frames with human wrist within 10 cm of chair: {near.sum()}")
print(f"  rest-pose (curled) hand at the human wrist pose: finger penetration > 5 mm in {(res[near,1] > 0.005).sum()} frames, mean {1000*res[near,1].mean():.1f} mm, max {1000*res[near,1].max():.1f} mm; palm penetration > 5 mm in {(res[near,2] > 0.005).sum()} frames")
print(f"  same with STRAIGHT fingers: finger pen > 5 mm in {(res[near,3] > 0.005).sum()} frames, mean {1000*res[near,3].mean():.1f} mm")
print("  per frame (t, finger pen mm rest / straight):", [(int(t), round(1000*a,1), round(1000*b,1)) for t, a, _, b, _ in res[[13, 20, 32, 50, 65, 80, 98, 110]]])
np.savez("results/figures/whitechair_target_pose_penetration.npz", res=res, human_wrist_chair=dh)
EOF
tail -1 results/figures/video_right.log; ls -la results/figures/*report*.mp4 2>/dev/null | cut -c30-

# ---------------- plot
/home/kaizhao/software/miniconda3/envs/handbench/bin/python - <<'EOF'
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = np.load("results/figures/whitechair_series.npz"); SW = np.load("results/figures/whitechair_shift_sweep.npz"); TP = np.load("results/figures/whitechair_target_pose_penetration.npz")
BODY, HAND, INK2, SURF = "#2a78d6", "#eb6834", "#52514e", "#fcfcfb"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), dpi=170, facecolor=SURF); t = np.arange(len(S["omni_gap_R"])) / 30.0
a = ax[0]; a.plot(t, 100 * S["human_wrist_chair_R"], color=INK2, lw=1.2, ls="--", label="human wrist to chair surface")
a.plot(t, 100 * S["umr_gap_R"], color=BODY, lw=2, label="UMR, fingers collide (original)"); a.plot(t, 100 * S["umr_fix_gap_R"], color=BODY, lw=2, ls=":", label="UMR, finger collision off")
a.plot(t, 100 * S["omni_gap_R"], color=HAND, lw=2, label="OmniRetarget, fingers constrained (original)"); a.plot(t, 100 * S["omni_fix_gap_R"], color=HAND, lw=2, ls=":", label="OmniRetarget, fingers free")
a.axvspan(8 / 30, 20 / 30, color="#e8e8e8", lw=0, zorder=0); a.text(14 / 30, 24, "approach", ha="center", fontsize=9, color=INK2)
a.set_xlabel("time (s)"); a.set_ylabel("cm"); a.set_title("Right wrist: robot-to-human wrist gap", fontsize=11, loc="left"); a.set_ylim(0, 26); a.legend(fontsize=7.5, frameon=False, loc="upper right")
res = TP["res"]; a = ax[1]; a.plot(t, 1000 * res[:, 2], color=BODY, lw=2, label="palm block"); a.plot(t, 1000 * res[:, 1], color=HAND, lw=2, label="curled fingers")
a.set_xlabel("time (s)"); a.set_ylabel("penetration (mm)"); a.set_title("Rest-pose Wuji hand placed exactly at the human wrist pose", fontsize=11, loc="left"); a.legend(fontsize=8, frameon=False)
a = ax[2]; sh = 100 * SW["shifts"]
for key, col, lab in (("t65_x_distal_curl", HAND, "curled fingers (rest pose)"), ("t65_x_distal_straight", "#8a5a2b", "straight fingers")):
    r = SW[key]; a.plot(sh, 1000 * np.maximum(r[:, 1], r[:, 2]), color=col, lw=2, label=lab)
r = SW["t65_x_distal_curl"]; a.plot(sh, 1000 * r[:, 2], color=BODY, lw=2, ls=":", label="palm block only (fingers ignored)")
a.axhline(2, color=INK2, lw=0.8, ls="--"); a.axvline(0, color=INK2, lw=0.8); a.set_ylim(0, 60)
a.set_xlabel("hand shift along its distal axis (cm)\n0 = human wrist pose, negative = retracted towards the forearm"); a.set_ylabel("max hand penetration (mm)")
a.set_title("Frame 65: collision-free placements along the hand axis", fontsize=11, loc="left"); a.legend(fontsize=7.5, frameon=False)
for x in ax: x.set_facecolor(SURF); [s_.set_visible(False) for s_ in (x.spines["top"], x.spines["right"])]
fig.suptitle("sub16_whitechair_002: why the fixed-finger hand ends up on the far side of the chair back", x=0.01, ha="left", fontsize=13, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("results/figures/sub16_whitechair_002_analysis.png", facecolor=SURF); print("plot ok")
EOF
echo "plot ok"
