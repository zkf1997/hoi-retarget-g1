#!/bin/bash
# Self-check of the handover deliverable: paths relocated, data + final lanes present, result npz readable with the robot MJCF,
# viewer starts and renders, one PNG renders.  Needs the hsretargeting env (viewer/render) and, if present, the result archive
# extracted over the clone.   usage: bash omomo_bench/scripts/check_handover.sh [port]     (exit code 0 = all checks passed)
set -u
PORT=${1:-8197}
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; OB=$ROOT/omomo_bench
PY=${HS_PYTHON:-${CONDA_ROOT:-$HOME/software/miniconda3}/envs/hsretargeting/bin/python}
[ -x "$PY" ] || PY=$(command -v python)
fail=0; ok() { echo "  [ok]   $1"; }; bad() { echo "  [FAIL] $1"; fail=1; }
echo "== handover check in $ROOT (python: $PY)"

echo "-- 1. absolute paths"
ORIG="/home/kaizhao/projects/""retarget"   # split so that relocate.sh does not rewrite this literal
if grep -rl --include='*.xml' --include='*.json' "$ORIG" "$OB/robots" "$OB/configs" 2>/dev/null | grep -q . && [ "$ROOT" != "$ORIG" ]; then
  bad "robots/configs still point to $ORIG -> run bash omomo_bench/relocate.sh"; else ok "robot MJCF / configs point to this clone"; fi

echo "-- 2. data and final lanes"
for sfx in "" _humoto _hiphi; do
  D=$OB/data$sfx; R=$OB/results$sfx
  if [ -f "$D/seqs_test.json" ] && [ -d "$D/source" ] && [ -d "$D/objects" ]; then ok "data$sfx: index, source/, objects/"; else bad "data$sfx missing (extract the result archive over the clone)"; continue; fi
  n=$(ls "$R/pipe_final" 2>/dev/null | grep -c '\.npz$'); m=$($PY -c "import json;print(len(json.load(open('$D/seqs_test.json'))))")
  [ "$n" = "$m" ] && ok "results$sfx/pipe_final: $n/$m sequences" || bad "results$sfx/pipe_final has $n of $m sequences"
done

echo "-- 3. result npz readable with the MJCF (forward kinematics, joints mapped by name)"
$PY - <<PYEOF || fail=1
import json, sys, numpy as np, mujoco
ob = "$OB"; bad = 0
for sfx in ("", "_humoto", "_hiphi"):
    try:
        idx = json.load(open(f"{ob}/data{sfx}/seqs_test.json")); seq = idx[0]["seq"]
        r = np.load(f"{ob}/results{sfx}/pipe_final/{seq}.npz", allow_pickle=True); src = np.load(f"{ob}/data{sfx}/source/{seq}.npz", allow_pickle=True)
        m = mujoco.MjModel.from_xml_path(f"{ob}/robots/g1_29dof_wuji_hand2.xml"); d = mujoco.MjData(m)
        names = [str(x) for x in r["robot_joint_names"]]; adr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): m.jnt_qposadr[j] for j in range(m.njnt)}
        assert r["qpos"].shape == (len(r["frame_ids"]), 76) and len(names) == 69 and all(n in adr for n in names)
        for n, v in zip(names, r["qpos"][0, 7:]): d.qpos[adr[n]] = v
        mujoco.mj_kinematics(m, d); s = float(np.asarray(r["smpl_scale"]).reshape(-1)[0]); assert abs(s - 1.32 / float(src["tpose_height"])) < 1e-3
        print(f"  [ok]   data{sfx}: {seq[:40]} T={len(r['qpos'])} s={s:.3f} stage={r['stage']}")
    except Exception as ex:
        print(f"  [FAIL] data{sfx}: {type(ex).__name__}: {ex}"); bad = 1
sys.exit(bad)
PYEOF

echo "-- 4. viewer starts (port $PORT, 60 s)"
LOG=/tmp/check_handover_viser.log
(cd "$OB" && BENCH_DATASET=humoto timeout 60 "$PY" -u viser_compare.py --port "$PORT" --all-objects --lane final=results_humoto/pipe_final > "$LOG" 2>&1; true)
if grep -q "viser ready" "$LOG" && grep -q "heartbeat" "$LOG"; then ok "viser ready + heartbeat ($(grep -c heartbeat "$LOG") beats; $(grep -o 'SMPL-X unavailable' "$LOG" | head -1 | sed 's/.*/human shown as joints/'))"
else bad "viewer did not start, see $LOG"; tail -5 "$LOG"; fi

echo "-- 5. offscreen render (MuJoCo EGL)"
seq=$($PY -c "import json;print(json.load(open('$OB/data_humoto/seqs_test.json'))[0]['seq'])" 2>/dev/null)
if [ -n "$seq" ] && (cd "$OB" && MUJOCO_GL=egl BENCH_DATASET=humoto timeout 300 "$PY" render_compare.py --seqs "$seq" --frames 2 --lane final=results_humoto/pipe_final --out /tmp/check_handover_render > /tmp/check_handover_render.log 2>&1) && ls /tmp/check_handover_render/*.png >/dev/null 2>&1; then
  ok "render_compare wrote $(ls /tmp/check_handover_render/*.png | head -1)"; else bad "render failed, see /tmp/check_handover_render.log"; fi

[ $fail = 0 ] && echo "== ALL CHECKS PASSED" || echo "== SOME CHECKS FAILED"; exit $fail
