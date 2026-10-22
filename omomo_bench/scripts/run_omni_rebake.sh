#!/bin/bash
# Re-run every OmniRetarget lane with the fixed hand bake (run_omni.py bake_hands, 2026-10-14), then re-evaluate and re-render.
# usage: bash results/run_omni_rebake.sh wuji | brainco | post
cd /home/kaizhao/projects/retarget/omomo_bench
export MUJOCO_GL=egl OMP_NUM_THREADS=1
PY=/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python
archive() { [ -d "$1" ] && [ ! -d "$1_oldbake" ] && mv "$1" "$1_oldbake"; return 0; }
case "$1" in
wuji)
  archive results/omni_wuji_umrscale; archive results/omni_wuji_umrscale_native; archive results/omni_wuji_fingersfree_test; archive results_humoto/omni_wuji_umrscale
  BENCH_DATASET=omomo  $PY run_omni.py --hand wuji --scaling umr --workers 10 >> results/omni_wuji_umrscale_full_run.log 2>&1
  BENCH_DATASET=humoto $PY run_omni.py --hand wuji --scaling umr --workers 8 >> results_humoto/omni_wuji_umrscale_full_run.log 2>&1
  NATIVE=$(ls results/omni_wuji_umrscale_native_oldbake/*_original.npz | xargs -n1 basename | sed 's/_original.npz//' | tr '\n' ' ')
  BENCH_DATASET=omomo $PY run_omni.py --hand wuji --scaling umr --native-collision --workers 8 --seqs $NATIVE > results/omni_wuji_native_run.log 2>&1
  BENCH_DATASET=omomo $PY run_omni.py --hand wuji --scaling umr --tag omni_wuji_fingersfree_test --workers 1 --seqs sub16_whitechair_002 --force -- --fingers-free > results/omni_wuji_fingersfree_test_run.log 2>&1
  touch results/.rebake_wuji_done; echo REBAKE_WUJI_DONE ;;
brainco)
  archive results/omni_brainco_umrscale; archive results_humoto/omni_brainco_umrscale
  BENCH_DATASET=omomo  $PY run_omni.py --hand brainco --scaling umr --workers 10 >> results/omni_brainco_umrscale_full_run.log 2>&1
  BENCH_DATASET=humoto $PY run_omni.py --hand brainco --scaling umr --workers 8 >> results_humoto/omni_brainco_umrscale_full_run.log 2>&1
  touch results/.rebake_brainco_done; echo REBAKE_BRAINCO_DONE ;;
post)
  while [ ! -f results/.rebake_wuji_done ] || [ ! -f results/.rebake_brainco_done ]; do sleep 60; done
  R=results
  for f in summary.md metrics.csv summary_native.md metrics_native.csv wrist_gap_omomo.csv wrist_gap_omomo.log; do [ -f $R/$f ] && [ ! -f $R/${f%.*}_oldbake.${f##*.} ] && cp $R/$f $R/${f%.*}_oldbake.${f##*.}; done
  for f in summary.md metrics.csv; do [ -f results_humoto/$f ] && [ ! -f results_humoto/${f%.*}_oldbake.${f##*.} ] && cp results_humoto/$f results_humoto/${f%.*}_oldbake.${f##*.}; done
  # OMOMO: 4 default lanes, 602 seqs, chunked 6-way then merged
  export BENCH_DATASET=omomo
  $PY - <<'PYEOF'
import json; s = [e["seq"] for e in json.load(open("data/seqs_test.json"))]; n = 6
for i in range(n): open(f"results/.rebake_chunk{i}", "w").write(" ".join(s[i::n]))
PYEOF
  for i in 0 1 2 3 4 5; do $PY compare.py --seqs $(cat results/.rebake_chunk$i) --out _rebake_part$i > results/compare_rebake_part$i.log 2>&1 & done; wait
  $PY compare.py --merge results/metrics_rebake_part?.csv --out "" 2>&1 | tail -2
  NATIVE=$(ls results/omni_wuji_umrscale_native/*_original.npz | xargs -n1 basename | sed 's/_original.npz//' | tr '\n' ' ')
  $PY compare.py --seqs $NATIVE --out _native --lane coacd=results/omni_wuji_umrscale:_original --lane native=results/omni_wuji_umrscale_native:_original 2>&1 | tail -2
  $PY wrist_gap_stats.py --out $R/wrist_gap_omomo.csv --lane omni_wuji=$R/omni_wuji_umrscale:_original --lane umr_wuji=$R/umr_wuji --lane omni_brainco=$R/omni_brainco_umrscale:_original --lane umr_brainco=$R/umr_brainco 2>&1 | grep -v Warning | tee $R/wrist_gap_omomo.log
  BENCH_DATASET=humoto $PY compare.py > results_humoto/compare_rebake.log 2>&1; tail -2 results_humoto/compare_rebake.log
  # white-chair report figures + videos (5 columns: source | UMR | UMR finger collision off | OmniRetarget | OmniRetarget fingers free)
  for side in right left; do
    $PY render_compare.py --seqs sub16_whitechair_002 --focus ${side}_hand --distance 0.6 --frames 6 --tag _report_$side --lane "UMR Wuji (original)=results/umr_wuji" --lane "UMR Wuji, finger collision off=results/umr_wuji_nofingercol" --lane "OmniRetarget Wuji (original)=results/omni_wuji_umrscale:_original" --lane "OmniRetarget Wuji, fingers free=results/omni_wuji_fingersfree_test:_original" 2>&1 | grep -v Warning | tail -1
    $PY render_compare.py --video --seqs sub16_whitechair_002 --focus ${side}_hand --distance 0.6 --tag _report_$side --lane "UMR Wuji (original)=results/umr_wuji" --lane "UMR Wuji, finger collision off=results/umr_wuji_nofingercol" --lane "OmniRetarget Wuji (original)=results/omni_wuji_umrscale:_original" --lane "OmniRetarget Wuji, fingers free=results/omni_wuji_fingersfree_test:_original" > results/figures/video_$side.log 2>&1; tail -1 results/figures/video_$side.log
  done
  echo REBAKE_ALL_DONE ;;
esac
