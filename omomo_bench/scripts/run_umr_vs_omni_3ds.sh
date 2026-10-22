#!/bin/bash
# UMR vs OmniRetarget on OMOMO / HUMOTO / HiPHI after the hand-bake fix. Phases: hiphi | native | post
cd /home/kaizhao/projects/retarget/omomo_bench
export MUJOCO_GL=egl OMP_NUM_THREADS=1
PY=/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python
case "$1" in
hiphi)   # OmniRetarget Wuji on HiPHI, CoACD (default) then native hull; per-sequence heights = UMR's per-sequence scale
  export BENCH_DATASET=hiphi
  $PY run_omni.py --hand wuji --scaling umr --index data_hiphi/seqs_test_perseq.json --workers 10 >> results_hiphi/omni_wuji_umrscale_full_run.log 2>&1
  $PY run_omni.py --hand wuji --scaling umr --index data_hiphi/seqs_test_perseq.json --native-collision --tag omni_wuji_umrscale_native --workers 10 >> results_hiphi/omni_wuji_native_run.log 2>&1
  touch results/.3ds_hiphi_done; echo HIPHI_DONE ;;
native)  # native-hull OmniRetarget Wuji on the full OMOMO and HUMOTO sets
  BENCH_DATASET=omomo  $PY run_omni.py --hand wuji --scaling umr --native-collision --tag omni_wuji_umrscale_native --workers 8 >> results/omni_wuji_native_run.log 2>&1
  BENCH_DATASET=humoto $PY run_omni.py --hand wuji --scaling umr --native-collision --tag omni_wuji_umrscale_native --workers 8 >> results_humoto/omni_wuji_native_run.log 2>&1
  touch results/.3ds_native_done; echo NATIVE_DONE ;;
post)
  while [ ! -f results/.3ds_hiphi_done ] || [ ! -f results/.3ds_native_done ]; do sleep 120; done
  for ds in omomo humoto hiphi; do
    export BENCH_DATASET=$ds; R=results; [ $ds != omomo ] && R=results_$ds
    L="--lane omni_wuji=$R/omni_wuji_umrscale:_original --lane omni_wuji_native=$R/omni_wuji_umrscale_native:_original --lane umr_wuji=$R/umr_wuji"
    [ $ds != hiphi ] && L="$L --lane omni_brainco=$R/omni_brainco_umrscale:_original --lane umr_brainco=$R/umr_brainco"
    DD=data; [ $ds != omomo ] && DD=data_$ds
    $PY -c "import json; s=[e['seq'] for e in json.load(open('$DD/seqs_test.json'))]; [open('$R/.3ds_chunk%d' % i, 'w').write(' '.join(s[i::6])) for i in range(6)]"
    for i in 0 1 2 3 4 5; do $PY compare.py --seqs $(cat $R/.3ds_chunk$i) --out _3ds_part$i $L > $R/compare_3ds_part$i.log 2>&1 & done; wait
    $PY compare.py --merge $R/metrics_3ds_part?.csv --out _3ds 2>&1 | tail -1
    $PY wrist_gap_stats.py --out $R/wrist_gap_3ds.csv $L 2>&1 | grep -v Warning > $R/wrist_gap_3ds.log
    $PY wrist_orientation_stats.py $L 2>&1 | grep -v Warning > $R/wrist_orientation_3ds.log
    echo "== $ds done"
  done
  echo ALL3DS_DONE ;;
esac
