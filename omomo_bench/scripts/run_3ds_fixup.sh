#!/bin/bash
cd /home/kaizhao/projects/retarget/omomo_bench
export MUJOCO_GL=egl OMP_NUM_THREADS=1
PY=/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python
for ds in omomo humoto hiphi; do
  export BENCH_DATASET=$ds; R=results; [ $ds != omomo ] && R=results_$ds
  L="--lane omni_wuji=$R/omni_wuji_umrscale:_original --lane omni_wuji_native=$R/omni_wuji_umrscale_native:_original --lane umr_wuji=$R/umr_wuji"
  [ $ds != hiphi ] && L="$L --lane omni_brainco=$R/omni_brainco_umrscale:_original --lane umr_brainco=$R/umr_brainco"
  $PY compare.py --merge $R/metrics_3ds_part?.csv --out _3ds $L 2>&1 | tail -1
  $PY wrist_orientation_stats.py $L > $R/wrist_orientation_3ds.log 2>&1 &
done
wait; echo FIXUP_DONE
