#!/bin/bash
# Final hand stages on top of an alternative UMR body variant.  Usage: bash run_variant_pipeline.sh <dataset> <variant> [umr_workers] [extra umr args]
set -u
DS=$1; V=$2; W=${3:-3}; UMR_EXTRA=${4:-}
cd /home/kaizhao/projects/retarget/omomo_bench
export BENCH_DATASET=$DS MUJOCO_GL=egl OMP_NUM_THREADS=1
PY=/home/kaizhao/software/miniconda3/envs/umr/bin/python
R=results_$DS
$PY run_umr.py --hand wuji --variant $V --workers $W $UMR_EXTRA 2>&1 | tee $R/umr_wuji_${V}_full_run.log | tail -1
$PY hand_pipeline.py --stage s2 v3 --body-lane umr_wuji_$V --v3-iters 12 --out-suffix _${V}final --workers 8 2>&1 | tee $R/pipe_run_${V}final.log | tail -1
$PY eval_pipeline.py --workers 8 --out _${V}final --lane umr_wuji_$V=$R/umr_wuji_$V --lane s2_${V}final=$R/pipe_s2_${V}final --lane v3_${V}final=$R/pipe_v3_${V}final 2>&1 | tee $R/pipe_eval_${V}final.log | tail -1
echo VARIANT_${DS}_${V}_DONE
