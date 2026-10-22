#!/bin/bash
# v3 (12 iterations) + UMR-style dense contact term, two weights, then evaluation against the chosen pipeline
set -u
DS=$1
cd /home/kaizhao/projects/retarget/omomo_bench
export BENCH_DATASET=$DS MUJOCO_GL=egl OMP_NUM_THREADS=1
PY=/home/kaizhao/software/miniconda3/envs/umr/bin/python
R=results_$DS
for W in 50 150; do
  $PY hand_pipeline.py --stage v3 --v3-iters 12 --w-dense $W --out-suffix _dense$W --workers 8 2>&1 | tee $R/pipe_run_dense$W.log | tail -1
done
$PY eval_pipeline.py --workers 8 --out _dense --lane final=$R/pipe_final --lane dense50=$R/pipe_v3_dense50 --lane dense150=$R/pipe_v3_dense150 2>&1 | tee $R/pipe_eval_dense.log | tail -1
echo DENSE_${DS}_DONE
