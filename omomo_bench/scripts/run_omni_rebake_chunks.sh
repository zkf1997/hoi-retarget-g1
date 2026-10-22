#!/bin/bash
# re-evaluate the two OMOMO compare chunks that died, then merge all six into results/summary.md
cd /home/kaizhao/projects/retarget/omomo_bench
export MUJOCO_GL=egl OMP_NUM_THREADS=1 BENCH_DATASET=omomo
PY=/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python
for i in 2 3; do $PY compare.py --seqs $(cat results/.rebake_chunk$i) --out _rebake_part$i > results/compare_rebake_part$i.log 2>&1 & done; wait
for i in 0 1 2 3 4 5; do printf "part%d rows %s\n" $i $(tail -n +2 results/metrics_rebake_part$i.csv | wc -l); done
$PY compare.py --merge results/metrics_rebake_part?.csv --out "" 2>&1 | tail -2
echo CHUNKS_DONE
