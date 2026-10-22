#!/bin/bash
cd /home/kaizhao/projects/retarget/omomo_bench
export BENCH_DATASET=omomo MUJOCO_GL=egl OMP_NUM_THREADS=1
R=results
/home/kaizhao/software/miniconda3/envs/umr/bin/python wrist_gap_stats.py --out $R/wrist_gap_omomo.csv --lane omni_wuji=$R/omni_wuji_umrscale:_original --lane umr_wuji=$R/umr_wuji --lane omni_brainco=$R/omni_brainco_umrscale:_original --lane umr_brainco=$R/umr_brainco 2>&1 | grep -v Warning | tee $R/wrist_gap_omomo.log
echo WRISTGAP_DONE
