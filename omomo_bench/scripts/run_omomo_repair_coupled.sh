#!/bin/bash
# OMOMO two-stage, coupled variant: UMR body with finger collision off -> coupled arm+finger repair with full collision
cd /home/kaizhao/projects/retarget/omomo_bench
export BENCH_DATASET=omomo MUJOCO_GL=egl OMP_NUM_THREADS=1
PY=/home/kaizhao/software/miniconda3/envs/umr/bin/python
R=results
$PY omomo_hand_repair.py --coupled --iters 8 --body-lane umr_wuji_nofingercol --out pipe_handrepair_coupled --workers 8 2>&1 | tee $R/handrepair_coupled_run.log | tail -1
$PY wrist_gap_stats.py --out $R/wrist_gap_omomo_coupled.csv --lane umr_wuji=$R/umr_wuji --lane coupled=$R/pipe_handrepair_coupled 2>&1 | grep -v Warning | tee $R/wrist_gap_omomo_coupled.log
/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python compare.py --out _handrepair_coupled --lane umr_wuji=$R/umr_wuji --lane coupled=$R/pipe_handrepair_coupled 2>&1 | tail -3
echo OMOMO_COUPLED_DONE
