#!/bin/bash
# OMOMO two-stage: UMR body with finger collision off -> finger repair with full collision; compare against the original umr_wuji
cd /home/kaizhao/projects/retarget/omomo_bench
export BENCH_DATASET=omomo MUJOCO_GL=egl OMP_NUM_THREADS=1
PY=/home/kaizhao/software/miniconda3/envs/umr/bin/python
R=results
$PY run_umr.py --hand wuji --variant nofingercol --workers 3 2>&1 | tee $R/umr_wuji_nofingercol_full_run.log | tail -1
$PY omomo_hand_repair.py --body-lane umr_wuji_nofingercol --out pipe_handrepair --workers 8 2>&1 | tee $R/handrepair_run.log | tail -1
$PY wrist_gap_stats.py --out $R/wrist_gap_omomo_repair.csv --lane umr_wuji=$R/umr_wuji --lane body_nofingercol=$R/umr_wuji_nofingercol --lane handrepair=$R/pipe_handrepair 2>&1 | grep -v Warning | tee $R/wrist_gap_omomo_repair.log
/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python compare.py --out _handrepair --lane umr_wuji=$R/umr_wuji --lane handrepair=$R/pipe_handrepair 2>&1 | tail -3
echo OMOMO_REPAIR_DONE
