#!/bin/bash
# coupled repair on the FULL-collision UMR body (for the "how much penetration does fingers-off add after all stages" question)
cd /home/kaizhao/projects/retarget/omomo_bench
export BENCH_DATASET=omomo MUJOCO_GL=egl OMP_NUM_THREADS=1
PY=/home/kaizhao/software/miniconda3/envs/umr/bin/python
HS=/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python
$PY omomo_hand_repair.py --coupled --iters 8 --body-lane umr_wuji --out pipe_handrepair_coupled_fullcol --workers 8 2>&1 | tee results/handrepair_coupled_fullcol_run.log | tail -1
L="--lane body_full=results/umr_wuji --lane body_nfc=results/umr_wuji_nofingercol --lane repair_on_full=results/pipe_handrepair_coupled_fullcol --lane repair_on_nfc=results/pipe_handrepair_coupled"
$HS wrist_gap_stats.py --out results/wrist_gap_repair_fullcol.csv $L 2>&1 | grep -v Warning | tee results/wrist_gap_repair_fullcol.log
for i in 0 1 2 3 4 5; do $HS compare.py --seqs $(cat results/.3ds_chunk$i) --out _repairfc_part$i --lane repair_on_full=results/pipe_handrepair_coupled_fullcol --lane repair_on_nfc=results/pipe_handrepair_coupled > results/compare_repairfc_part$i.log 2>&1 & done; wait
$HS compare.py --merge results/metrics_repairfc_part?.csv --out _repairfc --lane repair_on_full=results/pipe_handrepair_coupled_fullcol --lane repair_on_nfc=results/pipe_handrepair_coupled 2>&1 | tail -1
echo REPAIRFC_DONE
