#!/bin/bash
# Full-set validation of the body stage with finger collision OFF: UMR (nofingercol) -> AnyTeleop -> v3 (12 it), same hand-stage
# settings as the current final of each dataset (HUMOTO: multi-object step 2; HiPHI: single object), evaluated against pipe_final.
set -u
DS=$1
cd /home/kaizhao/projects/retarget/omomo_bench
export BENCH_DATASET=$DS MUJOCO_GL=egl OMP_NUM_THREADS=1
PY=/home/kaizhao/software/miniconda3/envs/umr/bin/python
R=results_$DS
MULTI=""; [ -f data_$DS/multi_objects.json ] && MULTI="--multi"
EXTRA=""; [ "$DS" = "hiphi" ] && EXTRA="--no-slot-reuse"
$PY run_umr.py --hand wuji --variant nofingercol --workers 3 $EXTRA 2>&1 | tee $R/umr_wuji_nofingercol_full_run.log | tail -1
$PY hand_pipeline.py --stage s2 v3 --v3-iters 12 $MULTI --body-lane umr_wuji_nofingercol --out-suffix _nfcfull --workers 8 2>&1 | tee $R/pipe_run_nfcfull.log | tail -1
$PY eval_pipeline.py --workers 8 $MULTI --out _nfcfull --lane umr_body=$R/umr_wuji --lane umr_body_nfc=$R/umr_wuji_nofingercol --lane final=$R/pipe_final --lane final_nfc=$R/pipe_v3_nfcfull 2>&1 | tee $R/pipe_eval_nfcfull.log | tail -1
echo NFCFULL_${DS}_DONE
