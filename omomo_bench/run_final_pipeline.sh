#!/bin/bash
# Final pipeline (chosen 2026-10-07): scaled world (objects x s), UMR body with the default hand weights (fingers at rest),
# AnyTeleop fingers (hand_bench, must already exist), v3 coupled refinement with 12 iterations, metrics.
# Usage: bash run_final_pipeline.sh humoto|hiphi|omomo [umr_workers] [extra run_umr args, e.g. --no-slot-reuse for HiPHI]
# Output lanes: humoto pipe_v3_final_multi, hiphi pipe_v3_final_it12, omomo pipe_handrepair_coupled (-> results*/pipe_final).
set -u
DS=$1; W=${2:-3}; UMR_EXTRA=${3:-}
cd /home/kaizhao/projects/retarget/omomo_bench
export BENCH_DATASET=$DS MUJOCO_GL=egl OMP_NUM_THREADS=1
PY=/home/kaizhao/software/miniconda3/envs/umr/bin/python
R=results_$DS; [ "$DS" = omomo ] && R=results   # OMOMO results live in results/ (no suffix)
# body stage: HUMOTO uses the finger-collision-off variant (validated 2026-10-13: better on every metric); HiPHI keeps the
# default (finger collision off trades a little contact / penetration for smoothness there). Override with BODY_VARIANT=....
case "$DS" in humoto|omomo) BV_DEFAULT=nofingercol;; *) BV_DEFAULT="";; esac
BV=${BODY_VARIANT-$BV_DEFAULT}; BODY=umr_wuji${BV:+_$BV}
$PY run_umr.py --hand wuji --workers $W ${BV:+--variant $BV} $UMR_EXTRA 2>&1 | tee -a $R/${BODY}_full_run.log | tail -1
# multi-object hand stage when data_<ds>/multi_objects.json exists (prepare_humoto_multi.py); identical to single-object otherwise
if [ "$DS" = "omomo" ]; then
  # OMOMO has no finger data: no AnyTeleop / v3; instead the coupled arm+finger repair with full collision (omomo_hand_repair.py)
  $PY omomo_hand_repair.py --coupled --iters 8 --body-lane $BODY --out pipe_handrepair_coupled --workers 8 2>&1 | tee $R/handrepair_coupled_run.log | tail -1
  ln -sfn pipe_handrepair_coupled $R/pipe_final
  $PY wrist_gap_stats.py --out $R/wrist_gap_omomo_coupled.csv --lane body=$R/$BODY --lane final=$R/pipe_final 2>&1 | grep -v Warning | tail -3
  echo FINAL_${DS}_DONE; exit 0
fi
MULTI=""; [ -f data_$DS/multi_objects.json ] && MULTI="--multi"
$PY hand_pipeline.py --stage s2 v3 --v3-iters 12 $MULTI --body-lane $BODY --out-suffix _final_$( [ -n "$MULTI" ] && echo multi || echo it12 ) --workers 8 2>&1 | tee $R/pipe_run_final.log | tail -1
ln -sfn pipe_v3_final_$( [ -n "$MULTI" ] && echo multi || echo it12 ) $R/pipe_final
$PY eval_pipeline.py --workers 8 --out _it12 --lane body=$R/$BODY --lane s2=$R/pipe_s2_final_$( [ -n "$MULTI" ] && echo multi || echo it12 ) --lane final=$R/pipe_final $MULTI 2>&1 | tee $R/pipe_eval_it12.log | tail -1
echo FINAL_${DS}_DONE
