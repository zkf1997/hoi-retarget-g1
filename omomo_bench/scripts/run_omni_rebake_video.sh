#!/bin/bash
cd /home/kaizhao/projects/retarget/omomo_bench
export MUJOCO_GL=egl OMP_NUM_THREADS=1 BENCH_DATASET=omomo
PY=/home/kaizhao/software/miniconda3/envs/umr/bin/python
for side in right left; do
  $PY render_compare.py --video --seqs sub16_whitechair_002 --focus ${side}_hand --distance 0.6 --tag _report_$side --lane "UMR Wuji (original)=results/umr_wuji" --lane "UMR Wuji, finger collision off=results/umr_wuji_nofingercol" --lane "OmniRetarget Wuji (original)=results/omni_wuji_umrscale:_original" --lane "OmniRetarget Wuji, fingers free=results/omni_wuji_fingersfree_test:_original" > results/figures/video_$side.log 2>&1; tail -1 results/figures/video_$side.log
done
echo VIDEO_DONE
