#!/bin/bash
# fix-ups after run_omni_rebake.sh: native ablation needs its own tag; videos need the umr env (imageio ffmpeg plugin)
cd /home/kaizhao/projects/retarget/omomo_bench
export MUJOCO_GL=egl OMP_NUM_THREADS=1 BENCH_DATASET=omomo
PY=/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python
NATIVE=$(ls results/omni_wuji_umrscale_native_oldbake/*_original.npz | xargs -n1 basename | sed 's/_original.npz//' | tr '\n' ' ')
$PY run_omni.py --hand wuji --scaling umr --native-collision --tag omni_wuji_umrscale_native --workers 9 --seqs $NATIVE > results/omni_wuji_native_run.log 2>&1; tail -1 results/omni_wuji_native_run.log
$PY compare.py --seqs $NATIVE --out _native --lane coacd=results/omni_wuji_umrscale:_original --lane native=results/omni_wuji_umrscale_native:_original 2>&1 | tail -1
echo NATIVE_DONE
