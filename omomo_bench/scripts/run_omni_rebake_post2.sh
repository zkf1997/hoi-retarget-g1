#!/bin/bash
# After run_omni_rebake.sh post: regenerate the white-chair analysis plot (uses the Omni lanes), the approach strip, and restart viser 8095.
cd /home/kaizhao/projects/retarget/omomo_bench
export MUJOCO_GL=egl OMP_NUM_THREADS=1 BENCH_DATASET=omomo
bash results/reports/whitechair_analysis.sh 2>&1 | grep -v Warning | tail -8
cd results/figures && for f in 8 11 14 17 20; do ffmpeg -loglevel error -y -i sub16_whitechair_002_report_right.mp4 -vf "select=eq(n\,$f)" -vframes 1 /tmp/wc_f$f.png; done && /home/kaizhao/software/miniconda3/envs/handbench/bin/python - <<'PYEOF'
from PIL import Image
ims = [Image.open(f"/tmp/wc_f{f}.png") for f in (8, 11, 14, 17, 20)]
W, H = ims[0].size; out = Image.new("RGB", (W, H * len(ims)), (255, 255, 255))
for i, im in enumerate(ims): out.paste(im, (0, i * H))
out.save("/home/kaizhao/projects/retarget/omomo_bench/results/figures/sub16_whitechair_002_approach_right.png"); print("approach strip", out.size)
PYEOF
cd /home/kaizhao/projects/retarget/omomo_bench
for p in $(ps -eo pid,args | grep "viser_compare.py" | grep "8095" | grep -v grep | awk '{print $1}'); do kill $p; done; sleep 2; tmux kill-window -t real_runs:viser8095 2>/dev/null
tmux new-window -d -t real_runs -n viser8095 "cd /home/kaizhao/projects/retarget/omomo_bench && export BENCH_DATASET=omomo && while true; do /home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python -u viser_compare.py --port 8095 --seqs sub16_whitechair_002 sub16_whitechair_009 sub17_woodchair_001 sub16_largebox_000 sub16_trashcan_000 --lane UMR_Wuji=results/umr_wuji --lane UMR_Wuji_finger_collision_off=results/umr_wuji_nofingercol --lane UMR_Wuji_two_stage_final=results/pipe_handrepair_coupled --lane OmniRetarget_Wuji=results/omni_wuji_umrscale:_original --lane OmniRetarget_Wuji_fingers_free=results/omni_wuji_fingersfree_test:_original >> /tmp/viser_pipe_omomo_wrongside.log 2>&1; sleep 5; done"
echo POST2_DONE
