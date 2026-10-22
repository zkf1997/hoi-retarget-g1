#!/bin/bash
# UMR finger-collision-off body stage vs full-collision UMR (and OmniRetarget CoACD / hull) on one dataset: usage: bash results/run_nfc_compare.sh <omomo|humoto|hiphi>
cd /home/kaizhao/projects/retarget/omomo_bench
export MUJOCO_GL=egl OMP_NUM_THREADS=1 BENCH_DATASET=$1
PY=/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python
R=results; [ $1 != omomo ] && R=results_$1
NFC="--lane umr_wuji_nofingercol=$R/umr_wuji_nofingercol"
ALL="--lane omni_wuji_hull=$R/omni_wuji_umrscale_native:_original --lane omni_wuji=$R/omni_wuji_umrscale:_original --lane umr_wuji=$R/umr_wuji $NFC"
for i in 0 1 2 3 4 5; do $PY compare.py --seqs $(cat $R/.3ds_chunk$i) --out _nfc_part$i $NFC > $R/compare_nfc_part$i.log 2>&1 & done; wait
# the hull lane was named omni_wuji_native in the 3ds csvs; rename on the fly
$PY - <<PYEOF
import csv, glob
rows = []
for f in sorted(glob.glob("$R/metrics_3ds_part?.csv")) + sorted(glob.glob("$R/metrics_nfc_part?.csv")):
    for r in csv.DictReader(open(f)):
        if r["method"] == "omni_wuji_native": r["method"] = "omni_wuji_hull"
        rows.append(r)
w = csv.DictWriter(open("$R/metrics_nfc_all.csv", "w", newline=""), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
PYEOF
$PY compare.py --merge $R/metrics_nfc_all.csv --out _nfc $ALL 2>&1 | tail -1
$PY wrist_gap_stats.py --out $R/wrist_gap_nfc.csv $ALL 2>&1 | grep -v Warning > $R/wrist_gap_nfc.log
$PY wrist_orientation_stats.py $ALL 2>&1 | grep -v Warning > $R/wrist_orientation_nfc.log
echo NFC_${1}_DONE
