#!/bin/bash
# ablation: OmniRetarget with holosoma's native object collision (one convex hull of the whole mesh) vs our CoACD pieces
cd /home/kaizhao/projects/retarget/omomo_bench
PY=/home/kaizhao/software/miniconda3/envs/hsretargeting/bin/python
$PY run_omni.py --hand wuji --scaling umr --native-collision --tag omni_wuji_umrscale_native --workers 9 --seqs sub16_largetable_000 sub16_largetable_001 sub16_largetable_002 sub17_smalltable_000 sub17_smalltable_001 sub17_smalltable_002 sub16_plasticbox_000 sub16_plasticbox_001 sub16_plasticbox_002 sub16_trashcan_000 sub16_trashcan_001 sub16_trashcan_002 sub17_woodchair_000 sub17_woodchair_001 sub17_woodchair_002 sub16_whitechair_000 sub16_whitechair_001 sub16_whitechair_002 2>&1 | tee results/omni_wuji_native_run.log | tail -3
echo NATIVE_DONE
