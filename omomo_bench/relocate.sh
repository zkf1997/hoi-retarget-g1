#!/bin/bash
# After cloning to a path other than /home/kaizhao/projects/retarget: rewrite the absolute paths baked into the robot MJCFs,
# the UMR configs and the driver scripts.  usage: bash omomo_bench/relocate.sh   (run from anywhere; idempotent)
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=/home/kaizhao/projects/retarget
[ "$ROOT" = "$OLD" ] && { echo "already at $OLD"; exit 0; }
grep -rl --include='*.xml' --include='*.json' --include='*.sh' --include='*.py' "$OLD" "$ROOT/omomo_bench" "$ROOT/hand_bench" 2>/dev/null \
  | grep -v '/results' | while read -r f; do sed -i "s#$OLD#$ROOT#g" "$f"; echo "relocated $f"; done
# conda env python paths used by the scripts
PYROOT=${CONDA_ROOT:-$HOME/software/miniconda3}
[ "$PYROOT" != /home/kaizhao/software/miniconda3 ] && grep -rl --include='*.sh' --include='*.py' /home/kaizhao/software/miniconda3 "$ROOT/omomo_bench" | grep -v '/results' | while read -r f; do sed -i "s#/home/kaizhao/software/miniconda3#$PYROOT#g" "$f"; done
echo done
