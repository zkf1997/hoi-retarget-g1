#!/usr/bin/env bash
# Fetch the vendor description packages used by build_g1_hand_urdfs.py into urdf/third_party/ (git-ignored).
# Commits are pinned to what the shipped URDFs were built from; bump them deliberately and rebuild.
set -euo pipefail
TP="$(cd "$(dirname "$0")/.." && pwd)/third_party"
mkdir -p "$TP"

UNITREE_ROS_COMMIT=ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee   # unitreerobotics/unitree_ros  2026-09-16
REVO2_COMMIT=92cc697c7fa691db59404ce52344f3969a5ef7a6         # BrainCoTech/revo2_description  2026-08-20
WUJI_COMMIT=c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95          # wuji-technology/wuji-description  2026-09-11

clone_pinned() {  # url dir commit [sparse paths...]
  local url=$1 dir=$2 commit=$3; shift 3
  if [ ! -d "$dir/.git" ]; then
    if [ $# -gt 0 ]; then
      git clone --filter=blob:none --sparse "$url" "$dir"
      git -C "$dir" sparse-checkout set "$@"
    else
      git clone "$url" "$dir"
    fi
  fi
  git -C "$dir" fetch -q origin "$commit" 2>/dev/null || git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q "$commit"
  echo "$dir @ $(git -C "$dir" rev-parse --short HEAD)"
}

# unitree_ros is large; only the G1 description folders are needed
clone_pinned https://github.com/unitreerobotics/unitree_ros.git "$TP/unitree_ros" "$UNITREE_ROS_COMMIT" \
  robots/g1_description robots/g1_with_brainco_hand
clone_pinned https://github.com/BrainCoTech/revo2_description.git "$TP/revo2_description" "$REVO2_COMMIT"
clone_pinned https://github.com/wuji-technology/wuji-description.git "$TP/wuji-description" "$WUJI_COMMIT"
