#!/usr/bin/env bash
# Conda environments used by omomo_bench (run once). Requires the miniconda at ~/software/miniconda3.
set -euo pipefail
source ~/software/miniconda3/etc/profile.d/conda.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- UMR (README recipe): python 3.12, torch 2.4.1 cu121, requirements-umr.txt
if ! conda env list | grep -q '^umr '; then
  conda create -n umr python=3.12 pip -y
  conda run -n umr python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
  conda run -n umr python -m pip install -r "$ROOT/UMR/requirements-umr.txt"
fi
# SMPL-X body models (license-gated, not in git): UMR looks in UMR/smpl/
for g in MALE FEMALE NEUTRAL; do
  ln -sfn /media/kaizhao/hdd/dataset/models_smplx_v1_1/models/smplx/SMPLX_$g.npz "$ROOT/UMR/smpl/SMPLX_$g.npz"
done

# --- holosoma_retargeting (OmniRetarget): python 3.11, editable install
if ! conda env list | grep -q '^hsretargeting '; then
  conda create -n hsretargeting python=3.11 -y
  conda run -n hsretargeting pip install -e "$ROOT/holosoma/src/holosoma_retargeting"
  conda run -n hsretargeting pip install smplx joblib matplotlib
fi
# IMPORTANT: mujoco >= 3.14 breaks holosoma_retargeting silently (all 29 joints stay at 0, only the root moves):
# in interaction_mesh_retargeter._build_transform_qdot_to_qvel_fast the test
#     jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)     # jt is a numpy.int32
# is False for the new pybind enums, so the hinge block of the qdot->qvel map is never filled. Pin an older mujoco:
conda run -n hsretargeting pip install "mujoco==3.3.5"
echo "envs ready: umr, hsretargeting"
