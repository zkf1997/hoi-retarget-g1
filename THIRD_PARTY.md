# Third-party components

| component | location | origin | notes |
|---|---|---|---|
| UMR (Unified Motion Retargeting) | submodule `UMR/` | https://github.com/hanyang9/UMR (commit b45d45c) | body stage; our changes are the patch `omomo_bench/patches/umr_multi_object_retarget_prop.patch` |
| holosoma (OmniRetarget) | submodule `holosoma/` | https://github.com/amazon-far/holosoma (commit bccd4d7) | body-stage comparison only; runtime patches in `omomo_bench/omni_entry.py` |
| Unitree G1 description (29 DoF) | `urdf/*/meshes/g1`, `urdf/*/g1_29dof_*.urdf` | https://github.com/unitreerobotics/unitree_ros (BSD-3-Clause) | hand mount frames added |
| BrainCo Revo2 hand | `urdf/g1_brainco_revo2/meshes/revo2_*` | BrainCo URDF release | check the vendor licence before redistribution |
| Wuji Hand 2 | `urdf/g1_wuji_hand2/meshes/wuji_hand2_*` | Wuji Technology URDF release (as used by the holosoma community forks) | check the vendor licence before redistribution |
| dex-retargeting (AnyTeleop) | pip dependency of `hand_bench` | https://github.com/dexsuite/dex-retargeting (MIT) | finger stage |
| hiphi2smplx | external, patch `omomo_bench/patches/hiphi2smplx_mixamo_hands_hoi.patch` | https://github.com/noitom-robotics/hiphi2smplx (commit 65e5131) | SMPL-X fitting of HUMOTO / HiPHI |
| SMPL-X body model | not included | https://smpl-x.is.tue.mpg.de (licence-gated) | needed only to draw the human mesh and to re-run data preparation |
| Datasets OMOMO, HUMOTO, HiPHI | not included | their respective licences | converted inputs are in the result archive |
