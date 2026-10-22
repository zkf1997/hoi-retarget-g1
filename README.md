# Whole-body retargeting of human-object interaction motion to Unitree G1 + Wuji Hand 2

Retargets three human-object-interaction datasets (OMOMO, HUMOTO, HiPHI: SMPL-X body + fingers + object trajectories) to the
Unitree G1 (29 DoF) with the Wuji Hand 2 (20 DoF per hand): a body stage (UMR), a finger stage (AnyTeleop) and a coupled
arm + finger refinement, with evaluation scripts, a web viewer and a benchmark of body-stage methods (UMR vs OmniRetarget).

Snapshot of the handover release 2026-10-22 (branch `handover-2026-10` of the development repository, commit `a6a8591`);
the development history is not included.

**Start here: [`omomo_bench/HANDOVER.md`](omomo_bench/HANDOVER.md)** — quick start (view the results in 10 minutes), pipeline,
environments, how to reproduce, viewers, result-archive format, pitfalls, open items.

| document | content |
|---|---|
| [`omomo_bench/HANDOVER.md`](omomo_bench/HANDOVER.md) | entry point |
| [`omomo_bench/RELEASE_FORMAT.md`](omomo_bench/RELEASE_FORMAT.md) | format of the result archive (`retarget_final_results_<date>.tar.gz`, distributed separately: source data, object meshes, final robot motions for 602 / 100 / 60 sequences) |
| [`omomo_bench/README_EN.md`](omomo_bench/README_EN.md) | stage-by-stage details and every ablation |
| [`omomo_bench/reports/`](omomo_bench/reports) | analysis reports: UMR vs OmniRetarget on three datasets, finger-collision study, pushed-off sequences, white-chair defect; figures |
| [`hand_bench/README_EN.md`](hand_bench/README_EN.md) | finger retargeting bench (AnyTeleop vs TopoRetarget) |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | provenance of the robot models and the submodules |
| `README_ZH.md`, `omomo_bench/README.md`, `hand_bench/README.md`, `omomo_bench/reports/*.md` without `_en` | Chinese originals |

Final pipeline in one picture:

![pipeline](omomo_bench/reports/figures/pipeline_overview.png)

Clone with submodules (UMR, holosoma) for the retargeting stages; the viewers and the result archive do not need them:
```bash
git clone --recursive <this repo>
bash omomo_bench/relocate.sh        # rewrites the absolute paths of the original machine
bash omomo_bench/scripts/check_handover.sh   # after extracting the result archive over the clone
```
