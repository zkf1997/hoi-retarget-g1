# retarget

人形机器人运动重定向（motion retargeting）工作区：上游算法以 git submodule 形式挂载，机器人模型（G1 + 各类灵巧手 URDF）在 `urdf/` 下自建、可复现。

> 本文档持续更新，改动记录见文末 [更新日志](#更新日志)。

> **English readers start here:** `omomo_bench/HANDOVER.md` (entry point: quick start, pipeline, environments, reproduction,
> viewers, pitfalls), `omomo_bench/README_EN.md` (stage details and all ablations), `omomo_bench/RELEASE_FORMAT.md` (result
> archive format), `omomo_bench/reports/*_en.md` (analysis reports), `hand_bench/README_EN.md` (finger retargeting bench).
> This file and the other `README.md` files are the Chinese originals. Branch `handover-2026-10`.

## 目录结构

```
retarget/
├── holosoma/        submodule  amazon-far/holosoma        OmniRetarget 官方实现（src/holosoma_retargeting/）
├── UMR/             submodule  hanyang9/UMR               Unified Motion Retargeting
├── urdf/
│   ├── g1_brainco_revo2/   g1_29dof_brainco_revo2.urdf + meshes/   G1 29DoF + BrainCo Revo2 五指手
│   ├── g1_wuji_hand2/      g1_29dof_wuji_hand2.urdf    + meshes/   G1 29DoF + Wuji Hand 2 (beta2) 五指手
│   ├── previews/           两张正交投影预览图（q=0）
│   ├── scripts/
│   │   ├── fetch_third_party.sh    拉取三家厂商的 description 仓库（固定 commit）到 urdf/third_party/
│   │   ├── build_g1_hand_urdfs.py  拼装脚本：G1 躯干 + 手 + 安装位姿 → 上面两个 URDF
│   │   ├── validate_urdf.py        MuJoCo 加载、关节/质量统计、指尖位置、自碰撞检查
│   │   └── preview_urdf.py         无 OpenGL 的预览图（MuJoCo FK + matplotlib）
│   └── third_party/        git-ignored，由 fetch_third_party.sh 生成
├── omomo_bench/     OmniRetarget vs UMR 在 OMOMO 上的对比基准（数据转换、批跑、指标、viser 并列可视化），见其 README
├── hand_bench/    AnyTeleop vs TopoRetarget 在 HUMOTO / HiPHI 上的手部重定向基准（G1+BrainCo / G1+Wuji），见其 README
├── third_party/     git-ignored，toporetarget 非官方实现的 clone
└── README.md
```

初始化：

```bash
git clone --recurse-submodules git@github.com:zkf1997/retarget.git
cd retarget/urdf && bash scripts/fetch_third_party.sh      # 仅重新生成 URDF 时需要
python scripts/build_g1_hand_urdfs.py                       # 重新生成两个 URDF（默认基于 g1_29dof_rev_1_0）
MUJOCO_GL=disable python scripts/validate_urdf.py g1_brainco_revo2/g1_29dof_brainco_revo2.urdf
```

`validate_urdf.py` / `preview_urdf.py` 需要 `mujoco`（本机 conda 环境 `isaac` 或 `gmr`）；`build` 只用标准库。

## 子项目

| 子项目 | 来源 | 说明 |
|---|---|---|
| `holosoma/` | https://github.com/amazon-far/holosoma | OmniRetarget（ICRA 2026）官方代码在 `src/holosoma_retargeting/`，另含 RL 训练、sim2real。自带 `models/g1/g1_29dof.urdf`（无手/球形手）。 |
| `UMR/` | https://github.com/hanyang9/UMR | 基于点云对应的统一重定向。自带 `assets/g1/g1_29dof.urdf`（同 Unitree `g1_29dof_rev_1_0` 躯干）。 |

## URDF 模型

### 数据来源（均已固定 commit，见 `fetch_third_party.sh`）

| 部件 | 仓库 | 文件 | commit |
|---|---|---|---|
| G1 躯干（29 DoF） | unitreerobotics/unitree_ros | `robots/g1_description/g1_29dof_rev_1_0.urdf` | `ccfc6fd` (2026-09-16) |
| BrainCo→G1 安装位姿 + 转接板 | unitreerobotics/unitree_ros | `robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf` | 同上 |
| BrainCo Revo2 手 | BrainCoTech/revo2_description | `urdf/revo2_{left,right}_hand.urdf` | `92cc697` (2026-08-20) |
| Wuji Hand 2 beta2 手 | wuji-technology/wuji-description | `hand2/hand2_beta2/body/urdf/{left,right}_with_mount.urdf` | `c2cd7f8` (2026-09-11) |

选择 `g1_29dof_rev_1_0` 作为躯干是为了和 UMR / holosoma 的 `g1_29dof.urdf` 保持一致（腕关节偏移 0.046、4010 腕电机）。如果实机是新款（mode_15，5010 腕电机、腕偏移 0.051），用 `--base g1_29dof_mode_15` 重新生成即可，输出文件名会带 `_mode_15` 后缀。

### G1 手腕坐标约定

以 `{left,right}_wrist_yaw_link` 为参考（零位时前臂水平向前）：

- **+x** 指向远端（手指方向）
- **+z** 拇指侧（向上）
- **掌心朝向身体**：右手 +y，左手 −y
- 法兰面在 x = 0.0415 m 处（Unitree 自带的橡胶手、Dex3、Inspire 手都装在这里）

这一约定是从 Unitree 官方 Dex3 / Inspire / BrainCo 装配文件中反推并验证的；`validate_urdf.py` 会把指尖坐标转换到该坐标系，用于核对方向。

### 1. G1 + BrainCo Revo2 —— `urdf/g1_brainco_revo2/g1_29dof_brainco_revo2.urdf`

- 安装：完全沿用 Unitree 官方装配。转接板 `{side}_base2_link` 装在 x=0.0415；手的 `{side}_base_link` 装在 x=0.0591，`rpy = (+π/2, 0, +π/2)`（右）/ `(−π/2, 0, −π/2)`（左）。
- 手模型采用 BrainCo 自己维护的 `revo2_description`（比 Unitree 那份新：拇指运动学已修订、`*_tip` 关节为 fixed、多了 `*_touch_link` 触觉传感器 frame）。
- 每只手 11 个 revolute 关节：拇指 3（metacarpal / proximal / distal）+ 四指各 2（proximal / distal）。实机每手只有 6 个电机，远端指节由连杆带动，因此脚本给 `*_distal_joint` 加了 `<mimic>`：拇指 ×1.0，四指 ×1.155（比例来自 Unitree 官方文件）。MuJoCo 会忽略 mimic，把 distal 当独立关节；Isaac Lab 导入时可选择转成 mimic 或独立关节。
- 统计：83 links，51 个自由度（29 躯干 + 22 手），总质量 33.83 kg（每手 0.39 kg + 转接板 0.023 kg）。
- 校验：q=0 与手部关节中位两种姿态下均无自穿透。零位指尖在手腕系中：中指 (0.219, ±0.019, 0.012)，拇指 (0.108, ±0.017, 0.108) —— 手指向前、拇指向上、掌心朝内。

### 2. G1 + Wuji Hand 2 (beta2) —— `urdf/g1_wuji_hand2/g1_29dof_wuji_hand2.urdf`

- 采用 Wuji 的 `with_mount` 版本：根链接 `{l,r}_mount` 即机械臂法兰座（69 g），法兰面在 mount 坐标系 z=0（mesh z ∈ [−0.042, 0]），手沿 −z 延伸，手指屈曲轴 x，掌心法向 +y（右）/ −y（左），食指侧 +x。
- 安装：`{l,r}_mount` 直接装在 G1 法兰面 x=0.0415，`rpy = (0, −π/2, 0)`，左右相同。映射关系：−z_w → +x_g（远端），y_w → y_g（掌心朝内），x_w → +z_g（食指/拇指侧向上）。
- **注意**：Wuji 没有发布 Hand 2 与 G1 之间的官方转接件（只有一代 Wuji Hand 的 `unitree-g1-docking-adapter.stl`），这里假定 mount 法兰面直接贴 G1 法兰面、无额外厚度。若实机有转接板，改 `build_g1_hand_urdfs.py` 里 `build_g1_wuji()` 的 `G1_FLANGE_X` 偏移即可。
- 每只手 20 个 revolute 关节（5 指 × 4：拇指 cmc_flex / cmc_abd / mcp / ip，四指 mcp_flex / mcp_abd / pip / dip），全部主动。指尖有 `{l,r}_*_tip` 空链接和 `*_tip_sensor_frame` 触觉垫链接。
- 统计：101 links，69 个自由度（29 + 40），总质量 34.39 kg（每手含 mount 0.69 kg）。
- 校验：零位指尖在手腕系中：中指 (0.250, ∓0.013, 0.003)，拇指 (0.147, ±0.020, 0.109)，方向正确。存在 5 对穿透：`{l,r}_{ring,pinky,thumb}_proximal_abd` 与 `{l,r}_wrist`，这是 Wuji 手模型自身相邻指节和手腕壳网格重叠，Wuji 自己的 MJCF 里用 `<exclude>` 排除了这些配对。用 MuJoCo 做物理仿真时需要同样加 exclude（或只用简化碰撞体）；只做运动学重定向不受影响。

### 加载说明

- 网格路径全部是相对 URDF 文件的 `meshes/<子目录>/xxx.STL`，两个目录各自自包含（约 30 MB）。
- URDF 顶部有 `<mujoco><compiler meshdir="." strippath="false" .../></mujoco>`：MuJoCo 直接 `MjModel.from_xml_path()` 即可（默认的 strippath=true 会截掉子目录）。
- 已验证的加载器：MuJoCo 3.2.3、yourdfpy。
- `preview_urdf.py` 生成的预览图见 `urdf/previews/`（红色为手）。

## 全身 + 手重定向基准（`omomo_bench/`）

两部分：(1) 身体阶段 OmniRetarget 对 UMR 的对比，OMOMO 602 段、HUMOTO 100 段，手固定在 rest pose；(2) 最终的解耦流水线 **UMR 身体 → AnyTeleop 手指 → v3 联合细化（12 次迭代）**，HUMOTO 100 段和 HiPHI 60 段全量跑完，HUMOTO 接触 F1 0.745、手-物穿透 0.71 mm，HiPHI 0.577 / 0.66 mm。每一步具体做什么、运行方式、指标、结果和所有消融（物体缩放约定、细化版本、AnyTeleop 必要性、UMR 身体阶段变体）都在 [`omomo_bench/README.md`](omomo_bench/README.md)。环境：`omomo_bench/env_setup.sh`（`umr`、`hsretargeting`；`hsretargeting` 必须钉 `mujoco==3.3.5`）。

## HiPHI 交互跟踪策略（`omomo_bench/wbt/`）

把 HiPHI 60 段的最终重定向结果转成 holosoma 动作格式，在 IsaacSim 上训练一个 G1 + Wuji Hand 2 的多序列人-物交互跟踪策略：每个 env 固定一段动作和它的物体（IsaacLab 多资产生成），手指动作是参考手指姿态上的残差，奖励加指尖 / 腕 / 手指关节跟踪项，`eval_tracking.py` 逐段评测手部精度。见 [`omomo_bench/wbt/README.md`](omomo_bench/wbt/README.md)。环境：venv `hswbt`。

## 手部重定向基准（`hand_bench/`，HUMOTO / HiPHI）

用 AnyTeleop（dex-retargeting）和 TopoRetarget（非官方实现）把 HUMOTO 的 21 点手部动作重定向到 `urdf/` 下两款手，
手持物体参与穿透 / 接触保持指标。hand-only URDF（`urdf/g1_*/hand_{left,right}.urdf`）由 `hand_bench/make_hand_urdfs.py` 从两个 G1 URDF 切出。
流程、坐标约定、对上游代码的改动和结果见 [`hand_bench/README.md`](hand_bench/README.md)。环境：conda `handbench`。

## 更新日志

- **2026-09-23** 新建仓库；加入 `holosoma`、`UMR` 两个 submodule；remote 设为 `git@github.com:zkf1997/retarget.git`。
- **2026-09-23** 新建 `urdf/`：拉取 Unitree / BrainCo / Wuji 的 description 仓库，编写拼装、校验、预览脚本，生成 `g1_29dof_brainco_revo2.urdf` 与 `g1_29dof_wuji_hand2.urdf`，MuJoCo 校验通过。
- **2026-09-23** 新建 `omomo_bench/`：OMOMO → OmniRetarget / UMR 输入转换（SMPL-X 前向、物体中心化 + CoACD），批跑脚本，`compare.py` 指标，`viser_compare.py` 三车道对比，`hand_rest_pose.json`。发现并规避 mujoco ≥ 3.14 导致 holosoma_retargeting 关节不动的问题（钉 3.3.5）。15 段试点完成，全量测试集在跑。
- **2026-09-25** 新建 `hand_bench/`：HUMOTO GLB 解析、hand-only URDF、AnyTeleop / DexPilot / TopoRetarget 批跑、指标与 viser 对比；102 段 × 2 手 × 2 款手全部跑完。
- **2026-10-01** `omomo_bench/` 支持 HUMOTO（GLB → BVH → hiphi2smplx SMPL-X 拟合 → `BENCH_DATASET=humoto`）：OmniRetarget 对 UMR 在 HUMOTO 100 段上的全量对比；OmniRetarget 用 Mixamo 关节直接映射的变体。
- **2026-10-03** `omomo_bench/hand_pipeline.py` / `eval_pipeline.py`：UMR 身体 + AnyTeleop 手指 + 细化（s3 / s3p / v2 / v3）流水线，HUMOTO 和 HiPHI 全量，分阶段 viser 与渲染对比。
- **2026-10-04** 真实尺寸物体的两种约定（`--scaling real|realhand`，UMR 补丁支持机器人侧独立物体轨迹）、AnyTeleop 必要性、手尺寸重映射、身体阶段腕朝向分析与 UMR 变体（接触项权重、表面权重、手指自由、模板手姿态）等消融；viser 切换序列崩溃修复。
- **2026-10-07** 选定最终流水线（UMR 身体 → AnyTeleop → v3 12 次迭代），HUMOTO / HiPHI 全量输出到 `results_*/pipe_final`；`omomo_bench/README.md` 重写为结构化文档。
- **2026-10-08** HUMOTO 多物体：手持物体和家具（GLB 网格 + CoACD / 表面分块）进入手部细化，HUMOTO final 改为多物体版本；UMR 多物体身体阶段（补丁）作为可选项；评测新增全身-场景穿透。稠密指节/掌心接触项（`--w-dense`）验证。
- **2026-10-09** OmniRetarget 物体碰撞消融（CoACD 对原生整网格凸包）；viser 改用 MuJoCo FK 与简化显示网格，修复切换序列卡死；对 UMR 与 hiphi2smplx 的改动以补丁形式放在 `omomo_bench/patches/`。
- **2026-10-09** `humoto_bench/` 改名为 `hand_bench/`（手部重定向基准，HUMOTO 与 HiPHI 共用）；`omomo_bench` 和文档里的路径同步更新。
- **2026-10-09** `omomo_bench/wbt/`：HiPHI 重定向结果 → holosoma WBT 训练（G1 + Wuji Hand 2 共 69 DoF、每 env 一个物体、手指残差动作、手部跟踪奖励与评测脚本）；定位 `env_spacing=0` 导致的 3 倍物理减速；启动 2040 env 全量训练。
- **2026-10-02** `hand_bench/` 增加 HiPHI 后端（`BENCH_DATASET=hiphi`，`hiphi_io.py` / `prepare_hiphi.py`）：60 段 HOI 动作上跑完 AnyTeleop / DexPilot / TopoRetarget 的手部重定向对比；HUMOTO 全量指标写入 README。
- **2026-10-14/16** 发现并修复 OmniRetarget 烘焙手的 bug（`bake_hands` 把 MuJoCo 编译后的 geom 位姿写回 XML，网格重心化被应用两次，每块手网格偏 1–4 cm、转动最多 171°）；所有 OmniRetarget lane 重跑，新增 HiPHI 的 OmniRetarget（逐段身高）和三个数据集的凸包消融；三数据集 UMR 对 OmniRetarget 重比见 `omomo_bench/results/reports/umr_vs_omni_3ds.md`；OMOMO 的 final 改为两步（身体阶段手指不碰撞 + 完整碰撞的手臂-手指联合修复）。
