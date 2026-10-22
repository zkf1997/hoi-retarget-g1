# omomo_bench — 人-物交互动作到 G1 + 灵巧手的全身重定向

把 OMOMO / HUMOTO / HiPHI 三个人-物交互数据集（SMPL-X 身体 + 手指 + 物体 6DoF 轨迹，30 fps）重定向到 Unitree G1 29-DoF 加 Wuji Hand 2（或 BrainCo Revo2）五指手，输出机器人全部关节轨迹和物体轨迹，并给出身体和手部两套指标与可视化。目录里有两类东西：

- **身体阶段的两种上游方法对比**：OmniRetarget（`holosoma_retargeting`）对 UMR，手指固定在 rest pose。结论是 UMR 穿透更少、更快、根更稳，OmniRetarget 跟踪更准但带手时 55–73% 的序列要回退。
- **最终的解耦流水线**（2026-10-07 选定）：UMR 身体 → AnyTeleop 手指 → v3 联合细化（12 次迭代），HUMOTO 100 段和 HiPHI 60 段已全量跑完，输出在 `results_*/pipe_final`。

下文先讲最终流水线每一步具体做了什么，再是运行方式、指标、结果，最后是所有设计决策背后的消融。

---

> **English readers start here:** `HANDOVER.md` (entry point: quick start, pipeline, environments, reproduction,
> viewers, pitfalls), `README_EN.md` (stage details and all ablations), `RELEASE_FORMAT.md` (result
> archive format), `reports/*_en.md` (analysis reports), `../hand_bench/README_EN.md` (finger retargeting bench).
> This file and the other `README.md` files are the Chinese originals. Branch `handover-2026-10`.

## 1. 目录结构

```
omomo_bench/
├── env_setup.sh                创建 conda 环境 umr / hsretargeting
├── prepare_omomo.py            OMOMO pickle → UMR / OmniRetarget 输入 + 物体网格（CoACD）+ source npz        [umr]
├── prepare_humoto.py           HUMOTO（经 hiphi2smplx 拟合的 SMPL-X）→ data_humoto/                           [umr]
├── prepare_hiphi.py            HiPHI（hiphi2smplx 的 SMPL-X + 物体轨迹）→ data_hiphi/                            [umr]
├── prepare_humoto_mixamo.py    OmniRetarget 直接用 Mixamo 关节的变体输入（mx/mxh/mxtoe/mxpel/mxleg）            [umr]
├── prepare_scaling_variants.py OmniRetarget 式物体 z 锚定的 UMR 输入（data/umr_omniscale）                       [umr]
├── prepare_realsize_hand_anchor.py 真实尺寸物体按手腕锚定的 UMR 输入（data_*/umr_realhand）                     [umr]
├── run_omni.py / omni_entry.py 批量跑 OmniRetarget（带手 URDF、UMR 式缩放、回退链、上游补丁）                    [hsretargeting]
├── run_umr.py                  批量跑 UMR（--hand, --scaling umr|omni|real|realhand, --variant）                 [umr]
├── hand_pipeline.py            阶段 2/3：AnyTeleop 手指替换 + s3 / s3p / v2 / v3 细化 → results_*/pipe_<stage>   [umr]
├── eval_pipeline.py            身体 + 手部指标 → results_*/pipe_metrics*.csv, pipe_summary*.md                    [umr]
├── compare.py                  身体指标 + 模型/物体工具函数（被 eval_pipeline / 可视化复用）                     [umr|hsretargeting]
├── wrist_orientation_stats.py  身体阶段腕朝向误差统计                                                           [umr]
├── viser_compare.py            viser 并列播放（--pipeline 预设 = 流水线各阶段）                                  [hsretargeting]
├── render_compare.py           静态 PNG 网格（列 = 源 + lane，行 = 帧），MuJoCo EGL 离屏渲染                     [umr|hsretargeting]
├── run_final_pipeline.sh       一键：UMR 身体 → 手部阶段 → 评测（按 dataset）
├── run_variant_pipeline.sh     同上，换一个 UMR 身体变体
├── scaling_conflict_figure.py  身体/手尺寸矛盾示意图 → results/figures/scaling_conflict.png
├── make_hand_mjcf.py           ../urdf/ 的 G1+手 URDF → robots/g1_29dof_<hand>.xml + configs/umr_g1_<hand>*.json
├── hand_rest_pose.json / hand_rest.py   手指 rest pose（两款手的 fnmatch 规则）
├── configs/umr_g1_wuji*.json   UMR 机器人配置及变体（_v2 _real _free _flat _contact0 _contact4 ...）
├── data[_humoto|_hiphi]/       (git-ignored) 转换后的数据：source/ umr/ omni/ objects/ seqs_test.json
├── results[_humoto|_hiphi]/    (git-ignored) 各 lane 的 npz、日志、指标、图
└── omni_workdir[_humoto]/      (git-ignored) holosoma 需要的工作目录
```

环境：`umr`（UMR、hand_pipeline、eval、渲染）、`hsretargeting`（holosoma，mujoco 钉 3.3.5，viser）、`handbench`（hand_bench 的 AnyTeleop）。所有脚本通过 `BENCH_DATASET=omomo|humoto|hiphi` 切换数据目录；MuJoCo 离屏渲染和碰撞检测需要 `MUJOCO_GL=egl`。

---

## 2. 最终流水线做了什么

### 2.0 输入与尺度约定

每段序列的输入（`data_*/source/<seq>.npz`）：SMPL-X 参数（16 维 betas、身体 63 维、手指 90 维）、55 个关节、主物体的位姿轨迹、T-pose 身高；物体网格在 `data_*/objects/<obj>/`（视觉网格 + CoACD 凸分解块）。手部 21 个关键点和 AnyTeleop 结果来自 `../hand_bench/`。

**缩放世界**：UMR 要求人体缩放到机器人身高，$s = 1.32 / \text{T-pose 身高}$（HUMOTO 0.85，OMOMO 约 0.77，HiPHI 0.72–0.85）。人体顶点、关节、物体位置和物体网格都乘 $s$，机器人保持真实尺寸。所有阶段、指标和可视化都在这个世界里；结果 npz 存 `smpl_scale` 和 `object_mesh_scale`（默认等于 $s$）。这意味着机器手（中指链长 20.9 cm，比人的 18.1 cm 长 15%）要去抓一个缩小到 0.77–0.85 倍的物体，手/物尺寸比是人的 1.35–1.5 倍。这个矛盾的几种替代方案在第 6 节，结论是缩放世界仍然最好。

### 2.1 阶段 1：UMR 身体重定向（`run_umr.py --hand wuji`）

UMR 用**点云表面对应**做重定向。离线在 SMPL-X T-pose（该段的 betas，手指伸直）和机器人 T-pose（`configs` 里的 `tpose_qpos`，手指在 rest pose）两片采样点云之间，用 template-residual 自编码器学出 4096 个共享顺序的 slot；每个 slot 一端绑在 SMPL-X 网格的一个面上，另一端绑在机器人某个 geom 的表面点上，随各自的姿态运动。在线每帧解一个高斯-牛顿式 QP，变量是 7 维根位姿加 29 个身体关节（40 个手指关节用 `joint_limits = [v−1e−4, v+1e−4]` 钉在 `hand_rest_pose.json` 的 rest pose），残差项：

| 项 | 作用的 slot | 残差 | 权重 |
|---|---|---|---|
| 表面位置 | 按身体分段采样的 456 个（每只手 15 / 80，躯干 50 / 714） | $p^{robot}_i(q) - s\,p^{human}_i$ | 手 10，躯干 20，头 1 |
| 表面法向 | 同上 | $n^{robot}_i(q) - \hat n_i$，目标法向按 T-pose 下人机法向夹角做固定修正 | 手 1，腿/躯干 5，前臂 0 |
| 物体接触 | 全部 4096 个里距物体 ≤ 10 cm 的最近 128 个 | $(p^{robot}_i - o_{k(i)}) - v_i$，$v_i$ 是人侧 slot 到最近物体采样点的偏移（< 5 mm 归零） | 5 × 强度(0.5–1)，直接乘在行上，有效二次权重 6–25 |
| 地面接触 / 自接触 / 时间平滑 | | | 默认值 |

约束：关节限位、地面非穿透、机器人-物体非穿透（`mj_geomDistance` 线性化，带 slack，slack 代价 5000）。输出 `results_*/umr_wuji/<seq>.npz`：`qpos (T, 7+69)` 按 `robot_joint_names` 顺序、`smpl_scale`、`ground_z`、`hand`、`object_mesh_scale`。

对应关系按 SMPL-X 模板复用：HUMOTO 一个模板（100 段共用），OMOMO 按受试者，HiPHI 每段一个模板（转换时逐段拟合 betas），所以 HiPHI 必须 `--no-slot-reuse`。每段约 40 s（HUMOTO）到 3 min（HiPHI 长序列）。

身体阶段给出的腕位姿只能当初值：腕位置系统性后退约 4 cm（机器手比人手长，表面匹配让指尖对齐就把腕往回推），腕朝向均值 10°，双手端碗、盘、花瓶时整只手会翻转 40–50°（接触项在手指钉死时的最小二乘妥协，见 6.5）。

### 2.2 阶段 2：AnyTeleop 手指（`../hand_bench/run_anyteleop.py`，`hand_pipeline.py --stage s2`）

手指不依赖身体阶段。对每段、每只手，在**规范手坐标系**（x = 腕→中指 MCP，z = 小指 MCP→食指 MCP 的正交化方向，y = 掌心法向；和 G1 两款手 URDF 的腕坐标约定一致）里取人的 21 个关键点，用 dex-retargeting 的 vector 优化器逐帧解 20 个手指关节：目标是腕→指尖的 5 个向量乘以 `scaling_factor`，比例 = 机器人中指链长 / 人中指链长（Wuji ≈ 1.15），带关节限位和时间平滑（`low_pass_alpha=1.0`，即不滤波）。这一步在手部 benchmark 里已经验证优于 TopoRetarget（接触 F1 0.74 对 0.61）。

`pipe_s2` 只是把 AnyTeleop 的 40 个手指关节值逐帧写进 UMR 的 qpos，手臂不动。它本身不能用（指尖目标不看物体，穿透 6.5 mm），但它提供了"已经包住物体"的全局手形，这是后面细化能收敛的前提（去掉它 F1 从 0.74 掉到 0.48，见 6.3）。

### 2.3 阶段 3：v3 联合细化（`hand_pipeline.py --stage v3 --v3-iters 12`）

每帧每只手解一个 27 变量的 QP（7 个手臂关节 + 20 个手指关节），从 AnyTeleop 手形（首帧）或上一帧解（其余帧）出发线性化，每帧 12 次内迭代。手臂的 7 个自由度就是 G1 从肩到法兰的全部关节（肩 3、肘 1、腕 3），6 维腕位姿目标之外剩一个冗余自由度给先验和非穿透用；腰不参与，避免改动身体阶段的躯干。

| 项 | 内容 | 权重 |
|---|---|---|
| 腕位置锚点 | 法兰 `{l,r}_mount` 到缩放后人腕关键点 | $50 \cdot \max(\alpha, 0.15)$ |
| 腕朝向 | `wrist_yaw_link` 到人手规范坐标系 | 5 |
| 接触指尖目标 | 人指尖距物体表面 < 2 cm（滞回：进 2 cm 出 4 cm，至少 3 帧）的指尖，目标 = 人接触点投影到物体表面后映射到机器人世界；拇指 / 食指 / 其余优先级 1.45 / 1.25 / 0.78 | 200 |
| 手指先验 | AnyTeleop 手形 | 1 |
| 手臂先验 | UMR 的手臂解 | $0.5 \cdot (1 + 4(1-\alpha))$ |
| 时间项 | 手指：上一帧 2、加速度 20；手臂：相对 UMR 解的修正量的速度 40、绝对加速度 60 | |
| 非穿透 | 手 + 前臂全部 geom 对物体 CoACD 块，2 cm 内的 pair 线性化，2 mm 余量，带 slack | slack 10⁴ |
| 关节限位 | 硬约束 | |

$\alpha(t)$ 是接触度：指尖距物体 < 5 cm 的指示量的 12 帧因果滑动平均。接触时腕锚点全权重、手臂先验弱；不接触时腕锚点降到 15%、手臂先验强 5 倍。接触目标**不**按手尺寸重映射（消融证明无用，见 6.4），手尺寸差由腕锚点的放松吸收。求解器 Clarabel。输出 `results_*/pipe_final/<seq>.npz`（`pipe_v3_it12` 的软链），npz 布局与阶段 1 相同，另有 `stage`、`hands`、`params`。

12 次迭代对 4 次的收益很小但一致（HUMOTO F1 0.739 → 0.745，穿透 0.77 → 0.71 mm；HiPHI 持平），代价是手部阶段慢 3 倍（HUMOTO 100 段 8 进程约 9 分钟）。

### 2.4 评测与可视化

`eval_pipeline.py` 对每条 lane 用 MuJoCo 正运动学 + 碰撞检测算第 4 节的指标；`viser_compare.py --pipeline` 把源人体和各阶段并列播放；`render_compare.py --pipeline --focus right_hand` 出静态网格图。

---

## 3. 运行方式

```bash
# 数据准备（见第 7 节）
export BENCH_DATASET=humoto            # 或 hiphi
# 阶段 1 + 2/3 + 评测，一键
bash run_final_pipeline.sh humoto 3                    # HUMOTO：UMR 3 进程
bash run_final_pipeline.sh hiphi  3 --no-slot-reuse    # HiPHI：每段重训 correspondence
```

分步：

```bash
conda run -n umr python run_umr.py --hand wuji --workers 3 [--no-slot-reuse]                         # -> results_*/umr_wuji
# AnyTeleop 手指：hand_bench 里 BENCH_DATASET=<ds> python run_anyteleop.py --hand wuji            # -> hand_bench/results*/anyteleop/wuji
MUJOCO_GL=egl conda run -n umr python hand_pipeline.py --stage s2 v3 --v3-iters 12 --out-suffix _it12 --workers 8   # -> pipe_s2_it12, pipe_v3_it12
MUJOCO_GL=egl conda run -n umr python eval_pipeline.py --out _it12 --lane v3_it12=results_$BENCH_DATASET/pipe_v3_it12 --workers 8
conda run -n hsretargeting python viser_compare.py --final --port 8090      # 源 | 阶段 1 | 2 | final（--pipeline 显示全部中间版本）
MUJOCO_GL=egl conda run -n umr python render_compare.py --pipeline --seqs <seq> --focus right_hand --distance 0.7 --tag _right
```

常用开关：`hand_pipeline.py --stage s2 s3 s3p v2 v3`（各版本细化，见 6.2）、`--body-lane`（换身体阶段 lane）、`--out-suffix`、`--finger-prior rest`、`--v3-remap ratio`、`--obj-mesh-scale 1.0`；`run_umr.py --scaling umr|omni|real|realhand`、`--variant v2|free|flat|contact0|contact4|...`（对应 `configs/umr_g1_wuji_<variant>.json`）；`eval_pipeline.py --lane NAME=DIR[:SUFFIX]`（可重复）、`--out`（必须带，否则覆盖默认 csv）、`--merge a.csv b.csv --out X`（合并出对比表）。长任务放在 tmux 里跑。

长期开着的 viser：HUMOTO 8090、HiPHI 8091、OMOMO 8092（OmniRetarget 对 UMR）。缩放世界对真实尺寸锚定的对比需要时用 `--lane` 手动起。

---

## 4. 指标

身体指标（`compare.py`，在缩放世界里算）：`obj_pen_max` 逐帧最深机器人-物体穿透的均值、`obj_pen_frac` 穿透 > 5 mm 的帧比例、`ground_pen`、`foot_slide`（人足支撑相中机器人踝 xy 位移）、`hand_contact`（人腕距物体 < 10 cm 的帧里机器人手也 < 10 cm 的比例）、`wrist_err / ankle_err`（对缩放后的 SMPL-X 关键点）、`joint_jerk / root_jerk`（三阶差分 × fps³ 的绝对值均值）。

手部指标（`eval_pipeline.py`，距离换算到人体单位 = 机器人世界 / s，每只手分别算再平均）：`flange_pos_err_cm` / `flange_rot_deg` 法兰相对人腕关键点和规范手坐标系的误差；`tip_pos_err_cm` 交互帧里机器人指尖对人指尖的距离（跟踪指标）；`tip_obj_dist_err_cm` 机器人指尖到物体表面的距离与人的差（抓取指标）；`contact_recall / precision / f1`，人和机器人指尖都以 2 cm 为接触阈值；`hand_pen_mm / hand_pen_frac` 手-物穿透（仅交互帧）；`finger_jerk`。

---

## 5. 结果

### 5.1 最终流水线（HUMOTO 100 段 / HiPHI 60 段，2026-10-07）

> **2026-10-13 更新**：HUMOTO 的 final 已改为手指不碰撞身体阶段上的 v3（`pipe_v3_nfcfull`，见 6.10）：接触 F1 0.799、手-物穿透 0.89 mm / 4.6% 帧、法兰 2.42 cm / 4.0°、关节 jerk 167 / 根 jerk 16（`results_humoto/pipe_summary_nfcfull.md`）。下表 HUMOTO 列是改动前的 `pipe_v3_it12`；HiPHI 列仍是 final。

| 指标 | HUMOTO 阶段 1 | HUMOTO 阶段 2 | **HUMOTO final** | HiPHI 阶段 1 | HiPHI 阶段 2 | **HiPHI final** |
|---|---|---|---|---|---|---|
| 接触 F1（召回 / 精度） | 0.26 | 0.46 | **0.745 (0.71 / 0.86)** | 0.17 | 0.29 | **0.577 (0.53 / 0.76)** |
| 指尖-物体距离误差 (cm) | 4.43 | 2.44 | **1.30** | 5.22 | 3.70 | **1.74** |
| 指尖位置误差 (cm) | 9.5 | 5.0 | **3.25** | 13.5 | 8.7 | **5.66** |
| 手-物穿透 (mm) / >5 mm 帧 | 2.6 / 0.18 | 6.5 / 0.48 | **0.71 / 0.044** | 3.1 / 0.18 | 12.4 / 0.64 | **0.66 / 0.016** |
| 全身物体穿透 (mm) / >5 mm 帧 | 4.5 / 0.32 | 9.9 / 0.70 | **1.7 / 0.11** | 5.7 / 0.44 | 15.6 / 0.75 | **3.1 / 0.24** |
| 法兰位置 (cm) / 朝向 (°) | 4.25 / 9.9 | 4.25 / 9.9 | **2.54 / 4.0** | 3.91 / 14.1 | 3.91 / 14.1 | **3.04 / 6.0** |
| 腕跟踪误差 (cm) | 3.7 | 3.7 | **2.2** | 3.0 | 3.0 | **2.3** |
| 身体关节 jerk / 根 jerk | 258 / 19.4 | 258 / 19.4 | **191 / 19.4** | 193 / 10.3 | 193 / 10.3 | **134 / 10.3** |
| 手指 jerk | 0.6 | 53 | 65 | 0.5 | 199 | 88 |

完整表 `results_*/pipe_summary_chosen.md`，逐段 `pipe_metrics_it12.csv`。HiPHI 的物体多为大件（箱子、椅子、水桶、长凳、拖把），双手操作，接触 F1 天然更低。图：`results_*/figures/*_final_*.png`（列 = 源 | 阶段 1 | 阶段 2 | final）。

### 5.2 细化版本演进（HUMOTO 100 段，4 次迭代）

| 指标 | s2 (+AnyTeleop) | s3 手指 QP | s3p v1 +手臂放置 | v2 联合 QP | v3 |
|---|---|---|---|---|---|
| 接触 F1 | 0.46 | 0.63 | 0.73 | 0.73 | **0.74** |
| 指尖-物体距离误差 (cm) | 2.44 | 1.58 | **1.23** | 1.33 | 1.30 |
| 手-物穿透 (mm) / >5 mm 帧 | 6.5 / 0.48 | 2.9 / 0.20 | 2.2 / 0.14 | 0.86 / 0.054 | **0.77 / 0.048** |
| 全身物体穿透 (mm) | 9.9 | 5.2 | 3.9 | 2.0 | **1.8** |
| 法兰位置 (cm) / 朝向 (°) | 4.25 / 9.9 | 4.25 / 9.9 | **1.67 / 1.0** | 2.05 / 3.8 | 2.56 / 4.0 |
| 身体关节 jerk | 258 | 258 | **182** | 207 | 191 |
| 手指 jerk | 53 | 144 | 113 | **64** | 66 |

HiPHI 同一组：F1 0.29 / 0.48 / 0.52 / 0.55 / **0.58**，手-物穿透 12.4 / 4.9 / 7.0 / 1.2 / **0.65** mm，法兰 3.9 / 3.9 / **2.1** / 2.5 / 3.0 cm，关节 jerk 193 / 193 / 145 / 163 / **135**。v1 → v2 的收益在穿透（非穿透约束和接触目标进同一个 QP，腕锚点让步），v3 再降穿透、升 F1、降 jerk；v1 的腕跟踪最好是因为它显式 IK 到人腕系，v2/v3 放松腕锚点是有意为之。

### 5.3 身体阶段两种方法对比（手指 rest pose）

**总结文档：`reports/body_stage_comparison_2026-10-17.md`**（UMR 对 OmniRetarget、CoACD 对凸包、被推开机制、手指碰撞开关、跑完全部阶段后的穿透、推荐配置）。OMOMO 602 段、HUMOTO 100 段、HiPHI 60 段（逐段身高），统一 UMR 缩放，所有 lane 都按同一套 CoACD 物体几何评测。OmniRetarget 用 CoACD 碰撞（我们的默认，见 6.9），2026-10-15 修正烘焙手后重跑。完整表、BrainCo、回退拆分见 `reports/umr_vs_omni_3ds.md`（`results*/summary_3ds.md`、`wrist_gap_3ds.log`、`wrist_orientation_3ds.log`）。

| 指标（G1 + Wuji） | OMOMO Omni | OMOMO UMR | HUMOTO Omni | HUMOTO UMR | HiPHI Omni | HiPHI UMR |
|---|---|---|---|---|---|---|
| 物体穿透 (mm) / >5 mm 帧 | 10.1 / 42% | **5.4** / 43% | 10.4 / 60% | **4.5 / 32%** | 13.5 / 47% | **5.7 / 44%** |
| 需要回退的段 | 359 / 602 | 0 | 71 / 100 | 0 | 32 / 60 | 0 |
| 手-物接触保持 | **0.97** | 0.95 | **1.00** | 0.87 | 0.96 | 0.96 |
| 腕位置误差 (cm) | **2.2** | 3.5 | **1.6** | 3.7 | **2.9** | 3.0 |
| 腕朝向误差 (°) / >30° 帧 | 32 / 43% | **13 / 6%** | 39 / 59% | **10 / 5%** | 47 / 67% | **14 / 3%** |
| 手被推到物体另一侧的段（按手计） | **10** | 62 | **0** | 16 | 6 | **5** |
| 地面穿透 (mm) | 4.3 | **0.2** | 4.2 | **0.05** | 2.4 | **0.2** |
| 脚滑 (mm/帧) | **1.0** | 2.6 | **1.0** | 2.8 | **0.9** | 1.9 |
| 踝误差 (cm) | **4.3** | 5.2 | 4.6 | **3.6** | 5.1 | **4.7** |
| 关节 jerk / 根 jerk | **201** / 56 | 273 / **14** | **238** / 62 | 258 / **19** | **140** / 35 | 193 / **10** |
| 每帧耗时 (s，20 进程并行下测得) | 2.9 | **0.25** | 1.1 | **0.15** | 2.3 | **0.10** |

OmniRetarget 的穿透几乎全来自回退：硬约束可行的段穿透 < 2 mm，但 CoACD 下有 53–71% 的段不可行（rest pose 的固定手指包不住把手、边沿），回退到放宽容差 / 手指不参与 / 关闭物体约束后平均 12–28 mm，占其总穿透的 97%。UMR 的物体约束带 slack，从不失败，穿透稳定在 4.5–5.7 mm。OmniRetarget 腕位置更准、接触保持更好、几乎不把手推到物体另一侧（它有腕关键点和交互网格；UMR 没有腕位置项，见 6.10）；UMR 腕朝向好 3–4 倍（OmniRetarget 没有朝向项，手绕前臂自由转）、地面穿透接近 0、根 jerk 低 3–4 倍、快 10–20 倍，但脚滑多 2–3 倍、关节 jerk 更高。BrainCo（OMOMO / HUMOTO）排序相同。OmniRetarget 写死 29 DoF，带手只能把手网格烘成固定 geom（`run_omni.py --hand`）。早先"OmniRetarget 在 HUMOTO 上直接用 Mixamo 关节比经 SMPL-X 更好"的试验（`results_humoto/omni_brainco_mx*`）是修正烘焙手之前跑的，未重跑。

---

## 6. 设计决策与消融

每条都在 HUMOTO 上验证过，20 段子集 = 5 段最差的双手端碗序列 + 15 段随机（`results_humoto/wrist_subset.json`）。

### 6.1 物体缩放还是真实尺寸

缩放世界里手/物尺寸比是人的 1.35 倍（`results/figures/scaling_conflict.png`）。试了两种真实尺寸约定，全量 HUMOTO：

| v3 之后 | 缩放世界（采用） | 真实尺寸，物体中心 × s（`--scaling real`） | 真实尺寸，按手腕锚定（`--scaling realhand`） |
|---|---|---|---|
| 接触 F1 | 0.74 | 0.71 | **0.76** |
| 手-物穿透 (mm) | **0.77** | 1.05 | 0.79 |
| 全身物体穿透 (mm) | **1.8** | 2.9 | 2.6 |
| 法兰位置 (cm) | **2.56** | 2.94 | 2.68 |
| 身体关节 / 根 jerk | **191 / 19** | 212 / 23 | 213 / 23 |

第一种差是因为人体整体 × s 而物体半径不缩，手离表面近了 0.15 × 半径。第二种（`prepare_realsize_hand_anchor.py`：机器人侧物体轨迹改成 $p' = a + (p-a)/s$，$a$ 为人腕，两手按指尖距表面加权；UMR 补丁让源侧接触沿原轨迹、机器人侧沿新轨迹，结果 npz 多 `obj_pos_robot`）手部指标追平并略超缩放世界，但 UMR 身体阶段在"身体贴 × s 的人"和"手上的点落到更大的物体"之间折中，jerk 和前臂穿透变差。两者兼得需要改 UMR 身体项，未做。HiPHI 的地面大件还要额外的 z 锚定。

### 6.2 细化的三个版本

v1（s3p）：先用高斯-牛顿 IK 把 7 个手臂关节放到人腕位姿（掌对物体非穿透 QP），再解 20 变量的手指 QP，事后 5 帧滑动平均。v2：27 变量联合 QP，接触判定滞回，因果时间项，手和前臂非穿透。v3：按 DexWeave（arXiv:2609.34724）加接触度门控腕锚点、指尖优先级、修正量时间项加绝对加速度项，接触目标用人的接触点本身。DexWeave 的手形初始化是 729 项码本检索而不是 AnyTeleop，联合细化 40 次迭代，场景默认不缩放，没有物理验证。

### 6.3 阶段 2 不能省

`--finger-prior rest`（v3 的手指先验和初值换成 rest pose）：F1 0.74 → 0.48，指尖误差 3.3 → 6.3 cm，100 段全部变差。逐帧线性化的 QP 带非穿透约束，从张开的手出发合拢 90° 的直线路径穿过物体，被挡在外面。需要的是全局手形初值，来源可换（码本检索也行），不能没有。

### 6.4 手尺寸重映射无用

v2 把人指尖相对腕的偏移乘 1.15 再投影到物体表面（且该因子在缩放世界里应为 ratio/s = 1.35）。`--v3-remap ratio` 全量验证：F1 0.741 对 0.739，穿透 0.84 对 0.77 mm，逐段 F1 胜率 49%，小物体也无差别。目标点最后都投影到表面，撑开再投影落点几乎不变，尺寸差已经由腕锚点吸收。v3 默认不重映射。

### 6.5 身体阶段的腕朝向和各种 UMR 变体

腕朝向误差（`wrist_orientation_stats.py`，HUMOTO 全部帧）：均值 9.9°，中位数 6.8°，4.8% 的帧 > 30°，偏置只有 3.5–5.8°，不是坐标约定问题；交互帧 12°，最差的双手端碗序列 40–54°。原因是接触项：它用全部 4096 个 slot 里离物体最近的 128 个，有效权重最高 25，而表面项在手上只有 15 个 slot、权重 10；手指钉死、手又长 15%，让更多手面点贴上碗沿的最省力解是翻转整只手。子集上的变体：

| UMR 变体 | 腕朝向 / >30° 帧 | 腕位置 (cm) | 全身穿透 (mm) | 关节 / 根 jerk | 接 v3 后：F1 / 穿透 (mm) / 关节 jerk |
|---|---|---|---|---|---|
| 基线 | 12.9° / 12% | 4.6 | 4.3 | 341 / 27 | **0.77** / 0.97 / 250 |
| `contact4` 接触项 ×4 | 14.6° / 14% | **3.9** | 9.6 | 391 / 37 | — |
| `contact0` 接触项关 | 7.0° / 1% | 4.9 | **2.3** | 307 / 24 | — |
| `v2` 手部表面 slot 15→40，位置 10→30，法向 1→5 | 6.4° / 0% | 4.9 | 7.2 | 299 / 28 | 0.76 / 0.83 / 231 |
| `free` 手指不钉死 | 9.9° / 2.6% | **3.7** | 2.5 | **261 / 24** | 0.76 / 1.07 / 217 |
| `flat` 对应模板用伸直的机器人手 | 14.1° / 9.9% | 4.6 | 4.0 | 317 / 27 | 0.75 / 0.48 / 233 |

`free` 的 jerk 下降全在肩肘腕（410 → 256）和腰，腿不变：手上逐帧变化的接触配对原来只能由手臂吸收，手指自由后由手指吸收（手指 jerk 184），但 UMR 自己的手指不能用（15 个 slot 分不清五指，F1 0.28）。`v2` 在全量上接触 F1 0.739 → 0.723（HUMOTO）、持平但更穿（HiPHI），被否决；`free` 的全量验证（`run_variant_pipeline.sh <ds> free`，接 AnyTeleop + v3 12 次迭代，`results_*/pipe_summary_freefinal.md`）：HUMOTO 接触 F1 0.741 对 0.745、手-物穿透 0.94 对 0.71 mm（更差）、关节 jerk 168 对 191（96% 的段更好）；HiPHI F1 0.580 对 0.577、穿透 0.63 对 0.66 mm、关节 jerk 117 对 134（98% 更好）、根 jerk 9.3 对 10.3。即身体更平滑是稳定收益，手部在 HUMOTO 上付出穿透代价、在 HiPHI 上持平。默认仍用基线身体；只看身体平滑或只做 HiPHI 这类大件任务时可换 `free`。身体阶段改不出朝向对、位置对、不穿透三者兼得，最终由 v3 修正到 4°。

### 6.6 UMR 式的稠密接触项加进第三阶段（2026-10-07）

v3 原来只有 5 个指尖有接触目标。仿照 UMR 的 contact map，但对应关系用 AnyTeleop 那套关键点映射而不是学出来的 slot：取 15 个指节关节中心（四指的 MCP/PIP/DIP、拇指的 CMC/MCP/IP）加 4 个由关键点线性组合出的掌心点，人侧每个点离物体表面 < 3 cm（滞回，6 cm 退出，至少 3 帧）时，机器人上对应的点必须复现人侧点到最近表面点的偏移向量，偏移按机器手/人手尺寸比放大（机器人关节中心在皮肤下更深）。实现：`hand_pipeline.py --w-dense W --tau-dense 0.03`（`DENSE_POINTS`、`contact_targets_dense`、`Robot.dense_points/dense_jac`），`eval_pipeline.py` 新增 `kp_contact_recall/precision/f1`（同一组点，3 cm）和 `kp_obj_dist_err_cm`。

| HUMOTO 100 段 | final（仅指尖） | + 稠密项 w=50 | + 稠密项 w=150 |
|---|---|---|---|
| 指尖接触 F1（召回 / 精度） | 0.745 (0.71 / 0.86) | 0.764 (0.73 / 0.86) | **0.781 (0.76 / 0.88)** |
| 指节+掌心接触 F1（召回 / 精度） | 0.528 (0.43 / 0.84) | 0.564 (0.46 / 0.86) | **0.604 (0.50 / 0.87)** |
| 指尖-物体距离误差 (cm) | 1.30 | 1.18 | **1.07** |
| 指节+掌心-物体距离误差 (cm) | 1.71 | 1.62 | **1.50** |
| 指尖位置误差 (cm) | 3.25 | 2.96 | **2.65** |
| 手-物穿透 (mm) / >5 mm 帧 | **0.71 / 0.044** | 0.83 / 0.048 | 0.89 / 0.052 |
| 全身物体穿透 >5 mm 帧比例 | **0.106** | 0.118 | 0.120 |
| 法兰位置 (cm) / 朝向 (°) | **2.54** / 3.99 | 2.67 / 4.01 | 2.80 / **3.84** |
| 关节 jerk / 手指 jerk | 191 / **65** | 191 / 72 | 192 / 81 |

逐段：w=150 对 final 的指尖 F1 胜率 78%，指节掌心 F1 89%，指尖-物体距离 91%；穿透只有 27% 更好，法兰 21%。即接触质量是一致的大幅提升（F1 +0.036，相当于 v1 到 v3 全部改进的总和），代价是手-物穿透多 0.2 mm、法兰误差多 0.26 cm、手指 jerk 多 25%，权重越大趋势越明显。
| HiPHI 60 段 | final（仅指尖） | + 稠密项 w=50 | + 稠密项 w=150 |
|---|---|---|---|
| 指尖接触 F1（召回 / 精度） | 0.577 (0.53 / 0.76) | **0.595 (0.55 / 0.78)** | 0.587 (0.54 / 0.76) |
| 指节+掌心接触 F1（召回 / 精度） | 0.449 (0.33 / 0.84) | 0.503 (0.38 / 0.86) | **0.564 (0.44 / 0.88)** |
| 指尖-物体距离误差 (cm) | 1.74 | 1.61 | **1.52** |
| 指节+掌心-物体距离误差 (cm) | 2.33 | 2.17 | **2.06** |
| 手-物穿透 (mm) / >5 mm 帧 | **0.66 / 0.016** | 0.85 / 0.042 | 1.17 / 0.063 |
| 全身物体穿透 >5 mm 帧比例 | **0.236** | 0.272 | 0.305 |
| 法兰位置 (cm) / 朝向 (°) | **3.04** / 5.99 | 3.19 / **5.84** | 3.52 / 5.85 |
| 关节 jerk / 手指 jerk | 134 / **88** | 134 / 96 | 134 / 105 |

结论：稠密接触项确实补上了指节和掌心的接触，两个数据集的指节掌心 F1 各升 0.04–0.08，指尖 F1 也跟着升 0.02–0.04，指尖位置误差降 0.3–0.9 cm；代价是穿透和法兰误差单调变差，权重越大越明显，HiPHI 的大件物体对此更敏感（w=150 时手-物穿透 0.66 → 1.17 mm，全身穿透帧比例 0.24 → 0.31）。w=50 是两个数据集都能接受的折中：接触 F1 +0.02，指节掌心 F1 +0.04–0.05，手-物穿透 +0.15 mm（仍在 1 mm 内），法兰 +0.15 cm。HiPHI 上 w=50 对 final 的逐段胜率：指尖 F1 67%，指节掌心 F1 98%，指尖-物体距离 75%；手-物穿透只有 13% 更好，法兰 32%。`pipe_final` 暂未切换到带稠密项的版本，lane 在 `results_*/pipe_v3_dense{50,150}`，表格 `pipe_summary_dense.md`。


### 6.7 多物体（HUMOTO，第一步：手持物体，2026-10-08）

HUMOTO 100 段里 41 段有多个物体，36 段手碰到了不止一个。原来只有接触帧最多的主物体参与重定向，其余物体在手部阶段既没有接触目标也没有非穿透约束。第一步只处理手持物体（`handheld` 标志，有现成网格和 CoACD）：

- `prepare_humoto_multi.py`：主物体之外，任一只手在 2 cm 内超过 10 帧的手持物体都入选，资产拷进 `data_humoto/objects/`，写 `data_humoto/multi_objects.json`（35 段多了 56 件物体，最多 5 件）。
- `compare.augmented_model` 接受物体列表，每件一个自由 body；`hand_pipeline.py --multi`：v3 的指尖目标、接触度 α 和稠密项都取每帧每点**最近的那件物体**，所有物体进非穿透约束；结果 npz 多一个 `objects` 键。额外物体的机器人世界位姿 = HUMOTO 轨迹 × s。
- `eval_pipeline.py --multi`：手部指标对所有物体算（点到最近物体的距离、对任一物体的穿透），两条 lane 用同一评测。viser 和 `render_compare.py` 会画出结果里的全部物体。

| 35 段多物体序列（按全部物体评测） | 原 final（仅主物体） | 多物体 |
|---|---|---|
| 指尖接触 F1（召回 / 精度） | 0.679 (0.60 / 0.87) | **0.767 (0.71 / 0.90)** |
| 指节+掌心接触 F1 | 0.550 | **0.578** |
| 指尖-物体距离误差 (cm) | 1.69 | **1.30** |
| 手-物穿透 (mm) / >5 mm 帧 | 4.25 / 0.29 | **0.79 / 0.034** |
| 法兰位置 (cm) / 朝向 (°) | 2.01 / **3.7** | **1.86** / 4.9 |
| 关节 jerk / 手指 jerk | **71 / 55** | 82 / 85 |

逐段胜率：接触 F1 86%，手-物穿透 97%，法兰位置 71%。另外 65 段单物体序列结果与原 final 逐位相同。代价是手指 jerk 和关节 jerk 上升：最近物体在两件物体之间切换时目标跳变（例如手从篮子把手移到碗上）。HUMOTO 的 `pipe_final` 已指向 `pipe_v3_multi`；全部 100 段的数字：F1 0.740 → 0.771，手-物穿透 2.05 → 0.85 mm（按全部物体评测）。（第一步的数字；第二、三步见下。）

#### 第二步：家具进入手部阶段；第三步：多物体 UMR 身体阶段（2026-10-08）

- **家具资产**：桌、椅、架、边桌、推车、垃圾桶、落地灯从各序列 GLB 取网格（`humoto_io.load_object_mesh`，比例 1），用 `hand_bench/prepare_humoto.export_object` 做 CoACD（阈值 0.04，≤32 块）。薄壁件 CoACD 会把空腔填上（假填充占包围盒 7–10%，手伸进架子或桶里会被挡），架子和边桌改用"表面片"分解：面细分到 ≤5 cm 后按面心 k-means 聚 96–128 类，每类取凸包（扁平类沿法向加厚 5 mm），假填充降到 2.6–2.7%；垃圾桶用 400 片表面片；推车保留 64 片 CoACD（5.9%）。`prepare_humoto_multi.py` 现在把序列里所有家具也放进 `multi_objects.json`（41 段有额外物体，共 109 件；第一步的纯手持列表在 `multi_objects_handheld.json`）。手部阶段加了包围球粗筛（`Robot.near_obj_geoms`），否则 400 片的家具让成对距离查询成为瓶颈。
- **多物体 UMR**（`prepare_umr_multi.py` → `data_humoto/umr_multi/`，每件额外物体写 `<obj>.xml` + `prop_<obj>.csv`，材质名逐件改名避免合并时冲突；`run_umr.py --scaling multi` + `configs/umr_g1_wuji_multi.json`）：给 UMR 打补丁（`solver.object_multi`）：所有物体各采 1024 个表面点拼成一个点集，接触项的最近邻自然取最近的物体；穿透模型每件物体一个自由 body，每帧把各自位姿写进去。接触点上限和穿透对上限从 128 提到 512，否则椅子、桌子会把手-杯子的约束挤掉。只对 41 段多物体序列跑 UMR，其余 59 段输入与基线相同，直接复用基线身体。
- **评测**新增 `scene_pen_mm / scene_pen_frac`：整个机器人（不含世界几何体）对所有物体的逐帧最深穿透（原有的 `obj_pen_*` 只看主物体）。
- 顺带修了一个 bug：`run_umr.py` 原来只在 `--scaling umr` 时把 `object_mesh_scale` 记为 s，其他变体都记 1.0；现在只有真实尺寸变体（`real`/`realhand`/`omni`）记 1.0。之前的 `--variant` 实验都用 `--scaling umr`，不受影响。

| 41 段多物体序列，按全部物体评测 | 仅主物体 | 第一步 手持物体 | **第二步 + 家具（final）** | 第三步 + 多物体 UMR |
|---|---|---|---|---|
| 指尖接触 F1 | 0.694 | 0.769 | **0.770** | 0.757 |
| 指节+掌心接触 F1 | 0.552 | **0.575** | 0.569 | 0.561 |
| 手-物穿透 (mm) / >5 mm 帧 | 5.05 / 0.31 | 2.14 / 0.11 | 1.18 / 0.056 | **0.61 / 0.040** |
| 全身-场景穿透 (mm) / >5 mm 帧 | 9.92 / 0.57 | 5.61 / 0.26 | 4.18 / **0.19** | **3.06** / 0.20 |
| 法兰位置 (cm) / 朝向 (°) | 1.99 / **3.8** | **1.86** / 4.9 | 2.01 / 5.3 | 2.08 / 5.5 |
| 关节 jerk / 根 jerk | **82 / 5.6** | 91 / 5.6 | 95 / 5.6 | 111 / 7.1 |
| 手指 jerk | **59** | 85 | 96 | 96 |

UMR 身体阶段本身（不接手部阶段）：全身-场景穿透 12.6 → 6.7 mm，手-物穿透 7.3 → 3.5 mm，但关节 jerk 106 → 147、腕误差 2.6 → 3.2 cm。逐段：第二步对第一步，手-物穿透 73%、场景穿透 80% 更好，F1 持平；第三步对第二步，手-物穿透 61%、场景穿透 56% 更好，F1 49%、关节 jerk 只有 12% 更好。所以第二步是稳定收益，定为 HUMOTO 的 final（`results_humoto/pipe_final -> pipe_v3_multi2`）；第三步把穿透再减一半，代价是关节 jerk +16%、根 jerk +27%，作为选项保留在 `pipe_v3_multi3` / `umr_wuji_multi`，对物理仿真更在意穿透时可换用。jerk 的来源推测是多物体时 UMR 接触/穿透活动集逐帧切换更多。全部表格 `results_humoto/pipe_summary_multi23.md`，图 `results_humoto/figures/*_multi23.png`。

### 6.9 OmniRetarget 的物体碰撞：CoACD 凸块对原生整网格（2026-10-09）

holosoma 自带示例的物体是箱子，碰撞直接用整个网格一个 geom，MuJoCo 按其凸包处理。我们的 `run_omni.py` 改成每个 CoACD 凸块一个 geom（与 UMR 共用）。消融 `run_omni.py --native-collision`（模型名加 `_native`，不覆盖原结果）在 6 种最不凸的 OMOMO 物体（大小桌、塑料箱、垃圾桶、木椅、白椅）各 3 段、Wuji 手、UMR 缩放上跑，两组都按 CoACD 几何评测（`compare.py --out _native`）。整网格凸包比 CoACD 多填实的体积：桌子 84%、塑料箱 70% 的包围盒，垃圾桶、木椅 35%，白椅 21%。

修正烘焙手后（2026-10-15）在三个数据集全量重跑了两种碰撞（lane `omni_wuji_umrscale` / `omni_wuji_umrscale_native`，`results*/summary_3ds.md`），都按 CoACD 几何评测：

| G1 + Wuji | OMOMO CoACD | OMOMO 凸包 | HUMOTO CoACD | HUMOTO 凸包 | HiPHI CoACD | HiPHI 凸包 |
|---|---|---|---|---|---|---|
| 物体穿透 (mm) / >5 mm 帧 | 10.1 / 42% | **2.1 / 9%** | 10.4 / 60% | **4.1 / 21%** | 13.5 / 47% | **10.5 / 33%** |
| 需要回退的段 | 359 | **39** | 71 | **26** | 32 | **22** |
| 地面穿透 (mm) | 4.3 | **1.6** | 4.2 | **2.0** | **2.4** | 3.8 |
| 手-物接触保持 | **0.97** | 0.83 | **1.00** | 0.98 | 0.96 | **0.97** |
| 腕位置误差 (cm) | **2.2** | 3.5 | **1.6** | 2.1 | **2.9** | 3.4 |
| 腕朝向误差 (°) | **32** | 37 | **39** | 41 | **47** | 49 |
| 手被推开的段（按手计） | **10** | 59 | 0 | 0 | 6 | **4** |
| 关节 / 根 jerk | **201 / 56** | 232 / 61 | **238 / 62** | 244 / 64 | **140 / 35** | 146 / 36 |
| 每帧耗时 (s) | 2.9 | **1.1** | 1.1 | **0.7** | 2.3 | **1.2** |

凸包让硬约束可行得多（OMOMO 回退 359 → 39），穿透因此很低、速度快一倍以上，但它实际上换了一个物体：凹处被填实，手进不去。OMOMO 上接触保持 0.97 → 0.83（落地灯 1.00 → 0.03、衣架 0.98 → 0.46），腕误差 2.2 → 3.5 cm，手被挡在真实表面之外的段 10 → 59，朝向和 jerk 也更差。HUMOTO 的物体是小件、近似凸，两者差别不大；HiPHI 的大件家具凸包仍需 22 次回退，穿透只降到 10.5 mm，凳子、衣架用凸包反而更穿。对人-物交互重定向，要的是手真正贴到物体上，所以主比较保留 CoACD，凸包作为"holosoma 默认配置"的对照列一起报告。图 `results/figures/sub17_woodchair_000_native.png`（修正后重渲染）：凸包时手悬在椅背外。

### 6.10 OMOMO：固定手指把手推到物体另一侧（2026-10-10）

`sub16_whitechair_002`（双手握椅背顶杆）里 UMR 的 Wuji lane 两只手在人手握住椅背的全程被推离人腕 13–14 cm，到了椅背另一侧（修正烘焙手之前 OmniRetarget 也有同样现象，重跑后腕偏移 5 cm、无被推开帧，见第 8 节）。原因不是源数据：人的腕离椅子 4 cm，平展的 SMPL-X 手指也没有穿过椅背。原因是 **rest pose 的固定手指 + 机器人-物体硬约束**：椅背顶杆只有 2 cm 粗，人是把杆握在拇指和四指之间，机器人手指钉死成微屈的"钩"，杆要落进钩里，线性化的非穿透约束从外面到不了那个位置，求解器只能把整只手挪到杆的另一侧（钩从那边套上去，或掌心贴到对侧）。UMR 的手还同时穿透 6 mm。`wrist_gap_stats.py` 统计（人腕距物体 < 10 cm 的帧里机器人腕离人腕 > 12 cm 的比例，535 段有接触）：

| lane | 平均腕偏移 (cm) | > 12 cm 帧比例 | 有一只手超过半程被推开的段数 |
|---|---|---|---|
| omni_brainco | 5.5 | 0.01 | 6 |
| omni_wuji | 5.6 | 0.02 | 10 |
| umr_brainco | 5.9 | 0.04 | 23 |
| umr_wuji | 7.4 | 0.09 | 62 |

（OmniRetarget 为 2026-10-15 修正烘焙手后的重跑，`results/wrist_gap_3ds.csv`；段数按手计。）

Wuji 比 BrainCo 严重（手更长、钩更深）。UMR 比 OmniRetarget 严重得多（62 对 10）：OmniRetarget 有腕关键点代价和交互网格，手挪到另一侧代价很大，而且硬约束不可行时它走回退链放松约束（UMR 被推开的 57 段里，OmniRetarget 有 44 段用了回退，这些段它的物体穿透 14.3 mm 对 UMR 7.2 mm），等于用穿透换腕位置；UMR 没有腕位置项，表面 / 接触项在杆的两侧几乎等价，带 slack 的逐帧线性化非穿透把手沿表面推过去后没有项把它拉回来，而且它从不放松碰撞。被推开的段清单和可视化：`reports/pushed_sequences_umr_wuji.md`。集中在细杆和薄板物体：椅子、桌子、箱子边沿、吸尘器杆。更准确的几何分析（把 rest pose 的手放到人腕位姿上算穿透、沿手轴平移扫描、第 8–20 帧的接近过程）见 `reports/whitechair_002_wrong_side.md`：直接压在杆上的是掌块，固定的手指把退路堵死，沿手轴 ±16 cm 内没有无碰撞位置；手指比人腕早 0.3 s 碰到椅背，硬约束从那一刻起把手沿椅背滑到另一侧。**对有手部阶段的数据集（HUMOTO / HiPHI）**：身体阶段的固定手指只是占位，不参与物体硬约束是合理的，手指和穿透由 AnyTeleop + v3 处理（见下文全量验证）。**对 OMOMO 不适用**：OMOMO 没有手指数据、没有后续手部阶段，身体阶段的输出就是最终结果，手指碰撞必须保留，否则手指穿进物体没有任何环节能修；下面两个变体在 OMOMO 上只作为定位原因的消融，不作为修法。OmniRetarget 用 `--fingers-free`（手掌、身体仍受约束），UMR 用 `configs/umr_g1_wuji_nofingercol.json`（`robots/g1_29dof_wuji_hand2_nofingercol.xml`，手指 geom 不碰撞，`run_umr.py --variant nofingercol`）。在该段上两种方法的腕偏移都从 13–14 cm 降到 5 cm，> 12 cm 的帧归零；按完整碰撞模型评测，UMR 的手-物穿透从 6.4 mm 降到 1.9 mm（手指不再被推进物体），OmniRetarget 为 0。代价是手指可能从物体里穿过（本段 > 5 mm 的帧为 0），要在接 AnyTeleop + v3 的数据集上才由手部阶段处理；OMOMO 的正式结果保持手指碰撞不变。

**OMOMO 的修法（保留手指碰撞）**：让 UMR 把手指当自由度解（`--variant free`，碰撞照常）没用（腕偏移 12.6 cm、76% 的帧仍被推开），因为 UMR 没有任何项会把手指张开让杆进来。可行的是两步：身体阶段手指不碰撞（手落到人腕处）→ `omomo_hand_repair.py` 用完整碰撞模型逐帧重解。试了两种重解：

- 只解手指（默认）：每帧每手 20 个手指关节，rest pose 先验 + 上一帧项 + 手/物非穿透带 slack + 关节限位（`refine_fingers` 去掉接触目标），手臂不动。全量（`results/run_omomo_repair_full.sh` → `pipe_handrepair`，`wrist_gap_omomo_repair.csv`）：腕偏移 6.0 cm，但手指碰撞下的穿透 6.48 mm，比原 lane（5.79）还差：手臂钉死时掌块已经压在箱子 / 垃圾桶上，光动手指推不出来。
- 手臂 + 手指联合（`--coupled`，**采用**）：每帧每手 7 臂 + 20 指关节一起解，腕位置 / 朝向锚到人手的 SMPL-X 规范手坐标系（`human_hand_frames`），rest pose 手指先验，上一帧 + 加速度项，手和前臂对物体非穿透带 slack，无接触目标（`refine_coupled`，w_tip=0、w_pose=50、w_rot=5，8 次迭代，每段约 6 s）。全量（`results/run_omomo_repair_coupled.sh` → `pipe_handrepair_coupled`，`wrist_gap_omomo_coupled.csv`；身体指标 `summary_handrepair_coupled.md` 由 `compare.py` 生成，跑完后补）：

| OMOMO 535 段有接触（同一次统计） | umr_wuji（原 final） | 两步（coupled） |
|---|---|---|
| 平均腕偏移 (cm) | 7.4 | **6.7** |
| 被推开（> 12 cm）帧比例 | 0.09 | **0.01** |
| 有一只手超过半程被推开的段数 | 62 | **5** |
| 手-物穿透 (mm) / > 5 mm 帧 | 5.79 / 0.44 | **1.23 / 0.06** |

逐物体穿透全部下降（小箱 4.4 → 0.1、大箱 5.1 → 0.4、桌子 7.1–7.3 → 0.5–0.6、椅子 5.5–6.6 → 1.1–2.7、垃圾桶 6.2 → 2.7 mm），被推开的帧只剩木椅 0.05、白椅 0.03。余下 5 段（`sub16_whitechair_009/037`、`sub17_woodchair_001/004/046`）是手指不碰撞的身体阶段本身就把手压进了椅背，联合重解只能拉回一只手。白椅子 002：腕偏移 5.5 cm、被推开帧 0、穿透 0.5–1.1 mm。已改为 OMOMO 的 final（`results/pipe_final -> pipe_handrepair_coupled`；`run_final_pipeline.sh omomo` 走这条链：`BODY_VARIANT=nofingercol` → `omomo_hand_repair.py --coupled`）。可视化：8095（原 lane / 手指不碰撞 / 两步 final / OmniRetarget）。

会不会给后面的手部阶段留下很大的初始穿透？在 HUMOTO 20 段子集上试了"身体阶段手指不碰撞 → AnyTeleop → v3 12 次迭代"（lane `umr_wuji_nofingercol` / `pipe_v3_nfcsub`，按完整碰撞模型评测）：

| 20 段子集 | 身体阶段（手指碰撞） | 身体阶段（不碰撞） | v3（手指碰撞的身体） | v3（不碰撞的身体） |
|---|---|---|---|---|
| 手-物穿透 (mm) / >5 mm 帧 | 2.8 / 0.22 | 5.9 / 0.40（最深 16 mm） | 0.89 / 0.060 | **0.73 / 0.049** |
| 法兰位置 (cm) / 朝向 (°) | 5.4 / 12.9 | 3.5 / 6.0 | 2.92 / 3.81 | **2.87 / 3.44** |
| 指尖接触 F1 | 0.27 | 0.33 | 0.773 | **0.784** |
| 全身物体穿透 (mm) | 4.3 | 8.2 | 1.8 | **1.4** |
| 关节 jerk | 341 | 225 | 251 | **207** |

身体阶段的初始穿透确实翻倍，但 v3 全部消掉了，最深的 16 mm 也归零；而且因为腕位姿更准（法兰 5.4 → 3.5 cm、朝向 12.9° → 6.0°），最终结果在所有指标上都更好，逐段 F1 胜率 80%。原因：第三阶段的非穿透是逐帧线性化约束，几厘米的穿透一两次迭代就能推出来；真正修不了的是腕被推到物体另一侧那种十几厘米的错误，手指不碰撞恰好避免了它。全量验证（2026-10-13，`run_nfc_full.sh <ds>`：UMR `--variant nofingercol` → AnyTeleop → v3 12 次迭代，HUMOTO 带多物体，按完整碰撞模型和全部物体评测，lane `umr_wuji_nofingercol` / `pipe_v3_nfcfull`，表 `pipe_summary_nfcfull.md`）：

| | HUMOTO final（手指碰撞） | HUMOTO 手指不碰撞 | HiPHI final（手指碰撞） | HiPHI 手指不碰撞 |
|---|---|---|---|---|
| 指尖接触 F1（召回 / 精度） | 0.771 (0.71 / 0.91) | **0.799 (0.74 / 0.92)** | **0.577** (0.53 / 0.76) | 0.572 (0.52 / 0.75) |
| 指节+掌心接触 F1 | 0.553 | **0.560** | 0.449 | 0.450 |
| 指尖-物体距离误差 (cm) | 1.33 | **1.19** | 1.74 | **1.73** |
| 手-物穿透 (mm) / >5 mm 帧 | 1.00 / 0.059 | **0.89 / 0.046** | **0.66 / 0.016** | 0.81 / 0.031 |
| 全身-场景穿透 (mm) / >5 mm 帧 | 2.95 / 0.163 | **2.77 / 0.139** | **3.06 / 0.236** | 3.38 / 0.267 |
| 法兰位置 (cm) / 朝向 (°) | 2.54 / 4.58 | **2.42 / 3.97** | 3.04 / 5.99 | **3.01 / 5.87** |
| 关节 jerk / 根 jerk | 197 / 19.4 | **167 / 16.2** | 134 / 10.3 | **113 / 8.7** |
| 身体阶段手-物穿透 (mm) | 4.7 | 8.2 | 3.1 | 13.0 |

三个数据集身体阶段本身的对比（含 OMOMO 上对 OmniRetarget CoACD / 凸包的四方对比、腕朝向、被推开段数）见 `reports/umr_finger_collision_3ds.md`（2026-10-17）：手指不碰撞让腕位置误差降 0.8–1.0 cm、腕朝向降 3–4°、被推开的段 57 → 5 / 11 → 0 / 4 → 2、关节 jerk 降三成，代价是身体阶段手指穿透 8–17 mm，必须接手部阶段。HUMOTO 上全面更好（逐段：F1 69%、法兰 73%、关节 jerk 94% 的段更好），已改为 HUMOTO 的 final（`results_humoto/pipe_final -> pipe_v3_nfcfull`，`run_final_pipeline.sh` 对 HUMOTO 默认 `BODY_VARIANT=nofingercol`）。HiPHI 上是取舍：平滑度明显更好（关节 jerk −16%，97% 的段），腕位姿略好，但接触 F1 −0.005、手-物穿透 0.66 → 0.81 mm、场景穿透 +0.3 mm。HiPHI 的物体是大件，身体阶段手指不碰撞时手会整个压进箱子、椅子（身体阶段穿透 13 mm，比 HUMOTO 的 8 mm 深），v3 没能全部推出来。HiPHI 保持原 final；需要更平滑的轨迹时可用 `BODY_VARIANT=nofingercol`。

### 6.11 其他

- 12 次迭代对 4 次：HUMOTO 小幅一致改善，HiPHI 持平，采用。
- UMR 手部表面权重提高只把腕朝向从 10° 降到 6°，腕位置 4 cm 的系统性后退不变。
- Wuji 手安装用官方 `_with_mount`（掌心法向 −y，同 Dex3 / Inspire 约定）；rew35860/holosoma 的 fork 用 Wuji Hand 1 且掌面转 90°。
- 关节限位：OmniRetarget 和 UMR 都是硬约束；v3 的 QP 也是。
- UMR 的 Clarabel 偶发 `InsufficientProgress`：默认 1e-8 容差过迭代导致数值崩溃，放宽到 1e-6/1e-7 并在迭代点可行时接受即可。

---

## 7. 数据准备

- **OMOMO**（`prepare_omomo.py`）：pickle 里的 SMPL-X 参数和 `obj_com_pos / obj_rot`；物体用中心化网格 + 每种物体的 `obj_scale` 中位数，CoACD 分解；身高用零姿态模板包围盒（sub16 1.723 m，sub17 1.737 m）。测试集 602 段，无手指数据，流水线只到阶段 1。
- **HUMOTO**（Mixamo GLB）：`../hand_bench/humoto_to_bvh.py` 导出 BVH → `hiphi2smplx --skeleton-map mixamo` 拟合 SMPL-X（16 维 betas 只拟合一次，`betas_humoto.npy`，身高 1.548 m，其余段固定体型 + 身体/手指 IK）→ `prepare_humoto.py` 转 Z-up、选接触帧最多的手持物体为主物体。100 段，32 256 帧，41 种物体。
- **HiPHI**：HOI 子集 7025 段里按 12 个物体类别各取 5 段（按指尖距物体 3 cm 内的帧占比排序，`../hand_bench/data_hiphi/selection.json`），共 60 段 109 788 帧，来自 13 个演员；SMPL-X 由 `hiphi2smplx --fit-shape-hands` **逐段**拟合 16 维 betas（没有按演员共享），90 Hz 按 3 抽帧。`prepare_hiphi.py` 写出 `data_hiphi/`。扩大规模很便宜（全部 7025 段 3 分钟），流水线每 100 段约 3.5 小时。
- 手部关键点、物体接触标签和 AnyTeleop 结果都在 `../hand_bench/data*/seqs/`、`results*/anyteleop/`。

---

## 7.1 对上游代码的改动

- **UMR**（子模块 `../UMR`，提交指针不变）：对 `scripts/retarget_smpl_to_humanoid_surface_vector.py` 的改动以补丁形式放在 `patches/umr_multi_object_retarget_prop.patch`：机器人侧独立物体轨迹 `retarget_prop_<obj>.csv`（真实尺寸按手锚定，6.1）和多物体加载 `solver.object_multi`（6.7）。克隆后执行 `git -C ../UMR apply ../omomo_bench/patches/umr_multi_object_retarget_prop.patch`；不打补丁时默认流程照常运行，只有 `--scaling realhand|multi` 需要它。
- **holosoma**（子模块）：未改动，所有修改都以运行时补丁的形式放在 `omni_entry.py`。
- **hiphi2smplx**（仓库外，github.com/noitom-robotics/hiphi2smplx @ 65e5131）：本地改动导出为 `patches/hiphi2smplx_mixamo_hands_hoi.patch`（`git apply` 即可），包括 Mixamo 骨架映射（`--skeleton-map mixamo`、`MIXAMO_MAP`）、`load_betas` 保留全部 16 维、体型 + 手指联合拟合（`shape_fitting.py`、`--fit-shape-hands`）、手指 IK（`hand_ik.py`）、HiPHI HOI 细化（`hoi_refine.py`、`hoi_cli.py`）。

## 8. 已知问题 / 注意

- **OmniRetarget 烘焙手的 bug（2026-10-14 发现，所有 OmniRetarget lane 都受影响，已于 2026-10-15/16 全部重跑）**：`run_omni.py bake_hands` 把手指 rest pose 下的手部网格作为固定 geom 写进 `wrist_yaw_link`，用的是 MuJoCo 编译后的 `geom_xpos/xmat`。编译后的 geom 坐标已经包含 MuJoCo 对网格的重心化（`mesh_pos/mesh_quat`），重新加载时原始 STL 又被重心化一次，等于平移/旋转应用了两遍：Wuji 每块手部网格平均偏 1.35 cm、最大 3.7 cm、朝向最大差 171°；BrainCo 平均 1.41 cm、最大 4.4 cm、122°。手掌、手指散成碎块（`reports/figures/omni_hand_bake_bug.png`，左两列旧烘焙，右两列修正后）。受影响：`results/omni_{wuji,brainco}_umrscale`、`results_humoto/omni_*`、CoACD 对原生整网格消融（6.9）、`--fingers-free` 试验、所有 UMR 对 OmniRetarget 的比较表（5.3、6.5、6.10 的 omni 行）和用 `compare.omni_model` 评测/渲染 OmniRetarget lane 的脚本；OmniRetarget 求解时的手-物碰撞和回退链（细杆被"拇指和四指夹住"）也是在这个错位的手上发生的。UMR lane、hand_pipeline、三个数据集的 final 都不用烘焙模型，不受影响。修法已进 `bake_hands`（按 `mesh_pos/mesh_quat` 补偿，修正后与带关节的 URDF 差 < 0.001 mm；输出带 `BAKE_MARK` 注释，`--setup` 会自动重建没有该标记的模板和逐物体 xml）。**已于 2026-10-15/16 全部重跑**（OMOMO 两只手各 602 段、HUMOTO 两只手各 100 段、HiPHI Wuji 60 段、三个数据集的凸包消融；旧结果保留为 `*_oldbake`，旧表为 `summary_oldbake.md` 等），5.3 / 6.9 / 6.10 的数字和白椅子报告的图、视频已更新。修正前 OmniRetarget 在白椅子段也被推到椅背另一侧，修正后消失。
- mujoco ≥ 3.14 让 holosoma_retargeting 关节不动（pybind 枚举比较问题），`hsretargeting` 钉 `mujoco==3.3.5`。
- holosoma 上游的 `no-activate-obj-non-penetration` 是空操作，`omni_entry.py` 做了真正的过滤；`task_name.split("_")[0]` 查身高字典，非 OMOMO 名字用 `--subject` 兜底。
- UMR 结果 npz 的 `robot_xml` 是每段临时文件、会被另一只手的运行覆盖，所有脚本按 `hand` 键重建模型；物体位姿不存，从 source × `smpl_scale` 重建（`realhand` lane 用 `obj_pos_robot`）。
- `eval_pipeline.py` 不带 `--out` 会覆盖默认的 `pipe_metrics.csv`。
- 评测里 trimesh 的 `on_surface` 对 9 万面的网格会吃 60 GB 内存，已分块；模型缓存每段清空，BLAS 线程钉 1。
- `pkill -f` / `pgrep -f` 在同一条命令里会匹配到自己的 shell，用方括号模式。
- viser 切换序列曾崩溃（加载时把顶点置空而播放线程还在渲染），已用锁和延后换句柄修复。
