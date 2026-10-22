# hand_bench — AnyTeleop vs TopoRetarget on HUMOTO / HiPHI (G1 + BrainCo Revo2 / G1 + Wuji Hand 2)


> 2026-10-09 起本目录由 `humoto_bench` 改名为 `hand_bench`：它是手部重定向基准，HUMOTO 和 HiPHI 两个数据集共用同一套脚本（`BENCH_DATASET`），旧名字只反映了最先接入的数据集。
把 HUMOTO（人-物交互，Mixamo 骨架含 21 点手部关键点 + 物体 6DoF 轨迹，30 fps）中的**手部动作**分别用
**AnyTeleop**（`dex-retargeting` 的 vector 优化器，另跑 DexPilot 变体）和 **TopoRetarget**（`third_party/toporetarget`，
Wu et al. 2026 的非官方实现）重定向到 G1 的两款灵巧手，输出统一指标并用 viser 并列播放。两种方法都是**手部**重定向：
机器人手悬浮在人手腕位姿上（TopoRetarget 额外优化手基座位姿），躯干不重定向。

```
hand_bench/
├── humoto_io.py          HUMOTO GLB 解析：Mixamo 65 关节 FK、每手 21 关键点（MediaPipe 顺序）、物体位姿、GLB 内嵌物体网格
├── make_hand_urdfs.py    从 urdf/ 的两个 G1 URDF 切出手部子树 → urdf/g1_*/hand_{left,right}.urdf（根 = {side}_wrist_yaw_link）
├── hand_model.py         MuJoCo 手模型：主动关节 / mimic、21 个机器人关键点、FK、关节限位、手尺寸
├── prepare_humoto.py     序列筛选 + data/seqs/<seq>.npz + 手持物体网格与 CoACD 凸分解                     [handbench env]
├── run_anyteleop.py      dex-retargeting（vector = AnyTeleop；--optimizer dexpilot）                          [handbench env]
├── run_toporetarget.py   TopoRetarget 流水线（骨方向初始化 + 交互网格 Laplacian 优化）                          [handbench env]
├── compare.py            指标 → results/metrics.csv, results/summary.md                                      [handbench env]
├── viser_compare.py      viser 三车道：HUMOTO 人+物 | 方法 A 机器人手 | 方法 B 机器人手                        [handbench env]
├── bench_common.py       共用 I/O（手坐标系、物体位姿、结果读写）
├── data/     (git-ignored)  seqs/*.npz, objects/<obj>/{visual.obj, coacd_*.obj}, selection.json, handheld_objects.json
└── results/  (git-ignored)  anyteleop/, anyteleop_dexpilot/, toporetarget/ 下 <hand>/<seq>__<side>.npz；metrics.csv；summary.md
```

环境：`conda create -n handbench python=3.11 && pip install mujoco==3.3.5 dex-retargeting torch(cpu) trimesh viser coacd pyyaml pandas`
（`dex-retargeting 0.5.0` 带 pinocchio/nlopt）。`third_party/toporetarget` 为 `git clone https://github.com/0iui0/toporetarget`（commit 8c7fbcb，git-ignored）。

## 一键流程

```bash
conda activate handbench; export MUJOCO_GL=disable
python make_hand_urdfs.py                                  # 4 个 hand-only URDF
python prepare_humoto.py --max-seqs 100 --per-object 3     # 102 段（每种手持物体最多 3 段，按接触帧比例排序），52 个物体，~11 min
python run_anyteleop.py --hand brainco --workers 16;  python run_anyteleop.py --hand brainco --optimizer dexpilot --workers 16
python run_anyteleop.py --hand wuji    --workers 16;  python run_anyteleop.py --hand wuji    --optimizer dexpilot --workers 16
python run_toporetarget.py --hand brainco --workers 22;  python run_toporetarget.py --hand wuji --workers 22     # 各 ~10–25 min
python compare.py --workers 16                             # → results/summary.md
python viser_compare.py --hand brainco --methods anyteleop toporetarget      # http://localhost:8080
```

## 数据（`humoto_io.py`, `prepare_humoto.py`）

- HUMOTO 每段一个 GLB：`Armature`（scale 0.01，骨长单位 cm）下挂 Mixamo 骨架，手部 `Thumb1..4 / Index1..4 / …` 正好是
  MediaPipe 21 点布局（0 腕 + 5 指 × 4）；每个物体是根节点，TRS 逐帧动画（米）。所有 735 段共用同一副骨架。
  常量通道只有 2 个关键帧，按 30 fps 网格重采样。glTF Y-up → Z-up。
- **物体网格用 GLB 内嵌的那份**（`humoto_objects_0805/*.obj` 与 GLB 节点坐标轴约定不一致，内嵌网格无歧义）。同一物体在各段中网格一致。
- **手持物体**：包围盒最大边 ≤ 0.55 m（桌椅、货架、水槽等排除）。每手每帧记录最近手持物体及关键点到物体表面的最小距离。
- 序列选择：含手持物体的序列按"任一手距物体 < 2 cm 的帧比例"排序，每种物体最多 3 段，共 102 段、32 597 帧、46 种物体实际被抓握。
- 碰撞体：CoACD（threshold 0.05，≤ 16 块）；可视化用原网格。

## 坐标约定

- **人手规范坐标系**（`humoto_io.hand_frame`）：原点腕关节；x = 腕→中指 MCP；z = 小指 MCP→食指 MCP（拇指侧，正交化）；y = z × x。
  这与 `urdf/` 里两款手共用的 G1 腕约定一致：+x 远端、+z 拇指侧、掌心朝 +y（右手）/ −y（左手），所以人手关键点变换到该坐标系后可直接喂给机器人手。
- **机器人 21 关键点**（`hand_model.py`）：0 = 法兰（BrainCo `*_base2_link` / Wuji `*_mount`，都在 x = 0.0415）；拇指 CMC/MCP/IP/TIP、
  四指 MCP/PIP/DIP/TIP 取对应关节体原点与 `*_tip` 链接；BrainCo 手指没有 DIP 关节，DIP := 远节中点。
- 法兰放在人腕关节上（腕→中指 MCP：人 9.3 cm，机器人法兰→MCP 10.3 cm）。**手尺寸比** = 机器人/人的中指链长（腕→MCP→PIP→DIP→指尖，与弯曲无关）：
  BrainCo ≈ 1.00，Wuji ≈ 1.15。
- **主动自由度**：BrainCo 每手 6（拇指 metacarpal + proximal，四指 proximal；5 个 distal 按 URDF `<mimic>` 联动，拇指 1.0×、四指 1.155×，与实机一致）；
  Wuji 每手 20（全部）。

## 两种方法

| | AnyTeleop (`run_anyteleop.py`) | TopoRetarget (`run_toporetarget.py`) |
|---|---|---|
| 实现 | `dex-retargeting 0.5.0` `VectorOptimizer`（`--optimizer dexpilot` 为 `DexPilotOptimizer`） | `third_party/toporetarget`（非官方复现）`TopoRetargetPipeline` |
| 输入 | 人手 5 个腕→指尖向量 × 尺寸比 | 人手 21 点（不缩放）+ 物体网格 + 物体位姿（都在人手规范坐标系） |
| 目标 | Huber(机器人法兰→指尖向量 − 目标)，+ 对上一帧的正则；SLSQP，关节限位为约束；mimic 由 `MimicJointKinematicAdaptor` 处理 | 阶段 1：骨方向能量 + 时序平滑（L-BFGS-B）；阶段 2：交互网格 Laplacian 能量（Delaunay on 21 点 + 50 个物体表面采样）+ 骨方向 + 时序平滑 + 穿透软罚 |
| 优化变量 | 主动关节 | 主动关节 + 手基座 6DoF（位置 + 四元数） |
| 低通滤波 | 关（alpha=1），官方 teleop 配置为 0.2 | 无 |
| 耗时 | ≈ 1–2 ms/帧 | ≈ 0.15–0.3 s/帧（单核，无并行争用时） |

对上游 toporetarget 代码的改动（全部在 `run_toporetarget.py` 内以子类/猴子补丁方式实现，未改子模块）：

1. `RobotHand` 用 `hand_model.HandModel`（MuJoCo FK）实现，`base_pos` 即法兰（关键点 0）；
2. 关节限位作为 L-BFGS-B 的 `bounds` 传入（上游调用 `scipy.optimize.minimize` 时**没有**限位）；
3. `_estimate_joints_from_keypoints` 上游硬编码每指 3 关节（对 6 自由度的 BrainCo 会越界），改为张开手初值；
4. `compute_laplacian` 换成数值等价的向量化版本（上游为 Python 双重循环，快 ~5×）。

已知局限（属于该非官方实现）：穿透罚项只作用于"关键点—物体采样点距离 < 2 mm"，对手指**几何体**穿透没有约束；
基座姿态从单位四元数出发、能量对整体旋转不敏感，常收敛到腕部大幅旋转的局部极小（见 `base_rot_deg`）；物体表面采样每帧随机（结果不完全可复现）。

## 指标（`compare.py`）

在"交互帧"（人手距物体 < 10 cm）上、把机器人手放在各方法自己的法兰位姿上计算：

| 指标 | 含义 |
|---|---|
| `pen_depth_mm` / `pen_frac` | 手部碰撞几何体与物体 CoACD 凸块的最深穿透（`mj_geomDistance`）均值 / 穿透 > 5 mm 的帧比例 |
| `contact_recall` / `precision` / `f1` | 人指尖距物体表面 < 1.5 cm 记为接触；机器人指尖同阈值；逐指逐帧统计 |
| `tip_obj_dist_err_mm` | \|机器人指尖-物体距离 − 人指尖-物体距离\| |
| `tip_pos_err_mm` | 机器人指尖 vs 按尺寸比缩放的人指尖（手坐标系）；`pinch_err_mm` 拇-食指尖距误差；`inter_tip_err_mm` 10 对指尖距误差 |
| `base_drift_mm` / `base_rot_deg` | 法兰相对人腕的平移 / 旋转（AnyTeleop 恒为 0） |
| `q_vel_mrad` / `q_jerk_mrad` / `limit_sat` | 关节速度、二阶差分（mrad/帧）、处于限位的关节样本比例 |
| `runtime_ms` | 每帧求解耗时（TopoRetarget 是 22 进程并行时测得，偏高） |

## 结果

102 段（`data/selection.json`，每种手持物体 ≤ 3 段）× 左右手 = 204 个 (序列, 手) 样本，46 种物体；2026-09-25 跑完。
完整表（含逐物体 contact_f1 / 穿透）见 `results/summary.md`，逐样本见 `results/metrics.csv`。

**G1 + BrainCo Revo2（每手 6 主动自由度 + 5 mimic）**

| method | n | pen_depth_mm | pen_frac | contact_recall | contact_precision | contact_f1 | tip_obj_dist_err_mm | tip_pos_err_mm | pinch_err_mm | inter_tip_err_mm | base_drift_mm | base_rot_deg | q_vel_mrad | q_jerk_mrad | limit_sat | runtime_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anyteleop | 204 | **9.089** | **0.713** | **0.864** | **0.883** | **0.856** | **5.947** | **15.9** | **12.9** | **8.840** | **0.000** | **0.000** | 3.329 | 1.690 | 0.060 | **1.515** |
| anyteleop_dexpilot | 204 | 9.380 | 0.715 | 0.815 | 0.873 | 0.821 | 6.573 | 17.4 | 14.6 | 9.529 | 0.000 | 0.000 | **3.177** | 1.967 | **0.058** | 1.744 |
| toporetarget | 204 | 11.3 | 0.748 | 0.617 | 0.831 | 0.702 | 11.9 | 27.6 | 23.0 | 13.3 | 31.3 | 40.3 | 3.240 | **1.522** | 0.155 | 367.9 |

**G1 + Wuji Hand 2（每手 20 自由度）**

| method | n | pen_depth_mm | pen_frac | contact_recall | contact_precision | contact_f1 | tip_obj_dist_err_mm | tip_pos_err_mm | pinch_err_mm | inter_tip_err_mm | base_drift_mm | base_rot_deg | q_vel_mrad | q_jerk_mrad | limit_sat | runtime_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anyteleop | 204 | **6.162** | **0.466** | **0.653** | **0.932** | **0.741** | **8.140** | **1.059** | **0.455** | **1.007** | **0.000** | **0.000** | 3.233 | 1.687 | **0.021** | **1.716** |
| anyteleop_dexpilot | 204 | 6.647 | 0.493 | 0.633 | 0.921 | 0.729 | 8.164 | 2.422 | 5.697 | 2.841 | 0.000 | 0.000 | 3.427 | 2.247 | 0.027 | 2.957 |
| toporetarget | 204 | 12.9 | 0.761 | 0.482 | 0.811 | 0.613 | 18.9 | 41.9 | 38.9 | 21.6 | 56.9 | 64.3 | **2.862** | **1.175** | 0.078 | 601.8 |

读法：
- **AnyTeleop（vector）在两款手上都是穿透最小、接触保持最好、指尖跟踪误差最小的方法**，且每帧 1–2 ms。DexPilot 变体与之接近，拇指投影先验在 Wuji 上略增加 pinch 误差。
- Wuji 20 自由度能把缩放后的人指尖几乎精确复现（`tip_pos_err` 1 mm），但接触召回反而低于 BrainCo：Wuji 手比人手大 15%，法兰固定在人腕上时指尖按比例外移，离物体更远；BrainCo 尺寸比 ≈ 1，指尖落点与人一致。
- 这里的 TopoRetarget 是**非官方复现**，结果弱于 AnyTeleop：基座 6DoF 自由度常收敛到腕部大幅旋转（平均 40°/64°、平移 3–6 cm）的局部极小，穿透罚项只约束关键点—采样点距离而非手指几何，交互网格的 Laplacian 项也不足以把手拉回物体表面。它唯一占优的是时序平滑（`q_jerk`）。因此这组数字反映的是该实现在 HUMOTO 上的表现，不能直接当作论文方法的上限。
- 穿透均值 6–13 mm 对两种方法都偏大，主要来自大物体双手抱持（woven_basket、wash_tub、step_stool：人手指本身就贴着/陷入物体表面，HUMOTO 的手部数据与物体之间已有 ~1 cm 量级穿透），这是数据源限制。

## HiPHI（`BENCH_DATASET=hiphi`）

同一套脚本也跑了 HiPHI 的 HOI 子集（`hiphi_io.py` + `prepare_hiphi.py`；`bench_common.py` 按环境变量 `BENCH_DATASET=hiphi`
切到 `data_hiphi/`、`results_hiphi/`，`run_*.py` / `compare.py` / `viser_compare.py` 不变）：

```bash
export BENCH_DATASET=hiphi MUJOCO_GL=disable
python prepare_hiphi.py --per-category 5 --workers 20      # 7025 段 HOI 原始动作全部解析（~3 min），每类物体取接触帧比例最高的 5 段
python run_anyteleop.py --hand brainco ...; python run_toporetarget.py --hand brainco ...   # 同上
python compare.py --workers 20; python viser_compare.py --hand brainco
```

- 数据：`/home/kaizhao/projects/HiPHI/data/<frame>/<lu>/<motion_id>/`，55 关节 BVH（Y-up，cm，90 Hz，ZXY 欧拉；位置通道即局部平移），
  手指每指 3 关节 + End Site 指尖 → 直接得到 MediaPipe 21 点；`object_tracks/<obj>.csv`（米，XYZW）+ `object_meshes/<mesh_id>.obj`（cm）。
  只用非镜像的原始动作；90 Hz 按 3 抽帧到 30 Hz。7 段骨架缺 `Spine2` 的动作被跳过。
- 物体是家具/容器尺度（球 24 cm、箱子 30–70 cm、桌椅 50–90 cm、瓶子 34 cm、拖把 137 cm），每段恰好一个，全部作为交互物体；
  BVH 关键点在手内部，接触阈值用 3 cm 排序。选中 60 段、109788 帧（30 Hz）、13 个物体网格；
  类别分布 {'ball': 5, 'bench': 5, 'bottle': 5, 'box': 5, 'bucket': 5, 'chair': 5, 'clothrack': 5, 'mop': 5, 'soccerball': 5, 'stepstool': 5, 'table': 5, 'trashbin': 5}；接触帧比例中位数 0.93。
- 手尺寸比（中指链长）：BrainCo ≈ 1.00–1.05，Wuji ≈ 1.15–1.21（HiPHI 演员手比 HUMOTO 角色略小）。

### HiPHI 结果

60 段（12 类物体各 5 段）× 左右手 = 120 个样本，13 个物体网格，平均每段 1830 帧（30 Hz），交互帧占 86%，人指尖处于接触状态的比例 50%。
完整表见 `results_hiphi/summary.md`，逐样本见 `results_hiphi/metrics.csv`。

**G1 + BrainCo Revo2**

| method | n | pen_depth_mm | pen_frac | contact_recall | contact_precision | contact_f1 | tip_obj_dist_err_mm | tip_pos_err_mm | pinch_err_mm | inter_tip_err_mm | base_drift_mm | base_rot_deg | q_vel_mrad | q_jerk_mrad | limit_sat | runtime_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anyteleop | 120 | **18.3** | 0.876 | **0.819** | 0.669 | **0.694** | **9.610** | **26.8** | **11.9** | 20.2 | **0.000** | **0.000** | 6.118 | 4.954 | **0.058** | **0.913** |
| anyteleop_dexpilot | 120 | 18.5 | **0.875** | 0.808 | 0.650 | 0.673 | 10.1 | 28.1 | 12.8 | **19.7** | 0.000 | 0.000 | 6.071 | 5.373 | 0.063 | 1.263 |
| toporetarget | 120 | 23.9 | 0.877 | 0.560 | **0.694** | 0.580 | 14.4 | 35.6 | 29.7 | 21.0 | 47.1 | 55.2 | **5.157** | **2.897** | 0.344 | 148.7 |

**G1 + Wuji Hand 2**

| method | n | pen_depth_mm | pen_frac | contact_recall | contact_precision | contact_f1 | tip_obj_dist_err_mm | tip_pos_err_mm | pinch_err_mm | inter_tip_err_mm | base_drift_mm | base_rot_deg | q_vel_mrad | q_jerk_mrad | limit_sat | runtime_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anyteleop | 120 | 23.4 | 0.924 | **0.734** | **0.811** | **0.743** | **7.139** | **2.773** | **0.674** | **2.316** | **0.000** | **0.000** | 6.018 | 4.866 | **0.033** | **1.465** |
| anyteleop_dexpilot | 120 | **23.4** | 0.924 | 0.725 | 0.799 | 0.732 | 7.270 | 3.717 | 4.750 | 3.360 | 0.000 | 0.000 | 6.856 | 6.008 | 0.034 | 2.362 |
| toporetarget | 120 | 24.2 | **0.883** | 0.564 | 0.718 | 0.610 | 12.6 | 34.9 | 26.3 | 19.4 | 51.3 | 51.0 | **5.798** | **2.858** | 0.042 | 532.5 |

读法：
- 结论与 HUMOTO 一致：**AnyTeleop（vector）接触保持最好（F1 0.69 / 0.74）、指尖-物体距离误差最小、指尖跟踪最准、最快**；DexPilot 变体几乎相同。
  TopoRetarget 仅在时序平滑上占优，接触 F1 低 0.1–0.13，基座依旧漂到 5 cm / 50° 量级，BrainCo 上 34% 的关节样本顶在限位。
- 穿透整体比 HUMOTO 大一倍以上（18–24 mm vs 6–13 mm），且三种方法接近：HiPHI 的物体是家具/容器尺度，人手是整掌贴着箱子、椅背、台阶推拉抱举，
  BVH 手部关键点在手掌内部、物体网格又是粗糙的扫描/建模件，源数据本身手掌与物体就有 1–3 cm 重叠；手部重定向无法修正这一点。
  对 HiPHI 更有区分度的指标是接触保持与指尖-物体距离误差，而不是穿透深度。
- 逐物体：球类、台阶凳、垃圾桶上 AnyTeleop 的接触 F1 0.8–0.9；晾衣架（细杆）和拖把（细柄）对所有方法都难（F1 0.3–0.5），
  因为 BrainCo 6 自由度、Wuji 指节较粗，难以复现人手握细杆的闭合程度。TopoRetarget 在 Box_A_1 / Table_A_1 / StepStool（大平面）上与 AnyTeleop 持平或略好，
  在球类上明显更差（F1 0.5 vs 0.8–0.9、穿透 33–38 mm）。
- Wuji 的指尖跟踪误差只有 2.8 mm（20 自由度能精确复现缩放后的人手），但穿透反而比 BrainCo 大 5 mm：手大 15–20%，法兰固定在人腕上时整只手更深地插进物体。

## viser（`viser_compare.py`）

三条车道沿 +y 排开：HUMOTO 人体骨架（黑）+ 手部关键点（红）+ 全部物体 | 方法 A 的两只机器人手 + 物体 | 方法 B。
GUI：序列、机器人手（brainco/wuji）、播放/暂停、速度、帧、车道间距、车道显隐。远程用 `ssh -L 8080:localhost:8080`。
