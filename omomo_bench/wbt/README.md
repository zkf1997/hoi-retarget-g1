# HiPHI 多序列人-物交互跟踪策略（G1 + Wuji Hand 2，holosoma / IsaacSim）

用 `results_hiphi/pipe_final`（最终流水线：UMR 身体 → AnyTeleop 手指 → v3 联合细化）的 60 段 HiPHI 重定向结果，
在 holosoma 的 whole-body tracking（WBT）框架上训练**一个**策略同时跟踪 60 段带物体的动作，重点保证手部跟踪精度。

```
wbt/
├── convert_hiphi_to_holosoma.py   重定向结果 → holosoma 动作 npz + 每段物体 URDF
├── hiphi_wbt/                     holosoma 插件（--import-file 注册）
│   ├── robot.py                   69 DoF G1 + Wuji Hand 2 的 RobotConfig（103 个 body）
│   ├── sim_multiasset.py          IsaacSim 子类：每个 env 一个物体资产（env i ← 资产 i % 60）
│   ├── command.py                 FixedMotionCommand：env i 固定跟踪动作 i % 60；手部指标
│   ├── action.py                  手指目标 = 参考手指姿态 + 残差 的关节位置动作
│   ├── rewards.py                 指尖 / 腕 / 手指关节 跟踪奖励
│   └── presets.py                 实验预设 exp:g1-wuji-hiphi-wbt（及 -fixhand 消融）
├── eval_tracking.py               逐段评测：成功率、身体 / 腕 / 指尖 / 手指关节 / 物体误差
├── plot_training.py               tensorboard 曲线 → PNG
├── assets/robot/                  训练用 URDF（加了脚底接触点 link）；meshes → ../../urdf/g1_wuji_hand2/meshes
├── assets/objects/<III>_<seq>.urdf  每段的物体 URDF（生成，git 忽略）
└── data/motions/<III>_<seq>.npz   holosoma 动作数据（生成，git 忽略）
```

环境：venv `/home/kaizhao/software/venvs/hswbt`（`--system-site-packages` 基于 conda `sonic` 的 IsaacSim 5.1 / IsaacLab 0.54.3，holosoma 以 editable 安装）。
所有命令在仓库根目录执行，`PYTHONPATH=omomo_bench/wbt` 让 `hiphi_wbt.*` 可导入。

## 1. 数据转换

```bash
BENCH_DATASET=hiphi conda run -n umr python omomo_bench/wbt/convert_hiphi_to_holosoma.py   # --fps 50 --lane pipe_final
```

- 30 fps 的 `qpos`（7 + 69）线性插值到 50 fps（IsaacSim 200 Hz 物理、decimation 4）；用 `robots/g1_29dof_wuji_hand2.xml` 做 MuJoCo 正运动学得到 103 个 body 的世界位姿，速度用差分。
- 关节顺序 = 重定向结果的 MJCF 顺序；holosoma 按名字对齐（启动日志里 "order of the joint_names ... does not match" 是正常提示）。
- 物体：位姿是源数据 × 该段的 `smpl_scale`（和重定向用的缩放世界一致）；URDF 用显示网格做 visual、CoACD 凸块做 collision，网格按 s 缩放，质量 = 名义质量 × s³（`NOMINAL_MASS` 按物体类别给出）。
- 结果：60 段、182,922 帧（每段 2942–3269 帧 ≈ 60 s），`data/manifest.json` 记录序号、物体、缩放、质量。

## 2. 任务定义（`hiphi_wbt/presets.py`）

在 holosoma 的 `g1_29dof_wbt`（带物体的 WBT）上替换：

| 部分 | 做法 |
|---|---|
| 机器人 | `g1_wuji`：69 DoF、103 body（`collapse_fixed_joints=False` 保留指尖等固定 link）；身体关节的增益 / armature 沿用 G1 预设，手指 kp 0.6（拇指 CMC 1.0）、kd 0.03、armature 2e-4；开启自碰撞 |
| 场景 | 每个 env 一个物体资产：`sim_multiasset.IsaacSim` 用 IsaacLab `MultiAssetSpawnerCfg(random_choice=False)`，env i 得到第 i % 60 个 URDF；`replicate_physics=False`；`env_spacing=6 m` |
| 动作分配 | `FixedMotionCommand`：env i 永远跟踪动作 i % 60（和物体一致）；起始帧在段内均匀采样（10% 从第 0 帧开始），评测时都从第 0 帧 |
| 跟踪 body | holosoma 的 14 个 body + 10 个指尖 |
| 动作 | `ResidualReferenceJointPositionActionTerm`：手指关节（`^[lr]_`）的 PD 目标 = 参考手指姿态 + 0.25·effort/kp·a，零动作就是重定向手指姿态；身体关节同 holosoma（默认姿态 + 缩放动作） |
| 观测 | actor：holosoma 共享观测 + 物体在机器人系下的位置、朝向；critic：holosoma 带物体的 critic 观测 |
| 奖励 | holosoma 带物体 WBT 的全部项 + 指尖位置 exp（σ 0.08 m，权重 1.0）+ 腕位置 exp（σ 0.1，0.5）+ 手指关节 exp（σ 0.3 rad，0.5）；`undesired_contacts` 排除所有手部 link（手和物体接触是期望的） |
| 终止 | `BadTrackingZOnly`（含物体位置 0.3 m / 朝向 1.0 rad 阈值）；训练 episode 10 s |
| 随机化 | holosoma 带物体的随机化，物体质量系数改为 [0.8, 1.5] |

## 3. 训练 / 评测

```bash
PY=/home/kaizhao/software/venvs/hswbt/bin/python
# 训练（2040 env = 34 轮 × 60 段；日志 logs/HiPHI_WBT/<time>-g1_wuji_hiphi_wbt-locomotion/，含 tensorboard 与 model_XXXXX.pt）
PYTHONPATH=omomo_bench/wbt $PY holosoma/src/holosoma/holosoma/train_agent.py --import-file omomo_bench/wbt/hiphi_wbt/presets.py \
    exp:g1-wuji-hiphi-wbt logger:disabled --training.num_envs 2040 --training.headless True \
    --algo.config.num_learning_iterations 20000 --algo.config.save_interval 500
# 曲线
$PY omomo_bench/wbt/plot_training.py logs/HiPHI_WBT/<run>
# 逐段评测：60 个 env 各从第 0 帧播放到结尾（或被终止），输出 <run>/eval_tracking/model_XXXXX.{md,csv,_summary.json}
PYTHONPATH=omomo_bench/wbt $PY omomo_bench/wbt/eval_tracking.py --checkpoint logs/HiPHI_WBT/<run>/model_XXXXX.pt [--init-noise 0]
```

评测指标（每段取存活帧的均值）：成功率（整段播完且未触发 bad-tracking 终止）、跟踪帧占比、身体 / 腕 / 指尖（均值与各帧最差指尖）/ 手指关节 / 物体位置与朝向误差。

## 4. 吞吐量（RTX 4090，2040 env，2026-10-09）

| 配置 | steps/s |
|---|---|
| 默认（`env_spacing=0`，异构物体，手碰撞 + 自碰撞） | 3,000 |
| 关自碰撞 | 3,400 |
| 去掉手部碰撞几何 | 4,500 |
| 同一物体 + `replicate_physics=True` | 8,900 |
| **`env_spacing=6 m`（异构物体，保留手碰撞 + 自碰撞）** | **8,500** |
| 去手碰撞 + 关自碰撞 + 同一物体 | 14,300 |

`env_spacing=0` 时 2040 个机器人在空间上重叠，PhysX 宽相位要处理大量跨 env 的候选对（即使最终被过滤），拉开 6 m 后速度恢复到和同构场景一样；异构资产本身几乎不影响速度。手部 54 个碰撞体约占剩余开销的一半，训练时保留。
`presets.py` / `robot.py` 里的环境变量 `HIPHI_WBT_ENV_SPACING`、`HIPHI_WBT_SELF_COLL=0`、`HIPHI_WBT_NO_HAND_COLLISION=1`、`HIPHI_WBT_SINGLE_OBJECT=1` 只用于这类剖析。

## 5. 接触感知的改进（2026-10-09 下午，快速迭代中）

参考 FlashDexRetarget / UniCross / DexRepNet++ / Weave 的做法（详见对话记录），在插件里加了可开关的项，用环境变量 `HIPHI_WBT_VARIANT=geo,contact,future,bps,pen` 选择：

| 旗标 | 内容 | 维度 / 权重 |
|---|---|---|
| （总是开） | actor 观测加参考物体在躯干系下的位置 / 朝向（holosoma 原版 actor 完全看不到物体目标） | 3 + 6 |
| `geo` | 42 个手部关键点（腕 + 每指近 / 中 / 远 / 指尖 link 原点）到物体表面的带符号距离（截 ±10 cm）和法向（躯干系）+ 10 个指尖的接触标志 | 178 |
| `bps` | 每个腕系下 4³ 个格点（±8 cm）采样的 SDF（动态 BPS / 局部占据） | 128 |
| `future` | 0.1 s、0.2 s 后的参考关节角减当前关节角 + 参考物体位置 | 2 × 72 |
| `contact` | 交互奖励 $r=\tfrac1{10}\sum_f \exp(-\max(0,d_f-d_f^{ref})/0.02)$：只罚"比参考更远"；机器人侧带符号距离、参考侧无符号距离；参考距离 > 5 cm 的指尖不参与 | 权重 1.0 |
| `pen` | 手部关键点进入物体超过 1 cm 的深度之和 | 权重 −5 |
| `vobj` | 物体虚拟控制器课程（ResMimic / DexMachina）：对物体施加朝参考位姿的 PD 外力 / 力矩（kp 60、kd 8，按质量缩放，限 150 N / 15 Nm），增益随平均 episode 长度从 50 步到 300 步线性衰减到 0 | 训练早期物体不掉、策略能看到更长的轨迹 |

几何来源：`build_object_sdf.py` 对每个物体的缩放后 CoACD 凸块算 96³ 带符号距离网格（AABB + 15 cm 边距，open3d 逐块取最小），`hiphi_wbt/geometry.py` 在 GPU 上三线性采样（60 个网格 212 MB），参考侧距离在启动时对全部 182,922 帧预先算好。参考里 51.6% 的"帧-指尖"距离 < 1.5 cm。

**快速迭代**：`HIPHI_WBT_MOTIONS=<tag>,<tag>` 把选中的动作 / 物体 / SDF 软链接到 `data/subsets/` 后只训练这些序列；预设 `exp:g1-wuji-hiphi-fast`（1024 env，项目 `HiPHI_WBT_fast`，运行名带旗标）。`wbt_logs/launch_fast.sh <window> '<flags>' [iters]` 一键在 tmux 里启动。

### 5.1 诊断：参考回放（2026-10-09 晚）

`eval_tracking.py --replay`（所有关节开环跟踪参考姿态，`--dump-joints` 打印偏差最大的关节 / body）发现：

- **机器人自碰撞把手指推开**：开启自碰撞时，刚从参考姿态出发一步，拇指 IP、小指 MCP 等就偏离 10–16°，几步后 30–40°；手指 PD 放大 4 倍也没用。关掉自碰撞后第 0 步手指误差从 3.5° 降到 0.6°。原因是相邻指节的凸包碰撞体在关节处重叠（PhysX 只过滤直接相连的父子 link）。因此 `robot.py` 默认 `enable_self_collisions=False`（手-物碰撞不受影响），之前所有运行（全量 1402 次迭代、快速 A/B/C/D）都带着这个问题。
- 开环回放下躯干在 0.3 s 内就开始倾倒（根位置误差 13 步 70 mm、12°），和 holosoma 一样需要策略闭环。
- 手部质量 0.69 kg / 只（holosoma 半球手 0.12 kg），腕 pitch / yaw 力矩上限 5 Nm，重力矩约 0.6 Nm，不是瓶颈。

快速实验 A/B/D 在 20:30 左右重启为 A2/B2/D2（自碰撞关闭）；同时跑一个 holosoma 原版 `exp:g1-29dof-wbt-w-object`（1024 env，400 次迭代，`wbt_logs/stock_ref.log`）作为学习速度参照。

### 5.2 诊断：为什么 episode 一直只有十几步（2026-10-13）

快速实验 A2 / B2 / D2（2 段，1024 env，3000 次迭代）跑完后 episode 长度仍只有 16–22 步（0.3–0.4 s），策略噪声已经降到 0.33，评测从第 0 帧开始 1.2 s 内就倒。逐项排查（`term/bad_*` 指标记录每步触发哪个终止条件；踝和腕的 0.25 m 阈值各占一半）：

| 试验 | 300 次迭代时 episode 长度 |
|---|---|
| holosoma 原版 `g1-29dof-wbt-w-object`（1024 env） | 23（500 次 28） |
| 我们 69 DoF（A2 / E，E 已换 holosoma 脚碰撞 + 50 Nm 踝） | 7–8 且在下降 |
| 我们 **手指关节固定**（F，29 DoF 动作，`HIPHI_WBT_FROZEN_HAND=1`） | 17 且在上升（400 次 20） |

结论：不是脚（rev_1_0 URDF 的脚是 4 个 5 mm 小球，已换成 holosoma 的圆柱脚，`HIPHI_WBT_URDF=hscoll` 为默认）、不是踝力矩（35 → 50 Nm）、不是自碰撞（已关），而是 **40 个手指动作维度**：手指残差的范围是 holosoma 的 0.25 rad / 单位动作，初始噪声 1.0 时每步 ±14° 的随机手指动作让身体学不起来。起初怀疑是手指的探索噪声，但 G（`HIPHI_WBT_FIXHAND=1`，策略手指输出被忽略）和 H（手指残差缩到 0.2 倍）同样在 200 次迭代后 episode 缩短到 7–8 步，所以不是动作本身。

**真正原因是 holosoma 的 `action_rate_l2` 惩罚（权重 −0.1）对全部动作维度求和**：单位探索噪声下每维每步 E[(Δa)²] ≈ 2，69 维比 29 维多出 2.8 倍，使起始回报从原版的 −0.4 变成 −2.4、每步期望奖励为负，于是策略学会尽快摔倒结束 episode（episode 长度随训练下降的原因）。F 没有手指动作维度所以正常。修正：`action_rate_l2` 改为只对身体关节求和（`hiphi_wbt.rewards:penalty_action_rate_subset`，权重 −0.1），手指单独一项权重 −0.01；起始回报回到 −0.7。保留 `reference_action_scale=0.2` 和 `HIPHI_WBT_FIXHAND` 两个开关。实验 I（基线）、J（`geo,contact`）用修正后的奖励在跑。

### 5.3 结果（I / J / F，1500 次迭代）与自适应采样（2026-10-13 晚）

| 运行 | 1500 次时 episode 长度 | 备注 |
|---|---|---|
| F 手指冻结（29 DoF） | 30 | 身体误差 0.13 m（张开的手指计入跟踪 body） |
| I 基线（修正后奖励） | 17 | 评测从第 0 帧起 1.3 s 内倒 |
| J `geo,contact` | 18 | 和 I 无差别；指尖-物体距离误差 39 mm |
| holosoma 原版（参照） | 56（1200 次） | 但它的 clip 只有 6.5 s，我们的是 62 s |

I / J 评测仍然 0 成功；手指关节误差随训练上升（残差被用来做别的事），腕 / 肘 / 膝在倒地前就偏 20–30°。

原版和我们另一个差别是 **自适应起始帧采样**（`use_adaptive_timesteps_sampler=True`，BeyondMimic 式：按失败位置的 1 s 分箱直方图采样起点），我们因为 env 和动作绑定一直关着。现在 `FixedMotionCommand` 加了按动作限制的版本（`adaptive_per_motion`，默认开，全局分箱概率按各 env 自己的 clip 掩码后归一化，`HIPHI_WBT_ADAPTIVE=0` 关）。实验 K（基线）、L（`geo,contact`）、N（只用 `002_rock`，几乎不走动的序列，看能不能学会站着操作）在跑。

### 5.4 开环回放诊断（2026-10-13 深夜）

`eval_tracking.py` 新增诊断开关：`--freeze`（参考停在第 0 帧）、`--replay`（全关节开环 PD 跟踪参考）、`HIPHI_WBT_OBJECT_OFFSET`（把物体挪走）、`HIPHI_WBT_ZERO_INIT_VEL`、`--trace-joints`（逐步打印 q / 参考 / 速度 / 施加力矩）、`--poke-ref`、`--urdf` / `--fresh-robot` / `--finger-kp` / `--finger-kd` / `--finger-armature` / `--self-coll`。评测时 `HIPHI_WBT_MOTIONS` 现在会覆盖 checkpoint 里的动作目录（之前评测"002 单段"实际跑的是 checkpoint 里的 000）。

结论：
- holosoma 原版 G1 在开环（冻结参考、PD 保持）下也在 0.4 s 内倒下：这套增益是实机级别的"软"增益，靠策略闭环补偿，开环站不住是正常的。
- 我们的机器人多一个问题：**肩关节增益（kp 14）扛不住 0.92 kg 的手**。HiPHI 这两段里手臂外展 73°，肩 roll 的静态重力矩约 9–10 Nm，开环下肩 roll 在 0.4 s 内下垂 50°，腕误差随之超过 0.25 m 触发终止。策略要用约 1.5 个动作单位的常值偏置来补，拖慢学习。新增 `HIPHI_WBT_ARM_KP`（肩 / 肘 / 腕 kp 倍数，kd 按 sqrt 缩放），实验 P（×3 基线）、Q（×3 + `geo,contact`）对比 K（×1）。
- 手指外展关节（pinky / ring / middle abd）在开环下以 3–5 rad/s 漂向负极限，PD 力矩饱和在 0.2 Nm 也拦不住；手指 kp ×4、kd ×10、armature ×50、去掉手部碰撞、挪走物体都不改变这个现象，原因未明（怀疑 PhysX 关节摩擦按约束力缩放或 USD 转换的轴向），暂不影响身体学习的结论。
- 回放 F（手指固定）的 checkpoint 时发现旧 `_frozenhand.urdf` 已被重新生成逻辑删除，`frozen2` 评测失败；仅诊断用途。

### 5.5 继续排查学习速度（2026-10-14 凌晨）

| 运行 | 设置 | 300 次 | 备注 |
|---|---|---|---|
| K | 基线 + 自适应起点采样 | 13.8 | 900 次 16 |
| P / Q | 肩肘腕 kp ×3（身体仍是默认姿态 + 动作） | 更差 | 刚性手臂把手拉向默认姿态，腕终止更多，停掉 |
| R | 全身"参考 + 残差"动作（`HIPHI_WBT_RESIDUAL=all`） | 14.8 | |
| S | R + 肩肘腕 kp ×3 | 15.3 | |
| stock_on_hiphi | holosoma 原版 29 DoF G1（球手、原版箱子）直接跟踪我们的 002 npz（`stock_on_hiphi.py`） | 跑中 | 区分"数据难"还是"我们的机器人 / 手慢" |

K 策略（1000 次）在 002 第 0 帧起：30 步内根位置误差 < 4 cm，但肘 / 腕逐渐偏 25–38°，指尖误差 100–200 mm，35 步后躯干倾斜，64 步倒下。

### 5.6 二分定位：是数据还是机器人（2026-10-14 凌晨）

`stock_on_hiphi.py` 让 holosoma **原版** 29 DoF G1（球手、原版箱子、原版奖励 / 观测 / 终止）直接跟踪我们的 002 npz（加载器按名字取 body / 关节，69 关节文件里多余的手指列被忽略）：

| 运行（都在 002 单段、1024 env） | 300 次 | 之后 |
|---|---|---|
| 原版 G1 + 我们的 002 数据 (`exp:stock-on-hiphi`) | **30** | 450 次 44、650 次 **92** |
| 原版任务配置 + 我们的 69 DoF 机器人 (`exp:stock-on-hiphi-wuji`) | 9 | 不涨 |
| 原版任务配置 + 我们的机器人、手指冻结（29 DoF，手质量保留） | 11 | 不涨 |
| 我们的任务配置 K / R / S | 14–15 | 1500 次 17 |

结论：**数据没问题，任务配置也不是主因，问题在机器人模型本身**（即使手指冻结成 29 DoF 也学不动）。剩下的差异：手质量 0.92 kg 对 0.12 kg、103 个 body（固定关节不合并）、URDF 本身。正在跑两个进一步的二分：原版 G1 把球手质量改成 0.92 kg（`exp:stock-on-hiphi-heavyhand`），以及我们的机器人手指冻结 + 固定关节合并成 32 个 body（`HIPHI_WBT_COLLAPSE=1`）。

### 5.7 根因：空 link 在 IsaacSim 里默认 1 kg（2026-10-14 上午）

二分继续：原版 G1 把球手改成 0.92 kg 后学习速度不变（300 次 29 步），所以不是手的质量；我们的机器人手指冻结 + 固定关节合并成 32 个 body 后恢复到 23 步 / 300 次（500 次 42），说明问题在那些被合并掉的 link 里。用 `EVAL_PRINT_MASSES=1` 打印仿真里的质量：**总质量 44.4 kg（URDF 34.4 kg），10 个指尖 link `<link name="l_pinky_tip" />` 没有 `<inertial>`，IsaacSim 给了每个 1 kg、惯量 1e-4**，还有 imu / d435 / mid360 4 个空 link 也各 1 kg。即每只手末端挂着 5 kg：手指 PD（0.2 Nm 上限）扛不住 1 kg 指尖所以"漂移"，肩关节扛不住 5 kg 所以开环 0.4 s 下垂 50°，策略当然学不动；手指冻结 / 合并固定关节的实验只是把这些质量挪到了腕上。

修复：三个 URDF 里 14 个空 link 都加 5 g / 1e-7 的 inertial。修复后仿真总质量 34.46 kg，开环回放里小指外展误差 < 1°、肩 roll 只是重力下垂到 37° 后回弹。同时保留 `collapse_fixed_joints=True`（82 个 body：指尖和脚底接触点 `dont_collapse`）。实验 V（基线）、W（`geo,contact`）用修复后的模型重跑。

之前所有训练结果（I–U、全量 1402 次）都带着这 14 kg 的错误质量，只能作废。

### 5.8 修复后的对照（2026-10-14 下午）

| 运行（002 单段、1024 env） | 200 次 | 300 次 | 400 次 |
|---|---|---|---|
| 原版任务配置 + 修复后的机器人（手指冻结 29 DoF，`stockWF2`） | 13 | 22 | 32（550 次 50） |
| 我们的配置 X（69 DoF） | 16.5 | 16 | 17.7，之后不涨 |
| X + 原版箱子（Y） | 17.5 | 18 | 19.6 |
| X − 手部奖励项（Z） | 14 | 16.5 | 14 |
| 我们的预设、所有开关调成原版行为、69 维动作（AA） | 8 | 14.6 | |
| AA 但 29 维动作（手指冻结，AB） | 跑中 | | |

机器人模型修好后，原版任务配置能正常学；我们的任务配置在同一段上 300 次后停在 16–20 步。物体、手部奖励、观测、自适应采样都不是原因（Y / Z / AA）。剩下的嫌疑是 **69 维动作空间本身**：手指 40 维即使被忽略（fixhand）也进入 PPO 的高斯策略、熵和 KL，KL 偏高时自适应学习率被压回 1e-5（tensorboard 里 AA 的 actor LR 在 1e-5 和 7.6e-5 之间跳，原版稳定在 7.6e-5），且噪声 std 衰减更慢（0.79 对 0.57）。AB 用来验证。

### 5.9 继续二分（2026-10-14 下午 – 晚）

| 运行（002 单段） | 200 | 300 | 400 | 说明 |
|---|---|---|---|---|
| AB：AA 的开关 + 手指冻结（29 维） | 14.5 | 30 | 44.5 | **和原版一样快，甚至更快** |
| AC：X + `desired_kl` 0.02 | 17.9 | 15 | | 无效 |
| AE：X + 手指协同动作（41 维，`HIPHI_WBT_SYNERGY=1`） | 15.6 | 16.2 | | 无效 → 不是动作维数 |
| AH：AA + 去掉手部碰撞几何 | 8.9 | | | 和 AA 一样 → 不是手指-物体接触 |
| AI：AA + 手指 kp ×0.1 | 9.1 | | | 同上 |
| AJ：AA + 观测只含身体关节（`HIPHI_WBT_BODY_OBS=1`） | 跑中 | | | 手指 40 个关节位置 / 速度 / 动作是否是观测噪声 |
| AK：X + 协同动作 + 身体观测 + 手指相对参考 | 跑中 | | | |

AA 与 AB 的唯一差别是手指关节"PD 跟踪参考（动作被忽略）"还是"固定关节"，两者的差距（19 对 44）说明问题来自带手指关节本身。继续拆：

| 运行（AA 的开关 + …，002） | 300 | 400 | 550 |
|---|---|---|---|
| AA：69 维动作 | 14.6 | 19.4 | |
| AJ：AA + 观测去掉手指 | 同 AA | | | 
| AL：41 维（协同） | 17.9 | 25 | 30 |
| AM：29 维（手指只回放参考，`HIPHI_WBT_SYNERGY=body`） | 20.4 | 27.7 | 36 |
| AB：29 维、手指固定关节 | 30 | 44.5 | 57 |

结论：**两个因素叠加**。(1) 策略动作维数：69 → 41 → 29 维逐级变快（无效的手指维度抬高 KL / 重要性比的方差）；(2) 手指按参考闭合抓箱子本身让平衡更难（AM 对 AB）。AN（AM + 去掉手部碰撞）500 次 20.8、AO（AM + 我们的 1.1 kg 箱子）500 次 21.6，都不比 AM 好（去掉手碰撞后箱子失去支撑反而更难），说明剩余差距是"用手指按参考抓着箱子"这个任务本身更难，以及单种子实验 ±5 步的噪声，不再是 bug。

**当前默认配置**：修复质量的 82 body 模型、`HIPHI_WBT_SYNERGY=1`（41 维：29 身体 + 每手 6 个协同残差）、自适应起点采样、`geo,contact`。AF（基线）/ AG（`geo,contact`）/ AP（`geo,contact,vobj`）在两段上跑 2000 次用于选型，之后回到 60 段全量。

### 5.10 评测里的真正难点（2026-10-14 晚）

AG（协同动作 + `geo,contact`，两段，1000 次）从第 0 帧闭环评测：身体误差 43 mm、腕 18 mm、指尖 37 mm 都不错，但 **第 64 步两段同时被 `bad_object_pos` 终止**：两段都是箱子放在地上、第 32 步左右手到达箱子、第 35–64 步以 1.3 m/s 把 1.1 kg 的箱子提起来；策略跟上了身体却没把箱子抓起来（指尖-物体距离从 42 mm 跳到 148 mm，箱子留在地上，误差超过 0.3 m）。所以从第 0 帧起的成功率衡量的是"快速抓起地上的箱子"这一最难的瞬间。

对策：物体虚拟控制器课程（`vobj`，AP 在跑）、更长训练、可能的手指增益；评测里增加从中段（已持物）起步的指标。

### 5.11 第二个根因：60 个 env 实际全在用 env_0 的箱子（2026-10-15）

全量 12500 次的策略评测：成功 3/60，5 段晾衣架在第 12 步全部被 `bad_object_pos` 终止。`EVAL_PRINT_OBJECTS=1` 显示 **59/60 个 env 的物体质量都是 1.1021 kg（env_0 的 Box_A）**。原因：holosoma 先生成物体再 `clone_environments(copy_from_source=False)`，env_i 以 inherit 方式实时继承 env_0 的整个子树，于是每个 env 里既有自己的物体又有 env_0 的物体，而 IsaacLab 刚体视图的路径模式取自 env_0 的 body 名，所有 env 都把 env_0 的箱子当成"物体"，真正的物体成了不受控的多余刚体。之前所有多物体训练（含全量 train2）都受影响；两段快速实验（000、002 都是 Box_A）不受影响。

试过：深拷贝克隆（PhysX GPU 视图崩溃）、停用继承来的副本（视图模式只匹配 env_0）、统一 body 名（质量等属性仍被 env_0 的强意见覆盖）。最终修复（`sim_multiasset.py`）：物体放到 env 子树之外 `/World/hiphi_objects/env_i/object`（单位变换，holosoma 本来就按世界坐标 + env 原点写位姿），克隆之后生成；物体 URDF 改为 `assets/objects/<tag>/hiphi_object.urdf`、body link 统一叫 `object_body`。验证：0/60 质量不匹配，晾衣架不再第 12 步失败；现在最早的失败是足球（`bad_object_ori`，球滚动时朝向本来就无意义）。

全量训练用修复后的场景重启（`geo,contact,vobj`，协同动作），4.9 s/迭代，目录 `logs/HiPHI_WBT/20261015_125909-g1_wuji_hiphi_wbt_contact-geo-vobj_syn-locomotion`，tmux `wbt:train4`。

### 5.12 物体辅助力泄漏到评测（2026-10-20）

`vobj` 课程的虚拟 PD 力按"训练平均 episode 长度"衰减（50 步全开、300 步关）。全量训练里 episode 一直 40–50 步，所以 **整个训练过程辅助力都是满的**（`assist/scale` ≈ 1.0），策略从来不需要自己搬动物体；评测时 checkpoint 的配置里也带着 `object_assist`，物体在不被接触时也会被拉向参考位姿。现在 `eval_tracking.py` 默认 `HIPHI_WBT_NO_ASSIST=1`。

`model_30499` 去掉辅助力后（第 0 帧起）：跑完 4/60（有辅助时 6/60），物体对 5.3%（7.8%），身体 55 mm、指尖 61 mm 基本不变。真正身体+物体都跟上的只剩 052（长凳，物体 15 mm，95%），050 物体对率从 53% 掉到 6%。

结论：vobj 的衰减条件不对，应按迭代数衰减；需要用修复后的物体、不带 vobj（或按迭代衰减）重新训练。

## 6. 状态

- 2026-10-09 `exp:g1-wuji-hiphi-wbt` 全量训练（2040 env，约 5.8 s/迭代，运行目录 `logs/HiPHI_WBT/20261009_124245-g1_wuji_hiphi_wbt-locomotion`）跑到 1402 次迭代后手动中断，转入快速迭代；奖励 −2.4 → 0.55，episode 长度 11 → 14 步，checkpoint `model_01000.pt`。
- 2026-10-14 21:48 全量训练重启：`exp:g1-wuji-hiphi-wbt`，`HIPHI_WBT_SYNERGY=1 HIPHI_WBT_VARIANT=geo,contact`，2040 env，30000 次，7.6 s/迭代，tmux `wbt:train2`，日志 `wbt_logs/train_full2.log`，目录 `logs/HiPHI_WBT/20261014_194823-g1_wuji_hiphi_wbt_contact-geo_syn-locomotion`。AP（两段，+`vobj`）并行用于决定是否加课程。
- 2026-10-09 15:05 三个并行快速实验（2 段：`000_Bringing-lug_0311`、`002_Cause_to_move_in_place-rock_0108`，1024 env，3000 次迭代，三个同时跑时各约 11 s/迭代）：A 基线、B `geo,contact`、C `geo,contact,future,bps,pen`（500 次迭代时三者无差别、episode 仅 11 步，C 于 18:50 停掉换成 D `geo,contact,vobj`），目录 `logs/HiPHI_WBT_fast/20261009_1505*`、`_1506*`，日志 `wbt_logs/fast_fast{A,B,C}.log`。
- 对齐检查（`eval_tracking.py --zero-actions --dump-steps 12`）：第 0 步身体误差 8.5 mm、指尖 7.4 mm、手指关节 4.2°、物体 11.6 mm，说明动作数据、物体资产、env 原点和机器人初始化一致；零动作下机器人随后倒下属正常。
- `model_00500`（迭代 500）逐段评测成功率 0（全部在 0.5 s 内被 bad-tracking 终止），这是训练早期的预期结果；holosoma 的 WBT 一般需要上万次迭代。后续按 `model_XXXXX.pt` 逐个评测。
- `exp:g1-wuji-hiphi-wbt-fixhand`（手指纯回放，策略不控制手指）作为消融待跑（和主训练同时跑会把两者速度都减半，等主训练结束或有空闲 GPU 再跑）。
