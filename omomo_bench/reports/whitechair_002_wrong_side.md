# OMOMO `sub16_whitechair_002`：机器人手落到椅背另一侧的分析（2026-10-13）

序列：人双手从上方把手搭到椅背顶杆上并握住（顶杆直径约 2 cm，椅背是薄网），然后提起椅子。131 帧，人腕在 109 帧里离椅子 10 cm 以内，握住时腕离杆面约 4 cm。原始结果（UMR Wuji、OmniRetarget Wuji）里两只手全程落在椅背另一侧，腕离人腕 13–20 cm。

## 1. 现象（右手）

| lane | 握持期间腕偏移均值 | 偏移 > 12 cm 的帧 | 手-椅穿透（完整碰撞模型） |
|---|---|---|---|
| UMR Wuji（原） | 13.9 cm | 97% | 6.4 mm |
| OmniRetarget Wuji（原） | 13.0 cm | 93% | 0 |
| UMR Wuji，手指不碰撞 | 4.9 cm | 0% | 1.9 mm |
| OmniRetarget Wuji，手指不进约束 | 5.1 cm | 0% | 0 |
| UMR BrainCo（原，参考） | 4.6 cm | 0% | 1.6 mm |

素材：`figures/sub16_whitechair_002_report_right.png` / `_left.png`（6 帧近景，列 = 源 | UMR 原 | UMR 修 | Omni 原 | Omni 修）；视频 `figures/sub16_whitechair_002_report_right.mp4` / `_left.mp4`（全部 131 帧，同样五列）；`figures/sub16_whitechair_002_approach_right.png`（第 8–20 帧的接近过程）；`figures/sub16_whitechair_002_analysis.png`（曲线）。

## 2. 排除源数据问题

- 人腕离杆面 3.6–4.6 cm，没有进入椅子。
- OMOMO 没有手指数据，SMPL-X 的手是平展的；平展的 15 个手指关节在握持期间没有一个落进椅子的碰撞凸块。
- 两条 BrainCo lane 中 UMR 的那条在这段上是正常的，说明问题与手的几何有关，不是椅子或轨迹本身。

## 3. 手的几何：机器手比缩放后的人手长 34%

右手，腕到中指远端关节：人 17.3 cm，乘缩放 s = 0.766 后在机器人世界里是 13.2 cm；Wuji rest pose 下 17.8 cm，到指尖 19.7 cm。手指钉死在微屈的 rest pose，整只手是一个固定的"钩"，开口（掌心到指尖）约 8 cm，比顶杆粗得多，本来是能容下杆的。

## 4. 真正卡住的位置

把 rest pose 的 Wuji 手原样放到人腕的位置和朝向（规范手坐标系）上，用 MuJoCo 算与椅子凸块的穿透：

- **掌块**（法兰到指根的那段，7–14 cm）在 109 帧里有 95 帧穿透 > 5 mm，均值 10–15 mm。人握杆时掌心包着杆，杆的位置正好在机器人掌块里。
- **手指**只有 5 帧穿透 > 5 mm。所以"指尖更长撞上椅背"不是直接原因；在目标位姿上穿透的是手掌。

沿手的轴向把整只手前后平移，看哪里能无碰撞地放下（第 65 帧，`analysis.png` 右图）：

| 手指处理 | 无碰撞的平移量 |
|---|---|
| 手指钉死、参与碰撞 | ±16 cm 内**没有**任何位置 |
| 手指伸直、参与碰撞 | 只有退后 8 cm 一处 |
| 忽略手指（只算手掌） | 退后 ≥ 6 cm，或前进 8–12 cm |

含义：手掌想避开杆只需退后 6 cm，但退后以后杆就落进了手指弯曲的区域里，固定的手指又撞上去；往前则掌块和手指都压在杆上。沿着手的轴向没有任何一个位置同时满足掌和指都不碰杆。手指的长度和弯曲把人手"包住杆"的位置变成了对机器人不可行的位置，这一点和用户的判断一致，只是直接撞上的是手掌，手指负责把退路堵死。

## 5. 时间上怎么被推过去的（`approach_right.png`、视频第 8–20 帧）

- 人的手从上方下落，掌心搭上顶杆，手指垂到杆的另一侧。
- 机器人手指比人长、且指向下方，第 8–11 帧人腕还离椅子 16–18 cm 时，机器人的指尖已经先碰到了椅背。硬约束在这里就挡住了下落。
- 人继续往下、往前，腕锚点和表面对应项拉着机器人的手跟上，但手不能穿过杆；逐帧线性化的求解器只能沿着约束面滑，把手从杆的下方绕到椅背另一侧。到第 13 帧腕偏差已经 13–19 cm，之后整段维持在这个局部最优，因为回到正确一侧必须穿过杆。
- 两种方法机制相同。UMR 没有腕位置项，偏得更远（13.9 cm，而且手指被压进椅子 6 mm）；OmniRetarget 有腕关键点项，但硬约束优先，仍是 13 cm。

## 6. 修法与验证

OMOMO 没有手指数据，rest pose 的手指只是占位，让它参与硬约束是本末倒置。

- OmniRetarget：`run_omni.py ... -- --fingers-free`，手指 geom 不进物体约束，手掌和身体照常。
- UMR：`run_umr.py --variant nofingercol`，`robots/g1_29dof_wuji_hand2_nofingercol.xml` 的手指 geom 不碰撞。

结果见第 1 节表格：腕偏移回到 5 cm，被推开的帧归零；按完整碰撞模型评测，UMR 的手-椅穿透从 6.4 mm 降到 1.9 mm，OmniRetarget 为 0。视频和近景里，修复后的手搭在顶杆上、手指垂向另一侧，和人一致。

副作用是手指可能穿过薄物体（这段没有 > 5 mm 的帧）。在 HUMOTO 和 HiPHI 上手指由后面的 AnyTeleop + v3 重解并处理非穿透；在 HUMOTO 20 段子集上验证过"身体阶段手指不碰撞 → v3"反而全面更好（接触 F1 0.773 → 0.784，手-物穿透 0.89 → 0.73 mm，法兰 2.92 → 2.87 cm）。

## 7. 范围

`wrist_gap_stats.py` 在全部 OMOMO 上的统计（人腕距物体 < 10 cm 的帧里机器人腕偏移 > 12 cm）：有一只手超过半程被推开的序列，umr_wuji 62 / 535，umr_brainco 23，omni_wuji 10，omni_brainco 6（OmniRetarget 为修正烘焙手之后的重跑；修正前 19 / 11）；集中在椅子、桌沿、箱子边、吸尘器杆这类细杆和薄板。Wuji 比 BrainCo 严重，因为手更长、钩更深。修正手部几何后，OmniRetarget 在本段不再把手推到椅背另一侧（腕偏移 13 → 5 cm，见第 9 节），所以本报告第 1–6 节里 OmniRetarget 列的现象主要来自错位的手；UMR 的结论不变。

## 8. OMOMO 的修法（手指碰撞必须保留）

OMOMO 的身体阶段就是最终输出，手指碰撞不能关。让 UMR 把手指当自由度解但保留碰撞（`--variant free`）无效（12.6 cm / 76%）。采用两步：身体阶段手指不碰撞（`run_umr.py --variant nofingercol`，手落到人腕处），再 `omomo_hand_repair.py --coupled` 用完整碰撞模型逐帧联合重解手臂 7 关节 + 手指 20 关节：腕位置 / 朝向锚到人手的规范手坐标系，rest pose 手指先验，上一帧 + 加速度项，手和前臂对物体非穿透带 slack，无接触目标，8 次迭代。只重解手指、手臂不动的版本（`pipe_handrepair`）腕位置也修好了（6.0 cm），但掌块压在箱子 / 垃圾桶上时推不出来，穿透 6.48 mm 反而比原来差，故弃用。

本段（coupled）：腕偏移左 5.5 / 右 5.6 cm，被推开的帧 0，手指碰撞下平均穿透 0.5 / 1.1 mm，> 5 mm 的帧 2% / 0。全量 535 段有接触（`results/wrist_gap_omomo_coupled.csv`）：平均腕偏移 7.4 → 6.7 cm，被推开帧比例 0.09 → 0.01，受影响段数 62 → 5，手-物穿透 5.79 → 1.23 mm，> 5 mm 帧 0.44 → 0.06。身体指标（`results/summary_handrepair_coupled.md`，602 段）：物体穿透 5.4 → 1.3 mm、穿透帧比例 0.43 → 0.07，腕误差 3.45 → 2.26 cm，关节 jerk 273 → 189，根 jerk 14.2 → 10.3。余下 5 段（白椅 009/037、木椅 001/004/046）是不碰撞的身体阶段本身把手压进椅背，联合重解只能拉回一只手。结果为 `results/pipe_handrepair_coupled`（= `results/pipe_final`），可视化 8095。

## 9. OmniRetarget 烘焙手的 bug 与重跑（2026-10-14/15）

本报告最初版本里 OmniRetarget 两列的手是错位的：`run_omni.py bake_hands` 把 MuJoCo 编译后的 geom 位姿（已含网格重心化 `mesh_pos/mesh_quat`）写回 XML，重新加载时又重心化一次，每块手部网格偏 1–4 cm、转动最多 171°。修正后全部 OmniRetarget lane 已重跑，图、视频、曲线（`sub16_whitechair_002_report_{right,left}.{png,mp4}`、`_analysis.png`、`_approach_right.png`）都已用新结果重新生成；旧结果保存在 `*_oldbake`。

重跑后本段 OmniRetarget Wuji 腕偏移左 12.7 → 5.0、右 13.3 → 5.3 cm，被推开帧 0.90–0.96 → 0；BrainCo 左 15.1 → 5.0 cm。全量（535 段有接触）：被推开的段数 omni_wuji 19 → 10、omni_brainco 11 → 6，UMR 两条 lane 不变（62 / 23）。
