# 身体阶段对比总结：UMR 对 OmniRetarget、CoACD 对凸包、手指碰撞开关（2026-10-17）

本文汇总 2026-10-14 修复 OmniRetarget 烘焙手 bug 之后，在 OMOMO（602 段）、HUMOTO（100 段）、HiPHI（60 段）上重跑和重评的全部身体阶段对比，以及手指碰撞开关对完整流水线的影响。所有 lane 用同一套 CoACD 物体几何、完整碰撞模型评测；机器人为 G1 + Wuji Hand 2（OMOMO / HUMOTO 另有 BrainCo，排序相同）。数据文件：`results*/summary_3ds.md`、`summary_nfc.md`、`wrist_gap_{3ds,nfc}.log`、`wrist_orientation_{3ds,nfc}.log`、`pipe_summary_nfcfull.md`、`results/summary_handrepair_coupled.md`、`summary_repairfc.md`；脚本 `results/run_umr_vs_omni_3ds.sh`、`run_nfc_compare.sh`、`run_repair_fullcol.sh`。

## 0. 前提：OmniRetarget 烘焙手的 bug

OmniRetarget 写死 29 个关节，手只能作为固定 geom 挂在 `wrist_yaw_link` 下。我们的 `run_omni.py bake_hands` 把 MuJoCo 编译后的 geom 位姿（已含网格重心化 `mesh_pos/mesh_quat`）写回 XML，重新加载时原始 STL 又被重心化一次：每块手部网格平均偏 1.4 cm、最大 3.7–4.4 cm、朝向最多差 171°，手呈碎块状（`reports/figures/omni_hand_bake_bug.png`）。2026-10-14 修正（按 `mesh_pos/mesh_quat` 补偿，与带关节 URDF 误差 < 0.001 mm，输出带 `BAKE_MARK`），所有 OmniRetarget lane 重跑；旧结果保留为 `*_oldbake`。UMR、hand_pipeline、三个数据集的 final 不用烘焙模型，不受影响。修正前 OmniRetarget 在白椅子段也把手推到椅背另一侧，修正后消失。

## 1. 设置

| | 说明 |
|---|---|
| 缩放 | 统一 UMR 式：人和物体按 1.32 / T-pose 身高缩放；HiPHI 逐段身高（`data_hiphi/seqs_test_perseq.json`） |
| UMR | 带手指关节的完整 MJCF，手指钉在 rest pose；物体非穿透为带 slack 的硬约束（slack 代价 5000）；接触项权重 5、阈值 10 cm、最多 128 个 slot |
| OmniRetarget | 15 个关键点 + 交互网格，硬非穿透；不可行时回退链 default → pen_tol_0.02 → fingers_free → fingers_free_pen_tol_0.02 → no_obj_nonpen |
| 碰撞几何 | CoACD 凸块（默认）或 holosoma 默认的整网格凸包（`--native-collision`） |
| 评测 | `compare.py`（穿透、接触保持、腕/踝误差、jerk）、`wrist_gap_stats.py`（人手在物体 10 cm 内的帧里机器人腕离人腕 > 12 cm 的比例；任一手超过半程即"被推开的段"，按手计）、`wrist_orientation_stats.py`（相对人手规范手坐标系） |

## 2. UMR 对 OmniRetarget（身体阶段）

| 指标 | OMOMO Omni | OMOMO UMR | HUMOTO Omni | HUMOTO UMR | HiPHI Omni | HiPHI UMR |
|---|---|---|---|---|---|---|
| 物体穿透 (mm) / >5 mm 帧 | 10.1 / 42% | **5.4** / 43% | 10.4 / 60% | **4.5 / 32%** | 13.5 / 47% | **5.7 / 44%** |
| 需要回退的段 | 359 / 602 | 0 | 71 / 100 | 0 | 32 / 60 | 0 |
| 手-物接触保持 | **0.97** | 0.95 | **1.00** | 0.87 | 0.96 | 0.96 |
| 腕位置误差 (cm) | **2.2** | 3.5 | **1.6** | 3.7 | **2.9** | 3.0 |
| 腕朝向误差 (°) / >30° 帧 | 32 / 43% | **13 / 6%** | 39 / 59% | **10 / 5%** | 47 / 67% | **14 / 3%** |
| 手被推开的段 | **10** | 62 | **0** | 16 | 6 | **5** |
| 地面穿透 (mm) | 4.3 | **0.2** | 4.2 | **0.05** | 2.4 | **0.2** |
| 脚滑 (mm/帧) | **1.0** | 2.6 | **1.0** | 2.8 | **0.9** | 1.9 |
| 关节 jerk / 根 jerk | **201** / 56 | 273 / **14** | **238** / 62 | 258 / **19** | **140** / 35 | 193 / **10** |
| 每帧耗时 (s) | 2.9 | **0.25** | 1.1 | **0.15** | 2.3 | **0.10** |

OmniRetarget 的穿透按回退模式拆分（OMOMO）：默认硬约束 243 段 0.7 mm；pen_tol 103 段 12.6 mm；fingers_free 145 段 15.2 mm；两者叠加 51 段 19.1 mm；关掉物体约束 60 段 24.6 mm。总穿透的 97% 来自回退段（HUMOTO、HiPHI 同为 97%）。

结论：OmniRetarget 腕位置更准、接触保持更好、几乎不把手推到物体另一侧，硬约束可行时几乎不穿透；但有腕朝向项缺失（手绕前臂自由转）、回退后穿透大、地面穿透和根 jerk 高、慢 10–20 倍。UMR 从不失败、穿透稳定、腕朝向好 3–4 倍、地面和根节点干净，但腕位置差 1 cm、被推开的段多、脚滑多。

## 3. OmniRetarget：CoACD 凸块对整网格凸包

| G1 + Wuji | OMOMO CoACD | OMOMO 凸包 | HUMOTO CoACD | HUMOTO 凸包 | HiPHI CoACD | HiPHI 凸包 |
|---|---|---|---|---|---|---|
| 物体穿透 (mm) / >5 mm 帧 | 10.1 / 42% | **2.1 / 9%** | 10.4 / 60% | **4.1 / 21%** | 13.5 / 47% | **10.5 / 33%** |
| 需要回退的段 | 359 | **39** | 71 | **26** | 32 | **22** |
| 手-物接触保持 | **0.97** | 0.83 | **1.00** | 0.98 | 0.96 | **0.97** |
| 腕位置误差 (cm) | **2.2** | 3.5 | **1.6** | 2.1 | **2.9** | 3.4 |
| 腕朝向误差 (°) | **32** | 37 | **39** | 41 | **47** | 49 |
| 手被推开的段 | **10** | 59 | 0 | 0 | 6 | **4** |
| 关节 / 根 jerk | **201 / 56** | 232 / 61 | **238 / 62** | 244 / 64 | **140 / 35** | 146 / 36 |
| 每帧耗时 (s) | 2.9 | **1.1** | 1.1 | **0.7** | 2.3 | **1.2** |

凸包让硬约束容易可行，穿透低、回退少、速度快一倍，但它换了一个更大的物体：凹处被填实，手进不去。OMOMO 上接触保持 0.97 → 0.83（落地灯 1.00 → 0.03、衣架 0.98 → 0.46），腕误差 2.2 → 3.5 cm，被挡在真实表面外的段 10 → 59。HUMOTO 的小件近似凸，差别不大；HiPHI 的大件凸包仍需 22 次回退，凳子、衣架反而更穿。**决定：主比较用 CoACD，凸包作为"holosoma 默认配置"的对照列。**

## 4. "手被推到物体另一侧"

触发条件两家一样：手指固定在微屈的 rest pose，机器手比缩放后的人手长 34%，形成一个钩；人握细杆、薄边、碗沿时掌块压在物体上，弯着的手指堵住退路。UMR 上发生得多（OMOMO 57 段、HUMOTO 11、HiPHI 4，按段计），OmniRetarget 少（10 / 0 / 6），原因：

1. OmniRetarget 有腕关键点代价和交互网格，手挪到另一侧代价很大；UMR 没有腕位置项，接触项的定义是"机器人 slot 到配对物体点的向量等于人侧的向量"，拉回的力只来自手上少数 slot、权重 5。
2. OmniRetarget 不可行时放松约束而不是挪手：UMR 被推开的 57 段里 OmniRetarget 有 44 段用了回退（这些段它的穿透 14.3 mm 对 UMR 7.2 mm），等于用穿透换腕位置；UMR 始终保留完整碰撞。
3. UMR 的非穿透是逐帧线性化的硬约束：手指比人腕早约 0.3 s 碰到物体，约束把手沿表面推到另一侧；回程要穿过物体，线性化约束看不到绕行路径，迭代停在"从另一侧压着物体"的约束驻点（不是接触项的极小值，是 KKT 点）。

分布：OMOMO 集中在大箱子 11、塑料箱 8、小桌子 6、木椅 6、吸尘器 5、白椅子 5（细杆、边沿）；HUMOTO 几乎全是花瓶、碗、盘等薄壁容器；HiPHI 的大件平掌按压少见。手越长越严重：UMR 配 BrainCo 23 段、Wuji 57 段。清单与可视化：`reports/pushed_sequences_umr_wuji.md`，viser 8096 / 8097 / 8098。详细几何分析：`reports/whitechair_002_wrong_side.md`。

对后续阶段的影响（只看被推开的段）：HUMOTO 11 段经 v3 后法兰误差 8.1 → 2.4 cm、被推开帧归零、F1 0.82，残留的是 jerk（284 对手指不碰撞身体的 216）；HiPHI 4 段 final 法兰仍 6 cm、15% 帧被推开，大件物体修不全；OMOMO 没有后续阶段，必须在身体阶段解决。

## 5. UMR 身体阶段：手指碰撞开 / 关

`--variant nofingercol`：50 个手指 geom 不碰撞，手掌块、安装件、前臂照常碰撞。OMOMO 四方对比：

| 指标 | Omni + 凸包 | Omni + CoACD | UMR 完整碰撞 | UMR 手指不碰撞 |
|---|---|---|---|---|
| 物体穿透，全身 (mm) / >5 mm 帧 | **2.1 / 0.09** | 10.1 / 0.42 | 5.4 / 0.43 | 14.2 / 0.67 |
| 手部穿透，接触帧 (mm) | 0.05 | 0.05 | 5.8 | 17.0 |
| 需要回退的段 | 39 | 359 | 0 | 0 |
| 手-物接触保持 | 0.83 | 0.97 | 0.95 | **1.00** |
| 腕位置误差 (cm) | 3.5 | **2.2** | 3.5 | 2.5 |
| 腕朝向误差 (°) / >30° 帧 | 37 / 49% | 32 / 43% | 13 / 6% | **8 / 0.3%** |
| 手被推开的段 | 59 | 10 | 57 | **5** |
| 地面穿透 (mm) | 1.6 | 4.3 | **0.2** | **0.2** |
| 关节 jerk / 根 jerk | 232 / 61 | 201 / 56 | 273 / 14 | **181 / 10** |
| 每帧耗时 (s) | 1.1 | 2.9 | 0.25 | **0.07** |

HUMOTO / HiPHI（完整碰撞 → 手指不碰撞）：腕位置 3.7 → 2.7 / 3.0 → 2.2 cm，腕朝向 9.9 → 6.8 / 14.1 → 11.6°，被推开 11 → 0 / 4 → 2 段，关节 jerk 258 → 170 / 193 → 130，手部穿透 3.2 → 8.1 / 3.4 → 12.8 mm。逐段：腕误差在 91–94% 的段更小，关节 jerk 在 94–100% 的段更低，穿透在 91–98% 的段更大。

## 6. 跑完全部阶段后的穿透

| 完整流水线 | 身体阶段带手指碰撞 | 身体阶段不带手指碰撞 | 差 |
|---|---|---|---|
| HUMOTO（AnyTeleop + v3）手-物穿透 (mm) / >5 mm 帧 | 1.00 / 5.9% | 0.89 / 4.6% | −0.11 |
| 全身-物体 / 场景穿透 (mm) | 1.7 / 2.95 | 1.5 / 2.77 | −0.2 / −0.18 |
| HiPHI（AnyTeleop + v3）手-物穿透 (mm) / >5 mm 帧 | 0.66 / 1.6% | 0.81 / 3.1% | **+0.15** |
| 全身-物体 / 场景穿透 (mm) | 3.1 / 3.06 | 3.4 / 3.38 | **+0.3 / +0.32** |
| OMOMO（联合修复）手-物穿透，接触帧 (mm) / >5 mm 帧 | 1.24 / 6% | 1.23 / 6% | 0 |
| 全身-物体穿透 (mm) / >5 mm 帧 | 1.4 / 6.7% | 1.3 / 6.8% | 0 |

其他最终指标：HUMOTO F1 0.771 → 0.799、法兰 2.54 / 4.6° → 2.42 / 4.0°、关节 jerk 197 → 167；HiPHI F1 0.577 → 0.572、法兰 3.04 / 6.0° → 3.01 / 5.9°、jerk 134 → 113；OMOMO 腕误差 2.35 → 2.26 cm、jerk 201 → 189、被推开 6 → 5 段。逐段看，HiPHI 的 +0.15 mm 中位数为 0，集中在两三段大件物体（最大 +5.7 mm）。

## 7. 结论与推荐配置

- **第一阶段用 UMR。** 第三阶段会重解手臂和手指并锚定腕，OmniRetarget 领先的腕位置和接触会被重做；它落后的根 jerk、地面穿透、身体-物体穿透、回退带来的不稳定是第三阶段碰不到的。UMR 还快 10–20 倍、支持多物体、模型格式与后续阶段一致。
- **OmniRetarget 的比较用 CoACD**，凸包只作对照。
- **身体阶段关掉手指碰撞，保留手掌和前臂碰撞**，前提是后面有带完整碰撞、真正解手指的阶段。HUMOTO（v3）和 OMOMO（联合修复）已采用，最终穿透不增加；HiPHI 保留手指碰撞（大件物体上关掉会多 0.15–0.3 mm 穿透，换 16% 的 jerk 下降，看重平滑度时可改）。
- 不要走的方向：UMR 自己解手指但保留碰撞（`free`，被推开不变）、给身体阶段加大接触权重（`contact4`，穿透和 jerk 一起涨）。

## 8. 相关文件

- 报告：`umr_vs_omni_3ds.md`（三数据集英文表）、`umr_finger_collision_3ds.md`、`pushed_sequences_umr_wuji.md`、`whitechair_002_wrong_side.md`
- 图：`reports/figures/omni_hand_bake_bug.png`、`sub16_whitechair_002_report_{right,left}.{png,mp4}`、`sub16_whitechair_002_analysis.png`、`sub17_woodchair_000_native.png`
- 可视化：8090 / 8091 最终流水线（HUMOTO / HiPHI），8095 白椅子等 5 段（OMOMO 四 lane），8096 / 8097 / 8098 被推开的段
- lane：`omni_wuji_umrscale`、`omni_wuji_umrscale_native`、`umr_wuji`、`umr_wuji_nofingercol`、`pipe_handrepair_coupled`（OMOMO final）、`pipe_handrepair_coupled_fullcol`、`pipe_v3_nfcfull`（HUMOTO final）、`pipe_v3_it12`（HiPHI final）
