# 最终实验矩阵

> 版本：公开同步 v1.3
>
> 同步日期：2026-10-05；已核对本地截至 2026-08-31 的现有产物，不代表新增实验。
> 项目：**Parameter Identification and Human Body Characterisation for Human-Robot Collaboration**

## 1. 用途与证据标签

本矩阵用于统一论文 Results、Discussion、答辩 PPT 和后续补充实验。它回答四个问题：

1. 每组实验验证什么研究问题；
2. 哪些数据属于独立重复，哪些只是工程试运行；
3. 最终结果是通过、混合、阴性、探索性还是被阻断；
4. 论文中可以写出多强的结论。

证据标签定义如下：

| 标签 | 含义 | 写作规则 |
|---|---|---|
| `正式通过` | 冻结协议下的 validity 与 replication 合同均通过 | 可作为主要正面结论 |
| `正式混合/阴性` | 实验完整执行，但至少一项冻结门限失败 | 必须保留失败门限，不得只汇报改善项 |
| `确认性完成` | 多轮或配对实验完成并能支持限定结论，但未声明完整正式合同通过 | 可作为主要或支持性证据，结论需限定范围 |
| `探索性` | 用于机制发现、参数选择或真实数据案例分析 | 不得泛化为正式性能证明 |
| `工程验证` | 验证数据链路、同步、部署或运行可行性 | 不得等同于算法准确度证据 |
| `被阻断` | 预先条件不满足，因此目标终点未计算 | 应报告阻断原因，不得补写推断结果 |

论文位置缩写：`M` = Methods，`R1` = baseline/component results，`R2` = temporal/kinematic results，`R3` = occlusion/multi-view results，`R4` = real-world results，`D` = Discussion，`A` = Appendix。

方法与指标缩写见[当前进度页](project_progress_cn.md#3-方法与指标说明)。CTS5 是五视角偏置校正后的稳健标量骨长融合；FS-CTS5 扩展到双侧上臂、前臂、大腿、小腿八个主要肢段，不表示完整骨架已验证。

## 2. 正文实验矩阵

| ID | 研究问题 | 数据与独立单位 | 方法对比 | 主要终点 | 最终结果与判定 | 证据等级 | 论文位置 |
|---|---|---|---|---|---|---|---|
| E01 | BlazePose + ZED baseline 在干净静态仿真中的基本准确度和覆盖率如何？ | 女性人物静态 10 s，59 帧；单序列基线 | BlazePose Full + `median_7x7` 对 Isaac GT | detection、13-joint valid rate、MPJPE、jitter、limb MAE | detection 与 valid rate 均 100%；MPJPE `99.23 mm`，jitter `4.96 mm`，limb MAE `23.99 mm`。用于建立参考点，不表示跨场景性能 | 确认性完成 | M、R1 |
| E02 | 距离、BlazePose 模型和处理阶段如何影响准确度与实时性？ | 距离扫描 31 次；Lite/Full 配对 18 次、978 帧；性能拆分多次重复 | 距离档位；Lite vs Full；采集/推理/采样/约束分解 | 可见边界、MPJPE、jitter、limb MAE、latency、FPS | 640×360 全图边界为 `[3.45 m, 3.50 m)`；Full 准确度略优，Lite 推理更快，故 Full 用于离线、Lite 用于 live；推理为主要计算耗时 | 确认性完成 | R1 |
| E03 | 稳健 depth sampling 是否优于单点采样？ | 9 个离线序列、489 帧 | point、不同窗口与 `median_7x7` | MPJPE、采样耗时、有效率 | `median_7x7` MPJPE `94.924 +/- 1.080 mm`，优于 point 的 `97.821 +/- 0.922 mm`；采样均值 `0.420 ms`，选为离线默认 | 确认性完成 | R1 |
| E04 | 针对腕部空洞的 `wrist_aware_v5` 能否在真实 ZED live 中提高 coverage 且保持实时？ | 4 组 A/B 对、8×60 s、14,405 帧 | `median_7x7` vs `wrist_aware_v5` | core/wrist measured coverage、输出 FPS、连续性门限 | core coverage `92.31% -> 99.59%`，左腕 `0% -> 98.71%`，输出约 30 FPS；但出现一帧 `51.34 mm` step 超过 50 mm 门限。作为 live 默认，但完整严格合同未全过 | 正式混合/阴性 | R1、D |
| E05 | 骨长约束是否能在不损失 measured coverage 的情况下改善关节与肢体估计？ | held-out 离线集；live pilot 607 帧 | K0 vs K2g guarded 8-bone projection | MPJPE、jitter、limb MAE、bone variability、coverage、额外耗时 | 离线 MPJPE `-0.087 mm`、jitter `-0.220 mm`、limb MAE `-1.843 mm`、bone variability `-1.165 mm`；coverage 100%，均值耗时 `1.092 ms`。live pilot 可运行，但正式 live recorder 并非全部通过 | 确认性完成 | R2 |
| E06 | 测量缺失时，K3/K4 推断能否恢复动态右臂输出？ | 3 次独立 dynamic GT confirmatory，每次 240 帧，共 720 帧 | Raw、K2、K3、K4 | right-arm two-joint coverage、mean/P95 position error、provenance | Raw/K2 coverage `73.75%`；K3/K4 达 100%。K4 mean error `32.573 +/- 2.613 mm`，但 P95 `55.646 +/- 5.992 mm`，存在尾部误差代价；干净输入下 K4 与 K3 基本等价 | 确认性完成 | R2 |
| E07 | K4 对 root-depth 异常和持续缺失的鲁棒边界是什么？ | 3 轮故障注入；多档持续缺失长度 | K3 vs root-guarded K4 | 异常衰减、coverage、最大可桥接时长、恢复帧数、clean non-degradation | +/-150 mm root-depth 异常衰减约 `61.2%–63.0%`；最多桥接 30 帧/0.5 s root-depth 缺失并在下一帧恢复；不能桥接完整 2D shoulder 丢失 | 确认性完成 | R2 |
| E08 | 人体被真实渲染遮挡时，深度采样、K2 与 K3/K4 的表现如何？ | 3 次独立 partial-occlusion formal | Raw、K2、K3、K4 | active coverage、mean/P95 error、恢复速度、11 项门限 | 10/11 门限通过；K3/K4 coverage `0.958 +/- 0.000`，未达到注册的 1.0；mean error `186.83 +/- 6.21 mm`、P95 `349.35 +/- 2.49 mm`。说明能恢复大部分输出，但不能消除 2D 遮挡偏差 | 正式混合/阴性 | R3、D |
| E09 | temporal tracking 能否同时提高动态连续性、静态稳定性和准确度？ | clean dynamic 3 轮；640×360 static 3 轮；fresh 960×600 static 3 轮 | T0/raw vs T2 temporal tracking | P95 step residual、jitter、coverage、lag、mean/P95 error | 动态 step residual 降低 `21.7%–30.7%`；静态 jitter 降低 `56.5%–77.1%`，coverage 1.0；但 fresh 960×600 mean error 增加 `16.1–17.9 mm`、P95 增加 `7.2–13.1 mm`。结论为“更平滑但更偏”，不能宣称全面改善 | 正式混合/阴性 | R2、D |
| E10 | 两个同帧视角可用时，GT-free measured-first best-view 是否能恢复遮挡关节？ | View A/C same-capture，3 次 formal；三轮共 180 个 active frames | 单 View A vs measured-first selector 使用 A/C | coverage、position error、选中视角、5 项冻结门限 | 5/5 门限全过；active coverage `1.000 +/- 0.000`，error `19.45 +/- 0.64 mm`，三轮 `180/180` active frames 全部选择 View C | 正式通过 | R3 |
| E11 | best-view 选择器能否脱离 GT 部署，并满足实时性？ | 等价性 720 帧/1,440 关节；双 worker 3 轮；conditional streaming 3 个独立 Isaac PID | formal selector vs GT-free deployment；always-on dual vs conditional View C | identity/coordinate equivalence、selector latency、paired FPS、coverage、active error、consumer FPS | GT-free 等价性 7/7 全过，identity mismatch 0、最大坐标差 `4.44e-16 m`；always-on 双 worker仅 `17.08 +/- 0.39 FPS`，失败；conditional repeatability 全部门限通过，coverage 1.0、mean/P95 `20.22 +/- 0.48 / 25.87 +/- 0.56 mm`、consumer `39.98 +/- 1.04 FPS` | 正式通过（等价性与条件工程）；正式阴性（always-on） | R3、D |
| E12 | 自适应相机目标函数 `J` 能否预测真正最低误差视角？ | 17-view bank，3 次 formal capture/evaluation | `J` 排名 vs oracle error ranking | usability、Spearman rho、oracle regret、计算开销 | 可用性门限通过，但 accuracy ranking 失败：rho `-0.3905 +/- 0.0195`，regret `12.31 +/- 0.35 mm`，开销 `19.06 ms` 超门限。`J` 只能作可用性筛选器，不能作准确度排序器 | 正式混合/阴性 | R3、D |
| E13 | 多视角加权能否提高 limb-length identification，并保持各肢段稳定性？ | fresh formal 3 轮；CTS5 follow-up 5 轮 | 优化共享凸权重 vs 等权 vs 固定 `p010` | PBMAE、各肢段 MAD、R1–R5 | formal 优化 PBMAE `7.307 +/- 0.363 mm`，CTS5 `4.571 +/- 0.058 mm`，均明显优于等权与 `p010`；但两组均因至少一轮 segment MAD 非劣门限 R5 失败，不能宣称完整合同通过 | 正式混合/阴性 | R3、D |
| E14 | 原始真实 SVO 中 baseline、depth sampler 与 K4 的行为如何？ | 单被试、单次 2,498 帧 SVO2；冻结 S1 相位标注 | D0 `median_7x7`、D1 `v5`、D3 `v5+K4` | detection、measured coverage、phase continuity、K4 inferred provenance | D0/D1/D3 core coverage 分别 `0.9922/0.9273/0.9429`；v5 在共同子集上与 D0 近似等价，主要起 reliability filter 作用；K4 将右腕 coverage `0.5927 -> 0.7959`。无人体 GT，只能作真实案例与机制分析 | 探索性 | R4、D |
| E15 | 三轮标准化真实采集能否支持同被试重复性、真实 K2g/K4 与 sampler 恢复结论？ | run02–04，3 次独立 recorder restart，共 12,648 帧 | D0 vs D1；预定 D2/K2g、D3/K4 | profile eligibility、coverage、wrist recovery、重复性 | artifact/verifier 全过，但 profile 仅 2/8 bones 达标，因此 D2/D3 按规则被阻断；D1 core coverage 比 D0 低 `5.18 +/- 0.31 pp`，六个 wrist recovery 门限全失败。支持真实重复性阴性结论，不支持真实准确度结论 | 正式混合/阴性；D2/D3 被阻断 | R4、D |
| E16 | ZED BODY_38 是否比 baseline 更准确或更稳定？ | S3 exact-frame agreement；Track A 3 次 formal live same-frame comparison | BlazePose baseline A1/A2_v5 vs BODY_38 A3 | availability、agreement、GT position error、跨轮方向一致性 | BODY_38 不是 GT。Track A 三轮 validity 8/8；R1、R3 通过，但 R2 原始优劣方向 `[A3, A2_v5, A3]` 不一致；A3 跨轮 SD/mean `0.42`，A1 约 `0.02`。不能宣布统一赢家，只能报告 BODY_38 在本协议下跨轮不可复现 | 正式混合/阴性 | R4、D |
| E17 | 实验室 OBJ 是否可注册到真实 SVO，并用于人体-环境 clearance？ | 12 个拟合帧 + 24 个不相交 held-out 帧 | local point cloud-to-mesh registration | held-out median/P95 depth residual、100 mm 内比例、通过帧率 | held-out 4/5 门限失败：median `102.26 mm`、P95 `1570.87 mm`、100 mm 内比例 `0.4938`、`0/24` 帧通过。OBJ 仅作可视化，禁止输出 clearance | 正式阴性；后续终点被阻断 | R4、D |
| E18 | FS-CTS5 能否在八个映射主要肢段和四种场景下稳定优于对照？ | 20/20 bundles，24,000 camera frames；152,320 行正式比较分母 | M3 vs M2 及冻结对照 | G1–G6，尤其 clean right-thigh P95 非劣 | G1/G2/G4/G5/G6 通过；G3 失败：M3 `26.509 mm`、M2 `23.746 mm`、margin `2.084 mm`，超界 `0.680 mm`。完整合同阴性，不得只报告通过门 | 正式混合/阴性 | R3、D |

## 3. 支持性、工程性与排除项矩阵

| ID | 项目 | 作用 | 最终状态 | 使用位置 |
|---|---|---|---|---|
| S01 | Isaac 场景、男女角色行走/挥手、ZED Body Tracking 检出 | 建立可控仿真与演示环境 | 工程验证完成；动作真实性不是主要科研终点 | M、A、演示视频 |
| S02 | Ground Truth exporter 与 15-joint mapping | 生成同步 RGB、metric depth、相机参数和 USD skeleton GT；统一 Isaac–ZED–BlazePose 定义 | 工程与方法基础完成；13 joints 用于核心 full-body 评价 | M |
| S03 | partial occlusion v1 | 首次尝试渲染遮挡 | 遮挡物挂在相机子节点下而未渲染；废弃且不进入任何统计 | A，仅说明排除原因 |
| S04 | 遮挡误差分解与 2D stabilization | 解释 E08 失败机制 | 约 `98.93%` degradation 来自 2D drift；时序 2D 稳定未解决系统偏差 | R3、D |
| S05 | View B pilot 与早期 View A/C selector | 搜索有效互补视角并冻结 measured-first 方法 | View B 因自遮挡淘汰；K4-only selector 阴性；用于方法演化，不进入最终效果统计 | A、D |
| S06 | common-frame plumbing 与 same-capture validity | 验证 timestamp、坐标变换和同帧配对链路 | 工程 gate 通过；不能单独作为同步融合性能证据 | M、A |
| S07 | conditional streaming pilot v1 | 发现激活瞬间两帧 RGB-D pre-roll 不足 | 12/12 基础门限通过，但 onset 有 K4 fallback、最大 wrist error `237.72 mm`；由 v2 四帧 pre-roll 修复 | A、D |
| S08 | conditional repeatability v1 | 首次三轮重复执行 | rep_01 在 capture 前退出，无有效帧；产物保留且不计重复 | A |
| S09 | multiview limb-length E0 | 开发共享凸权重方法并选择 fresh confirmatory 方案 | retrospective development，明确 excluded；只支持方法开发 | M、A |
| S10 | 多人物 frontend pilot | 验证多人体候选、ID 与日志前端可运行 | 工程验证；核心实验仍是单人，不能宣称多人算法已验证 | A、Future Work |
| S11 | S4 mapping bridge recorder | 未来同 session 采集 SVO、WORLD pose 和 spatial map | 工具与 24 个 fake-SDK 测试完成，但未做真实硬件采集、protocol 未冻结 | A、Future Work |
| S12 | FS-CTS5 cross-character canonical same-input replay | 保存第一条 replay execution 的工程终态 | worker 11 promotion 后停留 partial；closeout `FAIL_sealed_replay_preserved_no_rerun`，exact QC 与 selector 均未执行 | A、D |
| S13 | FS-CTS5 cross-character prospective recovery `rpr_v1` | 验证 fresh replay、same-input exact QC 与冻结 v8 selector | 24/24 workers、12/12 exact QC、832,896 bits 通过；selector 唯一调用在 6,126,120 pair 中 0 个合格，`FAIL_v8_selector_no_authorized_bank`。下游 confirmatory/overlay/G2/XCAL/formal 均未授权 | R3、D、A |
| S14 | FS-CTS5 successor-v3 transition probe | 验证不同候选布局能否通过 frozen selector | canonical raw 8/8 完成但 replay 因 `MAX_PATH` 在推理前工程失败；独立 short-path recovery 16/16 workers、8/8 exact QC 通过，但 selector 在 2,704,156 subsets 中 0 个合格。该路线科学终止，formal `0/36` | D、A |
| S15 | FS-CTS5 successor-v4 `v4p1r6` | 建立新的 12-session cross-character raw cohort | 早期失败分支均封存；`v4p1r6` raw capture 12/12 与全量验证通过，postcapture 工具定向测试 11/11 通过。2026-08-28 已冻结 replay-only authority，但尚无 worker 输出或 replay closeout；selector/calibration/confirmatory/formal 均无科学结果 | A、Future Work |

## 4. 论文结果章节建议

| 结果章节 | 必须包含的实验 | 核心图表 | 本章主结论 |
|---|---|---|---|
| R1 Baseline and component selection | E01–E04 | 距离边界图；Lite/Full trade-off；depth sampler coverage/latency | 建立 BlazePose + ZED baseline，并说明离线与 live 采用不同配置的依据 |
| R2 Kinematics and temporal estimation | E05–E09 | K0/K2g 指标表；K3/K4 coverage-error 图；temporal bias-variance 图 | 运动学约束与短时推断有用，但 completion 和平滑均存在清晰失效边界 |
| R3 Occlusion, multi-view and camera adaptation | E08、E10–E13、E18 | 遮挡分解图；best-view coverage/error；FPS 分解；viewpoint ranking；limb PBMAE；FS-CTS5 gate 分账 | 互补视角和条件激活有效；简单相机目标函数、肢段稳定性与 FS-CTS5 完整合同未完全成立 |
| R4 Real-world evaluation | E14–E17 | phase timeline；三轮 coverage；Track A 跨轮方向；OBJ held-out residual | 真实数据验证了系统可运行与若干失败模式，但不提供无 GT 的绝对准确度证明 |

## 5. 最终主张边界

论文可以支持的主张：

- 已建立可运行的 BlazePose + ZED RGB-D 3D joint 与 limb-length baseline，并以 Isaac 同步 GT 定量验证；
- K2g 可在保持 measured coverage 的条件下小幅改善骨长一致性；
- K4 可处理有限时长的 depth 缺失和异常，但不能恢复完整 2D landmark 丢失；
- measured-first 多视角 best-view 在同帧仿真遮挡条件下显著恢复 coverage，GT-free selector 与 formal selector 等价；
- conditional View C 能在当前仿真流式配置下同时保持高 coverage 与约 40 FPS consumer throughput；
- temporal tracking 存在明确 bias–variance trade-off；
- 多视角加权 limb-length estimation 明显降低平均误差，但 segment-level 稳定性非劣合同未完全通过；
- 三轮真实采集和 BODY_38 对比揭示了跨配置迁移、profile eligibility 和跨轮可复现性问题。

论文不能支持的主张：

- 不能把 BODY_38 当作 GT，也不能宣称它或 BlazePose 在所有场景统一更准确；
- 不能把 K3/K4 inferred joints 计入 measured coverage；
- 不能把单人仿真结果泛化为多人 HRC 已验证；
- 不能把 conditional View C 的双虚拟流结果写成真实双 ZED 硬件验证；
- 不能使用失败注册的 OBJ 输出 clearance、安全距离或碰撞风险数值；
- 不能把 temporal tracking 写成同时提高稳定性和准确度；
- 不能把 adaptive objective `J` 写成已验证的准确度最优视角目标函数；
- 不能因 PBMAE 改善而隐去 limb-length R5 稳定性门限失败；
- 不能把真实 SVO 的 coverage/agreement 指标解释为绝对 3D accuracy。
- 不能把 `rpr_v1` selector 阴性写成 36-bundle 方法学阴性；不得手工挑选 near-miss、二次调用 selector 或执行未授权下游。

## 6. 证据溯源索引

下表保留本地结果文档的文件名作为溯源标识，不承诺这些详细文件已公开。
公开数值入口是[结果总览](results_overview.md)、[当前进度](project_progress_cn.md)
和[FS-CTS5 汇总](../results/fs_cts5_20_bundle_summary.json)。原始冻结记录仍在本地保留。

| 实验 ID | 本地主要结果文档标识（非下载链接） |
|---|---|
| E01 | `first_static_experiment_results.md` |
| E02 | `performance_and_distance_experiment_summary.md`、`blazepose_model_complexity_experiment_results.md`、`pipeline_performance_breakdown_results.md` |
| E03–E04 | `depth_sampling_ablation_results.md`、`depth_sampling_live_ab_protocol.md` |
| E05–E07 | `kinematic_constraints_final_summary.md`、`dynamic_k4_gt_confirmatory_v1_results.md`、`k4_root_fault_injection_v1_results.md`、`k4_sustained_occlusion_sensitivity_v1_results.md` |
| E08、S03–S04 | `dynamic_partial_occlusion_formal_v1_results.md`、`occlusion_2d_decomposition_v1_results.md`、`occlusion_2d_stabilisation_v1_results.md` |
| E09 | `clean_temporal_tracking_v1_results.md`、`clean_static_temporal_followup_v1_results.md`、`clean_static_temporal_confirmatory_v1_results.md` |
| E10–E11、S05–S08 | `view_ac_measured_first_formal_v1_results.md`、`multiview_best_view_deployment_equivalence_v1_results.md`、`view_ac_performance_decomposition_v1_results.md`、`view_ac_conditional_stream_repeatability_v2_recovery.md` |
| E12 | `adaptive_camera_objective_a2_a4_formal_results.md`、`adaptive_camera_positioning_summary_bilingual.md` |
| E13、S09 | `multiview_weighted_limb_length_formal_results_cn.md`、`multiview_limb_length_cts5_formal_results_cn.md`、`multiview_weighted_limb_length_e0_development_results.md` |
| E14、E17 | `real_svo_case_study_v1_results.md`、`real_svo_s0_validity_and_camera_motion_v1_results.md` |
| E15 | `real_zed_same_subject_repeatability_v1_results.md` |
| E16 | `body38_track_a_consolidated_results_v1.md`、`body38_track_b_availability_continuity_v1_results.md` |
| E18 | `formal_results_report_v1.json`；公开副本是 [aggregate summary](../results/fs_cts5_20_bundle_summary.json)，不是原始数据 |
| S10 | `multiperson_frontend_bridge_v1_results.md` |
| S11 | 本地项目记忆的 S4 bridge 状态；公开边界见[当前进度](project_progress_cn.md) |
| S12–S13 | [公开终态结果](fs_cts5_common_bank_randomized_block_science_recovery_v3_terminal_results_cn.md) |
| S14 | 本地 canonical 与独立 recovery closeout；公开摘要见[结果总览](results_overview.md) |
| S15 | `fs_cts5_common_bank_successor_v4_rpr_v6_postcapture_replay.md`、`postcapture_replay_lock.json`；公开摘要见[结果总览](results_overview.md) |

## 7. 数据入口

- 结果总览：[`results_overview.md`](results_overview.md)
- 机器可筛选矩阵：[`final_experiment_matrix.csv`](final_experiment_matrix.csv)
- Ground Truth exporter：本地 `tools/isaac_export_ground_truth.py`，未随本次公开；合同见[系统架构](architecture.md)
- 关节映射：[`../configs/joint_mapping.csv`](../configs/joint_mapping.csv)
- 主要实验产物：本地 `output/experiments/`，不在公开仓库中；参见[数据可用性](../DATA_AVAILABILITY.md)
