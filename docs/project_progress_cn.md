# 项目当前进度

> GitHub 同步日期：2026-10-05。此页整理本地截至 2026-08-31 的现有实验和写作产物，
> 不代表在同步当天新增采集或重新运行实验。
>
> 正式标题：**Parameter Identification and Human Body Characterisation for Human-Robot Collaboration**

## 1. 当前定位与整体状态

项目已经完成从仿真动画、同步 Ground Truth（GT，真实值）导出、BlazePose + ZED RGB-D
baseline，到深度采样、运动学约束、时序处理、遮挡、多视角和真实重复采集的主要实验链。
现有正面、混合和阴性结果已足以构成一份有明确边界的 MSc 报告。

baseline 始终是 **BlazePose + ZED RGB-D**；ZED SDK `BODY_38` 只是比较方法。
仿真中的 GT 是同步 Isaac USD Skeleton；真实 SVO 没有独立人体关节 GT。
核心实验主要是单人场景，不能据此宣称已完成多人协作或安全认证。

完整比较方法、独立样本单位、终点和报告章节见[最终实验矩阵](final_experiment_matrix.md)，
主要数值和表述边界见[英文结果总览](results_overview.md)。

## 2. 工作包状态

| 工作包 | 当前状态 | 可以支持的结论及限制 |
|---|---|---|
| 仿真与动作 | 已完成 | 男女角色均可行走、挥手，ZED Body Tracking 可识别；动作演示不是准确度验证 |
| GT 与关节映射 | 已完成 | 同步 RGB、metric depth、相机参数、骨架 GT；15 个映射关节、13 个核心评价关节 |
| 静态与距离基线 | 已完成 | 女性静态 10 s、59 帧；检测和核心有效率均为 100%；距离边界仅适用于已测配置 |
| Lite/Full 与性能拆分 | 已完成 | Full 用于干净离线静态准确度；Lite 用于实时及冻结动态 replay；推理是主要计算开销 |
| 深度采样 | 已完成 | 离线选 `median_7x7`，特定 live 配置选 `wrist_aware_v5`；跨配置收益不保证 |
| K2g 与 K4 | 已完成 | K2g 约束有效测量；K4 短时补全明确标为 inferred，不覆盖 measured 值 |
| 动态同帧 GT | 已完成 | 三次独立 confirmatory、720 帧；右肘和右腕误差不是 full-body MPJPE |
| Clean temporal tracking | 已完成，混合/阴性 | 动态更连续、静态更平滑，但动态完整合同和 fresh 960×600 静态完整合同均未通过 |
| 640×360 静态 temporal follow-up | 已完成，6/6 通过 | jitter 降低 56.5%–68.0%；不能反向推翻其他分辨率的阴性合同 |
| K4 故障与持续缺失 | 已完成 | 在测试配置下可桥接最多 0.5 s root-depth 缺失；不能桥接完整 2D shoulder 丢失 |
| 三轮渲染遮挡 | 已完成，10/11 通过，总合同未通过 | K3/K4 inferred coverage 为 0.958，未达到注册的 1.000；恢复输出不等于消除位置误差 |
| A/C 同帧 measured-first best-view | 已完成，5/5 通过 | active measured coverage 1.000，右臂两关节 error 19.45 ± 0.64 mm |
| GT-free selector 等价性 | 已完成，7/7 通过 | 720 帧、1,440 关节无 identity mismatch；这是部署等价性，不是真实世界准确度证明 |
| 双路 always-on 性能 | 已完成，实时门限失败 | 17.08 ± 0.39 paired FPS，低于 27 FPS 注册门限 |
| 条件 View C streaming | 已完成，逐轮和跨轮门限均通过 | 四帧 pre-roll；三轮 consumer 39.98 ± 1.04 FPS，mean error 20.22 ± 0.48 mm；不是 camera-to-output FPS |
| 真实 SVO 资产与 S1/S2 | 已完成 | 原始片段有效分母 2,498 帧；相位标注已冻结；覆盖率、连续性和机制分析不等于绝对准确度 |
| 三轮标准化真实重复采集 | 已采集并正式分析 | 12,648 帧；profile 仅 2/8 骨达标，D2/D3 按规则未运行；D0→D1 为注册阴性 |
| BODY_38 比较 | 已完成，方向重复门限失败 | 仿真每轮 validity 8/8，但跨轮优劣方向不一致；BODY_38 不是 GT，也不能宣布统一赢家 |
| OBJ 配准 | 已完成，held-out 4/5 失败 | OBJ 只能作环境可视化，禁止输出 clearance 或安全距离 |
| 同 session mapping recorder | 工具已实现，尚无真实硬件验证 | fake-SDK 测试不等于真实 SVO、WORLD pose 与 mesh 配准通过 |
| Adaptive camera objective | 已收束，注册阴性 | 可作为可用性筛选器，不能准确排序机位；rank correlation -0.39、regret 12.31 mm |
| 加权多视角骨长与 CTS5 | 已完成，混合/阴性 | 平均骨长误差下降，但两条正式线均未通过 R5 肢段稳定性非劣门限 |
| FS-CTS5 20-bundle | 已完成，5/6 门族通过，总合同阴性 | 24,000 camera frames、152,320 比较行；G3 clean right-thigh P95 超界 0.680 mm |
| Cross-character canonical replay | 已封存工程失败 | worker 11 promotion 失败；没有 selector 结果，不解释为方法学阴性 |
| Cross-character `rpr_v1` | replay 及 exact QC 通过；selector 注册阴性 | 24/24 workers、12/12 exact QC；6,126,120 pairs 中 0 个合格；formal 0/36 |
| Successor-v2 | 渲染前工程失败，已封存 | 无科学数据，不计为科学阴性 |
| Successor-v3 | 独立 recovery 后 selector 注册阴性 | 16/16 workers、8/8 exact QC；2,704,156 subsets 中 0 个合格；formal 0/36 |
| Successor-v4 `v4p1r6` | raw 12/12 完成；replay-only lock 已冻结，尚未执行 replay | 已核对 2026-08-28 lock；无 worker 输出和 replay closeout；selector、calibration、confirmatory、formal 均无结果 |
| 深度存储审计 | 已完成，工程 PASS | 41,280 路径内容保持不变；同证据类别内硬链接去重释放约 79.07 GiB，不增加实验样本 |
| 报告及文献检查 | 已形成报告草稿和审计产物 | 本地已有 LaTeX、PDF、Word、引用及缩写检查和中文译稿；不代表已提交或通过学校评审 |

### FS-CTS5 20-bundle 的完整判定

四种仿真场景分别是 S0 干净运动、S1 可恢复右臂遮挡、S2 可恢复桌面腿部遮挡、
S3 严重桌面腿部遮挡。每种场景五次独立采集，每次 240 render steps、五路相机。
所有 prediction 在首次读取正式 GT 前完成冻结和验证。

G1 accuracy、G2 coverage、G4 accurate-output rate、G5 abstention、G6 recovery 通过；
G3 肢段 P95 非劣门失败。clean right-thigh 的 M3 P95 为 26.509 mm，M2 为 23.746 mm，
注册允许差为 2.084 mm，超界 0.680 mm。不得调宽门限或只汇报通过项。

参见[机器可读摘要](../results/fs_cts5_20_bundle_summary.json)。

## 3. 方法与指标说明

| 简写 | 解释与使用边界 |
|---|---|
| HRC | Human-Robot Collaboration，人机协作；不是安全认证标签 |
| RGB-D | 颜色图像与深度图像；ZED 估计深度不是独立人体 GT |
| GT | Ground Truth；本项目准确度主线采用同步 Isaac 骨架 GT |
| K2g | 带 profile guard 的 measured-joint 骨长约束；不补成缺失测量 |
| K3/K4 | 缺测时的推断补全；K4 增加 root-depth 保护，输出保留 inferred provenance |
| T0/T1/T2 | 时序实验的帧独立、tracker-only、tracker-plus-smoothing 配置；T2 更平滑不代表更准确 |
| CTS5 | 项目方法标签：逐视角、逐骨段偏置校正后融合五路标量骨长；四或五路时去两端再平均，三路取中位数，少于三路弃权。不凭字母虚构标准全称 |
| FS-CTS5 | CTS5 的八个映射主要肢段扩展：双侧上臂、前臂、大腿、小腿；不包含躯干、手、脚等完整骨架验证 |
| M0/M1/M2/M3 | 20-bundle 中分别是 raw 主视角、raw 多视角等权、偏置校正等权、FS-CTS5 稳健融合；标签不能无说明跨实验复用 |
| D0/D1/D2/D3 | 真实 repeatability 中分别为固定 depth sampler、wrist-aware sampler、其后 K2g、其后 K4；profile 不合格时 D2/D3 不运行 |
| MPJPE | Mean Per Joint Position Error，按声明关节集合计算的位置误差；动态右臂两点误差必须另用明确名称 |
| PBMAE | Pooled Bone Mean Absolute Error，骨段平均绝对误差汇总；与 mean joint position error 不同 |
| MAE/MAPE | Mean Absolute Error / Mean Absolute Percentage Error，平均绝对误差及平均绝对百分比误差 |
| MAD/P95/SD | Median Absolute Deviation / 95th Percentile / Standard Deviation，中位绝对偏差、95 分位数、标准差 |
| ROI | Region of Interest，感兴趣区域 |
| QC | Quality Control，质量检查；byte-exact QC 验证身份而非科学准确度 |
| FPS | Frames Per Second，帧率；必须声明 source、paired、consumer 或实际输出边界 |
| SVO/SVO2、OBJ、USD | ZED 录制容器、环境网格格式、仿真场景格式；它们本身不能替代人体 GT |
| BODY_38 | ZED SDK 的 38-keypoint 人体估计输出，仅作 reference/comparison |

## 4. 尚需完成的部分

1. **报告收尾。** 将现有表格、图、场景示例与最终实验矩阵逐一对应；清除待插图标记，
   检查引用与缩写、评价分母、单位和失败门限，再核对学校格式要求。
2. **实时性能表述。** 保留 consumer、source acquisition、paired FPS 与延迟的区别，
   不把组件速度写成完整传感器到输出的性能。
3. **真实部署边界。** 真实数据可以支持 coverage、continuity、agreement、repeatability，
   绝对 3D 关节准确度和人体尺寸准确度仍需要独立测量或参考系统。
4. **后续研究，不是当前报告阻塞项。** 新硬件双视角验证、合格的真实个体 profile、
   同 session mapping 的 held-out 验证，以及 successor-v4 的授权 replay 仍待执行。
   这些工作须遵守现有协议及状态门，不能续跑已终止的失败路线或手工选择 near-miss bank。

学校提交、导师反馈和未来实验不由当前 GitHub 同步自动完成。

## 5. 本次公开范围

本仓库公开可发布的原始软件、依赖闭合的分析组件、硬件无关测试、实验矩阵和汇总结果。
真实 SVO/RGB-D、个体骨长 profile、环境 mesh、第三方资产、完整运行 output、
工作站路径和私有项目记忆仍保留本地。报告二进制及写作临时目录不批量公开。

冻结研究文件不为发布而改写；公开代码的换行规则与历史原始字节 hash 也不能混为一谈。
公开单元测试通过不是对所有私有协议、硬件采集或科学合同的全量验证。
