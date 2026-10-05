# FS-CTS5 cross-character `rpr_v1` 终态结果

日期：2026-08-24  
分类：**有效的注册 selector 阴性；不是 replay 工程失败，也不是 36-bundle 方法学结果**

## 1. 结论

`rpr_v1` 在与旧 canonical replay 完全不同的 write-once 目的地完成了 24/24 fresh ordinary-Python
workers。12 个 role-session 的 A/B same-input binary exact QC 全部通过，随后冻结的 v8 selector 被唯一调用
一次。selector 在 6,126,120 个候选 pair 中没有找到任何同时跨三轮通过全部硬门的 pair，因此终态为：

```text
FAIL_v8_selector_no_authorized_bank
```

这是一项有效的注册阴性结果。它不授权第二次 selector、不允许放宽门限或手工挑选 near-miss，也不产生
B9/F5/M0。confirmatory、overlay、G2、XCAL 与 36-bundle formal 全部按冻结依赖链停止。

## 2. 与旧 canonical FAIL 的关系

旧 canonical replay 仍是独立的工程 FAIL：

- status：`FAIL_sealed_replay_preserved_no_rerun`；
- closeout SHA256：`cf35402319229e00c0e3000ded1bf4be9d3e4394455a385cc2fb473a38926a3f`；
- raw cohort seal SHA256：`59d4a058327f007a76c3e5c17d79812736dbed3499789488628a38ab14da4c2c`；
- worker 11 partial 原样保留；旧 selector 调用为 0。

`rpr_v1` 不覆盖、续跑、提升或复用旧目录。新 selector 阴性不能反向把旧工程 FAIL 改写成科学结果；旧 FAIL
也不能用来否定新 replay 的有效性。

## 3. 冻结合同

| 合同 | SHA256 |
|---|---|
| prospective replay recovery v1 | `dc1bc68f9cc8bd5b37755e4d1bd06d7c4c6aa01ab60fbb799e2e710f4dbf6205` |
| transport audit v2 | `7eb7d268ef23c755c1939e6da1c0ae0eb7676f2fe69e4e3d32dec698fbfa8163` |
| execution audit v3 | `52be6db014ae8122a2e27f6548a556e39803cd4a48c5e34947d0e3529ea9971e` |

## 4. Replay 与 transport 终态

- 计划/实际 worker invocation：`24/24`；
- startup receipt：`24/24`；terminal receipt：`24/24`；
- 全局唯一 kernel process identity：`24`；
- return code 0：`24/24`；
- terminal state `success_promoted_final`：`24/24`；
- transport promotion：24 worker + 1 replay root，`25/25 PASS_promoted_write_once`；
- promotion attempt：每项一次；error receipt：0；overwrite attempt：0；
- replay sibling `.partial`：不存在。

coordinator 终态：

```text
PASS_24_independent_workers_sealed_pending_exact_QC
```

`rpr_v1/replay/replay_closeout.json` SHA256：
`6d0ef13224c1b083d86e089004c86f08215f543301e0ba34d196cd5fdb9b608c`。

## 5. Exact QC 与 selector 终态

| 项目 | 结果 |
|---|---:|
| same-input sessions binary exact | `12/12` |
| globally unique worker identities | `24` |
| selection matrices | `3` |
| registered bits | `832,896` |
| third replay authorized | `false` |
| selector calls | `1` |
| candidate pairs | `6,126,120` |
| qualifying pairs | `0` |

失败原因为：

```text
no_pair_passed_all_hard_gates_in_all_three_repeats
```

硬门分账：

| hard gate | 淘汰 pair 数 |
|---|---:|
| `f_coverage_not_all_repeats` | 4,934,215 |
| `f_recoverable_events_not_all_repeats` | 1,191,905 |
| `f_distinct_event_views_not_all_repeats` | 0 |
| `f_not_all_repeats` | 0 |
| `r_gate_vacuous_not_all_repeats` | 0 |
| `r_member_without_swap_not_all_repeats` | 0 |
| `r_swap_coverage_not_all_repeats` | 0 |

结果身份：

- `selected_view_ids=null`；
- `bank_view_ids=null`；
- `m0_view_id=null`；
- `confirmatory_capture_authorized=false`；
- `calibration_capture_authorized=false`；
- `formal_capture_authorized=false`。

## 6. 终态文件

| 文件 | SHA256 |
|---|---|
| `same_input_qc.json` | `9a0f2fcd0d42e8b6a18e61530b66f935920a98e0d47f4dfb3ab344d59ebd53c9` |
| `selection_matrix_rep_01.json` | `60ad26267d6e6689b759b4f819c9ac1cc504e1049e8ecc6de1e9f4f326d078e0` |
| `selection_matrix_rep_02.json` | `290800a08bb05820e26dc2ac1c0d3015ffc3793530ad966c0d2771c798e72c2b` |
| `selection_matrix_rep_03.json` | `bdd788d418a86362fb3be17142567883c92a5cecd4e3b1c3bc0c153535f49cb2` |
| `cross_renderer_binary_report.json` | `966b35449a67fea111521e9d170bf88cf72af326d604ddef9b26ab98207e4d5c` |
| `selection_result.json` | `b072530f18f08080553302892eecdd7f525fd4b6a4394b2766b7f1e0ea61eb6f` |
| `selection_closeout.json` | `8eba24907f3fca161f92650ca389d7a283c2c2b43ef4012acc98ba8df11bfa47` |

失败分支按协议抑制 chosen、runner-up、leaderboard 与 provisional IDs，不能从内部矩阵自行导出替代 bank。

## 7. 独立终态验证

phase-aware v3 validator 在 replay final 后连续两遍以 exit 0 报告
`prospective_replay_pass_preselection`，并且只在该状态开放 `finalize_selection_authorized=true`。selection
完成后再次以 exit 0 报告：

```text
phase = prospective_selection_fail_preserved
selector_calls = 1
finalize_selection_authorized = false
```

postselection handoff 的只读 inspect 随后按设计拒绝，错误为 `Selection is not terminal PASS.`。

冻结前 v3 专项 suite 为 18/18 通过；真实 output 存在后原样重跑得到 15 pass、2 error、1 fail。三项均来自
pre-execution 测试对“recovery root 必须 pristine/不存在”的硬断言，不能通过删除真实 output 或修改冻结测试
来制造全绿。终态有效性由 phase-aware validator、receipt 与 closeout 决定。

## 8. 后果与表述边界

- confirmatory：`0/12`；
- 四角色 overlay：未 author/fresh-load；
- G2 excluded pilot：未执行；
- XCAL excluded calibration：`0/12`；
- formal：`0/36`；formal GT 未打开。

不得陈述：

- “36-bundle formal 已完成/失败”；
- “某个近似通过的 five-view bank 可替代 selector 输出”；
- “旧 canonical replay 与 `rpr_v1` 可以合并为重复实验”；
- “selector 阴性证明 FS-CTS5 方法本身准确或不准确”；
- “source-only 下游工具链等于已完成 capture”。

可以陈述：在冻结 raw science、fresh replay、same-input exact QC 与唯一 selector 调用下，本候选库没有任何
跨三轮同时满足注册 coverage 与 recoverable-event 硬门的 bank/reserve pair；因此预注册依赖链在 selector
处以有效阴性终止。
