# FQL Succession — P2 之前的施工记录（归档）

> 📦 **归档说明（2026-08-17）** — 本目录只表示**位置与角色**，**不含任何有效性判断**。
> 这里的文件结论没有被撤销，只是不再作为论文引用入口。

## 为什么分出这一层

FQL succession 线 **2026-05-23 NEGATIVE 闭环**——这是论文第 5 章 §5.9 的主体内容之一，
不是弃案。**负面结论照样是论文结论**。

但线内文档承担的角色差别很大。2026-08-16 的入链审计里，`docs/` 顶层 16 份 `fql_*` 分成两段：

- **P2 收口产物** —— 论文直接引用（`paper/` 侧共 27 处引用，分布在 6 份文档上）。留在 `docs/` 顶层。
- **P2 之前的施工记录** —— 论文一次未引，任何入口也够不到（`thes=0` 且 `entr=0`），
  只被同线文档与 notebook 互引。就是本目录这 7 份。

分界线**不是「FQL vs 其他」，是「收口产物 vs 通往收口的过程」**。这两段的文件名只差一个字符
（`p0p1_spec` / `p2_main_spec`），光看目录列表分不出来——这正是要分层的原因。

## 本目录的 7 份

| 文件 | 角色 |
|---|---|
| [`fql_succession_plan_v0.md`](fql_succession_plan_v0.md) | 线的最初 plan（P0–P4 phase 设计、gate 条件）。SUPERSEDED 2026-06-02 |
| [`fql_succession_p0p1_spec.md`](fql_succession_p0p1_spec.md) | P0+P1 可执行 spec（Task A–D）。CLOSED |
| [`fql_succession_gate_b_report.md`](fql_succession_gate_b_report.md) | Gate B 终报（Option B 2-seed）——**冻结超参的来源** |
| [`fql_succession_gate_b_interim_report.md`](fql_succession_gate_b_interim_report.md) | Gate B 单种子中间报告。SUPERSEDED，留作可追溯性 |
| [`fql_succession_bug2_fix_decision.md`](fql_succession_bug2_fix_decision.md) | P2 前置决策 1/2：`--episodes` vs manifest size |
| [`fql_succession_c4_threshold_revision.md`](fql_succession_c4_threshold_revision.md) | P2 前置决策 2/2：c4 阈值 slope ≥ 0 → no-major-collapse |
| [`fql_audit_dryrun_report.md`](fql_audit_dryrun_report.md) | `scripts/audit_multimodality.py` 的 dry-run 验收报告（4/4 PASS） |

## 没有跟着搬的三份，以及为什么

`thes=0 / entr=0` 是入链计数，它**分不出「历史记录」和「只有代码引用的活契约」**。这三份按引用数看
像孤儿，实际都指着仓库里现存的资产：

| 留在 `docs/` 顶层的 | 它记录的现存资产 |
|---|---|
| [`fql_pytorch_port_design.md`](../../fql_pytorch_port_design.md) | `auv_nav/fql.py` 的 design contract；模块头注与 `tests/test_fql.py` §13（12 个测试）都指向它。论文 §5.4 核对 loss 口径要读这个模块 |
| [`fql_audit_multimodality_design.md`](../../fql_audit_multimodality_design.md) | `scripts/audit_multimodality.py` 同上；`tests/test_audit_multimodality.py` §9 是它的测试计划 |
| [`fql_e_uni_anchor_dataset_card.md`](../../fql_e_uni_anchor_dataset_card.md) | `offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/` 的数据卡；论文引用的 [`fql_succession_p2_main_spec.md`](../../fql_succession_p2_main_spec.md) §Status 指着它作 Task D 闭环凭证 |

前两份的头注仍写着「**Active design**，待 executor 按此 doc 实现」——代码早已落地，
这行是陈旧的，已在 2026-08-17 改注。

## 论文要用的数字在哪

不在本目录。去 [`fql_succession_p2_results.md`](../../fql_succession_p2_results.md)（主报告，
NEGATIVE 闭环）与 [`fql_succession_p2_mechanism_diagnostic.md`](../../fql_succession_p2_mechanism_diagnostic.md)；
写作路由见 [`fql_succession_paper_writing_index.md`](../../fql_succession_paper_writing_index.md)。
线的时间轴见 [`offline_rl_line_summary.md`](../../offline_rl_line_summary.md)。
