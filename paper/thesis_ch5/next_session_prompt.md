# 第 5 章 — 轮次入口（**当前轮：等 Colab 探针结果 → 数据完整性处置裁决**）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与相关约束。章状态账本 = [`status.md`](status.md)（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = [`prompt_playbook.md`](prompt_playbook.md)。
> 上一轮（2026-08-16 独立复核轮）已闭环，产物见下。

---

## 走到哪儿了

第 5 章十节成文、五批整改闭环，2026-07-28 判定送审就绪。章外两条数据完整性问题（2026-08-09 核实成立）：

- **①** `crosscomp-2000` 的训练 episode 与终检评估是**同一批任务实例**（训练种子 0..1999 ⊇ 评估 1250..1349，reset RNG 重放 100/100 逐条相同）。
- **③** 选检查点的验证集（40 回合）是终报测试集（100 回合）的**前缀子集**；40+40 的单元两份 manifest 完全相同。

事实账本：[`../../docs/data_integrity_open_items.md`](../../docs/data_integrity_open_items.md)（含 2026-08-16 的三处订正与新增第 ⑤ 条）。
波及面评估：[`data_integrity_impact_assessment.md`](data_integrity_impact_assessment.md)。
**独立复核（上一轮产物，先读这份）**：[`data_integrity_impact_assessment_review.md`](data_integrity_impact_assessment_review.md)。

复核的三条硬结论：

1. **③ 已零成本量化完毕**（[`tools/ch5_holdout_split_audit.py`](tools/ch5_holdout_split_audit.py)：把已刊 100 回合劈成参与选点的 40 条与未参与的 60 条）。留出 60 条上组间差**全部保号且变大**——翻转 $+1.6\to+3.0$ pp（5/5 种子不为负）、基线差距 $23.0\to28.0$ / $32.2\to40.0$ pp、§5.6 回落 $-7.6\to-9.0$ pp。**唯一例外**：§5.7.2「差距闭合过半」$53.0\%\to\mathbf{48.9\%}$，不再过半。
2. **① 的污染幅度 $\delta$ 仍未测，且可测**——评估里「污染只会抬高 2000 格」是**未经验证的推断**，不该当处置前提用。
3. 评估另有 6 处漏项，其中 §5.6.2 的 noisy-support 诊断牵出一个**此前未计入污染面的含噪 2000 数据集**（事实账本第 ⑤ 条）。

---

## 本轮任务

### 第一步：Drive 侧探针（用户在 Colab 跑）

[`../../notebooks/ch5_data_integrity_probe.ipynb`](../../notebooks/ch5_data_integrity_probe.ipynb) —— 无 GPU、不训练、不重采。

- **§1 探针 A**：2000 与 1000 格的**选中**检查点（`.pt`）是否还在 Drive。**这一条决定处置格局**：在 → ①-c 成立；已清 → 只剩「重采重跑」与「保留数字 + 降级翻转论述」二选一。（本机 `results/offline/**` 下 `.pt` 计数为 0。）
- **§2 探针 B**：含噪 2000 数据集的 `seed`／`num_episodes`，并在 Drive 全量 `offline_data/` 上跑 `python -m scripts.audit_seed_overlap`。关掉第 ⑤ 条这个未知。
- **§3（`RUN_CLEAN_PROBE`，默认 `False`）**：若 §1 通过，在干净 manifest（`--seed 3000`，与训练 0..1999 及现测试集 1250..1349 均不相交）上补评**两格 × 5 种子 = 10 次纯评估**。两格必须同 manifest——翻转是两格之差，只补 2000 而拿 1000 的旧数字比是混口径。manifest 配方已在本机验证：`--seed 1250` 逐条重现现有 `benchmarks/single_u10_cross_tgt15_ep100.json`（100/100），零改代码；notebook 内置该自检，失败即中止。

回传 git：`benchmarks/clean_probe/single_u10_cross_tgt15_ep100_s3000.json` + `results/offline/rebrac/clean_probe/**/seed_*.json` + notebook 的 `_completed` 版。**不回传 `.pt`。**

### 第二步：拿读数做裁决（用户拍板）

判定门槛（复核文档 §4 ★3）：章内真实断言是「**未再检出**回落」而非「2000 > 1000」，故

- $\delta \gtrsim 2$ pp → 「点估计反而略高」半句失效；
- $\delta \gtrsim 5\text{–}7$ pp → 才构成与 TD3+BC（$-7.6$ pp）/ 纯 BC（$-5.4$ pp）同量级的回落，翻转论述才真被推翻。

分流：干净读数守得住 → 「披露 + 敏感性读数」，不必重采重跑；守不住 → ①-b 降级（但**是拿证据降级**）；介于两者 → ①-a 重采这才有正当理由。

### 第三步：一次性整改批（唯一动 `.tex` 的一步）

**① 与 ③ 必须同批**——分两批等于付两次重流程复审，而本章「送审就绪」正是靠五批整改逐字节证出来的。批内容随探针结果定，至少含：③ 的五处口径句 + 复核补的两处漏项（筛查单元无留出集、§5.6.2 而非 §5.6.1）、①-b/①-c 的披露与敏感性读数、§5.7.2 那条不利读数（48.9%）、以及第 ⑤ 条决定的 noisy-support 处理。

---

## 硬约束

1. **在裁决之前不动任何 `.tex`。** 动正文即触发 `prompt_playbook.md` 的重流程复审义务，五批整改逐字节证明过的「零论证／零数字漂移」性质随之失效。
2. **不凭记忆写数字。** 每条都应能核到 `.tex` 正文、[`../../docs/rebrac_experiment_report.md`](../../docs/rebrac_experiment_report.md)（数字唯一权威源）、数据集 `metadata.json`、`results/offline/**/test/seed_*.json` 的逐回合记录，或代码。
3. **处置不由 agent 决定。** 给判断和依据，裁决权在用户。
4. 已知的一条易复发错误：**§5.3.6 正文从未写过「独立测试集」**。该误称出自 `td3bc.tex` rev.6 头注对 GT 报告的转述，2026-08-16 已订正，别再写回去。

---

## 相关提交

`09d0a44`（① 核实）· `e31cc58`（波及面评估）· `36fc060`（独立复核 + hold60 工具）· `22dac93`（事实账本订正 + 探针 notebook）
