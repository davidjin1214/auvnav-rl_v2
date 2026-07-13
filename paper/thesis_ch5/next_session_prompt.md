# 第 5 章续写 — 本轮入口 Prompt（道 1 重排方案裁决 · §5.5 九宫格落地轮）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> **流程状态**：补充验证两道已分流（2026-07-12）——**道 2 全流程闭环**（seed 43 补种子：v2 report 三种子增补 + §2.4 改写 + `boundary.tex` rev.3 + 编译零警告 + spec/H1/status 勾銷）；**道 1 用户已裁决 (a)**（六格补跑值为新 ground truth），重排方案呈批稿已落档 `docs/rebrac_broad_validation_v2_seed43_supplement_plan.md` **附录 C**，**待批**。

---

本轮任务：**用户批附录 C 方案（含 4 个待批点）→ 按 C.3 执行序一轮落地 §5.5 九宫格重排；未获批则一字不动。**

## 附录 C 待批点（原文见 plan 附录 C）

1. **C.1 混用规则**：3 本机原件格 + 6 补跑格按「同协议分批执行」混作九宫格，分批事实在 §7.10 与 §5.5.m 各披露一句；
2. **C.2 措辞方向**：「几近失败」→「种子间剧烈分化（0.10–0.87）、均值 0.46」；gap 60→44pp；§5.5.3 补「方差单调坍缩 0.39→0.22→0.04」句；§5.8.2「不足以支撑」→「不足以稳定支撑」；
3. **C.3 ⑧ 复审选项**：落地后是否加一轮定点复审（只审数字忠实性与措辞越权）；
4. **C.4 raw 同步**：六份补跑 `final_eval.json` 是否从 Drive 同步回本机 canonical 树（否则 §7.10 证据列注明 Drive 为原件所在）。

## 获批后执行序（= plan 附录 C.3，越序即错）

① `docs/arrival_v2_experiment_report.md` §7.10 重写（ground truth 先行）→ ② fig 脚本 ×2 改值重绘（`sensing_crit` CRIT_SEEDS、`monotonic` CRIT_K4_S0）→ ③ `online.tex`（§5.5.2/瓶颈表/§5.5.3/§5.5.4 caption/§5.5.5/§5.5.m/头注）→ ④ `boundary.tex`（§5.8.2 限定词、§5.8.4 正文+层级表，头注 rev.4）→ ⑤ spec 0.26 数字源句 → ⑥ latexmk + `-c` + grep 残留（`0.26|0.61|60 个百分点|几近失败`）→ ⑦ E·M1 勾销 + `status.md` 勾销 + 本文件重写。

## 红线（本轮特有）

- 新值以 plan 附录 C.1 矩阵为准，落稿前逐格回查 raw（本机原件三格 + Drive/同步后的补跑六格）——不从附录 C 誊抄；
- 措辞改写限 C.2 列明方向：核心结论（瓶颈在时序利用）不得动摇也不得加码；「s2 与 s1 并列」不得引申出「空间参照无用」等新论点；
- C.4 已核零改动项（manifest floor 图与 §5.5.4、`discussion.tex`、特权消融 0.267 峰值）**不碰**；
- 道 2 已闭环，`boundary.tex` §5.8 除 C.3 ④ 两处外一字不动。

## 按序必读

1. plan 附录 C 全文（矩阵/影响评估/清单/已核事实）；
2. `sections/online.tex` §5.5.2–§5.5.5 + §5.5.m + 头注（rev 块含 2026-06-18 转录史）;
3. `docs/arrival_v2_experiment_report.md` §7.10 现状；
4. `figures/scripts/fig_ch5_online_sensing_crit.py` + `fig_ch5_online_monotonic.py` 头部数据块。

## 输出与工程

- 产出：§7.10 重写 + 两图重绘 + `online.tex`/`boundary.tex` 定点落地 + 编译验证 + 勾销链（或按用户改批的其它处置）；
- 可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存本轮相关文件。
