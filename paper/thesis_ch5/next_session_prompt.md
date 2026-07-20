# 第 5 章续写 — 本轮入口 Prompt（全章整体复审轮）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> **流程状态**：十节全闭环 + 补充验证两道全清（2026-07-19）后，用户 2026-07-20 拍板启动**全章整体复审轮**（复核轮候选 1）——对完整 58 页做一次性外审视角整体评审，覆盖历史逐节/分段复审从未做过的章级维度（十节连读论证弧、§5.1/§5.2 承诺 ↔ §5.7–§5.10 交付对齐、模拟答辩攻击面、章级学术分量终判）。

---

本轮任务：**在新对话中执行全章整体复审。**

- 启动方式：新开对话，整段粘贴 [`chapter_full_review_prompt.md`](chapter_full_review_prompt.md)（2026-07-20 起草，用户要求「保任务完成的硬护栏 + 复审维度与方法全放开」的精简版体例）。
- 该轮性质：**只评审、不改文件、不 commit**——产出分级问题清单 + 全章整体判定 + 答辩攻击面评估。
- 复审产出后的流程：用户确认必修范围 → 另开会话落地（locked `.tex` 只 Edit 微修；实质改写走「方案→批准→落地→复审」重流程）→ 落地后按惯例更新 `status.md` 与本文件。

## 上一轮（2026-07-20 复核收尾轮）交接摘要

- 道 1 落地轮（`be25765`）经逐项复核确认全部兑现；
- 账本滞后微修已提交（`ed19391`：status.md §5.8 行 rev.4 补录 + 已勾销行追注 + 「五处→六处」；CLAUDE.md 命令参考移交 `rl-v2-commands` skill 落库）；
- spec §8 #4（AsymCritic 并列呈现）状态行 + 两处交叉引用（§2 §5.5.m / §3 缝合点表）已同步为 resolved 2026-07-07。

## 章级现状（详见 `status.md`）

- §5.1–§5.10 十节全部成文；补充验证事项（道 1 / 道 2）全清；findings 各级（CRIT/HIGH/MED/LOW）全清；待决表零 ⏳ 项；
- 编译 58 页零警告；观察项：全章为单章编译，若后续并入全文模板需复查浮动与页码。

## 常备红线（跨轮有效）

- **不从记忆写数字**——一切刊值溯源 `docs/rebrac_experiment_report.md`（离线主线）、`docs/rebrac_broad_validation_v2_report.md`（§5.8）、`docs/arrival_v2_experiment_report.md` §7.10（§5.5 临界传感）与 `docs/online_rl_line_summary.md` §1.1（A0）；
- 已定稿节 `.tex` 视为 locked：只 Edit 微修、不 Write 覆盖；改动须走「方案→批准→落地→复审」重流程（实质改写）或轻流程（零论证微修）；
- 可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存当轮相关文件。
