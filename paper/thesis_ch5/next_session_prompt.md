# 第 5 章续写 — 本轮入口 Prompt（补充验证全线闭环后 · 待新指令）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> **流程状态**：补充验证两道全部闭环（2026-07-19）——**道 2**（§5.8 补种子 43）已于 2026-07-12 落地；**道 1**（§5.5 临界传感九宫格重排）已于 2026-07-19 落地并通过独立定点复审（PASS：15 格逐格溯源、措辞零越权、不动项零触碰）。

---

本轮任务：**无预定任务，等用户新指令。**

## 上一轮（2026-07-19 道 1 落地轮）交接摘要

- 用户批附录 C 四点，其中 C.1 按用户修正执行：2026-06-18 旧转录值定性为**引用错误、完全作废（占位数字）**，不作「分批执行」论证、不比对新旧差异；
- 新 ground truth（唯一权威 = `docs/arrival_v2_experiment_report.md` §7.10，✅ 9/9 本机实核）：s0 0.46±0.39（0.400/0.867/0.100，剧烈分化）、s1 0.90±0.00、s2 0.87±0.03、gap(s1−s0) ≈ 44pp；方差链 k=4→8→12 = 0.39→0.22→0.04；
- 六份补跑 `final_eval.json` 已从 Drive `rl_v2_5` 树溯源同步回本机 canonical 树（C.4）；
- 落地面：§7.10 重写、两图重绘（`sensing_crit`/`monotonic`）、`online.tex` rev.7（§5.5.2/caption/§5.5.3 方差收缩句/瓶颈表/§5.5.4 caption/§5.5.5）、`boundary.tex` rev.4（§5.8.2 限定词 + §5.8.4 三处 0.46±0.39）、spec 六处同步；latexmk 58 页 0 undefined / 0 multiply / 无 Overfull + `latexmk -c`；残留 grep 全清（旧值仅存于头注 rev 历史记录，体例合规）；
- 勾销链：findings E·M1 ✅、findings §5 交接残句追注、`status.md` 道 1 行 ✅。

## 章级现状（详见 `status.md`）

- §5.1–§5.10 十节全部成文；补充验证事项（道 1 / 道 2）全清；findings 各级（CRIT/HIGH/MED/LOW）全清；
- 待决表已无 ⏳ 项；观察项：全章为 58 页单章编译，若后续并入全文模板需复查浮动与页码。

## 常备红线（跨轮有效）

- **不从记忆写数字**——一切刊值溯源 `docs/rebrac_experiment_report.md`（离线主线）、`docs/rebrac_broad_validation_v2_report.md`（§5.8）、`docs/arrival_v2_experiment_report.md` §7.10（§5.5 临界传感）与 `docs/online_rl_line_summary.md` §1.1（A0）；
- 已定稿节 `.tex` 视为 locked：只 Edit 微修、不 Write 覆盖；改动须走「方案→批准→落地→复审」重流程（实质改写）或轻流程（零论证微修）；
- 可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存当轮相关文件。
