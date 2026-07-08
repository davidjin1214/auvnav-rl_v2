# 第 5 章续写 — 本轮入口 Prompt（补充验证执行轮 · Colab 回读判读）

> 体例（2026-07-07 起）：本文件**每轮重写**，只含当轮任务与相关约束。章状态账本 = `status.md`（唯一真相源）；逐轮历史 = git log 与各节头注；prompt 方法论 = `prompt_playbook.md`。
> **流程状态**：补充验证盘点轮已闭环（2026-07-08）——道 1 部分落地（report §7.10 增补，3/9 本机实核，六缺格取证缺口成立）；道 2 方案获批（推荐案全批）。用户同日拍板：六缺格走**同协议补跑重取证**并入执行轮。执行 notebook 已建 4 份（builder `scripts/_build_supplementary_verification_notebooks.py`）。
> **先决条件（本轮动笔前确认）**：用户已在 Colab 跑完下述 4 个 notebook 并把 results 同步回本地。未跑完则本文件继续等待，不做任何判读。

---

本轮任务：**回读 Colab 执行结果，按预登记门槛判读，一致则落地零论证微修、不一致则呈报硬停。**

## 执行物（用户 Colab 侧，本轮回读对象）

| notebook | 单元 | 验尸输出 |
|---|---|---|
| `sac_arrival_v2_sensing_crit_rescue_seed0.ipynb` | s1_k4/s2_k4 × seed 0（在线 1M ×2） | `experiments/arrival_v2_prototype/sensing_crit_rescue_summary/rescue_verdict_seed0.json` |
| `sac_arrival_v2_sensing_crit_rescue_seed7.ipynb` | s0/s1/s2_k4 × seed 7（×3） | `…/rescue_verdict_seed7.json` |
| `sac_arrival_v2_sensing_crit_rescue_seed42.ipynb` | s2_k4 × seed 42（×1） | `…/rescue_verdict_seed42.json` |
| `rebrac_broad_validation_v2_seed43_supplement.ipynb` | N0/N2′/N2′-asym × seed 43（离线 ×3） | `results/offline/rebrac/broad_validation_v2/summaries/seed43_supplement_verdict.json` |

## 道 1 回读：临界传感 6-cell 补跑（E·M1）

1. **数字一律回查 raw `final_eval.json`**（六个 canonical run dir），verdict JSON 只作索引；
2. 六格全部与转录值**逐位一致** → `docs/arrival_v2_experiment_report.md` §7.10：⚠ 全改 ✅（证据列改指补跑文件）、删 PARTIAL 状态段与处置决定引注；`chapter_acceptance_review_5_1_5_6_findings.md` E·M1 勾销；`status.md` 登记项勾销。**`.tex` 零改动**；
3. **任何偏差（含 1 episode 之差）→ 呈报硬停**：§7.10 与 `.tex` 一字不动，列出偏差明细（cell / 补跑值 / 转录值 / Δ）等用户裁决。预期后续 = 以补跑值为准重排 §5.5.2、瓶颈表 k=4 与 s1 行、§5.5.5 floor、fig sensing_crit、§5.8.4 对照句（0.26）——属「新的事实性问题」流程，**本轮不得先行动手**。

## 道 2 回读：§5.8 补种子（seed 43）

1. 数字回查三份 `test_result.json`（N0/N2p/N2p_asym 的 `seed_43`）；
2. 按 plan §2 预登记门槛判读（N0 均值 ∈ [0.70, 0.902)；N2′ 0/30；消融 0/30 且越界主导）；
3. **三者全一致** → 按 plan §4 清单落地（顺序固定）：① ground truth 先行——v2 report 头部 caveat、§2/§4.5 表补 seed 43 列、§6.3 勾销；② `boundary.tex` 定点 Edit（caption 种子数、per-seed 列、0/60→0/90、上界 0.05→约 0.033、层级表标注、§5.8.m 登记语换三种子如实登记含 {42, 0, 43} 与预登记组不重合披露）；③ latexmk 编译 + `latexmk -c`；④ spec §2 §5.8 答辩风险登记降级、`status.md` 勾销、`section_5_8_review_findings.md` H1 追注；
4. **任一 ESCALATE** → 全部微修冻结，呈报。

## 红线（本轮特有）

- 预登记门槛不允许 post-hoc 调整；判读只按 plan §2 / 附录 A 原文执行；
- `.tex` 十节 locked：道 1 一致时零 `.tex` 改动；道 2 一致时只碰 plan §4 清单内的 `boundary.tex` 定点（超出清单一字不动）；
- 数字不从 notebook 打印或 verdict JSON 誊抄，一律回查 raw 结果文件；
- 两道判读互相独立：一道呈报不冻结另一道的落地。

## 按序必读

1. `status.md` 登记项表两行（当前状态与勾销条件）；
2. `docs/rebrac_broad_validation_v2_seed43_supplement_plan.md` §2/§4 + 附录 A（预登记门槛与微修清单原文）；
3. `docs/arrival_v2_experiment_report.md` §7.10（九宫格、⚠/✅ 现状、缺失清单）；
4. `sections/boundary.tex` 头注 + §5.8.5/§5.8.m（道 2 一致时的编辑面现状）。

## 输出与工程

- 道 1 产出：§7.10 状态翻转（或呈报单）+ E·M1/status 勾销；
- 道 2 产出：v2 report 增补 + `boundary.tex` 定点微修 + 编译验证（或呈报单）；
- 可自行 commit：`docs:` 前缀 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer，只暂存本轮相关文件。
