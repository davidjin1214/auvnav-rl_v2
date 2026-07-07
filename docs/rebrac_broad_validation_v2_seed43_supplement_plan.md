# ReBRAC Broad Validation v2 — 补种子（seed 43）Supplement Plan

> **Status: 呈批稿（PENDING USER APPROVAL），2026-07-08 补充验证盘点轮产出。获批前不起 notebook、不跑任何实验。**
>
> **来源**：博士论文第 5 章定稿前补充验证事项——§5.8 两单元与消融现为 2 seed `[42, 0]`，所依计划预登记为 3 seed `[42, 43, 44]`（[`rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) §6.2 / 行 147）。缺口出处：spec §2 §5.8 答辩风险登记；[`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) 头部 seed-count caveat + §6.3 follow-up backlog；`paper/thesis_ch5/section_5_8_review_findings.md` H1（预登记种子缩水披露）。§5.8.m 已如实登记该缺额。
>
> **红线（承自本轮入口 prompt）**：补种子结果落地前，`boundary.tex` 的两种子 caveat 与引用限定**一字不动**；v2 report 为 §5.8 数字唯一权威。

---

## 1. 最小补种子矩阵（推荐）

固定基底与首轮完全一致（唯一新变量 = seed）：`cross_stream / target=1.5 / arrival_v2 / s0 / ReBRAC (β1=4, β2=2, hidden=256, critic_LN=on) / 64 epochs`，数据集与 eval manifest 复用首轮 Drive 上既有文件。

| # | 单元 | 配置（不变量） | 新增 seed | 终检 | 一致性基准（既有 2-seed） |
|---|---|---|---|---|---|
| 1 | **N0** anchor | crosscomp 数据 / sub-critical（u10/Re150 cross） | 43 | 30 deterministic ep | [0.867, 0.833] → HOLDS（≥0.70） |
| 2 | **N2′** critical probe | privileged 数据 / critical（u15/Re250 cross），vanilla critic | 43 | 30 deterministic ep | [0.000, 0.000] → STRONG_NEGATIVE |
| 3 | **N2′ asym 消融** | 同 N2′ + `--use-asymmetric-critic`（唯一变量） | 43 | 30 deterministic ep | [0.000, 0.000] → ACTOR_FUNDAMENTAL_CONFIRMED |

共 **3 个训练单元 + 3 次终检**。

**终检口径说明**：plan §6.2 原文写 `test_episodes=100`，但首轮实际执行为每 seed 30 回合终检（report §2 表 `(26/30)`、`boundary.tex`「每种子终检 30 回合」）。补种子**跟随实际执行口径（30 回合、同 manifest）**以保持与既有两种子严格可比；不回改为 100。

**种子组的如实登记**：补 43 后种子组为 `{42, 0, 43}`，与预登记 `{42, 43, 44}` 仍非同组（0 当年替换了 44）。落稿时 §5.8.m 登记语须如实写「三种子 {42, 0, 43}」并保留「与预登记组不完全重合」一句，**不得**宣称已回到预登记组。

**备选（供拍板）**：若要完整覆盖预登记组，则每单元补 seed 43 **和** 44（6 训练单元 + 6 终检，≈2 倍预算），种子组成为 `{42, 0, 43, 44}` ⊇ 预登记组。推荐仍为最小矩阵（+43）：N2′/消融为 deterministic 0.000、N0 判定余量大（0.850 vs 门槛 0.70），第三种子已足回应 H1 披露的答辩风险；+44 的边际收益主要是「组内完整」修辞。

## 2. 预登记判读门槛（跑前锁定，不允许 post-hoc 调整）

| 单元 | 「与两种子结论一致」判据 | 不一致时动作 |
|---|---|---|
| N0 seed_43 | 3-seed 均值 ≥ 0.70（HOLDS 不变）且 vs efficiency_v2 anchor 0.902 仍同向退化 | 按「新的事实性问题」呈报，§5.8 不动 |
| N2′ seed_43 | **0/30**（零成功完全一致） | 任何 >0 成功（即便 3-seed 均值仍 <0.15）都视为「零成功完全一致」这一事实性表述失效 → 呈报 |
| N2′ asym seed_43 | **0/30** 且终止构成仍由越界主导 | 同上呈报 |

三者全一致 → 只做 §4 的零论证定点微修；任一不一致 → 全部微修冻结，先呈报。

## 3. 执行与预算（Colab L4，单 session）

- **Notebook**：新 scaffold `notebooks/rebrac_broad_validation_v2_seed43_supplement.ipynb`——克隆 `rebrac_broad_validation_v2_core.ipynb` 的 N0/N2′ 段 + `rebrac_broad_validation_v2_n2p_asym_critic.ipynb` 的消融段，seed 参数改 43；一律 `!python -m scripts.train_offline --algo rebrac ...` shell magic（实时 stdout）；`[skip]` resume 以 `agent_final.pt` 判定。
- **输出落位**（与首轮同树）：`results/offline/rebrac/broad_validation_v2/{N0,N2p}/seed_43/test_result.json`、`results/offline/rebrac/broad_validation_v2_n2p_asym/seed_43/test_result.json`、checkpoints 落 `checkpoints/offline/rebrac/broad_validation_v2*/…/seed_43/`。
- **预算**：ReBRAC 64 epochs ≈ 30–45 min/单元（v2 smoke 外推；首轮 4-run 实测单 session ~30 min 亦有记录）→ 3 单元 + 终检保守 **≤ 2h L4，单 Colab session 完成**（+44 备选则 ≤ 4h / 1–2 session）。
- 依赖检查（notebook 首 cell）：两个数据集 `transitions.npz` 与 eval manifest 在 Drive 原路径存在即开跑，缺失即停呈报。

## 4. 一致时的定点微修清单（预登记编辑面，全部零论证改动）

**Ground truth 先行**（v2 report）：头部 seed-count caveat 更新；§2 / §4.5 表补 seed_43 列并重算 mean±std；§6.3 backlog 对应条目勾销；raw output 留痕表补三行。

**`boundary.tex`（locked，只按此清单定点 Edit）**：

1. 表（两单元与消融）caption「每单元两个随机种子」→「三个随机种子」；per-seed 列头「种子 42 / 0」→「种子 42 / 0 / 43」，补值并重算均值；
2. `0/60` → `0/90`（正文两单元 + 消融段，共约 3 处）；零计数 95% 上界「约 0.05」→「约 0.033」（3/90，约 2 处）；
3. 层级分解表两行「（两种子）」→「（三种子）」及其 caption 对应句；
4. §5.8.5 / §5.8.m：撤「预登记三种子缩水为二」缺额登记语，换为三种子如实登记（含 `{42, 0, 43}` 与预登记组不重合的披露）；
5. 编译验证 latexmk + 编后 `latexmk -c`；数字漂移零容忍（除上述清单外一字不动）。

配套勾销：spec §2 §5.8 答辩风险登记降级、`status.md` 登记项勾销、`section_5_8_review_findings.md` H1 追注。

## 5. 待批点

1. **矩阵规模**：最小矩阵（+43，3 单元，推荐）还是预登记组补全（+43+44，6 单元）；
2. **是否三单元全补**：也可只补 N2′ + 消融（N0 判定余量大）——但 H1 披露与 §5.8.m 登记语覆盖全部三行，推荐全补以免登记语出现「部分三种子、部分两种子」的碎口径；
3. Colab session 安排（获批后另起执行轮：建 notebook → 跑 → 按 §2 判读 → 按 §4 微修或呈报）。
