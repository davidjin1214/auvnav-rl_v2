# ReBRAC Broad Validation v2 — 补种子（seed 43）Supplement Plan

> **Status: ✅ APPROVED（用户批准 2026-07-08）**——按推荐方案（最小矩阵 +seed 43 × 3 单元全补）执行；同日用户追加批准**附录 A**：临界传感 6-cell 同协议补跑（道 1 rescue，Mac 不在身边、云端 final_eval 不可达）并入本执行轮。
> **⛔ 回读判读（2026-07-12）：两道均呈报硬停，微修全冻结**——道 1 六格 5/6 与转录值不一致（含 s0/seed_7 +60pp 重大偏差）；道 2 三门槛全过但 N0 seed 43（0.933）越过主线锚点 0.902、同向退化叙事失效，§4 零论证清单不足以自洽落地。偏差明细、影响面分析与待裁决点见**附录 B**；论文与两份 ground truth report 一字未动。
> **✅ 用户裁决（2026-07-12）：道 1 选 (a)、道 2 选 (a)**——道 2 扩权清单已同日落地（v2 report 三种子增补 + `boundary.tex` 定点微修 + 编译验证，见附录 B.2 追注）；道 1 走「新的事实性问题」重流程，重排方案呈批稿见**附录 C**，获批前 `.tex` 与 §7.10 不动。
> 执行 notebook（已建，2026-07-08）：`notebooks/rebrac_broad_validation_v2_seed43_supplement.ipynb` + `notebooks/sac_arrival_v2_sensing_crit_rescue_seed{0,7,42}.ipynb`（builder：`scripts/_build_supplementary_verification_notebooks.py`）。
> 原呈批稿正文（§1–§5）内容不变，作为已批协议保留。
>
> **来源**：博士论文第 5 章定稿前补充验证事项——§5.8 两单元与消融现为 2 seed `[42, 0]`，所依计划预登记为 3 seed `[42, 43, 44]`（[`rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) §6.2 / 行 147）。缺口出处：spec §2 §5.8 答辩风险登记；[`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) 头部 seed-count caveat + §6.3 follow-up backlog；`paper/thesis_ch5/notes/section_5_8_review_findings.md` H1（预登记种子缩水披露）。§5.8.m 已如实登记该缺额。
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

## 5. 待批点（✅ 已全部裁决 2026-07-08）

1. **矩阵规模**：~~最小矩阵 vs 预登记组补全~~ → **最小矩阵（+43，3 单元）**；
2. **是否三单元全补**：→ **全补**（N0 / N2′ / 消融）；
3. Colab session 安排：→ 执行轮 notebook 已建（见头注），用户在 Colab 跑完后由回读轮按 §2 判读、§4 微修或呈报。

---

## 附录 A：临界传感 6-cell 同协议补跑（道 1 rescue，用户批准 2026-07-08 并入本执行轮）

**背景**：`docs/arrival_v2_experiment_report.md` §7.10 取证缺口——§5.5.2 九读数中六份云端 `final_eval.json` 本机不可达（详见该节）；原件应在 Mac 上但 Mac 不在身边，用户裁决走**同协议补跑重取证**。

**矩阵（6 个 1M-step 在线训练单元，按 seed 分 3 道并行）**：

| 道（notebook） | 单元 | 转录期望值 | 预估 L4 |
|---|---|---|---:|
| `sac_arrival_v2_sensing_crit_rescue_seed0.ipynb` | s1_k4/seed_0、s2_k4/seed_0 | 0.900、0.800 | ~5h |
| `sac_arrival_v2_sensing_crit_rescue_seed7.ipynb` | s0_k4/seed_7、s1_k4/seed_7、s2_k4/seed_7 | 0.267、0.800、0.733 | ~7.5h（可 `[skip]`/resume 跨 2 session） |
| `sac_arrival_v2_sensing_crit_rescue_seed42.ipynb` | s2_k4/seed_42 | 0.733 | ~2.5h |

合计 ~15h L4（1M ≈ 2.5h/run）；与附正文 3 个离线单元（≤2h）互不依赖，可并行。

**协议**：与 §7.1/§7.6/§7.7 逐项一致（vanilla SAC / arrival_v2 / k=4 / 1M / num_envs=6 / random=update_after=5000 / eval 25k×30ep / 终检 30 deterministic ep / flow wake_v8_U1p50_Re250 / 固定 manifest `single_u15_cross_tgt15.json` 禁止再生成），唯一变量 = probe layout × seed；输出落 §7.10 缺失清单的 canonical 路径。

**预登记判读语义（跑前锁定）**：

- 补跑 `final_eval.json` 即该 cell 的**新 ground truth**（可追溯证据）；
- 六格全部与转录值逐位一致 → §7.10 ⚠→✅、E·M1 勾销（零论证改动）；
- **任何偏差 → 呈报硬停**：`.tex` 刊值与 §7.10 数值一字不动，由用户裁决（预期处置 = 以补跑值为准重排 §5.5.2/瓶颈表/floor 刊值，属「新的事实性问题」流程）；
- 若 Mac 文件先行同步到 Drive/本地，notebook 的 `[skip]` 判定自动跳过对应 cell，不覆盖原件。

**验尸输出**：每道写 `experiments/arrival_v2_prototype/sensing_crit_rescue_summary/rescue_verdict_seed{N}.json`（per-cell 补跑值 / 转录值 / EXACT_MATCH / 终止构成 / obs_dim 核对）。

---

## 附录 B：执行回读与呈报（2026-07-12，回读判读轮）——两道均呈报硬停

> **Status: ⛔ ESCALATED（两道并立，互相独立）**——道 1 触发附录 A 预登记的「任何偏差 → 呈报硬停」；道 2 预登记门槛全过但 §4 零论证清单不足以自洽落地（新的事实性情况）。**论文 `.tex` 十节、`docs/arrival_v2_experiment_report.md` §7.10、`rebrac_broad_validation_v2_report.md` 均一字未动**，等用户裁决。
>
> **回读取证方式披露**：4 个 notebook 均已在 Colab 执行完毕；本机仅同步回两份 verdict JSON，9 份 raw 结果文件（6 份 `final_eval.json` + 3 份 `test_result.json`）未同步到本机树。本轮经 Google Drive 连接器直接读取 Drive 上的 raw 原件，并逐层回溯父目录链（文件 → `results`/`seed_43` → `seed_N`/单元 → `s{X}_k4`/树根）确认每份文件归属；下表数字以 Drive raw 原件为准（与 verdict JSON 逐位一致，判读独立重算）。

### B.1 道 1（临界传感 6-cell 补跑）：5/6 与转录值不一致 → 呈报硬停

偏差明细（补跑值 = Drive raw `final_eval.json` 的 `eval_success_rate`，30 deterministic ep）：

| cell | 补跑值 | 终止构成（补跑） | 转录值（2026-06-18） | Δ | 判定 |
|---|---:|---|---:|---:|---|
| s1_k4/seed_0 | 0.900 (27/30) | goal 27 / OOB 3 | 0.900 | 0 | EXACT_MATCH |
| s2_k4/seed_0 | 0.867 (26/30) | goal 26 / OOB 4 | 0.800 | +0.067 | 偏差 |
| s0_k4/seed_7 | **0.867** (26/30) | goal 26 / OOB 4 | **0.267** | **+0.600** | 重大偏差 |
| s1_k4/seed_7 | 0.900 (27/30) | goal 27 / OOB 3 | 0.800 | +0.100 | 偏差 |
| s2_k4/seed_7 | 0.833 (25/30) | goal 25 / OOB 3 / timeout 2 | 0.733 | +0.100 | 偏差 |
| s2_k4/seed_42 | 0.900 (27/30) | goal 27 / OOB 3 | 0.733 | +0.167 | 偏差 |

obs_dim 核对全部通过（s0/s1/s2 × k4 = 48/56/72，与 §5.8.m 刊值一致）；manifest 均为 `single_u15_cross_tgt15`。

**判读分析（供裁决参考，非处置）**：

1. 五格偏差全部向上（+0.067 ～ +0.600），不是对称分布的复现噪声形态；
2. 对 1M-step 在线 SAC，同 seed 同协议重训本就不保证逐位复现（GPU 非确定性 + AsyncVectorEnv 调度），「六格逐位一致」的预登记门槛对在线补跑近乎必然触发呈报；但 s0_k4/seed_7 的 +60pp（8/30 → 26/30）远超一般训练随机性量级——或者 2026-06-18 转录值对应的是另一个 run / 存在誊抄层面的系统性问题（转录三源互核一致但可能同源），或者该 cell 的 run-to-run 方差本身巨大。两种读法都动摇「s0@k4 floor≈0.26」的稳定性；
3. 若按附录 A 预登记的预期处置「以补跑值为新 ground truth 重排」，s0 三种子成为 {0.400✅原件, 0.867补, 0.100✅原件}（mean 0.456 / std 0.385），s1 = {0.900, 0.900, 0.900}，s2 = {0.867, 0.833, 0.900}，gap(s1−s0) 由 0.61 缩为约 0.44 且 s0 组内离散巨大——影响面不止预告的 §5.5.2 / 瓶颈表 / §5.5.5 floor / fig sensing_crit / §5.8.4 对照句（0.26±0.15），「floor」概念本身及 §5.10 相关收束句都会被动摇；两个本机原件 ✅ 格与四个补跑格能否混作一个 3-seed 矩阵也须一并裁决；
4. s1_k4/seed_0 的 EXACT_MATCH 在 30 回合离散值空间内不排除巧合，不构成「其余格转录可信」的证据。

**待裁决**：(a) 是否接受补跑值为新 ground truth，启动「新的事实性问题」专轮重排 §5.5（影响面见上，建议专轮方案先行呈批）；(b) 或先等 Mac 原件可取时取证比对 2026-06-18 转录值来源再定；(c) 裁决前 §7.10 维持 PARTIAL 3/9、E·M1 不勾销。

### B.2 道 2（§5.8 补种子 seed 43）：预登记门槛全过，但落地清单不自洽 → 微修冻结呈报

Drive raw `test_result.json` 回查值（目录链溯源确认归属）：

| 单元 | seed 43 | 终止构成 | 门槛（plan §2 原文，均值读法） | 判读 |
|---|---:|---|---|---|
| N0 | 0.933 (28/30) | goal 28 / OOB 2 | 3-seed 均值 0.878 ∈ [0.70, 0.902) ✓ | CONSISTENT |
| N2′ | 0.000 (0/30) | OOB 29 / depth_hold_failure 1 | 0/30 零成功 ✓ | CONSISTENT |
| N2′-asym | 0.000 (0/30) | OOB 30（越界主导 ✓） | 0/30 且越界主导 ✓ | CONSISTENT |

N0 三种子 {42, 0, 43} = {0.867, 0.833, 0.933}：mean 0.878、std (ddof=1) 0.051（本轮独立重算，与 verdict JSON 一致）。

**呈报原因（新的事实性情况，非门槛失败）**：seed 43 = 0.933 **高于** efficiency_v2 主线锚点 0.902（+3.1pp），per-seed 退化方向不再一致（42: −3.5pp / 0: −6.9pp / 43: **+3.1pp**）。附注：plan §2 门槛第二子句「vs efficiency_v2 anchor 0.902 仍同向退化」按承前主语（3-seed 均值）读为「均值仍低于 0.902」时通过（0.878 < 0.902，Δ 收窄为 −2.4pp）；若按 per-seed 同向读法则该子句不满足、直接 ESCALATE——两种读法殊途同归到本呈报。其后果：

1. `boundary.tex` §5.8.1「两种子同向退化……同向性提示这更可能是……系统性效应，而非种子噪声」——三种子下同向性不成立，该段为论证级文本，如实改写超出 §4 零论证清单授权；
2. v2 report §2.4「Why the 5.2pp drop is not noise」同理失实（binomial 同向论证在 2/3 同向下削弱）；
3. §4 清单另遗漏两处纯事实性定点：§5.8 开篇段「各以两个随机种子训练／证据分量都应按两种子解读」句、§5.8.1 正文锚点数字（0.850±0.024 与 −5.2pp）——机械执行字面清单将产出表（0.878±0.051）与正文（0.850±0.024）直接矛盾的文稿。

按「超出清单一字不动」红线与「任一不自洽 → 全部微修冻结」的保守读法，v2 report 与 `boundary.tex` 全部未动。

**待裁决**：(a) 批准扩权清单一次性落地——原 §4 清单 + §5.8.1 同向段与 report §2.4 的如实改写（方向：三种子 Δ=−2.4pp、方向不一、同向性论据撤销、「系统性退化」降级为「幅度收窄且方向存疑」）+ 上述两处遗漏定点；(b) 或其他处置。N2′/消融两项（0/90、上界约 0.033）无叙事冲击，其微修只等与 N0 处置一并批准后执行。

> **✅ 裁决与落地追注（2026-07-12）**：用户批 (a)，扩权清单当日全部落地——v2 report：头部 Addendum/caveat、§1 核心表与结论句、§2.1/§2.2/§2.3 表、**§2.4 论证级改写**（同向性撤销、原判读存档）、§3.1/§3.2 表与计数（88/90）、§4.5 表与 0/90/0.033、§5.3/§5.4/§6.1/§6.3 联动、§7.2 留痕表补五行；`boundary.tex` rev.3：头注 ground truth 块与 rev 记录、开篇段种子数、汇总表（含排版收紧消 Overfull）、§5.8.1 锚点数字 + 同向段改写、§5.8.2/§5.8.3 计数与终止构成、层级表两行与 caption、§5.8.5/§5.8.m 登记语（{42, 0, 43} 不重合披露）、n=90 上界；latexmk 编译 58 页 0 undefined / 0 multiply / 无 Overfull>10pt + `latexmk -c`；spec §2 §5.8 答辩风险降级追注；findings H1 闭环追注。**§5.8.4 在线参照 0.26±0.15 未动（道 1 裁决范围）**。

---

## 附录 C：道 1 (a) 落地方案呈批稿——§5.5 临界传感九宫格以补跑值重排（2026-07-12 起草，✅ 已批并落地 2026-07-19）

> **性质**：用户已裁决道 1 (a)（补跑值为新 ground truth，启动「新的事实性问题」专轮）；按裁决轮红线，本方案先呈批，**获批前 `.tex`、report §7.10、fig 脚本一字不动**。方案已含本轮回读时核实的全部前置事实（C.4），获批后按 C.3 顺序一轮落地。
>
> **✅ 裁决与落地追注（2026-07-19）**：用户批四点——C.1 **按用户修正执行**（不作「同协议分批执行」的方法学论证：旧转录值直接定性为**引用错误、完全作废**，视作占位数字，不比对两次结果差异，§5.5.m 相应不加分批披露句）；C.2 措辞方向照批；C.3 ⑧ 定点复审加；C.4 六份补跑 raw 同步本机。当日按 C.3 执行序全部落地：report §7.10 重写（✅ 9/9 本机实核）→ 两图脚本改值重绘 → `online.tex` rev.7 → `boundary.tex` rev.4 → spec 六处同步 → latexmk 58 页零警告 + `-c` + 残留 grep 全清 → E·M1/status/next_session_prompt 勾销。独立定点复审（干净上下文子代理，只审数字忠实性与措辞越权）**PASS**：15 格逐格向 raw 溯源一致、措辞零越权、不动项零触碰（1 条 LOW——findings §5 旧交接残句「用 0.26」——已当轮追注存档）。本附录 C.1 矩阵中「混用规则（须批）」段的分批论证与转录疑点分析**以本追注为准作废**，仅存档。

### C.1 新 ground truth 矩阵（终检成功率，seeds 0 / 7 / 42）

| 配置 | 旧刊值（2026-06-18 转录） | 新值（补跑 + 本机原件） | 格来源 |
|---|---|---|---|
| s0（10-D деploy） | 0.400 / 0.267 / 0.100 | 0.400 / **0.867** / 0.100 | seed_0、seed_42 = 本机原件✅；seed_7 = 补跑 |
| s1（12-D 参照） | 0.900 / 0.800 / 0.900 | 0.900 / **0.900** / 0.900 | seed_42 = 本机原件✅；seed_0（与转录一致）、seed_7 = 补跑 |
| s2（16-D 参照） | 0.800 / 0.733 / 0.733 | **0.867** / **0.833** / **0.900** | 三格全补跑 |

聚合（mean ± std, ddof=1）：**s0 0.26±0.15 → 0.46±0.39**；**s1 0.87±0.06 → 0.90±0.00**（三种子同值）；**s2 0.76±0.04 → 0.87±0.03**；**gap(s1−s0) 0.61 → 0.44**。

**混用规则（须批）**：九格全部为同协议独立训练 run 的 30-ep 确定性终检值（附录 A 协议逐项一致 + obs_dim 核对通过），本机原件与补跑格的差别仅是训练批次时期（2026-05 vs 2026-07）——方法学上等同多种子实验分批执行，混用成立；分批事实在 report §7.10 与 §5.5.m 各披露一句。转录值来源疑点（5/6 全向上、s0/seed_7 旧值 0.267=8/30 与特权消融峰值 0.267 数值巧合，存在转录错位可能）作为历史存档写入 §7.10，不再影响刊值。

### C.2 叙事影响评估（获批后按此方向改写，越权即停）

1. **§5.5.2 核心叙述**：「s0 几近失败（约 0.26）」不再成立——新形态为**种子间剧烈分化**（0.10–0.87，std 0.39），均值 0.46；gap 缩至约 44pp 但仍显著。「s2 居中」失效（s2 0.87 与 s1 0.90 几乎并列），「更多通道未换来增益」保留但弱化。
2. **§5.5.3 剂量响应**：k=4→8→12 均值 0.46→0.76→0.88 仍单调，但新增更强信号——**跨种子方差单调坍缩（0.39→0.22→0.04）**：时序窗延长不仅抬升均值，更把种子间剧烈分化压成一致收敛。k=4↔k=8 单点均值差不再显著，单调性论证重心移到 §5.5.4 已有的「单次训练内部单调跃迁」与 k=12 收敛一致性。
3. **§5.5.5 在线参照句**：「均值仅约 0.26、几近失败」→「均值约 0.46 且种子间剧烈分化」；k=12 救援的对照从「失败→成功」改述为「不可靠→稳定」。
4. **§5.8.4（boundary.tex）**：层级表在线行 0.26±0.15 → 0.46±0.39；「离线 0 低于在线」的次序发现**更强**（0 vs 0.46）；§5.8.2「k=4 单点观测不足以支撑闭环校正」→「不足以稳定支撑」。
5. **红线核查**：§5.5 核心结论（瓶颈在时序利用）与 §5.8 临界结论在新数下均保持乃至增强；中心命题无损。§5.10 零波及（C.4）。

### C.3 编辑面清单（获批后执行序）

① **ground truth 先行**——`docs/arrival_v2_experiment_report.md` §7.10 重写：九宫格新值 + ⚠→✅（证据列改指补跑 `final_eval.json` 的 canonical 路径 + `rescue_verdict_seed{0,7,42}.json`）、删 PARTIAL 状态段、处置决定段改裁决记录、均值/gap 重算、补分批披露与转录疑点存档；
② **fig 脚本重绘**：`figures/scripts/fig_ch5_online_sensing_crit.py` CRIT_SEEDS 三行 + docstring；`fig_ch5_online_monotonic.py` CRIT_K4_S0 + docstring（[0.40, 0.867, 0.10]）；
③ **online.tex**：§5.5.2 正文与 caption（数字 + 「几近失败/居中」措辞）、瓶颈表 k=4 与 s1 行、§5.5.3 剂量响应句（补方差坍缩）、§5.5.4 ksweep caption「近底板」措辞、§5.5.5 参照句、§5.5.m 分批披露、头注 rev 块与 ground truth 指引；
④ **boundary.tex**：§5.8.2 限定词、§5.8.4 正文 + 层级表行，头注 rev.4；
⑤ **spec**：§5.5/§5.8 块内 0.26 数字源句更新；
⑥ latexmk 编译 + `-c` + 全文 grep 残留（`0.26|0.61|60 个百分点|几近失败`）；
⑦ 勾销：`chapter_acceptance_review_5_1_5_6_findings.md` E·M1 追注勾销、`status.md` 道 1 行勾销、`next_session_prompt.md` 重写；
⑧ **复审选项（呈批点）**：§5.5 为已闭环节，本次属实质数字重排 + 措辞连动——建议落地后加一轮定点复审（只审数字忠实性与措辞越权），是否加由用户批。

### C.4 已核事实（本轮回读时核实，防重复劳动）

- 补跑 s0_k4/seed_7 失败回合 = {ep 3, 8, 16, 28} **⊇** universal-floor 三回合 {8, 16, 28} → 评估集经验上界 27/30=0.90 不破坏；`fig_ch5_online_manifest_floor.py` 的 well-trained runs 集（k=12×3 + k=8×2）不含 k=4，**该图与 §5.5.4 叙述零改动**；
- `discussion.tex`（§5.10）对 0.26/60pp/floor **零直接引用**，收束层零波及；
- §5.5.3 特权消融峰值 0.267 与「标准 SAC 峰值区间 0.37–0.53」为周期评估 `eval_log` 数据源（本机），独立于终检重排，**不动**；
- s1 三种子同值 0.900（std=0.00），如实报；
- 六份补跑 raw 均已回查 Drive 原件并溯源目录链（附录 B 披露）；本机 canonical 路径尚无这六份 `final_eval.json`——落地前建议把六份从 Drive 同步到本机 `experiments/...` 树（或在 §7.10 证据列注明 Drive 为原件所在）。

**待批点**：(1) C.1 混用规则；(2) C.2 措辞方向；(3) C.3 ⑧ 是否加定点复审轮；(4) 六份 raw 是否同步本机。
