# Online RL 线总结报告

> ⚠ **写作出口 LOCKED 2026-06-02**：本线全部可引用素材的写作出口 = **博士论文第 5 章**（spec [`../paper/thesis_chapter_outline.md`](../paper/thesis_chapter_outline.md) rev.4）；A0 sensor screen → §N.2 节（online RL：可学性与信息瓶颈），SAC collector cross-source matrix → §N.6 末段章级 headline（algorithm × data-quality interaction）。**Online SAC standalone paper（曾候选）已撤销**；不另起独立投稿。文中残留的「ReBRAC 论文投稿/接收后」「未来 paper revision」等表述请理解为「第 5 章 §N.4 起草/收尾后」。
>
> 📍 **节号与版本对照（2026-07-28 全仓指针体检补注）**：本文头注写于 2026-06-02，其中 `§N.k` 是当时 8 节方案的记法、`rev.4` 是当时的 spec 版本。现行章结构为 **10 节**；spec 现行 rev **不在此写死**（写死正是本次体检查出的腐化源），以 [`CLAUDE.md`](../CLAUDE.md) 文档索引表为准。节号对照：**§N.2 → §5.5**（Online RL）、**§N.4 → §5.7**（ReBRAC-Q 主线）、**§N.5 → §5.8**（泛化边界）、**§N.6 → §5.9**（算法对比：FQL + SAC collector）。正文内 `§N.k` 一律照此读，**不逐处改写**。
>
> 文档版本：2026-05-06
> 作用：Online RL 线的**收口**报告。汇总可引用的正式结果、记录踩过的坑、整理仓库中所有 online 线相关产出，并给出 2026-05-06 战略下调后的下一步建议。
> 战略状态：thesis-grade 47-run 矩阵已撤销（详见 §3.1）；本线只保留 (a) 环境可行性 sanity（A0）、(b) Offline RL chapter 的 SAC collector 数据源（spec 已就位 = 第 5 章 §N.2 / §N.6 per 2026-06-02 LOCKED）两个角色。

---

## 目录

1. [可用于论文写作的正式结果](#1-可用于论文写作的正式结果)
2. [实验过程中踩过的坑](#2-实验过程中踩过的坑)
3. [仓库 online 线文件索引](#3-仓库-online-线文件索引)
4. [下一步计划与专业建议](#4-下一步计划与专业建议)

---

## 1. 可用于论文写作的正式结果

### 1.0 引用规则（最重要）

| 是否可作论文性能基线 | 实验 | 理由 |
|---|---|---|
| ✅ 可直接引用 | **A0 sensor screen**（[`experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/`](../experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/)） | thesis-grade：3 seed × 600k × 完整对照 cell × 收敛非平凡结论 |
| ⚠️ 仅作 reward preset 选择的方法学依据 | archived 双 sweep（[`experiments/objective_ablation_v1/`](../experiments/objective_ablation_v1/) + [`experiments/efficiency_gain_sweep_v1/`](../experiments/efficiency_gain_sweep_v1/)） | 200k 预算下 final success ≤ 6.67%，绝对数字不能引用；但其 preset 选择结论（`efficiency_v2 = energy=0, safety=0.25`）站得住，A0 作为后验证据 |
| ❌ 不可引用 | Sprint 0 preflight 5 run | 单 seed、smoke 性质、含已知 bug；价值在协议级发现而非性能数字 |

### 1.1 A0 — Sensor envelope on cross_u10（唯一 thesis-grade 多 seed 结果）

**实验设置**

| 项目 | 设置 |
|---|---|
| benchmark | `single_u10_cross_tgt15` |
| flow | `wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` |
| task geometry | `cross_stream` |
| target speed | `1.5 m/s` |
| 算法 | vanilla SAC + 4-step observation history |
| 对比 cell | `s0_k4`, `s1_k4`, `s2_k4` |
| objective | `efficiency_v2` 主线 + `arrival_v1` 诊断 |
| 训练步数 | `600 000` |
| `random_steps` / `update_after` | `5000` / `5000` |
| `eval_every` / `eval_episodes` | `10000` / `30` |
| `num_envs` | `6` |
| seeds | `46, 47, 50` |

**结果（success rate，3-seed mean ± std）**

| objective | s1_k4 | s0_k4 | s2_k4 |
|---|---|---|---|
| `efficiency_v2` | **0.967 ± 0.027** | 0.789 ± 0.211 | 0.856 ± 0.204 |
| `arrival_v1` | **1.000 ± 0.000** | 0.967 ± 0.027 | 0.711 ± 0.262 |

**结果（path efficiency，3-seed mean ± std）**

| objective | s1_k4 | s0_k4 | s2_k4 |
|---|---|---|---|
| `efficiency_v2` | **0.854 ± 0.024** | 0.801 ± 0.057 | 0.813 ± 0.100 |
| `arrival_v1` | **0.875 ± 0.013** | 0.853 ± 0.037 | 0.708 ± 0.133 |

**可用于论文的论点**

1. 在易学的 `cross_stream` benchmark 上，三种 sensor layout 都具备基础可学性（best per-cell success ≥ 70%），**没有任何 layout 应被先验地排除**。
2. `s1_k4`（DVL + 短程 ADCP）方差最低、均值最高，是 cross_u10 上最稳的协议候选。
3. `s2_k4`（DVL + 长程 ADCP + lateral）在 efficiency_v2 下与 s1 接近，但 seed 方差大；**这是 sensor 信息更丰富不必然更稳定的实证**——更多 channel 会让训练更难收敛。
4. `s0_k4`（DVL only，deployment-realistic）在 cross_u10 上**几乎能与 s1 持平**（arrival_v1 0.967 vs 1.000）；**这是 deployable-only sensor 在简单 wake 场景里足够用的实证**。
5. `efficiency_v2` 与 `arrival_v1` 在排序上一致（s1 ≥ s0 ≥ s2），且 `efficiency_v2` 的 success-conditioned path efficiency 不显著低于 `arrival_v1`——说明在简单场景里 weak safety shaping 没有破坏 arrival 学习信号。

**已知边界条件（论文要明示）**

- 这是 cross_u10 一个 difficulty cell 的结果。**u15_upstream 上 vanilla SAC 不能仅靠 600k step 收敛**（详见 §2.3）。
- 600k step、3 seed、num_envs=6 是 A0 的 protocol。任何后续工作如果要和 A0 数字直接对比，必须复用同一 protocol。
- 数据来源：[`experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/efficiency_v2/summary/ablation_summary.csv`](../experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/efficiency_v2/summary/ablation_summary.csv) 与同目录 `arrival_v1/ablation_summary.csv`。

### 1.2 Reward preset 选择的方法学依据（archived 报告）

[`results/archived/rl_navigation_experiment_report.md`](../results/archived/rl_navigation_experiment_report.md)（2026-04-14）记录了 **27 个 200k-step run** 的 reward preset 选择 sweep，结论用于支撑 A0 与后续所有 online 实验默认采用 `efficiency_v2`：

**Round 1（[`objective_ablation_v1`](../experiments/objective_ablation_v1/)，6 run）**：u15_upstream + arrival_v1 vs efficiency_v1，3 seed × 200k：
- arrival_v1：5.6% final success
- efficiency_v1：1.1% final success
- → 否决 efficiency_v1（`energy=5e-4, safety=2.0` 太强，压制探索）

**Round 2（[`efficiency_gain_sweep_v1`](../experiments/efficiency_gain_sweep_v1/)，21 run）**：u15_upstream + 7 gain × 3 seed × 200k：

| gain (energy, safety) | final success |
|---|---:|
| `(0, 0)` | 0.0% |
| `(0, 0.25)` | **6.67%** |
| `(0, 0.5)` | 2.22% |
| `(0, 1.0)` | 0.0% |
| `(1e-4, 0.5)` | 0.0% |
| `(2e-4, 0.5)` | 0.0% |
| `(5e-4, 2.0) = original efficiency_v1` | 0.0% |

- → `efficiency_v2 = (energy=0.0, safety=0.25)` 作为推荐 preset。

**论文使用方式**：可以在 method 节或 appendix 报告这两组 sweep 的"preset selection"角色，但**不要把 200k 的 success 数字当作性能基线**——在 A0 上同样 `efficiency_v2` 配置 + 600k 拿到 96.7%，说明 200k 不足以判断 reward 间真实优劣，只能判断"哪个 reward 让前 200k 更不容易学坏"。

### 1.3 工程协议落字（可写入论文方法节 / 附录）

来自 Sprint 0 preflight 的协议级发现，**可作为方法节的协议描述写进论文**：

1. **FLOW_PATH 由 BENCHMARK_KEY 决定的 invariant**（§2.2 介绍的 bug 修复后强制要求）：

   | benchmark prefix | flow file |
   |---|---|
   | `single_u10_*` | `wake_v8_U1p00_Re150_*` |
   | `single_u15_*` | `wake_v8_U1p50_Re250_*` |
   | `tandem_u15_*` | `wake_tandem_G35_v8_U1p50_Re250_*` |
   | `sbs_u15_*` | `wake_sbs_G35_v8_U1p50_Re250_*` |

2. **`num_envs` 是 protocol 的一部分**：A0 用 `num_envs=6`，preflight P0b 实测 6 → 12 加速 1.45×（IPC 是瓶颈，sublinear），但**任何跨阶段对比必须 num_envs 一致**。

3. **Best vs final checkpoint 双视角评估**：archived 报告 §4.3 提出，A0 + preflight 全部继承——任何 reward 设计变化都同时报 final 和 best periodic eval，避免 reward hacking 假阳性。

---

## 2. 实验过程中踩过的坑

按"症状 → 根因 → 修复 → 教训"的格式整理。每条都标了发现位置，便于以后追溯。

### 2.1 200k 预算太短，假性否决方法

- **症状**：archived 双 sweep 全员 final success ≤ 6.67%，看似所有 reward 都失败。
- **根因**：u15_upstream 难度下，vanilla SAC 在 200k 步还在 cold-start 阶段；在 A0 (cross_u10 + efficiency_v2 + 600k) 上同一 reward preset 拿到 96.7% 才证明问题不在 reward。
- **修复**：systematic plan 把默认预算从 100k/300k 一次性提到 600k；A0 的 18 run 全部 600k 完成。
- **教训**：在新 benchmark 上做 reward / sensor / 算法 ablation **之前**，必须先用一个已知能学好的对照（如 cross_u10）确认预算。**不能用未学会的曲线下结论**。
- **位置**：[`results/archived/rl_navigation_experiment_report.md`](../results/archived/rl_navigation_experiment_report.md) §2-§3、[`docs/systematic_improved_sac_experiment_report.md`](systematic_improved_sac_experiment_report.md) §2.5 结论 1。

### 2.2 训练 flow 与 eval flow 不一致（协议级 bug）

- **症状**：preflight P1 v1（s1, vanilla, 1M step）在 200k-600k 段看到 success rate 0.7-1.0，看似已收敛；落字到决策表里时被"曲线震荡 0.5-1.0"形容。
- **根因**：训练 `--flow` 用了 `wake_v8_U1p00_Re150_*`（U=1.0 m/s）但 eval manifest `single_u15_upstream_tgt15.json` 内置 flow_path 指向 `wake_v8_U1p50_Re250_*`（U=1.5 m/s）。Agent 在简单流场训练，被丢到困难流场评估，"高 success" 是 train/test mismatch 假阳性。
- **修复**：[`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) §10.0 写入 invariant —— FLOW_PATH 必须由 BENCHMARK_KEY 决定，notebook 不可独立设置；P1 v2_flowfix 用正确 flow 重跑后，best success 仅 0.60 @ 130k 后 collapse。
- **教训**：(a) eval manifest 与训练 flow **必须有一致性 check**（写入 `train_sac.py` 启动校验）；(b) helper function 自动从 manifest 读 flow_path，杜绝同类 bug 再现。
- **位置**：[`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) §10.0 + §10.3；[`experiments/online_thesis_v1/preflight/p1_budget_calibration_v2_flowfix/`](../experiments/online_thesis_v1/preflight/p1_budget_calibration_v2_flowfix/)。

### 2.3 efficiency_v2 在 u15_upstream 上 reward hacking

- **症状**：preflight P1 v2_flowfix（1M step）`eval_return` 后期持续上升（-260 → -80），但 `eval_success_rate` 从 0.60 @ 130k 一路 collapse 到 0；final eval termination_counts = 30 out_of_bounds。
- **根因**：`efficiency_v2` 的 `step_penalty=-1.0` 在 240s episode 上累计到 -480，远大于 `success_reward=+100` 与 `failure_penalty=-20`。Agent 在 600k+ 后学会"快速出界"——episode 短则时间惩罚少，return 反而更高。Reward 缺少 terminal dominance 不变量。
- **诊断指标**：
  - `corr(return, success) = -0.63`（return 升 ⇒ success 降）
  - `corr(return, time) = -0.96`（return 升 ⇒ episode 变短）
  - `corr(return, progress) = -0.51`（return 升 ⇒ 距离目标更远）
- **修复**：[`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) v1 → v4 提出 `arrival_v2`（8 参数，含 terminal dominance、early-failure penalty、final-distance penalty、no-fast 默认），并加 unit test 强制 `fast_OOB << any_timeout` 等不变量。**实施搁置**（详见 §4）。
- **教训**：(a) reward 设计**必须**有 terminal ordering 的 unit test；(b) eval pipeline **必须**输出 termination_counts 与 success-conditioned 指标，而不只是均值；(c) "return 上升"不等于"训练成功"，必须配 success rate 看。
- **位置**：[`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) §2-§3。

### 2.4 num_envs > 1 时数据流的几个潜在 bug（已修）

- **症状**：早期版本里 `--num-envs 6` 与 `--num-envs 1` 行为有微妙差异；privileged_obs 在 RLPD 路径下 key 不一致导致 `DualBufferSampler` 崩溃；single env reset 后 obs 可能被终止态覆盖。
- **根因**：(a) `train_sac.py` 中 `num_envs` 标志在某些分支没真正传到 vector env；(b) `(privileged_obs, next_privileged_obs)` 双键存储未实现，把"下一时刻 priv"错当"当前时刻 priv"用；(c) reset 与 done 处理顺序不当。
- **修复**：commit [`7e1d27a`](../auv_nav/replay.py)（"已修复会影响实验有效性的几处问题..."）。
- **教训**：替换 SAC 内部数据流的改动必须配 unit test 验证 step-by-step 一致性（priv key 对齐、reset 后 obs 不被覆盖、num_envs=1 与 >1 行为等价）。**这次修复同时让 offline 线 RLPD/AsymCritic ablation 受益**——是 online 线给 offline 线留下的最大工程遗产之一。
- **位置**：commit message [`7e1d27a`](../auv_nav/replay.py)。

### 2.5 AsyncVectorEnv IPC 是 wallclock 主要瓶颈（不是 NN）

- **症状**：L4 上 600k step run ≈ 1.5h，远高于 NN 规模（256-hidden 3-layer MLP × 4）理论 GPU 忙时 ~50min。
- **诊断**：preflight P0a（cProfile）显示训练 wallclock 分布：
  - ~55% AsyncVectorEnv IPC（`posix.read` from `connection.py`，pickle obs/info 跨 pipe）
  - ~30% env 物理（`flow.bilinear` + `vehicle.dynamics` + `autopilot.sample`）
  - ~6% PyTorch（NN forward/backward）
- **修复**：preflight P0b 实测 num_envs=6 → 12 wallclock 5.61 → 3.88 min（**1.45× 加速**），切到 12。
- **教训**：(a) **优先级反转**——`AsyncVectorEnv(shared_memory=True)` 上限收益 ~50% wallclock，远高于 Numba JIT vehicle.py 的 ~30% 上限收益；(b) NN 不是瓶颈说明加大 batch / hidden_dim 不会显著拖慢 wallclock，可以放心加。
- **位置**：[`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) §10.1-§10.2、§10.5。
- **Update 2026-05-17**：[`docs/wallclock_profile_2026_05_17.md`](wallclock_profile_2026_05_17.md) §4.5 用 5-bucket `perf_counter` + CUDA sync profile 在 L4 + thesis baseline (n=6, UTD=4) 上重测，**修正归因**：实际瓶颈是 SAC update (64-77% wallclock)，不是 IPC；cProfile 看到的 `posix.read` ~55% 反映的是 worker 等待 main 而非 IPC pickle 本身。本节"`num_envs 6→12` 拿加速"的实操修复仍然有效，但教训 (a) 中 "~50% wallclock 上限收益" 乐观了，实测在 effective UTD 不变前提下加速上限只有 1.17×；教训 (b) 中 "加大 hidden_dim 不会拖慢" 需要重新 profile 才能成立。归因解释以新归档为准。

### 2.6 跨阶段 num_envs 不一致破坏可比性

- **症状**：A0 用 num_envs=6，preflight P0b 切到 12；后续若用 12 跑 §2 sensor envelope，与 A0 不能直接比。
- **根因**：num_envs 影响每 env step 收集多少 transition、replay 分布、UTD 实际比、wallclock 与 sample efficiency 之间的换算关系。
- **修复**：online_rl_thesis_plan §1.2 明文写入 invariant —— `num_envs` 是实验 protocol 的一部分，**跨阶段不可混用**；A0 在 num_envs=6 上锁定后，后续若要切到 12 必须先在 cross_u10 上重跑一组以建立 6 ↔ 12 的协议桥梁。
- **教训**：训练加速选项（num_envs、shared_memory、UTD）一旦改动，等同于换 protocol；ablation 矩阵开始之前必须冻结一切训练加速选项。
- **位置**：[`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) §1.2 + §10.7。

### 2.7 arrival_v2 reward 设计需要 4 轮 review 才稳

- **症状**：[`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) v1 → v4 4 轮迭代，每轮发现新的不严谨：
  - v1：`w_safety=1-5` 是经验猜测，未与真实 P1 数据校准
  - v2：dominance flip 阈值估错（4.5 → 实际 3.69），late_OOB 漏算 R_early_failure
  - v3：fast_OOB 不是绝对最差（mid_OOB 才是），potential-based shaping 在 γ=0.995 下不严格 invariant
  - v4：fast_success 步数不一致，R_fast_success 默认值需要降级为 ablation
- **教训**：reward 设计是数值工程，**任何参数取值必须由 unit test fixture 校准**，不能拍脑袋。terminal ordering、no-suicide-attractor、no-unsafe-shortcut 等不变量必须显式写测试。
- **位置**：[`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) §10-§11。

### 2.8 文档膨胀与文件名混淆

- **症状**：online 线先后产出 `systematic_improved_sac_experiment_plan.md` / `_report.md`、`online_rl_thesis_plan.md`、`online_sac_reward_redesign.md`、`SAC_improvements_survey.md`、`results/archived/rl_navigation_experiment_report.md`；同名 preset (`arrival_v2` doc 设计 vs `arrival_v2_simple` 实际入码) 容易混淆。
- **根因**：每个阶段都开新文件而非 in-place 更新；deprecate 标记加得不够前置。
- **教训**：(a) 累积式报告（如 thesis_report.md）应该早建空骨架；(b) deprecate 当天就在文件首部加红色 banner（已经做到）；(c) 同名概念在不同文件里指代不同内容时，必须显式写"X 在本文件指..."。
- **位置**：本文件 §3 整理后即可。

---

## 3. 仓库 online 线文件索引

按"如果将来要找 X，就看 Y"的角度组织。所有路径相对仓库根。

### 3.1 顶层导航：当前应该读什么

| 想找... | 看这里 |
|---|---|
| Online 线整体状态、可引用的结果、坑、下一步 | **本文件**（[`docs/online_rl_line_summary.md`](online_rl_line_summary.md)） |
| 唯一可引用的 thesis-grade 多 seed 结果 | [`experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/`](../experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/)（A0） |
| `efficiency_v2` reward preset 的来源依据 | [`results/archived/rl_navigation_experiment_report.md`](../results/archived/rl_navigation_experiment_report.md) §3.7、§4 |
| SAC 算法改进的设计 context | [`docs/SAC_improvements_survey.md`](SAC_improvements_survey.md) |
| Env / sensor / reward / benchmark 规格 | [`docs/environment_design.md`](environment_design.md) |
| AsymCritic + privileged_obs 精确语义 | [`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) §2（即使 plan deprecated，§2 的语义定义仍然 active） |
| Sprint 0 preflight 的协议级发现 | [`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) §10 |
| 为什么 thesis 矩阵被撤销 | 本文件 §4.1 |

### 3.2 文档（docs/）

| 文件 | 状态 | 角色 | 何时读 |
|---|---|---|---|
| [`online_rl_line_summary.md`](online_rl_line_summary.md) | **active（本文件）** | online 线收口报告 | 最先读 |
| [`online_rl_thesis_plan.md`](online_rl_thesis_plan.md) | **deprecated 2026-05-06**（本次更新落字） | 47-run thesis 矩阵；§2 priv_obs 语义定义、§10 preflight 决策仍 active | 找 priv_obs 精确定义 / preflight 数据 |
| [`online_sac_reward_redesign.md`](online_sac_reward_redesign.md) | **搁置（v4，2026-04-27）** | `arrival_v2` reward 8 参数设计 + 不变量测试 spec | 未来想恢复 reward 设计时 |
| [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) | **active archive（2026-05-23，§7.9.7 universal-floor closure）** | arrival_v2 prototype 实测档（19 组 experiments，§7 / §7.6 / §7.7 / §7.8 / §7.9 cross-seed closure + §7.9.7 manifest universal-floor finding） | paper revision / rebuttal cite arrival_v2 实测结果时 |
| [`arrival_v2_p0_variance_reduction_design.md`](arrival_v2_p0_variance_reduction_design.md) | **active design reference (DEMOTED-TO-FUTURE-WORK / POLISH-ONLY，2026-05-19)** | SAC variance reduction (DroQ / N-Step / REDQ) 候选矩阵 + 5-tier verdict schema | offline 线 variance reduction 复用 / 未来 paper revision 需 DroQ 类轴时 |
| [`arrival_v2_sac_collector_design.md`](arrival_v2_sac_collector_design.md) | **active — rev.2 用户已声明启动（2026-05-24）** | 19 ckpt 清单 + 按 offline 主线 s0_k4+arrival_v2 双约束筛选 8 ckpt + D4RL 路径 1 (cross_u15) / 路径 2 (cross_u10) 双轨 spec + sensor floor 方法学论证 | 启动 SAC collector 前 / 写 adapter 前 / FQL 再验证 |
| [`systematic_improved_sac_experiment_plan.md`](systematic_improved_sac_experiment_plan.md) | DEPRECATED 2026-04-26 | 旧版主计划 | 仅历史回溯 |
| [`systematic_improved_sac_experiment_report.md`](systematic_improved_sac_experiment_report.md) | DEPRECATED 2026-04-26 | A0 阶段实测记录（数据本身仍有效） | A0 数据来源（也可以直接看 ablation_summary.md） |
| [`SAC_improvements_survey.md`](SAC_improvements_survey.md) | active reference | 2020-2026 SAC 改进算法综述 | 写论文 related work / 算法选型 |
| [`environment_design.md`](environment_design.md) | active reference | Env / sensor / reward / benchmark 规格 | 写论文 method 节 |
| [`rlpd_design.md`](rlpd_design.md) | active reference | RLPD 设计档（offline-to-online） | 未来恢复 RLPD 路线时 |
| [`auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) | bridge | MBRL 计划，引用了 online 线产出的 AsymCritic / EquivalentCurrentModel | 跨线复用 infra 时 |
| [`offline_mbrl_plan/AUV_REBRAC_NeuralODE_OfflineRL_v2_report.md`](offline_mbrl_plan/AUV_REBRAC_NeuralODE_OfflineRL_v2_report.md) §7.3 | bridge | asymmetric critic 复用论述 | 同上 |

**`docs/online_rl_thesis_report.md` 不存在**——thesis plan §8.2 约定的累积报告未实例化（thesis 撤销后也不需要）。

### 3.3 实验数据（experiments/ + results/）

| 目录 | 体量 | 状态 | 角色 |
|---|---:|---|---|
| [`experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/`](../experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/) | 18 run | **完整可用** | A0 sensor screen，唯一 thesis-grade 多 seed 结果 |
| `experiments/arrival_v2_prototype/` (gitignored)              | 18 run | **prototype 实测档（2026-05-07 → 2026-05-19）** | §7 strict control / §7.6 s0 envelope / §7.7 AsymCritic / §7.8 k=8 / §7.9 multi-seed × k=12；gate JSONs 在 `s0_cross_k{8,12}_seed{0,7,42}_summary/` 下 |
| [`experiments/objective_ablation_v1/`](../experiments/objective_ablation_v1/) | 6 run | preset 选择依据（archived 报告分析） | u15_upstream + 200k arrival vs efficiency_v1 否决 |
| [`experiments/efficiency_gain_sweep_v1/`](../experiments/efficiency_gain_sweep_v1/) | 21 run | preset 选择依据（archived 报告分析） | u15_upstream + 200k 7-gain sweep，导出 efficiency_v2 |
| [`experiments/online_thesis_v1/preflight/`](../experiments/online_thesis_v1/preflight/) | 5 run | 协议级 smoke，不可作性能引用 | P0 profiling、num_envs benchmark、P1 budget calibration（v1 INVALID / v2 reveal hacking）、P2 AsymCritic smoke |
| [`experiments/_codex_rel_results/`](../experiments/_codex_rel_results/) | 12-step | CI smoke | 无研究价值 |
| [`experiments/_stage_wrapper_smoke/`](../experiments/_stage_wrapper_smoke/) | 1 stub | CI smoke | 无研究价值 |
| [`results/archived/rl_navigation_experiment_report.md`](../results/archived/rl_navigation_experiment_report.md) | — | archived 但仍为 preset 选择源头 | objective_ablation_v1 + efficiency_gain_sweep_v1 的正式分析报告 |
| [`results/archived/README.md`](../results/archived/README.md) | — | archived 索引 | 上面那份的 key takeaways |

### 3.4 Notebooks（notebooks/sac_thesis_*）

| Notebook | 状态 | 内容 |
|---|---|---|
| [`sac_thesis_s0_preflight.ipynb`](../notebooks/sac_thesis_s0_preflight.ipynb) | 旧 scaffold | Sprint 0 P1+P2 早期框架 |
| [`sac_thesis_s0_preflight_v2_completed.ipynb`](../notebooks/sac_thesis_s0_preflight_v2_completed.ipynb) | ✅ 完成 | Sprint 0 P1（含 v2_flowfix）+ P2 |
| [`sac_thesis_s0b_profiling.ipynb`](../notebooks/sac_thesis_s0b_profiling.ipynb) | scaffold | P0a + P0b 早期框架 |
| [`sac_thesis_s0b_profiling_completed.ipynb`](../notebooks/sac_thesis_s0b_profiling_completed.ipynb) | ✅ 完成 | P0a cProfile + P0b num_envs benchmark |
| [`_deprecated_sac_thesis_s2_sensor_envelope.ipynb`](../notebooks/_deprecated_sac_thesis_s2_sensor_envelope.ipynb) | **deprecated 2026-05-06** | thesis Sprint 1 §2 sensor envelope，未启动；保留作历史 scaffold |
| `sac_thesis_s3 / s4 / s5 / s6 / s7` | 从未创建 | thesis plan 中后续节的 notebook |

### 3.5 代码组件（auv_nav/）

Online 线开发出的、当前**仍 active 且 offline 线复用**的核心组件：

| 文件 / 类 | 起源 | 当前角色 |
|---|---|---|
| [`auv_nav/sac.py`](../auv_nav/sac.py) `AsymmetricQNetwork` | online thesis §3 设计 | offline RLPD ablation（也供 MBRL plan 参考） |
| [`auv_nav/replay.py`](../auv_nav/replay.py) `DualBufferSampler` | RLPD design | offline 线 broad validation 第三轴 |
| [`auv_nav/replay.py`](../auv_nav/replay.py) `(privileged_obs / next_privileged_obs)` 双键 replay | online preflight bug 修复（commit `7e1d27a`） | offline 线 priv ablation 必需 |
| [`auv_nav/autopilot.py`](../auv_nav/autopilot.py) `EquivalentCurrentModel` | online 线 hull-integral privileged flow 计算 | online + offline 共享 |
| [`auv_nav/reward.py`](../auv_nav/reward.py) `efficiency_v2` preset | archived 报告 + A0 验证 | online + offline 共享主线 |
| [`auv_nav/reward.py`](../auv_nav/reward.py) `arrival_v2_simple` preset (commit [`bd37412`](../auv_nav/reward.py)) | offline reward ablation（**与 reward_redesign doc 中的 v4 arrival_v2 不是同一物**） | offline 线 |
| `--num-envs > 1` 真正生效 + reset 一致性 | online preflight bug 修复（commit `7e1d27a`） | online + offline 共享 |

**`arrival_v2_simple` 与 reward redesign doc 的 `arrival_v2` 区别**：
- `arrival_v2_simple` 仅复用 `RewardModelConfig` 现有字段做 preset 重组，没有新加 8 参数 / terminal dominance test，主要服务 offline reward ablation。
- doc 里的 `arrival_v2` 是 8 参数完整版（含 R_early_failure / R_final_distance / dominance unit test），**未实施**。

### 3.6 Stage 脚本（scripts/）

| 文件 | 状态 |
|---|---|
| [`scripts/run_protocol_stage_common.sh`](../scripts/run_protocol_stage_common.sh) | active，被所有 online stage notebook 通过 env-var override 调用 |
| [`scripts/run_stage_a0_layout_screen.sh`](../scripts/run_stage_a0_layout_screen.sh) | active，A0 launcher |
| [`scripts/summarize_stage_a0_layout_screen.sh`](../scripts/summarize_stage_a0_layout_screen.sh) | active，A0 summarizer |
| [`scripts/run_stage_a1_layout_main.sh`](../scripts/run_stage_a1_layout_main.sh) | DEPRECATED（A1 已撤销） |
| [`scripts/summarize_stage_a1_layout_main.sh`](../scripts/summarize_stage_a1_layout_main.sh) | DEPRECATED |

---

## 4. 下一步计划与专业建议

### 4.1 战略下调的归因（为什么 thesis 矩阵不重启）

`docs/online_rl_thesis_plan.md` 47-run thesis 矩阵于 2026-05-06 下调，原因有三：

1. **Reward 主线不稳**：preflight P1 v2_flowfix 已经证明 vanilla SAC + efficiency_v2 在 u15_upstream 上 1M step 仍 collapse 到 success=0；要继续就必须先实施 reward redesign doc 里的 8 参数 `arrival_v2` + 全套 unit test，工作量约 2 工作日 + 多轮 Colab 重跑（详见 reward_redesign §8.1）。
2. **Offline 线已具备 publishable 结果**：ReBRAC 主线 rev.8 closed、paper drafting Phase 5 完成；论文叙事已经收敛到"deployable s0 + ReBRAC 持平 privileged-critic 协议"，**不需要 online 线再产出独立章节**。
3. **机会成本**：thesis 矩阵 47 run × 1.5h × 4-7 个 Colab session 与 ReBRAC paper revision / broad validation S2 P1（16 run × ~30min × 1 session）之间，后者距离投稿更近。

→ 因此 **online 线不应再独立产出论文章节**，只承担两个角色：(a) A0 已完成的环境可行性 sanity；(b) 待 spec 的 SAC collector 数据源（§4.3）。

### 4.2 即刻清理动作（约 1 小时）

**状态：本节全部动作已于 2026-05-06 执行完毕（commit 见 git log）。** 表中保留作为历史决策记录。

| 优先级 | 动作 | 状态 | 理由 |
|---|---|---|---|
| P0 | 在 [`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) 头部加 `**DEPRECATED 2026-05-06**` banner，指向本文件 | ✅ 已完成 | 与 systematic_*.md 一致 |
| P0 | 在 [`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) 头部加 `**SHELVED 2026-05-06**` banner，指向本文件 §4.4 | ✅ 已完成 | 避免未来误以为是 active spec |
| P0 | 删除 `package-lock.json` | ✅ 已删除（原本未被 git track） | 与本仓库 Python 项目无关的污染文件 |
| P1 | 把 s2 sensor envelope notebook 重命名为 [`notebooks/_deprecated_sac_thesis_s2_sensor_envelope.ipynb`](../notebooks/_deprecated_sac_thesis_s2_sensor_envelope.ipynb) | ✅ 已完成 | 防误开 |
| P1 | 在 [`scripts/run_stage_a1_layout_main.sh`](../scripts/run_stage_a1_layout_main.sh) 与 [`summarize_stage_a1_layout_main.sh`](../scripts/summarize_stage_a1_layout_main.sh) 头部加 deprecate 注释 | ✅ 已完成 | 防误跑 |
| P2 | [`experiments/_codex_rel_results/`](../experiments/_codex_rel_results/) 与 [`experiments/_stage_wrapper_smoke/`](../experiments/_stage_wrapper_smoke/) | ✅ 已自动满足 | `experiments/` 整体已在 .gitignore（第 55 行），smoke 目录天然不被 track |

附带同步：[`CLAUDE.md`](../CLAUDE.md) §Notebooks 表已更新（s2 重命名 + s3-s7 标记为 cancelled）。

### 4.3 SAC collector 数据源 spec（~~Online 线下一个真正要做的事~~ ✅ **已 CLOSED 2026-05-26**）

> 📍 **2026-08-09 复核补注**：本节标题与下文的「待办 / 等 offline 线确认再 freeze」措辞停在 2026-05-24（rev.2）。SAC collector 已于 **2026-05-26 CLOSED**（rev.3 Plan A + Sprint 1+2 + m_multi_mix supplement，cross-source 矩阵闭环），收口详情见 [`offline_rl_line_summary.md`](offline_rl_line_summary.md) §4.3、实施 spec 见 [`arrival_v2_sac_collector_design.md`](arrival_v2_sac_collector_design.md)（active rev.3）。**本节推荐 spec 表已被 design 文档取代，只作历史决策记录读。**

Memory 提到 "D4RL-style SAC collector 拟定为 broad validation 平行第四轴" 但 spec 未定。**这是 online 线唯一仍有意义的待办项**——为 offline RL chapter 提供一个 RL-trained behavior policy（区别于 worldcomp / crosscomp / privileged 这三个 hand-engineered baseline）。

> **2026-05-24 update（rev.2）**：用户已声明启动 SAC collector（动机：D4RL 范式对齐 + FQL 再验证）。完整 spec 已落到 [`docs/arrival_v2_sac_collector_design.md`](arrival_v2_sac_collector_design.md) rev.2：与 offline 主线（v2 + FQL P2）协议严格对齐 = **s0 + k=4 + arrival_v2** → 19 ckpt 中筛 8 ckpt 可用；推荐 **路径 1 (cross_u15 cell × 4 ckpt, D4RL random/medium-replay tier) + 路径 2 (cross_u10 cell × seed=46 expert + 补 seed=47/50)** 双轨；总预算 ~13h L4。本节保留作为旧 A0-ckpt 协议的兼容简述；rev.2 实施细节去看 collector design 文档。

#### 推荐 spec（最小集合，等 offline 线确认再 freeze）

| 项目 | 推荐取值 | 理由 |
|---|---|---|
| benchmark | `single_u10_cross_tgt15` | A0 已证明 vanilla SAC 能学到 ~80-97% success；**避开 u15_upstream 的 reward hacking 区** |
| sensor | `s0_k4` | offline 主线 deployable-only 协议已对齐到 s0 |
| algorithm | vanilla SAC（无 LayerNorm / 无 UTD / 无 AsymCritic） | 与 A0 vanilla baseline 一致；不引入 thesis 章未验证的算法 |
| reward | `efficiency_v2` | A0 已验证；与 offline worldcomp/crosscomp 数据集一致 |
| seeds | `46, 47, 50` | 复用 A0 seed 池 |
| 训练步数 | 600k | 与 A0 一致 |
| 数据收集 | 每 seed 跑 deterministic eval 收 1000 episode | offline 线现有 collector 协议（参考 [`scripts/collect_offline_data.py`](../scripts/collect_offline_data.py)） |
| 输出 | `offline_data/sac_s0_h4_efficiency_v2_re150_u10cross_ep1000_seed{46,47,50}/` | 命名与现有 worldcomp_*/crosscomp_* 对齐 |
| 是否记录 privileged_obs | **是**（即使主线协议不用，offline RLPD ablation 可能用） | 收集时几乎零成本；删数据比加数据贵 |
| 总预算 | 3 seed × 600k 训练（**已在 A0 跑过**，**直接用 A0 ckpt 收数据**即可，不重训）+ 3 × 1000 episode collect ≈ 0.5h | A0 ckpt 路径见 §1.1 数据来源 |

→ **真正的工作量是写一个把 A0 的 SAC ckpt 当作 baseline policy 注入 collect_offline_data.py 的 adapter**（约 1-2 小时编码），不是重训 SAC。

**触发条件**：等 offline 线 broad validation S2 P1 跑完、确认 SAC 数据源是否真的需要再启动。如果 ReBRAC 在 worldcomp/crosscomp 上的结果已经稳，这一项也可以**完全砍掉**。

### 4.4 关于 `arrival_v2` reward 的最终处置（2026-05-23 update — universal-floor closure）

**Status update vs 2026-05-06 撤销决定**：本节原写 "v4 已收敛但未实施"，**事实上 arrival_v2 v4 (8 参数完整版) 已在 prototype 分支 `codex-arrival-v2-prototype` commit `813096e`（2026-05-07）落地进 `auv_nav/reward.py`**，并在该分支跑了 19 组实验，2026-05-23 收口在 §7.9 cross-seed thesis-grade closure + §7.9.7 manifest universal-floor finding。`auv_nav/reward.py` 里因此**同时存在两套 arrival_v2 preset**（与 §3.5 的 `arrival_v2_simple` 注解一致，仍为同一文件中两个并存 preset）：

- `arrival_v2_simple`（commit `bd37412`）— offline reward ablation 用，本节原始关注对象
- `arrival_v2`（commit `813096e`）— v4 8 参数完整版 + dominance unit test + early-failure penalty + final-distance penalty + no-fast 默认，**prototype 分支实测产出 thesis-grade finding（详见 [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md)）**

**Prototype 实测主线（不更改 thesis 矩阵撤销决定）**：

- **§7 4-way strict-control × s1_k4**（topology × geometry，2026-05-09）：4/4 cell 全 5/5 gate PASS
- **§7.6 s0 sensor envelope**（2026-05-13）：上游 3 cell（tandem / sbs / single_upstream）s0 全 PASS（与 s1 等价）；`single_cross_s0` catastrophic FAIL（final=0.100, OOB=0.667）— 80pp s0–s1 gap 在 production-difficulty regime 显化
- **§7.7 AsymCritic 单变量 ablation**（pure B 路径，2-seed × 2-algo paired hardened negative finding）：critic-side info upgrade 不闭合 80pp gap；瓶颈从 critic estimation 重定位到 actor-side information access
- **§7.8 history k=4→8 actor-side ablation**（seed=42 anchor）：单变量 actor-side temporal info upgrade 一次性闭合 80pp gap，与 s1_k4 上界等价且 sample-efficiency 更好
- **§7.9 multi-seed × k-monotonicity closure**（2026-05-18 / 2026-05-19）：k=8 cross-seed 不鲁棒（1/3 strict PASS, σ_final=0.181）；k=12 在 2/2 strict 5/5 PASS, σ_final=0.064，含 seed=0 CROSS-SEED-RESCUE — final 0.500→0.900。**seed=0 跨 history 单调相位跃迁** k=4: 0.4 → k=8: 0.5 → k=12: 0.9 **直接证伪 H_seed-stall + H_optimization-noise，确立 H_information-bottleneck wins**
- **§7.9.2'' k=12 third anchor (seed=7) + §7.9.7 manifest universal-floor**（2026-05-23）：实测 final=0.833 / OOB=0.167 / mean39=**0.750 三 seeds 最高** / peak @ **275k 三 seeds 最早**。drill-down 30-ep manifest 揭示 **vanilla SAC s0 在此 manifest 的 inherent ceiling = 27/30 = 0.900**（ep {1208, 1216, 1228} 在 5/5 vanilla runs 全部 OOB → manifest universal floor）。k=12 s42 / s0 恰好 saturate floor；k12_s7 比 floor 多 OOB 2 ep（1 seed=7 cross-history-persistent + 1 k=12 specific near-miss progress=89%）。**3-seed σ_final = 0.038 << thesis target 0.10**；σ_mean39 跨 history 砍 41%。verdict tier 升格为 6-tier `NEAR-PASS-MANIFEST-FLOOR-PINNED`。

**Narrative shift（升格版，2026-05-23）**：deployment-realistic 路径从原 thesis plan 的「升级 sensor 到 s1（多一个空间探头）」改写为「**保持 s0 + 升级 actor 时序访问到 k=12 (~6 s ≈ 涡街周期 30–60%)，在 2/3 seeds 上 saturate manifest-inherent ceiling**」。**新概念产出**：「**manifest universal floor**」——某固定 (benchmark, manifest, sensor, reward) 配置下、跨所有 vanilla SAC runs 都 OOB 的 episode 集合，定义 vanilla SAC 在该 manifold 的物理性极限。这是 prototype 工作产出的**双重方法学发现**（H_information-bottleneck + universal-floor），独立于 thesis 矩阵撤销决定。

**§8 P0 SAC variance reduction**：原 2026-05-19 morning draft 是为压 §7.9.1 k=8 seed=0 stall 设计的 DroQ / N-Step / REDQ 候选矩阵；§7.9.2' k=12 seed=0 CROSS-SEED-RESCUE 已 short-circuit 该 motivation；§7.9.7 universal-floor 进一步加固论证 — **universal floor 是 manifest-inherent，variance reduction 救不了；3-seed σ_final 已自然达标**。P0 **维持 future work / polish only** 定位。design doc 保留作为 paper revision 或 offline 线 variance reduction 复用参考；详见 [`arrival_v2_p0_variance_reduction_design.md`](arrival_v2_p0_variance_reduction_design.md)。

**处置建议（更新版，2026-05-23）**：

- **不要因 prototype 实测 PASS / universal-floor finding 而重启 47-run thesis 矩阵**——撤销决定基于产品判断（offline 线优先 + 海试数据未到），实测结果**不改变这个判断**；只补完了"假设我们重启会发生什么"的方法学证据 + 产出了一个独立的新概念（universal floor）。
- **doc 保留作为完整设计 + 实测档案**：[`online_sac_reward_redesign.md`](online_sac_reward_redesign.md) (设计，已搁置 banner) + [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) (实测，19 组 prototype experiments + §7.9 closure + §7.9.7 universal-floor) + [`arrival_v2_p0_variance_reduction_design.md`](arrival_v2_p0_variance_reduction_design.md) (variance reduction 设计参考)。三份互相 cross-link，未来 paper revision / rebuttal / offline 线 reward ablation 直接 cite。
- **是否值得移植到 offline 线**：与原结论一致，由 offline 线决定。**新增**：若 offline 线确实启动 arrival-first reward ablation，建议**直接复用 prototype 的 commit `813096e` 完整版**（已有 dominance unit test + early-failure penalty + 19 组 prototype 跑过的 gate-stable 实测保证 + universal-floor 概念可复用），而非把 `arrival_v2_simple` 重新升级。
- **Paper rebuttal 可 cite 的语言**：3-seed σ_final=0.038 (< thesis target 0.10); k=12 saturates manifest universal ceiling (27/30 = 0.900) on 2/3 seeds; 3rd seed deviation explainable by 1 seed-bias OOB + 1 near-miss OOB (progress=89%); 详细 thesis-grade 语言 see [`arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) §7.9.7。

### 4.5 论文写作中的 online 线落字位置

[`paper/outline.md`](../paper/outline.md) 当前对 online 线的安排已经合理（review 后确认）：

- **§3.2 Observation: deployable s0** — 复用 online 线定义的 10-D obs space
- **§3.3 Privileged observation (critic-only)** — 复用 [`online_rl_thesis_plan.md`](online_rl_thesis_plan.md) §2 的精确语义
- **§3.4 Reward and termination** — 复用 `efficiency_v2` preset
- 不要在 paper 里写"online thesis chapter"——已撤销。

**可以考虑加进 paper 的 online 线产出**（在 appendix 或 method 节点过）：
- A0 sensor screen 的 1-2 行表格作为"deployable s0 在简单 wake 上是 viable" 的引证。
- archived 报告的 preset 选择 sweep（27 run × 200k）作为 efficiency_v2 选择的方法学依据。
- 工程协议：FLOW_PATH ↔ BENCHMARK_KEY invariant、num_envs 固定、best+final 双 checkpoint 评估。

### 4.6 长期建议：什么情况下重启 online 线

如果未来下列任一条件被满足，可以考虑重启 online 线：

1. **offline 线 ReBRAC 论文投稿/接收后**有余力做扩展；
2. **AUV 海试数据**到位，需要训练在线适应策略；
3. **AsymCritic + privileged hull-integral flow 在 offline 线被审稿人质疑** "为什么不在 online 线也验证"——届时可重启 §3 priv-critic ablation（仅 6 run × s0 × u10/u15）作为 rebuttal 实验，无需重启完整 47-run 矩阵。

---

## 5. 一句话总结

Online 线在 2026-04-14 → 2026-05-06 累计产出 **27 个 preset 选择 sweep run（archived）+ 18 个 A0 多 seed run + 5 个 preflight 协议 smoke run**；2026-05-07 → 2026-05-23 在 prototype 分支 `codex-arrival-v2-prototype` 增加 **19 组 arrival_v2 prototype experiments**（4 组 §7 strict control + 4 组 §7.6 s0 envelope + 3 组 §7.7 AsymCritic ablation + 1 组 §7.8 k=8 anchor + 2 组 §7.9.1 k=8 multi-seed + 2 组 §7.9.2 k=12 monotonicity + **1 组 §7.9.2'' k=12 third anchor (2026-05-23)** + 2 组 reference baselines），收口在 **§7.9 cross-seed thesis-grade closure + §7.9.7 manifest universal-floor finding**（k=12 saturates manifest universal ceiling on 2/3 seeds + 3rd seed near-PASS-FLOOR-PINNED; 3-seed σ_final=0.038 << target 0.10）。唯一可作论文性能基线的仍是 A0（**2026-08-09 补注**：本段末句「本线下一步仍只剩复用 A0 ckpt 给 offline 线做 SAC collector，触发条件取决于 offline 线 broad validation 是否真需 RL-trained behavior policy」写于 2026-05-23，**已过期**——broad validation v2 于 2026-05-19 PASS、SAC collector 于 **2026-05-26 CLOSED**（见 §4.3 补注），本线两个角色现均已交付完毕，无待办）；**arrival_v2 prototype 产出双重方法学发现**——(1) H_information-bottleneck 实证（deployment-realistic 路径从 "s0→s1" 改写为 "s0+k=12"），(2) **manifest universal-floor 新概念**（vanilla SAC s0 在 single_u15_cross_tgt15 30-ep manifest 的 inherent ceiling = 27/30 = 0.900；ep {1208, 1216, 1228} 在 5/5 vanilla runs 全部 OOB）——作为 paper revision / rebuttal / offline 线 reward ablation 的可 cite 档案。Thesis-grade 47-run 矩阵 **撤销决定不变**（实测结果 + universal-floor finding 不改变产品判断）；本线下一步仍只剩 **复用 A0 ckpt 给 offline 线做 SAC collector** 这一可选项，触发条件取决于 offline 线 broad validation 是否真需 RL-trained behavior policy。
