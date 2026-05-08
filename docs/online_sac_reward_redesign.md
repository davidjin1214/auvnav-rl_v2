# Online SAC Reward Redesign Note【SHELVED / v6 Pre-Integration Spec】

> **⚠️ 此设计稿已搁置（2026-05-06）。**
> **收口报告见 [`docs/online_rl_line_summary.md`](online_rl_line_summary.md) §4.4** —— Online 线 thesis 矩阵已撤销，本设计稿对应的 `arrival_v2` reward（8 参数完整版）**不在 online 线落地**。
> 本文件保留作为设计档案；如未来 online 线重启或 offline 线需要 arrival-first reward preset 升级，可直接参考此处 v6 规范。
> 注意：[`auv_nav/reward.py`](../auv_nav/reward.py) 中已有的 `arrival_v2_simple` preset（commit `bd37412`）**与本文件 v6 的 `arrival_v2` 不是同一物**——前者只复用现有 RewardModelConfig 字段做 preset 重组，未实施本文件 §5.1 的 8 参数 / §6 的 MDP/replay 语义 / §7 的 discounted terminal dominance unit test。
>
> ---
>
> 文档版本：2026-05-07（v6；v5 语义保留，补齐 discounted unsafe-shortcut 与 pre-integration gate）
> 适用范围：在线 SAC / improved SAC 在 wake navigation benchmark 上的奖励函数设计。
> 直接动机：`notebooks/sac_thesis_s0_preflight_v2_completed.ipynb` 中 P1 的 `single_u15_upstream_tgt15` 预检暴露出 `return` 上升但 `success_rate` 退化到 0 的现象。
>
> **修订记录**：
> - **v1 (2026-04-26)**：首版，基于经验估算给出 arrival_v2 框架与初始参数。
> - **v2 (2026-04-27)**：经第二轮 expert review 发现 v1 在两处不严谨：
>   (a) `w_safety = 1–5` 是基于「avg per-step safety ≈ 0.05」的经验猜测，未与真实 P1 数据校准；
>   (b) `w_progress = 20` 与 `efficiency_v2` 比较时单步信号弱 3.3×，可能拖慢 SAC cold-start。
>   v2 已用 P1 v2 真实 train_log 校准了 safety_cost 分布（详见 §3.2），并相应修正了 §5.1 参数表和 §7.1 单元测试 fixture 量级。其余结构性判断（terminal dominance、arrival-first、early-failure penalty）维持不变。
> - **v3 (2026-04-27)**：经第三轮 RL expert review 发现 v2 在数值与不变量描述上的若干错误，详见 §11。要点：
>   (a) §5.1 expected return 表 4 行 safety penalty 错算 2-3×、`late_OOB` 漏一项 terminal；
>   (b) `w_safety` dominance flip 阈值从 v2 估的 4.5 修正为 **3.69**，hard upper bound 应取 **3.0**（不是 v2 的 4.0），§7.1 stress test 上限同步下调；
>   (c) `fast_OOB` **不是绝对最差**（修正后 `mid_OOB` 反而最差），但 safety property「无 suicide attractor」依然成立——正确的不变量是「`fast_OOB` 比任何 timeout 都差至少 100」；
>   (d) 补充 potential-based shaping 性质（现 §4.8）、R_fast_success 是时间压力主导项的归因（§3.2 末）、γ=0.995 折扣对 effective signal 的影响（§11.3）；
>   (e) §7.1 新增 trajectory-level fixture（从真实 train_log 抽 200 条），§7.2 新增 critic_loss / Q-value / alpha 监控。
>   v3 不改变 v2 §5.1 的最终参数取值，只修正数值与边界、补充 RL 理论分析、收紧测试覆盖。
> - **v4 (2026-04-27)**：经第四轮 RL + robotics review，确认 v3 的主方向正确但第一版落地偏复杂。v4 做四个收敛：
>   (a) `R_fast_success` 从主线默认项降级为 optional ablation（默认 `0`，可用 `arrival_v2_fast` 设为 `20`）；
>   (b) 修正 potential shaping 表述（现 §4.8）：在 `gamma=0.995` 下，`ΔPhi` 不是严格 policy-invariant potential shaping，严格形式应为 `gamma Phi(s') - Phi(s)`；
>   (c) 统一 §5.1 / §7.1 fixture 步数与 expected return，修正 `fast_success` 数值和 OOB 排序；
>   (d) 明确 `HARD_FAILURE_REASONS = SAFETY_FAILURE_REASONS - {"timeout"}`，timeout 必须走单独分支。
> - **v5 (2026-05-07)**：根据实现审查补齐四个 blocking 语义：
>   (a) `arrival_v2` 使用 `elapsed_time_s` 与 `initial_distance_m`，因此它必须同步扩展 observation / critic state，不能让 reward 依赖 policy 不可见的 episode context；
>   (b) `timeout` 在 `arrival_v2` 中是语义 terminal，在线 SAC replay 必须把 `terminated or truncated` 作为 non-bootstrap done，不能继续只用 `terminated`；
>   (c) §7 测试必须通过实际 `RewardModel.compute()` / reward helper，且同时验证 undiscounted 与 `gamma=0.995` discounted return 不变量；
>   (d) `arrival_v2_simple` 明确为 offline-only ablation preset，不是 `arrival_v2` 的实现起点；若保留在共享 registry，online SAC 必须有 guard / warning。
> - **v6 (2026-05-07)**：根据 discounted-return 复核补齐两个实现前 gate：
>   (a) 即使 `R_fast_success=0`，`gamma=0.995` 也会让更早拿到 `R_success` 的 risky shortcut 获得隐式速度优势；v5 的 `w_safety=0.5` 在 discounted unsafe-shortcut fixture 上失败；
>   (b) 主线候选 `w_safety` 从 `0.5` 提高到 `2.0`，并新增 discounted unsafe-shortcut unit test；
>   (c) 在正式集成 `arrival_v2` 到 core reward/env/train 之前，必须先用独立候选奖励 validator 通过纯公式 gate，不允许先改一轮正式路径再发现参数无效。

---

## 1. 结论

P1 的异常不是偶然噪声，而是当前 `efficiency_v2` 奖励与任务主指标不完全对齐导致的退化策略：

> agent 后期学会了快速出界，从而减少每步时间惩罚，使平均 return 上升；但任务成功率降为 0。

因此，`efficiency_v2` 不应继续作为 `single_u15_upstream_tgt15` 在线 SAC 主实验的默认目标函数。下一步应新增一个 arrival-first 的 reward preset，用它重跑 P1，再进入 sensor envelope 主实验。

推荐路线（v6 当前规范）：

1. 新增 `arrival_v2`：以到达为绝对主目标，使用归一化 progress、温和时间惩罚、强 terminal dominance。
2. `arrival_v2` 默认不加入 `R_fast_success`；若需要比较 time-to-goal preference，单独新增 `arrival_v2_fast` 做 ablation。
3. 不新增 training-time `efficiency_v3` reward；energy / path / soft safety 全部作为 eval-only metrics。
4. `arrival_v2` 必须同步定义 MDP state contract：policy/critic observation 至少包含 `elapsed_time_s / max_episode_time_s` 与 `initial_distance_m` 的归一化 episode context，或等价地删掉所有依赖这些 hidden variables 的 reward 项。
5. `arrival_v2` 必须同步定义 replay terminal 语义：`timeout` 是带 terminal penalty 的 finite-horizon terminal，SAC target 里不能 bootstrap。
6. 给 reward 加最小不变量测试，确保快速失败不会再比接近目标或成功 episode 获得更高 return；测试必须覆盖 undiscounted 与 discounted return。
7. 正式改 `auv_nav/reward.py` / `auv_nav/env.py` / `scripts/train_sac.py` 之前，先运行独立候选奖励 validator，确认 v6 参数在 discounted unsafe-shortcut 与 terminal dominance 上成立。

---

## 2. P1 证据复核

P1 路径：

```text
experiments/online_thesis_v1/preflight/p1_budget_calibration_v2_flowfix/
  efficiency_v2/sac_vanilla/s1_k4/seed_46/
```

关键观测：

| checkpoint | step | success_rate | eval_return | mean_time_s | progress_ratio | termination |
|---|---:|---:|---:|---:|---:|---|
| best success | 130008 | 0.600 | -260.79 | 183.61 | 0.844 | 18 goal + 12 timeout |
| best return in log | 990000 | 0.000 | -79.61 | 25.00 | -0.122 | not stored in CSV |
| final eval | 1000000 | 0.000 | -81.56 | 25.17 | -0.145 | 30 out_of_bounds |

`eval_log.csv` 中 `eval_return` 与其他指标的相关性：

| pair | correlation |
|---|---:|
| return vs success_rate | -0.630 |
| return vs elapsed_time | -0.963 |
| return vs progress_ratio | -0.506 |
| return vs path_efficiency | -0.625 |
| return vs safety_cost | -0.323 |

这说明后期 return 的改善主要来自 episode 变短，而不是导航变好。

---

## 3. 当前奖励的根因

### 3.1 efficiency_v2 的结构性失衡（v1 原始诊断）

当前 `efficiency_v2` preset 在 [`auv_nav/reward.py`](../auv_nav/reward.py) 中定义为：

```text
step_penalty = -1.0
reward_progress_gain = 1.0
success_reward = 100.0
failure_penalty = -20.0
energy_cost_gain = 0.0
safety_cost_gain = 0.25
```

环境时间尺度在 [`auv_nav/env.py`](../auv_nav/env.py) 中为：

```text
control_dt = 0.5 s
max_episode_time_s = 240 s
```

由于 `time_penalty_per_second=None` 时，代码把 `step_penalty=-1.0` 转成每个 control step 约 `-1` 的时间惩罚，一个完整 240 秒 episode 最多会产生约 `-480` 的时间项。相比之下：

- 成功终止奖励只有 `+100`
- 失败终止惩罚只有 `-20`
- soft safety 在 `efficiency_v2` 中只乘 `0.25`

所以慢速接近目标甚至成功的 episode 会因为时间项过大而 return 很低；快速出界只吃少量 step penalty 加一个很轻的 failure penalty，反而可能更高。

根因可以概括为三点：

1. **时间惩罚量级过大**：它压过了 success bonus 和 progress shaping。
2. **failure penalty 太轻**：快速出界没有被充分区分于普通低质量 episode。
3. **terminal outcome 没有 dominance**：reward 没有保证 `success > near-timeout > failure` 的全局排序。

### 3.2 真实 safety_cost 分布（v2 新增 evidence）

v1 的参数推荐（特别是 `w_safety = 1–5`）依赖了「per-step safety ≈ 0.05」的经验猜测。重做 reward 之前必须用 P1 v2 的真实 train_log 校准这个数字，否则 unit test fixture 也会带着同一个偏差。

数据源：`experiments/online_thesis_v1/preflight/p1_budget_calibration_v2_flowfix/efficiency_v2/sac_vanilla/s1_k4/seed_46/logs/train_log.jsonl`，11 253 条训练 episode 记录。每条含 `episode_cost`（= Σ per-step safety_cost.total）和 `episode_length`，per-step safety = `episode_cost / episode_length`。

**Per-step `safety_cost ∈ [0,1]` 分布（按训练阶段 × 成功/失败）**：

| phase | success | n | mean | median | p95 |
|---|---|---:|---:|---:|---:|
| early (0–100k) | False | 433 | 0.131 | 0.087 | 0.412 |
| early | True | 35 | 0.011 | 0.000 | 0.049 |
| peak (100–200k) | False | 337 | 0.129 | 0.083 | 0.401 |
| peak | True | 33 | **0.009** | 0.000 | 0.058 |
| decay (200–500k) | False | 2 178 | 0.150 | 0.116 | 0.395 |
| decay | True | 95 | **0.001** | 0.000 | 0.009 |
| collapse (500k+) | False | 8 142 | **0.185** | 0.158 | 0.403 |

**两个非显然事实**（直接决定 §5.1 参数选择）：

1. **在 efficiency_v2 训练数据中，已达成的 success episode safety_cost 几乎为零**（mean 0.001–0.011）。在该 reward 景观下，到达目标的轨迹自然远离边界，`w_safety` 主要影响失败 policy 的 return，对成功 policy 几乎无副作用，因此可以放心提高。**注意**：这条结论限定在 efficiency_v2 训练分布之内——arrival_v2（尤其启用 `R_fast_success` 时）的 reward 景观不同，可能让 agent 学到贴边走的 shortcut-style success；arrival_v2 训练数据出来后必须重新校准这条分布（详见 §11.8 与 §10.4 后续 step）。
2. **collapse 阶段（OOB 自杀）per-step safety_cost 最高（0.185）**，因为 policy 在边界附近滑行。这意味着 **soft safety_cost 本身已经在惩罚自杀行为**，只是当前 v2 用 `safety_cost_gain = 0.25` 太弱，压不过 `step_penalty = -1` 触发的「快出界省时间」激励。

**Episode-total `Σ safety_cost` 的 reward 量级（候选 `w_safety` 下）**：

| phase | success | avg ep_cost | w=0.25 | w=0.5 | w=1.0 | w=2.0 |
|---|---|---:|---:|---:|---:|---:|
| peak | False | 21.80 | -5.5 | -10.9 | -21.8 | -43.6 |
| peak | True | 2.67 | -0.7 | -1.3 | -2.7 | -5.3 |
| collapse | False | 9.75 | -2.4 | -4.9 | -9.7 | -19.5 |

→ 即便 `w=2.0`，成功 episode 的 safety penalty 总额也只有 -2.7（噪音级别），不影响成功 policy。

**Dominance stress test（关键约束）**：把 safety penalty 加到 timeout 之上，会不会让 timeout 比 OOB 更差，从而**重新创造 suicide 激励**？

按 §5.1 设计，timeout terminal ≈ `R_timeout + R_final_distance × D_f/D_i ≈ -100`（worst case），中等 OOB terminal ≈ `R_failure + R_early_failure × 0.5 + R_final_distance ≈ -200`。OOB 比 timeout 至少差 100。要让 safety penalty 翻转排序，需要：

```
w_safety × (peak_fail_avg_cost − collapse_avg_cost) > 100
w_safety × (21.80 − 9.75) > 100
w_safety × 12.05 > 100
w_safety > 8.3   (broad failure 比较)
```

v2 曾用 episode-total 近似得到 `w_safety_max ≈ 4.5`，但 v3/v4 按 §7.1 fixture 一致口径重算后，flip threshold 是 **3.69**。因此当前 hard stress 上限取 **3.0**，任何高于 3.69 的取值都可能让 timeout 比 OOB 更差，重新引入 suicide 激励。

**速度/安全博弈点（v6 修正：discounted SAC 的隐式速度奖励）**：

v4 的 per-second break-even 只适用于 undiscounted 近似。在线 SAC 实际优化 `gamma=0.995` 的 discounted target，即使 `R_fast_success=0`，更早到达仍会因为更早拿到 `R_success=100` 而获得隐式速度优势。

用 §7.1 的同口径 synthetic shortcut fixture 复核：

```text
safe_success:   180 steps, D_i=65, D_f=4, avg_safety=0.005
risky_success:  120 steps, D_i=65, D_f=4, avg_safety=0.150
```

在 v5 默认 `w_safety=0.5` 下，discounted return 反而是：

```text
risky_success ≈ 82.7  >  safe_success ≈ 70.2
```

这说明“`R_fast_success=0` + `w_safety=0.5`”仍可能训练出贴边快速成功的 shortcut policy。按同一 fixture 解 discounted break-even：

| reward variant | discounted shortcut break-even `w_safety` | v6 implication |
|---|---:|---|
| no fast-success bonus | `> 1.46` | 主线候选值必须高于此下限 |
| with `R_fast_success=20` | `> 1.71` | `arrival_v2_fast` 必须单独验证，不可沿用 v5 结论 |

**v6 结论**：主线 `arrival_v2` 仍默认不使用 `R_fast_success`，但 `w_safety` 候选默认值从 `0.5` 提高到 **`2.0`**。它高于 discounted unsafe-shortcut 下限，同时仍低于 timeout-vs-hard-failure dominance flip 阈值 `3.69` 与 hard stress 上限 `3.0`。若未来启用 `arrival_v2_fast`，必须重新跑 §7.1 discounted shortcut test 与 §8.1 pre-integration gate。

---

## 4. 设计原则

新 reward 必须满足以下约束。

### 4.1 Arrival First

在线 SAC 主实验的第一目标应是学会到达目标。效率、能耗、路径长度和 soft safety 只能在可到达策略之间做二级排序，不能让失败策略优于成功策略。

### 4.2 Terminal Dominance

在合理轨迹集合上，累计 return 应满足：

```text
fast_success
  > slow_success
  > timeout_near_goal
  > timeout_far_goal
  > late_hard_failure
  > fast_hard_failure
```

这条排序比某个具体参数值更重要。任何 reward 调参都必须先通过这个不变量。

### 4.3 Dense Signal 只做辅助

progress shaping 应帮助 SAC 学习方向，但不能压过终局。建议使用归一化 progress，而不是直接使用米制距离：

```text
progress_delta_norm = (previous_distance - current_distance) / D_init_safe
```

这样不同 benchmark 的 start-goal 距离变化不会直接改变 reward 尺度。

### 4.4 不奖励快速结束

失败 episode 的 return 不应因更短而更高。对 hard failure 应加入 early-failure penalty：

```text
early_failure_weight = 1 - elapsed_time / max_episode_time
```

越早发生 `out_of_bounds`、姿态/速度失稳、非有限状态等 hard failure，惩罚越强。

### 4.5 分离 Hard Failure 与 Soft Risk

`out_of_bounds`、`attitude_limit`、`speed_limit` 等 hard failure 应由 terminal penalty 处理；soft safety cost 用于提前提示风险，不应承担主要的失败惩罚职责。

### 4.6 Reward 必须满足 MDP State Contract（v5 新增）

`arrival_v2` 中有两类项依赖 episode context：

```text
progress_delta_norm = Δd / D_init_safe
early_failure_weight = 1 - elapsed_time_s / max_episode_time_s
R_fast_success       = R_fast_success * (1 - elapsed_time_s / max_episode_time_s)
final_distance_ratio = final_distance_m / D_init_safe
```

这些变量当前在 `env.step()` 的 `info` 中可见，但不在 actor/critic observation 中。若直接把它们用于训练 reward，同一个 observation-action-next_observation transition 可能因为 hidden elapsed time 或 hidden initial distance 获得不同 reward/Q target。这会让 SAC 面对非 Markov 的 value function，尤其破坏 early-failure 与 timeout 的 credit assignment。

因此，实现 `arrival_v2` 时必须二选一：

1. **推荐**：为 `arrival_v2` 新增 objective-specific observation context，把下面两个标量加入 actor/critic observation：
   ```text
   elapsed_time_frac = elapsed_time_s / max_episode_time_s
   initial_distance_norm = D_init_safe / OBS_DISTANCE_SCALE
   ```
   这会改变 observation dimension；旧 checkpoint / 旧 offline dataset 与 `arrival_v2` 不兼容，必须重训 / 重收集。
2. **备选**：不扩 observation，但从 training reward 中删掉所有依赖 hidden episode context 的项，改用 observation 可恢复的固定尺度（例如 benchmark-level distance scale）和纯 terminal reason。这个版本应另命名，不应叫本文 `arrival_v2`。

v5 规范采用方案 1。`elapsed_time_frac` 是 finite-horizon MDP 的剩余时间状态；`initial_distance_norm` 是 reward normalization 与 terminal distance ratio 的 episode context。两者都应进入 actor 与 critic 的输入，AsymCritic 也不能只给 critic 看，否则 actor 执行期仍缺状态。

### 4.7 Timeout 是语义 Terminal（v5 新增）

本文 `arrival_v2` 给 `timeout` 单独 terminal penalty，因此 timeout 不再只是 Gymnasium 意义上的 time-limit truncation，而是 finite-horizon 任务失败的一种语义 terminal。实现要求：

```text
arrival_v2 SAC replay done = terminated or truncated
legacy SAC replay done     = terminated   # 可保持现状以兼容旧实验
offline dataset dones      = terminated or truncated
```

如果未来决定继续对 timeout bootstrap，就必须删除 `R_timeout` 与 timeout final-distance terminal 项，只把 timeout 当作 episode 切分边界。不能同时「给 timeout terminal penalty」又「在 Bellman target 中从 timeout bootstrap」；这会让 critic 学到的目标与 §7 的 episode-return dominance 测试不一致。

### 4.8 Potential-Like Progress Shaping（v4 修正）

`progress_delta_norm = (prev_dist − curr_dist) / D_init` 可以写成 potential difference：

```text
F(s, s') = Phi(s') - Phi(s),   Phi(s) = -dist(s) / D_init
```

但在 SAC 使用 `gamma=0.995` 的 discounted MDP 中，严格 policy-invariant 的 potential-based shaping 形式应为：

```text
F_gamma(s, s') = gamma * Phi(s') - Phi(s)
```

因此，当前 `progress_delta_norm` **不是严格意义上保证不改变最优策略的 Ng-style shaping**；它更准确地说是 bounded, path-independent 的距离变化 shaping。其重要性质仍然成立：在未折扣累计意义下，

```text
Σ progress_delta_norm = (D_init - D_final) / D_init
```

所以 policy 不能通过“前进-后退-前进”的振荡套利 progress reward。这足以支持工程使用，但 thesis 中不能声称它在 `gamma=0.995` 下严格 policy-invariant。

但 reward 中**其余项都不是 potential-based**，会改变最优政策：

| 项 | 是否 potential-based | 改变最优政策的方向 |
|---|---|---|
| `w_progress × Δd / D_init` | 近似 / undiscounted 是 | 提供方向信号；discounted SAC 下不严格保证 policy invariance |
| `w_time × dt / T_max` | ❌ 否 | 偏向更短 episode |
| `w_safety × safety_cost` | ❌ 否 | 偏向远离边界 |
| `R_success`（terminal） | ❌ 否（绝对 reward）| 偏向到达 |
| `R_fast_success × (1 − t/T_max)` | ❌ 否 | 偏向快速到达 |
| `R_failure / R_early_failure / R_final_distance × D_f/D_i` | ❌ 否 | 偏向避免 hard failure |

这是 **intended trade-off**：我们用 bounded progress shaping 提供 dense learning signal，用 non-potential terminal 项表达任务语义（arrival / timeout / hard failure）。Thesis 章节应明示这一点，否则审稿人会问「为什么不用纯 potential-based reward」。

答：**纯 potential-based reward 在稀疏成功+硬失败的环境下无法表达「成功 vs 失败」的语义差异**——所有 trajectory 的 Σreward 都等于 `−Φ(s_T) + Φ(s_0)`，与是否 success/failure 无关。我们必须引入 non-potential terminal 项来表达这个语义。

---

## 5. 推荐奖励结构

### 5.1 `arrival_v2`

建议新增 `arrival_v2` preset。v4 的第一版落地目标是**最小足够复杂度**：主线 reward 只表达 arrival-first 和 hard-failure avoidance；time-to-goal / energy / path quality 先放到 eval metrics。若需要显式训练“更快成功”，再单独开 `arrival_v2_fast` ablation。

形式如下：

```text
per-step:
  r = + w_progress * progress_delta_norm
      - w_time * (dt / max_episode_time)
      - w_safety * soft_safety_cost_norm

terminal:
  if reason == "goal":
      r += R_success
      r += R_fast_success * (1 - elapsed_time / max_episode_time)   # default 0

  elif reason in HARD_FAILURE_REASONS:   # excludes "timeout"
      r -= R_failure
      r -= R_early_failure * (1 - elapsed_time / max_episode_time)
      r -= R_final_distance * clipped(final_distance / D_init_safe, 0, 2)

  elif reason == "timeout":
      r -= R_timeout
      r -= R_final_distance * clipped(final_distance / D_init_safe, 0, 2)
```

实现中必须显式定义：

```python
HARD_FAILURE_REASONS = SAFETY_FAILURE_REASONS - {"timeout"}
```

当前 `SAFETY_FAILURE_REASONS` 包含 `"timeout"`，不能直接复用，否则 timeout 会错误进入 hard-failure 分支。

初始参数建议（v6 当前规范；正式集成前仍需通过 §8.1 独立 gate）：

| parameter | v6 candidate default | optional fast ablation | rationale |
|---|---:|---:|---|
| `w_progress` | **50** | 50 | 成功 episode 总 progress reward 约 50，为 `R_success` 的一半；dense signal 与旧 objective 同量级。 |
| `w_time` | **5** | 5 | 完整 episode 时间项约 -5，只提供温和时间压力，不主导终局。 |
| `w_safety` | **2.0** | 2.0 | 高于 discounted unsafe-shortcut break-even 1.46，且低于 timeout-vs-OOB flip threshold 3.69。 |
| `R_success` | **100** | 100 | 主任务语义。 |
| `R_fast_success` | **0** | 20 | 默认不进入主线；如需显式 time-to-goal training pressure，用 `arrival_v2_fast` 单独比较。 |
| `R_failure` | **100** | 100 | hard failure 基础重罚。 |
| `R_early_failure` | **100** | 100 | 阻止 fast-OOB 套利。 |
| `R_timeout` | **50** | 50 | timeout 应差于成功，但优于 hard failure。 |
| `R_final_distance` | **50** | 50 | 区分 timeout/failure 时是否接近目标。 |

**Hard parameter bounds（写入 §7.1 unit test stress 子测试）**：

- `w_safety < 3.0`（dominance flip threshold = **3.69**，留 ~20% 安全余量）— v2 估的 4.5 是用近似算法（peak vs collapse episode total）得到，用 fixture 一致重算后下调，详见 §11.1
- no-fast default 下 discounted unsafe-shortcut bound 约 `w_safety > 1.46`；`w_safety=2.0` 是当前候选值
- `arrival_v2_fast` 若启用 `R_fast_success=20`，discounted unsafe-shortcut bound 约 `w_safety > 1.71`，必须单独通过 ordering / behavior regression
- `w_progress < R_success`（terminal dominance，Σprogress 不能超过 R_success=100）

**预期 undiscounted episode return 排序**（v6 默认 `R_fast_success=0, w_safety=2.0`；与 §7.1 fixture 步数一致；discounted 不变量另见 §7.1）：

```text
fast_success      (120 步, D_f/D_i=0.06, sft 0.005)  ≈ +46.9 − 1.25 −   1.2 + 100                 = +144.5
slow_success      (240 步, D_f/D_i=0.06, sft 0.005)  ≈ +46.9 − 2.50 −   2.4 + 100                 = +142.0
unsafe_success    (180 步, D_f/D_i=0.06, sft 0.150)  ≈ +46.9 − 1.88 −  54.0 + 100                 =  +91.0
timeout_near_goal (480 步, D_f/D_i=0.20, sft 0.130)  ≈ +40.0 − 5.00 − 124.8 − 50 − 10             = −149.8
timeout_far_goal  (480 步, D_f/D_i=1.00, sft 0.130)  ≈   0.0 − 5.00 − 124.8 − 50 − 50             = −229.8
late_OOB          (240 步, D_f/D_i=1.00, sft 0.150)  ≈   0.0 − 2.50 −  72.0 − 100 − 50 − 50       = −274.5
fast_OOB          ( 60 步, D_f/D_i=1.20, sft 0.185)  ≈ −10.0 − 0.63 −  22.2 − 100 − 87.5 − 60     = −280.3
mid_OOB           (120 步, D_f/D_i=1.50, sft 0.180)  ≈ −25.0 − 1.25 −  43.2 − 100 − 75 − 75       = −319.5
```

每行的项依次为：`w_progress×Σprogress_norm`、`w_time×Σ(dt/T_max)`、`w_safety×Σsafety`（用 fixture per-step × n_steps）、`R_success/R_failure/R_timeout`、`R_early_failure`、`R_final_distance × D_f/D_i`。

> **v2 → v3 数值差异**：v2 表 timeout/OOB 行的 safety penalty 普遍低估 2-3×，且 `late_OOB` 漏算 R_early_failure 或 R_final_distance 一项。详见 §11.1。
>
> **v3 → v4 数值差异**：v3 表中 `fast_success` 写成 60 步，而 §7.1 fixture 是 120 步；v4 统一为 120 步。同时默认 `R_fast_success=0`，所以 success return 下降约 10-17.5，但 success-vs-failure margin 仍然充足。

**关键不变量**（每次调参都要用 §7.1 测试验证；v6 当前措辞）：

- undiscounted 排序：`fast_success > slow_success > unsafe_success > timeout_near > timeout_far > late_OOB > fast_OOB > mid_OOB` ✓
- discounted 核心排序：`success > timeout > hard_failure`，且 `fast_OOB << timeout` ✓
- discounted unsafe-shortcut：`safe_success(180 steps, sft=0.005) > risky_success(120 steps, sft=0.150)` ✓
- **canonical / realistic unsafe success fixtures 均显著优于 failure fixtures**（default fixture 下 margin ≥ 100；不声称对任意无限 soft-risk success 全局成立）✓
- **`fast_OOB` 比任何 timeout 都差至少 100**（防止 OOB 自杀套利；v3 修正：原 v2 「fast_OOB 是绝对最差」**与 fixture 数值不符**——`mid_OOB` 才是绝对最差。但「无 suicide attractor」只需要 fast_OOB << timeout，这个约束依然满足）✓
- 注意：`mid_OOB` 比 `fast_OOB` 更差是合理的——前者既走错方向又触发 hard failure，progress 项吃了 −25 而后者只 −10

### 5.2 `efficiency_v3`（v2 修订：限定为 evaluation-only metric）

**v1 的歧义**：v1 §5.2 写「在 arrival_v2 成功稳定后追加 efficiency_v3」，这有两种解读：

- 解读 A：finalist policy 用 efficiency_v3 reward **重新训练** → §2 vs §3+ 跨节 absolute return 不可比，破坏 thesis comparability；
- 解读 B：finalist policy 用 efficiency_v3 reward **重新评估**（policy 不动）→ 跨节可比性 OK，但仍需要在 eval loop 加 efficiency 计算逻辑；
- 解读 C：保留 arrival_v2 reward 不动，efficiency 指标（path_length / energy / soft_safety_avg）做成 **post-hoc evaluation metric**，不进入 reward → 最简洁。

**v2 明确选 C**：efficiency 不写进 training reward，做成 eval-time only metrics。

理由：
1. **跨节可比性**：online RL chapter 的 §2/§3/§4/§5 全部用单一 reward (arrival_v2)，确保 absolute return 可以直接比较 sensor envelope、AsymCritic 等不同处理。
2. **避免 second-order reward hacking**：把 efficiency 做成 reward 项会引入新的 trade-off 维度（safety vs energy vs path），每个新项都需要重做 dominance 测试，且容易在某个未预期的 benchmark 上 spec game。
3. **论文叙事更清晰**：「training objective 是到达；efficiency 是 evaluation-only 的 deployment metric」对应真实 AUV 部署逻辑（先把任务做完再谈效率）。

**`efficiency_v3` 的 v2 定义**（仅作为 eval summary 字段，不进入 reward）：

```text
eval_path_length_excess = (path_length - direct_distance) / direct_distance
eval_energy_total       = Σ |actuator_rpm| × dt    （已在 eval rollout 里收集）
eval_soft_safety_avg    = Σ safety_cost.total / N_steps  （= mean per-step safety）
eval_time_to_goal_s     = elapsed_time_s   （成功 episode）
```

eval summary 同时输出 success-conditioned 与 unconditioned 两个版本（参见 §7.3）。Finalist 排序：

1. **First**: success_rate（必须达到阈值，例如 ≥ 0.8）；
2. **Second**: 在 success rate 同档（差异 < 1σ）的方法之间按 efficiency 指标排序。

如果未来确实需要 efficiency 进入 training reward（例如做能耗-success Pareto 曲线），可以**新开一个章节**用 efficiency_v3 reward 重训单独 finalist，明确标注为「与主线 arrival_v2 不可直接比 absolute return」。但默认不做。

### 5.3 `arrival_v2_simple` 的定位（v5 新增）

当前代码中的 [`REWARD_OBJECTIVE_PRESETS["arrival_v2_simple"]`](../auv_nav/reward.py) 是 offline C1 reward ablation 的最小 preset，不是本文 `arrival_v2` 的第一阶段实现。它的目标是用现有字段快速改变 terminal 量级：

```text
step_penalty = -0.2
success_reward = +200
failure_penalty = -200
timeout_penalty = -50
safety_cost_gain = 0.5
```

它刻意没有实现本文完整版的四个关键机制：

1. 没有 normalized progress；
2. 没有 normalized time / early-failure penalty；
3. 没有 final-distance terminal penalty；
4. 没有 objective-specific observation context 与 discounted dominance tests。

因此它只能作为 **offline-only ablation preset** 保留。若它继续留在共享 `REWARD_OBJECTIVE_PRESETS` 中，`scripts/train_sac.py` 应对 `arrival_v2_simple` 做显式 guard：

```text
default: forbid online SAC training with arrival_v2_simple
override: require --allow-offline-only-objective or equivalent explicit flag
```

`arrival_v2_simple` 的现有测试也应只锁定它自己的简化行为。若继续保留这些测试，应通过 `RewardModel.compute()` 重放 fixture，而不是手写 `step_penalty * steps + progress + terminal`，以免漏掉 `safety_cost_gain` 或 timeout terminal-violation 行为。不要把这些测试当作本文 `arrival_v2` 的验收标准；未来实现完整版时应新增独立的 `arrival_v2` tests，而不是扩展 `arrival_v2_simple` 的含义。

---

## 6. 实现影响

当前 [`RewardModel.compute()`](../auv_nav/reward.py) 只接收：

```text
progress
safety_cost
reason
terminated / truncated
actuator_rpm
```

要实现 `arrival_v2`，需要额外提供：

```text
elapsed_time_s
max_episode_time_s
previous_distance_to_goal_m
current_distance_to_goal_m
initial_distance_to_goal_m
dt
```

推荐做法：

1. 扩展 `RewardModelConfig`，加入 `max_episode_time_s`、`d_init_min_m`、`w_progress` / `w_time` / `w_safety`、`R_success` / `R_fast_success` / `R_failure` / `R_early_failure` / `R_timeout` / `R_final_distance` 等 terminal 系列参数。保留 legacy 字段用于旧 objective，避免重写旧实验。
2. 扩展 `RewardModel.compute()` 参数，使 reward model 自己完成归一化，而不是把逻辑散落在 `env.step()` 中。`env.step()` 只负责传入原始量：
   ```text
   previous_distance_to_goal_m
   current_distance_to_goal_m
   initial_distance_to_goal_m
   elapsed_time_s
   max_episode_time_s
   dt
   reason / terminated / truncated
   safety_cost / actuator_rpm
   ```
3. `arrival_v2` 必须有 objective-specific observation contract（见 §4.6）。推荐实现为：
   ```text
   PlanarRemusEnvConfig.include_episode_context_obs: bool = False
   arrival_v2 / arrival_v2_fast preset 自动设为 True
   observation 追加 [elapsed_time_frac, initial_distance_norm]
   ```
   这会把 `s0/s1/s2` 的 observation dimension 分别从 `10/12/16` 变为 `12/14/18`（未加 history 前）。所有 `ObservationHistoryWrapper`、offline dataset metadata、checkpoint metadata 和 evaluation loader 都必须记录新维度。
4. `scripts/train_sac.py` 必须按 objective 选择 bootstrap 语义：
   ```text
   if reward_objective in {"arrival_v2", "arrival_v2_fast"}:
       replay_done = terminated or truncated
   else:
       replay_done = terminated
   ```
   vector env 与 single env 两条路径都要一致。`scripts/collect_offline_data.py` 当前已保存 `dones = terminated or truncated`，可保留；`train_offline.py` 已检查 dataset objective，仍需确保 `arrival_v2` dataset 是用新 observation context 重收集的。
5. 保留 `arrival_v1 / efficiency_v1 / efficiency_v2` 作为历史 preset，新增 `arrival_v2`；可选新增 `arrival_v2_fast` 只用于 ablation。`arrival_v2_simple` 保持 offline-only ablation，不能升级为 alias。
6. `efficiency_v3` 不作为 reward preset，仅作为 eval summary 字段，详见 §5.2。
7. 在 `scripts/train_sac.py` 的 CLI 层保持 `--objective` 接口不变，但新增 offline-only guard：默认禁止 `--objective arrival_v2_simple` 进入 online SAC，除非显式 override。
8. 新增 hard-failure 集合时不要复用包含 timeout 的 `SAFETY_FAILURE_REASONS`：

```python
TIMEOUT_REASONS = frozenset({"timeout"})
HARD_FAILURE_REASONS = SAFETY_FAILURE_REASONS - TIMEOUT_REASONS
SAFETY_TERMINAL_VIOLATION_REASONS = HARD_FAILURE_REASONS
```

`SafetyCostModel` 的 `terminal_violation` 也应使用 `SAFETY_TERMINAL_VIOLATION_REASONS`，不要把 timeout 计入 safety violation。否则 `arrival_v2` 的 timeout 会同时吃 `R_timeout` 与 soft safety terminal cost，且 eval safety metric 会把普通超时误记为安全违规。

### 6.1 边界情况与防御性实现（v2 新增）

**(a) `D_init` clamp**：

`progress_delta_norm = Δd / D_init` 在 `D_init` 极小（如 task sampler 偶尔采到 start ≈ goal 的 case）时会让单步 progress 奖励 blow up。例如 `D_init = 4 m`（约 1 个 goal_radius）时，AUV 一步走 1m 即触发 reward `w_progress × 1/4 = +12.5`，是正常 case 的 ~40 倍。

实现要求：

```python
D_INIT_MIN = 10.0  # ~1 vehicle length + 1 goal_radius
D_init_safe = max(initial_distance_m, D_INIT_MIN)
progress_delta_norm = (prev_dist - curr_dist) / D_init_safe
```

`D_init_safe` 同样用于 terminal 的 `D_f / D_i` 项。Clamp 阈值 10m 的选择基于：vehicle length ≈ 1.6m、goal_radius = 4m、再留一个 vehicle length 余量。task sampler 当前似乎不会采这种 case，但**未来 sensor envelope 实验可能加入「目标偶尔出现在 start 附近」的测试场景**，clamp 是必需防御。

unit test 必须覆盖 `D_init = 0.5 × D_INIT_MIN` 与 `D_init = 0` 两种 edge case，确认不会 NaN / Inf / blow up。

**(b) `final_distance` clamp**：

`R_final_distance × clipped(D_f / D_i, 0, 2)` 中的 `clip(..., 0, 2)` 已经处理「policy 越走越远」的极端 case（不让单一指标无限放大）。维持。

**(c) numerical failure 与 nonfinite_state 的告警**：

env 当前会在 `nonfinite_state` / `nonfinite_derivative` 时 terminate，触发 `R_failure` 重罚。这种 termination **不是 policy 的策略选择**，而是**物理引擎的数值崩溃**。把它当作普通 hard failure 处理是可接受的，但需要：

- eval summary 必须按 reason 分别输出 termination count（参见 §7.3）；
- 若任一 evaluation 中 `n_nonfinite / n_total > 1%`，训练日志打印 WARNING；
- 若 `n_nonfinite / n_total > 5%`，视为环境数值不稳定，停止训练并人工 inspect。

**(d) RewardModel 调用点**：

需要同步更新所有调用 `RewardModel.compute()` 的地方，传入新签名所需的 `previous_distance_to_goal_m / current_distance_to_goal_m / initial_distance_to_goal_m / elapsed_time_s / max_episode_time_s / dt`：

- `auv_nav/env.py` 的 `step()` 主路径
- `scripts/train_utils.py` 的 eval rollout（间接通过 env）
- `scripts/collect_offline_data.py` 的离线数据收集（用于 RLPD / offline RL 数据）
- 测试 fixture（参见 §7.1）

env 已经有 `self.initial_distance` 与 `self.last_distance`，env 可直接传入 reward model；但由于 v5 要求这些 episode context 也进入 observation，`_build_observation()` / observation layout / obs dim metadata 仍必须同步更新。

**(e) Objective metadata 与兼容性**：

实现 `arrival_v2` 后，checkpoint 与 offline dataset metadata 至少要记录：

```text
reward_objective
reward_config
include_episode_context_obs
observation_dim
history_length
probe_layout
timeout_bootstrap_semantics
```

加载 checkpoint / offline dataset 时必须拒绝以下混用：

- `arrival_v2` checkpoint 用 legacy observation layout eval；
- legacy checkpoint 用 `arrival_v2` observation layout eval；
- `arrival_v2_simple` dataset 当作 `arrival_v2` dataset 训练；
- `arrival_v2` dataset 与 `arrival_v2_fast` objective 混用。

---

## 7. 必须新增的测试

### 7.1 Reward Ordering Unit Test（v6 最小必需测试）

构造 synthetic episode summary，通过实际 `RewardModel.compute()` 或与其共享的 pure helper 逐 step 重放 reward，测试累计 return 排序。测试不能手写 `sp * steps + progress + terminal` 这种近似公式，否则会漏掉 `safety_cost_gain`、timeout 分支、terminal violation 等实现细节。

```text
fast_success
  > slow_success
  > unsafe_success
  > timeout_near_goal
  > timeout_far_goal
  > late_out_of_bounds
  > fast_out_of_bounds
  > mid_out_of_bounds
```

这类测试不依赖 PyTorch，也不依赖真实流场，但必须复用生产 reward 逻辑。

**v1 vs v2 fixture 量级差异**：v1 没有指定 fixture 中的 per-step safety_cost 量级。如果默认用 0.05（v1 的隐含猜测），unit test 会通过，但生产中 failure-policy 真实是 0.13–0.18，dominance 边界会被错估。v2 强制 fixture 使用 §3.2 实测数据量级。

**Recommended fixtures**（dataclass，见下方代码草案）：

```python
@dataclass(frozen=True)
class EpisodeFixture:
    name: str
    n_control_steps: int
    success: bool
    reason: str           # "goal" | "timeout" | "out_of_bounds" | ...
    initial_distance_m: float
    final_distance_m: float
    avg_per_step_safety: float   # ∈ [0,1]
    # 计算 derived: elapsed_time_s = n_control_steps × control_dt
    #               progress_sum_norm = (D_init - D_final) / D_init_safe
    #               total_safety_cost = avg_per_step_safety × n_control_steps

FIXTURES = [
    # name, steps, success, reason, D_init, D_final, avg_safety
    EpisodeFixture("fast_success",    120, True,  "goal",          65.0,  4.0,  0.005),  # success ≈ 0
    EpisodeFixture("slow_success",    240, True,  "goal",          65.0,  4.0,  0.005),
    EpisodeFixture("timeout_near",    480, False, "timeout",       65.0, 13.0,  0.130),  # 接近目标
    EpisodeFixture("timeout_far",     480, False, "timeout",       65.0, 65.0,  0.130),  # 距离没变
    EpisodeFixture("late_oob",        240, False, "out_of_bounds", 65.0, 65.0,  0.150),
    EpisodeFixture("mid_oob",         120, False, "out_of_bounds", 65.0, 97.5,  0.180),  # 走错方向
    EpisodeFixture("fast_oob",         60, False, "out_of_bounds", 65.0, 78.0,  0.185),  # 自杀
    EpisodeFixture("unsafe_success",  180, True,  "goal",          65.0,  4.0,  0.150),  # shortcut stress
]
```

`compute_return()` 必须走真实 reward 逻辑。推荐把归一化公式抽成 `RewardModel.compute()` 内部可复用 helper；测试逐 step 构造距离序列和终局 reason：

```python
def compute_return(
    objective: str,
    fixture: EpisodeFixture,
    *,
    gamma: float | None = None,
    w_safety_override: float | None = None,
) -> float:
    """Replay a synthetic episode through RewardModel, not a handwritten formula."""
    reward_model = make_reward_model(objective, w_safety_override=w_safety_override)
    distances = np.linspace(
        fixture.initial_distance_m,
        fixture.final_distance_m,
        fixture.n_control_steps + 1,
    )
    total = 0.0
    discount = 1.0
    for step_idx in range(fixture.n_control_steps):
        reason = fixture.reason if step_idx == fixture.n_control_steps - 1 else "running"
        terminated = reason not in {"running", "timeout"}
        truncated = reason == "timeout"
        reward = reward_model.compute(
            previous_distance_to_goal_m=float(distances[step_idx]),
            current_distance_to_goal_m=float(distances[step_idx + 1]),
            initial_distance_to_goal_m=fixture.initial_distance_m,
            elapsed_time_s=(step_idx + 1) * CONTROL_DT,
            max_episode_time_s=MAX_EPISODE_TIME_S,
            dt=CONTROL_DT,
            safety_cost=fixture.avg_per_step_safety,
            reason=reason,
            terminated=terminated,
            truncated=truncated,
            actuator_rpm=0.0,
        ).reward
        total += discount * reward
        if gamma is not None:
            discount *= gamma
    return float(total)
```

**Test cases**：

```python
def test_arrival_v2_terminal_dominance():
    """Asserts the canonical ordering at v6 candidate params."""
    returns = {f.name: compute_return("arrival_v2", f) for f in FIXTURES}
    expected_order = [
        "fast_success", "slow_success", "unsafe_success",
        "timeout_near", "timeout_far",
        "late_oob", "fast_oob", "mid_oob",
    ]
    actual = sorted(returns, key=returns.get, reverse=True)
    assert actual == expected_order, f"order broken: {returns}"

def test_success_dominates_failure():
    """Canonical and unsafe-success fixtures both dominate failures."""
    returns = {f.name: compute_return("arrival_v2", f) for f in FIXTURES}
    successes = [v for k,v in returns.items() if "success" in k]
    failures  = [v for k,v in returns.items() if "success" not in k]
    assert min(successes) > max(failures) + 100

def test_discounted_terminal_dominance_gamma_0995():
    """SAC optimizes discounted targets, so core dominance must hold under gamma."""
    returns = {f.name: compute_return("arrival_v2", f, gamma=0.995) for f in FIXTURES}
    success_min = min(v for k, v in returns.items() if "success" in k)
    timeout_max = max(v for k, v in returns.items() if "timeout" in k)
    hard_failure_max = max(v for k, v in returns.items() if "oob" in k)
    assert success_min > timeout_max + 50, returns
    assert returns["timeout_far"] > returns["late_oob"], returns
    assert returns["fast_oob"] < timeout_max - 100, returns
    assert timeout_max > hard_failure_max, returns
    assert returns["slow_success"] > returns["unsafe_success"] + 10, returns

def test_no_suicide_attractor():
    """Defends against suicide-via-fast-OOB attractor.
    v3 修正：v2 用的是 `fast_oob == min(returns)`，与 fixture 数值不符
    （`mid_oob` 走错方向 + 触发 hard failure，return 比 fast_oob 更负）。
    实际安全语义：fast_OOB 比任何 timeout 都差，suicide 不划算即可。"""
    returns = {f.name: compute_return("arrival_v2", f) for f in FIXTURES}
    timeout_max = max(v for k, v in returns.items() if "timeout" in k)
    assert returns["fast_oob"] < timeout_max - 100, \
        f"suicide attractor: fast_oob={returns['fast_oob']} vs timeout_max={timeout_max}"

def test_w_safety_dominance_bound():
    """Stress: w_safety in [1.5, 3.0] must preserve timeout > hard failure.
    v3 修正：v2 上限 4.0 是用 episode-total 法估算，按 fixture per-step × n_steps
    一致重算，flip 阈值是 3.69，hard upper bound 取 3.0（~20% 余量）。"""
    for w in [1.5, 2.0, 2.5, 3.0]:
        returns = {f.name: compute_return("arrival_v2", f, w_safety_override=w) for f in FIXTURES}
        # 关键约束：timeout_far 必须好于 late_oob，否则 timeout > OOB 失败
        assert returns["timeout_far"] > returns["late_oob"], \
            f"flipped at w={w}: timeout_far={returns['timeout_far']}, late_oob={returns['late_oob']}"
    # 阈值上方应当确实 flip（防御算法回归）
    returns_at_limit = {f.name: compute_return("arrival_v2", f, w_safety_override=4.0) for f in FIXTURES}
    assert returns_at_limit["timeout_far"] < returns_at_limit["late_oob"], \
        "expected flip at w=4.0; if not, dominance derivation is wrong"

def test_w_safety_lower_bound_blocks_unsafe_shortcut_no_fast():
    """Synthetic shortcut fixture: 30% shorter time but 30× safety cost.
    With v6 candidate R_fast_success=0, w_safety=2.0 should make the safe
    route better under both undiscounted and gamma=0.995 returns."""
    safe_route = EpisodeFixture("safe_succ",     180, True, "goal", 65, 4, 0.005)
    risky_route = EpisodeFixture("shortcut_succ", 120, True, "goal", 65, 4, 0.150)
    for gamma in [None, 0.995]:
        r_safe = compute_return("arrival_v2", safe_route, gamma=gamma)
        r_risky = compute_return("arrival_v2", risky_route, gamma=gamma)
        assert r_safe > r_risky, \
            f"unsafe shortcut wins at gamma={gamma}: safe={r_safe}, risky={r_risky}"

def test_v5_w_safety_0p5_would_fail_discounted_shortcut_gate():
    """Regression guard for the v6 design review finding."""
    safe_route = EpisodeFixture("safe_succ",      180, True, "goal", 65, 4, 0.005)
    risky_route = EpisodeFixture("shortcut_succ", 120, True, "goal", 65, 4, 0.150)
    r_safe = compute_return("arrival_v2", safe_route, gamma=0.995, w_safety_override=0.5)
    r_risky = compute_return("arrival_v2", risky_route, gamma=0.995, w_safety_override=0.5)
    assert r_safe < r_risky, \
        "this guard should fail if the shortcut fixture no longer catches v5's weak safety"

def test_arrival_v2_fast_optional_ablation_is_safe():
    """Only needed if enabling arrival_v2_fast with R_fast_success=20."""
    safe_route = EpisodeFixture("safe_succ",     180, True, "goal", 65, 4, 0.005)
    risky_route = EpisodeFixture("shortcut_succ", 120, True, "goal", 65, 4, 0.150)
    for gamma in [None, 0.995]:
        r_safe = compute_return("arrival_v2_fast", safe_route, gamma=gamma)
        r_risky = compute_return("arrival_v2_fast", risky_route, gamma=gamma)
        assert r_safe > r_risky, \
            f"unsafe shortcut wins under fast ablation at gamma={gamma}: safe={r_safe}, risky={r_risky}"

def test_d_init_clamp():
    """D_init < D_INIT_MIN must not blow up reward."""
    edge = EpisodeFixture("near_goal_start", 20, True, "goal", 4.0, 0.5, 0.001)
    r = compute_return("arrival_v2", edge)
    assert -1e3 < r < 1e3, f"reward exploded: {r}"
```

注意两点：

1. Discounted tests 不要求完整 OOB 内部排序与 undiscounted tests 完全一致。`gamma=0.995` 会让早期 OOB 的 terminal penalty 在 Q-space 中更大，`fast_oob` 与 `mid_oob` 的相对顺序可能交换；真正必须守住的是 `success > timeout > hard_failure` 与 `fast_oob << timeout`。
2. `success_dominates_failure` 不是数学上对任意 `Σsafety_cost` 的全局命题。若一个“成功”episode 长时间贴着 safety limit 运行，它可以被 soft safety penalty 拉低。v6 的验收语义是：canonical success、realistic unsafe-success fixture 与 discounted unsafe-shortcut fixture 都必须优于 failure / risky shortcut；若项目要求 **任何成功都绝对优于任何失败**，实现中需要新增 success-episode safety penalty cap，并为 cap 写独立测试。

**测试运行频率**：每次 `auv_nav/reward.py` 改动必须通过；CI 强制（pre-commit hook 或 GitHub Action）。

**测试不覆盖的内容**（已知 limitation）：
- 真实 SAC 训练动力学（policy 是否能找到次优解）—— 由 §7.2 P1 regression 覆盖
- 真实流场下的 trajectory 分布 —— 由 cross_u10 behavior regression 覆盖（详见 §8.3）

#### 7.1.1 Trajectory-Level Invariant Test（可选后置）

§7.1 fixture 是第一版必须测试。真实 SAC 探索会产生 fixture 之外的轨迹分布——边界附近 oscillate、绕大弯但成功、卡死在局部 optimum 等。因此 trajectory-level test 有价值，但 v4 将它从第一版 blocking item 降级为后置验证，避免 reward 修复被过度工程化。

**数据源**：`experiments/online_thesis_v1/preflight/p1_budget_calibration_v2_flowfix/efficiency_v2/sac_vanilla/s1_k4/seed_46/logs/train_log.jsonl`（11 253 episode，§3.2 已经用过）。

**步骤**：

```python
def test_arrival_v2_ordering_on_real_trajectories():
    """从真实 train_log 抽 200 条 episode（按 success/failure 分层），
    用 arrival_v2 重算 return，验证：
    (a) 同 success/failure 类别内，return 与 (success, time-to-goal, final_distance)
        的相关性方向正确（spearman ρ）；
    (b) success episode return min > failure episode return max（margin ≥ 50）；
    (c) 没有任何 trajectory 触发 NaN / Inf / blow-up。
    """
    episodes = sample_stratified(
        log_path="experiments/.../train_log.jsonl",
        n_per_class={"success": 100, "failure": 100},
        seed=0,
    )
    returns = [compute_return_from_episode("arrival_v2", ep) for ep in episodes]

    # (a) 单调性
    success = [(r, ep) for r, ep in zip(returns, episodes) if ep.success]
    rho_time = spearmanr([r for r, _ in success], [ep.elapsed_s for _, ep in success]).correlation
    assert rho_time < -0.3, f"success return should anti-correlate with time, got {rho_time}"

    # (b) 类间 margin
    success_min = min(r for r, ep in zip(returns, episodes) if ep.success)
    failure_max = max(r for r, ep in zip(returns, episodes) if not ep.success)
    assert success_min > failure_max + 50, \
        f"class margin too small: success_min={success_min}, failure_max={failure_max}"

    # (c) 数值健全
    assert all(np.isfinite(r) for r in returns), "non-finite return on real trajectory"
    assert max(abs(r) for r in returns) < 1e3, "return blow-up on real trajectory"
```

**为什么需要**：unit test fixture 的 dominance 通过 ≠ 真实 trajectory 分布上 dominance 通过。fixture 假设每个 episode 的 per-step safety 是常数（0.005 / 0.130 / 0.150 / 0.180 / 0.185），真实 episode 的 per-step safety 是 trajectory 而非常数。这条测试用真实数据封住这个缺口。

**实现注意**：efficiency_v2 训练出来的 policy 与 arrival_v2 想训练的 policy 不同分布（前者后期 OOB 自杀，后者期望避免 OOB），所以 success episode 数量较少（约 200 条总 success）。**抽 100 success / 100 failure 已经足够**；如果 success 不足 100 条，按实际数量抽（log 中应有 ≥ 200 条）。

**预期通过判据**（基于 §3.2 实测；v6 candidate no-fast）：
- success return 范围预期约 `[+90, +145]`（time / D_f / safety variation；unsafe-success fixture 会明显低于 low-safety success）
- failure return 范围预期 `[−330, −140]`
- success_min − failure_max ≥ 50 应当成立

如果失败，**优先怀疑 fixture safety 量级与真实分布脱节**，而不是直接调 reward 参数。

### 7.2 P1 Regression Check

重跑 P1 时必须同时看：

**(a) Eval 指标**（每 10k steps）：

```text
eval_success_rate
eval_return
eval_termination_counts
eval_time_s
eval_progress_ratio
eval_path_efficiency
return_success_only
return_failure_only
return_by_reason
final_distance_by_reason
observation_context_enabled
```

**(b) Training dynamics**（建议记录；不作为第一版实现阻塞项）：

```text
critic_loss_mean / critic_loss_max     # reward variance 直接放大 critic loss
q_value_mean / q_value_max / q_value_min  # 跨 episode 的 reward 跨度 ~470，需监控 Q blow-up
alpha (auto-tuned temperature)         # SAC 自动温度对 reward scale 敏感
actor_loss_mean
target_entropy_gap = H(π) − target_H   # SAC entropy 是否锁在 target 附近
```

**判据**：

1. 若 `eval_return` 上升但 `out_of_bounds` 增加，判定为 reward failure。
2. 若 `final` 明显差于 `best_success`，不能用 final checkpoint 代表训练预算。
3. 若 `arrival_v2` 下仍出现 success collapse，需要先检查任务难度、探索和算法稳定性，而不是继续调 efficiency 项。
4. 若 `q_value_max > 500` 或 `critic_loss_max > 100` 持续 ≥ 50k steps，判定为 reward scale 失稳。应对：将 terminal 项整体 ×0.5（不改变排序）后重跑。
5. 若 `alpha` 在前 100k 内飙升到 > 1.0 且持续不降，说明 entropy regularization 可能在和 dense progress reward 打架。应对：先固定 `alpha` 做 50k smoke test，再决定是否改 reward scale。
6. 若 checkpoint metadata 显示 `arrival_v2` 但 `observation_context_enabled=False`，该 run 无效；它没有满足 §4.6 的 MDP state contract。
7. 若在线 SAC replay 仍对 timeout bootstrap（`done=terminated`），该 run 无效；它没有满足 §4.7 的 timeout terminal 语义。

**为什么 critic 监控重要**：arrival_v2 的 terminal 项是 ±100 到 ±200 量级，per-step 项是 ±2 量级，**reward variance 跨度 ~100×**。这会显著放大 TD-target 的方差，潜在让 critic_loss 不稳。SAC 在 D4RL / Adroit 等环境下处理过类似 reward scale，通常 OK，但本环境的 terminal heavy 程度更极端，**早期监控这三件套比等到 eval 失败再 debug 便宜得多**。

### 7.3 Evaluation Summary Extension

建议扩展 eval summary，增加：

```text
eval_return_success
eval_return_failure
eval_return_by_reason
eval_final_distance_m
eval_final_distance_by_reason
eval_terminal_violation_by_reason
eval_timeout_count
eval_hard_failure_count
eval_observation_context_enabled
```

这些指标可以让 reward hacking 在训练中被直接发现，而不是等到画图后才解释。

---

## 8. 对当前论文实验计划的影响

当前 [`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) 中仍把 `efficiency_v2` 作为主线目标函数。基于 P1 证据，建议调整为：

1. Sprint 0 先实现并验证 `arrival_v2`。
2. 用 `arrival_v2` 重跑 P1，重新标定 `single_u15_upstream_tgt15` 的训练预算。
3. §2 sensor envelope 第一轮使用 `arrival_v2`，回答"是否能到达"。
4. **不再追加 efficiency_v3 reward**（v2 修订）；efficiency 改为 eval-time only metric（详见 §5.2）。

在这个调整完成前，不建议把 `efficiency_v2` 下的 P1 final curve 当作正式预算依据。

### 8.1 Pre-Integration Independent Validation Gate（v6 新增）

正式集成 `arrival_v2` 之前必须先过一个**独立有效性检验 gate**。这里的“独立”有两个含义：

1. 不修改 `auv_nav/reward.py`、`auv_nav/env.py`、`scripts/train_sac.py` 等正式路径；
2. 不复用未来要实现的 `RewardModel.compute()` 逻辑，避免“实现和测试复制同一个错误公式”。

推荐新增一个 standalone candidate validator（当前建议路径：[`scripts/validate_arrival_v2_candidate.py`](../scripts/validate_arrival_v2_candidate.py)）。该脚本只包含候选公式、synthetic fixtures 与 pass/fail gate，用于在正式集成前回答：

- v6 参数是否在 undiscounted return 下消除了 `efficiency_v2` 的 fast-failure 激励；
- v6 参数是否在 `gamma=0.995` discounted return 下仍满足 `success > timeout > hard_failure`；
- v6 参数是否修复 v5 暴露出的 discounted unsafe-shortcut，即 `safe_success(180 steps, sft=0.005) > risky_success(120 steps, sft=0.150)`；
- `w_safety ∈ {1.5, 2.0, 2.5, 3.0}` 的 stress scan 是否仍保持 timeout 优于 hard failure，且 `w_safety=0.5` 是否被 validator 明确判为失败。

**Gate A：纯公式 validator（必须先过）**

```bash
conda run -n mytorch1 python -m scripts.validate_arrival_v2_candidate
```

通过条件：

1. default candidate `w_safety=2.0` 全部 synthetic gates 通过；
2. `w_safety=0.5` 在 discounted unsafe-shortcut gate 上失败（证明 gate 能抓住 v5 问题）；
3. 输出的 return table 与 §5.1 / §7.1 量级一致；
4. 无 NaN / Inf / reward blow-up。

**Gate B：真实轨迹 shadow scoring（推荐，若日志/轨迹可用）**

如果本地或 Colab 保留了 P1 v2 episode-level 轨迹或 train_log summary，应在不改 core code 的情况下用同一 candidate formula 重算旧轨迹 return。通过条件：

1. success episode 的 candidate return 显著高于 collapse/OOB episode；
2. near-timeout 高于 fast-OOB suicide；
3. return 与 `success` / `final_distance` / `termination_reason` 的方向一致。

若现有 train_log 缺少逐步距离序列，只能做 summary 近似；该 gate 作为辅助证据，不替代 Gate A 和后续真实训练回归。

**Gate C：隔离原型分支行为回归（通过 Gate A 后再做）**

只有 Gate A 通过后，才允许在一个隔离分支中实现最小 `arrival_v2` 原型并跑 §8.3 的 cross_u10 behavior regression。这个分支仍不视为正式集成；若 cross_u10 失败，应先回到 reward 设计/参数或训练稳定性诊断，而不是继续把修改扩散到 sensor envelope 主实验。

### 8.2 Sprint 0 拆分与时间预算（v6 更新）

原计划的 Sprint 0 包含 P0（profiling）+ P1（u15_upstream 预算标定）+ P2（AsymCritic smoke），预算约 1 个 Colab session（~3h wallclock）。reward redesign 显著超出这个范围，必须拆分：

| Sprint | 内容 | 状态 |
|---|---|---|
| **0a** | P0 profiling + flow_path fix（_v2_flowfix） | ✅ 已完成 |
| **0b**（新增）| reward redesign + arrival_v2 independent gate + isolated prototype + cross_u10 regression + P1 v6 重跑 | ⬜ 进行中 |
| 1 | §2 sensor envelope（18 run） | 推迟到 0b 完成后 |

**Sprint 0b 工作量估算**（v6 重新估算，包含 pre-integration gate 与 MDP/replay/test 补丁）：

| 任务 | 估时 |
|---|---:|
| standalone candidate validator + 文档同步 | 1h |
| `RewardModelConfig` + `RewardModel.compute()` / helper 扩签名 | 1.5h |
| `arrival_v2` preset 写入 `REWARD_OBJECTIVE_PRESETS` | 0.5h |
| `env.step` 透传 `elapsed_time_s / D_init / D_prev / D_curr` | 1h |
| objective-specific observation context + obs dim metadata | 2h |
| `scripts/train_sac.py` timeout done 语义 + `arrival_v2_simple` online guard | 1h |
| `SafetyCostModel` timeout 从 terminal violation 中拆出 | 0.5h |
| `D_init` clamp + edge case test | 0.5h |
| `evaluate.py / collect_offline_data.py / train_offline.py` metadata 兼容检查 | 1.5h |
| §7.1 RewardModel-backed ordering tests（undiscounted + discounted + stress）| 2h |
| §7.3 eval success/failure return + by-reason summary 输出 | 1.5h |
| 现有测试防回归 + lint | 1h |
| **小计 coding** | **~14h（约 2 工作日）** |
| **Colab cross_u10 behavior regression**（1 seed × 600k）| 1.5h wallclock |
| **Colab P1 v6 重跑**（s1 × u15_upstream × 1M）| 2.5h wallclock |
| **plan/doc 更新 + review 周转** | ~0.5 工作日 |
| **总 end-to-end** | **~3 工作日** |

### 8.3 cross_u10 Behavior Regression（v2 新增，必须通过才能进 §2）

unit test 只验证「预设 episode 的 return 排序」，验证不了「真实 SAC 训练在新 reward 下不会崩」。在 fire §2 sensor envelope 之前，必须用一个**已知 vanilla SAC 能学好**的 baseline 任务做行为回归：

| 项 | 值 |
|---|---|
| benchmark | `single_u10_cross_tgt15`（A0 时代 vanilla SAC s0 已达 ~95% success）|
| algo | vanilla SAC（无 LayerNorm / 无 asym / UTD=1）|
| reward | `arrival_v2` |
| sensor | `s0_k4`（最弱 sensor，最严苛测试）|
| seed | 46（单 seed 即可，仅做回归不做统计推断）|
| total_steps | 600 000 |
| eval | `benchmarks/single_u10_cross_tgt15.json`，30 ep |

**通过判据**（必须全部满足才能进 §2）：

1. 训练终点 success_rate ≥ 0.85（A0 efficiency_v2 是 ~0.95；留 10pp 容忍因 reward 改变带来的损失）；
2. 最后 100k 平均 success ≥ 整个训练 peak 的 0.9（无 collapse）；
3. eval termination_counts 中 `n_oob / n_total ≤ 0.10`（不出现 v2 的 OOB-suicide 模式）。

如果 cross_u10 都过不了 → arrival_v2 在更弱的任务上反而 break，说明 reward 设计有未发现的问题，**必须 debug 完才能进 §2**。

选择 vanilla SAC + `s0_k4` 是保守回归：actor/critic 只能看到最弱单点流速历史，不能依赖 AsymCritic 的 privileged flow。若这个设置都能达到 success ≥ 0.85 且不出现 OOB collapse，那么后续 AsymCritic / 更强 sensor 配置更可能受益；若它失败，则应先定位 reward/训练稳定性，而不是把失败归因给 sensor envelope。

L4 wallclock 1.5h，可与 P1 v6（2.5h）背靠背放在 Sprint 0b 的最后一个 Colab session 一起跑。

---

## 9. v1 审查意见（保留作历史）

上一轮分析的核心判断成立：

> P1 中 return 上升、success 降为 0 的原因，是当前 reward 对快速失败的惩罚不足，且每步时间惩罚过强，导致 SAC 后期优化出快速出界策略。

需要补充的严格表述是：问题不是单个参数失衡，而是缺少 terminal dominance 约束。只把 `failure_penalty` 从 `-20` 调到更大，可能缓解当前 case，但不能系统性防止未来在 energy、path、safety 或其他 benchmark 上再次出现类似 reward hacking。

因此，正确修复应是：

1. 重新定义 arrival-first reward；
2. 把 terminal outcome 的排序写成测试；
3. 把 success/failure 分组 return 和 termination counts 纳入标准 eval；
4. 再用新的 reward 重跑在线 SAC 预检。

---

## 10. v2 第二轮校准与修正（2026-04-27）

v1 提出了正确的结构（arrival-first + terminal dominance + ordering test），但**两处参数选择基于经验估算而非真实数据**，被第二轮 expert review 抓出。本节记录修正的来龙去脉，避免未来再次踩同一坑。

### 10.1 不严谨之处与触发原因

| 议题 | v1 取值/表述 | v2 修正后 | 触发原因 |
|---|---|---|---|
| `w_safety` | 1–5 (range) | **0.5** (v2-v5); **2.0** (v6 candidate) | v2 用真实 safety 分布把 v1 range 收缩到 0.5；v6 进一步发现 discounted unsafe-shortcut 要求 `w_safety > 1.46`，因此候选默认值上调到 2.0 |
| `w_progress` | 20 | **50** | v1 让 per-step progress 信号比 efficiency_v2 弱 3.3×（0.23 vs 0.75），早期 SAC cold-start 会显著拖慢；50 把 per-step 拉回 0.58，与 efficiency_v2 同量级 |
| `efficiency_v3` 定位 | 「在 arrival_v2 稳定后追加」（语义模糊）| **eval-time only metric，不进 reward** | v1 表述允许「重新训练 finalist」解读，会破坏 §2/§3+ 跨节 absolute return 可比性 |
| `D_init` 边界 | 未提 | **`max(D_init, 10.0)` clamp** | task sampler 偶尔可能采到 start ≈ goal 的 case，未防御会让单步 progress reward blow up |
| Sprint 0 范围 | 未估时 | **拆 0a/0b，0b ≈ 3 工作日** | reward redesign 远超 1 个 Colab session 的原 Sprint 0 预算 |
| 安全网 | 仅 unit test | **unit test + cross_u10 behavior regression** | unit test 不能验证真实 SAC dynamics；必须有 known-good 任务做行为回归 |

### 10.2 数据校准过程

> 历史说明：本节记录 v2 calibration 过程。`w_safety≈4.5 / [0.5,4.0]` 等边界已被 v3/v4 修正，v6 又用 discounted unsafe-shortcut fixture 修正下限；当前规范以 §3.2 与 §5.1 为准。

校准对象：v2 P1 训练 11 253 episode 的真实 safety_cost 分布（详见 §3.2）。

**关键发现**：

1. **成功 vs 失败 episode 的 safety 差距是 1–2 个数量级**（成功 ~0.001，失败 ~0.13–0.18）。这意味着 `w_safety` 的取值对成功 policy 几乎无副作用，主要塑造失败 policy 的 return → **可以放心提高 w_safety**。

2. **v2 当时估计 dominance flip 阈值为 `w_safety ≈ 4.5`**；v3/v4 用 fixture 一致口径修正为 **3.69**，hard stress 上限取 **3.0**。

3. **v2/v3/v4 的不安全捷径 break-even 没有充分纳入 discounted terminal success**。v4 的 undiscounted no-fast break-even 约 **0.08**；v6 按 `gamma=0.995` 重算后，no-fast discounted break-even 约 **1.46**，`arrival_v2_fast` 约 **1.71**。

4. **当前候选默认值取 `w_safety=2.0`**：保留 timeout-vs-OOB dominance margin，同时修复 v5 的 discounted unsafe-shortcut 漏洞。

### 10.3 推广的方法论教训

这两个不严谨之处的共同根因是：**reward 参数推荐不能只依赖「概念上合理的数量级」，必须用真实 trajectory 数据校准量纲**。

未来任何 reward 修改流程应是：

1. **先看真实数据分布**（per-step safety、progress、energy、… 的实际量级）；
2. **再算 dominance 边界**（每个 reward 项的 hard bounds）；
3. **再选取参数**（在 [lower, upper] 区间内挑值）；
4. **写 unit test fixture 时用真实量级**（不要用乐观估计）；
5. **用 known-good benchmark 行为回归**（unit test 之外的最后保险）。

v1 跳过了 (1) 和 (5)，所以 v2 必须补上。这套流程应当**写进 [`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) 的 reward design 章节**，作为 thesis methodology 的一部分（而不是隐藏的工程经验）。

### 10.4 v6 完成后的下一步

1. ✅ 本文档（v6）落字
2. ⬜ 用户 review v6
3. ⬜ 运行 §8.1 standalone candidate validator；若失败，不进入 core 实现
4. ⬜ 若 validator 通过，在隔离分支实现 `arrival_v2` preset（默认 `R_fast_success=0, w_safety=2.0`）+ 扩 `RewardModel` 签名 + 调用点更新
5. ⬜ 扩展 observation context（`elapsed_time_frac`, `initial_distance_norm`）并更新 metadata / checkpoint 兼容检查
6. ⬜ 修正 online SAC timeout replay done 语义；给 `arrival_v2_simple` 加 online guard
7. ⬜ 将 timeout 从 safety terminal violation 中拆出，hard failure 与 timeout 分支分离
8. ⬜ 写 §7.1 tests（RewardModel-backed ordering / discounted dominance / unsafe-success / discounted unsafe-shortcut / no suicide / `D_init` clamp / `w_safety` stress）
9. ⬜ 扩展 §7.3 eval summary（success/failure return、by-reason return、final distance by reason、terminal violation by reason）
10. ⬜ Colab 跑 cross_u10 behavior regression（1.5h × 单 seed，可选 ×3 seed = 4.5h，参见 §11.5），验证 §8.3 三条判据
11. ⬜ Colab 跑 P1 v6 重跑（2.5h），同时观察 §7.2 eval 指标；training dynamics 监控可后置
12. ⬜ 若 no-fast policy 明显磨蹭，再实现 `arrival_v2_fast` ablation
13. ⬜ 把 P1 v6 结果落到 active plan / report；online thesis 线若仍 shelved，则不要恢复 sensor envelope 矩阵

---

## 11. v3 第三轮校核与修正（2026-04-27）

> 历史说明：本节保留 v3 review 过程。v4 已进一步修正 `R_fast_success` 默认值、`fast_success` fixture 步数、`w_time` 每秒收益，以及 discounted potential shaping 的严格表述。v5 又补齐 MDP state / timeout replay / discounted tests。当前规范以 §1、§3.2、§4.6-§4.8、§5.1、§6、§7.1 为准。

v3 当时的判断是：v2 的 reward 结构与 §5.1 参数取值基本可用，但 v2 在数值计算与不变量描述上有若干不严谨之处。本节记录那轮修正；v4 对第一版落地复杂度又做了进一步收敛。

### 11.1 §5.1 expected return 表的数值错误

v2 §5.1 表中 4 行的 safety penalty 与一个终局组合项算错，根因是 v2 在 §3.2（用 episode-total ep_cost ≈ 21.80）和 §5.1（隐含用 fixture per-step × n_steps）之间用了**不一致的 safety 计算口径**。

**v2 vs v3 数值对照**（按 §7.1 fixture 的 per-step × n_steps 一致重算）：

| episode | v2 给的 return | v3 修正 | 主要差异来源 |
|---|---:|---:|---|
| `fast_success` | +163 | +163.7 | time penalty 应为 −0.6（v2 写成 −0.1 typo） |
| `slow_success` | +153 | +153.9 | basically OK |
| `timeout_near` | −36 | **−56.2** | safety 应为 −31.2（v2 写成 −11，差 20） |
| `timeout_far` | −116 | **−136.2** | safety 应为 −31.2（v2 写成 −11，差 20） |
| `late_OOB` | −163 | **−220.5** | v2 漏算 R_early_failure 或 R_final_distance 一项 |
| `mid_OOB` | −259 | −287.1 | safety 应为 −10.8（v2 −8）+ R_early_failure 应为 −75（v2 −50） |
| `fast_OOB` | −263 | −263.7 | basically OK ✓ |

**含义**：
- 排序 (success ≫ failure) 与 (fast_OOB ≪ timeout) 这两个**关键安全不变量在修正后依然成立**，design 没有破。
- 但 v2 声称的「`fast_OOB` 是绝对最差」**与 fixture 数值不符**——`mid_OOB` 才是绝对最差。这不影响安全性（无 suicide attractor 只需 fast_OOB ≪ timeout），但 §7.1 `test_fast_failure_is_worst` **写错了**，会在 v2 fixture 下直接断言失败。v3 已替换为 `test_no_suicide_attractor`，正确表达约束。
- `w_safety` dominance flip 阈值 v2 估为 4.5，v3 用一致口径解 `0 − 5 − 480·0.13·w − 50 − 50 = 0 − 2.5 − 240·0.15·w − 100 − 50 − 50` 得 **w = 3.69**。Hard upper bound 应取 **3.0**（~20% 余量），v2 取的 4.0 已经超过 flip 点。§7.1 stress test 同步修正。

### 11.2 R_fast_success 是 unsafe-shortcut 安全约束几乎唯一的成因

> v4 修正：本节是 v3 历史分析。`5/240` 应为 0.0208，且若保留 `R_fast_success=20`，时间压力应近似相加为 `(5+20)/240`。当前数值以 §3.2 为准。

v3 的定性结论仍有用：`R_fast_success` 是额外时间压力的主要来源，保留它会显著收紧 safety-weight 下限；因此 v4 将它降级为 optional ablation。

工程含义：若想降低 reward 耦合度，**先移除 `R_fast_success`，保留 `w_time`**。前者只在 success 路径上给学习信号，后者是 dense signal，对早期 critic value bootstrap 更有用。

### 11.3 γ=0.995 折扣对 effective signal 的影响（v2 完全没分析）

`γ=0.995` 在 480-step max episode 里：

| 信号 | 名义量级 | 起始时刻 effective Q 贡献（按平均折扣估计） |
|---|---:|---:|
| `R_success`（terminal）| +100 | `0.995^480 × 100 ≈ 9`（slow_success 240 步时 ≈ 30；fast_success 120 步时 ≈ 55） |
| `R_failure`（terminal）| −100 | `0.995^60 × 100 ≈ 74` (early OOB 时)；`0.995^240 × 100 ≈ 30` (late_oob fixture 步数) |
| `Σ progress`（dense）| +47 | 与 episode 长度强耦合：120 步 fast_success 平均折扣 ≈ 0.75 → **effective ≈ 35**；240 步 ≈ 0.58 → ≈ 27；480 步 timeout 平均折扣 ≈ 0.38 → ≈ 18 |
| `R_early_failure`（terminal）| ≤ 100 | 同 R_failure，早期 OOB 时强 |

**两个 RL 含义**：

1. **冷启动阶段 SAC critic 主要从 dense progress 学方向，而 R_success 要靠多步 Bellman bootstrap 才能传到起始 state**。R_success 的起始 effective 贡献跨场景在 9–55 之间，dense progress 跨场景在 18–35 之间——**真正使 progress 在「冷启动早期」占主导的不是 33 vs 9 那种数量级压制，而是「dense 信号每个 transition 都贡献，sparse terminal 信号要等数百步 backup」**。这就是为什么 §8.3 cross_u10 regression 设 600k 而不是 300k 的 RL 理由——600k 才足够 Bellman backup 把 +100 R_success 完整传到起始 state。
2. **early OOB 的 R_failure 在早期就强**（~74），意味着冷启动阶段 critic 学到的主要 negative signal 是「不要 OOB」，而 positive signal 来自 progress。这是个**有利的归纳偏差**——SAC 先学避免 OOB、再学到达，与论文 narrative「arrival first」也一致。

### 11.4 R_fast_success 是否值得保留：A/B 实验比经验判断更稳

> v4 处理结果：`R_fast_success` 已从默认 `arrival_v2` 移出，保留为 `arrival_v2_fast` optional ablation。本节保留 v3 的判断过程。

v2 §5.2 把 efficiency 推到 eval-only，但 reward 里仍保留 `R_fast_success = 20`。这条「success 越快越好」的梯度对 SAC 在 success 维度的 time-to-goal 学习有 ±7.5 的 Q 差距贡献（见 §5.1 表 fast vs slow）。

**历史 v3 判断**：
- 砍掉后 success 维度内部失去明确梯度，可能让 SAC 收敛到「能到就好，磨蹭也行」的 policy
- 但保留它就要承受额外 unsafe-shortcut 约束；v6 按 discounted fixture 重算后，`arrival_v2_fast` 至少需要 `w_safety > 1.71`（§3.2），不是历史 v3 的 `>0.32`

**v3 推荐方案**：cross_u10 regression 阶段做小型 A/B 实验，多花 1.5h L4 wallclock：
- 历史 v3 主分支：`arrival_v2_fast` with R_fast_success = 20（v6 中仅作为 optional ablation）
- 历史 v3 A/B 分支：`arrival_v2` with R_fast_success = 0（v6 default）

若 A/B 分支 success_rate 与主分支 ≤ 5pp 差距，且 mean_time_to_goal 没有显著恶化（≤ +20%），则可以**继续砍掉 R_fast_success 简化 reward**，同时避免把 discounted shortcut margin 压得太窄。

如果 A/B 分支 mean_time_to_goal 显著恶化（>+20%），则保留 R_fast_success 是有价值的。

### 11.5 cross_u10 single seed 的统计风险

§8.3 行为回归用 single seed (46) 跑 600k。这是 v2 的合理 trade-off（节省 wallclock），但**RL 标准做法是 ≥ 3 seed**——single seed 失败时无法区分「reward 设计差」vs「seed 运气不好」。

**v3 不强制改**，但建议：
- 如果 single seed 通过 §8.3 三条判据（success ≥ 0.85、无 collapse、n_oob/n_total ≤ 0.10），可以直接进 §2
- 如果 single seed **edge-case 通过**（例如 success 在 0.85-0.90 之间，刚刚踩线），强烈建议**多花 3h 补两个 seed**（47, 48），用 3-seed mean ± std 重新判据
- 如果 single seed 失败，**先补 2 个 seed 确认是否一致失败**，再决定是 reward debug 还是其它问题

### 11.6 AsymCritic 与新 reward 的相互作用

这是个 **AsymCritic thesis**，最终 §2 sensor envelope 用 AsymCritic 跑。但 §8.3 cross_u10 用 vanilla SAC——**这是个保守测试**：

- vanilla SAC critic 只能看到 actor 的 single-point 流速观测（s0），对 OOB 风险预测能力有限
- AsymCritic critic 看 hull-integral 流（privileged_obs），能更早识别「这个位置流场推力大，未来 OOB 概率高」，对 `R_failure + R_early_failure` 这套大额惩罚的 credit assignment **更有利**
- 所以「vanilla 能过 0.85 → AsymCritic 应当更好」是一个有合理性的演绎

这条 reasoning 已在 v6 写入 §8.3，作为「为什么 single seed vanilla 可以接受」的辅证，而不是只说「单 seed 仅做回归不做统计推断」（这个理由本身偏弱）。

### 11.7 Reward variance 与 SAC 训练动力学的潜在风险

arrival_v2 的 reward range：
- per-step：±2（progress 主导）
- terminal：±200（success / hard failure）
- **跨 episode variance ~100×**

SAC 对 reward scale 不是不敏感的——TD-target 方差直接决定 critic_loss 大小、间接影响 policy gradient noise。Default `lr=3e-4 / batch_size=256 / grad_clip_norm=10` 在 D4RL 等 ±200 量级 reward 上能跑，但本环境 terminal 占 reward 总量比例更高，**有 nontrivial 的 critic 失稳风险**。

v3 §7.2 加入 `critic_loss / Q-value / alpha` 监控就是为这条风险设保险——失稳征兆出现时（Q blow-up 或 alpha 飙升）有预案：terminal 项整体缩放 ×0.5、或固定 alpha。

### 11.8 v3 没改、v4 部分处理的事

- §3.2 `w_safety` 推导用 efficiency_v2 训练数据估的 fail-policy avg cost。**arrival_v2 训练出的 policy 不一定有同样的 safety 分布**——若 arrival_v2 真的更安全，failure-policy avg safety 会更低，break-even 阈值也会变。第一轮 P1 v6 跑完后应当用新数据重新校准一次（«v2 → v6 流程的二次迭代»，而不是再用 v2 数据）。
- v6 保留 v4 的单一 optional fast ablation，但把主线 `w_safety` 候选值提高到 2.0。若 cross_u10 + P1 v6 都顺利通过，可以考虑把 `arrival_v2` vs `arrival_v2_fast` 做成短附录；不建议再做大规模 `w_safety` sweep，除非 behavior regression 暴露明确问题。

---

## 12. v7 Prototype Validation Results（2026-05-08）

> **状态**：§8.3 cross_u10 behavior regression 与 §8.2 P1 v6 重跑双双通过；arrival_v2 在 §2 P1 证据复核所记录的 efficiency_v2 collapse 上拿到 100% success 修复。本节只报告实测，不修订上文设计。
>
> **作用范围**：单 seed prototype 验证，不替代多 seed thesis-grade statistics。
>
> **不修改顶部 SHELVED 标记**：thesis 矩阵撤销是产品决定（见 [`docs/online_rl_line_summary.md`](online_rl_line_summary.md) §4.4），与此处技术验证结论独立。是否据此重启 online 线由用户决定。
>
> **复现路径**：[`notebooks/sac_arrival_v2_cross_extension_and_p1_v6_completed.ipynb`](../notebooks/sac_arrival_v2_cross_extension_and_p1_v6_completed.ipynb)（已 mount Drive、`!python -u` 实时输出风格）。原始 600k cross_u10 prototype run 的归档：[`notebooks/sac_arrival_v2_cross_u10_regression_completed.ipynb`](../notebooks/sac_arrival_v2_cross_u10_regression_completed.ipynb)（已加 ARCHIVED banner）。

### 12.1 实施跟踪

- arrival_v2 8 参数完整版按 §5.1 v6 spec 在 commit `813096e`（2026-05-07）落地进 `auv_nav/reward.py`，与 `arrival_v2_simple`（commit `bd37412`）非同一物。
- §8.1 Gate A pure-formula validator（`scripts/validate_arrival_v2_candidate`）通过：default `w_safety=2.0` discounted unsafe-shortcut + terminal dominance + OOB ordering 全部成立；`w_safety=0.5` 在 discounted unsafe-shortcut 上被明确判失败（与 v6 设计预言一致）。
- 隔离 prototype 分支 `codex-arrival-v2-prototype` 跑了两组实验：
  - **§8.3 cross_u10 behavior regression**：先按计划跑 600k，未通过 last100k_mean gate（曲线仍在上升），延到 1M 后 5/5 gate 全过；
  - **§8.2 P1 v6 重跑**：从原计划 1M 提到 1.5M（s1 + upstream + 12-D 比 cross + s0 + 10-D 难，留缓冲）。
- 实施期间发现并修了一个 SAC trainer resume 路径上的 silent bug，参见 §12.6。

### 12.2 §8.3 cross_u10 Behavior Regression（PASS）

**配置**：vanilla SAC, `s0_k4`, seed=46, num_envs=6, total_steps=600k → **1M**。eval 25k 一次，30 ep / 次。

**§8.3 三条 gate 实测**（600k 数据来自 [`notebooks/sac_arrival_v2_cross_u10_regression_completed.ipynb`](../notebooks/sac_arrival_v2_cross_u10_regression_completed.ipynb)，1M 数据来自本次 extension notebook）：

| Gate | 阈值 | 600k 首跑 | 1M 续训 | 结论 |
|---|---|---|---|---|
| final_success_rate | ≥ 0.85 | 0.867 PASS（仅 1.7pp 余量） | **1.000** PASS | ✓ |
| last100k_mean / peak | ≥ 0.90 | 0.6533 / 0.867 = 0.754 **FAIL** | 0.9833 / 1.000 = **0.983** PASS | ✓ |
| OOB rate | ≤ 0.10 | 0.033 PASS | **0.000** PASS | ✓ |

外加 v5 / v6 spec 要求的 MDP / replay 语义 check（trainer_state.json）：

| Check | 期望 | 实测 |
|---|---|---|
| `include_episode_context_obs` | True | True ✓ |
| `timeout_bootstrap_semantics` | terminal | terminal ✓ |
| obs_dim (s0 / k4 / arrival_v2) | 48 | 48 ✓ |
| reward_objective | arrival_v2 | arrival_v2 ✓ |

**末段 eval 轨迹**（900k–1M，5 个 eval 点，30 ep / 点）：

| step | success | safety_cost | eval_time_s | progress_ratio | path_efficiency |
|---:|---:|---:|---:|---:|---:|
| 900 000 | 0.967 | 2.41 | 38.8 | 0.898 | 0.867 |
| 925 002 | 0.967 | 1.85 | 39.0 | 0.888 | 0.870 |
| 950 004 | 1.000 | 2.43 | 39.6 | 0.908 | 0.884 |
| 975 000 | 1.000 | 2.60 | 39.9 | 0.908 | 0.877 |
| **1 000 000 (final)** | **1.000** | 1.63 | 38.4 | 0.908 | 0.896 |

**收敛形状**：
- 600k → 1M 续训窗口内 success ∈ [0.867, 1.000] sustained（17 个 eval 点中 14 个 ≥ 0.967），无 collapse；
- `eval_time_s` 从 600k 时 65s 降到 1M 时 39s（× 0.59），策略在 sustained 高 success 同时显著缩短 time-to-goal —— arrival_v2 的 `R_timeout=50` per-step 时间压力按设计奏效；
- `safety_cost` 从 4.67 降到 1.63（× 0.35），`termination={goal: 30}` 全 goal，无 OOB / timeout —— **arrival_v2 在 cross_u10 上不引入 efficiency_v2 的 OOB-suicide 失败模式**。

**结论**：§8.3 PASS。arrival_v2 在「最弱 sensor + vanilla SAC + 已知可学」的保守回归中没有 break，可以作为后续工作的稳态起点。

**预算偏差**：§8.2 给 cross_u10 regression 1.5h L4，实际 ~3h（600k 首跑 + 续 400k 到 1M）。差因是 600k 时未稳态 —— 这是 prototype 实测发现，不是 v6 spec 的 bug；下次类似 prototype 应在 spec 里改成「先按 1M cap 跑完再判，而不是固定 600k」。

### 12.3 §8.2 P1 v6 重跑：efficiency_v2 旧 failure mode 修复（PASS）

**Benchmark**：`single_u15_upstream_tgt15`，即 §2 P1 证据复核所记录的 efficiency_v2 collapse 现场（agent 后期学会快速出界，return 上升但 success → 0）。

**配置**：vanilla SAC, `s1_k4`, seed=46, num_envs=6, total_steps=1.5M。eval 25k 一次，30 ep / 次。

**Final eval (30 ep) 与旧 efficiency_v2 P1 对照**：

| 指标 | efficiency_v2 P1（§2 旧版） | arrival_v2 P1 v6 (1.5M) |
|---|---|---|
| eval_success_rate | **0.0**（collapse） | **1.000** |
| eval_termination_counts | OOB-dominated | `{goal: 30}` |
| eval_return | 上升但与 success 反向 | 143.87 ± 1.78 |
| eval_progress_ratio | n/a（collapse） | 0.9355 |
| eval_path_efficiency | n/a | 0.7787 |
| eval_safety_cost | high（OOB suicide） | **0.292** |
| eval_time_s | n/a | 111.18 |

**学习曲线分阶段**（60 个 eval 点 = 25k → 1.5M，30 ep / 点）：

| 阶段 | env_step 区间 | success 区间 | 现象 |
|---|---|---|---|
| 探索期 | 25k–175k | 0.00–0.50 | safety_cost 20–35；upstream 推进未学到 |
| 爬升期 | 200k–350k | 0.23–0.83 | progress_ratio 上行，safety_cost 下行 |
| 稳定爬升 | 375k–600k | 0.83–0.93 | dominant policy 形成 |
| 首达饱和 | 675k | **1.000** | 首次 30/30 |
| 准稳态 | 675k–1.4M | ≥ 0.967（少数 outlier） | 单点最低 0.767 @ 850k，前后 825k=1.0 / 875k=1.0 → 30-ep eval 噪声 |
| 末段稳态 | 1.4M–1.5M | **全 1.000** | safety_cost ∈ [0.07, 0.50]；progress_ratio 锁定 0.9356；eval_time 锁定 ~111s |

**v6 spec 关键不变量实测**：
- **terminal dominance**：30/30 全 goal，无 timeout / OOB → R_success=100 在 γ=0.995 discounted return 下确实主导（与 §11.3 计算预期一致）；
- **timeout-as-terminal**：`trainer_state.timeout_bootstrap_semantics=terminal`，SAC target 不再 bootstrap 超时 episode（v5 §4.7 / §6 修订生效）；
- **MDP state contract**：`include_episode_context_obs=True` 实测生效（v5 §4.6 / §6 修订生效）；
- **discounted unsafe-shortcut**：末段 safety_cost 长期 < 0.5，与 §8.1 Gate A validator 的 `safe_success > risky_success` 预期一致 —— policy 不学短路绕飞。

**对 §2 P1 证据复核的回应**：
§2 记录的失败模式（"return 上升但 success=0；agent 学会快速出界"）在 arrival_v2 下被消除。P1 v6 1.5M 末段 `eval_return=143.87` 与 `success=1.0 / termination={goal:30}` **方向一致** —— return 与 task 主指标重新对齐，arrival-first reward 设计的核心立论得到证据支持。

**结论**：arrival_v2 在 P1 v6 这个 §2 旧 failure mode 现场实现了彻底修复。这是奖励重设计相对于 efficiency_v2 的关键证据。

### 12.4 与 §8 / §11 设计预言的对照

| §X 预言 | 实测 | 对照 |
|---|---|---|
| §8.1 Gate A 纯公式 validator pass | ✓ default `w_safety=2.0` 全过；`w_safety=0.5` 在 discounted unsafe-shortcut 上 fail | 与 v6 预言一致 |
| §8.3 cross_u10 final ≥ 0.85 | ✓ final=1.000（1M 时） | PASS（600k 阶段刚过阈值，需要续 budget） |
| §8.3 cross_u10 OOB ≤ 0.10 | ✓ OOB=0.000 | PASS |
| §8.3 last100k_mean ≥ 0.9 × peak | ✓ 0.9833 / 1.000 | PASS（说明续训到稳态判据是合理的） |
| §11.7 critic 失稳风险（reward variance × terminal dominance） | 未触发 | 1.5M 训练干净收敛，不需要 reward scale ×0.5 / 固定 alpha 备案 |
| §11.8 「arrival_v2 真的更安全则 failure-policy avg safety 会更低」 | ✓ P1 v6 末段 safety_cost < 0.5，远低于 efficiency_v2 P1 时 fail-policy 的 ~20 | 验证 §11.8 二次校准的方向正确 |
| §11.6 AsymCritic × 新 reward 交互 | 未在本次 prototype 检验（vanilla SAC，无 asym critic） | 留作后续 |

### 12.5 时间预算实测 vs §8.2

| 任务 | §8.2 预算 (L4) | 实测 (L4) | 说明 |
|---|---:|---:|---|
| §8.3 cross_u10 regression | 1.5h（600k） | ~3h（600k 首跑 + 续 400k） | 600k 未稳态，按收敛形状续到 1M |
| §8.2 P1 v6 重跑 | 2.5h（1M） | ~5h（1.5M） | 上行 budget cap，留缓冲 |
| **总 wallclock** | ~4h | ~8h | × 2 |

差因都是「按收敛形状现场判断 budget」而非「先固定 cap」。这两次差距没有触发 §6 / §11 的任何风险，但记录在此供 §8.2 下次 review 参考。

### 12.6 隐藏 SAC trainer bug：resume 路径修复（2026-05-08）

实施 §12.2 / §12.3 期间发现 `scripts/train_sac.py` resume 路径上一个 silent bug 链，**与 reward 设计无关**，但会让任何 `--resume <save_dir> --total-steps <N>` 静默退化为 no-op，所以记录在此：

| Bug | 位置 | 现象 | 修复 |
|---|---|---|---|
| start_step 单位混淆 | `train_sac.py:467` | trainer_state 保存 `env_step` 是 global step，主循环 `range` 把它当 per-env 计数器，resume 后 `range` 为空，循环不进入 | 把 `start_step` 在 range 入口处除以 `num_envs` |
| 空跑仍写 trainer_state | `train_sac.py` 主循环出口 | 上一 bug 让循环空过后，`save_training_state(env_step=total_env_steps)` 仍执行，把 trainer_state.env_step 从 600k 错改成 1M（agent / replay 实际未变） | maybe_resume 后加 early-return：`if start_step >= total_env_steps: return` |
| checkpoint_dir 路径累积错算 | `train_sac.py:164` | trainer_state 里 `checkpoint_dir` 是 save_dir 相对路径，但 line 409 当 cwd 相对路径解释，每次 resume `../` 数翻倍（7 → 13 → 19） | resume 时把相对路径用 `args.resume` 解析成绝对路径再写回 args |

三个修复合计 17 行 diff，与 reward 设计正交。建议未来加一个 mini regression test（fresh 1k → resume 续到 2k，断言 `trainer_state.env_step` 真的推进且 agent path 可解析），但不阻塞本节结论。

### 12.7 后续可选工作（不阻塞 SHELVED 状态）

prototype 验证仅证明 arrival_v2「在 online 单 seed 上 work」。若 thesis 重启或 offline 线决定升级 reward preset，可基于本节做下一步：

1. **多 seed 复现**（thesis statistics）：cross 1M × 4 seeds ≈ 4h L4；P1 v6 1.5M × 4 seeds ≈ 24h L4。
2. **横向 benchmark 扩展**：`tandem_u15_upstream_tgt15` / `sbs_u15_upstream_tgt15` 的同 reward 验证。
3. **`arrival_v2` vs `arrival_v2_fast` 短附录**（§11.8 留口）：`R_fast_success=20` 看 time-to-goal 在 P1 v6 上的边际改进。
4. **`w_safety` 二次校准**（§11.8）：用 P1 v6 的 actual failure-policy safety 分布重算 break-even 阈值，看 v6 候选 2.0 是否需要降到 1.5 或保持。
5. **`scripts/train_sac.py` resume regression test**（§12.6 留口）。

是否启动以上任何一项，由 [`docs/online_rl_line_summary.md`](online_rl_line_summary.md) 的产品决策驱动，**不应自动从 §12 PASS 跳到展开**。
