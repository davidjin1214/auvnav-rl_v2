# Online SAC Reward Redesign Note【SHELVED】

> **⚠️ 此设计稿已搁置（2026-05-06）。**
> **收口报告见 [`docs/online_rl_line_summary.md`](online_rl_line_summary.md) §4.4** —— Online 线 thesis 矩阵已撤销，本设计稿对应的 `arrival_v2` reward（8 参数完整版）**不在 online 线落地**。
> 本文件保留作为设计档案；如未来 online 线重启或 offline 线需要 arrival-first reward preset 升级，可直接参考此处 v4 规范。
> 注意：[`auv_nav/reward.py`](../auv_nav/reward.py) 中已有的 `arrival_v2_simple` preset（commit `bd37412`）**与本文件 v4 的 `arrival_v2` 不是同一物**——前者只复用现有 RewardModelConfig 字段做 preset 重组，未实施本文件 §5.1 的 8 参数 / §7 的 terminal dominance unit test。
>
> ---
>
> 文档版本：2026-04-27（v4）  
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
>   (d) 补充 potential-based shaping 性质（§4.6）、R_fast_success 是时间压力主导项的归因（§3.2 末）、γ=0.995 折扣对 effective signal 的影响（§11.3）；
>   (e) §7.1 新增 trajectory-level fixture（从真实 train_log 抽 200 条），§7.2 新增 critic_loss / Q-value / alpha 监控。
>   v3 不改变 v2 §5.1 的最终参数取值，只修正数值与边界、补充 RL 理论分析、收紧测试覆盖。
> - **v4 (2026-04-27)**：经第四轮 RL + robotics review，确认 v3 的主方向正确但第一版落地偏复杂。v4 做四个收敛：
>   (a) `R_fast_success` 从主线默认项降级为 optional ablation（默认 `0`，可用 `arrival_v2_fast` 设为 `20`）；
>   (b) 修正 §4.6：在 `gamma=0.995` 下，`ΔPhi` 不是严格 policy-invariant potential shaping，严格形式应为 `gamma Phi(s') - Phi(s)`；
>   (c) 统一 §5.1 / §7.1 fixture 步数与 expected return，修正 `fast_success` 数值和 OOB 排序；
>   (d) 明确 `HARD_FAILURE_REASONS = SAFETY_FAILURE_REASONS - {"timeout"}`，timeout 必须走单独分支。

---

## 1. 结论

P1 的异常不是偶然噪声，而是当前 `efficiency_v2` 奖励与任务主指标不完全对齐导致的退化策略：

> agent 后期学会了快速出界，从而减少每步时间惩罚，使平均 return 上升；但任务成功率降为 0。

因此，`efficiency_v2` 不应继续作为 `single_u15_upstream_tgt15` 在线 SAC 主实验的默认目标函数。下一步应新增一个 arrival-first 的 reward preset，用它重跑 P1，再进入 sensor envelope 主实验。

推荐路线（v4 当前规范）：

1. 新增 `arrival_v2`：以到达为绝对主目标，使用归一化 progress、温和时间惩罚、强 terminal dominance。
2. `arrival_v2` 默认不加入 `R_fast_success`；若需要比较 time-to-goal preference，单独新增 `arrival_v2_fast` 做 ablation。
3. 不新增 training-time `efficiency_v3` reward；energy / path / soft safety 全部作为 eval-only metrics。
4. 给 reward 加最小不变量测试，确保快速失败不会再比接近目标或成功 episode 获得更高 return。

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

**速度/安全博弈点（v4 修正）**：

v3 对时间压力的拆分有两个问题：

1. `w_time = 5` 的每秒收益应为 `5 / 240 ≈ 0.0208 reward/sec`，不是 0.010。
2. 若保留 `R_fast_success = 20`，unsafe-shortcut 的时间收益应近似相加：`(5 + 20) / 240 ≈ 0.104 reward/sec`。

走危险捷径每秒成本仍可粗略写成：

```text
failure-policy avg safety × 2 step/sec × w_safety
≈ 0.13 × 2 × w_safety
= 0.26 × w_safety reward/sec
```

因此：

| reward variant | time pressure | break-even `w_safety` | implication |
|---|---:|---:|---|
| no fast-success bonus | `5 / 240 = 0.0208` | `> 0.08` | `w_safety=0.5` 有充足 margin |
| with `R_fast_success=20` | `(5+20)/240 = 0.104` | `> 0.40` | `w_safety=0.5` 仍可行，但 margin 明显变窄 |

**v4 结论**：主线 `arrival_v2` 默认不使用 `R_fast_success`。理由不是“速度不重要”，而是 SAC 的折扣因子已经偏好更早成功，且 efficiency/time-to-goal 在本 thesis 主线中应作为 eval-only metric。若实验显示无 fast bonus 的 policy 明显磨蹭，再用 `arrival_v2_fast`（`R_fast_success=20`）做受控 ablation。

`w_safety=0.5` 保留为第一版固定值。它高于 no-fast 版本的 unsafe-shortcut 下限约 6×，同时远低于 v3 修正后的 dominance flip 阈值 `3.69`。因此 v4 不再把 `w_safety` 当作 sweep 维度；stress test 只验证 `[0.5, 3.0]` 内 ordering 不翻转。

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
progress_delta_norm = (previous_distance - current_distance) / initial_distance
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

### 4.6 Potential-Like Progress Shaping（v4 修正）

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
      r -= R_final_distance * clipped(final_distance / initial_distance, 0, 2)

  elif reason == "timeout":
      r -= R_timeout
      r -= R_final_distance * clipped(final_distance / initial_distance, 0, 2)
```

实现中必须显式定义：

```python
HARD_FAILURE_REASONS = SAFETY_FAILURE_REASONS - {"timeout"}
```

当前 `SAFETY_FAILURE_REASONS` 包含 `"timeout"`，不能直接复用，否则 timeout 会错误进入 hard-failure 分支。

初始参数建议（v4 当前规范）：

| parameter | v4 default | optional fast ablation | rationale |
|---|---:|---:|---|
| `w_progress` | **50** | 50 | 成功 episode 总 progress reward 约 50，为 `R_success` 的一半；dense signal 与旧 objective 同量级。 |
| `w_time` | **5** | 5 | 完整 episode 时间项约 -5，只提供温和时间压力，不主导终局。 |
| `w_safety` | **0.5** | 0.5 | 高于 no-fast unsafe-shortcut break-even 约 6×，且远低于 dominance flip threshold 3.69。 |
| `R_success` | **100** | 100 | 主任务语义。 |
| `R_fast_success` | **0** | 20 | 默认不进入主线；如需显式 time-to-goal training pressure，用 `arrival_v2_fast` 单独比较。 |
| `R_failure` | **100** | 100 | hard failure 基础重罚。 |
| `R_early_failure` | **100** | 100 | 阻止 fast-OOB 套利。 |
| `R_timeout` | **50** | 50 | timeout 应差于成功，但优于 hard failure。 |
| `R_final_distance` | **50** | 50 | 区分 timeout/failure 时是否接近目标。 |

**Hard parameter bounds（写入 §7.1 unit test stress 子测试）**：

- `w_safety < 3.0`（dominance flip threshold = **3.69**，留 ~20% 安全余量）— v2 估的 4.5 是用近似算法（peak vs collapse episode total）得到，用 fixture 一致重算后下调，详见 §11.1
- no-fast default 下 unsafe-shortcut toy bound 约 `w_safety > 0.08`；`w_safety=0.5` 有充分余量
- `arrival_v2_fast` 若启用 `R_fast_success=20`，unsafe-shortcut toy bound 约 `w_safety > 0.40`，必须单独通过 ordering / behavior regression
- `w_progress < R_success`（terminal dominance，Σprogress 不能超过 R_success=100）

**预期 episode return 排序**（v4 默认 `R_fast_success=0`；与 §7.1 fixture 步数一致）：

```text
fast_success      (120 步, D_f/D_i=0.06, sft 0.005)  ≈ +46.9 − 1.25 − 0.30 + 100                 = +145.4
slow_success      (240 步, D_f/D_i=0.06, sft 0.005)  ≈ +46.9 − 2.50 − 0.60 + 100                 = +143.8
timeout_near_goal (480 步, D_f/D_i=0.20, sft 0.130)  ≈ +40.0 − 5.00 − 31.2 − 50 − 10             =  −56.2
timeout_far_goal  (480 步, D_f/D_i=1.00, sft 0.130)  ≈   0.0 − 5.00 − 31.2 − 50 − 50             = −136.2
late_OOB          (240 步, D_f/D_i=1.00, sft 0.150)  ≈   0.0 − 2.50 − 18.0 − 100 − 50 − 50       = −220.5
fast_OOB          ( 60 步, D_f/D_i=1.20, sft 0.185)  ≈ −10.0 − 0.63 − 5.55 − 100 − 87.5 − 60     = −263.7
mid_OOB           (120 步, D_f/D_i=1.50, sft 0.180)  ≈ −25.0 − 1.25 − 10.8 − 100 − 75 − 75       = −287.1
```

每行的项依次为：`w_progress×Σprogress_norm`、`w_time×Σ(dt/T_max)`、`w_safety×Σsafety`（用 fixture per-step × n_steps）、`R_success/R_failure/R_timeout`、`R_early_failure`、`R_final_distance × D_f/D_i`。

> **v2 → v3 数值差异**：v2 表 timeout/OOB 行的 safety penalty 普遍低估 2-3×，且 `late_OOB` 漏算 R_early_failure 或 R_final_distance 一项。详见 §11.1。
>
> **v3 → v4 数值差异**：v3 表中 `fast_success` 写成 60 步，而 §7.1 fixture 是 120 步；v4 统一为 120 步。同时默认 `R_fast_success=0`，所以 success return 下降约 10-17.5，但 success-vs-failure margin 仍然充足。

**关键不变量**（每次调参都要用 §7.1 测试验证；v4 当前措辞）：

- 排序：`fast_success > slow_success > timeout_near > timeout_far > late_OOB > fast_OOB > mid_OOB` ✓
- **任何 failure mode 比任何 success mode 差至少 100**（default fixture 下约 200）✓
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
current_distance_to_goal_m
initial_distance_to_goal_m
dt
```

推荐做法：

1. 扩展 `RewardModelConfig`，加入 `max_episode_time_s` 和 `terminal` 系列参数。
2. 扩展 `RewardModel.compute()` 参数，使 reward model 自己完成归一化，而不是把逻辑散落在 `env.step()` 中。
3. 保留 `arrival_v1 / efficiency_v1 / efficiency_v2` 作为历史 preset，新增 `arrival_v2`；可选新增 `arrival_v2_fast` 只用于 ablation。
4. `efficiency_v3` 不作为 reward preset，仅作为 eval summary 字段，详见 §5.2。
5. 在 `scripts/train_sac.py` 的 CLI 层保持 `--objective` 接口不变。
6. 新增 hard-failure 集合时不要复用包含 timeout 的 `SAFETY_FAILURE_REASONS`：

```python
HARD_FAILURE_REASONS = SAFETY_FAILURE_REASONS - {"timeout"}
```

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

需要同步更新所有调用 `RewardModel.compute()` 的地方，传入新签名所需的 `elapsed_time_s / max_episode_time_s / current_distance_to_goal_m / initial_distance_to_goal_m / dt`：

- `auv_nav/env.py` 的 `step()` 主路径
- `scripts/train_utils.py` 的 eval rollout（间接通过 env）
- `scripts/collect_offline_data.py` 的离线数据收集（用于 RLPD / offline RL 数据）
- 测试 fixture（参见 §7.1）

env 已经有 `self.initial_distance` 与 `self.last_distance`，env 直接传入即可，不需要改 env state。

---

## 7. 必须新增的测试

### 7.1 Reward Ordering Unit Test（v4 最小必需测试）

构造 synthetic episode summary，直接测试累计 return 排序：

```text
fast_success
  > slow_success
  > timeout_near_goal
  > timeout_far_goal
  > late_out_of_bounds
  > fast_out_of_bounds
  > mid_out_of_bounds
```

这类测试不依赖 PyTorch，也不依赖真实流场。

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
    #               progress_sum_norm = (D_init - D_final) / D_init
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
]
```

**Test cases**：

```python
def test_arrival_v2_terminal_dominance():
    """Asserts the canonical ordering at v4 default params."""
    returns = {f.name: compute_return("arrival_v2", f) for f in FIXTURES}
    expected_order = [
        "fast_success", "slow_success",
        "timeout_near", "timeout_far",
        "late_oob", "fast_oob", "mid_oob",
    ]
    actual = sorted(returns, key=returns.get, reverse=True)
    assert actual == expected_order, f"order broken: {returns}"

def test_success_dominates_failure():
    """Any success > any failure (margin ≥ 100)."""
    returns = {f.name: compute_return("arrival_v2", f) for f in FIXTURES}
    successes = [v for k,v in returns.items() if "success" in k]
    failures  = [v for k,v in returns.items() if "success" not in k]
    assert min(successes) > max(failures) + 100

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
    """Stress: w_safety up to 3.0 must preserve ordering. >3.69 will flip.
    v3 修正：v2 上限 4.0 是用 episode-total 法估算，按 fixture per-step × n_steps
    一致重算，flip 阈值是 3.69，hard upper bound 取 3.0（~20% 余量）。"""
    for w in [0.5, 1.0, 2.0, 3.0]:
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
    With v4 default R_fast_success=0, w_safety=0.5 should make the safe
    route better despite the risky route's shorter duration."""
    safe_route = EpisodeFixture("safe_succ",     180, True, "goal", 65, 4, 0.005)
    risky_route = EpisodeFixture("shortcut_succ", 120, True, "goal", 65, 4, 0.150)
    r_safe = compute_return("arrival_v2", safe_route)
    r_risky = compute_return("arrival_v2", risky_route)
    assert r_safe > r_risky, f"unsafe shortcut wins: safe={r_safe}, risky={r_risky}"

def test_arrival_v2_fast_optional_ablation_is_safe():
    """Only needed if enabling arrival_v2_fast with R_fast_success=20."""
    safe_route = EpisodeFixture("safe_succ",     180, True, "goal", 65, 4, 0.005)
    risky_route = EpisodeFixture("shortcut_succ", 120, True, "goal", 65, 4, 0.150)
    r_safe = compute_return("arrival_v2_fast", safe_route)
    r_risky = compute_return("arrival_v2_fast", risky_route)
    assert r_safe > r_risky, f"unsafe shortcut wins under fast ablation: safe={r_safe}, risky={r_risky}"

def test_d_init_clamp():
    """D_init < D_INIT_MIN must not blow up reward."""
    edge = EpisodeFixture("near_goal_start", 20, True, "goal", 4.0, 0.5, 0.001)
    r = compute_return("arrival_v2", edge)
    assert -1e3 < r < 1e3, f"reward exploded: {r}"
```

**测试运行频率**：每次 `auv_nav/reward.py` 改动必须通过；CI 强制（pre-commit hook 或 GitHub Action）。

**测试不覆盖的内容**（已知 limitation）：
- 真实 SAC 训练动力学（policy 是否能找到次优解）—— 由 §7.2 P1 regression 覆盖
- 真实流场下的 trajectory 分布 —— 由 cross_u10 behavior regression 覆盖（详见 §8.2）

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

**预期通过判据**（基于 §3.2 实测；v4 default no-fast）：
- success return 范围预期约 `[+135, +150]`（time / D_f / safety variation）
- failure return 范围预期 `[−300, −80]`
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

**为什么 critic 监控重要**：arrival_v2 的 terminal 项是 ±100 到 ±200 量级，per-step 项是 ±2 量级，**reward variance 跨度 ~100×**。这会显著放大 TD-target 的方差，潜在让 critic_loss 不稳。SAC 在 D4RL / Adroit 等环境下处理过类似 reward scale，通常 OK，但本环境的 terminal heavy 程度更极端，**早期监控这三件套比等到 eval 失败再 debug 便宜得多**。

### 7.3 Evaluation Summary Extension

建议扩展 eval summary，增加：

```text
eval_return_success
eval_return_failure
eval_return_by_reason
eval_final_distance_m
eval_final_distance_by_reason
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

### 8.1 Sprint 0 拆分与时间预算（v2 新增）

原计划的 Sprint 0 包含 P0（profiling）+ P1（u15_upstream 预算标定）+ P2（AsymCritic smoke），预算约 1 个 Colab session（~3h wallclock）。reward redesign 显著超出这个范围，必须拆分：

| Sprint | 内容 | 状态 |
|---|---|---|
| **0a** | P0 profiling + flow_path fix（_v2_flowfix） | ✅ 已完成 |
| **0b**（新增）| reward redesign + arrival_v2 实现 + cross_u10 regression + P1 v4 重跑 | ⬜ 进行中 |
| 1 | §2 sensor envelope（18 run） | 推迟到 0b 完成后 |

**Sprint 0b 工作量估算**（基于 v2 §6.1 列出的所有调用点）：

| 任务 | 估时 |
|---|---:|
| `RewardModelConfig` + `RewardModel.compute()` 扩签名 | 1h |
| `arrival_v2` preset 写入 `REWARD_OBJECTIVE_PRESETS` | 0.5h |
| `env.step` 透传 `elapsed_time_s / D_init / D_curr` | 1h |
| `D_init` clamp + edge case test | 0.5h |
| `scripts/train_sac.py / evaluate.py / collect_offline_data.py` 调用点更新 | 2h |
| §7.1 最小 reward ordering tests（含 stress test）| 1.5h |
| §7.3 eval success/failure return + final_distance_by_reason 输出 | 1.5h |
| 现有测试防回归 + lint | 1h |
| **小计 coding** | **~8h（约 1 工作日）** |
| **Colab cross_u10 behavior regression**（1 seed × 600k）| 1.5h wallclock |
| **Colab P1 v4 重跑**（s1 × u15_upstream × 1M）| 2.5h wallclock |
| **plan/doc 更新 + review 周转** | ~0.5 工作日 |
| **总 end-to-end** | **~2 工作日** |

### 8.2 cross_u10 Behavior Regression（v2 新增，必须通过才能进 §2）

unit test 只验证「预设 episode 的 return 排序」，验证不了「真实 SAC 训练在新 reward 下不会崩」。在 fire §2 sensor envelope 之前，必须用一个**已知 vanilla SAC 能学好**的 baseline 任务做行为回归：

| 项 | 值 |
|---|---|
| benchmark | `single_u10_upstream_tgt15`（A0 时代 vanilla SAC s0 已达 ~95% success）|
| algo | vanilla SAC（无 LayerNorm / 无 asym / UTD=1）|
| reward | `arrival_v2` |
| sensor | `s0_k4`（最弱 sensor，最严苛测试）|
| seed | 46（单 seed 即可，仅做回归不做统计推断）|
| total_steps | 600 000 |
| eval | `benchmarks/single_u10_upstream_tgt15.json`，30 ep |

**通过判据**（必须全部满足才能进 §2）：

1. 训练终点 success_rate ≥ 0.85（A0 efficiency_v2 是 ~0.95；留 10pp 容忍因 reward 改变带来的损失）；
2. 最后 100k 平均 success ≥ 整个训练 peak 的 0.9（无 collapse）；
3. eval termination_counts 中 `n_oob / n_total ≤ 0.10`（不出现 v2 的 OOB-suicide 模式）。

如果 cross_u10 都过不了 → arrival_v2 在更弱的任务上反而 break，说明 reward 设计有未发现的问题，**必须 debug 完才能进 §2**。

L4 wallclock 1.5h，可与 P1 v4（2.5h）背靠背放在 Sprint 0b 的最后一个 Colab session 一起跑。

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
| `w_safety` | 1–5 (range) | **0.5** (固定常数) | v1 默认 per-step safety ≈ 0.05；§3.2 实测 failure-policy 是 0.13–0.18（高 2.6–3.6×），v1 的 1.0 在真实数据上效果接近 v1 的 2.0，已偏严格 |
| `w_progress` | 20 | **50** | v1 让 per-step progress 信号比 efficiency_v2 弱 3.3×（0.23 vs 0.75），早期 SAC cold-start 会显著拖慢；50 把 per-step 拉回 0.58，与 efficiency_v2 同量级 |
| `efficiency_v3` 定位 | 「在 arrival_v2 稳定后追加」（语义模糊）| **eval-time only metric，不进 reward** | v1 表述允许「重新训练 finalist」解读，会破坏 §2/§3+ 跨节 absolute return 可比性 |
| `D_init` 边界 | 未提 | **`max(D_init, 10.0)` clamp** | task sampler 偶尔可能采到 start ≈ goal 的 case，未防御会让单步 progress reward blow up |
| Sprint 0 范围 | 未估时 | **拆 0a/0b，0b ≈ 3 工作日** | reward redesign 远超 1 个 Colab session 的原 Sprint 0 预算 |
| 安全网 | 仅 unit test | **unit test + cross_u10 behavior regression** | unit test 不能验证真实 SAC dynamics；必须有 known-good 任务做行为回归 |

### 10.2 数据校准过程

> 历史说明：本节记录 v2 calibration 过程。`w_safety≈4.5 / [0.5,4.0]` 等边界已被 v3/v4 修正；当前规范以 §3.2 与 §5.1 为准。

校准对象：v2 P1 训练 11 253 episode 的真实 safety_cost 分布（详见 §3.2）。

**关键发现**：

1. **成功 vs 失败 episode 的 safety 差距是 1–2 个数量级**（成功 ~0.001，失败 ~0.13–0.18）。这意味着 `w_safety` 的取值对成功 policy 几乎无副作用，主要塑造失败 policy 的 return → **可以放心提高 w_safety**。

2. **v2 当时估计 dominance flip 阈值为 `w_safety ≈ 4.5`**；v3/v4 用 fixture 一致口径修正为 **3.69**，hard stress 上限取 **3.0**。

3. **v2/v3 的不安全捷径 break-even 是基于 `R_fast_success = 20` 的历史分析**。v4 默认 `R_fast_success=0` 后，no-fast break-even 约为 **0.08**；若启用 `arrival_v2_fast`，约为 **0.40**。

4. **当前默认值仍取 `w_safety=0.5`**：保留对成功 policy 的 minimal interference，同时与 no-fast 主线有充足 unsafe-shortcut margin。

### 10.3 推广的方法论教训

这两个不严谨之处的共同根因是：**reward 参数推荐不能只依赖「概念上合理的数量级」，必须用真实 trajectory 数据校准量纲**。

未来任何 reward 修改流程应是：

1. **先看真实数据分布**（per-step safety、progress、energy、… 的实际量级）；
2. **再算 dominance 边界**（每个 reward 项的 hard bounds）；
3. **再选取参数**（在 [lower, upper] 区间内挑值）；
4. **写 unit test fixture 时用真实量级**（不要用乐观估计）；
5. **用 known-good benchmark 行为回归**（unit test 之外的最后保险）。

v1 跳过了 (1) 和 (5)，所以 v2 必须补上。这套流程应当**写进 [`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) 的 reward design 章节**，作为 thesis methodology 的一部分（而不是隐藏的工程经验）。

### 10.4 v4 完成后的下一步

1. ✅ 本文档（v4）落字
2. ⬜ 用户 review v4
3. ⬜ 实现 `arrival_v2` preset（默认 `R_fast_success=0`）+ 扩 `RewardModel` 签名 + 调用点更新
4. ⬜ 写 §7.1 最小 unit tests（ordering / success dominance / no suicide / `D_init` clamp / `w_safety` stress）
5. ⬜ 扩展 §7.3 eval summary（success/failure return、by-reason return、final distance by reason）
6. ⬜ Colab 跑 cross_u10 behavior regression（1.5h × 单 seed，可选 ×3 seed = 4.5h，参见 §11.5），验证 §8.2 三条判据
7. ⬜ Colab 跑 P1 v4 重跑（2.5h），同时观察 §7.2 eval 指标；training dynamics 监控可后置
8. ⬜ 若 no-fast policy 明显磨蹭，再实现 `arrival_v2_fast` ablation
9. ⬜ 把 P1 v4 结果落到 [`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md) §10 的新 entry
10. ⬜ Sprint 1 §2 sensor envelope（18 run）才能 fire

---

## 11. v3 第三轮校核与修正（2026-04-27）

> 历史说明：本节保留 v3 review 过程。v4 已进一步修正 `R_fast_success` 默认值、`fast_success` fixture 步数、`w_time` 每秒收益，以及 discounted potential shaping 的严格表述。当前规范以 §1、§3.2、§4.6、§5.1、§7.1 为准。

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

1. **冷启动阶段 SAC critic 主要从 dense progress 学方向，而 R_success 要靠多步 Bellman bootstrap 才能传到起始 state**。R_success 的起始 effective 贡献跨场景在 9–55 之间，dense progress 跨场景在 18–35 之间——**真正使 progress 在「冷启动早期」占主导的不是 33 vs 9 那种数量级压制，而是「dense 信号每个 transition 都贡献，sparse terminal 信号要等数百步 backup」**。这就是为什么 §8.2 cross_u10 regression 设 600k 而不是 300k 的 RL 理由——600k 才足够 Bellman backup 把 +100 R_success 完整传到起始 state。
2. **early OOB 的 R_failure 在早期就强**（~74），意味着冷启动阶段 critic 学到的主要 negative signal 是「不要 OOB」，而 positive signal 来自 progress。这是个**有利的归纳偏差**——SAC 先学避免 OOB、再学到达，与论文 narrative「arrival first」也一致。

### 11.4 R_fast_success 是否值得保留：A/B 实验比经验判断更稳

> v4 处理结果：`R_fast_success` 已从默认 `arrival_v2` 移出，保留为 `arrival_v2_fast` optional ablation。本节保留 v3 的判断过程。

v2 §5.2 把 efficiency 推到 eval-only，但 reward 里仍保留 `R_fast_success = 20`。这条「success 越快越好」的梯度对 SAC 在 success 维度的 time-to-goal 学习有 ±7.5 的 Q 差距贡献（见 §5.1 表 fast vs slow）。

**不主张直接砍掉**，因为：
- 砍掉后 success 维度内部失去明确梯度，可能让 SAC 收敛到「能到就好，磨蹭也行」的 policy
- 但保留它就要承受 `w_safety > 0.32` 的 break-even 约束（§3.2）

**v3 推荐方案**：cross_u10 regression 阶段做小型 A/B 实验，多花 1.5h L4 wallclock：
- 主分支：`arrival_v2` with R_fast_success = 20（默认）
- A/B 分支：`arrival_v2_no_fast` with R_fast_success = 0

若 A/B 分支 success_rate 与主分支 ≤ 5pp 差距，且 mean_time_to_goal 没有显著恶化（≤ +20%），则可以**砍掉 R_fast_success 简化 reward**，同时 hard lower bound `w_safety > 0.32` 也消失，参数空间从 9 降到 8、安全约束从 2 个降到 1 个。

如果 A/B 分支 mean_time_to_goal 显著恶化（>+20%），则保留 R_fast_success 是有价值的。

### 11.5 cross_u10 single seed 的统计风险

§8.2 行为回归用 single seed (46) 跑 600k。这是 v2 的合理 trade-off（节省 wallclock），但**RL 标准做法是 ≥ 3 seed**——single seed 失败时无法区分「reward 设计差」vs「seed 运气不好」。

**v3 不强制改**，但建议：
- 如果 single seed 通过 §8.2 三条判据（success ≥ 0.85、无 collapse、n_oob/n_total ≤ 0.10），可以直接进 §2
- 如果 single seed **edge-case 通过**（例如 success 在 0.85-0.90 之间，刚刚踩线），强烈建议**多花 3h 补两个 seed**（47, 48），用 3-seed mean ± std 重新判据
- 如果 single seed 失败，**先补 2 个 seed 确认是否一致失败**，再决定是 reward debug 还是其它问题

### 11.6 AsymCritic 与新 reward 的相互作用

这是个 **AsymCritic thesis**，最终 §2 sensor envelope 用 AsymCritic 跑。但 §8.2 cross_u10 用 vanilla SAC——**这是个保守测试**：

- vanilla SAC critic 只能看到 actor 的 single-point 流速观测（s0），对 OOB 风险预测能力有限
- AsymCritic critic 看 hull-integral 流（privileged_obs），能更早识别「这个位置流场推力大，未来 OOB 概率高」，对 `R_failure + R_early_failure` 这套大额惩罚的 credit assignment **更有利**
- 所以「vanilla 能过 0.85 → AsymCritic 应当更好」是一个有合理性的演绎

这条 reasoning 应当**显式写入 §8.2**作为「为什么 single seed vanilla 可以接受」的辅证，而不是只说「单 seed 仅做回归不做统计推断」（这个理由本身偏弱）。v3 此处只在本节记录，§8.2 文本暂不改，留给下次评审决定。

### 11.7 Reward variance 与 SAC 训练动力学的潜在风险

arrival_v2 的 reward range：
- per-step：±2（progress 主导）
- terminal：±200（success / hard failure）
- **跨 episode variance ~100×**

SAC 对 reward scale 不是不敏感的——TD-target 方差直接决定 critic_loss 大小、间接影响 policy gradient noise。Default `lr=3e-4 / batch_size=256 / grad_clip_norm=10` 在 D4RL 等 ±200 量级 reward 上能跑，但本环境 terminal 占 reward 总量比例更高，**有 nontrivial 的 critic 失稳风险**。

v3 §7.2 加入 `critic_loss / Q-value / alpha` 监控就是为这条风险设保险——失稳征兆出现时（Q blow-up 或 alpha 飙升）有预案：terminal 项整体缩放 ×0.5、或固定 alpha。

### 11.8 v3 没改、v4 部分处理的事

- §3.2 `w_safety` 推导用 efficiency_v2 训练数据估的 fail-policy avg cost。**arrival_v2 训练出的 policy 不一定有同样的 safety 分布**——若 arrival_v2 真的更安全，failure-policy avg safety 会更低，break-even 阈值也会变。第一轮 P1 v4 跑完后应当用新数据重新校准一次（«v2 → v4 流程的二次迭代»，而不是再用 v2 数据）。
- v4 已将 9 参数体系收敛为固定比例 + 单一 optional fast ablation。若 cross_u10 + P1 v4 都顺利通过，可以考虑把 `arrival_v2` vs `arrival_v2_fast` 做成短附录；不建议再做大规模 `w_safety` sweep，除非 behavior regression 暴露明确问题。
