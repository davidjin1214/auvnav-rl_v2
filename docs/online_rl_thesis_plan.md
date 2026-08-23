# Online RL Thesis 实验计划【DEPRECATED】

> **⚠️ 此文档已 deprecated（2026-05-06）。**
> **收口报告见 [`docs/online_rl_line_summary.md`](online_rl_line_summary.md)** —— Online 线 thesis-grade 47-run 矩阵已撤销，仅保留 (a) A0 sensor screen 已完成结果、(b) 待 spec 的 SAC collector 数据源两个角色。
> 本文件保留作为历史决策依据；其中 §2（privileged_obs 精确语义）与 §10（Sprint 0 preflight 决策记录）的协议与定义仍 active，可继续被论文 method 节引用。**§4-§7 的 thesis 矩阵不再执行**。
>
> ---
>
> 文档版本：2026-04-26
> 对应报告（累积式）：[`docs/online_rl_thesis_report.md`](online_rl_thesis_report.md)（**从未创建**——thesis 撤销后也不会创建）
> 取代：[`docs/systematic_improved_sac_experiment_plan.md`](systematic_improved_sac_experiment_plan.md)（已 deprecated）
> 适用范围：当前仓库 `auv_nav.sac` 全部组件——`SAC / LayerNorm / Dropout / UTD / AsymmetricQNetwork`。RLPD 留给下一章 offline RL chapter。

---

## 1. 目的、研究问题与三条原则

### 1.1 章节定位

本文件定义 thesis 中 **Online RL 章** 的实验计划。该章在 thesis 中的角色：

> 在真实可部署的 sensor 约束下（DVL water-track only），刻画 SAC 在 wake field 导航任务上的能力边界，并通过仿真训练时的 privileged hull-integral flow 信息，把这个边界往外推。

### 1.2 三条不可妥协的原则（章节设计的 invariant）

1. **Thesis-grade systematic**：实验矩阵需 cleanly factorial、每个 cell 独立可解读；至少一处 5+ seed confirmation pass。
2. **Sensor 是一等公民**：probe layout `s0` 是**主轴 / realistic baseline**，`s1, s2` 是**参考天花板**。所有矩阵按 layout 切片。
3. **Online 先做透再升 offline**：本章主线只有 **vanilla SAC** 与 **AsymmetricCritic**；RLPD 一律留给 offline chapter。

### 1.3 待回答的科学问题

| Q# | 问题 | 对应节 |
|---|---|---|
| Q1 | 在 vanilla SAC 下，sensor × difficulty 二维空间内 SAC 能学到什么？ | §2 |
| Q2 | 在 deployment-realistic 的 s0 上，给 critic 喂 privileged hull-integral flow 能不能补上 vanilla SAC 的 gap？ | §3 |
| Q3 | 这个 privileged-critic 改进对更富 sensor (s1/s2) 是否仍然有效？ | §4 |
| Q4 | 训练于 single-cylinder wake 的策略，能否 zero-shot 迁移到 tandem / sbs 拓扑？ | §5 |
| Q5 | finalist 配置在更多 seed 下统计可信度如何？ | §6 |
| Q6 | observation history 长度 k=4 是否合理？ | §7 |

---

## 2. Privileged_obs 的精确语义

> **这一节是后续 §3 contribution 强度的下限。** 在写论文方法节时必须照抄此处定义，不要替换为模糊措辞。

### 2.1 物理与代码定义

`privileged_obs` 在 [`auv_nav/env.py:947`](../auv_nav/env.py) 的定义：

```python
"privileged_obs": equivalent_body[:2].astype(np.float32),  # dim = 2
```

其中 `equivalent_body` 由 [`auv_nav/autopilot.py`](../auv_nav/autopilot.py) 的 `EquivalentCurrentModel.sample()` 计算：

```
equivalent_world  = Σ_i  weights[i] * flow_at_world(hull_offset_i)
equivalent_body   = R(ψ)^{w→b} · equivalent_world
privileged_obs    = equivalent_body[:2]   # 仅取 (u_eq, v_eq)
```

- `hull_offset_i` 是沿 vehicle 长度（`vehicle_length_m × sample_fractions`）的多个采样点，落在 body-frame 的 x 轴上
- `weights` 是采样点的归一化权重（用于近似 hull 上对动力学贡献的积分）
- 输出维度 = **2**

### 2.2 与 s0 单点 probe 的关键区别

| 量 | 内容 | 数据来源 |
|---|---|---|
| s0 actor 看到的 probe channel | flow 在 body-frame **(0, 0)** 的单点采样 | 单点 sample（实际 DVL 物理上能给到的） |
| privileged_obs（critic 仅训练时可见） | **沿 hull 多点加权积分** 后的 effective flow | sim-only，需要全场流场 |

二者不同，因为：
- 单点 sample 看不到 hull 上别处的流场梯度
- 但驱动 AUV 6-DOF 动力学的恰恰是积分量（详见 vehicle.py 的力学模型）

→ **AsymCritic 的 contribution 假设**：让 critic 在训练时直接看到驱动动力学的真实 effective flow（而不只是 actor 拿到的单点观测），可以让 Q-value 估计更精确，从而把 actor 在 s0 sensor 限制下的策略推到更高水平。

### 2.3 Contribution 强度的天然上限

- privileged_obs 维度 = 2，远低于"完整流场状态"
- 对于 s2（4 个 probe + 含 lateral）的 actor，s2 单点采样已经能近似覆盖 hull 上的流场分布 → AsymCritic 在 s2 上的预期增益**应当显著小于**在 s0 上的增益
- 这正是 §4 要验证的"sensor × algo 交互效应"

### 2.4 实现层细节（写论文方法节用）

[`auv_nav/sac.py:241-298`](../auv_nav/sac.py) 的 `SACAgent.update()` 中：

| 步骤 | 是否使用 privileged_obs | 为什么 |
|---|---|---|
| Critic TD target（`q_target_net(s', a', priv')`） | ✅ 是 | 让 target 受惠于真实 effective flow |
| Critic prediction loss（`q_net(s, a, priv)`） | ✅ 是 | 让 critic 学会在 priv 条件下估值 |
| Actor improvement（`q_net(s, π(s))`，**不传 priv**） | ❌ 否（用 zeros 填充） | Mimic deployment：actor 只能基于 s0 obs 改进 |

这是 *deployment-mimicking asymmetric critic*（区别于 Pinto et al. 2018 的 fully-asymmetric 版本，后者 actor 也用 priv-conditioned Q）。**论文方法节必须明示这个选择**。

---

## 3. 实验骨架（v5，47 run 总预算）

```
§2  Sensor envelope on vanilla SAC                        18 run
    {s0, s1, s2} × {u10_upstream, u15_upstream} × 3 seed

§3  Privileged-critic on s0 (clean ablation)              6 run
    s0 × {对称 LN+UTD SAC, AsymCritic+LN+UTD with priv}
       × u15_upstream × 3 seed

§4  Cross-sensor validation of AsymCritic                 6 run
    AsymCritic+LN+UTD × {s1, s2} × u15_upstream × 3 seed

§5  Topology generalization                               0 run
    §3 finalist + §2 vanilla baseline → tandem / sbs eval

§6  Statistical confirmation                              10 run
    {vanilla s0, AsymCritic finalist} × 5 new seeds × u15

§7  History length defense                                6 run
    finalist × {k=1, k=4} × 3 seed × u15

----------------------------------------------------------------
Sprint 0 (Pre-flight, 必做)                                2 run
    P1: s1 × u15 × seed=46 × 1M 步 (vanilla)              [预算标定]
    P2: s0 × u15 × seed=46 × 50k 步 (AsymCritic)          [smoke]
================================================================
                                                  TOTAL  49 run
```

L4 单 600k run ≈ 1.5h，§ 47 训练 run + 2 preflight ≈ **49 × 1.5h ≈ 74 h**。
按 Colab Pro 单 session 12-24h 估算：**4-7 个 session**。

---

## 4. 各节实验细节

### 4.1 §2 Sensor envelope on vanilla SAC

**目标**：建立"在不做任何算法改进时，sensor × difficulty 决定了什么操作包络"的 baseline matrix。

| 维度 | 取值 |
|---|---|
| sensor | `s0_k4`, `s1_k4`, `s2_k4` |
| difficulty | `single_u10_upstream_tgt15`, `single_u15_upstream_tgt15` |
| algorithm | vanilla SAC（**无 LayerNorm / 无 UTD / 无 AsymCritic**） |
| training steps | `600k`（u10）/ **u15 见 Sprint 0 P1 的标定结果** |
| seeds | `46, 47, 50` |
| objective | `efficiency_v2` 主线；仅当某 cell 全 0% 时补 `arrival_v1` |
| num_envs | `6`（与 A0 一致，跨阶段不可比方差控制） |

**预期形态**：3×2 heatmap，三张图（success rate / variance / path_efficiency）。

**§2 → §3 的 trigger**：§2 出来后必须看到的特征——
- `vanilla s0 × u15_upstream` 显著低于 `vanilla {s1, s2} × u15_upstream`
- 这个 gap 是 §3 要 attack 的 headroom
- 如果 §2 显示 vanilla s0 在 u15 也 saturate（不太可能），则需要重新选难度（升到 single_u20_upstream 或类似）

### 4.2 §3 Privileged-critic on s0 (clean ablation)

**目标**：在 s0 × u15_upstream 这一格，干净地证明"AsymCritic + privileged hull-integral flow"是否能 close §2 看到的 gap。

| Cell | Actor | Critic | LayerNorm | UTD | privileged_obs |
|---|---|---|---|---|---|
| baseline (in §2) | s0 | symmetric | ❌ | 1 | ❌ |
| **§3 control (LN+UTD)** | s0 | symmetric | ✅ | 4 | ❌ |
| **§3 treatment (AsymCritic+LN+UTD)** | s0 | asymmetric | ✅ | 4 | ✅（hull-integral，dim=2） |

- 6 run = 2 cell × 3 seed
- 控制变量：除 "critic 是否看 privileged" 之外，**所有其它训练超参完全一致**
- 这是 §3 严格 ablation：differential between control 和 treatment 必须只来自 privileged info

**为什么需要 LN+UTD baseline 这个 control？**
- 如果只比 vanilla（§2）vs AsymCritic+LN+UTD（§3 treatment），二者差三件事（LayerNorm + UTD + AsymCritic），不能归因
- LN+UTD baseline 把"训练稳定性 / 样本效率"的影响隔离出去
- 这是论文方法节避免被审稿人质疑"也许 gain 来自 LN，不是 priv"的关键

**为什么不在 u10 也跑 §3？**
- u10 上 vanilla 大概率已经接近 saturate（sensor 不那么受限）
- privileged-critic 在饱和点上没有 headroom 可改
- 节省 6 run，论文里以 "saturated regime omitted" 一句带过

### 4.3 §4 Cross-sensor validation of AsymCritic

**目标**：验证 §3 的 contribution 是 **sensor-specific（s0 专属）** 还是 **universal（任何 sensor 都涨）**。前者让方法节 contribution 更 sharp。

| 维度 | 取值 |
|---|---|
| algo | `AsymCritic + LN+UTD`（§3 winner 配置） |
| sensor | `s1_k4`, `s2_k4` |
| difficulty | `single_u15_upstream_tgt15` |
| seeds | `46, 47, 50` |

- 6 run = 2 sensor × 3 seed
- 对应 baseline 已在 §2（vanilla {s1, s2} × u15）

**预期解读三种走向：**
- AsymCritic 在 s0 大涨、s1/s2 小涨或不涨 → contribution 是"privileged training 弥补 sensor 不足"，故事最强
- AsymCritic 全员涨同等幅度 → contribution 是"训练时多模信息 universal 有用"，也有 paper 价值
- AsymCritic 全员不涨 → §3 的 hull-integral 信号不够，方法节要回到"why didn't priv help"，contribution 弱化

### 4.4 §5 Topology generalization (zero-shot eval)

**目标**：用 zero training cost 回答 "训练于 single-cylinder 的策略能否泛化到 tandem / sbs"。

**操作**：
- 取 §3 winner（AsymCritic+LN+UTD on s0, u15）的 `agent_best.pt`
- 取 §2 vanilla s0 baseline 的 `agent_best.pt`
- 对每个 ckpt 在 `tandem_u15_upstream` + `sbs_u15_upstream` 的 manifest 上跑 `scripts/evaluate.py`
- 不做训练

**注意**：probe layout 都是 body-frame 相对的，理论上 layout 可直接迁移；变化的是流场 wake 拓扑。

**输出**：2 行（vanilla / AsymCritic）× 2 列（tandem / sbs）的 generalization gap 表。

### 4.5 §6 Statistical confirmation

**目标**：给 thesis hero number 拉到 5+ seed 的统计可信度。

| 维度 | 取值 |
|---|---|
| configs | `vanilla s0`（baseline）+ `AsymCritic+LN+UTD on s0`（finalist） |
| difficulty | `single_u15_upstream_tgt15` |
| seeds | **新 5 个**：`100, 101, 102, 103, 104`（与 pilot `46, 47, 50` 完全 disjoint） |

- 10 run = 2 config × 5 seed
- 论文报告：5-seed mean ± 95% CI，并明确说明 "3 pilot seeds + 5 confirmation seeds, disjoint sets, to address pilot-cherry-picking concerns"
- 如果 §3+§4 出现意外赢家（例如 AsymCritic+LN+UTD on s1 而非 s0 是真正 finalist），confirmation 跟随实际 winner

### 4.6 §7 History length defense

**目标**：在论文方法节防御"为什么 k=4"。examiner 必问的消融。

| 维度 | 取值 |
|---|---|
| config | §3 winner 配置 |
| history k | `1, 4` |
| difficulty | `single_u15_upstream_tgt15` |
| seeds | `46, 47, 50` |

- 6 run = 2 k × 3 seed
- 不测 k=16：长 history 在 wake field 这种局部信号下没有 prior reason 比 k=4 强；论文里以"k>4 无先验依据"一句带过
- 解读：k=4 显著优于 k=1 → history matters → defend k=4 选择

---

## 5. Sprint 0：Pre-flight（必须做完才能进 §2）

### 5.0 P0：Profiling + num_envs benchmark（addendum，2026-04-26 加入）

**Notebook**：[`notebooks/sac_thesis_s0b_profiling.ipynb`](../notebooks/sac_thesis_s0b_profiling.ipynb)

**触发原因**：L4 上 600k step run 实测 ~1.5h；以 NN 规模（256-hidden 3-layer MLP × 4）估算 GPU 理论忙时仅 ~50min，差额来自 CPU env stepping。在 thesis matrix 启动前是**最后窗口**——`num_envs` 一旦定下来，整章 47 run 不能再改（否则跨阶段不可比）。

**子任务**：
- **P0a**：cProfile 一次 10k 步 vanilla 训练，按模块分桶（`auv_nav.vehicle / flow / env / autopilot` vs `torch` vs `IPC`），定位瓶颈
- **P0b**：跑 `num_envs=6 × 50k` 与 `num_envs=12 × 50k` 两次干净 vanilla SAC，比较 wallclock ratio

**预算**：~15 min wallclock。

**判据**：
| ratio (t6 / t12) | 决策 |
|---|---|
| > 1.7 | 切 `num_envs=12`，更新 thesis plan §4.1 + §2 notebook |
| 1.4 - 1.7 | 切 12，但记录 IPC 损耗接近边界 |
| < 1.4 | 维持 `num_envs=6`，IPC 已成瓶颈 |

**注意**：P0 不阻塞 P1+P2；建议在 P1+P2 跑完后接力做（用同一个 Colab session 即可）。

### 5.1 P1：u15_upstream 训练预算标定

**问题**：A0 上 600k 是 cross_u10 学习曲线倒推的；u15_upstream 是更难的任务，可能需要更长。如果 §2 u15 列因为预算不够而全员卡 0，归因被污染。

**实验**：
```
sensor   = s1_k4         # 三个 layout 中预期最强，最不可能因 sensor 限制学不动
diff     = u15_upstream
seed     = 46
steps    = 1_000_000     # 比 600k 富裕，看 plateau 是否在 600k 后
其它     = vanilla SAC, num_envs=6, k=4
```

**判据**（看 `eval_log.csv` 曲线）：
- 600k 之前已经 plateau → §2 用 600k 即可
- 600k 后还在涨、800k 之前 plateau → §2 u15 列改用 800k
- 800k 后还在涨 → §2 u15 列改用 1M
- 1M 仍未 plateau → 难度太大；考虑改 difficulty（不直接 hack 预算）

**附带产出**：在 L4 上测 1 个 1M step 的真实 wallclock，用于校准后续 sprint 时长。

### 5.2 P2：AsymCritic online 路径 smoke test

**问题**：仓库里 AsymCritic 与 RLPD（offline data）一同被开发；纯 online 路径 + `--use-asymmetric-critic` 是否真的跑得通，需要 smoke 一次。

**实验**：
```
sensor   = s0_k4
diff     = u15_upstream
seed     = 46
steps    = 50_000        # 只看是否能跑通，不看性能
flag     = --use-asymmetric-critic
其它     = vanilla SAC + LayerNorm + UTD=4
```

**判据**：
- 训练能跑完 50k 步、不报错 → ✅ 通过，§3 可以照计划开
- 报错 → 修代码、再 smoke

**预期**：通过（基于代码读后的判断，env.py:947 无条件发 priv，online 路径在 train_sac.py 收集，sac.py update() 使用），但仍不省略 smoke。

### 5.3 Pre-flight 决策记录

P1 + P2 跑完后，在本文档 §10 末尾以"Pre-flight 决策记录"小节落字：
- u15_upstream 预算 = ? steps
- AsymCritic online 路径 = pass / 修复后 pass

---

## 6. Sprint 节奏与决策门

| Sprint | 内容 | 训练 run | session 数 | 决策门 |
|---|---|---|---|---|
| 0 | Pre-flight P1 + P2 | 2 | 1 | u15 预算？AsymCritic online 通？ |
| 1 | §2 全部（s0/s1/s2 × u10/u15） | 18 | 2 | u15 是否真的有 gap？vanilla s0 是否真的弱？ |
| 2 | §3 + §4 共 12 run | 12 | 1-2 | priv 在 s0 是否显著涨？是否对 s1/s2 也涨？ |
| 3 | §5 拓扑泛化（评估） | 0 | 0.5 | 拓扑泛化是否成立？ |
| 4 | §6 statistical confirmation | 10 | 1-2 | hero CI 是否稳？ |
| 5 | §7 history k 消融 | 6 | 1 | k=4 vs k=1 是否显著？ |

**每个 Sprint 的 abort 条件：**
- Sprint 1 失败：u15 全员卡 0，且 P1 已证明预算不是瓶颈 → §3 暂停，重选难度
- Sprint 2 失败：AsymCritic 在 s0 没涨 → §3 头条命题崩，论文方法节必须重新讨论 contribution
- Sprint 4 失败：confirmation 与 pilot 显著背离（5-seed mean 差异 > 2σ_pilot）→ thesis 必须诚实报告并讨论原因

---

## 7. 计算预算总账

| Sprint | 训练 run | 单 run 预算 | 累计 wallclock (L4) |
|---|---|---|---|
| 0 P1 | 1 | 1M 步 ≈ 2.5h | 2.5h |
| 0 P2 | 1 | 50k 步 ≈ 8min | ≈ 2.6h |
| §2 | 18 | 600k-800k 步 ≈ 1.5-2h | 27-36h |
| §3 | 6 | 同 §2 | 9-12h |
| §4 | 6 | 同 §2 | 9-12h |
| §5 | 0 | eval ≈ 30min/config | 1h |
| §6 | 10 | 同 §2 | 15-20h |
| §7 | 6 | 同 §2（k=1 略快） | 8-11h |
| **合计** | **49 run** | — | **72-95 h** |

→ **4-8 个 Colab Pro session**（按单 session 12-24h 计）。

---

## 8. 输出与归档约定

### 8.1 实验结果路径

按 [`docs/systematic_improved_sac_experiment_plan.md`](systematic_improved_sac_experiment_plan.md)（已 deprecated）§2.3 的"两棵镜像目录树"约定（这条约定本身仍有效）：

```
experiments/online_thesis_v1/<benchmark>/<objective>/<algo_tag>/<sensor>_k<k>/seed_<S>/
checkpoints/online_thesis_v1/<benchmark>/<objective>/<algo_tag>/<sensor>_k<k>/seed_<S>/
```

`<algo_tag>` 取值：
- `sac_vanilla`：§2 全部 + §6 baseline + §7 k=1 (on §3 winner)
- `sac_lnutd`：§3 control
- `sac_asym_lnutd`：§3 treatment + §4 + §6 finalist + §7 k=1/4 (on winner)

### 8.2 累积式报告

新报告 [`docs/online_rl_thesis_report.md`](online_rl_thesis_report.md) 在 Sprint 1 启动前创建，每完成一个 sprint 追加一节，结构：

```
§1 总览（章节定位、原则、Pre-flight 决策记录）
§2 Sensor envelope 实测结果         （Sprint 1 完成后写）
§3 Privileged-critic 实测结果       （Sprint 2 完成后写）
§4 Cross-sensor validation 实测     （Sprint 2 完成后写）
§5 Topology generalization 实测     （Sprint 3 完成后写）
§6 Statistical confirmation         （Sprint 4 完成后写）
§7 History length 消融              （Sprint 5 完成后写）
§8 章节级总结与 bridge to offline RL
```

### 8.3 Notebook 拆分（与 Colab 工作流对齐）

每节一个 notebook，互相 disjoint。所有 notebook 复用 ReBRAC notebook 的范式（drive mount → cd → env override → `!bash` → 分析 cell）。

| Sprint | Notebook |
|---|---|
| 0 | `notebooks/sac_thesis_s0_preflight.ipynb` |
| 1 | [`notebooks/_deprecated_sac_thesis_s2_sensor_envelope.ipynb`](../notebooks/_deprecated_sac_thesis_s2_sensor_envelope.ipynb)（原计划名 `sac_thesis_s2_sensor_envelope.ipynb`，撤销后加前缀） |
| 2 | `notebooks/sac_thesis_s3_privileged_critic.ipynb` |
| 2 | `notebooks/sac_thesis_s4_cross_sensor.ipynb` |
| 3 | `notebooks/sac_thesis_s5_topology_eval.ipynb` |
| 4 | `notebooks/sac_thesis_s6_confirmation.ipynb` |
| 5 | `notebooks/sac_thesis_s7_history.ipynb` |

§3-§7 的 notebook 在对应 Sprint 启动前再创建（避免基于错误假设写空骨架）。

> **2026-08-17 补注 —— 上表的实际落地情况**：thesis-grade 矩阵 2026-05-06 撤销，上表 7 行只有 Sprint 0 落地。`notebooks/sac_thesis_s0_preflight.ipynb` 仍在仓库里；Sprint 1 的 notebook 建过、随撤销改名为 [`notebooks/_deprecated_sac_thesis_s2_sensor_envelope.ipynb`](../notebooks/_deprecated_sac_thesis_s2_sensor_envelope.ipynb)，故表内原名不解析；`notebooks/sac_thesis_s3_privileged_critic.ipynb`、`notebooks/sac_thesis_s4_cross_sensor.ipynb`、`notebooks/sac_thesis_s5_topology_eval.ipynb`、`notebooks/sac_thesis_s6_confirmation.ipynb`、`notebooks/sac_thesis_s7_history.ipynb` **从未创建**（git 全历史零次新增）——不是丢失，是上面这句"启动前再创建"按字面生效了。

---

## 9. 待回答的悬案

1. **u15_upstream 预算**：由 Sprint 0 P1 决定
2. **是否扩展 privileged_obs 维度**：当前 dim=2（仅 hull-integral 流速）。如果 §3 在 s0 上效果不显著，可能需要扩展到包含上游 cell 流场 / 流场梯度等。**这是论文 contribution 强度的关键变量**——但属于 thesis depth 升级选项，先不动手，看 §3 实测。
3. **论文中 §3 contribution 的措辞**：在 Sprint 2 实测出"涨多少"之前，方法节标题保留两种候选：
   - `Privileged Hull-Integral Flow as Critic Augmentation for DVL-Limited Sensors`（gain 大时用）
   - `Probing the Limits of Asymmetric Critic Training under Minimal Privilege`（gain 小时用，把 negative result 框架化为 contribution）

---

## 10. Pre-flight 决策记录（Sprint 0 跑完，2026-04-26 落字）

### 10.0 ⚠️ 发现的协议级 bug

**问题**：训练 `--flow` 与 eval manifest 内置 `flow_path` 不一致。

- Sprint 0 全部 run 都用 `FLOW_PATH=wake_v8_U1p00_Re150_*.npy`（U=1.0 m/s）训练
- 但 `single_u15_upstream_tgt15.json` manifest 指定 eval flow = `wake_v8_U1p50_Re250_*.npy`（U=1.5 m/s）
- → train/test flow 不一致，agent 在简单条件下学习、被丢到困难条件评估

**Benchmark → flow 对照表**（必须遵守）：

| benchmark prefix | flow file |
|---|---|
| `single_u10_*` | `wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` |
| `single_u15_*` | `wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy` |
| `tandem_u15_*` | `wake_tandem_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy` |
| `sbs_u15_*` | `wake_sbs_G35_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy` |

**协议级 invariant（写入 thesis 方法节）**：FLOW_PATH 必须由 BENCHMARK_KEY 决定，notebook 不可独立设置。所有 thesis 阶段 notebook 应自动从 manifest 读 flow_path。

### 10.1 P0a — cProfile 瓶颈分布（valid，不依赖 flow 一致性）

总 wallclock 140s 中 ~62s 是结尾 final_eval（30 ep），训练实际 ~80s。**训练时间分布**：

| 占比 | 类别 | 备注 |
|---|---|---|
| ~55% | **AsyncVectorEnv IPC**（`posix.read` from `connection.py`） | 主要瓶颈，pickle obs/info 跨 pipe |
| ~30% | env 物理（flow.bilinear + vehicle.dynamics + autopilot.sample） | 第二瓶颈 |
| ~6% | PyTorch（NN forward/backward） | NN 几乎闲置 |
| 余 | numpy / gymnasium / 其他 | |

→ **关键洞察**：瓶颈不是物理仿真本身，而是**子进程间通过 pipe 传输 pickled obs**。这解释了 P0b 1.45x 而非 2x 的 sublinear scaling。

### 10.2 P0b — num_envs benchmark：valid

| num_envs | 50k step wallclock |
|---|---|
| 6 | 5.61 min |
| 12 | 3.88 min |

**speedup ratio = 1.45x**（落在 1.4-1.7 临界区）

**决策**：✅ **切到 `num_envs=12`**——47 thesis run × 1.45x ≈ 节省 13-15 小时总 wallclock；IPC 已是边界但仍有正收益。

外推：600k step 单 run 在 num_envs=12 上预期 ~47 min（vs num_envs=6 的 ~67 min）。

### 10.3 P1 — u15_upstream 预算标定：❌ INVALID，需重跑

P1 用 U=1.0 flow 训练、U=1.5 flow 评估，success rate 在 200k 后剧烈震荡（0.5-1.0 之间）——**这是 train/test mismatch 的产物，不是真实 plateau**。

→ **action**：用正确的 `wake_v8_U1p50_Re250_*.npy` 重跑 P1，~2.5h Colab session。

### 10.4 P2 — AsymCritic online smoke：✅ PASS

50k 步 vanilla SAC + LayerNorm + UTD=4 + AsymmetricCritic：
- 训练完成无报错
- `final_eval.json` 写出（success rate=0.0 是 50k 还在起步阶段，不是 bug）
- → **online + asym 路径无需改代码即可工作**，§3 可以照计划开

### 10.5 Tier 3 backlog（thesis 进行中可选）

cProfile 显示 IPC 占 55% → Numba JIT 上限只有 ~30% wallclock 改善，**而 `AsyncVectorEnv(shared_memory=True)` 上限可达 ~50%**。优先级反转：

| backlog 项 | 预期收益 | 工作量 | 优先级 |
|---|---|---|---|
| `AsyncVectorEnv(shared_memory=True)` | wallclock -30~40% | 1-2h | **§3 启动前考虑做**（可再砍 thesis 总预算 1/3） |
| Numba JIT vehicle.py + flow.py | wallclock -15~25% | 1-2 天 | thesis 后期备选 |

### 10.6 Action items（按顺序）

1. **修复 FLOW_PATH 自动化**：notebook 调用 helper 从 BENCHMARK_KEY 推 flow（杜绝同类 bug）
2. **改 §2 / §0 notebook**：FLOW_PATH 自动化 + NUM_ENVS=12
3. **重跑 P1**：取得真正的 u15_upstream plateau 数据
4. （可选）评估 `AsyncVectorEnv(shared_memory=True)` 可行性
5. P1 数据落字到本节，然后才进 Sprint 1（§2）

### 10.7 协议状态快照

| 协议项 | 取值 | 来源 |
|---|---|---|
| `num_envs` | **12** | P0b 实测 |
| `--flow` | 由 BENCHMARK_KEY 决定 | 10.0 invariant |
| u15_upstream 预算 | **待 P1 重跑** | 10.3 |
| AsymCritic online | ✅ 可用 | P2 |
