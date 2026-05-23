# AUVHamNODE-based Offline RL:Cross-Domain Transfer via Frozen Physics-Structured 1-Step Dynamics

> **⚠ PAUSED 2026-05-13** — 本线已暂停。详情、累计决策、恢复条件全部记录在 [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md)。本文(v2.1 plan)保留作历史 + resume 起点;**直接照此 plan 实施前**必须先读 pause memo §4(累计决策)与 §5(未做的事),否则会重复已做过的 Step 0-4 审计。
>
> Pause 期间影响:v2.1 §11.1 fire-condition 路径 A/B/C 已被 v3 pre-notes §3 的 4 项硬接口差异叠加暂停决策**取代**;v2.1 §4.2 A 阈值"normalized MSE < 0.1"在 wake U=1.5 上已被审计判定不可达。

> **ℹ FQL Succession（Paper 2）NEGATIVE 闭环旁注（2026-05-23）** — 平行的 FQL vs ReBRAC 线已闭环为**诚实负面 + 机制发现**:"表达力更强的 flow-matching 先验在本 AUV 导航任务上无 leverage"（FQL 不系统性超过 ReBRAC;详见 [`docs/fql_succession_p2_results.md`](fql_succession_p2_results.md)）。这对本线 §10.1 的 expressive-prior framing 是**相关上下文**,但**不改变 PAUSED 状态**,也**不是** §11.1 的 fire-condition discriminator —— 后者指 ReBRAC broad-val **C1 BC-penalty sweep**,与 FQL Succession 的 **C-1**(`distill_alpha_bc` 复赛)是两件不同的事,勿混淆。

**版本**:v2.1(2026-05-10 amendment:§11.1.0 零号前置增补)
**日期**:2026-05-09(原版)/ 2026-05-10(amendment)
**状态**:**PAUSED 2026-05-13**(原状态:locked plan,待 §11.1.0 零号前置 + §11.1.1 fire-condition 路径满足后启动)
**前序**:v1.0(2026-05-06)consolidated plan → v2.0(2026-05-08)经 v2.1/v2.2 严格审查迭代后定型 → **v2.1(2026-05-09)RL 专家 meta-review patch:Phase 0 新增 F 测试(critic Q overestimation hard gate)、§5.2 明确 done flag / wake time index / σ_a sampling 协议、§5.4 主对照升 5-seed 加 mix-ratio pilot、§5.5 改 paired bootstrap 95% CI 检验、§11.1 重写 fire-condition** → **v2.1 amendment(2026-05-10):新增 §11.1.0 零号前置(checkpoint + wrapper + audit table),修正 §11.1 路径 A/B/C 共享前置的疏漏**
**作者**:基于 [`docs/offline_mbrl_plan/`](offline_mbrl_plan/) 下 6 份草案的批判性合并 + Plan B(`NODE_IQL_FQL_SORL_revised_roadmap_v3.md`)的选择性吸收 + audit 后简化 + v2.1 RL 专家审查反馈整合

---

## v2.1 修订摘要(P0 + P1 patch 索引)

| # | 优先级 | 修订点 | 落点 |
|---|---|---|---|
| 1 | P0 | augmented `done` / `terminated` / `truncated` flag 协议明确化 | §5.2 步骤 5、§5.3 hint |
| 2 | P0 | Critic Q overestimation 升为 Phase 0 hard gate(新增测试 F) | §4.2 F |
| 3 | P0 | "AUVHamNODE-aug ≥ MLP-ensemble-aug" 改 paired bootstrap 95% CI > 0 | §5.5 |
| 4 | P0 | Fire-condition 与 broad_val C1 sweep 排期 reconcile | §11.1 |
| 5 | P1 | Phase 0 之后做 mix-ratio 1-seed pilot,主对照升 5-seed | §5.4 |
| 6 | P1 | plain-MLP-ensemble checkpoint protocol audit(不必然重训) | §1.1、§4.2 G |
| 7 | P1 | Phase 0 B vehicle.py oracle 协议 spell out + §1.2.5 字面豁免 | §1.2、§4.2 B |
| 8 | P1 | σ_a Gaussian + clip → truncated normal,report effective σ | §4.2 B、§5.2 步骤 2 |
| 9 | P1 | Probe wake field 时间 index 协议明确(用 dataset frame t+1) | §5.2 步骤 4 |
| 10 | P1 | privileged_obs 在 augmented transition 上的 hull-integral 生成(Path B 工程量警示) | §5.3 hint |
| 11 | P1 | Reward 重算的 progress / terminal 项分别处理 | §5.2 步骤 5 |
| **12** | **P0** | **§11.1.0 零号前置:checkpoint + wrapper + audit table 入库(2026-05-10 amendment 增补)** | **§11.1.0** |

---

## 0. 文档定位

本方案是 [`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) §3.4 中标记的 **Phase 3 — AUVHamNODE Offline MBRL** 的具体落地方案。

它**不替代**当前 ReBRAC 主线工作(broad validation / paper revision)。它是这些工作收敛后的下一阶段研究规划,目标是:

> **以已冻结、已发布过对照实验的 AUVHamNODE 为 1-step 动力学先验,通过 (s, a) 邻域 augmentation 增强 ReBRAC 的 sample efficiency,并在论文中把贡献定位为"frozen physics-structured dynamics 的跨域迁移",而不是 in-domain MBPO 拓展。**

相对 v1.0,v2.0 的核心修订:
- **删除** Phase 2 MVE 多步 rollout 和 Phase 3 SHAC pathwise gradient(工程量大、流场 propagation 风险高、收益不 justify)
- **删除** XQL/FQL/SORL/IQL 算法栈(Plan B 主轴,与 ReBRAC paper-ready 不连续)
- **保留** ReBRAC base + 1-step augmentation 主轴
- **新增** plain-MLP-ensemble 对照(load 已有 checkpoint,无需重训)
- **升级** paper claim 为"cross-domain transfer of frozen physics-structured dynamics"(因 audit #4:AUVHamNODE 训练数据 ≠ 当前 wake_data)

---

## 1. 问题精确陈述

### 1.1 给定资源

| 资源 | 说明 |
|---|---|
| **AUVHamNODE checkpoint**(冻结) | 已训好的 port-Hamiltonian Neural ODE,接收 (state, action, **当前位置流速**) → 输出 next state derivative;**训练数据 ≠ 当前 wake_data**(在其他数据上预训练并冻结) |
| **plain-MLP / 其他 dynamics checkpoints** | AUVHamNODE 工作中已做过对照实验的 dynamics models;**与 AUVHamNODE checkpoint 一同稍后入库**。**v2.1 注**:checkpoint 入库时必须同时入库 protocol audit table(参数量、训练数据、训练 epoch、LR schedule、ensemble size N),由 §4.2 G 检查是否 capacity-matched;若已 matched 则直接复用,若未 matched 则在 Phase 0 内补齐对照训练 |
| **`vehicle.py` ground truth** | 6-DOF 解析动力学,作为 oracle |
| **Wake field data** | [`wake_data/*.npy`](../wake_data/),shape `(T, Nx, Ny, C)` |
| **Offline dataset** | [`offline_data/<dataset>/transitions.npz`](../offline_data/),由 [`scripts/collect_offline_data.py`](../scripts/collect_offline_data.py) 用 baseline policies 在 `vehicle.py` 中收集;含 `privileged_obs`(hull-integral)字段 |
| **ReBRAC baseline** | paper-ready 4/4 闭环(rev.8) |
| **Codebase 已有组件** | [`auv_nav/replay.py`](../auv_nav/replay.py) `DualBufferSampler`(RLPD-style);[`auv_nav/sac.py`](../auv_nav/sac.py) `AsymmetricQNetwork`;[`auv_nav/autopilot.py`](../auv_nav/autopilot.py) `EquivalentCurrentModel`;[`auv_nav/rebrac.py`](../auv_nav/rebrac.py) ReBRAC agent |

### 1.2 关键约束

1. **AUVHamNODE 是冻结的、out-of-distribution 的 1-step dynamics model**
   - 训练数据 ≠ 当前 wake_data → 跨域迁移成立与否是 Phase 0 必须回答的硬问题
   - 不允许 finetune;只能"用"或"不用"
2. **流场是 exogenous,且对部署 actor 不可知**
   - 部署中 AUV 仅能测当前位置流速 \(w_t\)(DVL/ADCP);未来 \(w_{t+\tau}\) 不可知
   - 训练时不允许让 actor 依赖未来流场信息
3. **NODE 限定 1-step 用途**
   - 不做 multi-step rollout / MVE / SHAC pathwise gradient
   - 不做未来流场 query / forecast
   - 这把 v1.0 的"流场如何 propagate"问题彻底消掉
4. **AUVHamNODE 是 ODE(非 SDE)**
   - augmentation 是 deterministic;每条 dataset transition 产出 1 条 augmented transition
   - 多样性来源仅 σ_a(action 扰动),不来自 model stochasticity
5. **保持 offline 性质 + 现有 env/task 不变**
   - **RL 训练阶段(Phase 1+)**不再访问真实 AUV / `vehicle.py`
   - **Phase 0 例外**:`vehicle.py` 允许作为 ground-truth oracle,仅用于 §4.2 测试 B 的 σ_a 扰动 MSE 校准(离线 ablation,不进入 RL agent training loop)
   - 不引入 obstacle / sonar / map-aware 观测层
   - 沿用现有 `PlanarRemusEnv` + s0/s1/s2 obs 协议

### 1.3 目标(按优先级)

1. 在固定 dataset 规模下,提升 ReBRAC 的 final return / success rate
2. 在缩减 dataset 规模(25%)下,达到接近 100%-data ReBRAC 的性能
3. 证明物理结构先验(port-Hamiltonian)在跨域迁移中胜过 plain-MLP ensemble

---

## 2. 5 条核心设计原则

1. **NODE 限定 1-step**。v1.0 的"流场如何 propagate"是结构性决策;v2.0 通过"不 propagate"消掉这个决策。
2. **训练-部署一致性 > privileged 信息利用**。Privileged 仅限 in-domain 量(hull-integral [u_eq, v_eq],部署时也能算),绝不 leak 未来流场到 actor。
3. **物理结构 + twin-Q min,不上 ensemble pessimism**。port-Hamiltonian 结构先验已部分替代 plain ensemble OOD 机制;MOPO/COMBO/MOBILE/LEQ 全部不上 default。
4. **(s, a) 邻域 augmentation 是合法收益来源**。MBPO/SynthER 一线已建立 motivation;跨域迁移设定下,收益主要来自 (s, a) 维 interpolation,不来自外推到新物理。
5. **paper claim 锚在跨域迁移**,不锚在"more sophisticated MBRL"。AUVHamNODE 训练数据 ≠ 当前 wake_data 是这条 claim 的关键证据。

---

## 3. 方案速览

```
Phase 0:Pre-flight (1 周)
  └─ 5 个 ablation,domain-shift 特征化 + go/no-go

Phase 1:ReBRAC + 1-step Augmentation (2.5-3 周)
  └─ AUVHamNODE-aug vs plain-MLP-ensemble-aug vs no-aug
     流场策略:dataset 真实 w_t(不 propagate)
     默认 AsymCritic OFF;Path B 作为末尾 ablation

Phase 2A(条件触发):Inference-time Risk-aware Verifier (1-2 周)
  └─ Phase 1 训好的 actor + NODE 1-step 评分,纯 inference-time 无需重训

Phase 2B(可选,仅 2A margin 小时):NODE-feature Critic (3-4 周)
  └─ critic input concat φ_θ(s, a),需要 retraining
```

**总时长**:5-6 周(取决于 Colab L4 quota 和 checkpoint 就绪时机)。

---

## 4. Phase 0:Pre-flight Ablation(1 周)

### 4.1 目的

在跨域迁移设定下,回答 5 个 go/no-go 问题。

### 4.2 必做的 5 项 ablation

#### A. NODE 1-step prediction MSE on offline dataset(domain-shift 特征化)

**为什么测**:AUVHamNODE 训练数据 ≠ 当前 wake_data。要先回答"NODE 在新域上还准不准"。这是整个方案的 make-or-break gate。

**怎么测**:
1. 在每个 offline dataset(`crosscomp-1000` / `crosscomp-2000` / `worldcomp-1000` 等)上算 normalized 1-step MSE = ‖f_θ(s, a, w) − s'‖² / Var(s' − s)
2. **分桶报告**(关键):
   - 按 flow magnitude(low / mid / high):分位数 [0-33%, 33-67%, 67-100%]
   - 按 task geometry(downstream / cross / upstream)
   - 按是否在涡核 / 剪切层附近(用 wake field gradient 阈值判断)

**判断**:
- 整体 normalized MSE < 0.1 且各桶都 < 0.2 → **跨域迁移成立**,进 Phase 1,完整 paper claim
- 整体 < 0.2 但某桶 > 0.5 → 进 Phase 1,但 augmentation 限定在 low-MSE region;paper claim 中等
- 整体 > 0.3 → **暂停方案**;考虑寻求 AUVHamNODE 在当前 wake_data 上的轻量 finetune(虽然冻结假设,但跨域失败时可重审);或换论文方向

**记录**:NODE 在哪些状态区域误差大 — 后续作为 Phase 1 σ_a 的桶特定上限依据。

#### B. NODE 1-step MSE 在 σ_a 扰动下的退化曲线(关键 gate)

**为什么测**:augmentation 用的是 (s, a + ε),NODE 在 (s, a) 上的精度不等于在扰动 action 上的精度。这是 Phase 1 σ_a 起点的硬依据。

**怎么测**:
1. 对 dataset transitions 中的每条,生成 4 组扰动 a' = a + ε,σ_a ∈ {0.05, 0.1, 0.2, 0.4} × action range
   - **v2.1 注**:`ε` 不直接用 `N(0, σ_a²)` + clip(saturated dataset action 上 clip 会引入 truncation artifact,使实际扰动幅度远小于 σ_a)。实施二选一:
     - (a) `truncated normal` 在 actuator 限制内采样;
     - (b) 在 logit/tanh-space 扰动后 squash 回 actuator range
   - **必须 report effective σ**(实际生成的 ‖a' − a‖ 的 std),按 dataset action 在边界附近的比例分桶,与 σ_a 一同列入 Phase 0 报告
2. 用 NODE 算 s̃_{t+1};与 vehicle.py 用 (s, a') 跑出的 s_{t+1}^true 对比
   - **v2.1 注**:vehicle.py oracle 协议必须在 Phase 0 报告里 spell out:
     - rollout 时长:1 个 control step(与 NODE 输出对应)
     - 内部积分:与 vehicle.py 默认 RK4 一致
     - **wake field 输入**:固定为 dataset 在 t 时刻记录的 w_t snapshot(不在 1-step 内做时间插值,与 §1.2.3 "不做未来流场 query" 约束一致)
     - 给出 oracle wrapper 代码 snippet(放入 [`tests/test_vehicle_oracle_phase0.py`](../tests/) 单测验证)
3. 报告 MSE(σ_a) 曲线

**判断**:选择 MSE(σ_a) < 1.5 × MSE(0) 的最大 σ_a 作为 Phase 1 起点。若所有 σ_a 都退化严重,σ_a = 0.05 起步并升级 OOD penalty。

#### C. Dataset action coverage density

**为什么测**:augmentation 的"邻域"要有意义,dataset action 必须有一定 spread。

**怎么测**:
1. 对 100 个随机 \(s\) 做 nearest-neighbor 查询(state-space K=10)
2. 看这 10 个 neighbor 对应 action 的边缘 std
3. 比较 std 与 action range 的 ratio

**判断**:
- ratio > 0.1 → action 多样性足
- ratio < 0.05 → dataset action 单峰窄,**Phase 1 σ_a 必须 ≥ 0.2**(若 B test 允许)

#### D. ReBRAC baseline saturation

**为什么测**:如果 ReBRAC 还没收敛,model-based 提升可能只是把 ReBRAC ceiling 提前实现,不是真实增益。

**怎么测**:在最强 baseline 配置下跑 1.5 × current `--total-steps`,看 final return / success rate 是否还在涨。

**判断**:已饱和 → 进 Phase 1;未饱和 → 先把 ReBRAC 跑透。

#### E. Dataset coverage audit(Plan B §3.2)

**为什么测**:augmentation 收益与 dataset 多模态性弱相关,但 audit 本身能为后续 risk-aware verifier(Phase 2A)的 R_θ 设计提供分布信息。

**怎么测**(报告即可,不卡阈值):
- action 模式分布(直方图 / KDE)
- 成功 / 失败轨迹比例
- 高速逆流 / 顺流 / 横流分桶样本数量
- 动作饱和比例
- 回报分布

**v2.1 注**:测试 C 中的 0.1 / 0.05 binary cutoff 撤销,改为 report **action-spread ratio 完整分布**(直方图 + 25th / 50th / 75th percentile),σ_a 起步是否 ≥ 0.2 由 Phase 0 实测分布的 25th percentile 与 dataset action 边界饱和比例联合决定,而不是预设阈值。

#### F. **(v2.1 新增,P0 hard gate)** Critic Q overestimation under augmentation

**为什么测**:跨域 OOD frozen NODE 的最大风险是 1-step prediction systematic bias 通过 Bellman bootstrap 累积成 critic Q overestimation。ReBRAC 的 twin-Q min 抑制 stochastic noise 但**不**抑制 systematic bias;一旦 critic 系统性偏移,augmentation 越多偏移越严重(deterministic NODE 100% 重复同向错误)。该测试是 Phase 1 起 σ_a / mix ratio 的**第二硬约束**。

**怎么测**:
1. 取 Phase 0 测试 D 已收敛的 ReBRAC checkpoint(无 augmentation)。
2. 在 100 条 holdout dataset trajectories 上,对每个 (s_t, a_t) 算:
   - critic 预测:Q_θ(s_t, a_t) = min(Q_1, Q_2)
   - Monte Carlo return:G_t = Σ γ^k · r_{t+k}(从 dataset trajectory 累加,折扣 γ 取与 ReBRAC 训练一致)
3. baseline:report ⟨Q_θ − G_t⟩ 与 ⟨|Q_θ − G_t|⟩ 的均值与 95% CI。
4. 干预:用 σ_a = Phase 0 B 选定值生成 100% augmentation buffer,继续训练 1k gradient step,重测同样的 Q_θ − G_t。
5. 关键指标:**Δ overestimation = (Q_θ^aug − Q_θ^baseline) on holdout**。

**判断**:
- |Δ| / |Q_θ^baseline| < 5% → 进 Phase 1,不需要 OOD penalty
- 5–15% → 进 Phase 1,**Phase 1 必须开启 λ_ood = 0.01 起步**(§5.4 二级 ablation 提前到默认)
- > 15% → **Phase 1 暂停**,先在 Phase 0 内调 σ_a / mix ratio 把 Δ 压回 15% 以下,或触发 §8.1 的 plain-MLP-ensemble disagreement 作 OOD penalty 路径
- 该指标的 baseline trace 同时保留为 Phase 1 全程的 monitor signal(每 50k step 重测一次)

#### G. **(v2.1 新增,P1)** Plain-MLP-ensemble checkpoint protocol audit

**为什么测**:§10.1 中等档 paper claim("physics structure enables transfer")完全建立在 "AUVHamNODE-aug > plain-MLP-ensemble-aug" 上。如果两份 checkpoint 在(参数量、训练数据、训练 epoch、LR、ensemble size)上不 match,审稿人会立刻 attack:"是 physics 的功劳,还是 plain-MLP 在 OOD 数据上 underfit?"

**怎么测**(报告即可,不消耗 compute):
- 从 AUVHamNODE 工作中提取两份 checkpoint 的 protocol table
- 列出:total params、training dataset(name + size)、training epoch、LR schedule、batch size、ensemble size N、validation MSE、训练时是否见过 wake field input distribution

**判断**:
- 全部 match(差异 < 10%)→ 直接复用,论文 §10.1 中等档 claim 成立条件:**"under matched training budget"**
- 关键 axis 不 match → Phase 0 内补齐 capacity-matched MLP-ensemble 重训(单次 dynamics 训练而非 RL 训练,工程量 < 1 周)
- audit table 必须公开进论文 appendix

### 4.3 Phase 0 产出

[`docs/offline_mbrl_plan/phase0_ablation_report.md`](offline_mbrl_plan/phase0_ablation_report.md),包含:
- **7 项 ablation**(A–G,v2.1 增 F + G)的具体数值 + 分桶 MSE 表
- AUVHamNODE 在哪些状态区域误差大(后续作为 Phase 1 OOD penalty 触发条件)
- σ_a 起点选择(B test 派生,含 effective σ report)
- F test 的 Q overestimation Δ 数值与 λ_ood 起步决定
- G test 的 capacity-match audit table(若 mismatch,记录补训进度)
- go/no-go 决定

---

## 5. Phase 1:ReBRAC + 1-step Augmentation(2.5-3 周)

### 5.1 目的

正面回答两个问题:
1. **AUVHamNODE 是否能让 ReBRAC 更快/更稳?**(基础 sample efficiency 对照)
2. **physics structure 是否真比 plain-MLP ensemble 强?**(novelty 对照,#3 audit 让这条变便宜)

### 5.2 设计

**Augmentation pipeline**:
1. 起点:dataset 真实 transition (s_t, a_t^data, w_t, s_{t+1}^data, r_t^data, done_t^data)
2. 扰动 action:a_t' = a_t^data + ε,**采样 ε 用 truncated normal 在 actuator 限制内**(或 logit-space 扰动后 squash;**不**用 N(0, σ_a²) + clip,见 §4.2 B 注)。**生成 effective σ 与 σ_a 一同记录入 augmentation metadata**
3. 1-step model 推理:
   - **AUVHamNODE 路径**:s̃_{t+1} = ODE_solve(f_θ^Ham, s_t, a_t', w_t, Δt)
   - **plain-MLP-ensemble 路径**:s̃_{t+1} = aggregate({f_θ^MLP_i(s_t, a_t', w_t)}_{i=1..N}),aggregation 用 random-pick(MBPO 风格)
   - **ODE solver 一致性**:wrapper 必须读取 NODE checkpoint 训练时的 solver(dopri5 / rk4 / euler)并锁定使用;不允许在 augmentation 推理时换 solver
4. 重新查询 wake field at s̃_{t+1}'s pose,生成 probe channels
   - **wake time index 协议(v2.1)**:固定使用 **dataset frame index t+1** 对应的 wake snapshot W_{t+1}^data(因为 s̃_{t+1} 与 s_{t+1}^data 处于同一 episode 时间步,只是位置不同)。**不**用 w_t snapshot,**不**用任何前向插值——这与 §1.2.3 "不做未来流场 query" 兼容(t+1 index 已经是 dataset 给出的当前观测,而不是 forecast)
   - **augmentation pipeline 必须有 wake field memmap 访问**(probe channel 重生成 + Path B hull-integral 重生成)
5. Reward & done flag 重算(v2.1 P0):**分项处理**,因为 [`RewardModel.compute()`](../auv_nav/reward.py) 需要 `reason / terminated / truncated` flag,这些 flag 不是 (s, a, s')-pure 函数:
   - **可重算项**(从 (s_t, a_t', s̃_{t+1}) 推):`progress`、`safety_cost`、`energy_cost`、`task_reward`(time penalty)
   - **不可重算项**:`reason / terminated / truncated`(episode 终止判定逻辑由 [`PlanarRemusEnv`](../auv_nav/env.py) 决定,需要 boundary check + goal radius check + step count)
   - **协议(v2.1 默认)**:**复用 dataset done_t^data**——augmented transition 的 terminal flag = dataset 同步 flag,augmented step reward 只更新可重算项,terminal_reward 沿用 dataset r_t^data 中的 terminal 部分
   - **理由**:augmented s̃_{t+1} 偏离真值通常很小(NODE 1-step MSE 已由 §4.2 A 校验),terminal flag 反转概率低;直接复用 dataset flag 比新跑 env 终止 check 实现成本低 5–10×,且确保 reward 与 done 一致
   - **审计要求**:Phase 1 进度报告必须 report **augmented terminal-flag 反转率的事后估计**(用 vehicle.py oracle 在 1% 子样本上重跑 termination 判定),如果反转率 > 5%,触发 §5.6 fall-back
6. 写入 `model_transitions.npz`,schema 与现有 offline data 一致(privileged_obs 字段在 Path A 下为空;Path B 下需走 §5.3 hint 的 hull-integral 流程)

**关键设计决策**:
- **严格 1 步**:完全规避 "w 怎么 propagate" 问题;w_t 直接从 dataset 取
- **不混入 actor action**:a_t' = a_t^data + ε,**不**用 0.5·a_β + 0.5·a_π + ε。理由:actor 还在变,混入会引入 distribution drift,augmentation 可重复性差
- **σ_a 起步**:由 Phase 0 B 选定;若 C 25th-percentile spread < 0.05 × action range,起步 ≥ 0.2(在 B 允许范围内)
- **mix ratio**:**v2.1 改为先 pilot 再固定**——Phase 0 完成后用 1 seed 在 {25:75, 50:50, 75:25, 90:10} 上各跑 0.5 × current `--total-steps` 的 ReBRAC + AUVHamNODE-aug,选择 final return 最高的作为 Phase 1 主跑起点。pilot 默认对照保留 50:50(与仓库 `DualBufferSampler` 默认对齐)
- **OOD penalty**:λ_ood **由 §4.2 F 测试 F 的 Δ overestimation 决定**(< 5% 用 λ=0;5–15% 用 λ=0.01;> 15% 触发 fall-back),不再是"出现 overestimation 时再加"的事后反应
- **AsymCritic**:**默认 OFF**(Path A);Path B 作为末尾 ablation,**baseline 与 augmented 必须用同一 critic 配置配对**

### 5.3 实现 hint

**新增模块**:
- [`auv_nav/auvhamnode.py`](../auv_nav/auvhamnode.py) — 包装 AUVHamNODE checkpoint;接口对齐 [`auv_nav/vehicle.py`](../auv_nav/vehicle.py) `Remus100.dynamics(x, ui, Vc, beta_Vc, w_c)`(注:checkpoint + 该 wrapper 代码 stub 稍后随 commit 入库;wrapper 必须从 checkpoint metadata 读取并锁定 ODE solver 类型,augmentation 与 Phase 2A verifier 共用同一 solver)
- [`auv_nav/dynamics_ensemble.py`](../auv_nav/dynamics_ensemble.py) — 包装 plain-MLP / 其他 dynamics checkpoints(对照实验,稍后随 commit 入库;入库时附 §4.2 G 要求的 protocol audit table);support `--dynamics-source {auvhamnode, plain_ensemble, ...}` dispatch
- [`scripts/generate_model_augmented_buffer.py`](../scripts/generate_model_augmented_buffer.py) — load offline dataset + wake_data + dynamics checkpoint → augmentation 推理 → 写出 `model_transitions.npz`
   - **必须**:wake field memmap 访问(probe channel 重生成 + Path B hull-integral 重生成)
   - **必须**:复用 [`auv_nav/autopilot.py`](../auv_nav/autopilot.py) `EquivalentCurrentModel` 在 s̃_{t+1} 上做 hull 多点采样(`hull_flow_sample_fractions = (-0.4, -0.2, 0.0, 0.2, 0.4)` 5 点 + weighted integration)生成 augmented `privileged_obs`(Path B 必需,Path A 字段为空)。**v2.1 注**:这一步在 v2.0 文本里被简化成"wake field 单点 query",实际 Path B 工程量更大,需提前 budget
   - **必须**:对每条 augmented transition 同步写出 `done_t^aug = done_t^data`(沿用 dataset done flag,见 §5.2 步骤 5 协议)
- [`tests/test_auvhamnode_wrapper.py`](../tests/test_auvhamnode_wrapper.py) — 单元测试,验证接口与 `Remus100` 一致
- [`tests/test_vehicle_oracle_phase0.py`](../tests/test_vehicle_oracle_phase0.py) — Phase 0 B 的 vehicle.py oracle 单测,固定 wake snapshot 协议(见 §4.2 B v2.1 注)

**复用现有**:
- [`scripts/train_offline_rebrac.py`](../scripts/train_offline_rebrac.py) 加 `--model-data` flag,复用 RLPD 的双 buffer 路径([`auv_nav/replay.py`](../auv_nav/replay.py) `DualBufferSampler`)
- `RewardModel` 直接用,**不要改 reward**;按 §5.2 步骤 5 的"分项重算 + dataset done 复用"协议调用
- evaluation pipeline([`scripts/evaluate.py`](../scripts/evaluate.py) + benchmark manifests)不动

**绝对不要**:
- 不要碰 ReBRAC 的 actor / critic loss
- 不要加 ensemble + uncertainty penalty(plain-MLP-ensemble 仅作 augmentation 数据源,不进 RL agent)
- 不要改 [`auv_nav/env.py`](../auv_nav/env.py)
- 不要给 NODE 做 multi-step rollout

### 5.4 实验矩阵

**Step 1 — Mix ratio pilot(v2.1 新增,1-seed,先做)**:

| Mix ratio (real:model) | data scale | 跑 0.5 × `--total-steps` | 选优指标 |
|---|---|---|---|
| 25:75 / 50:50 / 75:25 / 90:10 | 25% real | 1 seed × 4 配置 | final eval return |

固定 AUVHamNODE-aug variant,选 final return 最高的 mix ratio 作为 Step 2 主跑起点(若多个并列,取 RLPD 默认 50:50)。

**Step 2 — 主对照矩阵(v2.1 升级 5-seed)**:

| 真实数据比例 | ReBRAC(已有) | + AUVHamNODE-aug | + plain-MLP-ensemble-aug |
|---|:-:|:-:|:-:|
| 25% | ✓(broad validation 已有,5-seed) | **✓ 5 seeds** | **✓ 5 seeds** |
| 100% | ✓(broad validation 已有,5-seed) | **✓ 5 seeds** | **✓ 5 seeds** |

(共 4 cell × 5 seed = 20 paper-grade run)Benchmarks:`single_u15_upstream_tgt15`、`tandem_u15_upstream_tgt15`(与 broad validation 标准对齐)。

**Step 3 — Ablation(3 seeds)**:
- σ_a sweep:继承 Phase 0 B 的 4 个值
- AsymCritic on/off(Path B,baseline 与 augmented 配对同步切换)
- λ_ood:由 §4.2 F 的 Δ overestimation 决定起步值;额外做 ±1 阶 sensitivity ablation
- mix ratio 完整 sweep:{25:75, 50:50, 75:25, 90:10} 在 Step 2 选定的 best variant 上重做(确认 pilot 不是 1-seed noise)

### 5.5 进入 Phase 2A 的条件

**v2.1 升级:全部满足才进,且关键比较使用统计检验**:

- 25% real + AUVHamNODE-aug 的 final return ≥ 25% real-only × 1.10(**5-seed mean,paired bootstrap 95% CI 下界 > 1.05**)
- success rate 上升,**paired bootstrap n=10k,95% CI 下界 > 0.5 std**
- action support distance(policy action 到 dataset action 的均距)< 1.2 × baseline
- held-out flow generalization gap 不退化(**held-out flow 定义**:不在 augmentation 训练 dataset 出现的 wake_data 文件,Phase 0 报告中预先指定;典型选 `wake_v8_U1p00_Re150` 之外的另一组 Re/U 配置)
- **AUVHamNODE-aug > plain-MLP-ensemble-aug**(novelty 关键)
   - **统计检验(v2.1 P0)**:per-benchmark paired bootstrap n=10k across 5 seeds,**要求 95% CI 下界 > 0**;若 CI 跨 0,论文 claim 退化为描述性("trends in favor of physics structure but not statistically significant")
   - 对齐 [`docs/rebrac_statistical_test_followup.md`](rebrac_statistical_test_followup.md) 的 ReBRAC paper 主线 statistical protocol

### 5.6 Phase 1 fall-back

若不满足:
- 检查 σ_a / mix ratio / NODE prediction MSE on dataset neighborhood
- 不要直接跳 Phase 2,先在 Phase 1 内迭代
- 若迭代 1-2 轮后仍无效,**接受 ReBRAC 已是 ceiling**;成果 reframe 为 ReBRAC paper 的 **appendix-level ablation**("we attempted physics-structured offline MBRL augmentation; ReBRAC saturates at deployable ceiling"),不写独立 negative-result paper

### 5.7 Phase 1 退出产出

[`docs/offline_mbrl_plan/phase1_results.md`](offline_mbrl_plan/phase1_results.md):
- ablation 矩阵全数值
- σ_a / mix ratio sensitivity 曲线
- AUVHamNODE vs plain-MLP-ensemble 对照(novelty key)
- 进入 Phase 2A 的判定

---

## 6. Phase 2A(条件触发):Inference-time Risk-aware Verifier(1-2 周)

仅当 Phase 1 通过且想进一步提升 paper claim 时才做。**核心特点:不重训,纯 inference-time**。

### 6.1 候选选择

部署期对每一步:
1. 从已训好 actor 采 N 个候选:a_i = π(s_t) + δ_i,δ_i ~ N(0, σ_eval²),i=1..N
2. **σ_eval = Phase 1 选定的 σ_a**(同源约束 — NODE 在这个范围内已知精度)
3. 对每个候选用 NODE 算 1-step 预测:s̃_{t+1}^{(i)} = f_θ(s_t, a_i, w_t)
4. 算风险项:
   \[ R_θ(s_t, a_i) = \lambda_E \cdot \hat E(a_i) + \lambda_{smooth} \cdot \|a_i - a_{t-1}\|^2 \]
   (注:**不引入 P_abnormal** — 当前 env 是 wake nav,无 abnormal-pose 检测信号)
5. **Conservative score**:S_i = min(Q_1, Q_2)(s_t, a_i) − λ_R · R_θ(s_t, a_i)
6. 执行:a_t* = argmax_i S_i

N 起步 = 4,扫到 16 看 best-of-N 是否放大 Q 过估计(碰撞率 / 最小障碍距离监控)。

### 6.2 Evaluation 协议

verifier 改变 inference 分布,不可与 deterministic eval 直接比较。报告 4 组对比:

| 组 | Actor | Verifier | 目的 |
|---|---|---|---|
| baseline | ReBRAC(无 aug) | none | 已有 |
| aug-only | ReBRAC + AUVHamNODE-aug | none | Phase 1 主结果 |
| **aug + verifier** | ReBRAC + AUVHamNODE-aug | conservative-Q-risk | Phase 2A 主结果 |
| baseline + verifier(control) | ReBRAC(无 aug) | conservative-Q-risk | verifier 对任何 policy 都有用吗? |

**指标**:success rate / collision-free rate / min obstacle distance / mean+p95 latency / action jerk / N=1,4,8,16 best-of-N 曲线。

### 6.3 verifier 内部对比

固定 Phase 1 best variant + AUVHamNODE-aug,扫 verifier 形式:

| Verifier | 公式 | 目的 |
|---|---|---|
| Q only(baseline) | Q | best-of-N 上限 |
| Conservative Q | min(Q_1, Q_2) | 抑制 Q 过估计 |
| **Conservative Q-risk** | min(Q_1, Q_2) − λ_R R_θ | 主方法 |

---

## 7. Phase 2B(可选,仅 Phase 2A margin 小时):NODE-feature Critic(3-4 周,不默认做)

### 7.1 设计

critic input 从 (s, a) 扩展为 (s, a, φ_θ(s, a)),其中:
\[ \phi_\theta(s, a) = [\tilde s_{t+1} - s_t,\ \tilde\nu_{t+1},\ \Delta d_{goal}^\theta] \]

这是 Plan B §5.3.2 的简化版,但**只做一组对比**:"ReBRAC + aug" vs "ReBRAC + aug + NODE-feat critic"。

### 7.2 触发条件

- Phase 2A 通过
- 但 verifier 提升 margin < 一个 std,想要 critic-side 也吸收 NODE 信号

### 7.3 不做 IQL,不做 FQL

NODE feature 可独立叠到 ReBRAC critic 上,不需要换算法主干。

---

## 8. 横切关注点

### 8.1 是否用 ensemble for OOD signal

**Default:不用**。理由:
- AUVHamNODE 是 physics-structured,single model 的 OOD 行为受结构先验约束
- plain-MLP-ensemble 已作为 Phase 1 augmentation 对照存在,**不**作为 OOD signal 提供方
- 留作 Phase 1 出现 critic over-estimation 时的二级 fall-back(用 plain-MLP-ensemble 内部 disagreement 做 OOD penalty)

### 8.2 悲观主义强度

| Phase | 机制 | 强度 |
|---|---|---|
| 1 | ReBRAC 已有 dual BC penalty;无额外悲观主义 | 0 |
| 1 二级 | 出现 Q overestimation 时:OOD action distance penalty \(\tilde r = r - \lambda_{ood}\|a' - a^{data}\|^2\) | λ_ood 起步 0.01 |
| 1 三级 | 仍不稳:plain-MLP-ensemble disagreement 作 OOD signal | conditional |
| 2A | twin-Q min + risk penalty(verifier 内部) | inference-time only |

**绝对不要**默认上 MOPO uncertainty / COMBO conservative critic / MOBILE MBI / LEQ — 它们是为 plain NN ensemble 设计,且 Phase 0 测试 A 已经通过 NODE 自身精度判定 OOD region。

### 8.3 部署一致性表

| 量 | 训练 | 部署 |
|---|---|---|
| AUV state s_t | dataset / model rollout(1-step) | 真实测量 |
| Current flow w_t | dataset / wake_data query | DVL/ADCP 实测 |
| **Future flow w_{t+τ}** | **不使用**(1-step only) | **不使用** |
| Actor 输入 | (s_t, history if any) | 同 |
| Critic 输入 | Phase 1 默认普通 obs;Phase 2B 选项加 φ_θ(s, a) | 部署不需要 critic |
| AsymCritic privileged_obs | Phase 1 默认 OFF;Path B ablation 时 ON | 部署仍可算([u_eq, v_eq] 在测时刻 hull 多点采样) |

### 8.4 Reward 设计

**Phase 1-2 全程沿用现有 `efficiency_v2` reward**,不引入额外 reward shaping。理由:
- 减少混淆变量,让 model-based 增益清晰可归因
- 与 online thesis 主线、ReBRAC paper 主线 reward 一致,便于跨论文对比

---

## 9. 风险与 fall-back

| 风险 | 触发条件 | Fall-back |
|---|---|---|
| AUVHamNODE 跨域迁移失败 | Phase 0 A:整体 normalized MSE > 0.3 | **暂停整个方案**;考虑申请 AUVHamNODE 在当前 wake_data 上轻量 finetune;或换论文方向 |
| AUVHamNODE 跨域 partial fail | Phase 0 A:整体 < 0.2 但某桶 > 0.5 | augmentation 限定在 low-MSE region;paper claim 中等 |
| σ_a 扰动下 NODE 精度差 | Phase 0 B:所有 σ_a 都退化严重 | σ_a = 0.05 起步;升级 OOD penalty 强度 |
| Dataset action 单峰窄 | Phase 0 C:ratio < 0.05 | σ_a ≥ 0.2;若 B 不允许,augmentation 收益预期低 |
| 1-step augmentation 无收益 | Phase 1 不达标 | 检查 σ_a / mix ratio / dataset support;不行就 reframe 为 ReBRAC paper appendix ablation |
| AUVHamNODE 不胜 plain-MLP-ensemble | Phase 1 主对照 95% CI 跨 0 或下界 ≤ 0 | paper claim 降级为描述性;考虑 Phase 2A 找 verifier 端 novelty |
| **Critic Q overestimation 失控** | **§4.2 F:Δ > 15%** | **Phase 1 暂停,先调 σ_a / mix ratio 把 Δ 压回 15% 以下;若仍不行,触发 §8.1 plain-MLP-ensemble disagreement 作 OOD penalty** |
| **Augmented terminal-flag 反转率高** | §5.2 步骤 5 审计:反转率 > 5% | augmentation 限定在 mid-trajectory(过滤 dataset done_t=True ± 5 step),或换协议为"dropping terminal augmentation" |
| Phase 2A verifier 改善小 | Phase 2A margin < 1 std | 进 Phase 2B(NODE-feat critic),retraining |
| Held-out flow generalization gap 显著 | 任一 Phase 评估时 | 减少 augmentation ratio,或加 in-domain regularization |
| ReBRAC broad validation 还在跑,资源冲突 | Colab L4 quota 不足 | 本方案推迟到 broad validation 收敛后启动 |

---

## 10. 论文 Framing 与 Contribution Claim

### 10.1 v2.0 升级版 contribution claim

按"实验结果支持的强度"由弱到强:

**最低必有(即便 Phase 1 仅 AUVHamNODE-aug 通过)**:
> "We demonstrate that a frozen, physics-structured (port-Hamiltonian) 1-step Neural ODE prior, pretrained on out-of-distribution data, transfers to AUV offline RL augmentation in new flow domains without any finetuning, improving ReBRAC sample efficiency by X% at 25% data scale."

**中等(Phase 1 plain-MLP 对照通过)**:
> "Furthermore, the physics structure is the key enabler of cross-domain transfer: a parameter-matched plain-MLP ensemble pretrained on the same out-of-distribution data fails to transfer, whereas the port-Hamiltonian model retains predictive accuracy and downstream offline-RL augmentation gains."

**最强(Phase 2A verifier 通过)**:
> "An inference-time risk-aware verifier, scoring actor candidates by conservative Q minus a one-step physical risk term computed by the same NODE, further improves collision-free success rate by Y% with no retraining."

### 10.2 与已有文献的区分

| 文献线 | 与本方案的关系 |
|---|---|
| MOPO / COMBO / MOBILE / LEQ | 它们的 motivation 是 NN ensemble OOD,在 in-distribution dataset 上训练 dynamics。本方案是**out-of-distribution pretrained + frozen** + physics structure,是不同 setting |
| MBPO / SynthER | 它们是 in-domain augmentation。本方案是 cross-domain transfer |
| SHAC / DiffRL | 它们假设可微 simulator 给定。本方案不走 actor 端可微路径(Phase 3 SHAC 已删除) |
| DreamerV3 / TD-MPC2 | 它们用 latent world model 做 multi-step planning。本方案保留物理可解释性 + 1-step only |
| **Foundation models for control(RT-2 / Octo / TD-MPC2 cross-domain / Action Chunking Transformers)** | 它们是 large-scale data-driven cross-task / cross-embodiment transfer,依赖海量异构数据。本方案是 **small-scale physics-structured single-domain transfer**,setting 不同;且本方案的 "frozen + OOD pretrained" 不依赖大数据,而是依赖结构先验 |
| AUV path-following CMQL(Drones 2025) | 领域相关,但其算法主线偏 conservative model-based offline RL;本方案不走 conservatism heavy 路线 |

---

## 11. 与现有 research lines 的关系

| Research line | 状态 | 与本方案的关系 |
|---|---|---|
| Online RL thesis | 已下调,仅承担「环境可行性 + offline 数据源」角色 | 本方案的 dataset 来源由 online 线提供 |
| ReBRAC mainline + paper-readiness | 4/4 闭环(rev.8) | 本方案的 ReBRAC baseline 来自这里 |
| ReBRAC broad validation(S2 P1 等) | 在跑;**B1 sensor spoke** 已 paper-quality 闭环;**C1 BC-penalty sweep** 列为 future work §5.3 未排期 | 本方案的 dataset 选择和 reward 配置由这里产出 |
| **本方案(offline RL with frozen AUVHamNODE)** | **planning(v2.1 locked)** | fire-condition 见 §11.1 |

### 11.1 Fire-condition(v2.1 重写;2026-05-10 修订增补 §11.1.0 零号前置)

v2.0 原文 fire-condition "ReBRAC paper revision 收尾 + broad validation 至少 1 项 mechanism discriminator 闭环" 与 broad validation 实际排期不匹配:

- broad validation **B1 (sensor s1)** 已闭环,但**不是 mechanism discriminator**(B1 是协议变量改变,不是机制 ablation)
- broad validation **C1 BC-penalty sweep**(首选 mechanism discriminator)被列为 [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md) 的 future work §5.3,**未排期**

为避免无限期等待,v2.1 重新定义 fire-condition。

#### 11.1.0 零号前置(2026-05-10 增补,**所有路径共享的硬前置**)

下列 dynamics-model 侧基础设施**必须先入库**,否则即使 fire-condition 满足也无法启动 Phase 0:

| 资产 | 当前状态(2026-05-10 核查) | 责任方 |
|---|---|---|
| `auv_nav/auvhamnode.py`(wrapper) | ✗ 不存在 | 本方案作者编写,wraps 上游 checkpoint |
| `auv_nav/dynamics_ensemble.py`(wrapper) | ✗ 不存在 | 同上 |
| `scripts/generate_model_augmented_buffer.py` | ✗ 不存在 | 同上 |
| `tests/test_auvhamnode_wrapper.py` | ✗ 不存在 | 同上 |
| `tests/test_vehicle_oracle_phase0.py`(§4.2 B) | ✗ 不存在 | 同上 |
| `checkpoints/auvhamnode/<name>.pt` + metadata | ✗ 不存在 | **AUVHamNODE 上游工作 export** |
| `checkpoints/dynamics_ensemble/<name>.pt` + N 个 ensemble 成员 | ✗ 不存在 | 同上 |
| **§4.2 G 要求的 protocol audit table**(参数量、训练数据、训练 epoch、LR、ensemble size N) | ✗ 不存在 | AUVHamNODE 上游工作提供,本方案作者归档 |

**已就绪侧**:

| 资产 | 状态 |
|---|---|
| `wake_data/` | ✓ 5 组 wake field(dummy / sbs Re150 + Re250 / tandem Re150 + Re250),含 `_meta.json` 与 `_phase.npy` |
| `offline_data/` | ✓ broad_validation 多个 cell 的 dataset(crosscomp / worldcomp 等)已生成 |
| `auv_nav/rebrac.py` + `scripts/train_offline_rebrac.py` | ✓ ReBRAC mainline 已 paper-readiness 4/4 闭环(rev.8) |
| `auv_nav/replay.py` `DualBufferSampler` | ✓ RLPD 双 buffer 路径已实现 |
| `auv_nav/autopilot.py` `EquivalentCurrentModel` | ✓ Path B privileged_obs 重生成所需 |

**关键路径风险**:checkpoint export 时间不在本方案作者掌控之内,而是 AUVHamNODE 上游工作的 release 排期决定。**该 ETA 是整个方案启动的唯一外部依赖**。建议:
- 在 [`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) §3.4 增加一行 status note 跟踪 checkpoint 入库 ETA
- 若 checkpoint ETA 比 ReBRAC paper drafting 收尾还晚,wrapper / generate_model_augmented_buffer.py / 单测**可以提前**编写(用 mock NODE 跑通 pipeline),避免 checkpoint 入库后才开始 4 周 wrapper 工程

#### 11.1.1 三条 fire-condition 路径

**前提**:§11.1.0 零号前置全部满足。否则路径 A/B/C 全部 N/A。

| 路径 | 满足条件(在零号前置之上) | paper claim 支撑度 |
|---|---|---|
| **路径 A(优选)** | ReBRAC paper revision 收尾 + C1 BC-penalty sweep 闭环 | 完整论文级 |
| **路径 B(备选,C1 sweep 排期注定晚于 ReBRAC 投稿时启用)** | ReBRAC paper revision 收尾 + B1 已闭环(✓ 当前已满足) + 本方案 §4.2 D 测试同时承担 mechanism discriminator 角色 | 论文级,但需在 §10.1 paper claim 中说明 mechanism discriminator 由本方案自带 |
| **路径 C(应急)** | ReBRAC paper revision 收尾 + B1 已闭环 | appendix-level,作为 ReBRAC paper appendix ablation,无独立 mechanism 证据 |

**当前(2026-05-10)各路径前置达成度速查**:

| 前置项 | 状态 |
|---|---|
| §11.1.0 零号前置(checkpoint + wrapper + audit table) | ✗ 全部空白 |
| ReBRAC paper revision 收尾 | drafting 中(rev.3 已合并 broad_val §3.5;method/experiment section 编写中) |
| C1 BC-penalty sweep 闭环(路径 A) | ✗ 未排期(工程量小:1-seed probe ~5 L4-hour;5-seed 升级 ~25 L4-hour) |
| B1 已闭环(路径 B/C) | ✓ success 0.900±0.028,Δ=−0.2pp,5-seed 完整 |

#### 11.1.2 资源冲突管理

- 本方案 Phase 0–2A 累计 5–6 周(offline RL run 单 run ~30–45 min on L4,显著快于 online SAC)
- Colab L4 quota 与 broad validation 共享;启动时机由 §11.1.0 零号前置 + §11.1.1 路径选择联合决定
- **预算估算**:Phase 1 主对照 4 cell × 5 seed × 1.5M-grad-step ≈ 25 L4-hour;σ_a / mix ratio sweep 各 ~12 hour;总 Phase 1 约 60–80 L4-hour

---

## 12. 一句话总结

> **以冻结、out-of-distribution pretrained 的 AUVHamNODE 为 1-step 动力学先验,通过 (s, a) 邻域 augmentation 增强 ReBRAC,Phase 0 七项 ablation(A-G,v2.1 增 F=critic Q overestimation hard gate / G=MLP-ensemble protocol audit)验证跨域迁移可行后进 Phase 1,主对照 4 cell × 5 seed × paired bootstrap 95% CI,以"physics structure enables cross-domain transfer"为核心 paper claim;Phase 2A 加 inference-time verifier 提升 paper claim 强度;Phase 2B(NODE-feat critic retraining)和 v1.0 的 Phase 2 MVE / Phase 3 SHAC 全部删除;不引入 ensemble RL agent / MOPO / COMBO / IQL / FQL / SORL / map-aware obstacle 观测;5-6 周一篇 paper 级 contribution。**

---

## 附录 A. 各版本关键修订

### A.1 v2.0 → v2.1(2026-05-09,RL 专家 meta-review patch)

| 修订 | 优先级 | 落点 | 理由 |
|---|---|---|---|
| **新增** Phase 0 测试 F:critic Q overestimation hard gate | P0 | §4.2 F | 跨域 OOD frozen NODE 最大风险是 1-step prediction systematic bias 经 Bellman bootstrap 累积;v2.0 把它列为 Phase 1 二级 fall-back,实质风险被低估 |
| **明确** augmented `done` flag 协议(默认复用 dataset done_t) | P0 | §5.2 步骤 5 | `RewardModel.compute()` 实际签名需要 `reason / terminated / truncated`,这些是 env 终止逻辑输出,不是 (s, a, s')-pure 函数;v2.0 文本沉默会导致不同执行者实现不一致 |
| **改** "AUVHamNODE-aug ≥ MLP-ensemble-aug" 为 paired bootstrap 95% CI > 0 | P0 | §5.5 | v2.0 用 "≥" 缺统计检验;3-5 seed 下 1 个 outlier seed 可反转 mean,与 ReBRAC paper 主线 statistical protocol 不对齐 |
| **重写** Fire-condition,新增路径 B/C | P0 | §11.1 | v2.0 fire-condition 与 broad_val C1 sweep(future work §5.3 未排期)不匹配,严格解读下当前无 candidate 满足 |
| **新增** mix-ratio 1-seed pilot(Step 1)+ 主对照升 5-seed(Step 2) | P1 | §5.4 | v2.0 90:10 起点未经 pilot 验证,有 Phase 1 主跑跑出 null result 风险;主对照 3-seed 不对齐 ReBRAC mainline 的 5-seed anchor |
| **新增** Phase 0 测试 G:plain-MLP-ensemble checkpoint protocol audit | P1 | §4.2 G | v2.0 audit #3 "无需重训" 默认两份 checkpoint capacity-matched,无证据;若 mismatch,§10.1 中等档 paper claim 会被审稿人 attack |
| **明确** Phase 0 B vehicle.py oracle 协议 + §1.2.5 字面豁免 | P1 | §1.2、§4.2 B | v2.0 "重跑 vehicle.py" 与 §1.2.5 "不再访问 vehicle.py" 字面冲突;且 wake snapshot 协议未定义,影响 ground-truth MSE 物理含义 |
| **改** σ_a Gaussian + clip → truncated normal 或 squash;report effective σ | P1 | §4.2 B、§5.2 步骤 2 | saturated dataset action 上 clip 引入 truncation artifact,使实际扰动幅度远小于 σ_a,污染 σ_a-MSE 曲线分析 |
| **明确** Probe wake field 时间 index = dataset frame t+1 snapshot | P1 | §5.2 步骤 4 | v2.0 文本未明确 augmented s̃_{t+1} 用 w_t / w_{t+1} / 插值,会偷偷引入时间错位 bias |
| **明确** Path B privileged_obs 必须走 hull-integral 流程(非单点) | P1 | §5.3 hint | v2.0 简化为单点 wake query,Path B 实际工程量被低估(5-point hull sampling + EquivalentCurrentModel weighted integration) |
| **明确** Reward 重算分项处理(progress 重算 + terminal 沿用 dataset) | P1 | §5.2 步骤 5 | 与"明确 done flag 协议"配套,确保 reward / done 一致性 |
| **(2026-05-10 amendment)新增** §11.1.0 零号前置:checkpoint + wrapper + audit table 入库 | P0 | §11.1.0 | v2.1(2026-05-09)原文 §11.1 路径 A/B 条件未明示 dynamics-model 侧基础设施依赖,实际所有路径都共享此硬前置;2026-05-10 仓库现状核查显示 checkpoint / wrapper / audit table 全部空白,该 ETA 是整个方案启动的唯一外部依赖 |

### A.2 v1.0 → v2.0(2026-05-08)

| 修订 | v1.0 → v2.0 | 理由 |
|---|---|---|
| **删除** Phase 2 MVE 多步 rollout | ✓ | 工程量大;流场 propagation 风险高;1-step only 已足以支撑论文 |
| **删除** Phase 3 SHAC pathwise gradient | ✓ | 同上;且 framing 上脱离严格 offline RL,审稿风险大 |
| **删除** XQL/FQL/SORL/IQL 算法栈 | ✓ | Plan B(`NODE_IQL_FQL_SORL_revised_roadmap_v3.md`)的算法 zoo 与 ReBRAC paper-ready 不连续;FQL 多模态前提在确定性 baseline 数据上无 evidence;observation contract 变更要求大 |
| **新增** plain-MLP-ensemble 对照 | ✓ | novelty key — "physics structure is the enabler";audit #3 让对照变便宜(无需重训) |
| **升级** paper claim 至 cross-domain transfer | ✓ | audit #4:AUVHamNODE 训练数据 ≠ 当前 wake_data;升级后 framing 更新颖 |
| **升级** Phase 0 测试 A 为 domain-shift 特征化 | ✓ | 跨域迁移成立与否是 Phase 0 必须回答的硬问题;原 v1.0 测试 A 的 closed-loop value error 在 1-step-only 设定下不再适用 |
| **新增** Phase 0 测试 B(σ_a 扰动 MSE 曲线) | ✓ | augmentation 用的是 (s, a + ε),NODE 在 (s, a) 上的精度不等于在扰动 action 上的精度 |
| **新增** Phase 0 测试 E(dataset audit) | ✓ | 来自 Plan B §3.2;为 Phase 2A risk verifier R_θ 提供分布信息 |
| **明确** augmentation pipeline 必须有 wake field 访问 | ✓ | v1.0 没说清楚;probe channels 必须在新 pose 重新查询 |
| **降级** AsymCritic 为 Phase 1 末尾 ablation | ✓ | 默认开启会让 aug 收益和 privileged 收益混在一起,审稿人会问"是 aug 起的作用还是 AsymCritic?" |
| **明确** mix ratio 90:10 起步 + sweep | ✓ | offline aug 应 real-heavy;75:25 太激进 |
| **明确** OOD penalty λ_ood = 0 起步 | ✓ | 与 ReBRAC 已有 dual BC penalty 不重复正则 |
| **简化** R_θ 移除 P_abnormal | ✓ | 当前 env 是 wake nav,无 abnormal-pose 检测信号 |

---

## 附录 B. 已被 v2.0 吸收/取代的草案

| 文档 | 主线 | 处置 |
|---|---|---|
| [`offline_mbrl_plan/AUV_REBRAC_NeuralODE_quick_route_v3.md`](offline_mbrl_plan/AUV_REBRAC_NeuralODE_quick_route_v3.md) | REBRAC + ODE 1/3-step aug | deprecated:1-step aug 思路被吸收;multi-step rollout 中 w 缺失说明被新方案的 1-step-only 约束消除 |
| [`offline_mbrl_plan/AUV_REBRAC_NeuralODE_OfflineRL_v2_report.md`](offline_mbrl_plan/AUV_REBRAC_NeuralODE_OfflineRL_v2_report.md) | REBRAC + ODE ensemble + MOBILE/COMBO/LEQ + MPPI | deprecated:conservative critic 概念留作 fall-back;default 全套悲观主义弃用 |
| [`offline_mbrl_plan/AUV_offline_RL_neural_ODE_report.md`](offline_mbrl_plan/AUV_offline_RL_neural_ODE_report.md) | LEQ 主算法 + Neural SDE + recurrent belief + MPC | deprecated:belief encoder 想法留作 future;LEQ / Neural SDE / 完整 POMDP framing 弃用 |
| [`offline_mbrl_plan/auv_fql_flow_offline_rl_method_selection_report_zh.md`](offline_mbrl_plan/auv_fql_flow_offline_rl_method_selection_report_zh.md) | model-free flow-based offline RL | deprecated:macro-action / MPC 安全层留作部署期参考;FQL/FAC/ReFORM 主线弃用 |
| [`offline_mbrl_plan/offline_mbrl_for_auv_navigation_report.md`](offline_mbrl_plan/offline_mbrl_for_auv_navigation_report.md) | SHAC teacher-student + 可微 rollout | deprecated:conditional dynamics + exogenous input framing 已被本方案吸收;SHAC 作首选 / teacher-student 蒸馏弃用 |
| [`offline_mbrl_plan/NODE_IQL_FQL_SORL_revised_roadmap_v3.md`](offline_mbrl_plan/NODE_IQL_FQL_SORL_revised_roadmap_v3.md) | NODE + IQL/FQL/SORL 算法栈 + adaptive test-time compute | deprecated:dataset coverage audit / "NODE as 1-step only" 约束 / risk-aware verifier 三个工具被吸收;IQL/FQL/SORL 算法主轴弃用(observation contract 不与本仓库现状对齐 + FQL 多模态前提无 evidence) |

---

*文档版本:v2.1(2026-05-09 RL 专家 meta-review patch,2026-05-10 §11.1.0 零号前置 amendment)*
*下次更新:Phase 0 ablation 完成后写入实测数值;P2 条目(critic 配对、held-out flow、σ_eval、verifier sampling、ODE solver、Q overestimation monitor、近期 foundation model baseline 对比、25% baseline ceiling sanity)在作者下一轮 review 时整合。*
