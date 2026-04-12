# RLPD：利用离线数据加速 AUV 流场导航策略学习

> 本文档对应代码库中 RLPD（RL with Prior Data）方法的引入，详细阐述算法选择依据、实现规划与实验设计，可作为代码实现和论文写作的参考。
> 前序：离线-在线 RL 综述见 `docs/world_model_and_offline_rl_survey.md` Part 3

---

## 1. 背景与动机

### 1.1 为什么在本项目中引入离线 RL

本项目当前使用纯在线 SAC 训练 AUV 在卡门涡街中的导航策略。训练初期，智能体需要经历大量随机探索（`random_steps=2000–5000`），才能积累足够的有效经验开始有意义的策略更新。在逆流欠驱动场景（`U_flow=1.5 m/s`, `target_speed=1.5 m/s`）下，随机策略几乎不可能到达目标，导致早期 replay buffer 中缺乏成功轨迹，收敛速度受限。

与此同时，项目已具备多个可用的 baseline 策略（`baselines.py`），它们虽然性能有限，但已能产生结构化的导航行为：

- `GoalSeekPolicy`：始终朝向目标，提供基础的目标导向行为
- `CrossCurrentCompensationPolicy`：利用探针读数补偿横向流，体现简单的流场响应
- `WorldFrameCurrentCompensationPolicy`：利用真实局部流速进行航向补偿（特权信息）
- `PrivilegedCorridorPolicy`：利用完整流场信息进行路线选择（强特权信息）

这些策略的 rollout 数据可以作为"先验经验"加速 SAC 的训练。

### 1.2 关键约束：仿真代价极低

本项目的流场数据由 LBM 预生成，环境步进（6-DOF RK4 积分 + 流场插值）计算代价极低。这意味着：

- **纯离线 RL 方法（CQL、IQL、TD3+BC）的核心价值——减少真实环境交互——在此不成立**
- 离线数据的价值不在于"省交互"，而在于：（1）初期收敛加速；（2）探索覆盖度提升；（3）数据质量与分布偏移效应的研究价值
- 不应将 offline RL 定位为"性能突破"手段，而应定位为"训练效率工具 + 数据科学研究载体"

---

## 2. 算法选择：为什么是 RLPD

### 2.1 候选方法对比

| 方法 | 核心机制 | 是否需要离线预训练 | 代码改动量 | 与现有 SAC 兼容性 | 离线→在线过渡稳定性 |
|------|---------|-----------------|-----------|-----------------|-------------------|
| **RLPD** | SAC + 双 buffer 对称采样 | **否** | **~80 行** | **完全兼容** | **无过渡问题** |
| CQL | Q 函数保守惩罚 | 是 | ~300 行 | 需改 critic loss | 差（Q 值崩溃） |
| Cal-QL | CQL + 校准下界 | 是 | ~350 行 | 需改 critic loss | 中（修复了 CQL 崩溃） |
| IQL | 期望值回归 + SARSA 型 Q | 是 | ~400 行 | 需重写 critic | 中（Q 值重标定冲击） |
| TD3+BC | TD3 + BC 正则化 | 是 | ~200 行 | 需切换到 TD3 | 差（BC 项限制改进） |
| WSRL | BC 预热 + 纯在线 | 预热阶段 | ~150 行 | 较兼容 | 好 |

### 2.2 选择 RLPD 的理由

**RLPD（Ball et al., ICML 2023）** 的核心发现是：不需要任何特殊的离线 RL 算法。标准 SAC 配合三个设计选择即可有效利用离线数据：

1. **对称采样（symmetric sampling）**：每个 mini-batch 中 50% 来自离线 buffer，50% 来自在线 buffer
2. **Critic 加 LayerNorm**：防止 Q 值在分布外（OOD）动作上外推崩溃
3. **大 Q ensemble**（原文用 10 个 critic）：降低 Q 值过估计偏差

**与本项目的契合度：**

- 本项目的 SAC 已支持 LayerNorm（`--use-layernorm`）和 Dropout（`--dropout-rate`），满足条件 2
- DroQ 配置（LayerNorm + Dropout + UTD=4）在功能上部分替代了大 ensemble 的作用（详见 §2.3）
- 核心实现只需：创建第二个 `TransitionReplay` + 修改 `sample_batch` 为双 buffer 采样，约 80 行代码
- 不需要离线预训练阶段，不修改 SAC 的 actor/critic 损失函数，不引入新的超参数敏感性

### 2.3 关于 Q Ensemble 规模的讨论

RLPD 原文使用 10 个 critic 来控制 OOD 动作的 Q 值过估计。本项目仅有 2 个 critic（`q1`, `q2`），但已具备以下正则化机制：

| 机制 | 原文 RLPD | 本项目 DroQ 配置 | 功能等价性 |
|------|----------|----------------|-----------|
| LayerNorm | 有 | 有（`--use-layernorm`）| 完全等价 |
| Q ensemble 大小 | 10 | 2 | **不等价**——需要观察 |
| Dropout | 无 | 有（`--dropout-rate 0.01`）| 部分替代 ensemble 效果 |
| UTD (updates per step) | 20 | 4（DroQ 配置）| 部分补偿 |

**实施建议**：先用现有 2-critic + DroQ 配置运行 RLPD。若观察到 Q 值过估计或训练不稳定，再考虑增加 critic 数量至 5 或 10。增加 critic 的代码改动集中在 `SACAgent.__init__` 和 `SACAgent.update` 中，属于可控的增量修改。

### 2.4 不推荐的方向

- **CQL / IQL / TD3+BC**：需要独立的离线预训练阶段，且离线→在线过渡存在已知不稳定性（Q 值崩溃 / 重标定冲击）。在仿真代价极低的场景下，额外的工程复杂度不值得
- **SHARSA**：需要超大规模离线数据集，不适合本项目的数据量级
- **DreamerV3 / TD-MPC2**：属于世界模型方法（model-based RL），不属于 offline RL 范畴，应作为独立研究线规划

---

## 3. 离线数据来源分析

### 3.1 Baseline 策略的特权性分级

本项目的 baseline 策略在决策时使用不同级别的信息，这直接影响其生成的离线数据的分布特性：

| 策略 | 决策信息来源 | 特权等级 | 数据质量预期 |
|------|------------|---------|------------|
| `GoalSeekPolicy` | `env.goal_heading()`（可从 obs 推导） | **无特权** | 低——无流场感知，逆流性能差 |
| `CrossCurrentCompensationPolicy` | `env.decode_observation(obs)` 中的探针读数 | **无特权** | 中低——简单反应式补偿 |
| `WorldFrameCurrentCompensationPolicy` | `env.last_equivalent_current_world`（AUV 位置的真实流速） | **中度特权** | 中高——真实流速补偿，但该信息部署时不可用 |
| `PrivilegedCorridorPolicy` | `env.flow_sampler.sample_world()`（任意位置的完整流场） | **强特权** | 高——全局路线规划，但分布偏移最严重 |

**关键洞察**：特权策略生成的动作分布与基于探针观测的可部署策略存在本质差异。例如，`PrivilegedCorridorPolicy` 可能选择一条绕行涡旋的路线，而仅凭探针观测的策略在相同位置可能无法判断应该绕行。这种**决策信息不对等**会导致离线数据中的 `(obs, action)` 对在可部署策略的分布之外，Q 函数可能学到"在某些 obs 下选择某个 action 有高回报"，但该 action 实际上只有在拥有特权信息时才是合理的。

### 3.2 数据维度约束

不同 probe layout 产生不同维度的观测：

| Probe Layout | 探针数 | 观测维度 | 说明 |
|-------------|--------|---------|------|
| `s0` | 1 | 10-D（8 base + 1×2 velocity） | DVL 基线 |
| `s1` | 2 | 12-D（8 base + 2×2 velocity） | DVL + 短程前向 ADCP |
| `s2` | 4 | 16-D（8 base + 4×2 velocity） | DVL + 长程 ADCP + 侧向梯度 |

**离线数据必须与在线训练使用相同的 probe layout。** 不同 layout 的数据不能直接混入同一个 replay buffer（维度不匹配）。因此，离线数据收集需要针对目标 probe layout 分别进行。

### 3.3 数据量估计

参考 RLPD 原文在 D4RL 上的实验，离线数据量在 100k–500k transitions 时提供实质性收益。本项目中：

- 单个 episode 约 100–480 步（`max_episode_time_s=240s`, `control_dt=0.5s`）
- 中等质量 baseline（`WorldFrameCompensation`）的成功率约 30–60%（逆流工况）
- 收集 100k transitions 需要约 200–1000 个 episode

**建议**：每个 baseline 策略收集 500–1000 个 episode（约 50k–200k transitions），覆盖多个随机种子和起终点配置。

---

## 4. 实现规划

### 4.1 总体架构

```
新增文件：
  scripts/collect_offline_data.py     # 离线数据收集脚本

修改文件：
  auv_nav/replay.py                   # 新增 load_from_npz() + DualBufferSampler
  scripts/train_sac.py                # 新增 --offline-data, --offline-ratio 参数
  scripts/run_suite.py                # 新增 RLPD METHOD_SPECS 条目

数据产出目录：
  offline_data/                       # 离线数据集（gitignore）
    ├── goalseek_s0_hard_U1p50/
    │   ├── transitions.npz
    │   └── metadata.json
    ├── worldcomp_s0_hard_U1p50/
    │   ├── transitions.npz
    │   └── metadata.json
    └── privileged_s0_hard_U1p50/
        ├── transitions.npz
        └── metadata.json
```

### 4.2 离线数据收集脚本

**文件**：`scripts/collect_offline_data.py`

**职责**：运行指定 baseline 策略，收集 `(obs, action, reward, cost, next_obs, done)` 元组并存储为 `.npz` 文件。

**核心逻辑**：

```python
# 伪代码
env = make_planar_env(flow_path, probe_layout=probe_layout, history_length=history_length)
policy = build_baseline_policy(policy_name)  # GoalSeek / WorldComp / Privileged / ...

all_obs, all_actions, all_rewards, all_costs, all_next_obs, all_dones = [], [], [], [], [], []

for ep in range(num_episodes):
    obs, info = env.reset(seed=seed + ep, options=reset_options)
    done = False
    while not done:
        action = policy.act(env, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # 存储 transition
        all_obs.append(obs)
        all_actions.append(action)
        all_rewards.append(reward)
        all_costs.append(info["step_safety_cost"])
        all_next_obs.append(next_obs)
        all_dones.append(terminated)
        obs = next_obs
        done = terminated or truncated

np.savez_compressed(output_path, obs=..., actions=..., rewards=..., ...)
# 同时保存 metadata.json：策略名、probe layout、流场路径、episode 数、transition 数、成功率等
```

**CLI 接口**：

```bash
python -m scripts.collect_offline_data \
    --policy worldcomp \
    --flow wake_data/wake_v8_U1p50_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --difficulty hard \
    --target-speed 1.5 \
    --episodes 500 \
    --seed 0 \
    --output-dir offline_data/worldcomp_s0_hard_U1p50
```

支持的策略名称映射：

| CLI 名称 | 类 |
|---------|---|
| `goalseek` | `GoalSeekPolicy` |
| `crosscomp` | `CrossCurrentCompensationPolicy` |
| `worldcomp` | `WorldFrameCurrentCompensationPolicy` |
| `privileged` | `PrivilegedCorridorPolicy` |

### 4.3 Replay Buffer 扩展

**文件**：`auv_nav/replay.py`

**新增 1**：`TransitionReplay.load_from_npz(path)` 方法

从 `.npz` 文件加载离线数据到 buffer。需要校验 `obs_dim` 和 `action_dim` 一致性。加载后 `self.size` 设为加载的 transition 数，`self.ptr` 设为 `self.size`（在固定离线 buffer 场景下不再追加新数据）。

**新增 2**：`DualBufferSampler` 类

```python
class DualBufferSampler:
    """RLPD-style symmetric sampling from offline + online buffers."""

    def __init__(
        self,
        offline_buffer: TransitionReplay,
        online_buffer: TransitionReplay,
        offline_ratio: float = 0.5,
    ) -> None:
        self.offline = offline_buffer
        self.online = online_buffer
        self.offline_ratio = offline_ratio

    def ready(self, batch_size: int) -> bool:
        online_needed = max(1, int(batch_size * (1 - self.offline_ratio)))
        return self.online.ready(online_needed) and len(self.offline) > 0

    def sample_batch(self, batch_size: int, device) -> dict[str, torch.Tensor]:
        n_offline = int(batch_size * self.offline_ratio)
        n_online = batch_size - n_offline
        offline_batch = self.offline.sample_batch(n_offline, device)
        online_batch = self.online.sample_batch(n_online, device)
        return {k: torch.cat([offline_batch[k], online_batch[k]], dim=0) for k in offline_batch}
```

### 4.4 训练脚本修改

**文件**：`scripts/train_sac.py`

**新增 CLI 参数**：

```python
parser.add_argument(
    "--offline-data", type=Path, default=None,
    help="Path to offline transitions.npz (RLPD mode).",
)
parser.add_argument(
    "--offline-ratio", type=float, default=0.5,
    help="Fraction of each batch drawn from offline buffer (default: 0.5).",
)
```

**训练循环修改**（仅在 `--offline-data` 提供时生效）：

```python
# 初始化阶段
if args.offline_data:
    offline_replay = TransitionReplay(obs_dim, action_dim, TransitionReplayConfig(...))
    offline_replay.load_from_npz(args.offline_data)
    sampler = DualBufferSampler(offline_replay, replay, args.offline_ratio)
    print(f"[rlpd] loaded {len(offline_replay)} offline transitions, ratio={args.offline_ratio}")
else:
    sampler = None

# 更新阶段（替换原来的 replay.sample_batch 调用）
if sampler is not None and sampler.ready(train_cfg.batch_size):
    batch = sampler.sample_batch(train_cfg.batch_size, agent.device)
else:
    batch = replay.sample_batch(train_cfg.batch_size, agent.device)
last_update = agent.update(batch)
```

**注意**：当 `--offline-data` 未提供时，训练行为与现有纯在线 SAC 完全一致——不引入任何性能回归。

### 4.5 实验套件注册

**文件**：`scripts/run_suite.py`

新增 `METHOD_SPECS` 条目：

```python
"rlpd_worldcomp": MethodSpec(
    key="rlpd_worldcomp",
    train_module="scripts.train_sac",
    extra_args=(
        "--use-layernorm", "--history-length", "1",
        "--offline-data", "offline_data/worldcomp_s0_hard_U1p50/transitions.npz",
        "--offline-ratio", "0.5",
    ),
    description="RLPD: SAC + WorldFrameCompensation offline data (50/50).",
),
"rlpd_privileged": MethodSpec(
    key="rlpd_privileged",
    train_module="scripts.train_sac",
    extra_args=(
        "--use-layernorm", "--history-length", "1",
        "--offline-data", "offline_data/privileged_s0_hard_U1p50/transitions.npz",
        "--offline-ratio", "0.5",
    ),
    description="RLPD: SAC + PrivilegedCorridor offline data (50/50).",
),
```

### 4.6 Checkpoint 与日志扩展

在 `save_training_state` 的 `extra_state` 中记录 RLPD 配置：

```python
extra_state={
    "algorithm": "rlpd" if args.offline_data else "sac",
    "offline_data_path": str(args.offline_data) if args.offline_data else None,
    "offline_ratio": args.offline_ratio if args.offline_data else None,
    "offline_transitions": len(offline_replay) if args.offline_data else 0,
    ...
}
```

在 `train_log.jsonl` 中增加字段（可选）：离线 buffer 采样次数、在线 buffer 实际大小等。

---

## 5. 实验设计

### 5.1 实验矩阵

**基础对比（先做）**：验证 RLPD 在本任务中的基本效果

| 实验 ID | 方法 | 离线数据 | 配置 |
|---------|------|---------|------|
| `baseline_sac` | 纯在线 SAC | 无 | `--use-layernorm --history-length 1` |
| `rlpd_goalseek` | RLPD | GoalSeek（无特权） | `--offline-ratio 0.5` |
| `rlpd_worldcomp` | RLPD | WorldFrameComp（中度特权） | `--offline-ratio 0.5` |
| `rlpd_privileged` | RLPD | PrivilegedCorridor（强特权） | `--offline-ratio 0.5` |

统一条件：`--difficulty hard --target-speed 1.5 --probe-layout s0`，5 个种子，200k 步。

**数据质量消融**：

| 实验 ID | 离线数据 | 预期效果 |
|---------|---------|---------|
| `rlpd_goalseek` | GoalSeek 数据 | 有限加速——数据质量低，但无分布偏移 |
| `rlpd_worldcomp` | WorldFrameComp 数据 | 显著加速——数据质量中高，轻度分布偏移 |
| `rlpd_privileged` | PrivilegedCorridor 数据 | **待验证**——数据质量高，但分布偏移严重 |
| `rlpd_mixed` | 上述三者混合 | 综合效果 |

**采样比例消融**：

| 实验 ID | 离线比例 | 预期效果 |
|---------|---------|---------|
| `rlpd_r25` | 25% offline / 75% online | 较弱加速，分布偏移影响小 |
| `rlpd_r50` | 50% / 50%（RLPD 默认） | 标准 RLPD 效果 |
| `rlpd_r75` | 75% offline / 25% online | 强初期加速，但后期可能被低质量数据拖累 |

### 5.2 核心研究问题

**Q1：在仿真代价低时，离线数据的价值有多大？**

对比 `baseline_sac` vs `rlpd_worldcomp` 的收敛曲线。预期：初期 2–5x 收敛加速，但渐近性能无显著差异。

**Q2：特权数据的分布偏移在 POMDP 环境中如何表现？**

对比 `rlpd_worldcomp` vs `rlpd_privileged`。这是本实验的核心研究贡献。

假设：`PrivilegedCorridorPolicy` 的路线选择基于全局流场信息，而训练中的 actor 只有探针观测。当 Q 函数在特权数据上学习到"某些 obs 下某些 action 有高回报"时，这些 action 实际上只有在拥有全局流场信息时才合理。在线训练阶段，actor 尝试复现这些 action 但缺乏决策依据，可能导致：

- **正迁移**：特权数据覆盖了在线探索难以到达的高回报区域，为 Q 函数提供更好的价值估计
- **负迁移**：actor 无法稳定复现特权策略的行为，Q 值过估计导致策略退化

这种正/负迁移的平衡点取决于特权策略的行为与探针可观测信息的相关程度——如果特权策略的大部分"好决策"恰好也能从探针信号中推断出来（例如，特权策略选择低流速走廊，而探针也能感知到局部流速下降），则正迁移占主导；反之则负迁移。

**Q3：不同探针布局下，离线数据的边际价值是否不同？**

对 s0、s1、s2 分别收集离线数据并训练 RLPD。预期：s0（信息最少）从离线数据中获益最多，因为其随机探索效率最低；s2（信息最丰富）获益最少。

### 5.3 评估指标

| 指标 | 含义 | 来源 |
|------|------|------|
| `eval_success_rate` | 到达目标的成功率 | `eval_log.csv` |
| `eval_return` | 平均累积回报 | `eval_log.csv` |
| `eval_time_s` | 平均到达时间 | `eval_log.csv` |
| 收敛步数（达到 50% 成功率） | 学习效率 | 从 eval 曲线提取 |
| Q 值均值与方差 | Q 函数健康度 | `train_log.jsonl` 中的 `mean_q` |
| alpha（温度参数） | 探索-利用平衡 | `train_log.jsonl` 中的 `alpha` |

### 5.4 对照与统计

- 每组实验运行 5 个种子（42, 43, 44, 45, 46）
- 报告均值 ± 标准差
- 收敛曲线使用平滑窗口（10 eval 点）
- 显著性检验：对最终 eval 指标使用 Welch t-test（样本量小，不假设等方差）

---

## 6. 实施步骤

### Step 1：离线数据收集基础设施

1. 新建 `scripts/collect_offline_data.py`
2. 实现 CLI 参数解析（policy、flow、probe-layout、difficulty、target-speed、episodes、seed、output-dir）
3. 实现 baseline 策略名称到类的映射
4. 运行 baseline 策略，收集 transitions 并存储为 `.npz`
5. 生成 `metadata.json`（策略名、参数、统计信息）
6. 在 `.gitignore` 中添加 `offline_data/`

### Step 2：Replay Buffer 扩展

1. 在 `TransitionReplay` 中新增 `load_from_npz(path)` 方法
2. 新增 `DualBufferSampler` 类
3. 编写单元测试：验证 `load_from_npz` 的维度校验、`DualBufferSampler` 的采样比例正确性

### Step 3：训练脚本集成

1. 在 `train_sac.py` 中新增 `--offline-data` 和 `--offline-ratio` 参数
2. 修改训练循环，在 `--offline-data` 提供时使用 `DualBufferSampler`
3. 扩展 `extra_state` 记录 RLPD 配置
4. 确保 `--offline-data` 未提供时行为与原版完全一致

### Step 4：数据收集与验证

1. 生成流场数据（如尚未存在）
2. 收集各 baseline 策略的离线数据
3. 检查离线数据统计（transition 数、成功率、回报分布）
4. 运行快速 pilot 实验（20k 步）验证 RLPD 训练管线无 bug

### Step 5：正式实验

1. 运行基础对比实验（§5.1 实验矩阵）
2. 运行数据质量消融
3. 运行采样比例消融
4. 收集和可视化结果

### Step 6：（可选）方法扩展

根据 Step 5 的结果决定：

- 若 RLPD 效果显著 → 增加 Q ensemble 规模实验（5/10 个 critic）
- 若特权数据负迁移严重 → 尝试优势加权采样（AWR 风格）替代均匀采样
- 若需要对比基线 → 实现 Cal-QL（在 CQL 基础上改一行校准下界）

---

## 7. 与论文写作的关联

### 7.1 可能的叙事结构

本实验的论文贡献不在于"RLPD 比 SAC 快"——这在仿真环境中说服力有限。更有价值的叙事是：

> 在部分可观测的非定常流场环境中，研究不同信息特权等级的先验数据对策略学习的影响：特权策略（拥有全局流场信息）生成的离线数据，在训练仅有局部探针感知的策略时，是否会因分布偏移而产生负迁移？

这个问题在 AUV 领域有直接应用意义：实际部署前可能有仿真环境中的"理想策略"数据，但部署时传感器信息远不如仿真完整。理解这种信息不对称对数据利用的影响，是一个有实用价值的研究问题。

### 7.2 可支撑的论文图表

| 图表 | 内容 | 对应实验 |
|------|------|---------|
| 收敛曲线对比 | SAC vs RLPD（不同数据源）的 eval_success_rate 曲线 | §5.1 基础对比 |
| 数据质量柱状图 | 不同 baseline 数据的 RLPD 最终性能对比 | §5.1 数据质量消融 |
| 分布偏移分析 | Q 值均值/方差随训练步数的变化（特权 vs 非特权数据） | §5.2 Q2 |
| 采样比例敏感性 | 不同 offline_ratio 下的收敛速度与最终性能 | §5.1 采样比例消融 |
| 探针布局交互效应 | s0/s1/s2 在 RLPD 下的边际收益对比 | §5.2 Q3 |

---

## 参考文献

- **RLPD**：Ball et al., "Efficient Online Reinforcement Learning with Offline Data," ICML 2023. arXiv: 2302.02948
- **DroQ**：Hiraoka et al., "Dropout Q-Functions: Doubly Efficient Reinforcement Learning," ICLR 2022
- **Cal-QL**：Nakamoto et al., "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning," NeurIPS 2023. arXiv: 2303.05479
- **WSRL**：Zhou et al., "Warm Start RL," ICLR 2025. arXiv: 2412.07762
- **D4RL**：Fu et al., "D4RL: Datasets for Deep Data-Driven Reinforcement Learning," arXiv 2004.07219
- **MAIOOS**：Multi-AUV navigation with offline-online RL, Ocean Engineering 2025
