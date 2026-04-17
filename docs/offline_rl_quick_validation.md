# 离线 RL 快速验证方案

> 文档版本：2026-04-17 rev.3
> 定位：`offline_rl_implementation_plan.md` 的 Phase 0，目标是在 1-2 天内得到**可信**的离线 RL 先验结论
> 核心问题：**在当前 repo 的数据接口和评估协议下，纯离线 RL 是否能稳定优于纯 BC，并在 deployable teacher 数据上显示真实价值？**

---

## 0. 当前实现状态

状态标签说明：

- `已实现`：代码已落地，并已做最小静态检查或 smoke test。
- `部分实现`：已有基础代码支持，但未形成完整自动化闭环，或只覆盖文档中的一部分范围。
- `未实现`：文档中的建议尚未编码。
- `未跑实验`：代码已可执行，但仓库中尚未提交对应 Phase 0 正式实验结果。

截至目前，Phase 0 **没有全部实现完**。当前状态更准确地说是：

- `已实现`：`TD3+BC` agent、`train_offline.py`、`evaluate_offline.py`、`evaluate_baseline_on_manifest.py`、observation normalizer、checkpoint 中的 `obs_normalizer` 持久化、`privileged-critic` 可选协议、Phase 0 启动脚本 [`scripts/run_offline_td3bc_phase0.sh`](../scripts/run_offline_td3bc_phase0.sh)。
- `部分实现`：训练/评估协议校验、离线数据 metadata 对齐检查、`best agent` 选择与最终评估。
- `未实现`：support-broadened collector（如 `--action-noise-std`）、`mixed-deployable` 数据收集、holdout/index-aware sampler。
- `未跑实验`：文档中 Step A / B / C 的正式实验结论还没有在仓库内固化。

说明：

- 文档中的命令前缀请按运行环境替换，可用 `python`，也可用 `conda run -n <env> python`。
- 下文标题后的状态标签，表示“当前代码实现状态”，不是“研究上已经得到结论”。

---

## 一、总体判断

本阶段仍然只做一个算法：**TD3+BC**。

原因不变：

- 实现最小，最适合作为纯离线 RL 管线的第一块试金石。
- 可以直接复用现有 `replay`、环境构造和 [`evaluate_agent()`](../scripts/train_utils.py)。
- `alpha = 0` 天然给出一个 MLP BC 下界。

但要把这句话说精确：

- **可以复用**：`TransitionReplay`、`make_planar_env()`、`evaluate_agent()`、manifest 工具；
- **不能直接复用**：现有 [`scripts/evaluate.py`](../scripts/evaluate.py)，因为它是 SAC 专用。

Phase 0 不能照搬通用 offline RL recipe。必须先锁死下面四件事，否则实验结果不可解释：

1. **数据收集、训练、评估的观测协议必须完全一致。**
2. **固定 benchmark 必须真的固定到同一个 flow case 和同一个 manifest。**
3. **deployable 结论必须先在 `crosscomp` 这类非特权 teacher 数据上建立。**
4. **`privileged_obs` 的使用语义必须先定义清楚。**

---

## 二、Phase 0 到底要回答什么

### 2.1 主问题：deployable viability

在固定离线数据集上训练 TD3+BC，是否能在固定评估集上稳定优于 `alpha=0` 的纯 BC？

这个问题的主结论应来自：

- `crosscomp` 数据；
- 不使用 privileged critic 的 deployable protocol。

### 2.2 次要问题：teacher gap

如果数据来自 `worldcomp`，TD3+BC 是否能接近甚至超过该 teacher？

这里必须注意：`worldcomp` 策略直接使用环境内部的等效流信息，不是仅由 actor 观测 `obs` 就能完整恢复的信号。因此：

- **超过 `worldcomp`** 是强阳性结果；
- **接近 `worldcomp`** 仍然是有价值结果；
- **低于 `worldcomp`** 不能直接推出“纯离线 RL 不行”，也可能是 teacher 信息优势导致。

### 2.3 当前 repo 的 `privileged_obs` 只覆盖局部等效流

当前环境输出的 `privileged_obs` 只是 body-frame equivalent flow 的前两维 `[u_eq, v_eq]`。

这意味着：

- 它与 `worldcomp` 使用的局部等效流信息强相关；
- 但它**不等于** `privileged` policy 的 corridor-level wake access。

因此在 Phase 0：

- privileged critic 对 `worldcomp` gap 的分析是有意义的；
- 但不能把它当成“足以消除所有 privileged teacher gap”的机制。

### 2.4 Phase 0 的判定标准

建议把结论分成两层：

| 结论层级 | 判断标准 | 含义 |
|---|---|---|
| 管线成立 | `TD3+BC(best alpha) > BC(alpha=0)`，且提升在固定 manifest 上可复现 | 离线 Q-learning 提供了额外价值 |
| 任务可行 | 上述增益出现在 `crosscomp` 数据上，且结果稳定 | 纯离线 RL 对 deployable policy 具有现实潜力 |

不建议把“必须超过 `worldcomp`”作为唯一 gate。

---

## 三、必须锁死的实验协议

### 3.1 推荐的 Phase 0 固定配置

为减少变量，Phase 0 统一使用：

- `flow = wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy`
- `benchmark = benchmarks/single_u10_cross_tgt15.json`
- `probe_layout = s0`
- `history_length = 4`
- `task_geometry = cross_stream`
- `target_speed = 1.5`
- `objective = efficiency_v2`

这里不再采用“脚本默认的 `history_length = 1`”，而是显式对齐 [`scripts/run_stage_a0_layout_screen.sh`](../scripts/run_stage_a0_layout_screen.sh) 的协议：

- A0 screen 已经把 `single_u10_cross_tgt15 + k4` 当作当前 benchmark screen 的标准入口；
- offline RL 若想与当前主实验轨道保持一致，就应该直接复用这套感知协议；
- 因此数据收集、训练、评估都必须显式写 `--history-length 4`，不能依赖默认值。

### 3.2 与 benchmark manifest 的关系

[`benchmarks/single_u10_cross_tgt15.json`](../benchmarks/single_u10_cross_tgt15.json) 已绑定：

- `flow_path = ...Re150...`
- 固定的 episode seeds
- 固定的 start / goal / flow_time

但它的：

- `probe_layout = null`
- `history_length = null`

这意味着 manifest **没有**冻结感知协议，感知协议仍然要由数据集和 checkpoint 自己保证一致。

### 3.3 objective 的语义

训练时使用的是数据集中保存的 `reward`，不是从环境现算。

因此：

- 数据收集用什么 objective，训练就学什么 reward；
- 评估环境也必须使用同一个 objective；
- `objective mismatch` 会让结果失去意义。

---

## 四、数据策略

### 4.1 主数据集：`crosscomp-500` `【部分实现】`

Phase 0 的主数据集应从 `crosscomp` 收集，因为它更接近 deployable teacher。

推荐命令：

```bash
conda run -n mytorch1 python -m scripts.collect_offline_data \
    --policy crosscomp \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --task-geometry cross_stream \
    --target-speed 1.5 \
    --objective efficiency_v2 \
    --episodes 500 \
    --seed 0 \
    --output-dir offline_data/crosscomp_s0_h4_effv2_re150_u10cross
```

### 4.2 二级数据集：`worldcomp-500` `【部分实现】`

`worldcomp` 不是 Phase 0 主结论的数据源，而是用来分析局部 teacher-gap。

推荐命令：

```bash
conda run -n mytorch1 python -m scripts.collect_offline_data \
    --policy worldcomp \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --task-geometry cross_stream \
    --target-speed 1.5 \
    --objective efficiency_v2 \
    --episodes 500 \
    --seed 0 \
    --output-dir offline_data/worldcomp_s0_h4_effv2_re150_u10cross
```

### 4.3 可选后续：support-broadened 数据集 `【未实现】`

当前 [`scripts/collect_offline_data.py`](../scripts/collect_offline_data.py) 默认执行确定性 baseline，不带动作噪声，也不混合多个 teacher。

所以如果出现下面这种情况：

- `alpha=0` 和 `alpha>0` 几乎一样；
- `mean_q` 没明显发散；
- 但 success rate 也没有提升；

优先怀疑数据支持集过窄，而不是立刻升级到 FQL。

Phase 0 之后最小的正确扩展方向是：

- 给 collector 增加 `--action-noise-std`；
- 或新增 `mixed-deployable` 数据收集；
- 或显式加入 recovery transitions。

### 4.4 数据检查 `【部分实现】`

收集完成后检查 `metadata.json`，至少确认：

| 检查项 | 预期 | 解释 |
|---|---|---|
| `flow_path` | 与 Re150 benchmark 一致 | 保证与 manifest 可比 |
| `probe_layout` | `s0` | 感知协议锁定 |
| `history_length` | `4` | 观测维度锁定 |
| `objective` | `efficiency_v2` | reward 语义锁定 |
| `num_transitions` | `> 100k` 为宜 | 太少时过拟合风险高 |
| `success_rate` | 不宜太低 | 成功轨迹稀缺会压低上界 |

当前状态说明：

- `metadata.json` 已由 [`scripts/collect_offline_data.py`](../scripts/collect_offline_data.py) 输出。
- [`scripts/train_offline.py`](../scripts/train_offline.py) 已对 `probe_layout`、`history_length`、`objective`、`reward_config`、`obs_dim`、`action_dim` 做一致性检查。
- 但还没有单独的“dataset validator”脚本把这些检查独立成一条预检命令。

### 4.5 过拟合风险估算 `【未实现】`

```text
采样次数 / transition ≈ total_steps × batch_size / num_transitions
```

建议控制在 **500 次以内**。如果明显超出：

- 降低 `total_steps`；
- 或增加 episode 数；
- 不建议在 Phase 0 里靠更复杂正则化硬扛。

---

## 五、实现边界

### 5.1 建议新增的文件 `【已实现】`

```text
auv_nav/td3bc.py
scripts/train_offline.py
scripts/evaluate_offline.py
scripts/evaluate_baseline_on_manifest.py
```

补充：当前还新增了一个批量启动脚本 [`scripts/run_offline_td3bc_phase0.sh`](../scripts/run_offline_td3bc_phase0.sh)，用于按本文档的默认协议直接发起 Phase 0 实验。

### 5.2 可以直接复用的部分

```text
auv_nav/replay.py
scripts/train_utils.py
auv_nav/sac.py 中的 QNetwork
auv_nav/networks.py 中的 build_hidden_layers
```

这里要特别注意：

- 当前 `QNetwork` 定义在 [`auv_nav/sac.py`](../auv_nav/sac.py)，不在 `networks.py`；
- 当前 [`scripts/evaluate.py`](../scripts/evaluate.py) 是 **SAC 专用**，不能假定 TD3+BC checkpoint 可直接复用。

### 5.3 为什么推荐单独做 `evaluate_offline.py` `【已实现】`

因为当前 `evaluate.py` 假定：

- `agent = SACAgent(...)`
- `trainer_state["agent_config"]` 能直接构造 `SACConfig`

Phase 0 更稳妥的做法是：

- 新增 `scripts/evaluate_offline.py`；
- 只要求 `TD3BCAgent + TD3BCConfig + evaluate_agent()` 跑通；
- 等纯离线算法不止一个后，再统一做 evaluator dispatch。

### 5.4 `trainer_state.json` 的最小字段 `【部分实现】`

为了保证训练和评估协议一致，建议离线 checkpoint 至少保存：

```json
{
  "algo": "td3bc",
  "agent_config": "...",
  "offline_data_path": "...",
  "flow_path": "...",
  "probe_layout": "s0",
  "history_length": 4,
  "reset_options": {
    "task_geometry": "cross_stream",
    "target_auv_max_speed_mps": 1.5
  },
  "env_config_overrides": {
    "reward_objective": "efficiency_v2"
  },
  "obs_normalizer": {
    "enabled": true,
    "mean": "...",
    "std": "..."
  },
  "agent_path": "agent_best.pt"
}
```

当前状态说明：

- 核心字段已经实现，且实际会写入 `algo`、`agent_config`、`flow_path`、`probe_layout`、`history_length`、`reset_options`、`env_config_overrides`、`obs_normalizer`、`agent_path`。
- 真实实现里字段名是 `offline_data_path`，不是 `offline_data`。
- 当前还额外保存了 `protocol`、`best_eval_metrics`、`best_eval_step`、`offline_metadata` 等辅助字段。

### 5.5 关于 holdout `【未实现】`

当前 [`TransitionReplay.sample_batch()`](../auv_nav/replay.py) 直接从全 buffer 均匀采样。只记录 `holdout_indices` 并不能真正把验证集排除出训练。

所以 Phase 0 的建议是：

- **先不做 holdout 结论**；
- 除非同时实现基于 index 的 train / holdout sampler；
- 否则只保留训练集指标和固定 manifest 的在线评估指标。

---

## 六、TD3+BC 的最小实现建议

### 6.1 网络与接口 `【已实现】`

`TD3BCAgent` 只需满足以下约束：

- `act(obs, policy_state, deterministic)`
- `update(batch)`
- `save(path)` / `load(path)`
- `reset_policy_state()`

这样即可与 [`evaluate_agent()`](../scripts/train_utils.py) 兼容。

### 6.2 状态归一化是硬要求 `【已实现】`

TD3+BC 的最小正确实现必须包含 observation normalization。

也就是说：

- 训练前从 offline dataset 计算 `obs_mean / obs_std`
- actor 和 critic 都使用同一组统计量
- 评估时也必须应用同一 normalizer
- checkpoint 中必须保存该 normalizer

若缺少这一步，不应称为“原始 TD3+BC”。

### 6.3 `privileged_obs` 的正确定位 `【已实现（代码支持）】`

建议把它做成**可选二级 ablation**，而不是 Phase 0 的默认主线。

原因：

- Phase 0 的主结论应先建立在 deployable protocol 上；
- `privileged_obs` 与 `worldcomp` gap 分析相关，但不应混入主表；
- 当前 repo 中 `privileged_obs` 只覆盖局部等效流，不等于完整特权流场信息。

因此建议：

- `crosscomp` 主实验：默认不开 privileged critic；
- `worldcomp` 二级实验：可额外做 `privileged-critic` ablation；
- 所有图表和表格必须显式标注 protocol。

当前状态说明：

- [`auv_nav/td3bc.py`](../auv_nav/td3bc.py) 已支持 `privileged_obs_dim > 0` 的 critic。
- actor update 的语义已显式做成 `privileged_actor_update_mode ∈ {zeros, batch}`，不再隐含在实现细节里。
- 但 `worldcomp + privileged-critic` 的正式实验结果还没有在仓库里沉淀。

### 6.4 关键超参 `【已实现】`

保留最少的超参扫描：

| 超参 | 默认值 | 备注 |
|---|---|---|
| `alpha` | `2.5` | 唯一主扫超参 |
| `policy_noise` | `0.2` | TD3 默认设置 |
| `noise_clip` | `0.5` | TD3 默认设置 |
| `policy_freq` | `2` | TD3 默认设置 |
| `gamma` | `0.995` | 与 SAC 对齐 |
| `tau` | `0.005` | 与 SAC 对齐 |

---

## 七、训练与评估协议

### 7.1 Step A：`crosscomp` smoke test `【可执行，未跑正式实验】`

- 数据集：`crosscomp_s0_h4_effv2_re150_u10cross`
- `alpha ∈ {0.0, 2.5}`
- `seed = 42`
- `total_steps = 100000`

目的：

- 确认管线跑通；
- 确认状态归一化闭环正确；
- 先判断 `alpha > 0` 相对 BC 是否出现正信号。

### 7.2 Step B：`crosscomp` full quick validation `【可执行，未跑正式实验】`

- `alpha ∈ {0.0, 1.0, 2.5, 5.0}`
- `seed ∈ {42, 43}`
- `total_steps = 200000`

只有当 Step A 正常时才做 Step B。

### 7.3 Step C：`worldcomp` teacher-gap diagnostic `【可执行，未跑正式实验】`

在 `crosscomp` 得到 best alpha 之后，再去 `worldcomp` 上做诊断：

- 先跑 deployable protocol；
- 如有需要，再加 `privileged-critic` ablation；
- `worldcomp` 结果不作为 Phase 0 主 gate。

### 7.4 推荐训练命令 `【已实现】`

先跑 `crosscomp` smoke test：

```bash
conda run -n mytorch1 python -m scripts.train_offline \
    --offline-data offline_data/crosscomp_s0_h4_effv2_re150_u10cross/transitions.npz \
    --alpha 0.0 \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --task-geometry cross_stream \
    --target-speed 1.5 \
    --objective efficiency_v2 \
    --manifest benchmarks/single_u10_cross_tgt15.json \
    --total-steps 100000 \
    --eval-every 10000 \
    --eval-episodes 30 \
    --save-dir checkpoints/offline/td3bc/crosscomp_u10cross_alpha0_seed42 \
    --seed 42 \
    --device cuda
```

```bash
conda run -n mytorch1 python -m scripts.train_offline \
    --offline-data offline_data/crosscomp_s0_h4_effv2_re150_u10cross/transitions.npz \
    --alpha 2.5 \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --task-geometry cross_stream \
    --target-speed 1.5 \
    --objective efficiency_v2 \
    --manifest benchmarks/single_u10_cross_tgt15.json \
    --total-steps 100000 \
    --eval-every 10000 \
    --eval-episodes 30 \
    --save-dir checkpoints/offline/td3bc/crosscomp_u10cross_alpha2p5_seed42 \
    --seed 42 \
    --device cuda
```

`train_offline.py` 建议支持 `--manifest`，中间评估就直接走固定 benchmark，不要用随机 reset。

当前状态说明：

- [`scripts/train_offline.py`](../scripts/train_offline.py) 已支持 `--manifest`。
- [`scripts/run_offline_td3bc_phase0.sh`](../scripts/run_offline_td3bc_phase0.sh) 已把 Step A / B 所需的 benchmark、dataset、alpha sweep 和 final eval 串成一条可复用工作流。

### 7.5 最终评估 `【已实现】`

最终结果必须在固定 manifest 上重新评估，建议 100 episodes：

```bash
conda run -n mytorch1 python -m scripts.evaluate_offline \
    --checkpoint checkpoints/offline/td3bc/crosscomp_u10cross_alpha2p5_seed42 \
    --manifest benchmarks/single_u10_cross_tgt15.json \
    --episodes 100 \
    --device cuda
```

### 7.6 行为策略对照 `【已实现】`

如果要和 teacher 做严格对比，建议补一个轻量 baseline evaluator，而不是依赖当前 `demo.py`。原因是：

- 现有 `demo.py` 不是基于 benchmark manifest 设计的；
- Phase 0 的所有可报告数字都应来自同一个固定评估集。

推荐最小方案：

- 在 `evaluate_offline.py` 旁边加一个 `evaluate_baseline_on_manifest.py`；
- 或给 `evaluate_offline.py` 增加 `--baseline-policy` 模式。

当前状态：前一种方案已经实现，即 [`scripts/evaluate_baseline_on_manifest.py`](../scripts/evaluate_baseline_on_manifest.py)。

---

## 八、监控与诊断

### 8.1 训练时至少记录这些指标 `【部分实现】`

| 指标 | 优先级 | 用途 |
|---|---|---|
| `eval_success_rate` | P0 | 唯一可信的主指标 |
| `eval_return` | P0 | 辅助判断 reward 质量 |
| `mean_q` | P0 | 检测 Q 高估 |
| `critic_loss` | P1 | 训练稳定性 |
| `actor_loss` | P1 | 策略更新方向 |
| `bc_loss` | P1 | `alpha=0` 与 `alpha>0` 的行为偏离程度 |

当前状态说明：

- 已实现：`eval_success_rate`、`eval_return`、`mean_q`、`critic_loss`、`actor_loss`、`bc_loss`。
- 指标输出位置：
  - 训练日志：`train_log.jsonl`
  - 周期评估：`eval_log.csv`
  - 最终评估：`final_eval.json`
- 当前 `eval_log.csv` 主要记录评估指标，训练损失仍以 `train_log.jsonl` 为主，没有统一汇总到一张表里。

### 8.2 最常见失败模式

**模式 A：`mean_q` 上升，`eval_success_rate` 不升反降`**

- 解释：Q 高估，策略被拉向 OOD 动作。
- 处理：增大 `alpha`，或减少训练步数。

**模式 B：`alpha=0` 和 `alpha>0` 几乎一样`**

- 解释：Q 信号没有提供额外价值，或数据支持集过窄。
- 处理：优先检查**状态归一化是否正确实现**，再检查**数据覆盖率是否过窄**；不要立刻否定 offline RL。

**模式 C：`crosscomp` 上都不优于 BC`**

- 解释：这更像 deployable offline RL 本身尚未建立，而不是 teacher 信息差。
- 处理：先修 normalizer、协议或数据支持集，不建议直接上 FQL。

**模式 D：`worldcomp` 上明显低于 teacher`**

- 解释一：teacher 有信息优势；
- 解释二：`privileged_obs` 语义没有被正确使用或正确标注；
- 解释三：数据覆盖不足。

所以这类失败需要与 `crosscomp` 结果一起解释，不能单独下结论。

---

## 九、Phase 0 的决策规则

### 9.1 推荐的判定表

| 结果 | 判定 | 下一步 |
|---|---|---|
| `TD3+BC(best alpha) > BC(alpha=0)`，且在 `crosscomp` 上稳定复现 | 管线成立 | 推进完整方案 |
| 同时在 `crosscomp` 上提升明显 | 强阳性 | 推进 ReBRAC / XQL |
| `crosscomp` 上只略优于 BC，但方向稳定 | 弱阳性 | 先做更多种子或扩支持集 |
| `crosscomp` 上不优于 BC，且 `mean_q` 发散 | 算法/正则问题 | 收紧 `alpha` 或减少步数 |
| `crosscomp` 上不优于 BC，且 `mean_q` 正常 | 数据问题更可疑 | 扩数据支持集或引入 mixed/noisy dataset |

### 9.2 何时推进完整方案

满足以下任一条件即可推进 `offline_rl_implementation_plan.md`：

- `TD3+BC(best alpha)` 在 `crosscomp` 上稳定优于 BC；
- 或虽然增益有限，但诊断显示主要瓶颈是数据支持集，而不是算法直接崩溃。

如果连 `crosscomp` 上的 BC 都稳定赢不了，就不建议直接上 FQL/XQL，应该先修数据协议、normalizer 或数据来源。

---

## 十、快速参考

### 10.1 最小执行顺序

```bash
# 1. 收集主数据集：crosscomp
conda run -n mytorch1 python -m scripts.collect_offline_data \
    --policy crosscomp \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --task-geometry cross_stream \
    --target-speed 1.5 \
    --objective efficiency_v2 \
    --episodes 500 \
    --seed 0 \
    --output-dir offline_data/crosscomp_s0_h4_effv2_re150_u10cross

# 2. 收集二级数据集：worldcomp
conda run -n mytorch1 python -m scripts.collect_offline_data \
    --policy worldcomp \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --task-geometry cross_stream \
    --target-speed 1.5 \
    --objective efficiency_v2 \
    --episodes 500 \
    --seed 0 \
    --output-dir offline_data/worldcomp_s0_h4_effv2_re150_u10cross

# 3. 先跑 crosscomp smoke test: alpha=0
conda run -n mytorch1 python -m scripts.train_offline \
    --offline-data offline_data/crosscomp_s0_h4_effv2_re150_u10cross/transitions.npz \
    --alpha 0.0 \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --task-geometry cross_stream \
    --target-speed 1.5 \
    --objective efficiency_v2 \
    --manifest benchmarks/single_u10_cross_tgt15.json \
    --total-steps 100000 \
    --eval-every 10000 \
    --eval-episodes 30 \
    --save-dir checkpoints/offline/td3bc/crosscomp_u10cross_alpha0_seed42 \
    --seed 42 \
    --device cuda

# 4. 再跑 crosscomp smoke test: alpha=2.5
conda run -n mytorch1 python -m scripts.train_offline \
    --offline-data offline_data/crosscomp_s0_h4_effv2_re150_u10cross/transitions.npz \
    --alpha 2.5 \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --task-geometry cross_stream \
    --target-speed 1.5 \
    --objective efficiency_v2 \
    --manifest benchmarks/single_u10_cross_tgt15.json \
    --total-steps 100000 \
    --eval-every 10000 \
    --eval-episodes 30 \
    --save-dir checkpoints/offline/td3bc/crosscomp_u10cross_alpha2p5_seed42 \
    --seed 42 \
    --device cuda
```

### 10.2 Phase 0 的一句话结论模板

- 若 `crosscomp` 上 `alpha=2.5 > alpha=0`：说明纯离线 Q-learning 在该任务上提供了真实增益。
- 若该增益在多个种子和固定 manifest 上稳定：说明 deployable offline RL 在当前 AUV 流场导航问题上具有现实可行性。
- 若只在 `worldcomp` 上失败：优先检查信息差和 protocol label，不要直接否定 offline RL。

---

## 十一、参考资料

- **TD3+BC**: Scott Fujimoto, Shixiang Shane Gu. *A Minimalist Approach to Offline Reinforcement Learning*. NeurIPS 2021.
- **完整方案**: [offline_rl_implementation_plan.md](offline_rl_implementation_plan.md)
