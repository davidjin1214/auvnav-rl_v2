# 改进 SAC 的系统实验方案

> 文档版本：2026-04-14  
> 适用范围：当前仓库已有的 `SAC / LayerNorm / Dropout / Asymmetric Critic / RLPD` 实现  
> 目标：用**可归因、可复现、可直接执行**的方式，验证这些改进是否真的提升了 AUV 在复杂流场中的导航性能

---

## 1. 复核后的结论

在重新审查 [auv_nav/sac.py](../auv_nav/sac.py)、[scripts/train_sac.py](../scripts/train_sac.py)、[scripts/run_suite.py](../scripts/run_suite.py)、[docs/rlpd_design.md](rlpd_design.md) 和 [results/rl_navigation_experiment_report.md](../results/rl_navigation_experiment_report.md) 后，我建议采用下面的实验原则。

### 1.1 当前必须坚持的原则

1. **先冻结观测协议，再比较算法组件。**  
   `probe_layout` 和 `history_length` 会显著影响结果，不能和 `LayerNorm / Dropout / Asymmetric Critic / RLPD` 混在一起比较。

2. **主实验目标函数固定为 `efficiency_v2`。**  
   现有结果已经说明 `efficiency_v1` 在关键 upstream benchmark 上过强，不适合作为主结论目标函数。

3. **`arrival_v1` 只作为诊断目标，不作为主线。**  
   如果 `efficiency_v2` 下所有方法成功率都很低，再用 `arrival_v1` 判断问题到底出在“方法无效”还是“目标函数压制到达学习”。

4. **RLPD 必须单独成组，不与结构改动混合归因。**  
   RLPD 的问题是“离线数据是否帮助在线学习”，不是“离线数据 + 新网络结构是否帮助在线学习”。

5. **效率指标必须在成功率达标后再比较。**  
   失败更快、越界更早，有时会让能耗更低，但这不代表策略更优。

### 1.2 当前仓库里不建议直接拿来做主结论的 preset

- `sac_ablation_v1` 不适合直接作为论文主 ablation。  
  原因是其中不同 method 同时改变了 `history_length`，不能干净归因。

- `objective_ablation_v1` 和 `efficiency_gain_sweep_v1` 的结论仍然有效。  
  它们已经足以支持“主线目标函数应切换到 `efficiency_v2`”这个决定。

---

## 2. 全局实验协议

本方案采用四个阶段：

1. **阶段 A：观测协议预筛选**
2. **阶段 B：在线算法组件消融**
3. **阶段 C：RLPD 数据源消融**
4. **阶段 D：确认性实验与拓扑泛化**

### 2.1 固定超参数

除非某一阶段明确说明，否则统一使用：

| 项目 | 设置 |
|------|------|
| 主目标函数 | `efficiency_v2` |
| 训练步数 | `200000` |
| `random_steps` | `5000` |
| `update_after` | `5000` |
| `update_every` | `1` |
| `batch_size` | `256` |
| `hidden_dim` | `256` |
| `checkpoint_every` | `10000` |
| `eval_every` | `10000` |
| `eval_episodes` | `30` |
| `num_envs` | `4` |
| pilot seeds | `42, 43, 44` |
| confirmatory seeds | `42, 43, 44, 45, 46, 47, 48, 49` |

### 2.2 指标优先级

主指标按以下顺序解释：

1. **最后 3 次 periodic eval 的平均成功率**
2. **success-rate 曲线的整体趋势**
3. **best periodic eval success**
4. **success-conditioned** `time / energy / safety / path_efficiency`
5. 终止原因分布：`goal / out_of_bounds / timeout / depth_hold_failure`
6. 训练稳定性诊断：`alpha`、`mean_q`、seed 间方差

### 2.3 淘汰规则

- 若一个方法在主 benchmark 上 `best periodic eval success < 0.10`，则视为**非可行方法**，不进入效率指标排名。
- 若方法成功率相近，优先选择：
  1. seed 方差更小者
  2. success-conditioned `path_efficiency` 更高者
  3. success-conditioned `energy` 更低者

---

## 3. 阶段 A：观测协议筛选

### 3.1 目标

阶段 A 只回答一个问题：

> 后续所有算法实验，应统一使用哪一套 `(probe_layout, history_length)`？

因此，阶段 A 的选择标准不是“绝对最强”，而是：

1. 在 upstream 场景中**可学**
2. 对 seed **稳定**
3. 在 hard upstream 上**不崩**
4. 若性能接近，则优先选择**更简单的协议**

### 3.2 设计原则

本阶段采用 `A0 -> A1 -> A2 -> A3` 四步：

1. **A0：可行性筛选**
   先在更简单的 `cross_stream` benchmark 上快速筛掉明显弱的 layout
2. **A1：layout 主筛选**
   再在 `upstream` benchmark 上筛选 `probe_layout`
3. **A2：history length 筛选**
   在前 2 名 layout 上比较 `history_length`
4. **A3：hard-check**
   在关键 hard benchmark 上做最终风险排查

所有 A 阶段实验统一使用：

- 算法：基础 `SAC`
- 目标函数：`efficiency_v2`
- 不启用：`LayerNorm / Dropout / Asymmetric Critic / RLPD`

### 3.3 benchmark 前提

本方案新增一个更简单的 benchmark：

- `single_u10_cross_tgt15`

其建议定义为：

| 字段 | 值 |
|------|------|
| `flow_path` | `wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` |
| `task_geometry` | `cross_stream` |
| `target_speed` | `1.5` |
| `action_mode` | `auto` |

注意：该 benchmark **当前不在** [scripts/benchmark_catalog.py](../scripts/benchmark_catalog.py) 中。  
在执行 A0 前，需要先新增对应的 `BenchmarkSpec`，并生成其 manifest。

### 3.4 实验矩阵

| 子阶段 | 目的 | benchmark | 比较对象 | 固定设置 | budget | seeds | 预计 run 数 |
|------|------|------|------|------|------:|------|------:|
| `A0` | layout 可行性筛选 | `single_u10_cross_tgt15` | `s0_k4`, `s1_k4`, `s2_k4` | `objective=efficiency_v2` | `100k` | `42,43` | `6` |
| `A1` | layout 主筛选 | `single_u10_upstream_tgt15` | A0 晋级 layout，默认 `s0_k4`, `s1_k4`, `s2_k4` | `history_length=4` | `100k` | `42,43,44` | `6-9` |
| `A2` | 筛 `history_length` | `single_u10_upstream_tgt15` | A1 前 2 名 layout 各比 `k=1,4,16` | `probe_layout` 来自 A1 | `100k-120k` | `42,43,44` | `18` |
| `A3` | hard-check | `single_u15_upstream_tgt15` | A2 前 2 名配置 | 最佳 `(probe, k)` 候选 | `100k` | `42,43` | `4` |

若 A0 没有明显淘汰，则阶段 A 总 run 数约为：

- `6 + 9 + 18 + 4 = 37`

若 A0 只保留 2 个 layout，则阶段 A 总 run 数约为：

- `6 + 6 + 18 + 4 = 34`

### 3.5 A0：可行性筛选

#### 设计理由

- `cross_stream` 明显比 `upstream` 更容易学，适合做前置筛选。
- A0 的目的不是最终选型，而是尽早淘汰明显弱的 `probe_layout`。
- 固定 `history_length=4`，只比较空间感知，不混入时间上下文因素。
- 将 budget 提高到 `100k`，可以减少“早期随机波动”导致的误判。

#### 比较对象

| 配置名 | `probe_layout` | `history_length` |
|------|------|------:|
| `s0_k4` | `s0` | `4` |
| `s1_k4` | `s1` | `4` |
| `s2_k4` | `s2` | `4` |

#### 判据

A0 只做淘汰，不做最终选型。

主判据：

- `best periodic eval success`

辅助判据：

- `out_of_bounds` 比例
- 成功 episode 数

#### 晋级规则

- 若某 layout 在两个 seed 上都几乎无成功，且大量 `out_of_bounds`，则直接淘汰。
- 默认保留前 2 名进入 A1。
- 若三者差距很小，也可三者全部进入 A1。

#### 执行命令

先新增 benchmark 并生成 manifest，然后运行：

```bash
for PROBE in s0 s1 s2; do
  for SEED in 42 43; do
    conda run -n mytorch1 python -m scripts.train_sac \
      --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
      --task-geometry cross_stream \
      --target-speed 1.5 \
      --objective efficiency_v2 \
      --probe-layout "$PROBE" \
      --history-length 4 \
      --total-steps 100000 \
      --random-steps 5000 \
      --update-after 5000 \
      --batch-size 256 \
      --hidden-dim 256 \
      --num-envs 4 \
      --eval-every 10000 \
      --eval-episodes 30 \
      --checkpoint-every 10000 \
      --eval-manifest benchmarks/single_u10_cross_tgt15.json \
      --seed "$SEED" \
      --save-dir "experiments/protocol_screen_v2/A0_single_u10_cross_tgt15/efficiency_v2/${PROBE}_k4/seed_${SEED}"
  done
done
```

### 3.6 A1：layout 主筛选

#### 设计理由

- A1 才是阶段 A 的主筛选步骤。
- 使用 `single_u10_upstream_tgt15`，因为它比 hard upstream 更容易拉开协议差异，同时仍然保留 upstream 的核心挑战。
- 继续固定 `history_length=4`，保证只比较 `probe_layout`。
- 将 budget 设为 `100k`，比以往更稳，适合做协议选择。

#### 比较对象

- A0 晋级的 layout
- 若 A0 不淘汰，则比较 `s0_k4`, `s1_k4`, `s2_k4`

#### 主判据

- 最后 3 次 periodic eval success 的平均值

#### 辅助判据

- `best periodic eval success`
- `out_of_bounds` 比例
- seed 方差
- success-conditioned `path_efficiency`

#### 选择逻辑

1. 先按主判据排序
2. 若差距很小，优先选更简单的 layout
3. 若 `s2` 仅略优于 `s1`，优先考虑 `s1`

#### 执行命令

```bash
for PROBE in s0 s1 s2; do
  for SEED in 42 43 44; do
    conda run -n mytorch1 python -m scripts.train_sac \
      --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
      --task-geometry upstream \
      --target-speed 1.5 \
      --objective efficiency_v2 \
      --probe-layout "$PROBE" \
      --history-length 4 \
      --total-steps 100000 \
      --random-steps 5000 \
      --update-after 5000 \
      --batch-size 256 \
      --hidden-dim 256 \
      --num-envs 4 \
      --eval-every 10000 \
      --eval-episodes 30 \
      --checkpoint-every 10000 \
      --eval-manifest benchmarks/single_u10_upstream_tgt15.json \
      --seed "$SEED" \
      --save-dir "experiments/protocol_screen_v2/A1_single_u10_upstream_tgt15/efficiency_v2/${PROBE}_k4/seed_${SEED}"
  done
done
```

### 3.7 A2：history length 筛选

#### 设计理由

- `probe_layout` 与 `history_length` 之间可能有交互，因此不应只在 A1 冠军 layout 上筛 `k`。
- 建议保留 A1 前 2 名 layout，并分别比较 `k=1,4,16`。
- 只有当 `k=16` 提供稳定、明确的收益时，才选择长历史；否则优先选择 `k=4`。

#### 比较对象

假设 A1 前 2 名是 `s1` 和 `s2`，则 A2 比较：

- `s1_k1`, `s1_k4`, `s1_k16`
- `s2_k1`, `s2_k4`, `s2_k16`

#### 主判据

- 最后 3 次 periodic eval success 的平均值

#### 辅助判据

- seed 方差
- `best periodic eval success`
- success-conditioned `path_efficiency`

#### 选择逻辑

1. 先看主判据
2. 若主判据接近，优先选更短的 `history_length`
3. 若 `k=16` 仅带来边际收益，不选 `k=16`

#### 执行命令模板

将 `TOP1_PROBE` 和 `TOP2_PROBE` 替换为 A1 的前 2 名：

```bash
for PROBE in TOP1_PROBE TOP2_PROBE; do
  for K in 1 4 16; do
    for SEED in 42 43 44; do
      conda run -n mytorch1 python -m scripts.train_sac \
        --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
        --task-geometry upstream \
        --target-speed 1.5 \
        --objective efficiency_v2 \
        --probe-layout "$PROBE" \
        --history-length "$K" \
        --total-steps 100000 \
        --random-steps 5000 \
        --update-after 5000 \
        --batch-size 256 \
        --hidden-dim 256 \
        --num-envs 4 \
        --eval-every 10000 \
        --eval-episodes 30 \
        --checkpoint-every 10000 \
        --eval-manifest benchmarks/single_u10_upstream_tgt15.json \
        --seed "$SEED" \
        --save-dir "experiments/protocol_screen_v2/A2_single_u10_upstream_tgt15/efficiency_v2/${PROBE}_k${K}/seed_${SEED}"
    done
  done
done
```

### 3.8 A3：hard-check

#### 设计理由

- A3 不重新做完整排名，只做最终风险排查。
- 目的在于防止某个协议只在较易 upstream 有效，但在关键 `single_u15_upstream_tgt15` 上崩掉。

#### 比较对象

- A2 前 2 名配置

#### 预算

- `100k`
- `2` 个 seed：`42, 43`

#### 判据

- `best periodic eval success`
- 最后 3 次 eval 平均成功率
- `out_of_bounds` 比例

若 A2 第一名在 A3 中明显崩掉，而第二名更稳，则优先选第二名作为最终协议。

#### 执行命令模板

将 `CFG1` 和 `CFG2` 替换成 A2 的前 2 名，例如 `s1_k4`, `s2_k4`：

```bash
for CFG in CFG1 CFG2; do
  case "$CFG" in
    s1_k4) PROBE=s1; K=4 ;;
    s2_k4) PROBE=s2; K=4 ;;
    s1_k16) PROBE=s1; K=16 ;;
    s2_k16) PROBE=s2; K=16 ;;
  esac

  for SEED in 42 43; do
    conda run -n mytorch1 python -m scripts.train_sac \
      --flow wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
      --task-geometry upstream \
      --target-speed 1.5 \
      --objective efficiency_v2 \
      --probe-layout "$PROBE" \
      --history-length "$K" \
      --total-steps 100000 \
      --random-steps 5000 \
      --update-after 5000 \
      --batch-size 256 \
      --hidden-dim 256 \
      --num-envs 4 \
      --eval-every 10000 \
      --eval-episodes 30 \
      --checkpoint-every 10000 \
      --eval-manifest benchmarks/single_u15_upstream_tgt15.json \
      --seed "$SEED" \
      --save-dir "experiments/protocol_screen_v2/A3_single_u15_upstream_tgt15/efficiency_v2/${CFG}/seed_${SEED}"
  done
done
```

### 3.9 阶段 A 的最终输出

阶段 A 结束后，应明确输出：

- `SELECTED_PROBE`
- `SELECTED_K`

以及一段简短的决策理由，说明：

1. 该协议在 `u10_upstream` 上表现靠前
2. 在 `u15_upstream` hard-check 中未崩
3. 若有多个协议接近，为什么选择了更简单的那个

### 3.10 阶段 A 的建议汇总命令

每个子阶段结束后，建议分别汇总：

```bash
conda run -n mytorch1 python -m scripts.summarize_suite \
  --suite-root experiments/protocol_screen_v2/A0_single_u10_cross_tgt15

conda run -n mytorch1 python -m scripts.summarize_suite \
  --suite-root experiments/protocol_screen_v2/A1_single_u10_upstream_tgt15

conda run -n mytorch1 python -m scripts.summarize_suite \
  --suite-root experiments/protocol_screen_v2/A2_single_u10_upstream_tgt15

conda run -n mytorch1 python -m scripts.summarize_suite \
  --suite-root experiments/protocol_screen_v2/A3_single_u15_upstream_tgt15
```

如需画图：

```bash
MPLCONFIGDIR=/tmp/mpl conda run -n mytorch1 python -m scripts.plot_suite \
  --suite-root <A_STAGE_ROOT>
```

### 3.11 阶段 A 的预期失败模式

1. `s0` 在 A0 或 A1 中大量 `out_of_bounds`
   说明单探针不足以支撑复杂流场中的横向修正和逆流决策。

2. `k=16` 在 A2 中出现“偶尔 peak 很高，但 seed 方差大”
   说明长历史增加了输入维度和训练不稳定性。

3. `s2` 在 A1/A2 中均值更高，但 A3 崩掉
   说明高维感知在较易任务上有利，但在 hard upstream 上不够稳。

4. 所有配置都接近零成功
   说明当前瓶颈不在观测协议，而在训练稳定性或目标函数压制，需要暂停后续阶段，先做诊断实验。

---

## 4. 阶段 B：在线算法组件消融

### 4.1 目的

在固定观测协议后，干净验证下列改进是否有效：

- `LayerNorm`
- 高 UTD
- `Dropout`
- `Asymmetric Critic`
- 全部组合

### 4.2 设计

训练 benchmark：

- `single_u15_upstream_tgt15`

评估 manifest：

- [benchmarks/single_u15_upstream_tgt15.json](../benchmarks/single_u15_upstream_tgt15.json)

比较方法：

| 方法名 | 额外 flag | 目的 |
|------|------|------|
| `sac_base` | 无 | 原始基线 |
| `sac_ln` | `--use-layernorm` | 单独验证 LN |
| `sac_utd4` | `--use-layernorm --updates-per-step 4` | 区分高 UTD 与 Dropout |
| `sac_droq` | `--use-layernorm --updates-per-step 4 --dropout-rate 0.01` | 验证 DroQ 风格改进 |
| `sac_asym` | `--use-layernorm --use-asymmetric-critic` | 验证 asymmetric critic |
| `sac_full` | `--use-layernorm --updates-per-step 4 --dropout-rate 0.01 --use-asymmetric-critic` | 验证联合方案 |

注意：本项目的 `Asymmetric Critic` 使用的是环境提供的 `privileged_obs`，本质上是**局部特权流信息 critic**，不是 full-state oracle critic。论文写作时必须按此表述。

### 4.3 执行命令

将下面的 `SELECTED_PROBE` 和 `SELECTED_K` 替换成阶段 A 的胜出配置：

```bash
for METHOD in sac_base sac_ln sac_utd4 sac_droq sac_asym sac_full; do
  case "$METHOD" in
    sac_base) EXTRA_ARGS=() ;;
    sac_ln) EXTRA_ARGS=(--use-layernorm) ;;
    sac_utd4) EXTRA_ARGS=(--use-layernorm --updates-per-step 4) ;;
    sac_droq) EXTRA_ARGS=(--use-layernorm --updates-per-step 4 --dropout-rate 0.01) ;;
    sac_asym) EXTRA_ARGS=(--use-layernorm --use-asymmetric-critic) ;;
    sac_full) EXTRA_ARGS=(--use-layernorm --updates-per-step 4 --dropout-rate 0.01 --use-asymmetric-critic) ;;
  esac

  for SEED in 42 43 44; do
    conda run -n mytorch1 python -m scripts.train_sac \
      --flow wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
      --task-geometry upstream \
      --target-speed 1.5 \
      --objective efficiency_v2 \
      --probe-layout SELECTED_PROBE \
      --history-length SELECTED_K \
      --total-steps 200000 \
      --random-steps 5000 \
      --update-after 5000 \
      --batch-size 256 \
      --hidden-dim 256 \
      --num-envs 4 \
      --eval-every 10000 \
      --eval-episodes 30 \
      --checkpoint-every 10000 \
      --eval-manifest benchmarks/single_u15_upstream_tgt15.json \
      --seed "$SEED" \
      --save-dir "experiments/improved_sac_components_v1/single_u15_upstream_tgt15/efficiency_v2/${METHOD}/seed_${SEED}" \
      "${EXTRA_ARGS[@]}"
  done
done
```

汇总：

```bash
conda run -n mytorch1 python -m scripts.summarize_suite \
  --suite-root experiments/improved_sac_components_v1
```

### 4.4 决策规则

- 选出 **前 2 个在线方法** 进入阶段 D。
- 若所有方法都低于可行阈值（`best periodic eval success < 0.10`），则触发补充诊断：
  - 在同一 benchmark 上，仅对 `sac_base`、`sac_ln`、`sac_droq` 追加 `arrival_v1` 实验
  - 目的不是改主结论，而是判断 `efficiency_v2` 是否仍然压制了到达学习

---

## 5. 阶段 C：RLPD 数据源消融

### 5.1 目的

固定一个在线 backbone，单独回答：

> 离线数据是否帮助在线训练？  
> 哪种离线数据最有帮助？  
> 特权数据是否存在明显分布偏移风险？

### 5.2 设计

1. 先从阶段 B 中选出**最强在线 backbone**，记为 `ONLINE_BACKBONE`。
2. 所有 RLPD 实验都使用**同一个 backbone**，只改变离线数据源。
3. 比较以下数据源：
   - `goalseek`
   - `crosscomp`
   - `worldcomp`
   - `privileged`

训练 benchmark 与评估 manifest：

- `single_u15_upstream_tgt15`

主设置：

- `offline_ratio = 0.5`
- `objective = efficiency_v2`
- `probe_layout = SELECTED_PROBE`
- `history_length = SELECTED_K`

### 5.3 先收集离线数据

```bash
for POLICY in goalseek crosscomp worldcomp privileged; do
  conda run -n mytorch1 python -m scripts.collect_offline_data \
    --policy "$POLICY" \
    --flow wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --task-geometry upstream \
    --target-speed 1.5 \
    --objective efficiency_v2 \
    --probe-layout SELECTED_PROBE \
    --history-length SELECTED_K \
    --episodes 600 \
    --seed 0 \
    --output-dir "offline_data/rlpd_main_${POLICY}_SELECTED_PROBE_kSELECTED_K_effv2"
done
```

### 5.4 数据公平性检查

读取各目录下的 `metadata.json`，核对：

- `objective`
- `probe_layout`
- `history_length`
- `privileged_obs_dim`
- `num_transitions`

若不同 policy 的 `num_transitions` 差异超过 `10%`，则不要直接写出“数据源质量优劣”的强结论。此时应调整 `episodes` 后重收集，直到 transition 数量大致匹配。

### 5.5 RLPD 训练命令

先定义 backbone flag。根据阶段 B 的胜者填写：

| `ONLINE_BACKBONE` | 对应 flag |
|------|------|
| `sac_ln` | `--use-layernorm` |
| `sac_utd4` | `--use-layernorm --updates-per-step 4` |
| `sac_droq` | `--use-layernorm --updates-per-step 4 --dropout-rate 0.01` |
| `sac_asym` | `--use-layernorm --use-asymmetric-critic` |
| `sac_full` | `--use-layernorm --updates-per-step 4 --dropout-rate 0.01 --use-asymmetric-critic` |

然后运行：

```bash
for POLICY in goalseek crosscomp worldcomp privileged; do
  for SEED in 42 43 44; do
    conda run -n mytorch1 python -m scripts.train_sac \
      --flow wake_data/wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
      --task-geometry upstream \
      --target-speed 1.5 \
      --objective efficiency_v2 \
      --probe-layout SELECTED_PROBE \
      --history-length SELECTED_K \
      --total-steps 200000 \
      --random-steps 5000 \
      --update-after 5000 \
      --batch-size 256 \
      --hidden-dim 256 \
      --num-envs 4 \
      --eval-every 10000 \
      --eval-episodes 30 \
      --checkpoint-every 10000 \
      --offline-data "offline_data/rlpd_main_${POLICY}_SELECTED_PROBE_kSELECTED_K_effv2/transitions.npz" \
      --offline-ratio 0.5 \
      --eval-manifest benchmarks/single_u15_upstream_tgt15.json \
      --seed "$SEED" \
      --save-dir "experiments/rlpd_source_ablation_v1/single_u15_upstream_tgt15/efficiency_v2/rlpd_${POLICY}/seed_${SEED}" \
      <BACKBONE_FLAGS>
  done
done
```

将 `<BACKBONE_FLAGS>` 替换为上表中的具体 flag。

同时保留一个对照组：

- 阶段 B 中同一 backbone 的纯在线结果

### 5.6 决策规则

- 选出 **前 2 个 RLPD 方案** 进入阶段 D。
- 如果 `privileged` 只提升 early learning，但最终成功率不稳定，结论应写为：
  - “privileged data can accelerate early learning”
  - 不应写为 “privileged data yields a better deployable policy”

---

## 6. 阶段 D：确认性实验与拓扑泛化

### 6.1 目的

对最终候选方法做正式确认，并检查是否能泛化到多圆柱尾迹拓扑。

### 6.2 参赛方法

进入阶段 D 的方法上限为 4 个：

- 阶段 B 的前 2 个在线方法
- 阶段 C 的前 2 个 RLPD 方法

### 6.3 正式训练

主 benchmark：

- `single_u15_upstream_tgt15`

正式 seeds：

- `42 43 44 45 46 47 48 49`

训练参数与阶段 B/C 相同，唯一变化是 seed 数量增加。  
目录建议：

- `experiments/improved_sac_confirmation_v1/...`

### 6.4 transfer evaluation

训练完成后，不直接只看 final checkpoint。  
应先在每个 run 目录中找到**在训练 benchmark 上表现最好的 periodic checkpoint**：

```bash
conda run -n mytorch1 python -m scripts.evaluate_best_checkpoint \
  --run-dir <RUN_DIR> \
  --output-json <RUN_DIR>/best_checkpoint.json
```

然后用该 checkpoint 去评估两个 transfer benchmark：

- [benchmarks/sbs_u15_upstream_tgt15.json](../benchmarks/sbs_u15_upstream_tgt15.json)
- [benchmarks/tandem_u15_upstream_tgt15.json](../benchmarks/tandem_u15_upstream_tgt15.json)

命令模板：

```bash
conda run -n mytorch1 python -m scripts.evaluate \
  --checkpoint <RUN_DIR> \
  --agent-file agent_step_<BEST_STEP>.pt \
  --manifest benchmarks/sbs_u15_upstream_tgt15.json \
  --output-json <RUN_DIR>/transfer_sbs.json

conda run -n mytorch1 python -m scripts.evaluate \
  --checkpoint <RUN_DIR> \
  --agent-file agent_step_<BEST_STEP>.pt \
  --manifest benchmarks/tandem_u15_upstream_tgt15.json \
  --output-json <RUN_DIR>/transfer_tandem.json
```

### 6.5 最终结论的成立条件

一个方法只有同时满足以下条件，才可以写成“推荐方法”：

1. 在 `single_u15_upstream_tgt15` 上显著优于 `sac_base`
2. seed 间方差可接受，没有明显不稳定训练
3. 在 `sbs` 和 `tandem` 上没有出现灾难性退化
4. success-conditioned `path_efficiency` 和 `energy` 至少有一项优于基线，且不以成功率大幅下降为代价

---

## 7. 建议的最终产出物

每个阶段结束后，至少保存以下内容：

1. `ablation_summary.csv`
2. `ablation_runs.csv`
3. `ablation_report.md`
4. 每个 run 的 `final_eval.json`
5. 每个 run 的 `eval_log.csv`
6. RLPD 数据目录下的 `metadata.json`

论文或报告中建议至少包含 4 张图/表：

1. 观测协议预筛选结果表
2. 在线组件消融主表
3. RLPD 数据源消融主表
4. 主 benchmark 与 transfer benchmark 的最终汇总表

---

## 8. 一句话执行建议

如果只按最小可行路径推进：

1. 先做阶段 A，冻结 `SELECTED_PROBE` 和 `SELECTED_K`
2. 再做阶段 B，找出最强在线方法
3. 再做阶段 C，验证 RLPD 是否真的有增益
4. 最后做阶段 D，用 8 seeds + transfer benchmark 给出正式结论

这条路线最重要的优点是：**每一步的结论都能归因，不会把“传感器变化、时间上下文变化、网络改动、离线数据改动”混在一起。**
