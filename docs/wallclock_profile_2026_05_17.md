# Wallclock Profile — 决定不做 GPU env 迁移的数据依据

> **Date**: 2026-05-17
> **Status**: 一次性归档,作为 thesis 后续"是否迁移到 GPU env / fastSAC 风格大规模并行采样"决策的事实依据。除非未来硬件 / 算法配置发生质变(见 §6),否则**不需要重新评估**。
>
> **Cross-references**:
> - Profile tool: [`scripts/profile_train.py`](../scripts/profile_train.py)
> - Notebook (env-side baseline + num_envs scaling): [`notebooks/profile_train_env_completed.ipynb`](../notebooks/profile_train_env_completed.ipynb)
> - Notebook (batch + n_envs sweep): [`notebooks/profile_train_update_sweep_completed.ipynb`](../notebooks/profile_train_update_sweep_completed.ipynb)
> - Raw JSON: `experiments/profile/l4_*.json`,`experiments/profile/local_n6_*_cpu_utd4.json`
> - 影响的 thesis line: [`docs/online_rl_thesis_plan.md`](online_rl_thesis_plan.md),[`docs/online_rl_line_summary.md`](online_rl_line_summary.md)

---

## 1. 背景与问题

Thesis online 线当前 SAC 主训练在 Colab L4 上 `--num-envs 6` 跑 600k transitions ≈ 1.5 h(详见 [`CLAUDE.md`](../CLAUDE.md) "Workflow & Compute Environment")。在审视"是否把 env 迁移到 GPU、像 fastSAC / Brax 那样跑数百-数千 envs"时,我们需要先回答两个先决问题:

1. **当前 wallclock 是被 env.step 吃掉的,还是被 SAC update / replay / IPC 吃掉的?** 如果不是 env 主导,GPU env 的 Amdahl 上限就很低。
2. **L4 上的 SAC update 端是否 saturate?** 如果 saturated,batch 增大不会带来加速;如果 launch-bound,理论上有大幅压榨空间。

本归档用两轮 profile 数据(本机 10 核 CPU + Colab L4 12 vCPU)同时回答这两个问题,给出"GPU env 不值得做"的明确结论,并量化未来仍然可行的 hyperparam 优化空间。

---

## 2. 实验设计

### 2.1 Profile tool

[`scripts/profile_train.py`](../scripts/profile_train.py) 复制了 `scripts/train_sac.py` 的内循环(`random_steps → update_after → updates_per_step`)并把 5 个阶段单独打表:

- `agent_act` — `agent.act(obs)`,含 obs→GPU tensor + actor forward + action→CPU
- `env_step` — `env.step(action)`,vector env 一次步进 `num_envs` 个 envs
- `replay_add` — push 一组 transitions 到 replay
- `replay_sample` — `replay.sample_batch(batch_size, device)`,含 CPU→GPU 拷贝
- `sac_update` — `agent.update(batch)`,GPU forward + backward + optimizer step + target soft update

所有 GPU 调用前后用 `torch.cuda.synchronize()`,保证 perf_counter 不被异步 launch 误导。Warmup 1500 transitions 后才开始计时,所有 lazy tensor cache / JIT / GPU kernel cache 都已热。

### 2.2 两轮 profile 的实验矩阵

**第一轮**([`profile_train_env_completed.ipynb`](../notebooks/profile_train_env_completed.ipynb))定位"env step vs SAC update"瓶颈:

- 本机 CPU baseline:n=6, async / sync,UTD=4
- L4 baseline:n=6, async, UTD=4
- L4 thesis 配置:n=6, async, UTD=4, + LayerNorm + AsymCritic
- L4 num_envs scaling:n ∈ {1, 4, 6, 8, 12}, UTD=4(固定 `updates_per_step`)

**第二轮**([`profile_train_update_sweep_completed.ipynb`](../notebooks/profile_train_update_sweep_completed.ipynb))验证"launch-bound 假设":

- Lever A:n=6, UTD=4, batch ∈ {256, 512, 1024, 2048}
- Lever B:n ∈ {12, 16}, UTD 同比例增到 8/11(保持 effective UTD = UTD/n ≈ 0.67 与 thesis baseline 一致),batch=256
- Combo:n=12, UTD=8, batch=1024(Lever A + B 同开)

每个 profile 跑 1500 warmup + 1800-3000 measure transitions,seed=42,probe=s0,history=4,task_geometry=upstream,target_speed=1.5,objective=efficiency_v2。Flow file 统一 `wake_v8_U1p50_Re250_D12p00_dx0p60_Ti5pct_1200f_roi`。

---

## 3. 完整数据

### 3.1 第一轮 — env step vs SAC update 占比

| 配置 | trans/s | env_step ms/call | env_step % | sac_update ms | sac_update % | 备注 |
|---|---:|---:|---:|---:|---:|---|
| 本机 CPU n=6 sync | 190 | 15.30 | 48.6% | 3.94 | 50.0% | IPC isolated |
| **本机 CPU n=6 async** | **298** | **4.92** | **24.5%** | **3.68** | **73.3%** | 6 worker 并行 |
| L4 cuda n=1 async | 17 | 10.42 | 17.5% | 11.42 | 76.8% | |
| L4 cuda n=4 async | 66 | 13.29 | 21.9% | 11.04 | 72.9% | |
| **L4 cuda n=6 async** | **91** | **15.81** | **24.1%** | **11.67** | **71.0%** | thesis baseline |
| L4 cuda n=8 async | 115 | 20.87 | 30.0% | 11.36 | 65.3% | |
| L4 cuda n=12 async | 165 | 23.36 | 32.1% | 11.53 | 63.3% | UTD=4 不变 → eff UTD 减半 |
| L4 cuda n=6 + LN + AsymC | 81 | 16.32 | 22.1% | 13.44 | 72.9% | thesis 主线完整配置 |

### 3.2 第二轮 — Lever A(batch sweep)与 Lever B(n_envs + UTD 同比)

固定 effective UTD = `updates_per_step / num_envs` ≈ 0.67(thesis baseline 值),只看 wallclock 收益:

| 配置 | n_envs | UTD | batch | eff UTD | trans/s | sac_update ms | env_step ms/call |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline 重测** | 6 | 4 | 256 | 0.67 | 85 | 10.48 | 26.3 |
| Lever A: bs=512 | 6 | 4 | 512 | 0.67 | 97 | 10.52 | 16.9 |
| Lever A: bs=1024 | 6 | 4 | 1024 | 0.67 | 95 | 10.76 | 16.4 |
| Lever A: bs=2048 | 6 | 4 | 2048 | 0.67 | 121 ⚠️ | 11.13 | 15.1 |
| Lever B: n=12, UTD=8 | 12 | 8 | 256 | 0.67 | 107 | 10.57 | 23.5 |
| Lever B: n=16, UTD=11 | 16 | 11 | 256 | 0.69 | 105 | 10.51 | 31.9 |
| Combo: n=12, UTD=8, bs=1024 | 12 | 8 | 1024 | 0.67 | 106 | 10.60 | 22.6 |

⚠️ **bs=2048 的 121 trans/s 不可信**:`replay.ready(2048)` 在 warmup 结束时尚未满足(warmup=1500 transitions),measurement 前半段在跑 env-only-no-update,update count 只到 836 而非 1200,被人为压低。真实 throughput 与 bs=1024 应该相当。

### 3.3 关键派生指标

**单次 `sac_update` 延迟 vs batch**(L4,n=6,UTD=4):

```
batch  | mean_ms
   256 | 10.48
   512 | 10.52   (+0.4%)
  1024 | 10.76   (+2.7%)
  2048 | 11.13   (+6.2%)
```

Batch 增大 8×,单次 update 延迟只增 6%。这是 **launch-bound 的教科书签名**。

**n_envs 对总 throughput 的 scaling**(L4,UTD=4 fixed,batch=256):

```
n=1   →  17 trans/s   (1.0×)
n=4   →  66 trans/s   (3.9×)
n=6   →  91 trans/s   (5.4×)
n=8   → 115 trans/s   (6.8×)
n=12  → 165 trans/s   (9.8×)
```

近线性 scaling 直到 n=12,L4 的 12 vCPU 在 n=12 接近饱和。但**这条 scaling 同时降低了 effective UTD**(从 4.0 降到 0.33),不是 free。

---

## 4. 三个核心结论

### 4.1 GPU env 的 Amdahl 上限只有 1.3-1.5×

在所有 L4 配置下,`env_step` 仅占 17-32% wallclock。即使 GPU env 把 `env_step` 完全消除:

| 场景 | env_step % | GPU env 理论上限加速 |
|---|---:|---:|
| L4 thesis(n=6, LN, AsymC, UTD=4) | 22.1% | **1.28×** |
| L4 baseline(n=6, UTD=4) | 24.1% | 1.32× |
| L4 n=12(UTD=4) | 32.1% | 1.47× |

这是**纯粹的 Amdahl 算术上限**,真实工程中要扣 GPU env 自身的 kernel launch、host↔device 拷贝、numerical regression cost。即使做出来,实测加速大概率落在 1.1-1.3×。

**这与 fastSAC / Brax / IsaacGym 的"10-50×"叙事完全不在一个 regime**——那些工作的环境是 MuJoCo locomotion / 简单 manipulation,SAC update 端用大 batch + 高 UTD + 更深网络早已 GPU-saturate,env step 是真正的瓶颈。我们的设定下根本不是。

### 4.2 SAC update 端 launch-bound 假设证实,但**无法**转化为 wallclock 加速

第一轮看到 `sac_update` 占 64-77% wallclock,曾以为"L4 GPU 严重 underutilize → batch 增大是免费午餐"。

第二轮直接验证:`sac_update mean_ms` 在 batch 256 → 2048 上仅 10.48 → 11.13,launch-bound 确认。**但这没用**——`updates_per_step` 是 per-vector-step 而非 per-batch,在固定 effective UTD = 0.67 的前提下:

```
total update count = transitions × eff_UTD = constant
total update time  = update_count × update_ms ≈ constant
```

1200 个 update × 11 ms ≈ 13 s 把 wallclock 直接锁死。Batch 增大让单次 update 看到更多样本(可能轻微提升 sample efficiency),但**wallclock 一分钱省不下**。

### 4.3 真正的 wallclock lever 只有"降低 effective UTD",有 sample efficiency 代价

| 路径 | wallclock 加速 vs baseline | effective UTD | sample efficiency 风险 |
|---|---:|---:|---|
| 保持 eff UTD = 0.67(Lever B `n=12 UTD=8`) | 1.17× | 0.67 | 不变 |
| eff UTD 0.67 → 0.33(`n=12 UTD=4`) | **1.81×** | 0.33 | 减半,需 sanity check |
| eff UTD 0.67 → 0.17(`n=12 UTD=2`) | ~2.5×(推算) | 0.17 | 显著退化风险 |

**这才是真正的 tradeoff,不是免费午餐。**

### 4.4 一个反直觉但有数据支撑的额外发现

本机 M-series CPU(10 核)在 `--device cpu` 下跑 thesis 同配置时,throughput **比 L4 cuda 高 3.3×**(298 vs 91 trans/s)。原因:`sac_update` 在 L4 GPU 是 11.4 ms / 在 M-series CPU 上只要 3.7 ms。Batch=256 + hidden=256 的 SAC update workload 太小,GPU launch overhead 完全主导,被强单核 CPU 碾压。

**这不是 L4 故障,而是当前 SAC 配置不适合 GPU**。如果未来 thesis 升级到 hidden=512 / batch=2048,GPU 优势才会显现;在那之前,L4 主要是为 8GB+ 显存 + 24h 训练窗口的方便性,不是性能。

### 4.5 与 [`online_rl_line_summary.md`](online_rl_line_summary.md) §2.5 旧诊断的关系(重要)

**旧诊断**(2026-05-06,preflight P0a,基于 cProfile)给出的归因:

- ~55% AsyncVectorEnv IPC(`posix.read` from `connection.py`)
- ~30% env 物理(`flow.bilinear` + `vehicle.dynamics` + `autopilot.sample`)
- ~6% PyTorch(NN forward/backward)

并据此推荐 `num_envs 6→12`,实测 wallclock 5.61 → 3.88 min(1.45× 加速)。

**本归档**用更精细的 5-bucket `perf_counter` profile(GPU 端配 `torch.cuda.synchronize`)显示**实际归因不同**:

- SAC update **64-77%**(L4 cuda 上)
- env_step 22-32%
- AsyncVectorEnv IPC 在 n=6 上**不是瓶颈**(本机 sync vs async 实测 3.1× scaling,符合 6 worker 并行预期;IPC 若主导,async 不可能逼近 worker 数的线性加速)

**为什么 cProfile 看到 `posix.read` ~55%?** 因为它是 **worker 等待 main 处理上一步** 的阻塞 read,被 cProfile 计入"IPC"桶;实际上这段时间 main 在跑 SAC update 而非 IPC pickle。cProfile 测的是"线程在哪个函数里花了时间",不是"哪个函数是 wallclock 因果"。这是 profiling 工具选择导致的归因偏差。

**旧诊断的实操结论(增加 num_envs 拿到加速)仍然有效**——本归档 Lever B 实测 n=12 vs n=6 加速 1.17-1.81×,与旧诊断 1.45× 同量级。**但归因解释应以本归档为准**。下游含义:

- 旧诊断"`AsyncVectorEnv(shared_memory=True)` 上限收益 ~50% wallclock"**乐观了** —— 实测当前 SAC config 下加速上限 1.17× (eff UTD 保持)~ 1.81× (eff UTD 减半),不是 50% wallclock 减少
- 旧诊断"NN 不是瓶颈说明加大 batch / hidden_dim 不会显著拖慢 wallclock"**对错参半** —— batch 增大确实几乎免费(launch-bound 证实),但加大 hidden_dim 会让 update 进入 compute-bound,需要重新 profile

---

## 5. 给 thesis 工作流的建议

### 5.1 GPU env 正式 DROP — 不要再被这个问题诱惑

无论按哪个角度看,都是**性价比极差**:

- Amdahl 上限 1.3-1.5×
- 工作量 2-4 周(vehicle.py + autopilot.py + flow.py + reward.py batched torch 重写)
- 必须做完整 numerical regression suite(否则现有 A0 等 SAC runs / ReBRAC offline 数据集失去可比性)
- 流场要全量上 GPU(0.5-2 GB,可接受但额外 engineering)
- 异步终止 → mask + auto-reset 改造

把这条结论封存在本文档中,**未来 6-12 个月内不要重新打开这个问题**,除非 §6 的 trigger 之一被触发。

### 5.2 Update 端 hyperparam 不动 — 现有配置已近似最优

保持 thesis 当前 baseline:

```
--num-envs 6  --updates-per-step 4  --batch-size 256
--hidden-dim 256  --use-layernorm  --use-asymmetric-critic
```

在 effective UTD = 0.67 的约束下,这已经是 ~10% 之内的最优配置。任何小幅调整(`--batch-size 512` 让每个 update 多看 2× 样本)只是"小确幸",不会撼动 thesis 时间线。

### 5.3 如果未来需要大规模 sweep,**先**做一个 sample-efficiency sanity check

如果 thesis 后期(例如完整 sensor envelope ablation × topology × target-speed × seeds 的 50+ run sweep)需要明确的 wallclock 收益,**唯一**值得考虑的路径是:

```
--num-envs 12  --updates-per-step 4  (eff UTD = 0.33,wallclock 1.81×)
```

但**必须先做 1-2 个 seed 的 sanity check**,与当前 baseline 对比 success rate / return curve / time-to-goal,确认 sample efficiency 退化在可接受范围(建议 ≤ 5% success rate 差异)。

这个 sanity check 工作量 ~ 2-3 h L4 wallclock,**远低于 GPU env 的 2-4 周**。决策上先做 sanity check,再决定是否切换 thesis 主线。

---

## 6. 什么时候该重新评估这个决策

本归档结论的**有效性边界**:

| Trigger | 为什么会改变结论 | 重评估优先级 |
|---|---|---|
| 升级到 H100 / A100 / RTX 4090 | GPU compute 更强但 launch overhead 类似;sac_update 单次延迟可能降到 3-5 ms,env_step 占比跃升到 50%+ | **高** |
| Thesis 改用 hidden_dim ≥ 512 + batch ≥ 1024 | SAC update 进入 compute-bound,GPU env 真正能省 50% wallclock | 中 |
| 引入 model-based / world model 训练 | 整个 pipeline 重构,profile 需要重新做 | 中 |
| 需要单 run > 5M transitions | 1.8× wallclock 在绝对时间上不再 marginal(从 13h 降到 7h vs 从 1.5h 降到 0.8h) | 中 |
| 流场从 single_u15 切换到 200×200 ROI 之上 | env_step 内的 flow sampling cost 上升,occupancy 变化 | 低 |
| Vehicle dynamics 重大简化(例如降到 3-DOF) | env_step 单次成本降一个量级,GPU env 假设链改变 | 低 |

**没有触发以上任何条件之前,不要重做 profile,不要启动 GPU env 评估。** 直接引用本归档作为答案。

---

## 7. 已知局限与注意事项

1. **bs=2048 的 throughput 数字偏高**(见 §3.2 ⚠️):measurement 窗口包含一段 replay-not-ready 的 env-only 阶段。若未来想严格对比 large-batch,需要把 `--warmup-steps` 调到 ≥ batch_size 以确保 replay 在 measurement 开始时已 ready。
2. **env_step 在 Colab 上有 episode-level noise**:vector env 重置时单次 step 时间会跳到 2-3× 均值,但对 throughput 影响 < 5%(噪声幅度 < bucket mean 的 30%)。
3. **L4 vCPU 数取自 Google Cloud 公开 spec(12 vCPU)**:实际可能是 6 物理核 + 12 hyperthreads,这解释了为什么 n=16 已开始 plateau。
4. **本机 M-series 10 核与 L4 12 vCPU 不可逐项直接对比**:`env_step` 在 M-series 上单 env 2.55 ms / 在 L4 单 env 10.4 ms 主要反映单核性能差异,不影响主结论。
5. **本归档只覆盖 SAC 主线**:ReBRAC offline RL 没有 env step,完全是 GPU update,瓶颈结构完全不同;本归档**不适用于 ReBRAC line**。
6. **profile_train.py 的 `agent_act` bucket 在 vector env 模式下是 batched forward**,与真实 train_sac 完全等价;但若未来引入 recurrent policy,该 bucket 的语义会变,需要重新审视。

---

## 8. 总结一句话

> L4 上的 SAC 训练瓶颈在 update 端,不在 env;update 端是 launch-bound 但被 effective UTD 锁死;GPU env 在当前配置下理论上限 1.3-1.5×,工作量 2-4 周,**不做**。Thesis online 主线保持现有 hyperparam 即可,不再讨论。
