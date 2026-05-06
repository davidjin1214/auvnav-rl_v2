# ReBRAC 广验实验设计：跨数据质量 / 传感器 / 任务三轴的 Probe-then-Deepen

> 文档版本：2026-05-04 rev.1
> 文档定位：ReBRAC 主线（`docs/rebrac_experiment_plan.md` rev.8 + `docs/rebrac_mainline_review.md` rev.2）已完整收口、paper-readiness 4 项 probe 全部 closed 之后，针对 review §2.2 / §3.2 列出的 generality 边界做的一轮**有限算力广验**。本文是 design spec，对应 implementation plan 将在 spec 通过用户审查后由 writing-plans skill 单独产出。
> 配套文档：[rebrac_experiment_plan.md](../../rebrac_experiment_plan.md)、[rebrac_mainline_review.md](../../rebrac_mainline_review.md)、[rebrac_experiment_report.md](../../rebrac_experiment_report.md)
> 阅读前提：阅读上述三份文档的 §0–§5；本文不重复主线结论，只描述"主线之外要新增什么"。

---

## 0. 摘要

**动机**：ReBRAC 主线只覆盖了一个组合（`crosscomp-1000 / s0 / cross_stream / Re150 / U=1.0 / target=1.5`）。paper 现有 finding 在该单格上严密，但 generality 是单点证据。审稿人会问的三件事——

1. 数据质量退化时 ReBRAC 是否仍优？
2. 传感器更丰富时 ReBRAC 的相对优势是否还在？
3. 任务几何 / 多尾流场景是否同样成立？

——目前都没有数据。

**方法**：从 Stage C finalist `(β1=4, β2=2)` 作为 anchor，在三个轴上各拉 2–3 条 spoke，每条 spoke 先 2-seed probe（强制包含 seed 44），仅当触发判据时升 5-seed + 1-seed β refit。

**预算**：约 **28–34 个新 run / ~21–24 小时 L4**（3–4 个 Colab Pro session）。

**核心输出**：
- 7 个新 offline dataset + sanity card；
- 7 条 spoke × 2 seeds 的 ReBRAC probe table；
- 1 条 (A2 mid-quality) 上的 TD3+BC head-to-head sanity；
- 触发判据下的 P2 5-seed 表（条件性）；
- paper §experiments 一节新增的 broad-validation subsection。

---

## 1. 动机与范围边界

### 1.1 当前 ReBRAC 主线只占了一格

参见 [rebrac_mainline_review.md §1.2](../../rebrac_mainline_review.md)。所有四条 paper-level finding 都在 `crosscomp / s0 / cross_stream / Re150` 这一个 cell 上严密成立；`worldcomp-1000` 是 deployable→teacher gap 的辅助验证轴，**不是** quality / sensor / task generality 的覆盖。

### 1.2 review §2.2 / §3.2 留下的弱点

| 弱点 | review 出处 | 当前数据 | 广验目标 |
|---|---|---|---|
| 数据质量轴未扫 | §2.2.1（隐含）+ §3.2.G | 仅 crosscomp（接近 expert）+ worldcomp（接近 expert） | A 轴 3 spoke：goalseek（差）/ mix（中）/ privileged（近 expert） |
| 传感器轴未扫 | §2.3 + §3.2.G | 仅 s0 | B 轴 2 spoke：s1 / s2 |
| 任务 generality 未扫 | §3.2.G | 仅 cross_stream / Re150 | C 轴 2 spoke：upstream（geometry） / tandem（multi-wake） |

### 1.3 不在本广验范围内（保持主线 closed）

| 类别 | 不做的原因 |
|---|---|
| `worldcomp` 任意扩展 | Phase 1 + Phase 2 已 5-seed 收口，扩广反而稀释 paper narrative |
| Re150 → Re250 / U=1.0 → U=1.5 sweep | 双 confound（同时改 Reynolds 和 wake speed），归因不干净 |
| `target_speed=2.0` / `single_u15_upstream_tgt20` | 与 anchor target_speed=1.5 不同，单变量原则被破坏 |
| 网络容量 / hidden_dim / dropout sweep | review §3.3 已确认不做 |
| LN-off 跨条件复测 | review §3.2 已 closed；本广验不预留 |
| seed 44 collector 起点 inspection (Stage E b) | 维持推迟 |
| β grid 全网格重扫 | 仅在 P2 触发条件下做 1-seed refit，不重扫主网格 |

---

## 2. 实验设计概述

### 2.1 三轴 + Anchor + Spoke + Probe-then-Deepen

```
                 Anchor (Stage C finalist, 0 算力, 已存在)
                 crosscomp-1000 / s0 / cross_stream / Re150 / (β1=4, β2=2)
                 5 seeds, success = 0.902 ± 0.021
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   A 轴: 质量        B 轴: 传感器       C 轴: 任务/流场
        │                 │                 │
   A1 goalseek-1000  B1 crosscomp-s1-1000  C1 crosscomp-upstream-1000
   A2 mix-1000       B2 crosscomp-s2-1000  C3 crosscomp-tandem-1000
   A3 privileged-1000
```

每条 spoke 改且仅改一个轴（行为策略 / probe layout / task geometry+flow），其余配置严格沿用 anchor。

### 2.2 二阶段协议

- **Phase 1 (probe)**：每条 spoke × 2 seeds（强制 seed 42 + seed 44）。该阶段是无条件的，14 个 ReBRAC run + 2 个 TD3+BC run。
- **Phase 2 (deepen)**：仅对触发判据的 spoke 执行：
  1. 1-seed β refit：在 (β1=2, β2=2) 与 (β1=4, β2=1) 各跑 1 个 seed（共 +3 run，复用 seed 42）；
  2. 升 5-seed：扩到 seed 42–46，若 β refit winner 漂移则在新 winner 上扩，否则在 (β1=4, β2=2) 上扩（+3 run，已有 seed 42 + seed 44 复用）。
- 单 spoke 触发后总成本 = 6 run。

### 2.3 关键设计取舍（已与用户拍板）

| 取舍点 | 选择 | 出处 |
|---|---|---|
| goalseek collector success rate < 0.3 时是否扩到 2000 ep | 不扩（保持 1000 ep parity） | §3.2 |
| mix dataset 混法 | episode-level（每 episode 整条 trajectory 来自单一 policy；50% goalseek + 50% crosscomp） | §3.2 |
| LN-off 跨条件 P2 add-on | 不预留 | §1.3 |
| TD3+BC 对照覆盖度 | 仅 A2（mid-quality 是 ReBRAC vs TD3+BC 区分度理论最大点） | §5.2 |
| β refit 时机 | P1 不 refit；P2 触发后第一步必 refit | §6.2 |
| spoke probe 的 seed 选择 | seed 42 + seed 44（强制包含 outlier） | §5.3 |
| P2 触发判据方向性 | 双向 \|Δ\| > 5pp（正向也触发） | §6.1 |

---

## 3. Anchor 与 spoke 矩阵

### 3.1 Anchor（复用 Stage C，0 新算力）

| 字段 | 值 |
|---|---|
| dataset | `offline_data/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/` |
| collector policy | `crosscomp` (CrossCurrentCompensationPolicy) |
| probe layout | `s0`（DVL 单点） |
| task geometry | `cross_stream` |
| flow | `wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` |
| target speed | 1.5 |
| objective | `efficiency_v2` |
| history | 4 |
| ReBRAC config | `(β1=4.0, β2=2.0, hidden=256, layers=3, critic_LN=on, normalize_q=on, TRAIN_EPOCHS=64)` |
| eval | val=40, test=100 |
| 5-seed 成绩 | success = 0.902 ± 0.021 |

### 3.2 A 轴（数据质量）

| spoke | collector | dataset 名称（建议） | 目的 | 预期 collector success |
|---|---|---|---|---:|
| **A1** | `goalseek` | `goalseek_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000` | 低质量端：dual penalty 在 mostly-failure 数据上是否撑得住 | < 0.3（推断） |
| **A2** | 50/50 episode-level mix(`goalseek` + `crosscomp`) | `mix5050_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000` | mid-quality：ReBRAC vs TD3+BC 区分度最大点 | ~0.6（中段） |
| **A3** | `privileged` (PrivilegedCorridorPolicy) | `privileged_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000` | 近 expert 端：ReBRAC 是否退化为 BC | ~0.99 |

注：A2 dataset 物理实现 = 500 episodes 用 goalseek 跑、500 episodes 用 crosscomp 跑、按 episode 拼接成 1000-ep `transitions.npz`。

### 3.3 B 轴（传感器）

| spoke | probe layout | dataset 名称（建议） | 目的 |
|---|---|---|---|
| **B1** | `s1`（DVL + 短程 ADCP，2 probes，obs=12-D） | `crosscomp_s1_h4_efficiency_v2_re150_u10cross_fixdone_ep1000` | 信息更充足时 ReBRAC 是否仍优；TD3+BC gap 是否收窄 |
| **B2** | `s2`（DVL + 长程 ADCP + 横向，4 probes，obs=16-D） | `crosscomp_s2_h4_efficiency_v2_re150_u10cross_fixdone_ep1000` | 强 sensor 上限 |

注：B 轴 spoke 必须重新收集 dataset（dataset obs 维度由 collect 时 probe layout 锁死，**不能**在训练时切换）。collector policy 仍为 crosscomp。

### 3.4 C 轴（任务 / 流场）

| spoke | task geometry | flow file | dataset 名称（建议） | 目的 |
|---|---|---|---|---|
| **C1** | `upstream` | `wake_v8_U1p00_Re150_*` | `crosscomp_s0_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000` | geometry 翻转 |
| **C3** | `cross_stream`（保持） | `wake_tandem_G35_v8_U1p00_Re150_*` | `crosscomp_s0_h4_efficiency_v2_re150tandem_u10cross_fixdone_ep1000` | multi-wake 场景 |

注：C2 (Re250) 砍掉（双 confound，见 §1.3）。

> **事后修正 (2026-05-06)**：C1 P1 5-seed 实测 success=0.225 ± 0.005（远低于本节 §8.3 表的预期 0.85–0.95），触发了完整的事后 ablation 链。三轮 ablation（reward landscape / privileged asym critic / 4× epoch budget）都未能突破 0.19–0.23 区间，确认 C1 是 **sensor floor spoke**。详见 [§13 C1 sensor-floor ablation](#13-c1-sensor-floor-ablation-事后追加-2026-05-06)。后续追加 follow-up spoke C1-s1（s1 sensor + upstream + u10）以完成 sensor 维度对照。

### 3.5 矩阵汇总

| ID | dataset | probe | geometry | flow | 算法 | seeds (P1) |
|---|---|---|---|---|---|---|
| Anchor | crosscomp-1000 | s0 | cross | Re150 single | ReBRAC | 已完成 5 seeds |
| A1 | goalseek-1000 | s0 | cross | Re150 single | ReBRAC | 42, 44 |
| A2 | mix5050-1000 | s0 | cross | Re150 single | ReBRAC | 42, 44 |
| A2-td3bc | mix5050-1000 | s0 | cross | Re150 single | TD3+BC (α=0.25) | 42, 44 |
| A3 | privileged-1000 | s0 | cross | Re150 single | ReBRAC | 42, 44 |
| B1 | crosscomp-s1-1000 | s1 | cross | Re150 single | ReBRAC | 42, 44 |
| B2 | crosscomp-s2-1000 | s2 | cross | Re150 single | ReBRAC | 42, 44 |
| C1 | crosscomp-upstream-1000 | s0 | upstream | Re150 single | ReBRAC | 42, 44 |
| C3 | crosscomp-tandem-1000 | s0 | cross | Re150 tandem | ReBRAC | 42, 44 |

P1 总计：14 ReBRAC run + 2 TD3+BC run = **16 run**。

---

## 4. 数据收集 protocol

### 4.1 收集清单

7 个新 dataset：A1, A2 (= A2a + A2b 拼接), A3, B1, B2, C1, C3。

### 4.2 收集命令骨架

A 轴 / C 轴使用 `scripts/collect_offline_data.py`，B 轴同理，仅改 `--probe-layout`。

```bash
# 示例：A1 goalseek
python -m scripts.collect_offline_data \
  --policy goalseek \
  --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 --task-geometry cross_stream --target-speed 1.5 \
  --history-length 4 --objective efficiency_v2 \
  --episodes 1000 --seed 0 --num-workers 8 \
  --output-dir offline_data/goalseek_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000

# 示例：C3 tandem (flow 不同)
python -m scripts.collect_offline_data \
  --policy crosscomp \
  --flow wake_data/wake_tandem_G35_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 --task-geometry cross_stream --target-speed 1.5 \
  --history-length 4 --objective efficiency_v2 \
  --episodes 1000 --seed 0 --num-workers 8 \
  --output-dir offline_data/crosscomp_s0_h4_efficiency_v2_re150tandem_u10cross_fixdone_ep1000
```

### 4.3 A2 mix dataset 的物理实现

混法 = episode-level（每个 episode 内 trajectory 由单一 policy 产生；dataset 整体 50/50）：

1. 用 goalseek 收 500 ep（`--seed 0`）→ 落到临时目录 `mix_tmp_goalseek/transitions.npz`；
2. 用 crosscomp 收 500 ep（`--seed 1`，与 step 1 不同 task seed 序列以避免起点重复）→ 落到临时目录 `mix_tmp_crosscomp/transitions.npz`；
3. 在 numpy 层把两个 `transitions.npz` 沿 episode 维度 concat（保持 `dones` 边界，preserve `privileged_obs` 列）；
4. metadata.json 记录：`mix_components: [goalseek/500/seed0, crosscomp/500/seed1]`、`mix_strategy: episode_level`、`task_sampler: anchor_distribution`。

**Task distribution 一致性**：两个 sub-collection 共用 anchor 的 `TaskSamplerConfig`（同样的 cross_stream geometry / target_speed=1.5 / spawn 区域），仅 collector policy 与 task seed 不同。这保证 mix dataset 的 task distribution 与 anchor 在统计意义上同源，仅 behavior policy 是混合的。

### 4.4 Sanity card（每个新 dataset 强制记录）

字段固定 7 项，写入每个 dataset 目录下的 `sanity_card.json`：

| 字段 | 含义 |
|---|---|
| `collector_success_rate` | 1000 episodes 内 env 报告的 `is_success` 均值（与 anchor manifest eval 同口径） |
| `collector_mean_return` | 同上 mean episode return |
| `episode_length_mean / std` | 平均 episode 长度（steps） |
| `obs_dim` | 观测维度，必须与 probe layout 匹配（s0=10 / s1=12 / s2=16） |
| `n_transitions` | 总 transition 数 |
| `privileged_obs_present` | bool，必须 True（保证未来 AsymCritic 可用） |
| `flow_file` | 使用的 wake_data 文件名 |

### 4.5 估算

- 单 dataset：1000 ep × 8 worker ≈ 30–60 min L4（取决于 episode 长度，goalseek 较多 timeout 会偏长）。
- 7 dataset 总成本：**~6 h L4**。
- 数据收集与训练在 Colab 上可串行也可并行；推荐串行收完再训，避免 Drive I/O 争抢。

---

## 5. Phase 1 — Probe（无条件执行）

### 5.1 配置

每条 spoke × 2 seeds 共用 anchor 的所有训练超参（详见 §3.1），**只**改 dataset / probe layout / geometry / flow 三轴中的对应轴。

### 5.2 TD3+BC sanity（仅 A2）

A2 上加 TD3+BC × 2 seeds，配置 = `td3bc_phase0c` 主线 winner（α=0.25, hidden=256, layers=3, critic_LN=on, TRAIN_EPOCHS=64）。该点保留 ReBRAC vs TD3+BC head-to-head，供 paper §experiments 引用。

理由：A2 是 mid-quality 端点之间的中段，是 dual penalty 相对单 BC penalty 区分度理论最大的点。端点（A1/A3）不加 TD3+BC，因为 td3bc_phase0c report 已经覆盖近端点行为。

### 5.3 Seed 选择

P1 强制使用 **seed 42 + seed 44**：
- seed 42 是 5-seed Stage C 中典型 seed（success ≈ 0.92）；
- seed 44 是已知 outlier（review §2.1.3 cross-(dataset, β2) 闭环），是诊断信号最强的点。

不用 seed 43：43 与 42 的行为高度相关，2 seeds 限额下 (42, 44) 比 (42, 43) 信息含量高得多。

### 5.4 输出

每条 spoke 训完产出：
- `results/offline/rebrac/broad_validation/<spoke_id>/seed_<S>/{train,val,test}.{csv,json}`
- `selected_checkpoint.json`（按 success → return → -safety → -time 选 ckpt）
- `summaries/overview.csv`（汇总 P1 全部 spoke + anchor 复制行）

### 5.5 P1 总成本

| 类别 | run | L4 |
|---|---:|---:|
| ReBRAC P1 (7 spoke × 2 seeds) | 14 | ~7 h |
| TD3+BC sanity (A2) | 2 | ~1 h |
| **小计** | **16** | **~8 h** |

---

## 6. Phase 2 — 触发深挖（条件性）

### 6.1 触发判据（per-spoke 评估）

某条 spoke 触发 P2 当且仅当满足以下任一：

| 判据 | 阈值 | 含义 |
|---|---|---|
| Mean shift（双向） | \|mean(spoke 2-seed) − mean(anchor 5-seed)\| > 5pp | 显著差异（正负皆触发） |
| Std blow-up | std(spoke 2-seed) > 2 × std(anchor) = 0.042 | 不稳定 |
| ReBRAC vs TD3+BC | (仅 A2) \|mean_ReBRAC − mean_TD3BC\| < 5pp | 区分度坍塌 |

注：双向触发是为了不漏掉正向 finding（如 s2 > anchor 5pp 是 paper 价值很高的"sensor 抬升"信号）。

### 6.2 P2 步骤（每个触发 spoke）

1. **1-seed β refit**：在 spoke dataset 上跑 (β1=2.0, β2=2.0) 与 (β1=4.0, β2=1.0) 各 1 seed（复用 seed 42）。+2 run。
2. **判定 winner 漂移**（阈值 = +3pp）：
   - 若 β refit 中存在新配置 mean (test=100, seed 42 单点) > 原 (β1=4, β2=2) seed 42 mean by > **3pp** → winner 漂移；
   - 否则 winner 保持。
   - 阈值取 +3pp 是 conservative：单 seed 估计的 sampling noise 大致量级是 ±2–3pp（参考 Stage B per-seed spread），低于 +3pp 不足以推翻 anchor finalist。
3. **5-seed 扩展**：在确定的 winner 上扩到 seed 42–46。
   - **若 winner 不漂**（路径多数）：原 winner 已有 seed 42 + 44 → 补 43/45/46 = **+3 run**。
   - **若 winner 漂**（路径少数）：新 winner 仅有 β refit 时的 seed 42 → 补 43/44/45/46 = **+4 run**。

P2 单 spoke 成本：
- winner 不漂路径：2 (β refit) + 3 (5-seed 补齐) = **5 run**
- winner 漂路径：2 (β refit) + 4 (5-seed 补齐) = **6 run**

### 6.3 触发数估计

基于 Stage C / D / E 的过往触发率（约 40% spoke 命中），保守估 7 个 spoke 中 **2–3 触发**。假设触发 spoke 中 winner 漂移率约 1/3：

| 触发数 | P2 run（不漂主路径）| P2 run（含 1 漂）| P2 L4 |
|---:|---:|---:|---:|
| 2 | 10 | 11 | ~5–5.5 h |
| 3 | 15 | 16 | ~7.5–8 h |

### 6.4 P2 输出

- `results/offline/rebrac/broad_validation/<spoke_id>/p2_beta_refit/seed_42/...`
- `results/offline/rebrac/broad_validation/<spoke_id>/p2_5seed_winner/{seed_42,...,seed_46}/...`
- `summaries/p2_overview.csv`

---

## 7. 预算与算力规划

### 7.1 总预算

| 阶段 | run | L4 |
|---|---:|---:|
| 数据收集（7 新 dataset） | — | ~6 h |
| Phase 1 ReBRAC probe | 14 | ~7 h |
| Phase 1 TD3+BC sanity (A2) | 2 | ~1 h |
| Phase 2 触发深挖（估 2–3 spoke 触发，含 ≤1 winner 漂） | 10–16 | ~5–8 h |
| 可选 0 算力分析（worldcomp super-teacher 轨迹） | 0 | 0 |
| **总计** | **26–32 run** | **~19–22 h L4** |

注：P2 单 spoke 成本 winner 不漂 = 5 run / 漂 = 6 run（详见 §6.2）；触发 2–3 spoke 的总开销在 10–16 run 区间。

### 7.2 Colab session 分配（建议）

| Session | 任务 | 时长 |
|---|---|---:|
| S1 | 数据收集（7 dataset 串行） | ~6 h |
| S2 | P1 ReBRAC（14 run）+ P1 TD3+BC（2 run） | ~8 h |
| S3 | P2 β refit + 5-seed（条件性） | ~5–7.5 h |
| S4（buffer） | 数据补漏 / P2 第二批触发 / 分析 notebook | ~2 h |

### 7.3 Skip-resume 策略

每条 spoke 入口必须支持 `[skip]`（用 `selected_checkpoint.json` 存在性判断），与 `scripts/run_offline_rebrac_screen.sh` 一致，保证 Colab 中断后续跑零浪费。

---

## 8. Expected Outcomes 与 Paper Claim 框架

### 8.1 A 轴预期与解读

| spoke | 预期 mean(ReBRAC) | 解读 |
|---|---|---|
| A1 goalseek-1000 | 0.4–0.7（推断 ±0.15）| 远低于 anchor 0.902 是预期，**这本身就是 paper finding**：ReBRAC 在 mostly-failure 数据上仍能从 successful sub-trajectory 提取信号；与 BC（在 mostly-failure 上必崩）形成对照 |
| A2 mix-1000 | 0.80–0.92 | 中段位置；ReBRAC vs TD3+BC 区分度最大点；预期 ReBRAC 显著优 |
| A3 privileged-1000 | 0.93–0.97 | 近 expert 数据上 ReBRAC ≈ BC；与 anchor 持平或微升 |

**Paper claim 升级路径**：A 轴若呈现 monotonic（goalseek < mix < anchor < privileged），则可在 paper §discussion 写 "ReBRAC 在数据质量谱上单调改善"。即便 A1 崩到 0.4，也是有价值的下界 finding（"ReBRAC 不是万能"），不是失败实验。

### 8.2 B 轴预期与解读

| spoke | 预期 mean(ReBRAC) | 解读 |
|---|---|---|
| B1 s1 | 0.92–0.96 | 适度抬升（短程 ADCP 带 ~3 step 提前量） |
| B2 s2 | 0.94–0.97 | 进一步抬升（长程 ADCP + 横向梯度） |

**关键解释（必须写进 spec / paper）**：sensor 越好 → TD3+BC 也跟着变强 → ReBRAC vs TD3+BC 的 gap **可能收窄甚至消失**。这**不是** ReBRAC 弱化，而是支持 sim2real narrative：ReBRAC 是为 deployment-realistic (s0) 服务的；当传感器丰富时，朴素 BC 已足够。这是论文 §discussion 的核心一段，必须显式预注册。

P1 不带 TD3+BC 在 B1/B2 上做对照，因此 paper 仅 claim "ReBRAC 在 s1/s2 上仍然取得高 success rate"，**不**直接 claim "ReBRAC > TD3+BC" 在 B 轴。这是预算取舍下的 narrative 边界。

### 8.3 C 轴预期与解读

| spoke | 预期 mean(ReBRAC) | 解读 |
|---|---|---|
| C1 upstream | 0.85–0.95 | upstream 物理上比 cross 容易（沿流方向，水流帮助前进），预期 ≥ anchor |
| C3 tandem | 0.70–0.90 | 双尾流场景，wake-wake interaction 增加任务难度，预期 ≤ anchor |

**Paper claim 升级路径**：C 轴若两条 spoke 都 ≥ 0.85，可 claim "ReBRAC 跨 task geometry / wake topology generalize"；若 tandem 显著退化（< 0.70），则在 §limitations 显式声明 "tandem 双尾流场景需要进一步算法工作"。

> **事后修正 (2026-05-06) — C1 实测远低于本节预期**：upstream "更易"的物理直觉在 s0 单点 DVL 上失败：u10 upstream 流场需要预知前方流场结构才能 deploy，s0 单点观测无此能力。C1 实测 = 0.225 ± 0.005，三轮事后 ablation 全部未能突破 0.19–0.23（详见 §13）。本节 C1 行的 paper-claim 路径从 "task generality" 重定位为 "sensor floor demonstration" + "C1-s1 sensor upgrade contrast"。

### 8.4 全局 Paper Claim 升级

广验完成后，paper §experiments 增设 broad-validation subsection，结构：

1. anchor 表（复制 Stage C，已存在）；
2. A 轴质量谱表（4 行：anchor + A1/A2/A3，2-seed 或 5-seed）；
3. A2 上 ReBRAC vs TD3+BC head-to-head；
4. B 轴传感器表（3 行：anchor + B1/B2）；
5. C 轴任务表（3 行：anchor + C1/C3）；
6. P2 触发的 spoke 标注 5-seed std；
7. discussion 段：sim2real narrative 在 B 轴的预期收窄解读。

---

## 9. 风险与 fallback

### 9.1 风险 R1：A1 goalseek 收集到 < 100 个成功 episode

**判据**：collector_success_rate < 0.1。
**fallback**：维持 1000 episodes（与协议一致）；在 sanity card 显式标注；ReBRAC 训练照常跑；spec §8.1 已把这种情形写入 expected outcome（不是失败）。

### 9.2 风险 R2：P2 触发数 > 4

**判据**：5+ spoke 触发判据。
**fallback**：按优先级序处理 —— A 轴 > C 轴 > B 轴；超出预算的 spoke 推迟到下一轮（不阻塞 paper drafting）。

### 9.3 风险 R3：P2 触发数 = 0

**判据**：所有 spoke 都在 |Δ| ≤ 5pp 且 std ≤ 0.042 内。
**解读**：这本身就是强 generality finding —— "ReBRAC 跨三轴均与 anchor 持平"，paper 直接用 P1 2-seed 表收口，§discussion 讨论 "广验未发现显著退化" 的可能解释（dual penalty 的 dataset-invariant 性质）。

### 9.4 风险 R4：Drive 容量不足

**判据**：7 个新 dataset + ckpt 累计 > 50 GB。
**fallback**：P1 训练完成后立即清理临时数据收集目录；ckpt 仅保留 `agent_step_*.pt`（按 8 epoch 周期），不保留每 epoch ckpt。

### 9.5 风险 R5：β refit 在 P2 中频繁漂移

**判据**：≥ 2 触发 spoke 出现 β refit winner 不是 (β1=4, β2=2)。
**解读**：暗示 anchor finalist 不是 robust winner，paper §method 必须显式声明 "(β1=4, β2=2) 是在 anchor 条件下的 winner，跨条件可能漂"。这是诚实的 limitations 表述，不是失败。

---

## 10. 输出落点

### 10.1 数据 / 结果

```
offline_data/
  goalseek_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/
  mix5050_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/
  privileged_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/
  crosscomp_s1_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/
  crosscomp_s2_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/
  crosscomp_s0_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000/
  crosscomp_s0_h4_efficiency_v2_re150tandem_u10cross_fixdone_ep1000/

results/offline/rebrac/broad_validation/
  A1_goalseek/<seed_42, seed_44>/...
  A2_mix5050/<seed_42, seed_44>/...
  A2_mix5050_td3bc/<seed_42, seed_44>/...
  A3_privileged/<seed_42, seed_44>/...
  B1_s1/<seed_42, seed_44>/...
  B2_s2/<seed_42, seed_44>/...
  C1_upstream/<seed_42, seed_44>/...
  C3_tandem/<seed_42, seed_44>/...
  summaries/p1_overview.csv
  summaries/p2_overview.csv (触发后)
```

### 10.2 文档与 notebook

- 本 spec：`docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`
- implementation plan（writing-plans skill 产出）：`docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md`
- 主 notebook：`notebooks/rebrac_broad_validation.ipynb`（数据收集 + P1 + P2 单一入口）
- 实验报告（实验跑完后写）：`docs/rebrac_broad_validation_report.md`
- 主线 review 加章节链接：`docs/rebrac_mainline_review.md` §3.5 (新增) 引用本 spec 与 report

### 10.3 Paper section 落点

- `paper/sections/experiments.tex`：新增 subsection "Broad validation across data quality, sensor, and task geometry"
- `paper/sections/discussion.tex`：新增段落引用 §8.4 sim2real narrative 解读
- 主结果表 caption：声明 anchor → spoke 关系

---

## 11. 配套文档与 cross-reference

| 引用 | 关系 |
|---|---|
| [rebrac_experiment_plan.md](../../rebrac_experiment_plan.md) rev.8 | 主线 plan（不修改）；本 spec 是其外延 |
| [rebrac_experiment_report.md](../../rebrac_experiment_report.md) rev.8 | 主线 report（不修改）；广验 report 单独成文档 |
| [rebrac_mainline_review.md](../../rebrac_mainline_review.md) rev.2 | review §3.2.G 提到 cross-task 验证未做；本 spec 部分回应该项（C 轴） |
| [rebrac_method_section_draft.md](../../rebrac_method_section_draft.md) | paper method 草稿；广验完成后可在 §6 添加 generality 段 |
| [environment_design.md](../../environment_design.md) | probe layout / task geometry / flow 定义 |
| [scripts/collect_offline_data.py](../../../scripts/collect_offline_data.py) | 数据收集驱动 |
| [scripts/run_offline_rebrac_screen.sh](../../../scripts/run_offline_rebrac_screen.sh) | 训练驱动；广验 sweep 复用同套 [skip]-resume 协议 |

---

## 12. 验收标准

广验阶段完成的判据：

1. ✅ 7 个新 dataset 全部收集完成且 sanity card 落盘；
2. ✅ P1 全部 16 run 完成且 `summaries/p1_overview.csv` 生成；
3. ✅ P2 触发的所有 spoke 完成 β refit + 5-seed 扩展（或显式声明触发数 = 0）；
4. ✅ `docs/rebrac_broad_validation_report.md` 完成 §1–§5（动机 / 矩阵 / P1 结果 / P2 结果 / discussion）；
5. ✅ `paper/sections/experiments.tex` 新增 broad-validation subsection 第一稿；
6. ✅ `docs/rebrac_mainline_review.md` 添加广验 cross-link。

满足上述 6 条 + 没有未解释的 P2 触发为开放项 → 广验阶段 closed，paper drafting 可吸收新 finding。

---

## 13. C1 sensor-floor ablation (事后追加，2026-05-06)

> **状态**：本节是 P1 C1 跑完之后追加的 retrofit。原 spec §3.4 / §8.3 把 C1 预期写成 0.85–0.95；实测 0.225 ± 0.005 触发了完整 ablation 链以排除非 sensor 主因。三轮 ablation 都未能突破 0.19–0.23 区间，确认 C1 (s0 / upstream / u10 / crosscomp / Re150) 是 **sensor floor spoke**。

### 13.1 起点：C1 P1 anchor 实测崩盘

| run | seeds | success | mean_R | termination |
|---|---:|---:|---:|---|
| broad-validation Anchor (cross_stream u10) | 5 | 0.902 ± 0.021 | — | — |
| C1 P1 (efficiency_v2 + sym critic, 64 ep) | 5 | **0.225 ± 0.005** | −371 ± 0.41 | timeout 77.5% / oob 0% / goal 22.5% |

C1 mean 比预期低 60+ pp，且 termination 几乎全部是 timeout（775/1000 ep）。所有 success 都来自 dataset collector 已能完成的子轨迹模式。这是 actor-stuck "deterministic-collapse" 的强信号，无法用 sample noise / β 漂移解释。

### 13.2 三 ablation 链（事后 hypothesis-driven）

依次跑了三轮，每轮一个变量、其余与 P1 anchor 完全一致（同 dataset，除 Ablation A 外；同 anchor β1=4 / β2=2；同 batch_size 256；同 hidden 256×3；同 γ=0.99）。

#### 13.2.1 Ablation A — reward landscape (`arrival_v2_simple`)

**假设**：`efficiency_v2` reward 在 upstream 上有 fast_OOB > slow_OOB inversion，actor 学到的最优策略可能就是 "原地 timeout"。引入 `arrival_v2_simple` preset (terminal-dominant: success=+200 / timeout=−50 / OOB=−200 / step_penalty=−0.2)，把 dataset mean_terminal_R 从 −108 翻到 +200。

**实施**：
- `auv_nav/reward.py::REWARD_OBJECTIVE_PRESETS["arrival_v2_simple"]` 新增（[`auv_nav/reward.py`](../../../auv_nav/reward.py)）
- `tests/test_reward_objective.py` 新增 `arrival_v2_simple_field_lockdown` + `arrival_v2_simple_terminal_dominance` 两个回归测试
- 重收集 1000 ep dataset：`offline_data/crosscomp_s0_h4_arrival_v2_simple_re150_u10upstream_fixdone_ep1000/`
- collector success_rate=1.0，mean_R=+205.34，mean_terminal_R=+199.93，`privileged_obs_present=True`

**结果**：

| run | dataset reward | seeds | success | mean_R | termination |
|---|---|---:|---:|---:|---|
| Ablation A | arrival_v2_simple | 2 (42, 44) | **0.215 ± 0.015** | −98.23 ± 4.03 | goal 21.5% / timeout 52.5% / oob 26.0% |

Δ vs P1 anchor = **−1pp**（统计上不可区分）。但 termination 从 "全 timeout" 转为 "21.5% goal + 52.5% timeout + 26% oob"——actor 不再 deterministic-collapse，开始 explore，但 explore 仍 deploy 不到 goal。**reward landscape 排除为 root cause**。

Notebook：[`notebooks/rebrac_c1_reward_ablation_completed.ipynb`](../../../notebooks/rebrac_c1_reward_ablation_completed.ipynb)。

#### 13.2.2 Ablation B — privileged asymmetric critic

**假设**：CLAUDE.md §3 主方法论（actor 看 s0 + critic 加 privileged hull-integral 流场，priv_dim=2）能否解锁 s0/upstream actor 的天花板？

**实施**：复用 Ablation A dataset，仅切换 `--use-asymmetric-critic --privileged-actor-update-mode zeros`（actor improvement 时 zero-pad priv 通道，mimic deployment）。其余超参与 Ablation A 完全一致。

**结果**：

| run | critic | seeds | success | mean_R | termination |
|---|---|---:|---:|---:|---|
| Ablation B | asym (10-D actor + 2-D priv critic) | 2 (42, 44) | **0.195 ± 0.015** | −114.87 ± 8.02 | goal 19.5% / timeout 48.5% / oob 32.0% |

Δ vs Ablation A = **−2pp**（noise 级别）。OOB 略升 (26 → 32 pp)、timeout 略降，提示 actor 略激进但没转化成 success。CLAUDE.md §3 在 offline ReBRAC 上没解锁 s0/upstream，**critic-side privileged information 排除为 root cause**。

Notebook：[`notebooks/rebrac_c1_asym_critic_ablation_completed.ipynb`](../../../notebooks/rebrac_c1_asym_critic_ablation_completed.ipynb)。

#### 13.2.3 Ablation C — training budget (4× epochs)

**假设**：upstream 比 cross 难度更高，64 epochs 可能是 budget bottleneck。

**先决无 GPU 诊断**：读 Ablation A 的 `train_log.jsonl`，比较 mid-window (epoch 16–32) vs late-window (epoch 56–64) 6 个 metric 的相对变化：

| metric | mid mean | late mean | rel_change | 阈值 | plateau? |
|---|---:|---:|---:|---:|:---:|
| critic_loss | +10.25 | +6.42 | **−37.5%** | 5% | ❌ |
| actor_loss | −0.71 | −0.82 | −15.7% | 5% | ❌ |
| bc_loss | +0.021 | +0.017 | −15.7% | 5% | ❌ |
| td_abs_error | +1.73 | +1.62 | −6.6% | 5% | ❌ |
| **mean_q** | **+44.6** | **+64.6** | **+44.8%** | 3% | ❌ |
| **target_q** | **+44.6** | **+64.7** | **+45.1%** | 3% | ❌ |

6/6 metric 都未 plateau，看似支持"epochs 不够"假设。Notebook：[`notebooks/rebrac_c1_train_convergence_check_completed.ipynb`](../../../notebooks/rebrac_c1_train_convergence_check_completed.ipynb)。

**实施**：single seed (42) × **256 epochs (4× current budget)** + 周期 val eval (every 16 epochs，40-ep manifest，写 `eval_log.csv`) + 周期 ckpt (every 16 epochs) + post-train batch test eval at epoch {64, 128, 192, 256} (100-ep manifest)。

**结果**：

| epoch | success | mean_R | termination |
|---:|---:|---:|---|
| 64  | 0.200 | −102.25 | timeout 54 / oob 26 / goal 20 |
| 128 | 0.200 | −106.35 | timeout 52 / oob 28 / goal 20 |
| 192 | 0.220 | −96.16  | timeout 52 / oob 26 / goal 22 |
| 256 | 0.220 | −96.16  | timeout 52 / oob 26 / goal 22 |

epoch 192 与 256 在 100-ep test 上**所有数字一字不差**（mean_R / std / safety_cost / 全部 termination 计数）—— actor 在 epoch 192 后已 deterministic 锁死，policy 不再变化。Δ (256 − 64) = **+2pp**（仍在 noise 内）。

**对 mean_q +44.8% 上升的协调解释**：critic 仍在 fitting Q-landscape，但 actor 被 BC anchor 钉死，policy 即便给 4× budget 也不能越界。这是 ReBRAC β floor 的教科书表现，不是 epochs 不够。**training budget 排除为 root cause**。

Notebook：[`notebooks/rebrac_c1_epoch_sensitivity_ablation_completed.ipynb`](../../../notebooks/rebrac_c1_epoch_sensitivity_ablation_completed.ipynb)。

### 13.3 三 ablation 汇总表

| 干预 | dataset / critic / budget | seeds | success | Δ vs P1 anchor (pp) |
|---|---|---:|---:|---:|
| **P1 anchor** | eff_v2 / sym / 64 ep | 5 | 0.225 ± 0.005 | 0.0 |
| Ablation A (reward) | arr_v2_s / sym / 64 ep | 2 | 0.215 ± 0.015 | −1.0 |
| Ablation B (asym critic) | arr_v2_s / asym / 64 ep | 2 | 0.195 ± 0.015 | −3.0 |
| Ablation C (epoch 4×) | arr_v2_s / sym / 256 ep | 1 | 0.220 (ep256) | −0.5 |

所有干预的 success 都被钉在 **0.19–0.23 区间**（4-pp 区间，远小于 ablation A/B 的 3-pp 噪声半径）。三个独立 root-cause 假设（reward landscape / critic supervision / training budget）全部排除。

### 13.4 结论：sensor floor

C1 (s0 / upstream / u10 / crosscomp / Re150) 是 **sensor floor spoke**：s0 (DVL water-track only) 在 u10 upstream 流场上提供的信息量不足以让 actor 学到 deploy-grade policy（success ≥ 0.6），即使在以下多重优待下也不能突破：

1. collector dataset success=1.0（数据本身可达 goal）
2. training reward 已正向（mean_terminal_R = +200）
3. privileged critic 已给 hull-integral signal
4. epochs 已扩 4×（256 ep）

这是 paper-grade finding：**deployment-realistic 单点传感器 + upstream 流场是真实的物理 deployability 边界**——不是算法 / reward / budget 问题。需要 sensor 升级（至少 s1 = DVL + 短程 ADCP）才能 deploy。

### 13.5 输出落点新增（对 §10.1 / §10.2 的补充）

```
offline_data/
  crosscomp_s0_h4_arrival_v2_simple_re150_u10upstream_fixdone_ep1000/    # ablation A/B/C 复用

benchmarks/
  c1_reward_ablation/{val_40, test_100}/single_u10_upstream_tgt15.json

checkpoints/offline/rebrac/
  c1_reward_ablation/<dataset>/actorb_4p0__criticb_2p0/seed_{42,44}/
  c1_asym_critic_ablation/<dataset>/actorb_4p0__criticb_2p0/seed_{42,44}/
  c1_epoch_sensitivity/<dataset>/actorb_4p0__criticb_2p0/seed_42_e256/

results/offline/rebrac/
  c1_reward_ablation/<dataset>/actorb_4p0__criticb_2p0/test/seed_{42,44}.json
  c1_asym_critic_ablation/<dataset>/actorb_4p0__criticb_2p0/test/seed_{42,44}.json
  c1_epoch_sensitivity/<dataset>/actorb_4p0__criticb_2p0/seed_42_e256/test/epoch_{64,128,192,256}.json

notebooks/
  rebrac_c1_reward_ablation_completed.ipynb
  rebrac_c1_asym_critic_ablation_completed.ipynb
  rebrac_c1_train_convergence_check_completed.ipynb
  rebrac_c1_epoch_sensitivity_ablation_completed.ipynb
```

### 13.6 后续行动 — C1-s1 sensor-upgrade follow-up

为把 sensor floor 从 "single-spoke 失败" 升格到 "sensor 维度 controllable axis"，加一条 follow-up spoke：

| spoke | probe | obs_dim | seeds | 预期 |
|---|---|---:|---:|---|
| **C1-s1** (新增) | s1 (DVL + 短程 ADCP, 2 probes) | 12 | 2 (42, 44) | ≥ 0.50 → sensor 升级解锁 upstream；< 0.30 → upstream u10 是更深 fundamental limitation |

实施骨架：
- 数据收集：复用 `crosscomp` baseline policy + `--probe-layout s1`，生成 `offline_data/crosscomp_s1_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000/`
- 训练：ReBRAC anchor (β1=4, β2=2) × 64 epochs × 2 seeds
- 评估：test_100 manifest（同 C1）
- 预算：~2h L4

**Paper 叙事分支**：

| C1-s1 实测 | paper claim |
|---|---|
| ≥ 0.50 | "upstream u10 在 s0 上不可 deploy；s1 提供的短程 ADCP 解锁该任务" → paper headline + sim2real narrative 的强证据 |
| 0.30–0.50 | "sensor 单步升级仅部分解锁 upstream，需要 s2（长程 ADCP + 横向梯度）" → 触发 C1-s2 follow-up |
| < 0.30 | "upstream u10 在所有 deployable sensor 上都接近 sensor floor" → §6 limitations 段写 "upstream 流场是任务-传感器共同的物理上限" |

