# ReBRAC C1-s1 Sensor-Upgrade Follow-up — Standalone Report

> ---
> **⚠ SUPERSEDED 2026-05-18**：本 v1 C1-s1 follow-up 已被 v2 plan [`docs/rebrac_broad_validation_v2_plan.md`](rebrac_broad_validation_v2_plan.md) 取代。
>
> 本 follow-up 在 `efficiency_v2` + upstream geometry 下做的 sensor s0→s1 ablation 结论（C1 task-fundamental floor 在 upstream + crosscomp + u10 + target=1.5 下成立）保留作 v1 archive，不重跑、不进 paper。
>
> v2 plan 改变 framing：upstream geometry 整体砍掉（online §7.2 已 saturate），sensor envelope 故事重心转到 cross_stream 下的 s0 vs s1 对照（v2 N1 / N3 cell + online §7.6 平行证据）。原 follow-up 提出的「C1-s2 backlog」「target_speed=2.0 probe」「BC penalty sweep on C1」三项都已在 v2 plan 中以不同形式处理或砍掉，详见 v2 plan §3 diff 表 + §9 backlog。
> ---
>
> **Date**: 2026-05-07
> **Status (2026-05-07 user direction 升级)**：本 follow-up **保持 standalone exploratory side study**，**不回写**主报告 [`rebrac_experiment_report.md`](rebrac_experiment_report.md)。原稿 §1/§7.1/§9 多处「待 later main-report retrofit」「下一次主报告 retrofit 时统一集成」措辞已过期——按用户 2026-05-07 判断，task-fundamental floor claim 的 mechanism discriminator（**BC penalty 强度 sweep on C1**，β1 ∈ {0, 1, 2, 4, 8}）未做之前，结论尚不达 paper-quality；主报告 retrofit deferred until 该 sweep 闭环。
>
> 主报告 §10A 已**退回**为 pointer（删除 cf5cfff 的 sensor-floor framing 与 verdict gate），不再保留与本 follow-up 实测矛盾的「预期 sensor 升级解锁」论述。
>
> **Cross-references**:
> - Main report（已退回 §10A 为 pointer，不集成本 follow-up）: [`docs/rebrac_experiment_report.md`](rebrac_experiment_report.md) §10A
> - 同源 broad validation 报告（同样 standalone）: [`docs/rebrac_broad_validation_report.md`](rebrac_broad_validation_report.md)
> - Spec (full evidence): [`docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`](superpowers/specs/2026-05-04-rebrac-broad-validation-design.md) §13
> - Plan (Task 11A Step 8 closed 2026-05-07): [`docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md`](superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md)
> - Notebook archive: [`notebooks/rebrac_c1_s1_sensor_upgrade_completed.ipynb`](../notebooks/rebrac_c1_s1_sensor_upgrade_completed.ipynb)

---

## 1. Background

广验 C1 spoke (`crosscomp / s0 / upstream / u10 / Re150`，`efficiency_v2`，64 epochs，β1=4 / β2=2，5 seeds) P1 anchor 实测 **success = 0.225 ± 0.005**，远低于 spec §8.3 期望的 0.85–0.95。三轮 ablation 跟进（详见主报告 §10A.2）：

| 干预 | dataset / sensor / critic / budget | seeds | success | Δ vs P1 (pp) |
|---|---|---:|---:|---:|
| P1 anchor | s0 / eff_v2 / sym / 64 ep | 5 | 0.225 ± 0.005 | 0.0 |
| Ablation A: reward swap | s0 / arr_v2_s / sym / 64 ep | 2 | 0.215 ± 0.015 | −1.0 |
| Ablation B: asym critic | s0 / arr_v2_s / asym / 64 ep | 2 | 0.195 ± 0.015 | −3.0 |
| Ablation C: epoch 4× | s0 / arr_v2_s / sym / 256 ep | 1 | 0.220 (ep 256) | −0.5 |

主报告 §10A.3 收口为 **sensor floor spoke**：deployment-realistic 单点 DVL 在 u10 upstream 上信息不足；**预期** sensor 升级到 s1 (DVL + 短程 ADCP, 12-D) 能解锁 → §10A.5 列「C1-s1 未跑」为待验证 limitation。

本 follow-up 直接测试该预期。

## 2. Hypothesis & verdict gate（事先 commit）

唯一变量：`--probe-layout s0 → s1`（其余完全等同 C1 P1 anchor：crosscomp / upstream / target_speed=1.5 / history=4 / efficiency_v2 / β1=4 β2=2 / 64 ep）。

| C1-s1 实测 success | verdict | next step |
|---:|---|---|
| **≥ 0.50** | sensor 升级解锁 | paper headline + sim2real narrative；考虑 5-seed bootstrap CI |
| **0.30–0.50** | 部分有效 | 触发 C1-s2 follow-up（s2 = 4 probes, 16-D） |
| **< 0.30** | task-fundamental floor | C1 spoke 在所有 deployable sensor 上接近 floor；§6 limitations 表述升级 |

## 3. Setup

- **Sensor**: s1 = 2 probes at (0, 0) + (4.5, 0) m，DVL water-track + 2 MHz 短程 ADCP，~3 步前向流场 advance warning，per-step obs_dim=12 → dataset obs_dim=48 (× history=4)
- **Dataset**: `offline_data/crosscomp_s1_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000/`
  - collector_success_rate = **1.0**（s1 比 s0 多 advance warning，crosscomp 在 s1 上 100% 成功）
  - n_transitions = 268,329；mean_return = −108.21；episode_length = 268.3 ± 66.7
- **Train**: ReBRAC anchor (β1=4.0, β2=2.0) × seeds {42, 44} × 64 epochs，sym critic
- **Eval**: 复用 `benchmarks/c1_reward_ablation/test_100/single_u10_upstream_tgt15.json`（与 P1 anchor + 三轮 ablation 共用 manifest，跨 5 配置可比）
- **Cost**: ~2h L4

## 4. Result

### 4.1 Per-seed test (100 ep, deterministic)

| seed | success | mean_R | safety_cost | termination (goal / timeout / oob) |
|---:|---:|---:|---:|---|
| 42 | 0.210 | −379.98 | 26.06 | 21 / 53 / 26 |
| 44 | 0.200 | −383.42 | 25.27 | 20 / 54 / 26 |
| **mean** | **0.205 ± 0.005** | **−381.70 ± 1.72** | **25.66** | **20.5 / 53.5 / 26.0** |

### 4.2 Five-way comparison (all on `c1_reward_ablation/test_100`)

| config | n_seeds | success | mean_R | termination (goal / timeout / oob) |
|---|---:|---:|---:|---|
| s0 + eff_v2 + sym + 64 ep (P1) | 5 | 0.225 ± 0.005 | −371.0 | 22.5 / **77.5** / 0.0 |
| s0 + arr_v2_s + sym + 64 ep (Abl A) | 2 | 0.215 ± 0.015 | −98.2 | 21.5 / 52.5 / 26.0 |
| s0 + arr_v2_s + asym + 64 ep (Abl B) | 2 | 0.195 ± 0.015 | −114.9 | 19.5 / 48.5 / 32.0 |
| s0 + arr_v2_s + sym + 256 ep (Abl C) | 1 | 0.220 | −96.2 | 22.0 / 52.0 / 26.0 |
| **s1 + eff_v2 + sym + 64 ep (C1-s1)** | 2 | **0.205 ± 0.005** | **−381.7** | 20.5 / 53.5 / 26.0 |

- Δ vs P1 anchor = **−2.0 pp**（在 ±1.5 pp single-run noise radius 内）
- 全部 5 行 success 钉在 **0.195–0.225** 区间（3-pp 全幅，小于 single-run 噪声半径的 2×）

## 5. Verdict

**< 0.30 → task-fundamental floor**（事先 commit 的判定规则触发）。

主报告 §10A.3 的 "sensor floor" 框架在 C1-s1 实测下应升格为 **task-fundamental floor**：
- s0 (10-D, DVL only) 失败
- s1 (12-D, DVL + 短程 ADCP) 同样失败 → sensor 维度不是 deployability lever
- 4 个独立干预（reward / privileged critic / 4× budget / sensor 升级）全部钉在同一 ceiling

C1 spoke (`crosscomp / upstream / u10 / Re150 / target_speed=1.5`) 是一个 **deployment-impossible task-dataset combination at deployable sensors s0/s1**——upstream u10 流速 + crosscomp dataset + target_speed=1.5 在 deployable sensors s0/s1 上的物理上限。s2 (16-D) 与 target_speed=2.0 未测，记入 §7 backlog。

## 6. Findings

### 6.1 Primary: task-fundamental floor (vs sensor floor)

升格原因：sensor 升级是 deployment-realism 视角下的最 obvious lever（更多前向流场 advance warning → 理论上 actor 能预判流场结构）。该 lever 失效（−2 pp 在噪声内）说明边界**不在 sensor 维度**，而在「u10 upstream 流速 + crosscomp 行为分布」共同决定的 task-dataset 上限。

### 6.2 Secondary: reward 与 sensor 都是 failure-mode 的独立 driver，但都不动 ceiling

观察 termination 分布在 5 个配置之间的切换模式：
- **eff_v2 + s0 (P1)**：timeout-dominated (77.5 / 0)——actor 保守，几乎全程留在边界内但超时
- **arr_v2_s + s0 (Abl A/C)**：timeout/oob mixed (~53 / ~26)——保 reward 不变换 reward 后 actor 更激进
- **eff_v2 + s1 (C1-s1)**：timeout/oob mixed (53.5 / 26.0)——保 reward 不变只换 sensor，actor **同样**变激进

⚠ **重要订正**：原稿曾写「reward governs failure mode」是错的。C1-s1 与 P1 **同 reward** (eff_v2)，仅 sensor 升级 s0→s1 就把 0 oob 翻到 26 oob。**reward 与 sensor 都是独立 driver**：两条不相交的路径（换 reward 或换 sensor）都能把 timeout-only (77.5/0) 翻到 mixed (~53/~26)。

但 **无论 failure mode 如何切换**，goal-reaching ceiling 都钉在 0.195–0.225。这是更强的 task-fundamental 信号——actor 行为模式（保守 vs 激进）被两个独立 driver 各自调节，goal-reaching 的物理上限却不动 → ceiling 来自 task / data 更上游，不是 actor 探索风格的 function。

### 6.3 ReBRAC β floor signature（待 BC penalty sweep 验证）

Ablation C 的 epoch sensitivity（64 / 128 / 192 / 256）在 100-ep test 上 epoch 192 与 256 **所有数字一字不差**——actor 已 deterministic-locked。convergence diagnostic 显示 critic 末段还在 +44.8% 上升、actor 已被 BC anchor 钉死，看上去像 ReBRAC 在 sensor-info-deficient 条件下的 β floor signature：critic 仍在 fitting Q-landscape，但 actor 不能越界。C1-s1 的 sensor 升级也不能改变这一点。

⚠ **论证缺口**：当前论证依赖「critic 仍动 + actor 不动 = β floor」的解读，但严格的 β floor 判定应通过 **BC penalty 强度 sweep**（β1 ∈ {0, 1, 2, 4, 8}）：若降 β1 不能恢复 success → β floor 升格为 task-fundamental 现象（actor 即使从 BC anchor 释放也碰不到更高的 success）；若降 β1 能恢复 → 当前 ceiling 是 BC penalty 的副产品而非 task-fundamental。**该 sweep 未做**，记入 §7 backlog。

## 7. Implications

### 7.1 Paper-narrative 升级建议（**deferred — 待 mechanism discriminator 闭环后再考虑统一集成**）

> ⚠ **2026-05-07 status 升级**：原标题「主报告下一次 retrofit 时集成」已过期。task-fundamental floor claim 的 mechanism discriminator（BC penalty 强度 sweep on C1）未做之前，下面这段升级建议**仅作为 candidate paper narrative 留底**，主报告 retrofit deferred；paper writing 阶段如需 generality 章节，可直接综合 broad_validation + 本 follow-up，不必走主报告 retrofit。

C1 spoke 的论文角色（candidate narrative，待 BC penalty sweep 验证 task-fundamental claim 后再正式采用）：

> **C1 demonstrates a deployment-impossible task-dataset combination at deployable sensors s0/s1 and target_speed=1.5**: four independent interventions（reward landscape / privileged asym critic / 4× training budget / deployable sensor upgrade s0 → s1）all fail to break the 0.195–0.225 ceiling. Reward 与 sensor 各自独立 modulate failure mode (timeout-only ↔ timeout/oob mixed)，but neither moves the goal-reaching ceiling. upstream u10 + crosscomp dataset 在 deployable sensors s0/s1 与 target_speed=1.5 下构成 task-fundamental floor；s2 (16-D) 与 target_speed=2.0 未测，记入 backlog。

与 Stage D Phase 2 finding（cross_stream + worldcomp dataset 上 deployable→teacher gap 关闭 ~52~58%）形成完整 deployability map：
- **deploy-graded 区间**（cross_stream + worldcomp，s0）：algorithmic lever 有效（ReBRAC + deployable obs 已可关闭主要 gap）
- **deploy-impossible 区间**（upstream u10 + crosscomp，s0/s1）：4 维度 lever 全失效 → 任务-数据集物理边界

两个 finding 合在一起为 sim2real 论文提供**完整的 deployability 谱系**：哪些任务 deployable 可解（algorithmic）、哪些是物理边界（task fundamental）。

### 7.2 C1-s2 不再触发

原 verdict gate 中的 0.30–0.50 区间对应 C1-s2 (s2, 4 probes, 16-D) follow-up。C1-s1 实测落在 < 0.30 → 边界**不在 sensor 维度**，s2 升级（信息量更高）很可能也落在同一区间，trade-off 与 cost (~2h L4) 不值得；记入 backlog 但不在当前 broad validation 范围内执行。

### 7.3 可选 follow-up: target_speed=2.0 (downstream)

把 `target_speed` 从 1.5 提升到 2.0（顺流方向更快）测试是否解锁 deployability。单 seed P1 probe ~1h L4：
- success ≥ 0.5 → 验证「u10 upstream 是 task-fundamental，但更高目标速度反向更易」→ 触发更细 ablation
- success < 0.3 → 进一步确认「u10 upstream + crosscomp 在所有 task-tunable parameter 下都接近 floor」

记入 plan Task 11A Step 9 backlog。

## 8. Outputs

```
offline_data/
  crosscomp_s1_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000/
    transitions.npz
    metadata.json
    sanity_card.json    # collector_success_rate=1.0, obs_dim=48, n_transitions=268329

checkpoints/offline/rebrac/c1_s1_sensor_upgrade/
  crosscomp_s1_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000/
    actorb_4p0__criticb_2p0/
      seed_42/agent_final.pt + trainer_state.json + train_log.jsonl
      seed_44/agent_final.pt + trainer_state.json + train_log.jsonl

results/offline/rebrac/c1_s1_sensor_upgrade/
  crosscomp_s1_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000/
    actorb_4p0__criticb_2p0/test/
      seed_42.json   # success=0.21, mean_R=-379.98
      seed_44.json   # success=0.20, mean_R=-383.42

notebooks/
  rebrac_c1_s1_sensor_upgrade_completed.ipynb    # 完整 run-archive
```

## 9. Limitations of this follow-up

1. **2 seeds**：与 P1 anchor 5 seeds 不对称；3-pp 总区间小于 single-run 噪声半径已足以支撑 verdict，但严格的 effect-size CI 需要升 5-seed × bootstrap。优先级低，因为 5 个配置的 ceiling 一致性已是更强的统计证据。
2. **C1-s2 (4 probes 16-D) 未跑**：理论上更高维 sensor 升级可能在 < 0.30 verdict 之外突破，但 s0→s1 几乎同 success（差 −2 pp）已强烈暗示边界不在 sensor 维度；s2 backlog 优先级低。
3. **`target_speed=2.0` 未测**：当前 task 物理上限假设是 "u10 upstream 流速 + 1.5 m/s 目标速度" 的组合；速度提升是否解锁 deployability 未验证。
4. **主报告不集成（standalone status 已升级）**：本 follow-up 与同源 [`broad_validation_report`](rebrac_broad_validation_report.md) 一并保持 standalone exploratory side study，不回写主 `rebrac_experiment_report.md`。主报告 §10A 已**退回**为 pointer（不再保留 cf5cfff 的 sensor-floor framing 或 verdict gate）。Retrofit trigger condition：BC penalty 强度 sweep on C1（β1 ∈ {0, 1, 2, 4, 8}）闭环、task-fundamental floor 的 mechanism discriminator 给出确定结论之后，再考虑统一回写。在此之前 paper §experiments / §discussion 引用本 follow-up 仅以 cross-link 形式。
