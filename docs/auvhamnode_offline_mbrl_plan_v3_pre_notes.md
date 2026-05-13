# AUVHamNODE Offline MBRL — v3.0 Pre-Notes(交接备忘)

> **⚠ PAUSED 2026-05-13** — 本备忘的 α 路径(spike + audit → v3.0 plan)已完成 Step 0-4 审计,但因用户决定暂停本线,**v3.0 plan 未写、Path 1B spike-lite 未执行**。详情、累计决策、resume 起点见 [`docs/auvhamnode_mbrl_line_pause_memo.md`](auvhamnode_mbrl_line_pause_memo.md)。
>
> 本文继续作为 resume 时的次要 anchor:§3 的 4 项硬接口差异、§7 的下一步动作清单、§9 的 open questions 在恢复时仍是有效起点;但**必须先读 pause memo §4 与 §5** 才能避免重做。

**版本**:v3 pre-notes
**日期**:2026-05-13
**状态**:**PAUSED 2026-05-13**(原状态:战略转折已确定 + 接口差异已发现 + 用户选 α 推进路径;v3.0 plan 待写)
**前序**:[`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) v2.1(commits `4367474` + `e95dea5`)
**用途**:在 /compact 后,本备忘是 v3.0 plan 工作的自包含交接;不依赖任何对话历史

---

## 0. 一句话现状

> ReBRAC 主线与 broad_validation 全部搁置,paper 也不写;主线转 MBRL,使用 arrival_v2 reward + phnode_full_oc_clean(已入仓)的 AUVHamNODE checkpoint。但 checkpoint 与 PlanarRemusEnv 之间存在 4 项硬接口差异,需要先做 adapter spike + 流场分布审计验证可行性,再决定 v3.0 plan 怎么写。

---

## 1. 战略转折决定(2026-05-12)

| 项 | 旧状态 | 新状态 |
|---|---|---|
| ReBRAC mainline 实验 | rev.8 paper-readiness 4/4 闭环 | **暂停**——已有多份文档记录,不依赖 paper |
| ReBRAC paper drafting | rev.3,method/experiment section 编写中 | **搁置**——"一时兴起整理的" |
| ReBRAC broad_validation(8 spoke × 5 seed) | 部分闭环(B1 ✓,C1 collapse 现象) | **暂停**——不再扩展 |
| efficiency_v2 reward | 主线 default | **弃用**(upstream 任务 reward hacking;见 [`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) §3.1) |
| arrival_v2 reward | redesign 文档 SHELVED | **激活**为新主线 reward |
| 主线工作方向 | offline RL(ReBRAC 主轴) | **MBRL with frozen AUVHamNODE**(本 plan 主轴) |

---

## 2. 已 committed 的 v2.1 plan 状态

[`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) 在 branch `codex-arrival-v2-prototype` 上:

- `4367474` docs(auvhamnode): introduce offline MBRL plan v2.1 (cross-domain transfer)
- `e95dea5` docs(auvhamnode): v2.1 amendment — add §11.1.0 zero-th prerequisite

**v2.1 中仍然有效、v3.0 可继承的内容**:
- §1–§3 问题陈述、5 条核心设计原则、方案速览框架
- §4.2 测试 A–G 的方法论(MSE 分桶、σ_a 扰动曲线、Q overestimation hard gate 等)
- §5.2 augmentation pipeline 步骤、done flag 协议、wake time-index 协议
- §5.5 paired bootstrap 95% CI 统计检验框架
- §6 Phase 2A inference-time verifier 设计
- §8 横切关注点(悲观主义强度、部署一致性表)

**v2.1 中已失效、v3.0 必须重写的内容**:
- §11.1 fire-condition 路径 A/B/C —— **整章作废**(ReBRAC paper + broad_val 全部搁置)
- §11.1.0 零号前置 —— AUVHamNODE checkpoint 部分已解决(phnode_full_oc_clean 入仓);plain-MLP-ensemble 仍空白
- §8.4 "Phase 1-2 沿用 efficiency_v2" —— 改为 arrival_v2
- §5.3 hint "接口对齐 `Remus100.dynamics`" —— 失实(见 §3 接口差异)
- §1.1 资源表中"plain-MLP / 其他 dynamics checkpoints"行 —— 上游 export 是否包括待确认

---

## 3. ⚠️ phnode_full_oc_clean 的 4 项硬接口差异(本次会话首次发现,未落任何其他文件)

来源:[`phnode_full_oc_clean/README.md`](../phnode_full_oc_clean/README.md) + [`phnode_full_oc_clean/checkpoints/seed45/provenance.json`](../phnode_full_oc_clean/checkpoints/seed45/provenance.json) + [`phnode_full_oc_clean/phnode_full_oc/state_layout.py`](../phnode_full_oc_clean/phnode_full_oc/state_layout.py)

### 3.1 状态空间维度不匹配

```
AUVHamNODE state(27-D, SE(3) 6-DOF):
  [x(3) | R(9) | nu_r(6) | u_actual(3) | u_cmd(3) | v_c_n(3)]
  - x: inertial-frame position(meters)
  - R: rotation matrix body->inertial, row-major flattened
  - nu_r: body-frame velocity *relative to water* [v_r(3), omega(3)]
  - u_actual: actuator state with first-order lag, 3-vector
  - u_cmd: commanded actuator input held constant per block
  - v_c_n: inertial-frame ocean current(m/s)

PlanarRemusEnv state(planar 3-DOF):
  base 8 obs = [surge u, sway v, yaw rate r, cos(ψ), sin(ψ),
                goal body-frame x, y, distance-to-goal]
  + n_probes × 2 probe channels(单点流场采样)
```

**含义**:planar ↔ SE(3) 需要 lift/project 适配。z 维、roll、pitch、depth velocity 在 planar abstraction 中固定/缺失。

### 3.2 动作层级不匹配

```
AUVHamNODE u_cmd: [rudder_rad, stern_rad, propeller_RPM] — 低层执行器
PlanarRemusEnv action: [heading_cmd, speed_cmd]         — 高层自驾仪

PlanarRemusEnv.step() 内部经 PID(autopilot.py)把
[heading_cmd, speed_cmd] → [rudder_rad, stern_rad, propeller_RPM]
```

**含义**:augmentation 的 a_t' = a_t^data + ε,ε 加在高层还是低层?两种选择 fundamentally 不同,v2.1 §4.2 B σ_a 扰动曲线测试未考虑。

### 3.3 控制时间步不匹配

```
AUVHamNODE 训练:    dt_ctrl = 0.2 s(per block, u_cmd held constant)
PlanarRemusEnv:    control_dt = 0.5 s
```

**含义**:1-step augmentation 的 "1 step" 是 0.2s 还是 0.5s?0.5s 意味着 NODE 推理 2.5 个 training 时间步,超过训练 horizon。

### 3.4 流场输入语义不匹配

```
AUVHamNODE 训练:
  v_c_n(inertial-frame ocean current)作为"known exogenous input"
  每个 control block 内 held constant
  current_speed_range: [0.0, 0.5] m/s
  current_direction_range: [-π, π]
  current_vertical_std: 0.05
  current_drift_sigma: 0.02
  noise_current_std: 0.05

PlanarRemusEnv:
  wake field W(t, x, y, c)空间变化场
  每步在 body pose 处 query
  典型 dataset 用 u=1.0 m/s + 涡核结构(wake_v8_U1p00_Re150)
  远超训练 0.5 m/s 上限
```

**含义**:这是 **2–3× 速度区间外推 + 空间结构外推**,比 v2.1 §4.2 A 警告的"温和 cross-domain transfer"严重得多。v2.1 §4.2 A 阈值(整体 normalized MSE < 0.1)在这种 distribution shift 下大概率不可达。

### 3.5 其他硬限制(provenance.json limitations 字段)

- "Training horizon was block-iid(0.2 s blocks); long-horizon rollouts drift; benchmarked up to 60 s"
  → wake nav episode 长 240s 远超 60s benchmark;1-step only 约束被加强(理由更充分)
- "noise_profile=clean only; no noisy IC robustness expected"
  → AUVHamNODE 没见过 noisy IC;σ_a 扰动加在 IC 上需谨慎
- "u_dim=3: REMUS 100 4-DOF dynamics(roll passive)"
  → 与 PlanarRemusEnv 的 planar 3-DOF 在 roll 维度上有差异

---

## 4. arrival_v2 reward 实施状态(已 in-tree)

[`auv_nav/reward.py`](../auv_nav/reward.py):
- Line 34: `ARRIVAL_V2_OBJECTIVES = frozenset({"arrival_v2", "arrival_v2_fast"})`
- Line 116: `arrival_v2_simple` preset(offline-only,使用现有 RewardModelConfig 字段)
- Line 139: `arrival_v2` preset(完整 v6 spec)
- Line 167: `arrival_v2_fast` preset(可选 ablation,带 fast-success bonus)
- Line 278: dispatch to `_compute_arrival_v2()`
- Line 321: `_compute_arrival_v2` 方法实现

**已有 arrival_v2 prototype 实验数据**(session summary 提及):
- commit `1adee4e`:doc split + tandem/sbs notebook scaffold
- 已取回:`notebooks/sac_arrival_v2_tandem_sbs_validation_completed.ipynb` + experiments/arrival_v2_prototype/
- 已取回:`notebooks/sac_arrival_v2_single_u15_seed42_1M_validation_completed.ipynb`
- 报告:[`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md)

**v3.0 不需要做 reward 工程**,只需要 confirm arrival_v2 在 unit test(reward_redesign §7 spec)上通过,即可用于 MBRL baseline。

---

## 5. plain-MLP-ensemble checkpoint 现状

[`phnode_full_oc_clean/README.md`](../phnode_full_oc_clean/README.md) 与 [`provenance.json`](../phnode_full_oc_clean/checkpoints/seed45/provenance.json) **只提到 phnode_full**,未见 plain-MLP-ensemble checkpoint。

**待用户确认**:上游 AUVHamNODE 工作的 export 里是否包括 plain-MLP-ensemble?

**两种情况**:
- (a) 包括 → §10.1 中等档 "physics structure enables transfer" claim 保留
- (b) 不包括 → §10.1 中等档 claim 自动消失,paper 退到最低档 "frozen physics-structured NODE prior helps cross-domain transfer";少一个对照变量,少一个 reviewer attack 维度

不阻塞当前 α 路径推进。

---

## 6. 用户选定的推进路径:α(谨慎)

**含义**:先做 adapter spike + 流场分布审计 → 拿到 distribution shift 实测数字 → 再决定 v3.0 plan 怎么写。

**不选 β(直接写 v3.0)的理由**:接口差异 §3.4 暴露的流场速度外推幅度可能让 §4.2 A 直接 no-go,此时 v3.0 plan 是空中楼阁。

---

## 7. /compact 之后的具体 next actions(自包含,按顺序执行)

### 动作 1:Adapter spike(预计 1–2 天)

**目的**:验证 planar_state ↔ SE(3) state 的 lift/project 接口可工作,不进 RL agent。

**实施**:
1. 阅读 [`phnode_full_oc_clean/examples/01_single_step.py`](../phnode_full_oc_clean/examples/01_single_step.py) 与 [`02_rollout.py`](../phnode_full_oc_clean/examples/02_rollout.py),理解 phnode_full API
2. 新建 `scripts/spike_auvhamnode_planar_adapter.py`(spike 脚本,不进 `auv_nav/`):
   - lift:从 PlanarRemusEnv state(surge u, sway v, yaw rate r, ψ, position x/y)+ 假设 z=const、roll=pitch=0、depth_vel=0,组装 27-D SE(3) state
   - project:从 27-D next_state 投影回 planar state
   - PID:复用 [`auv_nav/autopilot.py`](../auv_nav/autopilot.py) 把 [heading_cmd, speed_cmd] → [rudder, stern, RPM]
   - 1-step 推理:`single_step(model, state_27d, dt=0.2, method='rk4')`
3. sanity check:在一个 dataset transition 上跑,对比 NODE 预测 next planar state 与 dataset s_{t+1} 的 MSE

**判断**:
- MSE 在合理范围(< 1 m position error / < 0.5 m/s velocity error)→ 接口可行,继续动作 2
- MSE 严重偏大 → adapter 假设(z=const、roll=0 等)与 dataset 不兼容,需要重新设计 adapter 或重新生成兼容 dataset

### 动作 2:流场分布审计(预计 0.5 天)

**目的**:量化 AUVHamNODE 训练流场 vs wake_data 流场的 distribution shift,在 Phase 0 A MSE 测试启动前知道预期 shift 幅度。

**实施**:
1. 提取 AUVHamNODE 训练流场统计:
   - `provenance.json` 中的 `current_speed_range: [0.0, 0.5]`、`current_vertical_std: 0.05` 等
2. 提取 wake_data 流场统计(对 [`wake_data/`](../wake_data/) 中的每个 npy 跑):
   - speed magnitude 分布(min/median/p95/max)
   - spatial gradient 分布(涡核检测)
3. 报告:speed shift ratio(wake_max / AUVHamNODE_max)+ spatial structure 差异(AUVHamNODE 是 spatially uniform,wake 是 spatially varying)
4. 写入 `experiments/auvhamnode_spike/flow_distribution_audit.md`

**判断**:
- speed ratio < 2× → §4.2 A 可能可行,继续
- speed ratio > 3× → §4.2 A 大概率 no-go,需要先讨论"是否限制 wake_data 速度上限 / 是否需要 finetune AUVHamNODE"

### 动作 3:决定 v3.0 plan 写法(动作 1+2 之后)

基于动作 1+2 的实测数字,决定:

- (a) 接口可行 + distribution shift 可接受 → 直接写 v3.0,继承 v2.1 §1–§8 框架,新增 Phase 0 H 测试(adapter validation 已通过,纳入 v3.0 phase 0)
- (b) 接口可行 + distribution shift 严重 → 写 v3.0,在 §4.2 A 重新校准 MSE 阈值,接受 paper claim 退档;或考虑限制 wake_data 速度
- (c) 接口不可行 → 不写 v3.0,先解决 adapter 问题,本 pre-notes 留作记录

---

## 8. /compact 后立即执行的检查清单

- [ ] 读本备忘 §3 4 项接口差异 + §6 α 选择 + §7 动作 1
- [ ] 确认 [`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) v2.1 已 committed(`4367474` + `e95dea5`)
- [ ] 确认 [`phnode_full_oc_clean/`](../phnode_full_oc_clean/) 入仓状态(可能还是 untracked,需要 user 决定是否纳入 git;currently in `.gitignore`?待 user 确认)
- [ ] 启动动作 1:Adapter spike

---

## 9. Open questions for user

1. **plain-MLP-ensemble checkpoint** 上游是否会一起 export?(影响 §10.1 paper claim 中等档)
2. **phnode_full_oc_clean/ 是否要 commit 到 git**?(checkpoints 较大,可能想 git-ignore;但 wrapper + examples 应该入库)
3. **adapter spike 在 conda mytorch1 环境是否需要安装 `torchdiffeq`**?([`phnode_full_oc_clean/requirements.txt`](../phnode_full_oc_clean/requirements.txt) 列了 torch + torchdiffeq)
4. **wake_data 速度限制的接受度**:如果动作 2 显示 speed ratio > 3×,接受"限制 wake_data 到 0.5 m/s 上限"作为 v3.0 的 framing 调整,还是接受"AUVHamNODE 在涡核高速区精度差"作为 §10.1 paper claim 中等档退档?

---

## 10. 不在本备忘内但相关的文档索引

- [`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) — v2.1 完整 plan(基础)
- [`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md) — arrival_v2 reward v6 spec(已 SHELVED 但可参考)
- [`docs/arrival_v2_experiment_report.md`](arrival_v2_experiment_report.md) — arrival_v2 prototype 实验报告
- [`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) — offline RL 主线 timeline overview
- [`phnode_full_oc_clean/README.md`](../phnode_full_oc_clean/README.md) — AUVHamNODE checkpoint 文档
- [`auv_nav/reward.py`](../auv_nav/reward.py) — arrival_v2 reward 实现
- [`auv_nav/env.py`](../auv_nav/env.py) — PlanarRemusEnv(planar 3-DOF)
- [`auv_nav/autopilot.py`](../auv_nav/autopilot.py) — PID 自驾仪(heading/speed → rudder/stern/RPM)
- [`auv_nav/vehicle.py`](../auv_nav/vehicle.py) — Remus100 6-DOF 真值动力学(可作为 adapter 验证的 oracle)

---

*文档版本:v3 pre-notes(2026-05-13,/compact 交接备忘,自包含)*
*下次更新:动作 1 adapter spike 完成后,把 MSE 实测数字与判断结论补入 §7;启动 v3.0 plan 时本备忘归档到 [`docs/archive/`](archive/)。*
