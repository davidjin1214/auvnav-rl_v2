# AUVHamNODE + Offline MBRL 线 — 暂停归档备忘

**日期:** 2026-05-13
**触发:** 用户决定 — *"考虑到 auvhamnode 短时间难以接入,我希望暂停这条线,即暂时不推进 mbrl + auvhamnode 这条线了"*
**状态:** **PAUSED**(不是 deprecated,不是 cancelled;条件满足后可恢复)
**前序文档:** [`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) §3.4 / §4.4(timeline)

---

## 0. 这份文档的用途

- 给"未来回到这条线的我"一个**自包含的 resume manual**:不必读对话历史就能理解为什么暂停、做过什么、留下什么、恢复条件是什么。
- 给"现在正在看 repo 的我"一个**单一权威 pause anchor**:所有相关文档都指向这里,避免每份文档自己一套"为什么 paused"的说法。

不在这份备忘里的 anti-goal:重新论证 AUVHamNODE 是否有价值,或者重写 plan v2.1。这些在原文档里已经做过了。

---

## 1. 一句话现状

> **AUVHamNODE checkpoint 的 6/7 个条件轴对齐良好,但唯一不对齐的那一个(海流 v_c_n 幅度)是 2-4× 训练分布外。Step 0-4 的廉价审计已经把"是否值得做完整 spike"这个问题答到了 80%——继续推进的下一步(Path 1B spike-lite,~5h)在用户当前精力下不愿启动。本线暂停。**

---

## 2. 暂停理由(用户原话 + 我的解读)

**用户原话:**
> 考虑到 auvhamnode 短时间难以接入,我希望暂停这条线。

**我的解读(可能不准确,以用户判断为准):**
1. AUVHamNODE 接入需要解决至少 4 个硬接口差异 + 1 个分布外推问题(详见 [`auvhamnode_offline_mbrl_plan_v3_pre_notes.md`](auvhamnode_offline_mbrl_plan_v3_pre_notes.md) §3 与 §4.2)
2. 即使走最便宜的 Path 1B(spike-lite, ~5h),也只是验证"U=1.0 wake 是否可行",并非端到端 paper-grade 结果
3. ReBRAC 主线已收口(rev.8 paper-ready 4/4 ✅),没有 burning need 立即转 MBRL
4. arrival_v2 reward 重新设计已落地,可以独立支撑 online SAC 或后续 MBRL 任何路线
5. 项目还有其他更确定能给 paper 加分的方向(具体哪些由用户判断)

---

## 3. 完整产物清单(已落 git)

### 3.1 计划文档(`docs/`)

| 文档 | 行数 | 描述 | 状态 |
|---|---:|---|---|
| [`auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) | 640 | v2.1 + amendment(cross-domain transfer framing;Phase 0-2A 5-6 周实施期) | PAUSED;banner 已加 |
| [`auvhamnode_offline_mbrl_plan_v3_pre_notes.md`](auvhamnode_offline_mbrl_plan_v3_pre_notes.md) | 252 | α 路径交接备忘(发现 4 项硬接口差异;ReBRAC 战略下调) | PAUSED;banner 已加 |
| 本备忘 `auvhamnode_mbrl_line_pause_memo.md` | (本文) | pause anchor + resume manual | active |

### 3.2 预 spike 工作区(`docs/auvhamnode_spike/`)

| 文件 | 描述 | 状态 |
|---|---|---|
| [`README.md`](auvhamnode_spike/README.md) | 工作区 index + 一页 TL;DR | banner 已加 |
| [`00_smoke_test_log.md`](auvhamnode_spike/00_smoke_test_log.md) | Step 0 ✅:`mytorch1` env + torchdiffeq + checkpoint + ODE solver 全绿 | complete |
| [`01_static_distribution_audit.md`](auvhamnode_spike/01_static_distribution_audit.md) | Step 1 ✅:7 维分布审计 + 4 条路径决策矩阵 | complete |
| [`02_spike_lite_design.md`](auvhamnode_spike/02_spike_lite_design.md) | Path 1B 蓝图(~5h kill-test,**未执行**) | designed only |
| [`03_dynamics_consistency_audit.md`](auvhamnode_spike/03_dynamics_consistency_audit.md) | Step 3 ✅:`vehicle.py` 与 `remus100_core.py` 7 个公式分歧 | complete |
| [`04_swap_vehicle_decision_memo.md`](auvhamnode_spike/04_swap_vehicle_decision_memo.md) | Step 4 ✅:是否要把 `vehicle.py` 换成 `remus100_core.py`?结论=不要 | complete |
| [`_wake_stats.py`](auvhamnode_spike/_wake_stats.py) | wake_data 速度统计脚本(可复用) | reusable |
| [`_wake_stats.out`](auvhamnode_spike/_wake_stats.out) | 3/6 文件原始统计(剩余 3 个 OneDrive 卡住) | partial |

### 3.3 Worktree 中未入 git 的资产

| 路径 | 内容 | 处理 |
|---|---|---|
| `phnode_full_oc_clean/` | AUVHamNODE seed 45 checkpoint + 推理代码 + reference simulator(`remus100_core.py`)+ examples + tests | `.gitignore`(此次归档加入);暂时留在 worktree |

---

## 4. 已完成的实质决策(Step 0-4 累计产出)

下列结论不会因为暂停而失效,后续(无论本线恢复或转其他方向)都可以直接引用:

### 4.1 环境层

- ✅ `mytorch1` conda env 已装 torch 2.10 + torchdiffeq,绝对路径 `/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin/python` 可直接驱动 checkpoint。
- ✅ `phnode_full_oc_clean/examples/01_single_step.py` 与 `02_rollout.py` 跑通,物理数值合理(actuator lag、surge 加速、横向偏移都对)。

### 4.2 接口层(4 项硬差异,在 pre-notes §3 锁定)

| 差异 | 训练侧 | 部署侧(PlanarRemusEnv) | 影响 |
|---|---|---|---|
| 状态维度 | 27-D SE(3) 6-DOF | planar 3-DOF(8 + n_probes×2) | 需 lift/project 适配 |
| 动作层级 | 低层 [rudder, stern, RPM] | 高层 [heading, speed] | 需 PID 桥接 |
| 控制时间步 | dt_ctrl=0.2s | control_dt=0.5s | augmentation 必须分 2-3 个 NODE block,**不**改 dt |
| 流场语义 | inertial `v_c_n`,空间均匀,≤0.5 m/s | 空间异质 wake field,U_ref=1.0–1.5 m/s | **2-4× OOD** ← 唯一硬障碍 |

### 4.3 分布层(Step 1 §2 实测,基于 3/6 wake 文件)

| Wake 配置 | 速度 median (m/s) | speed_p99 (m/s) | shift vs 训练 cap 0.5 m/s |
|---|---:|---:|---:|
| `v8_U1p00_Re150` (single) | 1.03 | 1.31 | **2.62×** (p99) |
| `v8_U1p50_Re250` (single) | 1.58 | 2.09 | **4.18×** (p99) |
| `tandem_G35_U1p00` | 1.05 | 1.29 | 2.57× (p99) |

**结论:U=1.0 边界(2× shift,marginal);U=1.5 catastrophic(4× shift)。**

### 4.4 物理一致性层(Step 3 / Step 4)

- `auv_nav/vehicle.py` 与 `phnode_full_oc_clean/reference_simulator/remus100_core.py` **不一致**——7 个公式分歧,Top 4 严重度:
  1. cross-flow drag Cd_2D 常数 ~1.20 vs Re/AR 函数 0.25–0.80(ratio 1.5–4.7×)
  2. actuator τ_fin 0.10s vs 0.25s(2.5× 慢)
  3. actuator 积分器 joint RK4 vs Euler+rate-limit
  4. `geometry_scale=1.0096` → mass +3%, Ix/Iy +5%
- **是否把 `vehicle.py` 换成 `remus100_core.py`?** 否。Step 4 论证:换 fixes 30%(公式),余 70%(wake current OOD + 空间异质 + 控制周期 + 帧表示)无解。
- arrival_v2 reward 在 [`auv_nav/reward.py`](../auv_nav/reward.py) 已实施(L34、L116、L139、L167、L278、L321),与本线无耦合,可独立支撑后续 MBRL/SAC 工作。

### 4.5 4 条候选路径(Step 1 §5 决策矩阵)

| 路径 | 一句话 | 这次的状态 |
|---|---|---|
| **Path 1** | 限定 v3.0 到 U=1.0 wakes,recommended first | 未执行(spike 未开) |
| Path 1B | spike-lite ~5h kill-test 验 Path 1 | 蓝图已就位([`02_spike_lite_design.md`](auvhamnode_spike/02_spike_lite_design.md)),未执行 |
| Path 2 | finetune NODE 到 wake-compatible flow | 未评估(需上游训练 repo 访问) |
| Path 3 | rescale wake ≤0.5 m/s | 论证为劣选(破坏 Re/St 相似性) |
| Path 4 | 弃 NODE,用 `vehicle.py` oracle 当 dynamics aug source | 提及但未设计 |

---

## 5. 哪些工作**没**做(留作未来债务表)

直接列出来,未来回到这条线时不需要重新猜测:

1. **Spike-lite (Path 1B) 未执行** — [`02_spike_lite_design.md`](auvhamnode_spike/02_spike_lite_design.md) 给出了完整蓝图(lift/project hand-rolled、5h 预算、go/no-go 阈值),实施前不需要再设计。
2. **完整 adapter spike 未启动** — pre-notes §7 动作 1 的 1-2 天版本。
3. **3/6 wake 文件统计缺失** — `tandem_U1.5`, `sbs_U1.0`, `sbs_U1.5` 因 OneDrive 按需同步超时未跑通;trend 可推断但未实测。在 Colab 上 wake_data fully cached 时可秒级补全。
4. **Path 2 (finetune) 可行性未确认** — 需要回到上游 repo (`g3_5_5` / `g3_5_7`) 看训练 pipeline + dataset 生成代码是否可访问。
5. **Path 4 (vehicle.py oracle) 未细化** — 替代 framing 的 plan 没写。
6. **AUVHamNODE 训练数据集(`auv_oc_traj1000_blk150_s23`)未取回** — 仅有 checkpoint + provenance,没有原始 transitions。用于 spike Check A(in-training-distribution 对照)的对照数据需要从上游 export。
7. **`plain-MLP-ensemble` checkpoint 是否在上游 export 中** — pre-notes §9 question 1 未答复;影响 paper claim 中等档存活。

---

## 6. 恢复条件(任意 N 项为真即可考虑 unpause)

以下条件不是 AND,是 OR——任何一条成立都可以触发"是否要 unpause"的重新评估。

| 条件 | 监测信号 | 备注 |
|---|---|---|
| (a) AUVHamNODE 上游训练 pipeline + dataset 生成代码可访问 | repo `g3_5_5` / `g3_5_7` clone | 解锁 Path 2 finetune 和 Path 1B 的 Check A 对照 |
| (b) 出现 wake-compatible (≤0.5 m/s current 上限) 的新 wake_data 生成需求 | CFD pipeline 重跑 | 解锁 Path 3 |
| (c) 项目转向其他低速 AUV 任务 | thesis scope 变更 | AUVHamNODE 训练域立即变 in-distribution |
| (d) 出现替代 physics-structured dynamics prior 且训练流场含强 wake | 文献监测 | 同效但不绑 AUVHamNODE |
| (e) 用户主动想恢复 | (任何时候) | 默认路径 |
| (f) Path 1B spike-lite 在 5h 内可以执行的时间窗口出现 | (用户判断) | 走 [`02_spike_lite_design.md`](auvhamnode_spike/02_spike_lite_design.md) |

---

## 7. 恢复时的 first step(顺序明确)

如果 (a)/(b)/(c)/(d)/(e)/(f) 任一触发,**第一步不需要重读 plan v2.1 全文**,按以下顺序:

1. 重读本备忘 §4(累计决策)和 §5(未做的事),确认初始状态没变
2. 重跑 [`docs/auvhamnode_spike/_wake_stats.py`](auvhamnode_spike/_wake_stats.py) 把 3 个缺失行补上(验证 wake_data 还在 + OneDrive 通)
3. 视触发条件选下一动作:
   - (a) → 启动 Path 2 finetune,本备忘暂停期间 plan v2.1 不动,新写 plan v3.0
   - (b)/(c) → 启动 Path 1B spike-lite([`02_spike_lite_design.md`](auvhamnode_spike/02_spike_lite_design.md))
   - (d) → 重新评估 paper framing,plan v2.1 重写
   - (e)/(f) → 直接走 Path 1B

---

## 8. 与项目其他线的关系

- **ReBRAC 主线**:无依赖。本备忘的暂停**不**影响 ReBRAC paper revision、broad validation、c1_s1 follow-up。
- **arrival_v2 reward**:在 [`auv_nav/reward.py`](../auv_nav/reward.py) 已 in-tree,与本备忘解耦。后续任何 SAC/MBRL 工作都可以使用。
- **efficiency_v2 reward hacking 调查**:见 [`docs/online_sac_reward_redesign.md`](online_sac_reward_redesign.md);本备忘的暂停**不**改变该文档的 SHELVED 状态。
- **Online SAC 线**:见 [`docs/online_rl_line_summary.md`](online_rl_line_summary.md);本备忘的暂停**不**改变 online 线状态。
- **`auv_nav/vehicle.py`**:Step 4 决策"不换",此判断在本备忘暂停后仍然成立。

---

## 9. 同步更新清单(本次归档时同时改的文件)

- [x] [`docs/auvhamnode_offline_mbrl_plan.md`](auvhamnode_offline_mbrl_plan.md) — 顶部加 PAUSED banner
- [x] [`docs/auvhamnode_offline_mbrl_plan_v3_pre_notes.md`](auvhamnode_offline_mbrl_plan_v3_pre_notes.md) — 顶部加 PAUSED banner
- [x] [`docs/auvhamnode_spike/README.md`](auvhamnode_spike/README.md) — 顶部加 PAUSED banner
- [x] [`docs/offline_rl_line_summary.md`](offline_rl_line_summary.md) §3.4 + §4.4 — 状态从"v2.0 locked, 待 fire"改为"paused 2026-05-13, 见本备忘"
- [x] [`docs/offline_rl_implementation_plan.md`](offline_rl_implementation_plan.md) DEPRECATED banner — 追加 "AUVHamNODE 后续也已 paused" 的说明
- [x] `.gitignore` — 加入 `phnode_full_oc_clean/`(formalize "暂时不入 git")
- [x] **[2026-07-28 指针体检补收]** [`docs/offline_mbrl_plan/`](offline_mbrl_plan/) 全部 6 份草案 — 归档时漏收：其 DEPRECATED 横幅当时写"**唯一活跃入口**是 v2.0 plan",而 v2.0 plan 五天后即 paused,该句遂全部失真。已逐份追注"该接替者已整线 PAUSED、本线现无活跃入口"

---

*文档版本:1.0(2026-05-13)。维护策略:本备忘是 read-mostly 的 anchor;只有 §6 触发条件或 §7 first-step 顺序变更时才修订。*
