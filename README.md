# AUV Navigation in Complex Flow Fields

一个面向研究的强化学习项目：让 REMUS-100 风格的 AUV 在复杂、非均匀、随时间变化的尾迹流场中学会高效导航，而不是只会“顶流直冲”。

> **一句话定位**：研究 AUV 能否在**部署受限的局部感知**下，主动利用尾迹涡结构与局部流动信息，在强逆流或复杂干涉流场中更高效地到达目标。核心命题——**用好已有的信息与数据，而非为系统增添能力**。

> 本 README 是**人类视角的入口**：讲清楚“这个仓库在干什么、闭环长什么样、第一次怎么上手”。**给 agent 的操作手册、完整命令参考、完整模块/脚本清单、权威实验数字都在 [`CLAUDE.md`](CLAUDE.md) 与各线总览文档里**——本文不复述，只指路。

## 当前研究状态（先读这一节）

项目有两条线，**离线 RL 是主线（驱动论文），在线 RL 自 2026-05-06 起降级为支撑角色**：

| 线 | 角色 | 一句话状态 | 入口 |
|---|---|---|---|
| **离线 RL（主线）** | paper-driving | 各阶段已全部收口（TD3+BC → ReBRAC-Q 主线 → broad-val v2 → FQL succession → AUVHamNODE ⏸） | [`docs/offline_rl_line_summary.md`](docs/offline_rl_line_summary.md) |
| **在线 RL（支撑）** | 环境 sanity + 数据采集 | thesis-grade SAC 矩阵已取消；保留 `A0 sensor screen` 与 **SAC collector**（为离线线采数据） | [`docs/online_rl_line_summary.md`](docs/online_rl_line_summary.md) |

**当前主要工作 = 博士论文第 5 章写作**（写作出口自 2026-06-02 起 LOCKED 为单一章节，standalone paper 均撤销）。项目总纲与写作入口见 [`CLAUDE.md`](CLAUDE.md)。

> **数字以 ground-truth 文档为准，勿凭记忆**：ReBRAC-Q → [`docs/rebrac_experiment_report.md`](docs/rebrac_experiment_report.md)（rev.8）；TD3+BC → [`docs/td3bc_phase0c_experiment_report.md`](docs/td3bc_phase0c_experiment_report.md)；FQL → [`docs/fql_succession_p2_results.md`](docs/fql_succession_p2_results.md)；在线 SAC 早期报告已归档至 [`results/archived/rl_navigation_experiment_report.md`](results/archived/rl_navigation_experiment_report.md)。

## 项目在做什么

- 载体：以 REMUS-100 风格 AUV 为原型的平面导航任务
- 环境：由二维 TRT-LBM 生成的单圆柱 / 串联或并排双圆柱尾迹流场
- 目标：在动态流场中从随机起点导航到目标点
- 难点：流场强、非均匀、时变，且存在欠驱动（临界）场景

这不是“静态路径规划”仓库，而更接近一个**闭环控制与决策系统**：上游离线生成流场 → 中游把流场接入 AUV 物理环境 → 下游用 RL 学习控制策略 → 最后评估、可视化、论文作图。

## 仓库的整体闭环

```text
generate_wake.py
    ↓
wake_data/*.npy + *_meta.json
    ↓
PlanarRemusEnv + WakeField + FlowSampler
    ↓
SAC / baseline policies
    ↓                              ↓
checkpoint / logs / metrics        collect_offline_data.py → offline_data/
    ↓                                       ↓                        ↓
evaluate / demo / visualize     train_sac.py --offline-data    train_offline.py
                                   (RLPD: offline+online)       (pure offline: td3bc/rebrac/fql)
                                                                        ↓
                                                              evaluate_offline.py
```

理解顺序通常是：**生成流场 → 采集离线数据（离线线）或直接在线训练 → 训练 → 在固定 benchmark manifest 上评估 → 可视化/作图**。做严格算法对比时优先用 `benchmarks/*.json` 固定评估 episode，让不同算法在同一组 episode 上比较。

## 核心研究设定

### 状态、动作与任务

环境主体是 [`auv_nav/env.py`](auv_nav/env.py) 的 `PlanarRemusEnv`：动作 2 维（航向 + 速度指令），内部含 REMUS-100 6-DOF 动力学与自动驾驶仪；任务是在流场 ROI 内从随机起点导航到目标，带最大时长、边界与姿态/速度安全约束。

任务几何用 `--task-geometry {downstream, cross_stream, upstream}` 指定。

> **难度怎么参数化**：本研究的难度由 **benchmark key**（流速/几何/目标速度，如 `single_u15_upstream_tgt15`）决定，**不是**旧的 `--difficulty {easy,medium,hard}` 标志（后者仅作 legacy 别名保留）。新实验请用 `--task-geometry` + benchmark manifest。

### AUV 能看到什么（关键）

观测**不是全局流场图像，而是局部传感器式观测**：自身运动状态 + 目标相对信息 + 机体附近若干流速探针读数。3 种 probe layout 均对应真实 REMUS-100 传感器：

- `s0`：1 个中心探针（DVL），**部署现实基线**，obs 维度 10
- `s1`：2 探针（DVL + 短程前向 ADCP），obs 维度 12（参考上界）
- `s2`：4 探针（DVL + 长程/侧向 ADCP），obs 维度 16（参考上界）

环境另在 `info` 发出 **`privileged_obs`**（机体系整船积分等效流 `[u_eq, v_eq]`），仅供 `AsymmetricQNetwork` 的特权 critic 使用；actor 部署时只有 `s0` 单点采样。**本仓库的核心难点正是：在有限局部感知下做控制决策。**

### 奖励目标

奖励/成本定义在 [`auv_nav/reward.py`](auv_nav/reward.py)，用 `--objective` 显式指定（无隐含默认）：`efficiency_v2`（在线“高效航行”推荐默认）、`arrival_v2`（SAC collector / 离线线使用）、以及兼容旧实验的 `arrival_v1` / `efficiency_v1`。

## 为什么要先生成流场

仓库没有在线 CFD，流场先离线生成、训练时按时空插值读取。生成脚本 [`scripts/generate_wake.py`](scripts/generate_wake.py) 用二维 TRT-LBM 模拟圆柱绕流（单/串联/并排），产出 `wake_data/` 下的 `wake_*.npy`（`(T,Nx,Ny,3)`）+ 同名 `_meta.json`（训练/评估真正依赖这两个）+ `_phase.npy`（涡相位）。用法见 [`docs/generate_wake_usage.md`](docs/generate_wake_usage.md)。

## 为什么 `--target-speed 1.5` 重要

AUV 满转速名义最大速度默认设为 `2.0 m/s`。若来流速度接近这个量级，问题变得尖锐：例如流场 `U = 1.5 m/s` 且 AUV 限速 `1.5 m/s` 时，逆流任务里 AUV 处于“靠本体推进难以直接胜出”的**临界状态**——直线迎流策略很差，agent 必须学会利用局部流动结构，这才能体现 RL 是否真的“借力流场”。

> 想测试策略是否真具备复杂流场利用能力时，优先用 `U_flow = 1.5` + `target-speed = 1.5` 的逆流任务（benchmark `single_u15_upstream_tgt15`）。

## 快速上手

两条线各一个最小示例。**完整命令参考（含 RLPD、非对称 critic、resume、各类评估与 sweep）见 [`CLAUDE.md`](CLAUDE.md) 的 Common Commands。**

### 离线主线（td3bc / rebrac / fql）

```bash
# 1. 采集离线数据
python -m scripts.collect_offline_data --policy worldcomp \
    --flow wake_data/<wake>.npy --probe-layout s0 \
    --task-geometry cross_stream --target-speed 1.5 \
    --history-length 4 --objective arrival_v2 --episodes 1000 \
    --output-dir offline_data/<dataset>

# 2. 训练（--algo 选 td3bc / rebrac / fql）
python -m scripts.train_offline --algo rebrac \
    --offline-data offline_data/<dataset>/transitions.npz \
    --objective arrival_v2 --probe-layout s0 --history-length 4 \
    --manifest benchmarks/single_u10_cross_tgt15.json --device cuda

# 3. 评估
python -m scripts.evaluate_offline \
    --checkpoint checkpoints/offline/rebrac/<run_dir> \
    --manifest benchmarks/single_u10_cross_tgt15.json
```

离线数据策略：`goalseek` / `crosscomp`（无特权）、`worldcomp`（中度特权）、`privileged`（强特权）。

### 在线 SAC（A0 sensor screen / collector 口径）

```bash
python -m scripts.train_sac --total-steps 600000 --device cuda --seed 46 \
    --task-geometry upstream --target-speed 1.5 \
    --probe-layout s0 --history-length 4 --num-envs 6 \
    --eval-every 10000 --eval-manifest benchmarks/single_u15_upstream_tgt15.json \
    --eval-episodes 30 --objective efficiency_v2 \
    --save-dir experiments/<study>/<cell>/seed_46
```

加 `--offline-data <npz> --offline-ratio 0.5` 即进入 RLPD（离线+在线）模式；设计见 [`docs/rlpd_design.md`](docs/rlpd_design.md)。

## 仓库结构与代码导航

核心逻辑落点（**完整模块表与脚本表见 [`CLAUDE.md`](CLAUDE.md) 的 Architecture**）：

- 环境：[`auv_nav/env.py`](auv_nav/env.py)（`PlanarRemusEnv`）
- 流场接入：[`auv_nav/flow.py`](auv_nav/flow.py)
- 在线算法：[`auv_nav/sac.py`](auv_nav/sac.py)（含特权 `AsymmetricQNetwork`）
- 离线算法：`auv_nav/td3bc.py` / `auv_nav/rebrac.py` / `auv_nav/fql.py`
- 训练入口：[`scripts/train_sac.py`](scripts/train_sac.py)（在线 / RLPD）、[`scripts/train_offline.py`](scripts/train_offline.py)（纯离线）
- 数据/评估：`scripts/collect_offline_data.py`、`scripts/evaluate{,_offline}.py`、`scripts/generate_standard_benchmarks.py`

顶层目录：`auv_nav/`（核心库）、`scripts/`（实验入口）、`tests/`（pytest）、`docs/`（研究+实现文档）、`benchmarks/`（固定评估 manifest）、`wake_data/` / `offline_data/`（数据，gitignored）、`experiments/` / `checkpoints/`（训练产物）、`paper/`（论文与第 5 章 LaTeX 工程）。

> 训练产物里最关键的是 `trainer_state.json`：它保存恢复训练/评估所需配置（flow 路径、history length、probe layout、agent config）。`[skip]` resume 以 `agent_final.pt` 为键，Colab 重启后可续跑。

## 第一次上手怎么读

按目标选一条路径：

- **理解“项目在干什么”**：本文档 → [`docs/offline_rl_line_summary.md`](docs/offline_rl_line_summary.md) → [`auv_nav/env.py`](auv_nav/env.py) 的 `PlanarRemusEnvConfig` → 训练入口脚本
- **理解“流场怎么接进环境”**：[`scripts/generate_wake.py`](scripts/generate_wake.py) → [`auv_nav/flow.py`](auv_nav/flow.py) → [`auv_nav/env.py`](auv_nav/env.py)
- **理解“离线 RL 主线到了哪”**：[`docs/offline_rl_line_summary.md`](docs/offline_rl_line_summary.md) → [`docs/rebrac_experiment_report.md`](docs/rebrac_experiment_report.md) → [`scripts/train_offline.py`](scripts/train_offline.py)
- **理解“SAC 改进 / RLPD”**：[`docs/SAC_improvements_survey.md`](docs/SAC_improvements_survey.md) / [`docs/rlpd_design.md`](docs/rlpd_design.md) → [`auv_nav/sac.py`](auv_nav/sac.py) → [`auv_nav/replay.py`](auv_nav/replay.py)

## 文档地图

| 想看的东西 | 文档 |
|---|---|
| 项目总纲 / 完整命令 / 架构 / 计算环境 | [`CLAUDE.md`](CLAUDE.md) |
| 离线 RL 线总览（主线入口） | [`docs/offline_rl_line_summary.md`](docs/offline_rl_line_summary.md) |
| 在线 RL 线总览（支撑入口） | [`docs/online_rl_line_summary.md`](docs/online_rl_line_summary.md) |
| ReBRAC-Q 权威数字（rev.8） | [`docs/rebrac_experiment_report.md`](docs/rebrac_experiment_report.md) |
| TD3+BC 主线报告 | [`docs/td3bc_phase0c_experiment_report.md`](docs/td3bc_phase0c_experiment_report.md) |
| FQL succession（NEGATIVE 闭环） | [`docs/fql_succession_p2_results.md`](docs/fql_succession_p2_results.md) |
| 环境 / RLPD 设计 | [`docs/environment_design.md`](docs/environment_design.md) / [`docs/rlpd_design.md`](docs/rlpd_design.md) |
| 流场生成用法 | [`docs/generate_wake_usage.md`](docs/generate_wake_usage.md) |

## 测试

```bash
pytest tests/                       # 全量
pytest tests/test_offline_td3bc.py  # 指定模块
```

---

> **本 README 的边界**：只负责“是什么 / 为什么 / 第一次怎么上手”。操作手册、完整命令、完整模块与脚本清单、权威实验数字 → [`CLAUDE.md`](CLAUDE.md) 及上方文档地图，不在此重复。
