# AUV Navigation in Complex Flow Fields

一个面向研究的强化学习项目：让 AUV 在复杂、非均匀、随时间变化的尾迹流场中学会高效导航，而不是只会“顶流直冲”。

## 项目在做什么

这个仓库研究的问题可以概括为：

- 载体：以 REMUS-100 风格 AUV 为原型的平面导航任务
- 环境：由 LBM 生成的单圆柱或多圆柱尾迹流场
- 目标：在动态流场中从起点到目标点导航
- 难点：流场强、非均匀、时变，且存在欠驱动场景
- 方法：基于改进版 Soft Actor-Critic (SAC) 学习策略，支持 RLPD（RL with Prior Data）利用离线基线数据加速训练

这不是一个“静态路径规划”仓库。它更接近一个闭环控制与决策系统：

- 上游先离线生成流场数据
- 中游将流场接入 AUV 物理环境
- 下游用 RL 学习控制策略
- 最后通过评估、可视化和论文作图分析结果

如果只看一句话：

> 本项目试图回答：AUV 能否学会主动利用尾迹中的涡结构和局部流动信息，在强逆流或复杂干涉流场中更高效地到达目标。

如果你想直接看已经整理好的实验结论，而不是翻 `experiments/` 下的原始日志，见：

- [`results/rl_navigation_experiment_report.md`](results/rl_navigation_experiment_report.md)

## 当前状态

如果你只想先建立正确的全局认识，而不想立刻深入源码，可以先记住下面 6 点：

- 这是一个“流场感知 + AUV 动力学 + RL 控制”的研究仓库，不是静态路径规划仓库。
- 环境核心在 `auv_nav/env.py`，流场读取在 `auv_nav/flow.py`，算法核心在 `auv_nav/sac.py`。
- 训练入口是 `scripts/train_sac.py`，标准化批量实验入口是 `scripts/run_suite.py`。
- 当前主研究 benchmark 是 `single_u15_upstream_tgt15`，也就是 `U_flow = 1.5 m/s`、`target-speed = 1.5 m/s` 的单圆柱逆流任务。
- 当前推荐的主实验目标函数是 `efficiency_v2`，不是旧的 `efficiency_v1`。
- 已整理好的实验结论在 `results/rl_navigation_experiment_report.md`，建议先看它再决定下一轮实验。

## 推荐阅读顺序

如果你第一次进入这个仓库，建议按下面顺序理解：

1. 先读本文档，建立整体闭环和当前推荐实验路线。
2. 再看 [`results/rl_navigation_experiment_report.md`](results/rl_navigation_experiment_report.md)，了解目前已经得到的结论。
3. 然后读 [`docs/environment_design.md`](docs/environment_design.md)，理解流场与任务设计。
4. 最后进入源码：先看 [`auv_nav/env.py`](auv_nav/env.py)，再看 [`auv_nav/flow.py`](auv_nav/flow.py) 和 [`auv_nav/sac.py`](auv_nav/sac.py)。

## 这个仓库的整体闭环

```text
generate_wake.py
    ↓
wake_data/*.npy + *_meta.json
    ↓
PlanarRemusEnv + WakeField + FlowSampler
    ↓
SAC / baseline policies
    ↓                          ↓
checkpoint / logs / metrics    collect_offline_data.py → offline_data/
    ↓                                                      ↓
evaluate / demo / visualize    train_sac.py --offline-data (RLPD mode)
```

按实验流程理解这个仓库，通常最清晰：

1. 用 `scripts/generate_wake.py` 生成尾迹流场数据。
2. 用 `scripts/train_sac.py` 在指定流场上训练策略。
3. （可选）用 `scripts/collect_offline_data.py` 收集基线策略的离线数据，再用 RLPD 模式训练。
4. 用 `scripts/evaluate.py` 批量评估 checkpoint。
5. 用 `scripts/demo.py`、`scripts/visualize.py`、`scripts/figure_paper.py` 做行为分析和结果展示。

做严格算法对比时，建议优先使用仓库内置的标准 benchmark：

- 先用 `scripts/generate_standard_benchmarks.py` 生成 `benchmarks/*.json`
- 再用 `scripts/run_suite.py --benchmark-group ...` 跑成组实验
- 或者单独用 `scripts/evaluate.py --manifest benchmarks/<benchmark>.json` 统一评测

## 核心研究设定

### 1. 状态、动作和任务

环境主体是 [`auv_nav/env.py`](auv_nav/env.py) 中的 `PlanarRemusEnv`。

它的基本设定是：

- 动作为 2 维：航向相关指令 + 速度相关指令
- 环境内部包含 REMUS-100 风格动力学与自动驾驶仪
- 任务是在流场 ROI 内，从随机采样起点导航到目标点
- 每个 episode 有最大时间限制、边界限制和姿态/速度安全约束

任务几何有三种：

- `downstream`：顺流
- `cross_stream`：横流
- `upstream`：逆流

脚本层面又把它们映射为难度：

- `easy -> downstream`
- `medium -> cross_stream`
- `hard -> upstream`

所以你在训练脚本里看到的 `--difficulty hard`，本质上是在测试最有挑战性的逆流任务。

### 2. AUV 能看到什么

观测并不是“全局流场图像”，而是局部传感器式观测。

默认观测由两部分组成：

- AUV 自身运动状态与目标相对信息
- 机体附近若干流速探针读数

项目提供 3 种 probe layout：

- `s0`：1 个中心探针，最接近基础 DVL 设定，观测维度 10
- `s1`：2 个探针（中心 DVL + 短程前向 ADCP），观测维度 12
- `s2`：4 个探针（中心 DVL + 前向/侧向 ADCP），观测维度 16

这点很重要：这个仓库不是让 agent 直接看完整流场，而是让它在有限局部感知下做控制决策。

### 3. 奖励和安全约束

奖励与成本定义在 [`auv_nav/reward.py`](auv_nav/reward.py)。

每一步奖励大致由以下部分组成：

- 时间/步长惩罚：鼓励尽快完成任务
- 朝目标前进的 progress reward：鼓励有效推进
- 成功终止奖励：到达目标时给正奖励
- 失败或超时惩罚：出界、失稳、超时等情况给负奖励
- 可选的 safety cost / energy cost：用于更细的风险整形

当前仓库把目标函数显式分成三个 preset：

- `arrival_v1`
  兼容旧实验的“到达优先”目标。主优化量仍然是时间惩罚 + progress + terminal reward。
- `efficiency_v1`
  在 `arrival_v1` 基础上额外惩罚能耗和软安全风险。它是第一版效率目标，但在硬 benchmark 上往往过强。
- `efficiency_v2`
  根据 `efficiency_gain_sweep_v1` 的结果收敛出来的弱 safety shaping 版本。
  当前推荐把它作为“高效航行”主实验的默认起点。

因此，现在不建议再笼统地说“默认 reward 就是高效导航目标”。你需要在实验表格和命令里明确写出 `--objective arrival_v1`、`--objective efficiency_v1` 或 `--objective efficiency_v2`。

## 为什么要先生成流场

这个仓库没有在线 CFD。流场是先离线生成，再在训练时按时空插值读取。

流场生成脚本是 [`scripts/generate_wake.py`](scripts/generate_wake.py)，其职责是：

- 用二维 TRT-LBM 模拟圆柱绕流
- 支持单圆柱、串联双圆柱、并排双圆柱等配置
- 记录 ROI 中的 `(u, v, ω)` 时空演化
- 产出下游环境可直接读取的数据文件

生成后的数据在 `wake_data/` 中，通常每个 case 对应：

- `wake_*.npy`：主数据，形状为 `(T, Nx, Ny, 3)`
- `wake_*_meta.json`：元数据，记录空间分辨率、ROI 边界、时间步长等
- `wake_*_phase.npy`：涡相位估计结果

训练和评估真正依赖的是：

- `.npy`
- 同名 `_meta.json`

如果你想细看这个脚本怎么用、有哪些 profile、哪些参数能调，见：

- [`docs/generate_wake_usage.md`](docs/generate_wake_usage.md)

## 仓库结构应该怎么理解

### `auv_nav/`

这是核心包，建议按下面方式理解。

- `env.py`
  研究主环境。负责 episode 采样、动作解释、奖励计算、终止条件和观测组织。
- `vehicle.py`
  AUV 动力学模型与状态定义。
- `autopilot.py`
  低层控制后端，例如深度保持、航向控制、等效流体处理等。
- `flow.py`
  负责读取 `wake_data/`，并做空间和时间上的插值采样，把离线流场变成环境中的“可查询背景流”。
- `reward.py`
  奖励和安全成本模型。
- `sac.py`
  SAC agent 主体实现。
- `networks.py`
  Actor / Critic 网络结构及相关工具。
- `replay.py`
  经验回放池。支持 `TransitionReplay.from_npz()` 加载离线数据，`DualBufferSampler` 实现 RLPD 双 buffer 对称采样。
- `baselines.py`
  非学习型基线策略，用于对照实验。

如果你想知道“项目最核心的逻辑在哪”，答案通常是：

- 环境在 `auv_nav/env.py`
- 流场接入在 `auv_nav/flow.py`
- 算法在 `auv_nav/sac.py`

### `scripts/`

这是实验工作流的入口层。

- `generate_wake.py`
  生成流场数据。
- `train_sac.py`
  训练 SAC 智能体。支持 `--offline-data` 和 `--offline-ratio` 参数开启 RLPD 模式。
- `collect_offline_data.py`
  用基线策略（goalseek / crosscomp / worldcomp / privileged）收集离线 transition 数据，输出 `.npz` + `metadata.json`。
- `evaluate.py`
  用训练好的 checkpoint 做批量评估；支持固定 benchmark manifest 进行可复现评测。
- `generate_benchmark_manifest.py`
  预先采样并固化一组评估 episode，供算法间公平对比。
- `generate_standard_benchmarks.py`
  生成仓库约定的标准 benchmark manifests，显式拆分 `geometry / flow / topology / speed` 因子。
- `demo.py`
  跑单次或少量 episode，对比 baseline 或策略行为。
- `visualize.py`
  生成更详细的轨迹与流场动画。
- `figure_paper.py`
  生成论文级静态图。
- `run_suite.py`
  按 `benchmark × objective × gain × method × seed` 组织批量实验；当某些维度未启用时会自动退化成更简单的目录结构。
- `summarize_suite.py`
  按实际启用的实验维度汇总多个实验日志，并输出统一指标。

当前代码里的 factorized benchmark presets 仍默认走 `efficiency_v1`，这是为了保持既有预设和实验目录兼容。
如果你要开始新的主实验，建议显式覆盖成 `--objective efficiency_v2`。
旧的 legacy preset 仍保持兼容旧实验口径。
如果要专门比较目标函数，可以直接使用 `objective_ablation_v1` preset。
如果 `efficiency_v1` 在硬 benchmark 上学不稳，可以直接使用 `efficiency_gain_sweep_v1` preset，围绕低安全/低能耗权重做小范围扫描。当前 sweep 的推荐结果是 `efficiency_v2`。
- `plot_suite.py`, `plot_training.py`
  绘制训练曲线与消融图。
- `train_utils.py`
  训练/评估的公共工具函数。

### `tests/`

当前测试重点在两类：

- 改进版 SAC
- 多圆柱 LBM 数据生成

这说明项目最容易出错、也最值得固定行为的部分，正是算法实现和流场生成。

### `docs/`

这里既有研究型说明，也有实现文档。

你可以优先看：

- `docs/generate_wake_usage.md`
- `docs/environment_design.md`
- `docs/SAC_improvements_survey.md`
- `docs/world_model_and_offline_rl_survey.md`

### `wake_data/`

离线流场数据库。没有它，环境无法运行。

### `benchmarks/`

固定评估 episode 的标准 manifests。它们只冻结任务实例，不绑定 `probe_layout/history_length`，所以可以在 `s0/s1/s2` 和不同 history 设置之间复用。

### `results/`

整理后的实验报告目录。当前最重要的汇总文件是：

- `results/rl_navigation_experiment_report.md`

如果你不想先逐个翻 `experiments/*/final_eval.json` 和 `ablation_summary.csv`，可以直接从这里进入。

### `checkpoints/`

训练产物目录，通常保存：

- agent 权重
- replay buffer
- RNG 状态
- `trainer_state.json`
- 训练日志与评估日志

## 一次训练到底发生了什么

从 [`scripts/train_sac.py`](scripts/train_sac.py) 看，单次训练的主流程大致是：

1. 读取指定流场或自动发现 `wake_data/` 下的第一个可用文件。
2. 用 `make_planar_env(...)` 构造环境。
3. 根据 `probe_layout` 和 `history_length` 确定观测维度。
4. 初始化 SAC agent、replay buffer 和日志路径。
5. 环境交互采样。
6. 在满足 `update_after` 等条件后执行 SAC 更新。
7. 周期性评估、存 checkpoint、写日志。

这个流程里有几个研究上很关键的控制项：

- `--probe-layout`
  控制 agent 的感知方式。
- `--history-length`
  控制是否把最近若干帧观测堆叠起来，缓解部分可观测性。
- `--use-layernorm`
  稳定网络训练。
- `--dropout-rate`
  对 critic 做 DroQ 风格正则化。
- `--use-asymmetric-critic`
  训练时让 critic 看到特权观测，而 actor 仍只用局部感知。
- `--difficulty` / `--task-geometry`
  控制任务类型。
- `--target-speed`
  控制 AUV 极限速度，是本项目里非常关键的实验旋钮。
- `--objective`
  显式指定 reward objective。做“高效航行”主实验时，建议优先试 `efficiency_v2`。
- `--num-envs`
  控制并行环境数。现在默认会按 CPU 核心数保守自适应，而不是固定 16；在 `6C/12T` 机器上默认会落到大约 `6`。

## 为什么 `--target-speed 1.5` 重要

项目默认把 AUV 在满转速下的名义最大速度设为 `2.0 m/s`。如果流场来流速度接近这个量级，问题会变得非常尖锐。

比如：

- 流场 `U = 1.5 m/s`
- AUV 最大速度也限制为 `1.5 m/s`

那么在逆流任务里，AUV 基本处于“靠本体推进难以直接胜出”的临界状态。此时：

- 直线迎流策略通常很差
- agent 必须学会利用局部流动结构
- 这更能体现 RL 是否真的“借力流场”，而不是只学会简单跟踪目标

所以 README 里最值得你记住的一条实验建议是：

> 当你想测试策略是否真的具备复杂流场利用能力时，优先尝试 `U_flow = 1.5 m/s` 且 `target-speed = 1.5 m/s` 的逆流任务。

## 常见使用路径

### 1. 先生成数据，再训练

```bash
python -m scripts.generate_wake --profile navigation
python -m scripts.train_sac \
    --flow wake_data/wake_v8_U1p50_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --difficulty hard \
    --target-speed 1.5 \
    --objective efficiency_v2 \
    --probe-layout s0 \
    --history-length 5 \
    --use-layernorm \
    --dropout-rate 0.01 \
    --use-asymmetric-critic \
    --seed 42 \
    --device cpu
```

### 2. 评估一个 checkpoint

```bash
python -m scripts.evaluate \
    --checkpoint checkpoints/sac/<run_dir> \
    --episodes 100 \
    --objective efficiency_v2 \
    --flow wake_data/wake_v8_U1p50_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --difficulty hard \
    --target-speed 1.5
```

### 3. 看基线策略在复杂流场里怎么失败

```bash
python -m scripts.demo \
    --policy goal \
    --flow wake_data/wake_v8_U1p50_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --difficulty hard \
    --target-speed 1.5 \
    --plot
```

### 4. 用 RLPD 加速训练（离线数据 + 在线 SAC）

```bash
# 第一步：收集基线策略的离线数据
python -m scripts.collect_offline_data \
    --policy worldcomp \
    --flow wake_data/wake_v8_U1p50_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 --difficulty hard --target-speed 1.5 \
    --objective efficiency_v2 \
    --episodes 500 --seed 0 \
    --output-dir offline_data/worldcomp

# 第二步：用 RLPD 模式训练（自动 50/50 采样离线+在线数据）
python -m scripts.train_sac \
    --flow wake_data/wake_v8_U1p50_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --difficulty hard --target-speed 1.5 \
    --objective efficiency_v2 \
    --probe-layout s0 --use-layernorm \
    --offline-data offline_data/worldcomp/transitions.npz \
    --offline-ratio 0.5 \
    --seed 42 --device cpu
```

支持的离线数据策略：`goalseek`（无特权）、`crosscomp`（无特权）、`worldcomp`（中度特权）、`privileged`（强特权）。详见 `docs/rlpd_design.md`。

### 5. 做目标函数对照实验（objective ablation）

```bash
conda run -n mytorch1 python -m scripts.run_suite \
    --preset objective_ablation_v1

conda run -n mytorch1 python -m scripts.summarize_suite \
    --suite-root experiments/objective_ablation_v1

conda run -n mytorch1 python -m scripts.plot_suite \
    --suite-root experiments/objective_ablation_v1
```

这个 preset 会在 `single_u15_upstream_tgt15` 上，用 `sac_stack4`、3 个 seeds，直接比较
`arrival_v1` 和 `efficiency_v1`。

### 6. 做 efficiency gain sweep

```bash
conda run -n mytorch1 python -m scripts.run_suite \
    --preset efficiency_gain_sweep_v1

conda run -n mytorch1 python -m scripts.summarize_suite \
    --suite-root experiments/efficiency_gain_sweep_v1

conda run -n mytorch1 python -m scripts.plot_suite \
    --suite-root experiments/efficiency_gain_sweep_v1
```

这个 preset 固定 `single_u15_upstream_tgt15 + sac_stack4 + efficiency_v1`，扫描：
`(energy,safety) = (0,0), (0,0.25), (0,0.5), (0,1.0), (1e-4,0.5), (2e-4,0.5), (5e-4,2.0)`。
推荐先用它确认“低强度 safety shaping”是否能保住 success，再决定是否重新引入 energy shaping。

如果你要在 factorized benchmark suites 上直接使用当前推荐的新目标函数，可以显式覆盖：

```bash
conda run -n mytorch1 python -m scripts.run_suite \
    --preset study_core_v1 \
    --objective efficiency_v2
```

### 7. 按最佳周期评估选 checkpoint

```bash
conda run -n mytorch1 python -m scripts.evaluate_best_checkpoint \
    --run-dir experiments/efficiency_gain_sweep_v1/single_u15_upstream_tgt15/efficiency_v1/e0_s0p25/sac_stack4/seed_44
```

这个脚本会读取 `eval_log.csv`，按
`eval_success_rate -> eval_return -> -eval_safety_cost -> -eval_time_s`
选择最佳周期评估点，并给出对应的 `scripts.evaluate` 命令。

### 8. 生成更复杂的多圆柱流场

```bash
python -m scripts.generate_wake --profile tandem_G35_nav
python -m scripts.generate_wake --profile side_by_side_G35_nav
```

## 运行脚本时要知道的约定

- 所有脚本都建议从仓库根目录以模块方式运行：
  `python -m scripts.<name>`
- 本仓库本地运行约定使用 Conda 环境 `mytorch1`
- 如果 `wake_data/` 下没有流场文件，训练和演示脚本会先报缺少数据
- 训练时如果不显式指定 `--flow`，会自动发现 `wake_data/` 下的第一个 `wake_*_roi.npy`

## 输出产物怎么看

### 流场生成阶段

输出在 `wake_data/`：

- `wake_*.npy`
- `wake_*_meta.json`
- `wake_*_phase.npy`

### 离线数据收集阶段

输出在 `offline_data/<policy_name>/`：

- `transitions.npz`：包含 obs, actions, rewards, costs, next_obs, dones
- `metadata.json`：策略名、流场路径、探针布局、统计信息（成功率、回报分布等）

### 训练阶段

输出保存在 `--save-dir` 指定的目录中，默认是 `checkpoints/sac/`。常见文件包括：

- `agent_latest.pt`
- `agent_step_<N>.pt`
- `agent_final.pt`
- `replay_latest.pkl`
- `rng_state.pkl`
- `trainer_state.json`
- `train_log.jsonl`
- `eval_log.csv`
- `final_eval.json`

其中最关键的是 `trainer_state.json`，因为它保存了恢复训练和评估所需的重要配置，例如：

- flow 路径
- history length
- probe layout
- reset options
- agent config

## 如果你第一次看这个仓库，建议这样读

### 目标是理解“这个项目在干什么”

建议顺序：

1. 先读本文档
2. 再读 [`docs/generate_wake_usage.md`](docs/generate_wake_usage.md)
3. 然后读 [`auv_nav/env.py`](auv_nav/env.py) 里的 `PlanarRemusEnvConfig`
4. 最后看 [`scripts/train_sac.py`](scripts/train_sac.py) 的参数入口

### 目标是理解“流场怎么接进环境”

建议顺序：

1. [`scripts/generate_wake.py`](scripts/generate_wake.py)
2. [`auv_nav/flow.py`](auv_nav/flow.py)
3. [`auv_nav/env.py`](auv_nav/env.py)

### 目标是理解”算法做了哪些改进”

建议顺序：

1. [`docs/SAC_improvements_survey.md`](docs/SAC_improvements_survey.md)
2. [`auv_nav/networks.py`](auv_nav/networks.py)
3. [`auv_nav/sac.py`](auv_nav/sac.py)
4. [`tests/test_improved_sac.py`](tests/test_improved_sac.py)

### 目标是理解”RLPD 离线-在线训练怎么用”

建议顺序：

1. [`docs/rlpd_design.md`](docs/rlpd_design.md) — 算法选择、实验设计、论文关联
2. [`auv_nav/replay.py`](auv_nav/replay.py) — `from_npz` 和 `DualBufferSampler`
3. [`scripts/collect_offline_data.py`](scripts/collect_offline_data.py) — 离线数据收集
4. [`scripts/train_sac.py`](scripts/train_sac.py) — `--offline-data` 参数入口

## 快速索引

- 想生成流场：[`scripts/generate_wake.py`](scripts/generate_wake.py)
- 想看流场生成文档：[`docs/generate_wake_usage.md`](docs/generate_wake_usage.md)
- 想训练 agent：[`scripts/train_sac.py`](scripts/train_sac.py)
- 想收集离线数据：[`scripts/collect_offline_data.py`](scripts/collect_offline_data.py)
- 想了解 RLPD 设计：[`docs/rlpd_design.md`](docs/rlpd_design.md)
- 想评估 checkpoint：[`scripts/evaluate.py`](scripts/evaluate.py)
- 想按最佳周期评估选择 checkpoint：[`scripts/evaluate_best_checkpoint.py`](scripts/evaluate_best_checkpoint.py)
- 想看环境定义：[`auv_nav/env.py`](auv_nav/env.py)
- 想看流场读取：[`auv_nav/flow.py`](auv_nav/flow.py)
- 想看奖励：[`auv_nav/reward.py`](auv_nav/reward.py)
- 想看基线：[`auv_nav/baselines.py`](auv_nav/baselines.py)
- 想直接看整理后的实验报告：[`results/rl_navigation_experiment_report.md`](results/rl_navigation_experiment_report.md)

## 测试

```bash
pytest tests/
```

或运行指定模块：

```bash
pytest tests/test_improved_sac.py
pytest tests/test_multi_cylinder_lbm.py
```

## 当前 README 的定位

这份 README 的目标不是完整替代代码文档，而是让你在不深入读源码的前提下，先回答下面这些问题：

- 这个仓库研究的是什么问题
- 实验闭环是怎样的
- 数据、环境、算法和可视化分别在哪里
- 训练时有哪些关键实验旋钮
- 我第一次上手应该先跑哪个脚本

如果这些问题你已经能答出来，这份 README 就达到了目的。
