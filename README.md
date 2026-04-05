# AUV Navigation in Complex Flow Fields

## 项目概述 (Project Overview)
本项目是一个基于深度强化学习（Deep Reinforcement Learning, DRL）的研究项目，旨在为自主水下航行器（Autonomous Underwater Vehicle, AUV，以 REMUS-100 为原型）在复杂、非均匀流场（如通过格子玻尔兹曼方法生成的单/多圆柱尾迹流场）中，学习高效、鲁棒的导航与轨迹规划策略。本项目主要采用了高度优化与改进版的 Soft Actor-Critic (SAC) 算法。

---

## 目录结构 (Repository Structure)

本项目代码模块化清晰，主要分为以下几个部分：

- **`auv_nav/`**: 核心代码包，包含环境定义、航行器动力学、流场建模和 RL 算法。
  - **`env.py`**: 基于 Gymnasium 封装的 AUV 导航环境 (`PlanarRemusEnv`)，处理部分可观测性与环境交互。
  - **`vehicle.py` & `autopilot.py`**: 实现了 REMUS-100 航行器的动力学模型与底层控制系统（如基于航向和深度的自动驾驶仪）。
  - **`flow.py`**: 流场环境建模、流体数据插值采样，将 LBM 数据接入实时物理仿真。
  - **`sac.py` & `networks.py` & `replay.py`**: 核心 SAC 算法实现、神经网络结构以及经验回放池。包含了如 LayerNorm、Dropout (DroQ) 和非对称 Critic 等多种改进架构。
  - **`reward.py`**: 统一的奖励建模体系 (`RewardModel`) 和安全性成本模型 (`SafetyCostModel`)，用于风险整形、步长惩罚与边界碰撞检测。
  - **`baselines.py`**: 用于对比测试的传统基线策略（如视线法目标追踪、纯迎流补偿、全球坐标补偿、无流体假设规划等）。

- **`scripts/`**: 项目的所有可执行脚本，涵盖生成、训练、评估、可视化与论文制图流程（详见“使用指南”）。

- **`tests/`**: 单元测试目录，基于 `pytest` 编写，包含针对改进版 SAC 模块 (`test_improved_sac.py`) 和多圆柱 LBM 生成 (`test_multi_cylinder_lbm.py`) 的自动化测试。

- **`docs/`**: 设计文档、调研报告（如 SAC 算法改进调研、世界模型调研等），以及核心功能的规格设计（在 `superpowers/` 中）。

- **`wake_data/`**: 存放通过 LBM (Lattice Boltzmann Method) 预先生成的流场数据（`.npy` 格式）。
- **`checkpoints/`**: 模型训练时的自动保存路径（权重和日志）。

---

## 详细使用指南 (Usage Guide)

### 1. 生成尾迹流场数据 (Generating Wake Data)
在进行训练之前，首先需要生成用于仿真环境的流场数据。执行 `scripts/generate_wake.py` 脚本（基于 TRT-LBM 流体动力学仿真）生成高质量尾迹数据。

该脚本支持 **单圆柱** 以及 **多种多圆柱（Multi-cylinder）尾迹配置**，模拟极其复杂的水下流场：
```bash
# 生成默认的标准导航尾迹流场（单圆柱）
python scripts/generate_wake.py --profile navigation

# 生成串联双圆柱 (Tandem) 尾迹流场，间距为 3.5D
python scripts/generate_wake.py --profile tandem_G35_nav

# 生成并排双圆柱 (Side-by-side) 尾迹流场，间距为 3.5D
python scripts/generate_wake.py --profile side_by_side_G35_nav

# 使用 GPU 加速生成（需要安装并配置 CuPy）
python scripts/generate_wake.py --profile navigation --gpu

# 甚至可以自定义雷诺数 (Reynolds number) 和自由流速 (Free-stream velocity)
python scripts/generate_wake.py --re 150 250 --u 1.0 1.5
```

### 2. 训练强化学习智能体 (Training the SAC Agent)
使用 `scripts/train_sac.py` 在指定的流场和任务难度下训练 SAC 模型。本项目实现了一系列前沿的 **SAC 改进技术 (Architecture Improvements)** 以应对复杂流场环境。

```bash
# 在具有挑战性的逆流环境（hard 难度）下训练模型，并启用多项算法改进
python scripts/train_sac.py \
    --flow wake_data/wake_v8_U1.50_Re150_D12.00_dx0.60_Ti5pct_1200f_roi.npy \
    --target-speed 1.5 \
    --difficulty hard \
    --total-steps 50000 \
    --use-layernorm \
    --dropout-rate 0.01 \
    --use-asymmetric-critic \
    --history-length 5 \
    --seed 42 \
    --device cpu
```

> **💡 核心实验技巧 (Pro Tip):** 
> 强烈建议在训练和测试时，配合 $1.5 \text{ m/s}$ 的流场使用 `--target-speed 1.5` 选项，将 AUV 的极限速度从默认的 $2.0 \text{ m/s}$ 降维锁死在 $1.5 \text{ m/s}$。这会制造出一个严苛的**欠驱动临界状态 ($U_{flow} / V_{AUV} = 1.0$)**。在此设定下，传统的直线抗流策略将寸步难行，而这正是展示 SAC 强化学习算法通过“借力涡流”实现智能突破的绝佳场景！

**关键改进参数解析：**
- `--use-layernorm`: 在 MLP 骨干中加入 Layer Normalization，有助于稳定深层网络在复杂观测下的训练。
- `--dropout-rate`: 设置 Dropout 概率（如 `0.01`，参考 DroQ 算法），有效缓解 Critic 早期过拟合问题。
- `--use-asymmetric-critic`: 启用非对称 Critic。允许 Critic 在训练阶段接收额外的特权观测（Privileged Observations，如全图真实流速），而在评估时 Actor 仅依赖机载传感器的局部观测，从而显著加速收敛。
- `--history-length`: 设置观测历史堆叠长度，赋予智能体时序记忆能力，有效处理环境的部分可观测性 (POMDP)。

### 3. 模型评估 (Evaluating a Checkpoint)
利用 `scripts/evaluate.py` 对特定的 checkpoint 进行大批量测试以统计如到达成功率、耗时、能量消耗等指标。
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/sac/<run_dir> \
    --episodes 100 \
    --seed 123 \
    --flow wake_data/wake_v8_U1.00_Re150...npy \
    --difficulty hard
```

### 4. 演示、轨迹分析与动画渲染 (Demos & Visualization)
- **基线策略交互对比 (`scripts/demo.py`)**:
  用于方便地调用各种预设基线策略在环境中单次或多次运行，并绘制静态轨迹对比图。
  ```bash
  # 运行 "目标追踪 (goal)" 并在屏幕弹窗展示轨迹（观察其在逆流中失败的极限情况）
  python scripts/demo.py --policy goal --flow wake_data/wake_v8_U1.50_Re150...npy --target-speed 1.5 --difficulty hard --plot

  # 所有 baseline 并排对比并保存图像
  python scripts/demo.py --policy all --flow wake_data/wake_v8_U1.50_Re150...npy --target-speed 1.5 --difficulty hard --plot --save comparison.png
  ```

- **四面板详细数据动画 (`scripts/visualize.py`)**:
  专为轨迹细节打造，可将单回合渲染为包含：① 全局涡量热力图、② 距离目标曲线、③ 累积奖励曲线、④ 目标速度分解 (Speed Decomposition，显示 AUV 自带动力与流场带来动力的拆解) 的 MP4 或 GIF 动画。
  ```bash
  python scripts/visualize.py --policy goal --flow wake_data/...npy --save animation.mp4
  ```

### 5. 论文级图表生成 (Generating Paper Figures)
使用 `scripts/figure_paper.py` 生成标准出版物级别的 PDF 静态科研插图。这会生成涵盖不同策略对 "流体动力利用率 (Quantitative Flow Contribution)" 指标影响的分析对比图 (`figure1_comparison.pdf` 与 `figure2_speed_decomposition.pdf`)。
```bash
python scripts/figure_paper.py \
    --flow wake_data/...npy \
    --checkpoint checkpoints/sac/<run_dir> \
    --outdir ./figures/
```

### 6. 批量消融实验与绘图 (Batch Suites & Plotting)
项目提供了一套自动化的批处理与统计流脚本，用于进行多随机种子（Multi-seed）与模块消融（Ablation）实验：
- `scripts/run_suite.py`: 根据预设文件连续跑大规模多参数组合。
- `scripts/summarize_suite.py`: 提取 Tensorboard 或 JSONL 日志数据，并合并多 Seed 统计信息。
- `scripts/plot_suite.py` & `scripts/plot_training.py`: 读取汇总数据，生成论文所需的平滑训练曲线柱状分析图。

```bash
# 启动多种子消融训练
python scripts/run_suite.py --preset ablation_improvements --seeds 42 43 44 --difficulty hard
```

### 7. 运行单元测试 (Running Unit Tests)
使用 `pytest` 运行针对代码核心模块的测试，确保你的环境以及安装正常工作：
```bash
pytest tests/
# 或测试指定模块
pytest tests/test_improved_sac.py
pytest tests/test_multi_cylinder_lbm.py
```
