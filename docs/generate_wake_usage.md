# `scripts/generate_wake.py` 使用说明

本文档说明如何使用 `scripts/generate_wake.py` 生成尾迹流场数据，包括：

- 基本运行方式
- 3 个内置 profile 的含义
- 命令行可调参数
- 代码层可调参数
- 输出文件结构与命名规则
- 典型使用示例与注意事项

## 1. 脚本作用

`scripts/generate_wake.py` 使用二维 D2Q9 TRT-LBM（Two-Relaxation-Time Lattice Boltzmann Method）模拟圆柱绕流尾迹，并将结果导出为供 AUV 环境直接读取的流场数据。

输出内容包括：

- `wake_*.npy`：主流场数据，形状为 `(T, Nx, Ny, 3)`
- `wake_*_meta.json`：元数据文件，记录空间分辨率、记录时间间隔、ROI 范围等
- `wake_*_phase.npy`：基于监测信号估计的涡脱落相位序列

其中主流场 `.npy` 的 3 个通道分别为：

- `u_mps`：x 方向速度，单位 m/s
- `v_mps`：y 方向速度，单位 m/s
- `omega_1ps`：涡量，单位 1/s

项目下游环境主要依赖：

- `wake_*.npy`
- 同名的 `wake_*_meta.json`

因此这两者必须成对保留。

## 2. 运行前准备

从仓库根目录运行，并按仓库约定切换到 Conda 环境 `mytorch1`：

```bash
conda activate mytorch1
python -m scripts.generate_wake --help
```

如果需要 GPU 加速，还需要安装并正确配置 `CuPy`。

## 3. 最基本的使用方式

### 3.1 生成默认单圆柱导航流场

```bash
python -m scripts.generate_wake --profile navigation
```

这会在 `./wake_data/` 下生成一组用于导航训练的单圆柱尾迹数据。

### 3.2 生成串联双圆柱尾迹

```bash
python -m scripts.generate_wake --profile tandem_G35_nav
```

### 3.3 生成并排双圆柱尾迹

```bash
python -m scripts.generate_wake --profile side_by_side_G35_nav
```

### 3.4 使用 GPU

```bash
python -m scripts.generate_wake --profile navigation --gpu
```

如果传入了 `--gpu` 但环境中没有 `CuPy`，脚本会打印警告并回退到 CPU。

## 4. 命令行参数

当前 CLI 暴露的参数很少，只有 4 个：

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--profile` | `str` | `navigation` | 选择预设流场 profile |
| `--gpu` | flag | `False` | 使用 CuPy 在 GPU 上运行 |
| `--re` | `float ...` | `None` | 覆盖 profile 默认的 Reynolds 数列表 |
| `--u` | `float ...` | `None` | 覆盖 profile 默认的来流速度列表，单位 m/s |

### 4.1 `--profile`

可选值：

- `navigation`
- `tandem_G35_nav`
- `side_by_side_G35_nav`

如果传入未知 profile，脚本会抛出 `ValueError`。

### 4.2 `--re`

可以传入一个或多个 Reynolds 数，例如：

```bash
python -m scripts.generate_wake --profile navigation --re 150 250
```

如果不传，使用 profile 内部默认值。

### 4.3 `--u`

可以传入一个或多个自由流速度，例如：

```bash
python -m scripts.generate_wake --profile navigation --u 1.0 1.5
```

如果不传，使用 profile 内部默认值。

### 4.4 `--re` 和 `--u` 的组合方式

脚本会对 `Re` 和 `U` 做笛卡尔积组合，每一个组合都会单独生成一组流场文件。

例如：

```bash
python -m scripts.generate_wake --profile navigation --re 150 250 --u 1.0 1.5
```

会生成 4 个 case：

- `(Re=150, U=1.0)`
- `(Re=250, U=1.0)`
- `(Re=150, U=1.5)`
- `(Re=250, U=1.5)`

## 5. 内置 profile 说明

CLI 的核心其实是选择 profile。每个 profile 决定了圆柱布局、仿真域大小、记录时长、ROI 范围、下采样倍率等关键物理参数。

### 5.1 `navigation`

默认单圆柱导航 profile，适合常规训练数据生成。

默认参数组合：

- `U_phys_values = (1.0, 1.5)`
- `Re_values = (150.0, 250.0)`

固定几何与记录设置：

- 单圆柱
- `D_phys = 12.0 m`
- `Lx_phys = 480.0 m`
- `Ly_phys = 180.0 m`
- `dx = 0.3 m`
- `base_dt = 0.015 s`
- `cyl_x_phys = 96.0 m`
- `cyl_y_center = 90.0 m`
- `T_spinup_phys = 720.0 s`
- `T_record_phys = 360.0 s`
- `record_interval = 0.3 s`
- `roi_x_end_D = 18.0`
- `roi_y_half_D = 2.5`
- `turbulence_intensity = 0.05`
- `turbulence_length_scale = 20`
- `roi_downsample = 2`

默认会生成 4 个 case。

### 5.2 `tandem_G35_nav`

串联双圆柱 profile。两个相同圆柱沿主流方向排布，表面间距为 `3.5D`。

默认参数组合：

- `U_phys_values = (0.8, 1.0, 1.2)`
- `Re_values = (150.0, 200.0)`

固定几何与记录设置：

- 双圆柱串联
- `gap_ratio = 3.5`
- `D_phys = 12.0 m`
- `Lx_phys = 540.0 m`
- `Ly_phys = 180.0 m`
- `dx = 0.3 m`
- `base_dt = 0.015 s`，随后按稳定性自动修正
- 第一个圆柱中心：`x = 96.0 m, y = 90.0 m`
- 第二个圆柱中心位于下游：`x = cyl1_x + D + 3.5D`
- `T_spinup_phys = 720.0 s`
- `T_record_phys = 360.0 s`
- `record_interval = 0.3 s`
- `roi_x_start_D = -1.0`
- `roi_x_end_D = 24.5`
- `roi_y_half_D = 3.0`
- `roi_downsample = 2`
- `case_tag = "tandem_G35_"`

默认会生成 6 个 case。

这个 profile 的特点是 ROI 会向上游多取 `1D`，从而覆盖两个圆柱之间的间隙区域。

### 5.3 `side_by_side_G35_nav`

并排双圆柱 profile。两个相同圆柱位于同一 `x` 位置，在 `y` 方向上相隔 `3.5D`。

默认参数组合：

- `U_phys_values = (0.8, 1.0, 1.2)`
- `Re_values = (150.0, 200.0)`

固定几何与记录设置：

- 双圆柱并排
- `gap_ratio = 3.5`
- `D_phys = 12.0 m`
- `Lx_phys = 480.0 m`
- `Ly_phys = 240.0 m`
- `dx = 0.3 m`
- `base_dt = 0.015 s`，随后按稳定性自动修正
- 两个圆柱共用 `x = 96.0 m`
- ROI 的 y 中心强制设为域中心 `120.0 m`
- `T_spinup_phys = 720.0 s`
- `T_record_phys = 360.0 s`
- `record_interval = 0.3 s`
- `roi_x_start_D = 2.0`
- `roi_x_end_D = 25.0`
- `roi_y_half_D = 6.0`
- `roi_downsample = 2`
- `case_tag = "sbs_G35_"`

默认会生成 6 个 case。

这个 profile 的特点是 ROI 以两个圆柱之间的中线为中心，垂向覆盖范围更大，便于同时捕获两股尾迹。

## 6. 输出文件位置与命名规则

CLI 模式下，输出目录固定为：

```bash
./wake_data
```

每个 case 会生成 3 个文件：

- `wake_<tag>.npy`
- `wake_<tag>_phase.npy`
- `wake_<tag>_meta.json`

其中 `<tag>` 的构成为：

```text
{case_tag}v8_U{U}_Re{Re}_D{D}_dx{dx_out}_Ti{TiPct}pct_{frames}f_roi
```

例如：

```text
wake_v8_U1p50_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy
wake_tandem_G35_v8_U1p00_Re200_D12p00_dx0p60_Ti5pct_1200f_roi.npy
wake_sbs_G35_v8_U0p80_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy
```

说明：

- 小数点会被替换为 `p`
- `dx` 是输出网格间距 `dx_out`，不是仿真内部细网格 `dx`
- `frames` 是最终保存的时间帧数

## 7. 输出内容的具体含义

### 7.1 主数组 `wake_*.npy`

数组形状：

```text
(T, Nx, Ny, 3)
```

最后一个维度的 3 个通道为：

- `[..., 0] = u_mps`
- `[..., 1] = v_mps`
- `[..., 2] = omega_1ps`

数据类型为 `float16`，用于降低数据体积。

### 7.2 元数据 `wake_*_meta.json`

会记录：

- `Re`
- `U_ref`
- `D_ref`
- `dx_m`
- `dx_sim_m`
- `record_interval_actual_s`
- `roi_x0_phys_m` / `roi_x1_phys_m`
- `roi_y0_phys_m` / `roi_y1_phys_m`
- `total_frames`
- `data_shape`
- `channels`
- `backend`
- `elapsed_s`

其中：

- `dx_m` 是输出网格分辨率，供下游 `WakeFieldMetadata` 直接使用
- `dx_sim_m` 是 LBM 仿真内部使用的细网格分辨率

### 7.3 相位文件 `wake_*_phase.npy`

该文件保存基于监测点横向速度信号估计得到的涡脱落相位。当前训练主流程不直接依赖这个文件，但它对后续分析、同步或可视化是有价值的。

## 8. 脚本内部如何决定时间步长

虽然 CLI 没有暴露 `dt` 参数，但脚本并不是简单使用固定 `dt`。它会调用 `stable_dt_for_case(...)` 自动寻找可行时间步长，以同时满足两个稳定性约束：

- `tau_s >= 0.52`
- 格子 Mach 数 `Ma < 0.15`

这意味着：

- 你可以覆盖 `--re` 和 `--u`
- 但如果组合过于激进，脚本可能判定无可行 `dt` 并跳过该 case

主循环中每个 case 的异常会被捕获，控制台会打印：

```text
Skipping case (...)
```

## 9. 代码层可调参数

如果你觉得 CLI 暴露得不够，可以直接修改 profile 构造逻辑，或者在 Python 中直接构造 `PhysicsConfig` 并调用 `run_simulation(...)`。

### 9.1 `PhysicsConfig` 中可调的主要字段

| 字段 | 含义 |
| --- | --- |
| `Re` | Reynolds 数 |
| `U_phys` | 来流速度，单位 m/s |
| `D_phys` | 参考圆柱直径，单位 m |
| `Lx_phys`, `Ly_phys` | 物理域大小 |
| `dx` | 仿真网格尺寸 |
| `dt` | 仿真时间步长 |
| `cyl_x_phys`, `cyl_y_center` | 主圆柱中心位置 |
| `cyl_y_jitter` | 主圆柱 y 方向扰动 |
| `turbulence_intensity` | 入口湍动强度 |
| `turbulence_length_scale` | 入口扰动的空间相关长度 |
| `T_spinup_phys` | spin-up 时长 |
| `T_record_phys` | 正式记录时长 |
| `record_interval` | 输出帧时间间隔 |
| `roi_x_start_D`, `roi_x_end_D` | ROI 在 x 方向相对圆柱的位置范围，单位为 `D` |
| `roi_y_half_D` | ROI 在 y 方向的半宽，单位为 `D` |
| `roi_downsample` | ROI 空间下采样倍率 |
| `extra_cylinders` | 额外圆柱列表，格式为 `(x_phys, y_phys, D_phys)` |
| `roi_y_center_override` | 覆盖 ROI 的 y 中心 |
| `case_tag` | 输出文件名前缀 |

### 9.2 哪些参数最常需要调

通常最值得调整的是：

- `Re`
- `U_phys`
- `turbulence_intensity`
- `turbulence_length_scale`
- `T_spinup_phys`
- `T_record_phys`
- `record_interval`
- `roi_x_start_D`
- `roi_x_end_D`
- `roi_y_half_D`
- `roi_downsample`
- `extra_cylinders`

如果你的目标是“生成更适合 RL 训练的数据”，这几个参数会直接影响：

- 流场复杂度
- 数据时长
- 数据体积
- AUV 可见的尾迹覆盖区域
- 多圆柱干涉结构

## 10. 高级用法：直接在 Python 中调用

如果你想自定义输出目录、随机种子，或者构造 CLI 不支持的几何，可以直接调用脚本内部函数。

```python
from scripts.generate_wake import PhysicsConfig, run_simulation

pc = PhysicsConfig(
    Re=180.0,
    U_phys=1.2,
    D_phys=12.0,
    Lx_phys=480.0,
    Ly_phys=180.0,
    dx=0.3,
    dt=0.015,
    cyl_x_phys=96.0,
    cyl_y_center=90.0,
    turbulence_intensity=0.05,
    turbulence_length_scale=20,
    T_spinup_phys=720.0,
    T_record_phys=360.0,
    record_interval=0.3,
    roi_x_start_D=2.0,
    roi_x_end_D=18.0,
    roi_y_half_D=2.5,
    roi_downsample=2,
)

run_simulation(
    pc,
    output_dir="wake_data/custom",
    rng_seed=123,
    use_gpu=False,
)
```

这种方式比 CLI 多出几个可控项：

- `output_dir`
- `rng_seed`
- 任意 `PhysicsConfig`

## 11. 典型命令示例

### 11.1 默认导航数据

```bash
python -m scripts.generate_wake --profile navigation
```

### 11.2 只生成一个 case

```bash
python -m scripts.generate_wake --profile navigation --re 150 --u 1.0
```

### 11.3 生成 6 个串联双圆柱 case

```bash
python -m scripts.generate_wake --profile tandem_G35_nav
```

### 11.4 自定义并排双圆柱速度扫描

```bash
python -m scripts.generate_wake --profile side_by_side_G35_nav --u 0.9 1.1 1.3 --re 150 200
```

### 11.5 GPU 模式

```bash
python -m scripts.generate_wake --profile navigation --gpu
```

## 12. 使用建议

### 12.1 如果你只是想快速得到可训练数据

优先使用：

```bash
python -m scripts.generate_wake --profile navigation
```

### 12.2 如果你要研究复杂尾迹干涉

优先使用：

- `tandem_G35_nav`
- `side_by_side_G35_nav`

### 12.3 如果你要控制生成规模

最直接的方式是减少：

- `--re` 的取值个数
- `--u` 的取值个数

因为总 case 数就是两者数量的乘积。

## 13. 当前 CLI 的限制

当前命令行无法直接设置以下内容：

- 输出目录
- 随机种子
- 圆柱位置
- ROI 位置与范围
- 记录时长
- 湍流强度
- 下采样倍率
- 多圆柱布局细节

如果要调整这些内容，需要：

- 修改 `make_training_configs(...)`
- 修改各个 profile 构造函数
- 或直接在 Python 中调用 `run_simulation(...)`

## 14. 与下游训练脚本的衔接

生成完成后，可将得到的 `.npy` 文件路径传给训练或评估脚本，例如：

```bash
python -m scripts.train_sac --flow wake_data/wake_v8_U1p50_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy
```

注意同名的 `_meta.json` 文件必须存在，因为 `auv_nav.flow.WakeField.from_files(...)` 会自动按文件名查找它。
