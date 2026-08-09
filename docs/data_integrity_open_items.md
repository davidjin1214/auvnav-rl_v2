# 数据完整性待核项（2026-08-02 记录，暂未处理）

三条在别的工作里顺带撞见、**已在本机核实过证据但尚未处理**的记账问题。都不影响方法本身，
但都会在投稿/返修阶段被审稿人问到，且第 1 条一旦成立会直接推翻一格主结果。

发现场景：跑一组与本仓库无关的对照测试时，两个独立 agent 各自翻了 `paper/` 与
`offline_data/`，报出前两条；第 3 条是本次核实时顺带发现的。**下面所有事实均已亲自核过，
路径与数字可直接复用；未核实的部分逐条标注。**

---

## 1. `crosscomp-2000` 的采集种子可能与评估 manifest 重叠 ⚠️ 最高优先

### 已核实的机制

- 采集：`scripts/collect_offline_data.py:315` → `episode_seed = base_seed + ep`，
  `base_seed` 来自 `--seed`（:619）。
- 评估 manifest 生成：`scripts/generate_standard_benchmarks.py:30` → `seed = manifest_seed + idx`。
- `benchmarks/single_u10_cross_tgt15_ep100.json`：100 条，种子 **1250..1349**。
- **`offline_data/` 下现存 9 个数据集，`metadata.json` 里 `seed` 全部 = 0、
  `num_episodes` 全部 = 1000** → 训练种子区间 0..999。

### 因此

对所有 **1000-episode 数据集**：训练 0..999 vs 评估 1250..1349，**不相交，安全**。

对 **`crosscomp-2000`**：该数据集**本机不存在**（`offline_data/` 下只有 `_ep1000` 目录），
无法核。但现存 9 个数据集的 `base_seed` **无一例外都是 0**，若 2000-episode 那次采集沿用
同样的调用，种子区间就是 **0..1999，完整包含评估的 1250..1349** —— 即 100 条评估 episode
逐条出现在训练集里。采集与 manifest 生成调的是同一个 `env.reset(seed=s, ...)`，同种子
⇒ 同 `flow_time`/`start_xy`/`goal_xy`/`initial_heading`，是完全相同的任务实例。

**这正好命中 `paper` Table 3 里 `cross-2000` 那一行，也就是全文最大的那个数字（+32.2pp）。**

### 怎么查（几分钟）

1. `crosscomp-2000` 数据集的 `metadata.json` 里的 `seed` 与 `num_episodes`（数据集若在 Mac 侧）；
2. 或 `docs/td3bc_phase0c_experiment_report.md` 里那次采集的命令行。

若 `base_seed` 是 1000 / 2000 → 不相交，只需在正文写明种子机制即可。
若是 0 → 需换 base_seed 重采并重跑该单元。

### 顺带建议

不管结果如何，都值得跑一次 overlap audit：只调 `env.reset` 不跑仿真，把训练种子区间的
`(flow_time, start_xy, goal_xy, initial_heading)` dump 出来与 manifest 的 100 条做集合比对。
注意 `transitions.npz` 里只有 `obs/actions/.../privileged_obs`，**不存** `flow_time`/`start_xy`，
所以不能直接从数据集比对，必须重放 reset RNG。

---

## 2. `paper` Table 1 的 transitions 数字与本机 metadata 对不上

`paper/sections/setup.tex:89-91`（`tab:dataset_matrix`）写的是：

| Alias | 论文 Transitions | 本机实测 | 比值 |
|---|---|---|---|
| `crosscomp-1000` | $\sim$2.4$\times 10^5$ | **152,683**（= `mean_episode_length` 152.683 × 1000） | 1.57× |
| `worldcomp-1000` | $\sim$2.0$\times 10^5$ | **104,360**（= 104.36 × 1000） | 1.92× |
| `crosscomp-2000` | $\sim$4.8$\times 10^5$ | 无数据集 | — |

`crosscomp-2000` 的 4.8e5 恰好是 2.4e5 的两倍，说明这一列很可能是按 episode 数线性外推的
估算值，而非从 metadata 读出的实测值。

两种可能，**未确认是哪一种**：(a) Table 1 的数字过时；(b) 论文主线用的是**加噪采集变体**
（`docs/td3bc_phase0c_experiment_report.md` 里区分过 deterministic / noisy 单元），而本机只
留了 deterministic 版。任何一种都要改表或在表注里写清口径。

---

## 3. 两个同前缀评估 manifest 并存，种子区间是包含关系

- `benchmarks/single_u10_cross_tgt15.json` —— **30 条**，种子 1250..1279
- `benchmarks/single_u10_cross_tgt15_ep100.json` —— **100 条**，种子 1250..1349

前者的种子区间是后者的**子集**。`docs/fql_succession_bug2_fix_decision.md:27` 记着
`single_u10_cross_tgt15.json` 曾经是 30 ep、导致"所有 in-training eval + final test 都是 30 ep"。

`paper/sections/experiments.tex` §5.1 写的是 val = 40 episodes 选 checkpoint、test = 100
episodes 一次性报告。**40 这个数字对不上上面任何一个文件，未确认它来自哪里。**

需要确认的是：**选 checkpoint 用的那批 episode 与最终报告用的那批是否互斥。**
若 val 用的是 30ep 文件、test 用的是 100ep 文件，则 val ⊂ test，等于在测试集上选模型。
查法：ReBRAC 主线 run 的 `trainer_state.json` 里的 `eval_manifest` 字段。

---

## 4. 一条不算缺陷但会被问到的

`offline_data/worldcomp_..._ep1000/metadata.json` 的采集策略 `success_rate` = **0.958**，
而论文报告 ReBRAC-Q 在 `worldcomp-1000` 上是 **0.928**。两者分布不同（采集在 seeds 0..999
上、评估在冻结的 100-episode manifest 上），**不能直接比**，但数字太接近，容易被问出
"离线策略打得过产生数据的行为策略吗"。

建议：把两个 behaviour policy 在**评估 manifest 上**的成功率跑出来，作为一行加进 Table 3。
纯 eval 开销，且顺带回应"最强的 baseline 其实就是数据采集策略本身"这个必然会有的质疑。
