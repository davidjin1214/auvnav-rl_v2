# ReBRAC Phase 1 deployable vs TD3BC privileged-critic — Statistical test follow-up

> 文档版本：rev.1（由 `notebooks/rebrac_paper_followup.ipynb` §4 自动生成）
> 配套：[docs/rebrac_mainline_review.md §2.2.3 / §3.1.C](./rebrac_mainline_review.md)
> 用途：paper drafting 时的 main results 表注脚 / discussion 引用。

## 1. Sample-level 数据

| Protocol | 来源 | n_seeds | n_eps/seed | seed-level mean | seed-level std |
|---|---|---:|---:|---:|---:|
| ReBRAC deployable Phase 1 | `results/offline/rebrac/worldcomp_teacher_gap/deployable/.../actorb_4p0__criticb_2p0/test/seed_*.json` | 5 | 100 | 0.9280 | 0.0858 |
| TD3BC privileged-critic | `results/offline/td3bc/phase0c/worldcomp_teacher_gap/privileged_final/.../test_selected/alpha_0p1/seed_*.json` | 5 | 100 | 0.9220 | 0.0958 |

Per-seed success rate（成功率 / 100 episodes）：

| seed | ReBRAC dep | TD3BC priv | Δ |
|---|---:|---:|---:|
| 42 | 0.990 | 0.970 | +0.020 |
| 43 | 0.930 | 0.980 | -0.050 |
| 44 | 0.780 | 0.910 | -0.130 |
| 45 | 0.980 | 0.990 | -0.010 |
| 46 | 0.960 | 0.760 | +0.200 |

## 2. Paired episode-level bootstrap（10000 resamples）

按 100 个 episode_id 重抽样（跨 5 seeds 取每个 episode 的成功率均值，再 bootstrap）：

- point estimate Δ = **+0.0060**
- 95% CI on Δ = [-0.0300, +0.0420]
- 99% CI on Δ = [-0.0420, +0.0520]
- bootstrap p (two-sided, H0: Δ=0) ≈ **0.7762**

## 3. Welch's t-test（5 vs 5 seed-level means）

- t = **0.104**
- p (two-sided) = **0.9195**
- 结论：**fail to reject H0（持平）**

## 4. Gap closure 95% CI（seed-level bootstrap）

deployable→teacher gap closure 定义 `(method - TD3BC_dep) / (teacher - TD3BC_dep)`，TD3BC_dep = 0.858, teacher = 0.99：

- ReBRAC dep gap closure point = **53.0%**
- TD3BC priv gap closure point = **48.5%**
- Δ point = +4.5pp
- 95% CI on Δ = [-62.12pp, +86.36pp]

## 5. Paper 写作建议（堵 review §2.2.3）

- main text 的 ReBRAC dep vs TD3BC priv 结论改为 **"持平（statistically not different）"**，附 Welch's p-value；
- main results 表注脚写入 paired bootstrap CI 与 t-test p-value；
- gap closure +4.5pp 仅在 discussion 中作为 directional 报告，不在 abstract / conclusion 中作为强 claim；
- 5 seeds 是 RL benchmark 的常见上限，但应在 limitations 显式声明 underpowered。
