# FQL E-uni Paper Anchor Dataset — Card

> **Status (2026-05-19)**: COLLECTED — privileged / cross_u10 / s0 / h4 / arrival_v2 / 1000-ep。FQL P0+P1 spec Task D 完成。
>
> **Branch**: `codex-arrival-v2-prototype`
> **Spec**: [`docs/archive/fql_succession/fql_succession_p0p1_spec.md`](archive/fql_succession/fql_succession_p0p1_spec.md) Task D
> **Plan**: [`docs/archive/fql_succession/fql_succession_plan_v0.md`](archive/fql_succession/fql_succession_plan_v0.md) §3 (E-uni anchor cell)
> **Path**: `offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/`
> **Pre-flight**: [`docs/archive/fql_succession/fql_audit_dryrun_report.md`](archive/fql_succession/fql_audit_dryrun_report.md) (Task A — audit tool verified on 200-ep dry-run of same manifold)

---

## 1. Purpose

FQL paper 主线对照实验 N0 cell 的 dataset：privileged behavior policy 在 cross_u10 / s0 / arrival_v2 上的 1000-ep transitions，作为 **E-uni 单模态 expert dataset** 的 anchor。同 setup 下 ReBRAC broad val v2 §2 已报 success **0.850** (2 seed, [42, 0])，FQL Gate B 期望持平或超过此 anchor。

---

## 2. Collection setup

| Item | Value |
|---|---|
| Policy | privileged (`PrivilegedCorridorPolicy`, hull-integral `[u_eq, v_eq]` driven) |
| Flow ROI | `wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` (U=1.0 m/s, Re=150, T=1200 frames) |
| Probe layout | s0 (单点 DVL @ (0,0)) |
| History length | 4 |
| Task geometry | cross_stream |
| Target speed | 1.5 m/s |
| Reward objective | arrival_v2 |
| Episodes | 1000 |
| Seed | 0 |
| Num workers | 6 (本机 M-series 10-core) |
| Output dir | `offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/` |

**Reproducible 命令**：
```bash
PATH="/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin:$PATH" \
python -m scripts.collect_offline_data \
    --policy privileged \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 --history-length 4 \
    --task-geometry cross_stream --target-speed 1.5 \
    --objective arrival_v2 \
    --episodes 1000 --seed 0 --num-workers 6 \
    --output-dir offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000
```

---

## 3. Dataset statistics

### 3.1 顶层

| Metric | Value |
|---|---:|
| num_episodes | 1,000 |
| num_transitions | 86,685 |
| **success_rate (goal_rate)** | **0.9850** (985 / 1000) |
| out_of_bounds_rate | 0.0150 (15 / 1000) |
| timeout_rate | 0.0000 |
| other_terminal_rate | 0.0000 |
| **mean_return** | **132.555** ± 46.378 |
| mean_episode_length | 86.69 steps |
| Wall-clock | **153.5s** (2.56 min, 6 workers) |
| File size | `transitions.npz` 9.5 MB + `metadata.json` 1.8 KB |

### 3.2 Cross-check vs Task A 200-ep dry-run

| Metric | 200-ep (seed=0) | 1000-ep (seed=0) | Δ |
|---|---:|---:|---:|
| success_rate | 0.9850 | 0.9850 | 0.00 |
| mean_return | 133.17 ± 45.92 | 132.56 ± 46.38 | −0.61 |
| OOB / 总 | 3 / 200 (1.5%) | 15 / 1000 (1.5%) | 同 |
| mean_episode_length | 86.14 | 86.69 | +0.55 |

200-ep dry-run 的 statistics 完美外推到 1000-ep — 同一 seed=0 + 同 collection 路径，前 200 个 episode 应严格相同，后 800 个 episode 在 stochastic flow time / spawn / heading 采样后保持相同分布 → 这是 collection 路径 deterministic 且 task sampler 行为稳定的强 evidence。

### 3.3 Arrays (transitions.npz)

| Key | Shape | dtype | 用途 |
|---|---|---|---|
| `obs` | (86685, 48) | float32 | 8-base × 4 history (32) + s0_probe × 2-D × 4 history (8) + 8 episode-context channels |
| `actions` | (86685, 2) | float32 | (heading_cmd, speed_cmd) ∈ [−1, 1] |
| `next_obs` | (86685, 48) | float32 | |
| `next_actions` | (86685, 2) | float32 | (placeholder for SARSA-style algos) |
| `rewards` | (86685,) | float32 | arrival_v2 reward |
| `costs` | (86685,) | float32 | safety cost (per-step) |
| `dones` | (86685,) | float32 | terminated OR truncated |
| `terminateds` | (86685,) | float32 | true terminal only |
| `truncateds` | (86685,) | float32 | timeout only |
| `terminal_reason_codes` | (86685,) | int8 | 0 running / 1 goal / 2 timeout / 3 OOB |
| `behavior_policy_codes` | (86685,) | int8 | 0 crosscomp / 1 goalseek / 2 privileged / 3 worldcomp |
| `privileged_obs` | (86685, 2) | float32 | body-frame `[u_eq, v_eq]` from `EquivalentCurrentModel` |
| `next_privileged_obs` | (86685, 2) | float32 | |

obs_dim=48 = base_8 × 4 history + s0_2D × 4 history + episode-context_8 (`include_episode_context_obs=true` in metadata).

### 3.4 Action 分布

| Dim | mean | std | min | max | 备注 |
|---|---:|---:|---:|---:|---|
| heading_cmd | −0.017 | 0.790 | −1.000 | +1.000 | 活跃控制维度，覆盖全 action space |
| speed_cmd | +0.760 | **0.000** | +0.760 | +0.760 | **常量** — privileged policy 在 cross_u10 sub-critical regime 下使用固定目标速度 |

**注（FQL audit implication）**：speed_cmd 在 privileged policy 下退化为常量；GMM mode-count audit 实际工作在 (heading, speed) 2-D 但 speed dim 完全 degenerate，等价于 1-D heading 的 multimodality 检测。Task A audit Gate A.2 PASS 与此 fact 一致（heading 维度的 unimodal vs multimodal 区别被 GMM 捕捉，Δp(≥2) +0.58 [CI +0.53, +0.63]）。日后 paper §discussion 若 reviewer 关心 audit metric 是否被 2-D 假设 inflate，可明标 "audit run on (heading, speed) but speed is policy-determined constant in our setup, so the multimodality detected is effectively on heading distribution"。

### 3.5 privileged_obs 分布

| Dim | mean | std | 物理含义 |
|---|---:|---:|---|
| u_eq | −0.523 | 0.412 | 沿 hull 积分的等价 surge flow（body frame），cross_stream 任务下负值 = 逆 surge 分量 dominant |
| v_eq | +0.016 | 0.770 | 沿 hull 积分的等价 sway flow（body frame），mean ≈ 0 + std 0.77 = lateral wake 信号在 cross_u10 上活跃 |

这两维是 `AsymCritic` 在 FQL 训练时通过 `--use-asymmetric-critic` flag 注入 critic 的 privileged input（actor 部分仍只看 obs 48-D / s0 single-point sample）。

---

## 4. ReBRAC anchor comparison

| Reference | Setup | success (eval) | source |
|---|---|---:|---|
| ReBRAC main paper N0 (efficiency_v2) | crosscomp / s0 / cross_u10 / efficiency_v2 / 5-seed | **0.902 ± 0.021** | [`rebrac_experiment_report.md`](rebrac_experiment_report.md) rev.8 |
| ReBRAC broad val v2 N0 (arrival_v2, this manifold) | crosscomp / s0 / cross_u10 / arrival_v2 / 2-seed [42, 0] | **0.850 ± 0.024** | [`rebrac_broad_validation_v2_report.md`](rebrac_broad_validation_v2_report.md) §2 |
| **Behavior policy on dataset** | privileged / cross_u10 / arrival_v2 / seed=0 | **0.985** | this dataset |

> ⚠ **追注（2026-08-16）**：表中 v2 N0 的 `0.850 ± 0.024` 是**当时**的两种子值。该 anchor 已于
> 2026-07-12 由 seed 43 supplement 更新为 **0.878 ± 0.051（3 seed {42, 0, 43}）**，且首轮「两种子同向退化」
> 的读法已在 report §2.4 撤销。**此处保留原值不改**——下方 Gate B 门槛是按当时 anchor 预登记的，
> 改数字等于篡改预登记判据。引用当前 anchor 请用 0.878。

**FQL Gate B target**：在 E-uni 1000-ep 上跑 64-epoch FQL 1-seed，eval success ≥ ReBRAC broad val v2 N0 anchor (≈ 0.85) — 即"FQL 在 sub-critical / s0 anchor 上至少 holds ReBRAC parity"。若 FQL 显著高于 ReBRAC（e.g. ≥ 0.90），则解锁 paper claim "FQL beats ReBRAC on uni-modal expert" 的 anchor evidence。

**注意 reward 差异**：
- Behavior policy 0.985 是**专家自身收集时的 success**（不是 deployment eval），意义是「这份 dataset 中 episode 平均朝目标走且无 closed-loop 失稳」
- ReBRAC 0.850 是**学到的 policy 在 deployment-realistic eval manifest 上的 success**，因此 0.985 是 FQL 学习上限的 ceiling reference，不是直接对照目标

---

## 5. Storage & sharing

- 完整数据 9.5 MB (compressed npz) + 1.8 KB metadata — 已在 `offline_data/` 目录，gitignored
- **同步到 Drive (FQL training in Colab)** 仍 TBD — 本机 Task E 完成后再考虑（本机若能完整跑通 FQL 64-epoch，则 Colab sync 不紧迫）
- 可被 `python -m scripts.train_offline --algo fql --offline-data offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000` 直接消费

---

## 6. Implication for Task E (FQL Gate B sanity)

| Item | Status |
|---|---|
| Dataset 完整性 | ✓ 13 arrays，shape / dtype / 字段全部正确 |
| Reward sanity | ✓ behavior success 0.985 + mean_return 132.6 — arrival_v2 reward 在数据上正常 |
| privileged_obs 字段 | ✓ AsymCritic 训练所需 `privileged_obs` + `next_privileged_obs` 已携带，可直接 enable `--use-asymmetric-critic` |
| Cross-validation with 200-ep dry-run | ✓ statistics 一致 |
| **Ready for FQL Gate B** | **✓** |

**Task E 推荐 CLI**（待 FQL succession plan §3 / spec §Task E 确认最终 hyperparam）：
```bash
PATH="/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin:$PATH" \
python -m scripts.train_offline \
    --algo fql \
    --offline-data offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000 \
    --eval-manifest benchmarks/single_u10_cross_tgt15.json \
    --num-epochs 64 --batch-size 256 \
    --seed 0 \
    --save-dir checkpoints/offline/fql/gate_b_sanity/seed_0
```

---

**Card 起草**: 2026-05-19
**Reproducibility**: 命令在 §2；统计在 §3
