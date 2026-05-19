# FQL Audit Multimodality — Task A Dry-Run Report

> **Status (2026-05-19)**: PASS — `scripts/audit_multimodality.py` 在真实小数据集上 Gate A.2 verdict 工作正常，4/4 criteria 全部通过。
>
> **Branch**: `codex-arrival-v2-prototype`
> **Spec**: [`docs/fql_succession_p0p1_spec.md`](fql_succession_p0p1_spec.md) Task A
> **Design**: [`docs/fql_audit_multimodality_design.md`](fql_audit_multimodality_design.md)
> **Raw outputs**: `results/audit_dryrun_2026-05-19/{audit_summary.json, mode_count_per_anchor.csv, mode_count_distribution.png}`
> **Data**: `offline_data/audit_dryrun_2026-05-19/{e_uni_priv_200, m_multi_mix_200}/`

---

## 1. Purpose

P0+P1 spec Task A：用真实小数据集（200-ep × 2）dry-run audit script，验证 Gate A.2 verdict 在 cross_u10 / s0 / arrival_v2 manifold 上能区分 unimodal vs multimodal action distribution，这是 FQL paper 主线 (E-uni vs M-multi-mix on N0 anchor) audit 工具的预飞行。

**不是**正式实验：episode 数量 (200) 远低于 paper anchor (1000)；目的纯粹是验证：(a) audit CLI 在真实流形上不崩；(b) verdict logic 给出预期方向（E-uni p_ge_2 低 / M-multi-mix p_ge_2 高）；(c) joblib 并行 GMM fit 稳定且可在 CPU 单机完成。

---

## 2. Dataset 合成

### 2.1 E-uni (uni-modal baseline)

| Item | Value |
|---|---|
| Policy | privileged (单一) |
| Manifest setup | cross_u10 / s0 / history=4 / target_speed=1.5 |
| Reward objective | arrival_v2 |
| Flow | `wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` (U=1.0, Re=150) |
| Episodes | 200 |
| Seed | 0 |
| **success_rate** | **0.985** (197 goal / 3 OOB) |
| **transitions** | 17,229 |
| Mean return | 133.17 ± 45.92 |
| Wall-clock collect | 31.3s (6 workers) |
| obs_dim | 48 |
| action_dim | 2 |

### 2.2 M-multi-mix (multi-modal target)

通过 `scripts/concat_offline_datasets.py` episode-level 合三个 component：

| Component | Policy | Episodes | success | mean_return | seed | transitions |
|---|---|---:|---:|---:|---:|---:|
| 1 | privileged | 67 | 0.970 | 128.29 | 100 | 5,568 |
| 2 | worldcomp | 67 | 1.000 | 131.24 | 200 | 7,280 |
| 3 | crosscomp | 66 | 0.818 | 50.87 | 300 | 9,657 |
| **concat** | mix | **200** | **0.930** | **103.73** | — | **22,505** |

三个 component 在 cross_u10 同一 manifold 上的 success / return 差异（97% / 100% / 82%）已经说明三者在相同 state 下会 emit visibly different actions — 这就是 audit script 应该能 detect 的 multimodality footprint。

---

## 3. Audit 结果

### 3.1 命令

```bash
python -m scripts.audit_multimodality \
    --dataset-a offline_data/audit_dryrun_2026-05-19/e_uni_priv_200 \
    --dataset-b offline_data/audit_dryrun_2026-05-19/m_multi_mix_200 \
    --output-dir results/audit_dryrun_2026-05-19 \
    --label-a "E-uni" --label-b "M-multi-mix" \
    --seed 0
```

默认参数：knn_k=50, gmm_max_components=3, gmm_n_init=3, mode_weight_floor=0.10, n_anchor_states=500, n_bootstrap=1000, n_jobs=-1 (全核)。

### 3.2 Mode distribution

| Dataset | p_1 | p_2 | p_3 | **p_ge_2** | mean mode count | GMM failures |
|---|---:|---:|---:|---:|---:|---:|
| E-uni (privileged) | 0.812 | 0.158 | 0.030 | **0.188** | 1.22 ± 0.48 | 0 / 500 |
| M-multi-mix | 0.230 | 0.488 | 0.282 | **0.770** | 2.05 ± 0.71 | 0 / 500 |

E-uni 81% 的 anchor 是单模态，符合「单一 deterministic policy → 同 state 下 action 高度集中」的预期；M-multi-mix 77% 的 anchor 是多模态（mean ≈ 2.05 mode），符合「三个 policy 混合 → 同 state 下 emit 不同 action cluster」的预期。

### 3.3 Bootstrap CI on Δp(≥2)

| Quantity | Value |
|---|---:|
| Δp(≥2) mean (B − A) | +0.582 |
| 95% CI | [+0.534, +0.632] |
| Δ std | 0.0255 |
| CI width | 0.098 (very tight; N=200ep × 2 已足够给出 ±5pp 精度) |

### 3.4 Welch's t (one-sided)

| Quantity | Value |
|---|---:|
| t-statistic | 21.66 |
| p-value (one-sided, H1: B > A) | 6.4e-84 |

### 3.5 Gate A.2 Verdict

| Criterion | Value | Threshold | Pass |
|---|---:|---|:---:|
| c1: Δp(≥2) CI lower bound > 0.10 | 0.534 | > 0.10 | ✓ |
| c2: Welch p < 0.07 (one-sided) | 6.4e-84 | < 0.07 | ✓ |
| c3: E-uni p(≥2) < 0.20 | 0.188 | < 0.20 | ✓ |
| c4: M-multi-mix p(≥2) > 0.30 | 0.770 | > 0.30 | ✓ |
| **overall** | | | **PASS** |

CLI exit code = 0；stdout 末行 `[audit] Gate A.2 PASS`。

### 3.6 Runtime

| Stage | Wall-clock |
|---|---:|
| audit core (joblib parallel, n_jobs=-1) | 4.1s |
| total (incl. CLI startup + sklearn import + plotting) | 6.1s |
| effective CPU utilization | 499% (5 cores effective on a 10-core M-series) |

GMM fits over 1000 anchors (500 per dataset) 全部成功 (failures=0/500 在每个数据集) — sklearn 1.8 GaussianMixture solver 在 (knn_k=50, action_dim=2) 上完全稳定。

---

## 4. 工具发现

### 4.1 Verified 行为

- CLI signature 与 spec 一致：`--dataset-a`, `--dataset-b`, `--output-dir`, `--label-a/b`, `--seed`, `--n-jobs`
- Verdict 退出码：PASS → 0；FAIL → 2；error → 1（本次 PASS，未触发 FAIL/error path）
- Output 文件 3 个：`audit_summary.json` (config + metadata + audit_a/b + bootstrap + welch + verdict) + `mode_count_per_anchor.csv` (per-anchor 行) + `mode_count_distribution.png` (2-panel plot)
- joblib `n_jobs=-1` 默认在本机 10-core 上 effective 5x speedup
- 二个 dataset 不要求相同 episode 数（E-uni 200ep / M-mix 200ep 一致；但 audit 内部按 anchor 数 sampling，不依赖 episode 平衡）

### 4.2 小观察（非 bug，不阻塞）

**concat metadata 的 policy field 仅保留首 component 的标签**：
- M-multi-mix 的 `metadata.json` 顶层 `policy` 字段 = "privileged+worldcomp+crosscomp"（拼接），但 `policy_mixture` 仍 = `[{policy: privileged, weight: 1.0}]`（继承自 first input）
- 真实组成在 `mix_components` 字段下完整保留（policy + episodes + success + return + seed + source_dir × 3）
- 不影响 audit 工作（audit 只读 `obs`/`actions` array）
- **若日后 paper review 需要从 concat metadata 直接 trace policy mixture**，建议 `scripts/concat_offline_datasets.py` 把 `policy_mixture` 也合并（按 episode-weighted average）；目前 mix_components 已足够 traceable

### 4.3 Spec 待办（保留）

P0+P1 spec §8 列出的 audit 后续验证项中，本次 dry-run 覆盖：
- ✓ CLI run-through on 真实小数据
- ✓ Gate A.2 PASS path（4 criteria 全过）
- ✓ joblib 并行稳定性
- ✓ Output 文件完整性

未在本次 dry-run 覆盖（**留待 paper anchor 1000-ep audit 闭环**）：
- FAIL path（需要 verdict 中某个 criterion 失败的数据）
- 极端 anchor sparsity（knn_k=50 但某些 state 邻居 < 50 → 应有 fallback）
- 多 seed determinism 验证（unit test `test_run_audit_deterministic` 已覆盖；CLI 路径未单独 verify）

---

## 5. 结论与后续

**结论**：`scripts/audit_multimodality.py` 在 cross_u10 / s0 / arrival_v2 真实流形上 Gate A.2 verdict 工作正常；E-uni 与 M-multi-mix 的 mode-count distribution 完全符合 design doc §4 的物理预期；joblib 并行稳定。**Task A 完成**，audit 工具在 FQL P0+P1 paper anchor (1000-ep) 实验中可信。

**对 Task D / Task E 的 implication**：
- **Task D** (E-uni 1000-ep 收集)：本次 200-ep dry-run success 98.5% — 1000-ep 上预计仍 ≥ 95%，paper anchor 收集 risk-free。Wall-clock 外推：200ep / 31s → 1000ep ≈ 2.6 分钟（6 workers），完全可在本机
- **Task E** (FQL Gate B sanity)：Task D 数据集就位即可启动；audit 工具已 ready，可在 Gate B 闭环后立刻补 N0 1000-ep audit verdict

**下一步建议**：直接启动 Task D（E-uni 1000-ep paper anchor 收集），完成后并行启动 Task E (FQL Gate B sanity) + N0 1000-ep audit (paper appendix evidence)。

---

**Report 起草**: 2026-05-19
**Audit version**: `scripts/audit_multimodality.py` (commit `a259489`, post-/simplify)
**Reproducibility**:
```
PATH="/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin:$PATH" \
python -m scripts.audit_multimodality \
    --dataset-a offline_data/audit_dryrun_2026-05-19/e_uni_priv_200 \
    --dataset-b offline_data/audit_dryrun_2026-05-19/m_multi_mix_200 \
    --output-dir results/audit_dryrun_2026-05-19 \
    --label-a "E-uni" --label-b "M-multi-mix" --seed 0
```
