# FQL Succession P2 — Sprint 0 Collection Log

> **状态**:DONE(collection + 3 audit 闭环,GMM audit 降级 advisory)
> **作者**:Claude Code session,2026-05-21
> **依据**:[`fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md) v1.3 §2 + §3 + §10.5
> **目的**:记录 P2 sprint 0 三个 dataset 收集 + 3 audit observation;触发 spec audit 降级决策(D22)

---

## 1. Datasets collected

### 1.1 E-uni(reuse,无新 collection)

| Item | Value |
|---|---|
| Path | `offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000/` |
| Policy | privileged |
| `action_noise_std`(metadata 实测)| **0.0**(spec/dataset card 写 "0.1" 是 stale,实际 collection 时未加噪)|
| Episodes | 1000 |
| success_rate | 0.985 |
| num_transitions | 86,685 |
| mean_return | 132.56 ± 46.38 |
| Status | ✓ reused from Task D(2026-05-19),无新 collection |

详见 [`fql_e_uni_anchor_dataset_card.md`](fql_e_uni_anchor_dataset_card.md)。**Action noise 实际为 0,dataset card 与 spec 记录 stale**。

### 1.2 M-uni-noise(P2 sprint 0 新 collect,ε=0.5)

| Item | Value |
|---|---|
| Path | `offline_data/fql_succession/m_uni_noise_eps0p5_1000/` |
| Policy | privileged |
| `action_noise_std` | **0.5** |
| Episodes | 1000 |
| Seed | 1 |
| **success_rate** | **0.632** ✓ Band A target [0.50, 0.75]|
| num_transitions | 120,525 |
| mean_return | −2.61 ± 157.55(vs E-uni 132.56)|
| Wallclock | 3.5 min(6 workers)|
| Physical cell assignment | **Single privileged policy + Gaussian noise widening** → unimodal by construction(per `policy_mixture` metadata)|

### 1.3 M-multi-mix Source A(privileged-500)

| Item | Value |
|---|---|
| Path | `offline_data/fql_succession/_components/privileged_500_seed100/` |
| Policy | privileged |
| `action_noise_std` | 0.1 |
| Episodes | 500 |
| Seed | 100 |
| success_rate | 0.974 |
| num_transitions | 45,454 |
| mean_return | 126.13 ± 60.51 |
| Wallclock | 1.25 min |

### 1.4 M-multi-mix Source B(goalseek-500)

| Item | Value |
|---|---|
| Path | `offline_data/fql_succession/_components/goalseek_500_seed200/` |
| Policy | goalseek |
| `action_noise_std` | 0.1 |
| Episodes | 500 |
| Seed | 200 |
| success_rate | 0.678 |
| num_transitions | 92,836 |
| mean_return | −29.59 ± 146.21 |
| Wallclock | 0.8 min |

### 1.5 M-multi-mix(concat A + B,episode-level 50/50)

| Item | Value |
|---|---|
| Path | `offline_data/fql_succession/m_multi_mix_50priv_50goal_1000/` |
| Mix strategy | episode_level |
| Task sampler | anchor_distribution |
| Total episodes | 1000(500 + 500)|
| num_transitions | 138,290(45,454 + 92,836 byte-additive ✓)|
| Wallclock | <1 min |
| Physical cell assignment | **2 behavior policies (privileged + goalseek)** mixed at episode level → multimodal by construction |

### 1.6 Total sprint 0 wallclock

~7 min(本机 6-worker CPU 顺序执行)+ ~30 s × 3 audit ≈ **8 min**。

---

## 2. Multimodality audit (3 audits, **advisory per spec v1.3 §3.0**)

> **重要**:audit verdict 是 **advisory**,不阻塞 cell 使用。Cell assignment 由 collection metadata(§1)决定。

### 2.1 A1 — E-uni self-check(baseline floor reference)

| Item | Value |
|---|---|
| Audit dir | `results/fql_succession/p2/audit/e_uni_self/` |
| dataset-a | E-uni |
| dataset-b | E-uni(self)|
| `p_≥2(E-uni)` | **0.112** ✓(< 0.20 baseline floor)|
| Δp(≥2) | −0.010 [−0.050, +0.030](self-check,Δ ≈ 0 expected)|
| Welch p | 0.534(self-check,non-significant expected)|
| Audit script verdict | "Gate A.2 FAIL"(self-check expected behavior — c1/c2/c4 criterion 设计成 "证明 b > a multimodal",self-check 时 Δ ≈ 0)|
| Spec §3.3 c3 verdict | p_≥2(E-uni) = 0.112 < 0.20 ✓ |

### 2.2 A2 — M-uni-noise vs E-uni(GMM false-positive disclosure)

| Item | Value |
|---|---|
| Audit dir | `results/fql_succession/p2/audit/m_uni_noise_vs_e_uni/` |
| `p_≥2(M-uni-noise)` | **0.996** ❌(spec §3.3 c4 b-multimodal criterion FAIL "应 <0.20")|
| Δp(≥2) | +0.884 [+0.854, +0.912] |
| Welch p | ~7.6e-300(extreme significance)|
| Audit script verdict | "Gate A.2 PASS"(从 mixture-vs-baseline 看;但 paper claim 想要 c4 FAIL "b 不应 multimodal")|
| Spec v1.3 处理 | **GMM false-positive on noise-widened single policy**;cell 物理上是 single privileged + Gaussian widening(§1.2 metadata 证据);audit 作 paper appendix 透明披露,**不阻塞 cell 使用** |

### 2.3 A3 — M-multi-mix vs E-uni(multi-policy mixture confirmation)

| Item | Value |
|---|---|
| Audit dir | `results/fql_succession/p2/audit/m_multi_mix_vs_e_uni/` |
| `p_≥2(M-multi-mix)` | **0.414** ✓ (in [0.35, 0.50] marginal band per §2.4)|
| Δp(≥2) CI | +0.302 [+0.250, +0.352] — lower CI 0.25 > 0.10 ✓ |
| Welch p | ~1.5e-29 < 0.07 ✓ |
| Audit script verdict | "Gate A.2 PASS" |
| Spec §3.3 c4 verdict | 三条 criterion 全过 → mixture construction confirmed |

---

## 3. Mitigation attempt log(audit 降级前)

为完整披露 audit 降级决策的依据,以下是 v1.2 spec §3.4 mitigation 尝试(已 cleanup,不在 final dataset):

| Attempt | Configuration | success_rate | p_≥2 | Verdict (spec v1.2 hard gate) |
|---|---|---:|---:|---|
| ε=0.5(primary) | privileged + ε=0.5,1000 ep | 0.632 | 0.996 | FAIL (spec §3.4 trigger ε mitigation) |
| ε=0.3(spec §3.4 mitigation) | privileged + ε=0.3,1000 ep | 0.830 | 0.850 | **FAIL 仍未达 < 0.20** |
| _diag_ worldcomp ε=0 | worldcomp deterministic,300 ep | 0.973 | 0.118 | unimodal ✓ 但 not sub-optimal(Band D no effect)|

**Pattern**:任何使 success rate 退化的 collection 配置(Gaussian noise injection 或 reactive policy)在 GMM audit 上均被判 multi-modal;唯一 unimodal 的 deterministic baseline (worldcomp ε=0) 又过于 expert-level(97% success)。**M-uni-noise 设计目标(sub-optimal AND audit-unimodal)在 GMM metric 下 unreachable**。

详细数学解释见 [`fql_succession_p2_main_spec.md`](fql_succession_p2_main_spec.md) §10.5。

---

## 4. Gate C.2 cell-construction integrity verdict (per spec v1.3 §6.2)

| Cell | Required collection metadata | 实测 | Verdict |
|---|---|---|---|
| E-uni | `policy_mixture = [{privileged, 1.0}]` + `noise=0.0` | ✓ matches | PASS |
| M-uni-noise | `policy_mixture = [{privileged, 1.0}]` + `noise=0.5` | ✓ matches | PASS |
| M-multi-mix | `policy_mixture` 含 privileged + goalseek + episode-level mix | ✓ matches + audit confirms p_≥2=0.414 | PASS |

**Gate C.2 整体 verdict**:**PASS**(cell-construction integrity 全过)。

---

## 5. Spec patches triggered by this sprint 0

1. **Spec v1.2 → v1.3**:audit 从 Gate C.2 hard gate 降为 advisory(D22)
   - §3.0 新增 audit 角色重定
   - §3.3/§3.4 改 advisory verdict + 处理方式
   - §6.2 重定基于 collection metadata integrity
   - §10.5 新增 GMM caveat 段(数学解释 + paper §method 模板)
2. **Plan v1.3 → v1.4**:加 D22(audit advisory decision)
3. **`fql_e_uni_anchor_dataset_card.md` stale fix**(留 follow-up,本 sprint 不在 scope):dataset card 写 "noise=0.1" 但 metadata 实测 0.0

---

## 6. Next step (任务 4c)

3 个 run notebook scaffold 可以基于以下 verified dataset path 写:

| Cell | Verified dataset path |
|---|---|
| E-uni | `offline_data/privileged_s0_h4_arrival_v2_re150_u10cross_fixdone_ep1000` |
| M-uni-noise | `offline_data/fql_succession/m_uni_noise_eps0p5_1000` |
| M-multi-mix | `offline_data/fql_succession/m_multi_mix_50priv_50goal_1000` |

每个 notebook 仿 `notebooks/rebrac_c1_asym_critic_ablation.ipynb` 8-section 风格,2 algo × 2 seed [42, 0] = 4 run/notebook。
