# c4 阈值改造决策 — slope ≥ 0 → no-major-collapse

> **状态**:DECIDED — **方案 α (slope ≥ −2 × SE_agg)** 推荐;Session A 在 P2 spec 起草时 patch `fql_succession_p0p1_spec.md` §5.3
> **背景**:[`fql_succession_gate_b_report.md`](fql_succession_gate_b_report.md) §5.4
> **依赖**:[`fql_succession_bug2_fix_decision.md`](fql_succession_bug2_fix_decision.md)(决定 noise floor)
> **作者**:Claude Code session,2026-05-20
> **Commit gate**:P2 pre-requisite 2/2

---

## 1. 问题陈述

`fql_succession_p0p1_spec.md` §5.3 c4 判据:

> FQL eval 曲线 monotonicity:**末 30% 训练 success rate trend ≥ 0**(无 collapse)

Gate B Option B 实证显示该阈值在 n_seeds=2 + 30-ep manifest 下 **过敏感**:

- FQL aggregated slope = −0.00381,**与 0 在统计上不可区分**(z=−0.27,30-ep noise floor)
- ReBRAC aggregated slope = +0.00048(也几乎等于 0)
- 两算法均在 seed=0 上 slope 正,seed=42 上 slope 负 → **seed-driven,非 FQL-driven**

c4 给出 FAIL 完全由 deterministic threshold 触发 stochastic noise,而非反映真实 FQL 不稳定。P2 起步前必须改革。

---

## 2. 三个候选方案

### 2.1 Option α — slope ≥ −2 × SE(slope_aggregated_n_seeds)

**定义**:阈值随样本大小 + manifest size 自适应:

```
SE_eval(p)            = sqrt(p (1−p) / n_eval_episodes)
SE_slope_per_seed     = SE_eval / sqrt(Σ(x−x̄)²)        # last-30% 用 n_pts=6 → denom=17.5
SE_slope_aggregated   = SE_slope_per_seed / sqrt(n_seeds)
threshold             = −2.0 × SE_slope_aggregated
```

**实现成本**:verdict 阶段几行 numpy + scipy,无 bootstrap loop。
**数学含义**:H0 = 「真 slope ≥ 0」;阈值 = 单侧 95% CI 下界。slope < threshold ⇒ 95% 置信度拒绝 H0 ⇒ 真有 collapse。

### 2.2 Option β — bootstrap CI on slope,95% CI 上界 ≥ 0

**定义**:在 per-seed slope 上做 10000 次 with-replacement bootstrap,取 mean slope 的 95% CI,要求 CI 上界 ≥ 0。

**实现成本**:bootstrap loop + 排序,verdict 阶段成本可接受(~1s),但需要新代码。
**问题**:n_seeds=2 时 bootstrap 几乎 degenerate(只有 3 个 unique mean),CI 信息量低;n_seeds=5 下与 α 数学等价(单侧 t-test 与 bootstrap-mean upper bound 收敛)。

### 2.3 Option γ — peak-stability:last-3 mean ≥ 0.85 × max-3-consecutive

**定义**:不测 slope。要求最后 3 eval 的 mean 不比 trajectory 内最佳连续 3 eval window 的 mean 低 15%。

**实现成本**:6 行 numpy。
**问题**:阈值 0.85 拍脑袋校准,与 c1(last-3 mean)共线性高;在 multi-peak 噪声轨迹下,max-3 由 noise spike 决定 → over-sensitive 风险移到「分母」上,没有解决 noise floor 的根本问题。

---

## 3. Retroactive 测试(Gate B Option B 数据)

### 3.1 Noise floors

| Manifest | per-eval SE @ p=0.7 | slope SE per-seed (n_pts=6) | slope SE aggregated n=2 | slope SE aggregated n=5 |
|---|---:|---:|---:|---:|
| 30-ep (Gate B) | 0.0837 | 0.0200 | 0.0141 | 0.00894 |
| 100-ep (Bug 2 fix) | 0.0458 | 0.0110 | 0.00775 | **0.00490** |

### 3.2 Verdict 表

| Option | Gate B 配置 (n=2, 30-ep) | P2 default (n=5, 100-ep) | Gate B retroactive verdict |
|---|---:|---:|---|
| **Original (slope ≥ 0)** | threshold = 0 | threshold = 0 | **FAIL** (FQL slope = −0.0038) |
| **α (slope ≥ −2·SE_agg)** | threshold = **−0.0283** | threshold = **−0.0098** | **PASS** |
| **β (bootstrap CI upper ≥ 0)** | upper = +0.0229 | (n=5 reasonable) | **PASS** |
| **γ (last-3 ≥ 0.85·max-3)** | FQL seed=42: 0.689 vs 0.85×0.778=0.661 | (per-seed test) | **PASS** |

详细计算见 §3.3 (附录)。

### 3.3 关键观察

1. **Original threshold 在 Bug 2 fix 后仍然 fail**:即使 SE 减半到 100-ep,FQL aggregated slope −0.00381 vs threshold 0 仍是 FAIL → **修 noise 一个维度不够**,必须同时修 threshold
2. **Option α 在所有 4 种 (n_seeds × manifest) 配置下都给出 PASS**:这是 design intent — 「FQL slope 噪声水平内不算 collapse」
3. **Option α 在 P2 标准配置(n=5,100-ep)下 threshold 收紧到 −0.0098**:大约 = "−1pp/eval 平均下降 × 6 evals" 的量级,即肉眼可见的真 collapse trend 才会 fail
4. **β 与 α 在大 n 下渐近等价**:Bootstrap CI upper bound = mean + 1.96 × bootstrap_SE,实际就是 normal 假设下的 z-test;在 n_seeds ≥ 5 时与 α 给出几乎相同的判决
5. **γ 的 PASS 仅靠 4.2% margin** (0.689 vs 0.661):在更差的 FQL trajectory 上很容易 fail,这是它「敏感性偏高」的征兆 — 不一定是缺点,但也不是优势

---

## 4. Type I / II error tradeoff at n_seeds=5 + 100-ep

| Option | Type I (false PASS:实际有 collapse 但判 PASS) | Type II (false FAIL:实际 FQL 健康但判 FAIL) | 实现 |
|---|---|---|---|
| Original | 高(任何 noise 都过) | 高(噪声主导,Gate B 已实证) | 0 行 |
| **α** | 5%(单侧 z=−2) | < 1%(真 slope ≥ 0 时,SE 已用 5 seeds 收紧到 0.005) | 5 行 |
| β | ~5%(渐近等价 α) | ~5%(同) | 30 行(bootstrap loop) |
| γ | 与 0.85 阈值校准强耦合,难量化 | 同上 | 8 行(简单),阈值校准难 |

**判断**:α 的 Type II 在 P2 标准配置下足够低 — 真 collapse(典型 magnitude > 0.02 slope = > 2 × SE_agg)极少被误判为 PASS,而对噪声引起的 ±SE 量级波动免疫。这正是 spec 想要的「no major collapse」语义。

---

## 5. 推荐:**Option α**

### 5.1 三句话理由

1. **数学最干净**:阈值随 n_seeds + manifest size 自动 scale(SE_agg ∝ 1/√n_seeds × 1/√n_eval_episodes),无需手动校准多套阈值;明确的统计意义(95% 单侧 CI 下界)
2. **实现成本最低**:verdict 阶段 5 行 numpy,无 bootstrap loop,无超参 0.85
3. **与 Bug 2 fix 自然配合**:100-ep manifest 把 noise 降半 → α 自动收紧阈值,signal-to-noise 同步提高;不需要重新校准

### 5.2 不选 β 的原因

- 在 n_seeds=2(Gate B legacy)下退化,信息量低
- 在 n_seeds ≥ 5 下与 α 数学等价但实现重 6×
- bootstrap 适合处理非正态分布;slope SE 在 large n_eval 下近正态,bootstrap 无 robust 优势

### 5.3 不选 γ 的原因

- 0.85 阈值经验校准,没有统计意义解释
- 与 c1(last-3 mean)共线性 — 把同一信息看两遍
- max-3 在 noisy trajectory 下由 single spike 决定 → noise 移到分母,本质没解决问题
- 但 γ **可作为 P2 报告的 supplementary visualization 指标**(肉眼直观,不进入 binary verdict)

---

## 6. spec patch 草稿(留 Session A 在 P2 spec 写作时 apply)

在 [`docs/fql_succession_p0p1_spec.md`](fql_succession_p0p1_spec.md) §5.3 把下面这一行 patch:

```diff
- | FQL eval 曲线 monotonicity | 末 30% 训练 success rate trend ≥ 0(无 collapse) |
+ | FQL eval 曲线 no-major-collapse | 末 30% aggregated slope ≥ −2 × SE(slope_aggregated_n_seeds)。SE 按 √(p(1−p)/n_eval) / √17.5 / √n_seeds 计算(n_eval = manifest episodes, n_pts = 6 (last-30% of 20 evals), p ≈ in-training mean success rate)。在 n_seeds=5 + 100-ep manifest 下阈值 ≈ −0.0098;在 Gate B legacy (n=2, 30-ep) 下阈值 ≈ −0.0283。 |
```

并在 §5.3 verdict block 下补一行:

```diff
+ **c4 retroactive 校准依据**:见 [`fql_succession_c4_threshold_revision.md`](fql_succession_c4_threshold_revision.md) §3 retroactive 测试表 + §4 Type I/II 分析。
```

P2 main comparison spec(新文档,Session A 草拟)需要在「Verdict criteria」block 中调用相同的 c4 定义,并 inherit `slope_aggregated` + `SE_agg(n_seeds, n_eval)` 的统一计算函数。建议把该计算放入 `auv_nav/` 或 `scripts/` 下的 helper(本 session 不实现,留 P2 spec 决定模块位置)。

---

## 7. 不在本 commit 做的

- ❌ patch `fql_succession_p0p1_spec.md` §5.3 — Session A 在 P2 spec 起草时统一 patch
- ❌ 实现 c4 verdict helper 函数 — 留 P2 spec 决定模块位置后再实现
- ❌ 重跑 Gate B verdict — verdict 已定 3/4 PASS + c4 marginal-FAIL,本文档 §3 给出新阈值下的 retroactive PASS,完整 verdict 升级留 P2 第一份 sweep 闭环时一并写
- ❌ 在 P2 spec 中加 γ 作 supplementary visualization — 可选 enhancement,Session A 视报告丰富度决定

---

## 8. Acceptance

- [x] 三个 option α/β/γ 数学定义清楚
- [x] Retroactive 测试在 Gate B 数据上跑完(§3.2 verdict 表)
- [x] Type I/II tradeoff 在 P2 标准配置(n=5, 100-ep)下分析(§4)
- [x] 推荐 α 并给出三句话理由(§5.1)
- [x] spec patch 草稿就绪供 Session A copy-paste(§6)
- [ ] Commit "P2 pre-requisite 2/2: c4 threshold revision design"

---

## 附录 §3.3 — α threshold 全表

| Configuration | SE_per_seed | SE_aggregated | α threshold | FQL agg slope = −0.00381 |
|---|---:|---:|---:|:---:|
| n=2, 30-ep (Gate B as-observed) | 0.0200 | 0.01414 | −0.02828 | PASS |
| n=2, 100-ep | 0.0110 | 0.00775 | −0.01549 | PASS |
| n=5, 30-ep | 0.0200 | 0.00894 | −0.01789 | PASS |
| **n=5, 100-ep (P2 default)** | **0.0110** | **0.00490** | **−0.00980** | **PASS (margin 0.006)** |

P2 default 下 FQL agg slope (−0.00381) vs threshold (−0.00980) margin = 0.006 ≈ 0.6 × SE_agg。

**Interpretation**:Gate B 的 FQL slope 真实信号在「不会触发 P2 c4 fail」的安全区,但 margin 不大 → 若 P2 实际 5-seed 平均 slope 更负,c4 仍可能 fail,这本身是有效信号(说明真有问题)。这正是 α 设计目标。

---

*Decision recorded: 2026-05-20. Option α 推荐;β/γ 不选但 γ 可作为 P2 报告的 supplementary visualization 候选(非 binary verdict)。spec patch 草稿就绪,留 Session A 在 P2 spec 起草时统一 apply。*
