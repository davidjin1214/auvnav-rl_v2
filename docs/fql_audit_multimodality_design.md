# `scripts/audit_multimodality.py` — Detail Design Doc

> **文档版本**：v1.0（2026-05-19）
> **作用**：把 [`fql_succession_p0p1_spec.md`](fql_succession_p0p1_spec.md) §1 Task A audit 步骤拆成算法、CLI、判定逻辑、产出 schema 可实现的契约。
> **状态**：**Active design**，待 executor 按此 doc 实现。
> **范围**：仅 `scripts/audit_multimodality.py` 内的算法 + CLI + 产出；不涉及数据收集（在 P0+P1 spec §1.2 内已覆盖）。
> **依赖约束**：本脚本是**分离 dev-time 工具**，允许使用 `scikit-learn`（仓库主依赖未引入）；implementer 在脚本顶部加 import guard 并在 README 注明 `pip install scikit-learn`。**不**把 sklearn 加进主 training requirements。

---

## 目录

1. [Goal + Gate A.2 角色](#1-goal--gate-a2-角色)
2. [CLI 契约](#2-cli-契约)
3. [算法 step-by-step](#3-算法-step-by-step)
4. [Gate A.2 verdict 逻辑](#4-gate-a2-verdict-逻辑)
5. [产出 schema](#5-产出-schema)
6. [Dependency 处理](#6-dependency-处理)
7. [边界条件与异常](#7-边界条件与异常)
8. [Implementation checklist](#8-implementation-checklist)
9. [Test plan（pytest）](#9-test-plan)
10. [Acceptance criteria](#10-acceptance-criteria)

---

## 1. Goal + Gate A.2 角色

### 1.1 一句话目标

> 给定两份 offline dataset (一份 unimodal candidate，一份 multimodal candidate)，量化 **action distribution conditional on similar states** 在两份 dataset 上的多模态强度差异；输出 paired bootstrap CI + Welch's t verdict + 可视化，作为 Gate A.2 通过/失败的判据。

### 1.2 为什么是 k-NN action GMM

- 直接对全局 action 分布跑 GMM 会被 state 分布污染（多 collector 可能去不同 state region，看似多峰其实是 state 异构）
- 真正想问：**"在状态相似的条件下，多个 collector 选不同动作"** —— 即 conditional action distribution 的多模态
- k-NN 在 obs 空间找邻居 ≈ 条件化 anchor state；邻居的 action 集合分布即 conditional p(a | s ≈ s_anchor)
- 对该条件分布跑 GMM with BIC mode selection → mode count

### 1.3 Plan v1 决策锚定

- **D11**：4 个 audit 指标砍到 1 个（仅 k-NN GMM mode count）；本 doc 不实现 left/right detour、KL 分布、return 分布
- **D7**：multimodality audit 是 Gate A.2 hard gate，不是 nice-to-have
- **D10**：spectrum 3 cell；audit 主要比较 E-uni vs M-multi-mix（M-uni-noise 在 P2 单独 audit，本 P0+P1 不评估）

---

## 2. CLI 契约

```bash
python -m scripts.audit_multimodality \
    --dataset-a <path/to/dataset_a>           # 单峰候选，e.g. e_uni_dryrun_200
    --dataset-b <path/to/dataset_b>           # 多峰候选，e.g. m_multi_mix_dryrun_200
    --output-dir <path/to/audit_output>       # 产出目录
    --knn-k 50                                # 每个 anchor 取多少近邻
    --gmm-max-components 3                    # BIC 选模上限
    --gmm-n-init 3                            # GMM EM 重启次数（防 local minimum）
    --mode-weight-floor 0.10                  # 单 mode 至少占比；否则降阶
    --n-anchor-states 500                     # 每 dataset 采样多少 anchor
    --n-bootstrap 1000                        # bootstrap 重采样次数
    --seed 0                                  # numpy + sklearn random seed
    --label-a "E-uni"                         # 图表 label
    --label-b "M-multi-mix"                   # 图表 label
```

**所有 args 都有合理默认值**（仅 `--dataset-a/b` 与 `--output-dir` 是必填）。

### 2.1 Input dataset 契约

每份 dataset 必须有：
- `transitions.npz` 含至少 `obs: [N, O]` 和 `actions: [N, A]`
- `metadata.json` 含 `obs_dim`, `action_dim`, `num_episodes`, `num_transitions`

兼容 broad val v2 + P0+P1 spec §1.2 产出的 dataset 格式。

### 2.2 不读什么

- 不读 `next_obs` / `rewards` / `dones` / `privileged_obs` / `next_actions`
- 仅 `obs` + `actions` 两列；其余忽略
- 这让 audit 对 schema drift 鲁棒（FQL plan 之外的 dataset 也可 audit）

---

## 3. 算法 step-by-step

### 3.1 加载与预处理

```python
def _load_obs_actions(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load (obs, actions) + metadata. Validates required fields."""
    transitions = np.load(dataset_dir / "transitions.npz")
    obs = np.asarray(transitions["obs"], dtype=np.float32)         # [N, O]
    actions = np.asarray(transitions["actions"], dtype=np.float32) # [N, A]
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    if obs.shape[0] != actions.shape[0]:
        raise ValueError(f"obs/actions length mismatch in {dataset_dir}")
    return obs, actions, metadata
```

**Compatibility check**（必须 dataset_a 与 dataset_b 间）：

```python
def _validate_compatibility(meta_a: dict, meta_b: dict) -> None:
    for key in ("obs_dim", "action_dim"):
        if int(meta_a[key]) != int(meta_b[key]):
            raise ValueError(f"{key} mismatch: a={meta_a[key]} vs b={meta_b[key]}")
```

### 3.2 Anchor state 采样

```python
def _sample_anchors(
    obs: np.ndarray,           # [N, O]
    n_anchors: int,
    rng: np.random.Generator,
) -> np.ndarray:               # indices [n_anchors]
    n = obs.shape[0]
    if n <= n_anchors:
        return np.arange(n)
    return rng.choice(n, size=n_anchors, replace=False)
```

**关键约定**：anchor 从对应 dataset 自己采样（不跨 dataset），但 k-NN 搜索时**也用同一 dataset**（每个 dataset 独立量化自己的内部多模态）。

### 3.3 k-NN 搜索

```python
def _build_knn_index(obs: np.ndarray) -> "NearestNeighbors":
    from sklearn.neighbors import NearestNeighbors
    # k+1 因为 anchor 自己也会出现在邻居里 —— 跳掉第一个
    nbrs = NearestNeighbors(n_neighbors=51, algorithm="auto", metric="euclidean")
    nbrs.fit(obs)
    return nbrs


def _query_neighbor_actions(
    nbrs,
    anchor_obs: np.ndarray,    # [n_anchor, O]
    all_actions: np.ndarray,   # [N, A]
    k: int,
) -> np.ndarray:               # [n_anchor, k, A]
    distances, indices = nbrs.kneighbors(anchor_obs, n_neighbors=k + 1)
    # 跳掉第一个 (anchor 自己)
    neighbor_idx = indices[:, 1:k + 1]     # [n_anchor, k]
    return all_actions[neighbor_idx]        # [n_anchor, k, A]
```

**为什么 `metric="euclidean"`**：obs 已经 normalized 或 bounded（s0 是 DVL + history broadcast，量级一致）；不需要 Mahalanobis 等高级 metric。如果未来发现 dimension scale 不一致，可在 P2 升级。

**计算复杂度**：500 anchor × kdtree on 200 sample → < 1s。

### 3.4 GMM with BIC mode selection

```python
def _gmm_mode_count(
    actions_k: np.ndarray,         # [k, A]，单 anchor 的 k 个邻居 actions
    max_components: int,
    n_init: int,
    weight_floor: float,
    rng_seed: int,
) -> int:                          # 1 / 2 / 3
    """Fit GMM with n_components in {1, ..., max_components}, return best by BIC.

    After BIC selection, applies weight_floor: if any component's mixture weight
    falls below weight_floor, drop it and re-evaluate with one fewer component.
    """
    from sklearn.mixture import GaussianMixture

    best_n = 1
    best_bic = float("inf")
    best_gmm = None
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            n_init=n_init,
            random_state=rng_seed,
            max_iter=200,
            reg_covar=1e-4,
        )
        try:
            gmm.fit(actions_k)
        except ValueError:
            # 数据 collapsed / 数量不足；跳过
            continue
        bic = gmm.bic(actions_k)
        if bic < best_bic:
            best_bic = bic
            best_n = n
            best_gmm = gmm

    if best_gmm is None:
        return 1   # fallback：所有 fit 都失败 → 当作单峰

    # Apply weight floor: drop modes below threshold
    weights = best_gmm.weights_
    n_effective = int(np.sum(weights >= weight_floor))
    return max(1, n_effective)
```

**关键 design 选择**：
- `covariance_type="full"`：2D action 下 full cov 只有 3 params/component，BIC 惩罚足够；不用 spherical/tied
- `reg_covar=1e-4`：防止 singular cov（k=50 邻居 actions 可能近共线，例如 cross_stream 任务下 heading 范围窄）
- `n_init=3`：EM 三起点，防 local minimum
- `weight_floor=0.10`：经验值，对应 50 neighbor 里 5 个支持一个 mode；过低易把 outlier 当 mode

### 3.5 主 audit loop

```python
def _audit_dataset(
    obs: np.ndarray,
    actions: np.ndarray,
    config: AuditConfig,
    rng: np.random.Generator,
) -> dict:
    nbrs = _build_knn_index(obs)
    anchor_idx = _sample_anchors(obs, config.n_anchor_states, rng)
    anchor_obs = obs[anchor_idx]                                   # [n_anchor, O]
    neighbor_actions = _query_neighbor_actions(
        nbrs, anchor_obs, actions, k=config.knn_k,
    )                                                              # [n_anchor, k, A]

    mode_counts = np.empty(len(anchor_idx), dtype=np.int32)
    for i in range(len(anchor_idx)):
        mode_counts[i] = _gmm_mode_count(
            neighbor_actions[i],
            max_components=config.gmm_max_components,
            n_init=config.gmm_n_init,
            weight_floor=config.mode_weight_floor,
            rng_seed=config.seed + i,   # different seed per anchor for n_init variability
        )

    n_anchor = len(mode_counts)
    p_distribution = {
        f"p_{j}": float(np.mean(mode_counts == j))
        for j in range(1, config.gmm_max_components + 1)
    }
    p_ge_2 = float(np.mean(mode_counts >= 2))
    return {
        "n_anchor": n_anchor,
        "anchor_indices": anchor_idx.tolist(),
        "mode_counts": mode_counts.tolist(),
        "p_distribution": p_distribution,
        "p_ge_2": p_ge_2,
        "mean_mode_count": float(np.mean(mode_counts)),
        "std_mode_count": float(np.std(mode_counts, ddof=1)),
    }
```

### 3.6 Paired bootstrap CI

```python
def _paired_bootstrap_delta_p_ge_2(
    mode_counts_a: np.ndarray,     # [n_anchor]
    mode_counts_b: np.ndarray,     # [n_anchor]
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    """Bootstrap CI for delta = p_ge_2(B) - p_ge_2(A).

    Paired: same bootstrap indices applied to both datasets per replicate.
    Assumes n_anchor matches between A and B (caller's responsibility).
    """
    n = len(mode_counts_a)
    if len(mode_counts_b) != n:
        raise ValueError("Paired bootstrap requires matched anchor counts")
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        p_a = np.mean(mode_counts_a[idx] >= 2)
        p_b = np.mean(mode_counts_b[idx] >= 2)
        deltas[i] = p_b - p_a
    return {
        "delta_mean": float(np.mean(deltas)),
        "delta_ci_2p5": float(np.percentile(deltas, 2.5)),
        "delta_ci_97p5": float(np.percentile(deltas, 97.5)),
        "delta_std": float(np.std(deltas, ddof=1)),
    }
```

**关于"paired" semantic**：A 与 B 的 anchor 不是同一 state（两 dataset 独立采 anchor），但**两 dataset 各 500 anchor → 两个 500-长 mode_count 数组同长度**。配对 bootstrap 用同样的 resample index，控制 anchor-level 抽样噪声 source。这是 D4RL audit 的 standard practice。

### 3.7 Welch's t one-sided test

```python
def _welch_t_one_sided(
    mode_counts_a: np.ndarray,
    mode_counts_b: np.ndarray,
) -> dict:
    """One-sided Welch's t-test: H1: mean(B) > mean(A)."""
    from scipy import stats
    t_stat, p_two_sided = stats.ttest_ind(
        mode_counts_b, mode_counts_a,
        equal_var=False, alternative="greater",
    )
    return {
        "welch_t": float(t_stat),
        "welch_p_one_sided": float(p_two_sided),
    }
```

scipy.stats 已在仓库环境内（不需要额外依赖）。

---

## 4. Gate A.2 verdict 逻辑

按 P0+P1 spec §1.3 **4 条**判据：

```python
def _verdict(
    audit_a: dict,
    audit_b: dict,
    bootstrap: dict,
    welch: dict,
) -> dict:
    """Apply 4 gate criteria from P0+P1 spec §1.3."""
    c1 = bootstrap["delta_ci_2p5"] > 0.10
    c2 = welch["welch_p_one_sided"] < 0.07
    c3 = audit_a["p_ge_2"] < 0.20
    c4 = audit_b["p_ge_2"] > 0.30

    criteria = {
        "c1_delta_p_ge2_ci_lower_above_0p10": {
            "value": bootstrap["delta_ci_2p5"],
            "threshold": 0.10,
            "pass": c1,
        },
        "c2_welch_p_below_0p07": {
            "value": welch["welch_p_one_sided"],
            "threshold": 0.07,
            "pass": c2,
        },
        "c3_a_unimodal_p_ge2_below_0p20": {
            "value": audit_a["p_ge_2"],
            "threshold": 0.20,
            "pass": c3,
        },
        "c4_b_multimodal_p_ge2_above_0p30": {
            "value": audit_b["p_ge_2"],
            "threshold": 0.30,
            "pass": c4,
        },
    }
    all_pass = all(c["pass"] for c in criteria.values())
    return {
        "criteria": criteria,
        "overall_pass": all_pass,
        "verdict": "Gate A.2 PASS" if all_pass else "Gate A.2 FAIL",
    }
```

**Fail handling**：本脚本只输出 verdict，不抛错。让 caller (e.g. p0p1 report) 决定 abort/mitigation 流程（P0+P1 spec §1.3）。`overall_pass=False` 时脚本以 exit code **2** 退出（CI/notebook 可捕获）。

---

## 5. 产出 schema

### 5.1 audit_summary.json

```json
{
  "version": "1.0",
  "generated_at": "2026-05-DD HH:MM:SS",
  "config": {
    "dataset_a": "offline_data/fql_succession/e_uni_dryrun_200",
    "dataset_b": "offline_data/fql_succession/m_multi_mix_dryrun_200",
    "label_a": "E-uni",
    "label_b": "M-multi-mix",
    "knn_k": 50,
    "gmm_max_components": 3,
    "gmm_n_init": 3,
    "mode_weight_floor": 0.10,
    "n_anchor_states": 500,
    "n_bootstrap": 1000,
    "seed": 0
  },
  "metadata_a": { "policy": "privileged", "num_episodes": 200, ... },
  "metadata_b": { "policy": "privileged+goalseek", "num_episodes": 200, ... },
  "audit_a": {
    "p_distribution": {"p_1": 0.86, "p_2": 0.12, "p_3": 0.02},
    "p_ge_2": 0.14,
    "mean_mode_count": 1.16,
    "std_mode_count": 0.39
  },
  "audit_b": {
    "p_distribution": {"p_1": 0.55, "p_2": 0.38, "p_3": 0.07},
    "p_ge_2": 0.45,
    "mean_mode_count": 1.52,
    "std_mode_count": 0.62
  },
  "bootstrap": {
    "delta_mean": 0.31,
    "delta_ci_2p5": 0.22,
    "delta_ci_97p5": 0.41,
    "delta_std": 0.048
  },
  "welch": {"welch_t": 9.6, "welch_p_one_sided": 1.2e-21},
  "verdict": {
    "criteria": { ... },
    "overall_pass": true,
    "verdict": "Gate A.2 PASS"
  }
}
```

### 5.2 mode_count_per_anchor.csv

| Column | Type | 说明 |
|---|---|---|
| `dataset_label` | str | "E-uni" / "M-multi-mix" |
| `anchor_idx` | int | 在 dataset 内的 transition index |
| `mode_count` | int | 1 / 2 / 3 |

每 anchor 一行，dataset_a 在前 500 行，dataset_b 在后 500 行。

### 5.3 mode_count_distribution.png

2-panel matplotlib figure：

```
┌──────────────────────────┬──────────────────────────┐
│ Left: histogram of       │ Right: bar chart of      │
│ mean_mode_count per      │ p_distribution {p_1,     │
│ anchor (overlaid A vs B) │ p_2, p_3} side-by-side   │
│                          │ A vs B with paired       │
│                          │ bootstrap CI error bars  │
└──────────────────────────┴──────────────────────────┘
Title: "Multimodality audit — {label_a} vs {label_b}"
Footer: "Δp_{≥2} = {delta_mean:.3f} (95% CI [{ci_lo:.3f}, {ci_hi:.3f}]),
         Welch p = {welch_p:.2e}, verdict = {verdict_str}"
```

DPI=120，size=(10, 4)；保存为 PNG。

---

## 6. Dependency 处理

### 6.1 sklearn import guard

脚本顶部：

```python
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:
    raise ImportError(
        "scripts/audit_multimodality.py requires scikit-learn. "
        "Install with: pip install scikit-learn\n"
        "(audit is a separate dev-time tool; sklearn is not in "
        "the repo's main training requirements.)"
    ) from exc
```

### 6.2 README / 注释

`audit_multimodality.py` module docstring 顶部：

```
Multimodality audit for offline RL dataset comparison
(FQL Succession Plan v1 §3.4, Gate A.2).

Dependency: scikit-learn (not in main repo requirements; install via
`pip install scikit-learn` before running this script). Audit is a
separate dev-time tool isolated from the training pipeline.
```

### 6.3 不修改主仓库 requirements

- 不更新仓库的 dependency manifest（若存在的话）
- 不在 `auv_nav/*` import sklearn
- 不在 `train_offline.py` / `collect_offline_data.py` 等主路径 import sklearn

---

## 7. 边界条件与异常

| 场景 | 处理 |
|---|---|
| `n_anchor_states > N` | 自动降级 `n_anchor = N`，warning print |
| `knn_k >= N` | 抛错（k-NN 在自己 dataset 上不可能 k=N 邻居） |
| `obs_dim` mismatch between a/b | 抛错 |
| `action_dim` mismatch | 抛错 |
| GMM fit `ValueError` 某 anchor | 该 anchor mode_count=1 (fallback)，记 warning count |
| 全部 anchor 都 GMM fit fail | 抛错，audit invalid |
| `transitions.npz` 缺 `obs` 或 `actions` | 抛错并指明缺哪个 key |
| `--seed` 不同每次跑结果有 jitter | 接受；reported in summary.json `config.seed` 字段 |

---

## 8. Implementation checklist

按此顺序写代码：

- [ ] 1. `AuditConfig` dataclass（保 args 顺序对齐 CLI）
- [ ] 2. `_load_obs_actions` + `_validate_compatibility`
- [ ] 3. `_sample_anchors`
- [ ] 4. `_build_knn_index` + `_query_neighbor_actions`
- [ ] 5. `_gmm_mode_count`
- [ ] 6. `_audit_dataset` 主 loop
- [ ] 7. `_paired_bootstrap_delta_p_ge_2`
- [ ] 8. `_welch_t_one_sided`
- [ ] 9. `_verdict`
- [ ] 10. `_save_summary_json` + `_save_per_anchor_csv`
- [ ] 11. `_plot_mode_count_distribution`
- [ ] 12. `main()` CLI orchestration + exit code 2 on fail
- [ ] 13. Module docstring + sklearn import guard

---

## 9. Test plan（pytest）

新文件 `tests/test_audit_multimodality.py`：

| Test | 验证 | 期望 |
|---|---|---|
| `test_load_obs_actions_roundtrip` | 写假 npz/metadata 再读 | shape 与写入一致 |
| `test_validate_compatibility_mismatch_raises` | obs_dim / action_dim 不同 | ValueError |
| `test_sample_anchors_no_replacement` | `n > n_anchor` 时 | 无重复 |
| `test_sample_anchors_full_when_small` | `n <= n_anchor` 时 | 返回所有 indices |
| `test_gmm_mode_count_unimodal_data` | 单高斯生成的 50 actions | mode_count == 1 |
| `test_gmm_mode_count_bimodal_data` | 双高斯 well-separated 生成 50 actions | mode_count == 2 |
| `test_gmm_mode_count_weight_floor` | 双高斯 95/5 mix（5 个 outlier） | mode_count == 1（weight_floor 起作用） |
| `test_audit_dataset_smoke` | 合成 obs/action × 500 transitions | 返回 dict 含 keys, p_ge_2 ∈ [0, 1] |
| `test_paired_bootstrap_ci_symmetric` | mode_a == mode_b | delta CI 含 0 |
| `test_paired_bootstrap_ci_skewed` | b 全 2，a 全 1 | delta_mean ≈ 1.0, CI 紧 |
| `test_welch_t_one_sided_directional` | b > a | p < 0.05; b < a; p > 0.95 |
| `test_verdict_all_pass_pathway` | 构造满足 4 条的输入 | overall_pass == True |
| `test_verdict_any_fail_pathway` | 各 break 一条 | overall_pass == False，criterion 字段标识失败项 |
| `test_main_e2e_synthetic` | 真跑 main()（合成两份 dataset） | exit code 0 (PASS) 或 2 (FAIL)；产出 3 个文件 |

**运行**：

```bash
pytest tests/test_audit_multimodality.py -v --cov=scripts.audit_multimodality
```

**覆盖率目标**：`scripts/audit_multimodality.py` ≥ 80%。

---

## 10. Acceptance criteria

| 标准 | 通过条件 |
|---|---|
| **Code quality** | `ruff check scripts/audit_multimodality.py` clean；type annotations on signatures；PEP 8 |
| **Unit test pass rate** | §9 全部 14 个 test pass |
| **Coverage** | ≥ 80% line |
| **End-to-end smoke** | 在 P0+P1 spec §1.2 实跑产出的 E-uni-200 与 M-multi-mix-200 上跑通；audit_summary.json + per-anchor CSV + distribution.png 三件齐 |
| **Deterministic** | 同 seed 跑两次，audit_summary.json 字节级相同（除 generated_at 时间戳） |
| **Sklearn isolation** | `grep -r 'sklearn' auv_nav/` 0 hit（sklearn 仅在 audit 脚本里） |
| **Wallclock** | 500 anchor × 50-NN × 3-init GMM 在 L4 CPU 上 < 60s（实际目标 ≤ 30s） |

任一不达标 → 不进 Gate A.2 实际 verdict run。

---

## 11. 与 P0+P1 spec 的引用关系

| P0+P1 spec 引用点 | 本 doc 对应章节 |
|---|---|
| §1.3 audit 脚本 spec（Inputs / 算法 / Verdict） | §2, §3, §4 |
| §1.3 Gate A.2 判据 4 条 | §4 verdict 逻辑 + 阈值 |
| §1.4 产出结构（audit_summary.json / CSV / PNG） | §5 |
| §1.5 时间预算 "Audit 脚本开发 2 天" | §8 + §9 |
| §6 Acceptance & Phase Exit "Gate A.2 PASS" | §10 + §4 exit code 约定 |

---

*Document version: v1.0 (2026-05-19). 维护策略：实现期若发现 GMM/k-NN 边界 case 漏洞回写本 doc + deviation 段；implementer 在 §10 acceptance pass 后给本 doc 打 "verified-by-impl" tag。*
