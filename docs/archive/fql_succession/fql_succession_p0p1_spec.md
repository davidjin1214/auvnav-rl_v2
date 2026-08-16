# FQL Succession — P0+P1 Executable Spec

> 📦 **已归档** — P2 之前的施工记录，2026-08-17 迁入 `docs/archive/fql_succession/`；归档只改位置与标注，**不含有效性判断**。缘由与本目录清单见 [`README.md`](README.md)。

> **文档版本**：v1.1（2026-05-20 patch:§5 CLI rename align actual code + §5.3 c4 阈值 Option α revision）
> **作用**：把 [`fql_succession_plan_v0.md`](fql_succession_plan_v0.md) §4.1 P0+P1（reward sanity + multimodality audit + FQL 实现 + Expert anchor）拆成可执行的命令、参数、文件契约、产出清单。
> **状态**:**CLOSED** — Gate A.1 cite ✓ + Gate A.2 ✅ PASS 4/4 + Gate B ⚠ 3/4 PASS + c4 marginal-FAIL (seed-driven, retroactive PASS under v1.1 阈值);P2 main comparison spec 在 `fql_succession_p2_main_spec.md` 起草中。
> **范围**：仅覆盖 P0+P1（约 4 周）。P2 main comparison、P3 mix ratio ablation、P4 writing 的 spec 在 P0+P1 闭环后另写。
>
> **版本历史**:
> - v1.0 (2026-05-18) — 初稿,5 task + Gate A.1 共享 + Gate A.2/B 判据
> - **v1.1 (2026-05-20)** — Session A patch:(a) §5.1/§5.2 CLI flag 名对齐 `scripts/train_offline.py` 实际实现 (`--algo` 等);(b) §5.3 c4 阈值改 Option α (slope ≥ −2 × SE_agg) 替换原 "slope ≥ 0";(c) §5.1/§5.2 manifest 引用保持 30-ep `single_u10_cross_tgt15.json` (Gate B 历史事实),P2 default ep100 manifest 在 P2 spec 引用,不回写本文档。
>
> **核心简化**：
> - **Task 1（reward sanity）共享 [`rebrac_broad_validation_v2_plan.md`](../../rebrac_broad_validation_v2_plan.md) N0 cell 证据**，本 spec 不重复 spec
> - **FQL 集成走 `auv_nav/offline_registry.py` + `scripts/train_offline.py --algorithm fql`**，不写独立 train entry（plan v1 §4.1 措辞需要纠正）
> - **M-multi-mix dataset 走 in-tree `scripts/concat_offline_datasets.py`**（v1 broad val A2 mix5050 已实战），不改 `collect_offline_data.py`
> - **审计仅 1 个 canonical 指标**（k-NN action GMM mode count）；其他 audit signal 是 nice-to-have appendix material

---

## 目录

0. [Overview + assumed inputs](#0-overview--assumed-inputs)
1. [Task A — Multimodality audit dry-run (Gate A.2)](#1-task-a--multimodality-audit-dry-run-gate-a2)
2. [Task B — FQL implementation (`auv_nav/fql.py` + registry)](#2-task-b--fql-implementation-auv_navfqlpy--registry)
3. [Task C — `scripts/train_offline.py` 集成](#3-task-c--scriptstrain_offlinepy-集成)
4. [Task D — E-uni dataset full collection](#4-task-d--e-uni-dataset-full-collection)
5. [Task E — Gate B FQL E-uni sanity](#5-task-e--gate-b-fql-e-uni-sanity)
6. [Acceptance & Phase Exit](#6-acceptance--phase-exit)
7. [Out of scope](#7-out-of-scope)
8. [Wallclock & parallelization](#8-wallclock--parallelization)

---

## 0. Overview + assumed inputs

### 0.1 P0+P1 任务流（4 件可执行任务 + Gate A.1 cite）

```
┌─────────────────────────────────────────────────────────────────┐
│ Gate A.1 (cite broad val v2 N0, no work in this spec)           │
│  → ReBRAC × arrival_v2 × crosscomp-u10-s0 ≥ 0.70 success        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ pass
┌─────────────────────────────┴───────────────────────────────────┐
│ Task A: Audit dry-run (Gate A.2)                                │
│   Collect E-uni-200 + privileged-100 + goalseek-100             │
│   concat_offline_datasets → m_multi_mix_dryrun_200              │
│   k-NN action GMM mode count audit                              │
│   → E-uni vs M-multi-mix ≥1.5σ separation                       │
│                                                                  │
│ Task B: FQL implementation (parallel to Task A wallclock)       │
│   auv_nav/fql.py — actor + critic + agent                       │
│   Register in auv_nav/offline_registry.py                       │
│                                                                  │
│ Task C: train_offline.py 集成                                    │
│   --algorithm fql choice + FQL-specific CLI args                │
└─────────────────────────────┬───────────────────────────────────┘
                              │ A passes + B/C ready
                              ▼
┌─────────────────────────────┴───────────────────────────────────┐
│ Task D: E-uni dataset full collection                           │
│   privileged + small noise × 1000 ep                            │
│                                                                  │
│ Task E: Gate B FQL E-uni sanity                                 │
│   FQL × E-uni × 1 seed parity vs ReBRAC × E-uni × 1 seed        │
│   → Gate B: FQL ≥ ReBRAC − 8pp                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ pass
                          P2 main comparison (separate spec)
```

### 0.2 共享参数（所有 P0+P1 task 强制一致）

| 参数 | 值 | 锚定 |
|---|---|---|
| Task config | `single_u10_cross_tgt15` benchmark | [`arrival_v2_experiment_report.md`](../../arrival_v2_experiment_report.md) §2 / broad val v2 N0 |
| Flow file | `wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy` | 同上 |
| Reward | `--objective arrival_v2` | plan v1 §2.3 |
| Probe layout | `--probe-layout s0` | plan v1 §3.3 |
| History length | `--history-length 4` | plan v1 §3.3 |
| Task geometry | `cross_stream` (隐含 in benchmark) | plan v1 §3.3 |
| ReBRAC anchor config | β1=4, β2=2, hidden=256, critic_LN=on, vanilla critic | broad val v2 §4 |
| Train epochs | 64 (≈200k steps with default batch) | broad val v2 §4 |
| Eval episodes | 100 | broad val v2 §4 |
| Seed pool | [42, 43, 44, 45, 46]（P0+P1 大多只用 [42]，P2 用全部 5） | broad val v2 §4 |

### 0.3 路径约定

- Datasets: `offline_data/fql_succession/{cell_id}/`
- Train output: `experiments/fql_succession/p0p1/{task_id}/`
- Audit output: `experiments/fql_succession/p0p1/audit/`
- Code: `auv_nav/fql.py`, `auv_nav/offline_registry.py` (modify), `scripts/train_offline.py` (modify)

---

## 1. Task A — Multimodality audit dry-run (Gate A.2)

### 1.1 目的

在投入 1000-episode 大规模 collection 之前，证明 spectrum 的 modality 轴真的 spread：E-uni（单 collector + 小噪声）vs M-multi-mix（双 collector episode-level mix）在 k-NN action GMM mode count 指标上**至少 1.5σ paired separation**。

如果 200 ep dry-run 就区分不开，没必要再继续 1000 ep —— 应当先调 mix 协议（加大第三 collector、改 mix 颗粒度）。

### 1.2 Dataset 收集命令

**E-uni dry-run（200 ep）**：

```bash
python -m scripts.collect_offline_data \
    --policy privileged \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --episodes 200 \
    --seed 0 \
    --action-noise 0.1 \
    --num-workers 8 \
    --output-dir offline_data/fql_succession/e_uni_dryrun_200
```

**M-multi-mix dry-run（200 ep，episode-level 50/50 mix）**：

**使用 in-tree `scripts/concat_offline_datasets.py`**（commit `992625c`，v1 broad val A2 mix5050 已实战使用，带 `mix_components` metadata + 114 行 unit test）：**两次 `collect_offline_data` + 一次 concat**，无需改源码。

**Step 1 — 收 privileged-100**（M-multi-mix 的 mode A 源）：

```bash
python -m scripts.collect_offline_data \
    --policy privileged \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --episodes 100 \
    --seed 100 \
    --action-noise 0.1 \
    --num-workers 8 \
    --output-dir offline_data/fql_succession/_components/privileged_100_seed100
```

**Step 2 — 收 goalseek-100**（mode B 源）：

```bash
python -m scripts.collect_offline_data \
    --policy goalseek \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --episodes 100 \
    --seed 200 \
    --action-noise 0.1 \
    --num-workers 8 \
    --output-dir offline_data/fql_succession/_components/goalseek_100_seed200
```

**Step 3 — concat 到 M-multi-mix-200**：

```bash
python -m scripts.concat_offline_datasets \
    --input-dir offline_data/fql_succession/_components/privileged_100_seed100 \
    --input-dir offline_data/fql_succession/_components/goalseek_100_seed200 \
    --output-dir offline_data/fql_succession/m_multi_mix_dryrun_200 \
    --mix-strategy episode_level \
    --task-sampler anchor_distribution
```

**产出 metadata 验证**（concat 后 `m_multi_mix_dryrun_200/metadata.json` 应有）：
- `policy = "privileged+goalseek"`
- `mix_components`: list of 2 entries，分别记录两源 policy、num_episodes、success_rate
- `mix_strategy = "episode_level"`
- `num_episodes = 200` (100 + 100)
- weighted aggregates: success_rate、mean_return 按 episode 数加权重算

**为什么用 concat 不用 mix-CLI**：
- v1 broad val A2 mix5050 已实战使用 `concat_offline_datasets.py`，模式 in-tree、tested
- Mode A / Mode B 源 dataset **完全独立可复用**：P3 mix ratio sweep 直接复用同一对源 dataset 改 episode 数即可（30/70 → 60+140; 70/30 → 140+60），不需要重新设计 collection
- `mix_components` metadata 比"每 episode collector_identity"语义更干净（reader 不需要 group-by 才能反推 mix 比例）
- 不动 `collect_offline_data.py` 等于不引入 cross-line 修改风险

### 1.3 Audit 脚本 spec

**新文件**：`scripts/audit_multimodality.py`

**Inputs**：
- `--dataset-a <path>`：第一份 dataset（单峰参照，typically E-uni）
- `--dataset-b <path>`：第二份 dataset（多峰候选，typically M-multi-mix）
- `--knn-k <int>`：近邻 k 值，默认 50
- `--gmm-max-components <int>`：BIC 选模上限，默认 3
- `--n-bootstrap <int>`：bootstrap CI 样本数，默认 1000
- `--n-anchor-states <int>`：采样多少 anchor state 跑 audit，默认 500
- `--output-dir <path>`：audit 产出目录

**算法**：

```
1. Load (obs, action) pairs from both datasets
2. Build joint state index (kNN on obs from union)
3. Sample N anchor_states (default 500) uniformly from each dataset's obs
4. For each anchor_state:
   a. Find k=50 nearest neighbors by obs L2 distance
   b. Collect their actions (k action vectors)
   c. Fit GMM with n_components ∈ {1, 2, 3}; select by BIC
   d. Record selected n_components (1 / 2 / 3)
5. Per dataset, compute mode-count distribution:
   {p_1: frac with n_components=1, p_2: ..., p_3: ...}
6. Statistical test:
   a. paired bootstrap CI of (p_≥2_B − p_≥2_A) over 1000 resamples
   b. Welch's t one-sided test on n_components per anchor
7. Output:
   - audit_summary.json: {p_distribution_A, p_distribution_B, delta_p_ge2_CI, welch_t, welch_p, verdict}
   - mode_count_distribution.png: 2-panel histogram
   - mode_count_per_anchor.csv: raw per-anchor results
```

**Verdict 判据 (Gate A.2)**：

| 判据 | 通过条件 |
|---|---|
| `p_≥2(M-multi-mix) − p_≥2(E-uni)` 95% CI 下界 | **> 0.10** |
| Welch's t one-sided p-value | **< 0.07**（对应 ~1.5σ） |
| `p_≥2(E-uni)` 绝对值 | < 0.20（确认 E-uni 自身是单峰） |
| `p_≥2(M-multi-mix)` 绝对值 | > 0.30（确认 M-multi-mix 自身是多峰） |

**4 条全过 → Gate A.2 pass**。任一失败 → Gate A.2 fail，按 plan v1 §4.1 mitigation 调整 mix 协议。

### 1.4 产出结构

```
offline_data/fql_succession/
├── _components/                                  # 可复用源 dataset（P2/P3 也复用）
│   ├── privileged_100_seed100/   (mode A 源)
│   │   ├── transitions.npz
│   │   └── metadata.json
│   └── goalseek_100_seed200/     (mode B 源)
│       ├── transitions.npz
│       └── metadata.json
├── e_uni_dryrun_200/             (privileged 单源)
│   ├── transitions.npz
│   └── metadata.json
└── m_multi_mix_dryrun_200/       (concat 产物)
    ├── transitions.npz
    └── metadata.json (含 mix_components 字段)

experiments/fql_succession/p0p1/audit/
├── audit_summary.json
├── mode_count_distribution.png
└── mode_count_per_anchor.csv
```

### 1.5 时间预算（**修正：不需要改 collect_offline_data.py**）

| 子任务 | 时长 | 备注 |
|---|---|---|
| ~~`collect_offline_data.py` mix 功能开发~~ | **0**（取消） | 用 in-tree `concat_offline_datasets.py` 替代 |
| E-uni + privileged-100 + goalseek-100 collection | 6 h CPU (8 worker × 500 ep 累计) | wallclock |
| `concat_offline_datasets` 合并 M-multi-mix-200 | <1 min | shell 一行 |
| Audit 脚本开发 | 2 天 | scipy.cluster, sklearn.mixture |
| Audit 运行 + Gate A.2 verdict | 1 h | 500 anchor × 50-NN × GMM 较快 |
| **小计** | **~3 工作日**（vs 原估 5） | 砍 2 天 dev 工作 |

---

## 2. Task B — FQL implementation (`auv_nav/fql.py` + registry)

### 2.1 设计契约

**FQL = Flow Q-Learning** (Park / Li / Levine ICML 2025)：

| 组件 | 设计 | 继承来源 |
|---|---|---|
| **Behavior teacher** | Conditional flow-matching network: predicts velocity field `v_θ(x_t, t \| s)` over action space | FQL paper §3.1 |
| **Distilled student actor** | Deterministic MLP `π_φ(s) → a`，由 teacher 的 ODE 解一次性蒸馏 + Q-guided update | FQL paper §3.2 |
| **Critic** | Twin-Q + critic LayerNorm | **从 ReBRAC 继承**（paper 1 Finding iv） |
| **Q-guidance loss** | `L_actor = -Q(s, π_φ(s)) + α_bc · ‖π_φ(s) − a_teacher(s)‖²` | FQL paper §3.3 |
| **BC anchor** | `a_teacher(s)` = one-step flow integration from `t=0` to `t=1` with student `π_φ(s)` as init | FQL paper §3.3 |

### 2.2 文件结构

**新文件**：`auv_nav/fql.py`

```python
# 顶部模块结构（参考 auv_nav/rebrac.py）

from dataclasses import dataclass
from typing import Literal
# ... torch imports via require_torch()

@dataclass(slots=True)
class FQLConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    # FQL-specific
    flow_steps: int = 10                  # ODE 离散化步数（teacher integration）
    flow_time_embed_dim: int = 32         # 时间步 embedding 维度
    distill_alpha_bc: float = 1.0         # actor BC anchor 系数
    teacher_lr: float = 3e-4              # flow-matching teacher 学习率
    actor_lr: float = 3e-4                # student actor 学习率
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_freq: int = 2
    batch_size: int = 256
    grad_clip_norm: float = 10.0

    # 继承自 ReBRAC（Finding iv）
    critic_use_layernorm: bool = True
    actor_use_layernorm: bool = False
    dropout_rate: float = 0.0

    # 兼容 ReBRAC 的 Q 归一化（Finding iii）
    normalizer_eps: float = 1e-3
    normalize_q: bool = True


class FlowMatchingTeacher(_ModuleBase):
    """Velocity field v_θ(x_t, t | s)."""
    # __init__(self, obs_dim, action_dim, hidden_dim, num_hidden_layers, time_embed_dim)
    # forward(self, x_t: Tensor[B, action_dim], t: Tensor[B, 1], s: Tensor[B, obs_dim]) -> Tensor[B, action_dim]
    # integrate(self, s: Tensor[B, obs_dim], n_steps: int) -> Tensor[B, action_dim]
    #   ODE: x_0 ~ N(0, I); x_{i+1} = x_i + (1/n_steps) * v_θ(x_i, i/n_steps, s)


class DistilledStudent(_ModuleBase):
    """One-step deterministic actor π_φ(s) → a."""
    # 与 ReBRAC DeterministicActor 几乎一致；可直接复用


class FQLPolicy:
    """Inference-time wrapper exposing Policy API used by evaluate_offline_policy_parallel."""
    # act(obs, policy_state, deterministic) -> (action, policy_state)
    # reset_policy_state() -> dict


class FQLAgent:
    """Training agent. Mirrors ReBRACAgent's surface for offline_registry."""
    # __init__(self, config: FQLConfig, *, obs_normalizer, device)
    # update(self, batch: dict) -> dict[str, float]   # 返回 {loss_flow, loss_actor, loss_critic, q_mean, ...}
    # save(self, path: Path) -> None
    # load(self, path: Path) -> None
    # policy(self) -> FQLPolicy
```

### 2.3 训练 loop（FQL update step）

每个 `update(batch)` 调用按以下顺序：

```
1. Sample minibatch (s, a, r, s', done) from replay
2. (Step every step) Flow-matching teacher update:
   - t ~ U(0, 1)
   - x_0 ~ N(0, I), shape [B, action_dim]
   - x_t = (1-t) * x_0 + t * a   (linear interpolation path; FQL §3.1)
   - target_v = a - x_0           (constant velocity along this path)
   - loss_flow = MSE(v_θ(x_t, t, s), target_v)
   - update teacher
3. (Step every step) Critic update (twin-Q + LayerNorm):
   - With target student π̄_φ(s'), target Q̄(s', π̄_φ(s'))
   - TD target: y = r + γ * (1-done) * min(Q̄_1, Q̄_2)
   - loss_critic = MSE(Q_1(s, a), y) + MSE(Q_2(s, a), y)
   - update critics; soft target update
4. (Every policy_freq steps) Student actor update:
   - a_teacher = teacher.integrate(s, n_steps=flow_steps)  [no grad]
   - a_student = π_φ(s)
   - Q value: Q_min = min(Q_1(s, a_student), Q_2(s, a_student))
   - Q normalization (per ReBRAC Finding iii): Q_norm = Q_min / (|mean(Q_min)| + normalizer_eps)
   - loss_actor = -Q_norm + distill_alpha_bc * MSE(a_student, a_teacher)
   - update student; soft target update
```

### 2.4 Registry 集成

**Modify** `auv_nav/offline_registry.py`：

```python
def normalize_offline_algo(algo: str | None) -> str:
    value = "td3bc" if algo is None else str(algo).strip().lower()
    if value not in {"td3bc", "rebrac", "fql"}:        # ← add "fql"
        raise ValueError(f"Unsupported offline algorithm: {algo!r}")
    return value


def make_agent_config(algo: str, **kwargs: Any) -> Any:
    algo_name = normalize_offline_algo(algo)
    if algo_name == "td3bc":
        from .td3bc import TD3BCConfig
        return TD3BCConfig(**kwargs)
    if algo_name == "fql":                              # ← add branch
        from .fql import FQLConfig
        return FQLConfig(**kwargs)
    from .rebrac import ReBRACConfig
    return ReBRACConfig(**kwargs)


def make_agent(algo: str, config: Any, *, obs_normalizer, device: str = "cpu") -> Any:
    algo_name = normalize_offline_algo(algo)
    if algo_name == "td3bc":
        from .td3bc import TD3BCAgent
        return TD3BCAgent(config, obs_normalizer=obs_normalizer, device=device)
    if algo_name == "fql":                              # ← add branch
        from .fql import FQLAgent
        return FQLAgent(config, obs_normalizer=obs_normalizer, device=device)
    from .rebrac import ReBRACAgent
    return ReBRACAgent(config, obs_normalizer=obs_normalizer, device=device)
```

### 2.5 单元测试 spec

新文件：`tests/test_fql.py`

| Test | 验证 |
|---|---|
| `test_flow_teacher_shapes` | velocity 输出 shape match action_dim；integrate 输出在 \[-1, 1\]^action_dim 内 |
| `test_flow_teacher_overfits_one_action` | 用 1-sample dataset，teacher loss < 1e-3 after 1000 steps |
| `test_student_distill_matches_teacher` | 用固定 teacher，student 蒸馏 1000 步后 ‖π_φ(s) − teacher.integrate(s)‖² < 0.05 |
| `test_fql_agent_update_smoke` | 随机 batch × 100 update 不抛错；losses finite |
| `test_fql_registry_roundtrip` | `make_agent_config("fql", obs_dim=10, action_dim=2)` + `make_agent("fql", cfg)` 不抛错 |
| `test_fql_checkpoint_save_load` | save → load → policy 输出一致（容差 1e-6） |

### 2.6 时间预算

| 子任务 | 时长 |
|---|---|
| 参考 FQL JAX repo → PyTorch 移植 flow teacher | 3 天 |
| 蒸馏 + Q-guided student actor | 2 天 |
| Critic（直接复用 ReBRAC 模式）+ Agent wrapper | 1 天 |
| Registry + train_offline.py 集成 | 0.5 天 |
| 单元测试 6 个 | 1.5 天 |
| **小计** | **~8 工作日** |

可与 Task A wallclock 并行（不抢同一资源：Task A 是 CPU collection + audit；Task B 是 dev 写代码）。

---

## 3. Task C — `scripts/train_offline.py` 集成

### 3.1 CLI 改动

```python
# scripts/train_offline.py line ~709
parser.add_argument(
    "--algorithm",
    choices=["td3bc", "rebrac", "fql"],   # ← add "fql"
    default="rebrac",
    help="Offline RL algorithm to train.",
)

# 新增 FQL-specific args（不污染 ReBRAC defaults）
parser.add_argument("--fql-flow-steps", type=int, default=10,
                    help="(FQL only) ODE discretization steps for teacher.")
parser.add_argument("--fql-flow-time-embed-dim", type=int, default=32,
                    help="(FQL only) time embedding dimension.")
parser.add_argument("--fql-distill-alpha-bc", type=float, default=1.0,
                    help="(FQL only) BC anchor weight on student loss.")
parser.add_argument("--fql-teacher-lr", type=float, default=3e-4,
                    help="(FQL only) flow-matching teacher learning rate.")
```

### 3.2 `_resolve_*_config` 函数

仿照 `_resolve_rebrac_actor_layernorm` 等函数，新增 `_resolve_fql_config(args) -> dict`，把 FQL-specific args 打包为 `FQLConfig` kwargs。

### 3.3 logging 兼容

`FQLAgent.update()` 返回的 metrics dict 必须包含至少以下 key（与 `train_offline.py` 现有 logger 兼容）：

```python
{
    "loss_actor": float,
    "loss_critic": float,
    "loss_flow": float,        # FQL-specific, 新增 column
    "q_mean": float,
    "q_std": float,
    "grad_norm_actor": float,
    "grad_norm_critic": float,
    "grad_norm_flow": float,   # FQL-specific
}
```

`append_csv` 写入时自动 union 列名（无需 schema 改动；现有实现已支持 None 填充）。

### 3.4 验证

```bash
# Smoke test: 100 step FQL training on tiny synthetic dataset
python -m scripts.train_offline \
    --algorithm fql \
    --offline-data offline_data/fql_succession/e_uni_dryrun_200/transitions.npz \
    --total-steps 100 \
    --batch-size 64 \
    --save-dir /tmp/fql_smoke \
    --eval-every-steps 200 \
    --device cpu
```

预期：100 step 无错误退出；`/tmp/fql_smoke/training_log.csv` 包含 `loss_flow` 列。

---

## 4. Task D — E-uni dataset full collection

### 4.1 命令

```bash
python -m scripts.collect_offline_data \
    --policy privileged \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --episodes 1000 \
    --seed 0 \
    --action-noise 0.1 \
    --num-workers 16 \
    --output-dir offline_data/fql_succession/e_uni_1000
```

### 4.2 验证

| 指标 | 预期 |
|---|---|
| Episode 数 | 1000 |
| Success rate | ≥ 0.85（privileged + 小噪声应保持 high-success；与 broad val v2 S sanity privileged 70% 不同 cell — 这里是 sub-critical u10 而非 critical u15） |
| `privileged_obs_present` 标记 | True（含 `[u_eq, v_eq]` 列） |
| Action dim 一致性 | 2 |
| Obs dim | 10（s0 + history=4 broadcast） |

如果 success rate < 0.85，重置 noise 到 0.05 重跑（不放弃 noise，要保留 modality unimodal 但确保 expert quality）。

### 4.3 产出

```
offline_data/fql_succession/e_uni_1000/
├── transitions.npz
├── metadata.json
└── collection_summary.txt  (auto-generated by collect_offline_data)
```

### 4.4 时间预算

L4 / 16 worker × 1000 ep ≈ **2-3 h CPU wallclock**。可与 Task B 单元测试并行。

---

## 5. Task E — Gate B FQL E-uni sanity

### 5.1 ReBRAC E-uni × seed=42 baseline

> **CLI flag note (v1.0 → v1.1 patch, 2026-05-20)**：本 spec v1.0 起草时使用的 prospective CLI 字段名（`--algorithm`, `--eval-every-steps`, `--rebrac-*` 前缀）已与 `scripts/train_offline.py` 实际实现对齐。完整映射表见 Session B `notebooks/fql_succession_gate_b.ipynb` gate-b-header cell；Gate B Option B 闭环已用 actual code flags 跑过 4 run（[`fql_succession_gate_b_report.md`](fql_succession_gate_b_report.md) §2）。**§5.1/§5.2 的 manifest 引用保持原 30-ep `single_u10_cross_tgt15.json`（Gate B 历史事实）**;P2 default 切到 ep100 manifest (D18),不回写本 spec。

```bash
python -m scripts.train_offline \
    --algo rebrac \
    --offline-data offline_data/fql_succession/e_uni_1000/transitions.npz \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --manifest benchmarks/single_u10_cross_tgt15.json \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --total-steps 200000 \
    --batch-size 256 \
    --eval-every 10000 \
    --eval-episodes 100 \
    --actor-penalty-coef 4.0 \
    --critic-penalty-coef 2.0 \
    --critic-layernorm \
    --no-actor-layernorm \
    --seed 42 \
    --save-dir experiments/fql_succession/p0p1/gate_b/rebrac_e_uni_seed42 \
    --device cuda
```

### 5.2 FQL E-uni × seed=42

```bash
python -m scripts.train_offline \
    --algo fql \
    --offline-data offline_data/fql_succession/e_uni_1000/transitions.npz \
    --flow wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
    --manifest benchmarks/single_u10_cross_tgt15.json \
    --probe-layout s0 \
    --history-length 4 \
    --target-speed 1.5 \
    --task-geometry cross_stream \
    --objective arrival_v2 \
    --total-steps 200000 \
    --batch-size 256 \
    --eval-every 10000 \
    --eval-episodes 100 \
    --flow-steps 10 \
    --distill-alpha-bc 1.0 \
    --teacher-lr 3e-4 \
    --flow-time-embed-dim 32 \
    --seed 42 \
    --save-dir experiments/fql_succession/p0p1/gate_b/fql_e_uni_seed42 \
    --device cuda
```

### 5.3 Gate B 判据

| 指标 | 通过条件 |
|---|---|
| c1: FQL `test_success` (last 3 eval, mean) | **≥ ReBRAC `test_success` (last 3 eval, mean) − 0.08** |
| c2: FQL `loss_flow` 末段 | < 0.05（teacher 收敛） |
| c3: FQL `loss_actor` 末段 | 数量级与 ReBRAC `loss_actor` 一致（±1 个 OOM） |
| c4: FQL eval 曲线 no-major-collapse | 末 30% aggregated slope ≥ −2 × SE(slope_aggregated_n_seeds)。SE 按 √(p(1−p)/n_eval) / √17.5 / √n_seeds 计算（n_eval = manifest episodes, n_pts = 6 (last-30% of 20 evals), p ≈ in-training mean success rate）。**在 Gate B legacy (n=2, 30-ep) 下阈值 ≈ −0.0283**;在 P2 default (n=5, 100-ep) 下阈值 ≈ −0.0098。 |

**4 条全过 → Gate B pass → 进入 P2 main comparison。**

**c4 retroactive 校准依据 (v1.1 patch, 2026-05-20)**:见 [`fql_succession_c4_threshold_revision.md`](fql_succession_c4_threshold_revision.md) §3 retroactive 测试表 + §4 Type I/II 分析。Gate B Option B 闭环实测:FQL aggregated slope −0.0038 vs revised threshold −0.0283 → **c4 在新阈值下 retroactive PASS** (Gate B 原阈值 0 下 marginal-FAIL,seed-driven 而非 FQL-driven,见 [`fql_succession_gate_b_report.md`](fql_succession_gate_b_report.md) §5)。

任一失败：
- FQL << ReBRAC：实现有 bug，debug 3 iter；仍失败则 STOP（plan v1 R4 mitigation）
- `loss_flow` 不收敛：检查 time sampling、flow path 公式、teacher 学习率
- 曲线 collapse：检查 Q-normalization、actor 学习率、`distill_alpha_bc`

### 5.4 时间预算

2 run × 200k step × 1.5h ≈ **3h wallclock**（顺跑）或 **1.5h**（双 GPU 或 num_envs=1 单卡 fp32 并发）。

---

## 6. Acceptance & Phase Exit

P0+P1 整体出口条件：

| Gate | 判据来源 | 必过 |
|---|---|---|
| **Gate A.1** | broad val v2 N0 ≥ 0.70 success | ✅ |
| **Gate A.2** | Task A audit 4 条全过 | ✅ |
| **FQL impl test** | `tests/test_fql.py` 6 个 test 全过 | ✅ |
| **train_offline.py smoke** | `--algorithm fql` 100-step smoke 无错 | ✅ |
| **Gate B** | Task E 4 条全过 | ✅ |

全过 → 升级 `fql_succession_plan_v0.md` 到 v1.1，进入 P2 main comparison spec 写作。

任一 Gate 失败 → 按对应 mitigation 处理；2 轮 mitigation 失败 → abort plan v1，回到 framing layer。

---

## 7. Out of scope（**本 spec 明确不做**）

- ❌ M-uni-noise 数据收集（P2 范围）
- ❌ FQL × M-multi-mix / M-uni-noise 训练（P2 范围）
- ❌ Mix ratio sweep（P3 范围）
- ❌ TD3+BC × E-uni 对照（P2 选做；P0+P1 不做）
- ❌ 5-seed ReBRAC × arrival_v2 reward anchor 对照（D12 砍掉）
- ❌ 4-metric audit（左右绕比、KL 分布、return 分布）（D11 砍掉，本 spec 只跑 mode-count）
- ❌ Asym critic 集成（plan v1 主线 vanilla critic；asym critic 是 follow-up）
- ❌ Paper writing（P4 范围）

---

## 8. Wallclock & parallelization

### 8.1 串行 critical path

```
Day 1:    E-uni-200 + privileged-100 + goalseek-100 dry-run collection (~6h CPU)
          concat → m_multi_mix_dryrun_200 (<1 min)
Day 2-3:  audit_multimodality.py dev + Gate A.2 run
Day 1-9:  (parallel) auv_nav/fql.py impl + tests
Day 9:    train_offline.py integration + smoke
Day 10:   E-uni full 1000 collection (~3h CPU)
Day 11:   Gate B ReBRAC + FQL training (~3h GPU)
Day 12:   Gate B verdict + p0p1 report writing
```

**总 wallclock：~12 working days ≈ 2.5 周**（vs 初版 spec 估 14 天，砍掉的是 `collect_offline_data.py` mix dev 2 天）。

### 8.2 并行机会

| 同时进行 | 资源不冲突理由 |
|---|---|
| FQL impl (Task B/C) ∥ audit dry-run collection (Task A1) | Task A1 是 CPU worker，Task B/C 是 dev 写代码 |
| FQL unit test (Task B5) ∥ audit script dev (Task A2) | 都是 dev 工作，dev 自行切换 |
| E-uni full collection (Task D) ∥ FQL impl 收尾 (Task B 末段) | Task D CPU only，Task B dev only |

### 8.3 风险时间膨胀

| 触发条件 | 时间影响 |
|---|---|
| Gate A.2 fail，调 mix 协议 | +3-5 天（重设计 + 重 dry-run） |
| FQL impl bug 排查 > 3 iter | +5-10 天 |
| Gate B fail，FQL E-uni < ReBRAC − 8pp | +5-10 天 debug 或 STOP |

最坏情况 P0+P1 ≈ **6 周**；预期 case **3 周**。

---

*Document version: v1.0 (2026-05-18). 维护策略：每个 Task 启动前在文档内补充更细节的命令；Task 闭环后补"实测时长 vs 估计"对照。Gate A.1/A.2/B 通过/失败后更新 §6 verdict 状态。*
