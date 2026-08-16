# `auv_nav/fql.py` — PyTorch Port Design Doc

> **文档版本**：v1.0（2026-05-19）
> **作用**：把 [`fql_succession_p0p1_spec.md`](archive/fql_succession/fql_succession_p0p1_spec.md) §2 (Task B) 的 FQL 实现拆成 tensor-shape-level 可实现的 design contract。
> **状态**：**IMPLEMENTED**（2026-08-17 订正——原写「Active design，待 executor 实现」，代码早已落地）。本文是 `auv_nav/fql.py` 的**现行 design contract**：模块头注与 [`../tests/test_fql.py`](../tests/test_fql.py)（§13 的 12 个测试）都指回本文。FQL succession 线虽已 NEGATIVE 闭环，该模块仍在仓库内、仍被测试覆盖，且论文 §5.4 核对 loss 口径要读它——故本文留在 `docs/` 顶层，未随该线迁入 [`docs/archive/fql_succession/`](archive/fql_succession/README.md)。
> **范围**：仅 `auv_nav/fql.py` 内的算法与接口设计 + `auv_nav/offline_registry.py` 的 dispatch 改动；不涉及 audit / collection / training entry CLI（在 P0+P1 spec 内已覆盖）。
> **目标读者**：implementer（executor agent 或人），需要按此 doc 写出能跑通 P0+P1 spec §3.4 smoke test 的实现。
>
> **核心约束**：
> - 算法严格遵循 FQL paper (Park, Li, Levine, ICML 2025) §3.1-3.3
> - **Critic 直接继承 ReBRAC 实现**（critic LN + twin-Q + Q-normalization Finding iii）—— 这是 paper claim 干净性的关键，不允许 FQL 自己重新设计 critic
> - 接口契约（API surface）严格 mirror `ReBRACAgent` —— 保证 `train_offline.py` / `evaluate_offline_policy_parallel` / checkpoint pipeline 零改动接入
> - **Python 风格**：PEP 8、type annotations on all signatures、`@dataclass(slots=True)` for config、`from __future__ import annotations`、conditional torch import guard

---

## 目录

1. [FQL 算法 recap（仅 paper 关键公式）](#1-fql-算法-recap)
2. [模块与 tensor shape 契约](#2-模块与-tensor-shape-契约)
3. [FQLConfig dataclass](#3-fqlconfig-dataclass)
4. [FlowMatchingTeacher 设计](#4-flowmatchingteacher-设计)
5. [DistilledStudent 设计](#5-distilledstudent-设计)
6. [Critic（继承 ReBRAC 的 QNetwork）](#6-critic继承-rebrac-的-qnetwork)
7. [FQLAgent.update — 训练 loop](#7-fqlagentupdate--训练-loop)
8. [Metrics dict 契约](#8-metrics-dict-契约)
9. [Checkpoint schema + FQLPolicy](#9-checkpoint-schema--fqlpolicy)
10. [Registry 集成](#10-registry-集成)
11. [JAX → PyTorch 已知 pitfall](#11-jax--pytorch-已知-pitfall)
12. [Implementation checklist](#12-implementation-checklist)
13. [Test plan（pytest）](#13-test-plan)
14. [Acceptance criteria](#14-acceptance-criteria)

---

## 1. FQL 算法 recap

仅复述实现需要的关键公式。完整 derivation 见 FQL paper。

### 1.1 Behavior teacher（flow matching）

学一个 velocity field `v_θ(x_t, t | s)`，shape: `[B, A]`，参数化为 `(x_t, t, s) → ∂x/∂t`。

**Linear interpolation path** (FQL §3.1)：

```
x_0 ~ N(0, I)              # shape [B, A], 起点
x_1 = a                    # 数据 action，shape [B, A]
t ~ U(0, 1)                # shape [B, 1]
x_t = (1 - t) * x_0 + t * a    # shape [B, A]
v_target = x_1 - x_0 = a - x_0  # 常速度（沿 linear path）
```

**Teacher loss**：

```
L_flow = E_{(s, a) ~ D, t ~ U(0,1), x_0 ~ N(0,I)} [
    || v_θ(x_t, t, s) - (a - x_0) ||²_2
]
```

### 1.2 Teacher integration（推理时）

从噪声 `x_0 ~ N(0, I)` 跑 Euler ODE 到 `x_1`：

```
x_0 ~ N(0, I)
for i = 0, ..., n_steps - 1:
    t = i / n_steps                  # scalar
    dt = 1 / n_steps                 # scalar
    x_{i+1} = x_i + dt * v_θ(x_i, t, s)
return x_{n_steps}                   # = teacher action prediction
```

`n_steps` 默认 10（FQL paper 用 5-10，取上界保稳）。

### 1.3 Student distillation + Q-guidance

Student `π_φ(s) → a` 是 1-step deterministic actor。**Loss 由两部分**：

```
a_teacher = teacher.integrate(s, n_steps)   # detached, no_grad
a_student = π_φ(s)                          # with grad
Q_min = min(Q_1(s, a_student), Q_2(s, a_student))
λ = 1 / (|mean(Q_min)|.detach() + ε)        # ReBRAC Finding iii: Q normalization

L_actor = - λ · mean(Q_min) + α_bc · || a_student - a_teacher ||²_2 / A
```

注：
- BC term 在 FQL paper 是 distill 项（match teacher's expensive integration）；在 ReBRAC 是 match dataset action。**FQL 用 teacher action**，不是 dataset action。
- Q 项的 `λ` 因子直接继承 ReBRAC `normalize_q` 逻辑（Finding iii）

### 1.4 Critic update

完全等同 TD3+BC 的 critic update（无 BC penalty on critic side）：

```
a_next_student = π_φ_target(s')    # student target policy
a_next_noisy = (a_next_student + clipped_noise).clamp(-1, 1)
y = r + γ · (1 - done) · min(Q_1_target(s', a_next_noisy), Q_2_target(s', a_next_noisy))
L_critic = MSE(Q_1(s, a), y) + MSE(Q_2(s, a), y)
```

**注**：与 ReBRAC 不同，FQL critic **不带 critic-side BC penalty**（FQL 的 BC anchor 全在 student loss 里走 teacher distillation）。这是 paper 的设计选择，不可改。

---

## 2. 模块与 tensor shape 契约

`B` = batch size, `A` = action_dim (=2 for AUV), `O` = obs_dim (=10 for s0+history=4), `H` = hidden_dim (=256), `T` = time_embed_dim (=32), `K` = num_hidden_layers (=3), `N` = flow_steps (=10).

| 模块 | Input shapes | Output shape | 说明 |
|---|---|---|---|
| `TimeEmbed` | `t: [B, 1]` | `[B, T]` | Sinusoidal positional encoding |
| `FlowMatchingTeacher` | `x_t: [B, A]`, `t: [B, 1]`, `s: [B, O]` | `[B, A]` | velocity prediction |
| `FlowMatchingTeacher.integrate` | `s: [B, O]`, `n_steps: int` | `[B, A]` | ODE integration result |
| `DistilledStudent` | `s: [B, O]` | `[B, A]` | 1-step deterministic actor |
| `QNetwork` (inherited from ReBRAC) | `obs: [B, O]`, `action: [B, A]` | `[B]` | scalar Q-value per sample |

**Critical convention**：所有 action 在 `[-1, 1]` 区间（与 ReBRAC `DeterministicActor` `tanh` 输出一致）。Teacher 的 `integrate` 输出**不带** tanh —— 由 student 通过 `(a_teacher - a_student)²` BC loss 学到 bounded 输出。这是 FQL paper 的设计：teacher 输出 unbounded，student 输出 bounded via tanh。

---

## 3. FQLConfig dataclass

```python
@dataclass(slots=True)
class FQLConfig:
    # === core dimensions（必填）===
    obs_dim: int
    action_dim: int

    # === network 架构（继承 ReBRAC defaults）===
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    # === FQL-specific ===
    flow_steps: int = 10
    flow_time_embed_dim: int = 32
    distill_alpha_bc: float = 1.0

    # === learning rates ===
    teacher_lr: float = 3e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # === RL hyperparams ===
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    batch_size: int = 256
    grad_clip_norm: float = 10.0

    # === regularization（继承 ReBRAC Finding iv）===
    critic_use_layernorm: bool = True
    actor_use_layernorm: bool = False
    teacher_use_layernorm: bool = False
    dropout_rate: float = 0.0

    # === Q normalization（继承 ReBRAC Finding iii）===
    normalizer_eps: float = 1e-3
    normalize_q: bool = True
```

**说明**：
- `slots=True` 与 ReBRACConfig 一致，禁止意外加字段
- 默认值锚定 FQL paper appendix table 1 (D4RL hyperparams) + ReBRAC paper 1 Finding iii/iv
- 不引入 `privileged_obs_dim` —— plan v1 §3.3 明确 FQL 不用 asym critic（vanilla critic only）

---

## 4. FlowMatchingTeacher 设计

### 4.1 TimeEmbed (sinusoidal positional encoding)

参考 DDPM / FQL 标准实现：

```python
class TimeEmbed(_ModuleBase):
    """Sinusoidal time embedding similar to Transformer positional encoding."""

    def __init__(self, embed_dim: int) -> None:
        require_torch()
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"time_embed_dim must be even, got {embed_dim}")
        self.embed_dim = embed_dim
        half = embed_dim // 2
        # freqs shape [half], values exp(-log(10000) * k / half) for k in 0..half-1
        freqs = torch.exp(
            -np.log(10000.0)
            * torch.arange(half, dtype=torch.float32)
            / half
        )
        self.register_buffer("freqs", freqs)   # not learnable

    def forward(self, t: "torch.Tensor") -> "torch.Tensor":
        # t: [B, 1] or [B]; output: [B, embed_dim]
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        # t * freqs broadcast: [B, 1] * [half] → [B, half]
        args = t * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

**Reference**：与 FQL paper appendix A、DDPM (Ho et al.) 时间嵌入一致。

### 4.2 FlowMatchingTeacher 网络

```python
class FlowMatchingTeacher(_ModuleBase):
    """Velocity field v_θ(x_t, t | s)."""

    def __init__(self, config: FQLConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        self.time_embed = TimeEmbed(config.flow_time_embed_dim)

        # input: concat(x_t [B, A], s [B, O], t_embed [B, T]) → [B, A + O + T]
        input_dim = config.action_dim + config.obs_dim + config.flow_time_embed_dim
        self.net = MLP(
            input_dim,
            config.hidden_dim,
            config.action_dim,
            use_layernorm=config.teacher_use_layernorm,
            dropout_rate=config.dropout_rate,
            num_hidden_layers=config.num_hidden_layers,
        )

    def forward(
        self,
        x_t: "torch.Tensor",       # [B, A]
        t: "torch.Tensor",         # [B, 1]
        obs: "torch.Tensor",       # [B, O]
    ) -> "torch.Tensor":           # [B, A]
        t_embed = self.time_embed(t)   # [B, T]
        x_in = torch.cat([x_t, obs, t_embed], dim=-1)
        return self.net(x_in)           # NO tanh — velocity is unbounded

    @torch.no_grad()
    def integrate(
        self,
        obs: "torch.Tensor",       # [B, O]
        n_steps: int | None = None,
    ) -> "torch.Tensor":           # [B, A]
        """Euler ODE integration from x_0 ~ N(0, I) to x_1."""
        if n_steps is None:
            n_steps = self.config.flow_steps
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        B = obs.shape[0]
        x = torch.randn(B, self.config.action_dim, device=obs.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B, 1), fill_value=i * dt, device=obs.device)
            v = self.forward(x, t, obs)
            x = x + dt * v
        return x   # [B, A]; NOT clamped — student learns tanh-bounded via BC
```

**关键 caveat**：
- `integrate` 必须 wrap `@torch.no_grad()` —— student loss 路径上 teacher 是 frozen target
- `n_steps` 默认从 config 读，allows ablation override（P3 mix ratio sweep 时可用同一 teacher 跑 n_steps=1 / 5 / 10 比较）

### 4.3 Flow loss computation

不在 module 内（保持 module 是 forward-only），放在 `FQLAgent._teacher_loss(batch)`：

```python
def _teacher_loss(self, batch: dict) -> tuple[Tensor, dict[str, float]]:
    obs = self._normalize_obs(batch["obs"])      # [B, O]
    actions = batch["actions"]                    # [B, A]
    B = actions.shape[0]

    t = torch.rand(B, 1, device=actions.device)   # [B, 1] uniform on [0, 1]
    x_0 = torch.randn_like(actions)               # [B, A]
    x_t = (1.0 - t) * x_0 + t * actions           # [B, A]
    v_target = actions - x_0                       # [B, A]

    v_pred = self.teacher(x_t, t, obs)             # [B, A]
    loss = F.mse_loss(v_pred, v_target)
    return loss, {"loss_flow": float(loss.item())}
```

---

## 5. DistilledStudent 设计

**几乎等于 ReBRAC `DeterministicActor`**，只是显式 rename + 保留 tanh bound：

```python
class DistilledStudent(_ModuleBase):
    """One-step deterministic actor π_φ(s) → a ∈ [-1, 1]^A."""

    def __init__(self, config: FQLConfig) -> None:
        require_torch()
        super().__init__()
        self.net = MLP(
            config.obs_dim,
            config.hidden_dim,
            config.action_dim,
            use_layernorm=config.actor_use_layernorm,
            dropout_rate=config.dropout_rate,
            num_hidden_layers=config.num_hidden_layers,
        )

    def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
        return torch.tanh(self.net(obs))
```

**Verification**：layout 与 ReBRAC `DeterministicActor` 1:1，差别仅在 config type (FQLConfig vs ReBRACConfig)。如果 future refactor 想合并，可以引入 protocol type，但此次不做。

---

## 6. Critic（继承 ReBRAC 的 QNetwork）

**直接 `from .sac import QNetwork`，不重新实现**。配合 `FQLConfig` 实例：

```python
from .sac import QNetwork

# 在 FQLAgent.__init__：
self.q1 = QNetwork(config).to(self.device)     # QNetwork 读取 config.obs_dim, action_dim, hidden_dim, critic_use_layernorm
self.q2 = QNetwork(config).to(self.device)
self.q1_target = QNetwork(config).to(self.device)
self.q2_target = QNetwork(config).to(self.device)
```

**前提验证**：`QNetwork.__init__` 必须接受 `FQLConfig` 与 `ReBRACConfig` 形态等价的对象 —— 检查 `auv_nav/sac.py` `QNetwork.__init__` 的字段访问：

| 访问字段 | ReBRACConfig | FQLConfig | 一致 |
|---|---|---|---|
| `obs_dim` | ✓ | ✓ | ✓ |
| `action_dim` | ✓ | ✓ | ✓ |
| `hidden_dim` | ✓ | ✓ | ✓ |
| `num_hidden_layers` | ✓ | ✓ | ✓ |
| `critic_use_layernorm` | ✓ | ✓ | ✓ |
| `dropout_rate` | ✓ | ✓ | ✓ |

如果 `QNetwork` 还访问其他字段（如 `privileged_obs_dim`），需在 FQLConfig 补 `privileged_obs_dim: int = 0` 默认（plan v1 vanilla critic）。**implementer 必须验证此项 before 实现**，否则会 silent fail。

**Asym critic 故意不集成**：plan v1 §3.5 明确 sensor envelope 不是本 paper 轴；asym critic 是 follow-up scope。

---

## 7. FQLAgent.update — 训练 loop

按 §1.4 + §1.3 + §1.1 顺序在每个 `update(batch)` 内执行：

```python
def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
    obs = self._normalize_obs(batch["obs"])
    next_obs = self._normalize_obs(batch["next_obs"])
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["dones"]

    # ─── 1. Teacher update (every step) ───
    loss_flow, flow_metrics = self._teacher_loss(batch)
    self.teacher_opt.zero_grad(set_to_none=True)
    loss_flow.backward()
    teacher_grad_norm = nn.utils.clip_grad_norm_(
        self.teacher.parameters(),
        self.config.grad_clip_norm,
    )
    self.teacher_opt.step()

    # ─── 2. Critic update (every step, no critic-side BC) ───
    with torch.no_grad():
        noise = torch.randn_like(actions) * self.config.policy_noise
        noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
        next_actions = (self.student_target(next_obs) + noise).clamp(-1.0, 1.0)
        target_q = torch.min(
            self.q1_target(next_obs, next_actions),
            self.q2_target(next_obs, next_actions),
        )
        q_target = rewards + self.config.gamma * (1.0 - dones) * target_q

    q1_pred = self.q1(obs, actions)
    q2_pred = self.q2(obs, actions)
    q1_loss = F.mse_loss(q1_pred, q_target)
    q2_loss = F.mse_loss(q2_pred, q_target)

    self.q1_opt.zero_grad(set_to_none=True)
    q1_loss.backward()
    q1_grad_norm = nn.utils.clip_grad_norm_(self.q1.parameters(), self.config.grad_clip_norm)
    self.q1_opt.step()

    self.q2_opt.zero_grad(set_to_none=True)
    q2_loss.backward()
    q2_grad_norm = nn.utils.clip_grad_norm_(self.q2.parameters(), self.config.grad_clip_norm)
    self.q2_opt.step()

    # ─── 3. Student actor update (policy_freq steps) ───
    self.update_count += 1
    should_update_actor = (self.update_count % max(1, self.config.policy_freq)) == 0

    if should_update_actor:
        with torch.no_grad():
            a_teacher = self.teacher.integrate(obs, n_steps=self.config.flow_steps)
        a_student = self.student(obs)
        q_min = torch.min(self.q1(obs, a_student), self.q2(obs, a_student))

        if self.config.normalize_q:
            lambda_coef = q_min.abs().mean().detach().clamp_min(1e-6).reciprocal()
        else:
            lambda_coef = torch.ones((), dtype=torch.float32, device=obs.device)

        bc_loss = (a_student - a_teacher).pow(2).mean()   # mean over (B, A)
        actor_loss = -lambda_coef * q_min.mean() + self.config.distill_alpha_bc * bc_loss

        self.student_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        student_grad_norm = nn.utils.clip_grad_norm_(self.student.parameters(), self.config.grad_clip_norm)
        self.student_opt.step()
        self._soft_update_targets()
        self._last_actor_metrics = {
            "actor_loss": float(actor_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "mean_q": float(q_min.detach().mean().item()),
            "lambda": float(lambda_coef.item()),
            "student_grad_norm": float(student_grad_norm.item()),
        }
        self._has_actor_metrics = True
    elif not self._has_actor_metrics:
        # 初次 sentinel：avoid NaN in CSV
        ...

    return {**self._last_actor_metrics, "loss_flow": ..., "q1_loss": ..., ...}
```

**关键不变式**：
- Teacher update 用 frozen `actions`（不通过 student loss 影响 teacher）
- Critic update 用 `student_target` 算 next action（**不**用 `teacher.integrate` —— next action 来自 distilled policy，不每 step 跑昂贵 ODE）
- Student update 才调用 `teacher.integrate`，并 `no_grad` 把 teacher 锁死
- Student target update 走 `policy_freq` gating，与 ReBRAC 一致

### 7.1 _soft_update_targets

只更新 student & critic targets；teacher **不需要 target**（teacher 是 stable BC objective，不参与 TD chain）：

```python
def _soft_update_targets(self) -> None:
    tau = self.config.tau
    with torch.no_grad():
        for src, tgt in zip(self.student.parameters(), self.student_target.parameters(), strict=True):
            tgt.data.mul_(1.0 - tau).add_(tau * src.data)
        for src, tgt in zip(self.q1.parameters(), self.q1_target.parameters(), strict=True):
            tgt.data.mul_(1.0 - tau).add_(tau * src.data)
        for src, tgt in zip(self.q2.parameters(), self.q2_target.parameters(), strict=True):
            tgt.data.mul_(1.0 - tau).add_(tau * src.data)
```

---

## 8. Metrics dict 契约

`FQLAgent.update(batch)` 必须返回以下 keys（**train_offline.py 写入 CSV 的 schema**）：

| Key | 类型 | 来源 | 是否每 step 更新 |
|---|---|---|---|
| `loss_flow` | float | §7 step 1 | ✓ 每 step |
| `q1_loss` | float | §7 step 2 | ✓ |
| `q2_loss` | float | §7 step 2 | ✓ |
| `critic_loss` | float | `0.5*(q1_loss+q2_loss)` | ✓ |
| `target_q` | float | `q_target.detach().mean()` | ✓ |
| `td_abs_error` | float | `(q1_pred - q_target).abs().mean()` | ✓ |
| `actor_loss` | float | §7 step 3 | only every `policy_freq` |
| `bc_loss` | float | §7 step 3 | only every `policy_freq` |
| `mean_q` | float | `q_min.mean()` | only every `policy_freq` |
| `lambda` | float | Q-norm coef | only every `policy_freq` |
| `teacher_grad_norm` | float | from clip_grad_norm_ | ✓ |
| `q1_grad_norm` / `q2_grad_norm` | float | from clip_grad_norm_ | ✓ |
| `student_grad_norm` | float | from clip_grad_norm_ | only every `policy_freq` |
| `policy_updated` | float (0/1) | actor 是否本 step 更新 | ✓ |

**Convention**：当 `actor_loss` 那组未更新时，返回 `self._last_actor_metrics`（与 ReBRAC `_has_actor_metrics` 模式一致）。

---

## 9. Checkpoint schema + FQLPolicy

### 9.1 Save payload

```python
payload = {
    "config": asdict(self.config),
    "teacher": self.teacher.state_dict(),
    "student": self.student.state_dict(),
    "student_target": self.student_target.state_dict(),
    "q1": self.q1.state_dict(),
    "q2": self.q2.state_dict(),
    "q1_target": self.q1_target.state_dict(),
    "q2_target": self.q2_target.state_dict(),
    "teacher_opt": self.teacher_opt.state_dict(),
    "student_opt": self.student_opt.state_dict(),
    "q1_opt": self.q1_opt.state_dict(),
    "q2_opt": self.q2_opt.state_dict(),
    "obs_normalizer": self.obs_normalizer.state.as_dict() if ... else None,
    "update_count": self.update_count,
    "algorithm": "fql",
}
torch.save(payload, path)
```

### 9.2 FQLPolicy（eval workers 用）

仿 `ReBRACPolicy`：只需 student（不需要 teacher / critics）：

```python
class FQLPolicy:
    """Lightweight deterministic policy wrapper for evaluation workers."""

    def __init__(
        self,
        config: FQLConfig,
        *,
        student_state: dict[str, Any],
        obs_normalizer: ObservationNormalizer | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        require_torch()
        self.config = config
        self.device = torch.device(device)
        self.obs_normalizer = obs_normalizer or ObservationNormalizer.identity(...)
        self.student = DistilledStudent(config).to(self.device)
        self.student.load_state_dict(student_state)
        self.student.eval()

    @classmethod
    def from_payload(cls, payload: dict, *, device: str = "cpu") -> "FQLPolicy":
        config = FQLConfig(**payload["config"])
        # ... extract normalizer, build policy ...
        return cls(config, student_state=payload["student"], ...)

    def reset_policy_state(self) -> None:
        return None

    @_no_grad()
    def act(self, obs, policy_state=None, deterministic=True):
        # 与 ReBRACPolicy.act 几乎相同
        ...
```

**关键**：eval-time 只用 1-step student forward，**不**调用 `teacher.integrate`。这是 FQL 的核心优势 —— inference 不需要 ODE。

---

## 10. Registry 集成

在 `auv_nav/offline_registry.py`：

```python
def normalize_offline_algo(algo: str | None) -> str:
    value = "td3bc" if algo is None else str(algo).strip().lower()
    if value not in {"td3bc", "rebrac", "fql"}:
        raise ValueError(f"Unsupported offline algorithm: {algo!r}")
    return value


def make_agent_config(algo: str, **kwargs: Any) -> Any:
    algo_name = normalize_offline_algo(algo)
    if algo_name == "td3bc":
        from .td3bc import TD3BCConfig
        return TD3BCConfig(**kwargs)
    if algo_name == "fql":
        from .fql import FQLConfig
        return FQLConfig(**kwargs)
    from .rebrac import ReBRACConfig
    return ReBRACConfig(**kwargs)


def make_agent(algo: str, config, *, obs_normalizer=None, device: str = "cpu"):
    algo_name = normalize_offline_algo(algo)
    if algo_name == "td3bc":
        from .td3bc import TD3BCAgent
        return TD3BCAgent(config, obs_normalizer=obs_normalizer, device=device)
    if algo_name == "fql":
        from .fql import FQLAgent
        return FQLAgent(config, obs_normalizer=obs_normalizer, device=device)
    from .rebrac import ReBRACAgent
    return ReBRACAgent(config, obs_normalizer=obs_normalizer, device=device)
```

**Verify**：`make_agent_from_config_dict` (offline_registry.py 后段) 也需要加 `fql` 分支 —— 用于 checkpoint resume。

---

## 11. JAX → PyTorch 已知 pitfall

参考 FQL 官方 JAX repo (`https://github.com/seohongpark/fql`，ICML 2025) 移植时的高风险点：

| Pitfall | JAX 行为 | PyTorch 等价 | 检查方式 |
|---|---|---|---|
| `vmap` 隐式 batching | `jax.vmap` 自动加 leading batch dim | 手动 `unsqueeze(0)` / 显式 `(B, ...)` | 在 `forward` 顶部 assert `x_t.ndim == 2` |
| RNG 状态 | 显式 PRNGKey | 隐式 `torch.Generator` | Smoke test 设 `torch.manual_seed(42)`，反复 `update` loss 序列 deterministic |
| `jax.lax.scan` 训练 loop | 编译期展开 | Python for-loop（足够慢但不影响正确性） | 不必移植；接受性能 ~10x JAX |
| `optax` learning rate schedule | 显式 schedule fn | 用 `torch.optim.lr_scheduler` 或 constant lr | FQL 默认 constant；本计划用 constant，**不**引入 schedule |
| Linear path 公式细节 | `x_t = (1-t)*x_0 + t*a`（FQL 是这版） | 同上 | 不要写成 `x_t = t*x_0 + (1-t)*a`！方向反了 → velocity 反号 |
| `t.reshape(-1, 1, 1)` for broadcasting | JAX broadcast 很宽容 | PyTorch 要求 explicit broadcastable shape | `t` 始终 `[B, 1]`，不 squeeze 也不 unsqueeze 多余维 |
| `jnp.random.normal` device 默认 | JAX random 默认 CPU 再 jit | PyTorch `torch.randn` 必须显式 `device=obs.device` | 所有 random 张量构造都带 `device=...` |
| Velocity field output activation | FQL 用 unbounded（无 tanh） | 同上 | 在 `FlowMatchingTeacher.forward` 末尾**不**加 tanh |
| Sinusoidal time embedding 公式 | DDPM 标准式 | 同 §4.1 | unit test `test_time_embed_t0_t1_distinct` |

**Reference implementations to cross-check**：
1. FQL official JAX: https://github.com/seohongpark/fql （主源）
2. CleanRL flow matching demo (if exists)
3. DDPM PyTorch 时间嵌入：https://github.com/lucidrains/denoising-diffusion-pytorch

---

## 12. Implementation checklist

按此顺序写代码，每完成一项跑对应 test：

- [ ] 0. **Pre-impl verification**：跑 `python -c "from auv_nav.sac import QNetwork; help(QNetwork.__init__)"` 看 QNetwork 实际访问 config 哪些字段；若访问 `privileged_obs_dim` / `dropout_rate` 等，在 FQLConfig 补字段
- [ ] 1. `FQLConfig` dataclass（§3）
- [ ] 2. `TimeEmbed` + `test_time_embed_shape` + `test_time_embed_t0_t1_distinct`
- [ ] 3. `FlowMatchingTeacher.forward` + `test_flow_teacher_shapes`
- [ ] 4. `FlowMatchingTeacher.integrate` + `test_flow_integration_no_nan`
- [ ] 5. `_teacher_loss` 函数 + `test_flow_teacher_overfits_one_action`（保证 algorithm 自洽）
- [ ] 6. `DistilledStudent`（直接复用 ReBRAC pattern）
- [ ] 7. `FQLAgent.__init__` + 3 个 optimizer
- [ ] 8. Critic update 部分（直接搬 ReBRAC `update` 的 critic 段，去掉 critic-side BC penalty）+ `test_fql_critic_update_smoke`
- [ ] 9. `_soft_update_targets`
- [ ] 10. Student actor update 部分 + Q-normalization + `test_fql_actor_update_smoke`
- [ ] 11. `update()` 完整 metrics dict + `test_fql_update_metrics_keys`
- [ ] 12. `save` / `load`
- [ ] 13. `FQLPolicy` + `test_fql_policy_act_shape`
- [ ] 14. `offline_registry.py` 三个 dispatch 改动 + `test_fql_registry_roundtrip`
- [ ] 15. `scripts/train_offline.py` `--algorithm fql` choice 与 FQL CLI args + 100-step smoke test
- [ ] 16. **End-to-end smoke**：在 1000-transition 合成 dataset 上跑 200 step，确认无 NaN / loss 趋势合理

---

## 13. Test plan（pytest）

新文件 `tests/test_fql.py`，至少 8 个 test：

| Test name | 验证 | 期望 |
|---|---|---|
| `test_fql_config_defaults` | FQLConfig 默认值 | `flow_steps=10`, `distill_alpha_bc=1.0`, `critic_use_layernorm=True` |
| `test_time_embed_shape` | TimeEmbed 输出 shape | `[B, T]` for `t: [B, 1]` 和 `t: [B]` 两种 input |
| `test_time_embed_t0_t1_distinct` | t=0 与 t=1 embedding 不同 | `||embed(0) - embed(1)||_2 > 0.1` |
| `test_flow_teacher_shapes` | Teacher.forward 输出 shape | `[B, A]`；无 NaN |
| `test_flow_integration_no_nan` | Teacher.integrate over random seed | 输出无 NaN/Inf；shape `[B, A]` |
| `test_flow_teacher_overfits_one_action` | 用 1-sample dataset，teacher loss → 0 | 1000 step 后 loss < 0.05 |
| `test_fql_update_metrics_keys` | `update(batch)` 返回 dict 包含 §8 所有 keys | set(keys) ⊇ expected_keys |
| `test_fql_critic_update_smoke` | 100 step critic update on random batch | Q losses finite，target_q 不爆炸 (\|target_q\| < 1e4) |
| `test_fql_actor_update_smoke` | 100 step student update | actor_loss finite，bc_loss 递减或 plateau |
| `test_fql_registry_roundtrip` | registry dispatch | `make_agent_config("fql", obs_dim=10, action_dim=2)` 返回 `FQLConfig` 实例；`make_agent("fql", cfg)` 返回 `FQLAgent` |
| `test_fql_checkpoint_save_load` | save → load → policy 输出一致 | 容差 1e-6 |
| `test_fql_policy_act_shape` | FQLPolicy.act on (10,) obs and (B, 10) obs | 返回 `(2,)` 和 `(B, 2)` |

**运行**：

```bash
pytest tests/test_fql.py -v --cov=auv_nav.fql --cov-report=term-missing
```

**覆盖率目标**：`auv_nav/fql.py` ≥ 80% lines。

---

## 14. Acceptance criteria

FQL 实现接受标准（满足后 P0+P1 spec Task B 闭环）：

| 标准 | 通过条件 |
|---|---|
| **Code quality** | `ruff check auv_nav/fql.py` clean；type annotations on all signatures；PEP 8 compliant |
| **Unit test pass rate** | §13 全部 12 个 test 通过 |
| **Coverage** | `auv_nav/fql.py` line coverage ≥ 80% |
| **Registry integration** | `python -m scripts.train_offline --algorithm fql --offline-data <small> --total-steps 100 --device cpu` 100% finish 无错；CSV 含所有 §8 metrics keys |
| **Determinism** | 同 seed 跑两次 `update()`，loss 序列字节级相同 |
| **Memory** | 200k step training peak GPU memory < 2GB（L4 应 ~500MB） |

任一不达标 → 不进 Task E (Gate B sanity)。

---

*Document version: v1.0 (2026-05-19). 维护策略：实现期发现 design 漏洞时回写本 doc + 维护一份 deviation 段；implementer 在 §14 acceptance pass 后给本 doc 打 "verified-by-impl" tag。*
