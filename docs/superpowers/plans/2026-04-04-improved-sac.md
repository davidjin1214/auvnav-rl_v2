# Improved SAC (LayerNorm + DroQ + Asymmetric Critic) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three flag-controlled SAC improvements — LayerNorm, DroQ-style Dropout, and Asymmetric Critic using privileged environment observations — enabling a 5-variant ablation study with zero changes to existing default behavior.

**Architecture:** `SACConfig` gains three new boolean/numeric flags (`use_layernorm`, `dropout_rate`, `privileged_obs_dim`); `MLP` gains two optional parameters; a new `AsymmetricQNetwork` handles the privileged-obs path; the env exposes `privileged_obs` in `info`; `TransitionReplay` optionally stores it; the training loop wires everything together; `run_suite.py` defines ablation variants.

**Tech Stack:** PyTorch, NumPy, Gymnasium, Python 3.10+ dataclasses.

---

## File Map

| File | Change |
|------|--------|
| `auv_nav/networks.py` | Extend `MLP.__init__` with `use_layernorm`, `dropout_rate` |
| `auv_nav/sac.py` | New fields in `SACConfig`; update Actor/Critic; add `AsymmetricQNetwork`; update `SACAgent.update()` |
| `auv_nav/env.py` | Add `privileged_obs` key to `info` dict |
| `auv_nav/replay.py` | Add `privileged_obs_dim` to `TransitionReplayConfig`; optional buffer in `TransitionReplay` |
| `scripts/train_sac.py` | Wire `privileged_obs` through replay add + agent update; add CLI flags |
| `scripts/run_suite.py` | New METHOD_SPECS entries + `sac_ablation_v1` SUITE_PRESET |
| `tests/test_improved_sac.py` | New test file (create) |

---

### Task 1: Extend MLP with LayerNorm and Dropout

**Files:**
- Modify: `auv_nav/networks.py`
- Create: `tests/test_improved_sac.py`

- [ ] **Step 1.1: Create test file with failing tests**

Create `tests/test_improved_sac.py` (requires `tests/__init__.py` — already created in Plan A Task 1 if done, otherwise create it empty):

```python
"""Tests for improved SAC components — no training loop is run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")

from auv_nav.networks import MLP


def _layer_types(module: nn.Module) -> list:
    return [type(m).__name__ for m in module.net]


def test_mlp_baseline_no_layernorm_no_dropout():
    """Default MLP has no LayerNorm or Dropout layers."""
    m = MLP(8, 16, 4)
    types = _layer_types(m)
    assert "LayerNorm" not in types
    assert "Dropout" not in types


def test_mlp_layernorm_present():
    """use_layernorm=True inserts LayerNorm after each hidden Linear."""
    m = MLP(8, 16, 4, use_layernorm=True)
    types = _layer_types(m)
    assert "LayerNorm" in types
    # Two hidden layers → two LayerNorm
    assert types.count("LayerNorm") == 2


def test_mlp_dropout_present():
    """dropout_rate > 0 inserts Dropout after each hidden activation."""
    m = MLP(8, 16, 4, dropout_rate=0.01)
    types = _layer_types(m)
    assert "Dropout" in types
    assert types.count("Dropout") == 2


def test_mlp_layernorm_before_relu():
    """LayerNorm appears immediately before ReLU in each hidden block."""
    m = MLP(8, 16, 4, use_layernorm=True)
    types = _layer_types(m)
    for i, t in enumerate(types):
        if t == "LayerNorm":
            assert types[i + 1] == "ReLU"


def test_mlp_forward_shape():
    """MLP forward pass produces correct output shape."""
    m = MLP(8, 16, 4, use_layernorm=True, dropout_rate=0.01)
    x = torch.randn(5, 8)
    y = m(x)
    assert y.shape == (5, 4)


def test_mlp_backward_runs():
    """Gradients flow through MLP with all options enabled."""
    m = MLP(8, 16, 4, use_layernorm=True, dropout_rate=0.01)
    x = torch.randn(3, 8)
    loss = m(x).sum()
    loss.backward()   # must not raise
```

- [ ] **Step 1.2: Run to confirm failure**

```bash
cd /Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/new_off_rl/rl_v2
python -m pytest tests/test_improved_sac.py::test_mlp_layernorm_present -v
```

Expected: FAILED — `TypeError: MLP.__init__() got unexpected keyword argument 'use_layernorm'`

- [ ] **Step 1.3: Extend MLP in networks.py**

Replace the entire `MLP` class in `auv_nav/networks.py`:

```python
class MLP(_ModuleBase):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_layernorm: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        require_torch()
        super().__init__()
        layers = []
        for i in range(2):
            in_features = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)
```

- [ ] **Step 1.4: Run MLP tests**

```bash
python -m pytest tests/test_improved_sac.py -k "mlp" -v
```

Expected: all 6 MLP tests PASSED.

- [ ] **Step 1.5: Commit**

```bash
git add auv_nav/networks.py tests/test_improved_sac.py
git commit -m "feat: extend MLP with optional LayerNorm and Dropout support"
```

---

### Task 2: Update SACConfig, SquashedGaussianActor, and QNetwork

**Files:**
- Modify: `auv_nav/sac.py`
- Modify: `tests/test_improved_sac.py`

- [ ] **Step 2.1: Write failing tests**

Append to `tests/test_improved_sac.py`:

```python
from auv_nav.sac import SACConfig, SquashedGaussianActor, QNetwork


def _cfg(**kw) -> SACConfig:
    defaults = dict(obs_dim=8, action_dim=2, hidden_dim=16)
    defaults.update(kw)
    return SACConfig(**defaults)


def test_sac_config_defaults_unchanged():
    """Existing SACConfig fields still work with no new args."""
    cfg = _cfg()
    assert cfg.use_layernorm is False
    assert cfg.dropout_rate == 0.0
    assert cfg.privileged_obs_dim == 0


def test_actor_backbone_has_layernorm_when_enabled():
    """SquashedGaussianActor backbone contains LayerNorm when use_layernorm=True."""
    cfg = _cfg(use_layernorm=True)
    actor = SquashedGaussianActor(cfg)
    types = [type(m).__name__ for m in actor.backbone]
    assert "LayerNorm" in types


def test_actor_backbone_no_layernorm_by_default():
    """SquashedGaussianActor backbone has no LayerNorm by default."""
    cfg = _cfg()
    actor = SquashedGaussianActor(cfg)
    types = [type(m).__name__ for m in actor.backbone]
    assert "LayerNorm" not in types


def test_q_network_uses_mlp_layernorm():
    """QNetwork MLP contains LayerNorm when use_layernorm=True."""
    cfg = _cfg(use_layernorm=True)
    q = QNetwork(cfg)
    types = [type(m).__name__ for m in q.q.net]
    assert "LayerNorm" in types


def test_actor_sample_shape():
    """Actor sample returns correct shapes with layernorm enabled."""
    cfg = _cfg(use_layernorm=True, dropout_rate=0.01)
    actor = SquashedGaussianActor(cfg)
    obs = torch.randn(4, 8)
    action, log_prob = actor.sample(obs)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)
```

- [ ] **Step 2.2: Run to confirm failure**

```bash
python -m pytest tests/test_improved_sac.py -k "sac_config_defaults or actor_backbone" -v
```

Expected: FAILED — `TypeError: SACConfig.__init__() got unexpected keyword argument 'use_layernorm'`

- [ ] **Step 2.3: Add new fields to SACConfig**

`SACConfig` is `@dataclass(slots=True)`. Add three fields at the end (after `grad_clip_norm`):

```python
@dataclass(slots=True)
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.995
    tau: float = 0.005
    init_alpha: float = 0.2
    target_entropy: float | None = None
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    batch_size: int = 256
    updates_per_step: int = 1
    grad_clip_norm: float = 10.0
    # --- NEW: architecture improvements ---
    use_layernorm: bool = False
    dropout_rate: float = 0.0       # 0.0 = off; DroQ recommends 0.01
    privileged_obs_dim: int = 0     # 0 = off; set to 2 for asymmetric critic
```

- [ ] **Step 2.4: Update SquashedGaussianActor to use config flags**

In `SquashedGaussianActor.__init__()`, replace the hardcoded backbone:

```python
    def __init__(self, config: SACConfig) -> None:
        require_torch()
        super().__init__()
        self.config = config
        backbone_layers = []
        for i in range(2):
            in_f = config.obs_dim if i == 0 else config.hidden_dim
            backbone_layers.append(nn.Linear(in_f, config.hidden_dim))
            if config.use_layernorm:
                backbone_layers.append(nn.LayerNorm(config.hidden_dim))
            backbone_layers.append(nn.ReLU())
            if config.dropout_rate > 0.0:
                backbone_layers.append(nn.Dropout(p=config.dropout_rate))
        self.backbone = nn.Sequential(*backbone_layers)
        self.mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Linear(config.hidden_dim, config.action_dim)
```

- [ ] **Step 2.5: Update QNetwork to pass flags to MLP**

```python
class QNetwork(_ModuleBase):
    def __init__(self, config: SACConfig) -> None:
        require_torch()
        super().__init__()
        self.q = MLP(
            config.obs_dim + config.action_dim,
            config.hidden_dim,
            1,
            use_layernorm=config.use_layernorm,
            dropout_rate=config.dropout_rate,
        )

    def forward(self, obs: "torch.Tensor", action: "torch.Tensor") -> "torch.Tensor":
        x = torch.cat([obs, action], dim=-1)
        return self.q(x).squeeze(-1)
```

- [ ] **Step 2.6: Run Task 2 tests**

```bash
python -m pytest tests/test_improved_sac.py -k "sac_config or actor or q_network" -v
```

Expected: all 6 tests PASSED.

- [ ] **Step 2.7: Commit**

```bash
git add auv_nav/sac.py tests/test_improved_sac.py
git commit -m "feat: add use_layernorm/dropout_rate to SACConfig, Actor, QNetwork"
```

---

### Task 3: Add AsymmetricQNetwork and update SACAgent

**Files:**
- Modify: `auv_nav/sac.py`
- Modify: `tests/test_improved_sac.py`

- [ ] **Step 3.1: Write failing tests**

Append to `tests/test_improved_sac.py`:

```python
from auv_nav.sac import AsymmetricQNetwork, SACAgent


def test_asymmetric_q_forward_with_privileged():
    """AsymmetricQNetwork forward pass with privileged_obs returns scalar per sample."""
    cfg = _cfg(privileged_obs_dim=3)
    q = AsymmetricQNetwork(cfg)
    obs = torch.randn(5, 8)
    act = torch.randn(5, 2)
    priv = torch.randn(5, 3)
    out = q(obs, act, priv)
    assert out.shape == (5,)


def test_asymmetric_q_forward_without_privileged_uses_zeros():
    """AsymmetricQNetwork forward without privileged_obs zero-pads automatically."""
    cfg = _cfg(privileged_obs_dim=3)
    q = AsymmetricQNetwork(cfg)
    obs = torch.randn(5, 8)
    act = torch.randn(5, 2)
    # No privileged_obs argument
    out = q(obs, act)
    assert out.shape == (5,)


def test_sac_agent_uses_asymmetric_critic_when_enabled():
    """SACAgent instantiates AsymmetricQNetwork when privileged_obs_dim > 0."""
    cfg = _cfg(privileged_obs_dim=2)
    agent = SACAgent(cfg, device="cpu")
    assert isinstance(agent.q1, AsymmetricQNetwork)
    assert isinstance(agent.q2, AsymmetricQNetwork)


def test_sac_agent_uses_standard_critic_by_default():
    """SACAgent uses QNetwork by default (privileged_obs_dim=0)."""
    cfg = _cfg()
    agent = SACAgent(cfg, device="cpu")
    assert isinstance(agent.q1, QNetwork)
    assert isinstance(agent.q2, QNetwork)


def test_sac_agent_update_with_privileged_obs():
    """SACAgent.update() accepts batch with 'privileged_obs' key."""
    cfg = _cfg(privileged_obs_dim=2)
    agent = SACAgent(cfg, device="cpu")
    batch = {
        "obs":            torch.randn(4, 8),
        "actions":        torch.randn(4, 2),
        "rewards":        torch.randn(4),
        "next_obs":       torch.randn(4, 8),
        "dones":          torch.zeros(4),
        "privileged_obs": torch.randn(4, 2),
    }
    metrics = agent.update(batch)
    assert "q1_loss" in metrics


def test_sac_agent_update_without_privileged_obs():
    """SACAgent.update() works normally when 'privileged_obs' not in batch."""
    cfg = _cfg()
    agent = SACAgent(cfg, device="cpu")
    batch = {
        "obs":      torch.randn(4, 8),
        "actions":  torch.randn(4, 2),
        "rewards":  torch.randn(4),
        "next_obs": torch.randn(4, 8),
        "dones":    torch.zeros(4),
    }
    metrics = agent.update(batch)
    assert "q1_loss" in metrics
```

- [ ] **Step 3.2: Run to confirm failure**

```bash
python -m pytest tests/test_improved_sac.py -k "asymmetric" -v
```

Expected: FAILED — `ImportError: cannot import name 'AsymmetricQNetwork'`

- [ ] **Step 3.3: Add AsymmetricQNetwork class**

Add immediately after `QNetwork` in `auv_nav/sac.py`:

```python
class AsymmetricQNetwork(_ModuleBase):
    """Critic that accepts privileged observations during training.

    Input: obs + action + privileged_obs (when provided) or zero-padded.
    The privileged_obs argument should be None at evaluation time.
    """

    def __init__(self, config: SACConfig) -> None:
        require_torch()
        super().__init__()
        self._priv_dim = config.privileged_obs_dim
        total_in = config.obs_dim + config.action_dim + config.privileged_obs_dim
        self.q = MLP(
            total_in,
            config.hidden_dim,
            1,
            use_layernorm=config.use_layernorm,
            dropout_rate=config.dropout_rate,
        )

    def forward(
        self,
        obs: "torch.Tensor",
        action: "torch.Tensor",
        privileged_obs: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        if privileged_obs is None:
            privileged_obs = obs.new_zeros(obs.shape[0], self._priv_dim)
        x = torch.cat([obs, action, privileged_obs], dim=-1)
        return self.q(x).squeeze(-1)
```

- [ ] **Step 3.4: Update SACAgent.__init__() to choose critic class**

In `SACAgent.__init__()`, replace the four QNetwork instantiation lines:

```python
        # Choose standard or asymmetric critic based on config.
        _CriticCls = AsymmetricQNetwork if config.privileged_obs_dim > 0 else QNetwork
        self.q1 = _CriticCls(config).to(self.device)
        self.q2 = _CriticCls(config).to(self.device)
        self.q1_target = _CriticCls(config).to(self.device)
        self.q2_target = _CriticCls(config).to(self.device)
```

- [ ] **Step 3.5: Update SACAgent.update() to pass privileged_obs**

In `SACAgent.update()`, change the critic forward calls. Find the existing `update()` method and replace the critic-related lines:

```python
    def update(self, batch: "dict[str, torch.Tensor]") -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        priv = batch.get("privileged_obs")   # None if not present

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            # Target critics do not use privileged obs (unavailable at eval time)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            q_target = rewards + self.config.gamma * (1.0 - dones) * q_next

        q1_pred = self.q1(obs, actions, priv)    # passes priv to AsymmetricQNetwork
        q2_pred = self.q2(obs, actions, priv)    # QNetwork.forward() ignores extra args
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        # ... rest of update() unchanged ...
```

Note: `QNetwork.forward(obs, action)` only accepts two positional args. To avoid needing to handle this difference, update `QNetwork.forward()` to accept and ignore an optional third argument:

```python
class QNetwork(_ModuleBase):
    ...
    def forward(
        self,
        obs: "torch.Tensor",
        action: "torch.Tensor",
        _privileged_obs: "torch.Tensor | None" = None,  # ignored, for uniform API
    ) -> "torch.Tensor":
        x = torch.cat([obs, action], dim=-1)
        return self.q(x).squeeze(-1)
```

Also update actor loss lines which call `self.q1(obs, pi_action)` — add `None` for privileged_obs:

```python
        pi_action, log_prob = self.actor.sample(obs)
        q1_pi = self.q1(obs, pi_action)   # no priv — actor optimized without it
        q2_pi = self.q2(obs, pi_action)
```

These are unchanged since we want the actor to learn without privileged info.

- [ ] **Step 3.6: Run Task 3 tests**

```bash
python -m pytest tests/test_improved_sac.py -v
```

Expected: all tests PASSED (no failures).

- [ ] **Step 3.7: Commit**

```bash
git add auv_nav/sac.py tests/test_improved_sac.py
git commit -m "feat: add AsymmetricQNetwork and update SACAgent for privileged critic"
```

---

### Task 4: Expose privileged_obs in environment info

**Files:**
- Modify: `auv_nav/env.py`
- Modify: `tests/test_improved_sac.py`

- [ ] **Step 4.1: Write failing test**

Append to `tests/test_improved_sac.py`:

```python
import numpy as np


def test_env_make_info_has_privileged_obs():
    """_make_info() dict includes 'privileged_obs' sliced from equivalent_current_body."""
    from unittest.mock import MagicMock, patch
    from auv_nav.env import PlanarRemusEnv

    # Build a minimal fake env object (bypasses __init__ entirely).
    env = object.__new__(PlanarRemusEnv)
    env.last_equivalent_current_world = np.zeros(3, dtype=np.float32)
    env.last_equivalent_current_body = np.array([0.7, -0.4, 0.1], dtype=np.float32)
    env.last_center_current_world = np.zeros(3, dtype=np.float32)
    env.reference_flow_world = np.zeros(3, dtype=np.float32)
    env.reference_flow_speed_mps = 0.0
    env.elapsed_time = 0.0
    env.cumulative_rpm = 0.0
    env.cumulative_safety_cost = 0.0
    env.step_count = 0
    env.last_reward_breakdown = MagicMock(
        task=0.0, progress=0.0, terminal=0.0,
        energy=0.0, safety=0.0, total=0.0,
    )

    from auv_nav.vehicle import StateIndex as S
    state = np.zeros(12, dtype=np.float64)
    env.state = state

    cfg = MagicMock()
    cfg.goal_radius_m = 4.0
    cfg.max_episode_time_s = 240.0
    cfg.z_ref_m = 0.0
    cfg.depth_fail_tol_m = 1.5
    cfg.roll_fail_tol_rad = 0.785
    cfg.pitch_fail_tol_rad = 0.61
    cfg.max_body_speed_mps = 3.5
    cfg.max_relative_speed_mps = 4.5
    cfg.max_angular_rate_radps = 5.0
    env.config = cfg

    task = MagicMock()
    task.goal_xy_world = np.zeros(2, dtype=np.float32)
    task.start_xy_world = np.zeros(2, dtype=np.float32)
    task.task_geometry = "downstream"
    task.target_auv_max_speed_mps = 1.5
    env.task = task

    # Patch _build_observation and safety cost so _make_info works standalone
    env.safety_cost_model = MagicMock()
    env.safety_cost_model.compute = MagicMock(return_value=MagicMock(total=0.0))

    # Call _make_info and check the key exists with correct values
    info = env._make_info(reason="timeout")
    assert "privileged_obs" in info
    priv = info["privileged_obs"]
    assert priv.shape == (2,)
    assert priv.dtype == np.float32
    assert float(priv[0]) == pytest.approx(0.7)
    assert float(priv[1]) == pytest.approx(-0.4)
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
python -m pytest tests/test_improved_sac.py::test_env_info_has_privileged_obs -v
```

Expected: FAILED — `AssertionError: 'privileged_obs' not in info`

- [ ] **Step 4.3: Add privileged_obs to env info**

In `auv_nav/env.py`, locate the `_make_info()` method (around L832). It already computes `equivalent_body = self.last_equivalent_current_body` and includes it under `"equivalent_current_body"`.

Find the `return` dict in `_make_info()` (around L865). Add one new key:

```python
            # flow field
            "center_current_world": self.last_center_current_world.astype(np.float32),
            "equivalent_current_world": equivalent_world.astype(np.float32),
            "equivalent_current_body": equivalent_body.astype(np.float32),
            # Privileged observation for asymmetric critic training (not used by actor).
            # Contains body-frame equivalent flow [u_eq, v_eq] in m/s.
            "privileged_obs": equivalent_body[:2].astype(np.float32),
            "reference_flow_world": self.reference_flow_world.copy(),
```

That's the entire change to `env.py` — one line.

- [ ] **Step 4.4: Run test**

```bash
python -m pytest tests/test_improved_sac.py::test_env_info_has_privileged_obs -v
```

Expected: PASSED.

- [ ] **Step 4.5: Commit**

```bash
git add auv_nav/env.py tests/test_improved_sac.py
git commit -m "feat: expose privileged_obs (u_eq, v_eq body frame) in env step info"
```

---

### Task 5: Add privileged_obs buffer to TransitionReplay

**Files:**
- Modify: `auv_nav/replay.py`
- Modify: `tests/test_improved_sac.py`

- [ ] **Step 5.1: Write failing tests**

Append to `tests/test_improved_sac.py`:

```python
from auv_nav.replay import TransitionReplay, TransitionReplayConfig


def test_replay_without_privileged_obs_unchanged():
    """TransitionReplay with no privileged_obs_dim behaves as before."""
    cfg = TransitionReplayConfig(capacity=100)
    replay = TransitionReplay(obs_dim=4, action_dim=2, config=cfg)
    replay.add(
        obs=np.zeros(4), action=np.zeros(2),
        reward=0.0, cost=0.0, next_obs=np.zeros(4), done=False,
    )
    assert len(replay) == 1
    batch = replay.sample_batch(1, device=torch.device("cpu"))
    assert "privileged_obs" not in batch


def test_replay_with_privileged_obs_stores_and_returns():
    """TransitionReplay stores privileged_obs and returns it in sample_batch."""
    cfg = TransitionReplayConfig(capacity=100, privileged_obs_dim=2)
    replay = TransitionReplay(obs_dim=4, action_dim=2, config=cfg)
    priv = np.array([1.5, -0.3], dtype=np.float32)
    replay.add(
        obs=np.zeros(4), action=np.zeros(2),
        reward=0.0, cost=0.0, next_obs=np.zeros(4), done=False,
        privileged_obs=priv,
    )
    assert len(replay) == 1
    batch = replay.sample_batch(1, device=torch.device("cpu"))
    assert "privileged_obs" in batch
    assert batch["privileged_obs"].shape == (1, 2)
    assert float(batch["privileged_obs"][0, 0]) == pytest.approx(1.5)


def test_replay_privileged_obs_none_not_stored():
    """If add() is called without privileged_obs, buffer stays at zeros."""
    cfg = TransitionReplayConfig(capacity=100, privileged_obs_dim=2)
    replay = TransitionReplay(obs_dim=4, action_dim=2, config=cfg)
    replay.add(
        obs=np.zeros(4), action=np.zeros(2),
        reward=0.0, cost=0.0, next_obs=np.zeros(4), done=False,
        # No privileged_obs
    )
    batch = replay.sample_batch(1, device=torch.device("cpu"))
    assert "privileged_obs" in batch   # key still present (zeros)
```

- [ ] **Step 5.2: Run to confirm failure**

```bash
python -m pytest tests/test_improved_sac.py -k "replay" -v
```

Expected: FAILED — `TypeError: TransitionReplayConfig.__init__() got unexpected keyword argument 'privileged_obs_dim'`

- [ ] **Step 5.3: Update TransitionReplayConfig**

`TransitionReplayConfig` is `@dataclass(slots=True)`. Add one field:

```python
@dataclass(slots=True)
class TransitionReplayConfig:
    capacity: int = 1_000_000
    privileged_obs_dim: int = 0   # NEW: 0 = disabled
```

- [ ] **Step 5.4: Update TransitionReplay.__init__()**

In `TransitionReplay.__init__()`, after the existing buffer allocations, add:

```python
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        # Optional privileged observation buffer.
        if config.privileged_obs_dim > 0:
            self.privileged_obs: np.ndarray | None = np.zeros(
                (capacity, config.privileged_obs_dim), dtype=np.float32
            )
        else:
            self.privileged_obs = None
```

- [ ] **Step 5.5: Update TransitionReplay.add()**

Add optional `privileged_obs` parameter:

```python
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        cost: float,
        next_obs: np.ndarray,
        done: bool,
        privileged_obs: np.ndarray | None = None,   # NEW
    ) -> None:
        idx = self.ptr
        self.observations[idx] = np.asarray(obs, dtype=np.float32)
        self.actions[idx] = np.asarray(action, dtype=np.float32)
        self.rewards[idx] = float(reward)
        self.costs[idx] = float(cost)
        self.next_observations[idx] = np.asarray(next_obs, dtype=np.float32)
        self.dones[idx] = float(done)
        if self.privileged_obs is not None:
            if privileged_obs is not None:
                self.privileged_obs[idx] = np.asarray(privileged_obs, dtype=np.float32)
            # else: slot stays as zeros (safe default)
        self.ptr = (self.ptr + 1) % self.config.capacity
        self.size = min(self.size + 1, self.config.capacity)
```

- [ ] **Step 5.6: Update TransitionReplay.sample_batch()**

Add privileged_obs to the returned dict when buffer is active:

```python
    def sample_batch(self, batch_size: int, device: "torch.device") -> "dict[str, torch.Tensor]":
        require_torch()
        if self.size <= 0:
            raise RuntimeError("Replay buffer is empty.")
        batch_size = max(1, int(batch_size))
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs":      torch.as_tensor(self.observations[indices], device=device),
            "actions":  torch.as_tensor(self.actions[indices], device=device),
            "rewards":  torch.as_tensor(self.rewards[indices], device=device),
            "costs":    torch.as_tensor(self.costs[indices], device=device),
            "next_obs": torch.as_tensor(self.next_observations[indices], device=device),
            "dones":    torch.as_tensor(self.dones[indices], device=device),
        }
        if self.privileged_obs is not None:
            batch["privileged_obs"] = torch.as_tensor(
                self.privileged_obs[indices], device=device
            )
        return batch
```

- [ ] **Step 5.7: Run Task 5 tests**

```bash
python -m pytest tests/test_improved_sac.py -k "replay" -v
```

Expected: 3 PASSED.

- [ ] **Step 5.8: Commit**

```bash
git add auv_nav/replay.py tests/test_improved_sac.py
git commit -m "feat: add optional privileged_obs buffer to TransitionReplay"
```

---

### Task 6: Wire privileged_obs through training loop + add CLI flags

**Files:**
- Modify: `scripts/train_sac.py`

- [ ] **Step 6.1: Add CLI flags for new SACConfig fields**

In `train_sac.py`, locate the `argparse` block. Add three new arguments after the existing `--hidden-dim` argument:

```python
    parser.add_argument(
        "--use-layernorm", action="store_true", default=False,
        help="Enable LayerNorm in Actor and Critic networks (DroQ/improved SAC).",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0, metavar="P",
        help="Dropout rate for hidden layers (0.0 = off). DroQ recommends 0.01.",
    )
    parser.add_argument(
        "--use-asymmetric-critic", action="store_true", default=False,
        help="Enable Asymmetric Critic: Critic sees privileged_obs from env info.",
    )
```

- [ ] **Step 6.2: Pass new flags to SACConfig**

Locate the `SACConfig(...)` construction in `train_sac.py` (around L151):

```python
    agent_cfg = SACConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        batch_size=train_cfg.batch_size,
        updates_per_step=train_cfg.updates_per_step,
        use_layernorm=args.use_layernorm,
        dropout_rate=args.dropout_rate,
        privileged_obs_dim=2 if args.use_asymmetric_critic else 0,
    )
```

- [ ] **Step 6.3: Pass privileged_obs_dim to TransitionReplayConfig**

Update the `TransitionReplay` construction (around L159):

```python
    replay = TransitionReplay(
        obs_dim=agent_cfg.obs_dim,
        action_dim=agent_cfg.action_dim,
        config=TransitionReplayConfig(
            capacity=train_cfg.replay_capacity,
            privileged_obs_dim=agent_cfg.privileged_obs_dim,
        ),
    )
```

- [ ] **Step 6.4: Extract privileged_obs from info and pass to replay.add()**

In the training loop, locate the `has_step_cost` detection block (around L200):

```python
            has_final_obs = "final_observation" in info
            has_final_info = "final_info" in info
            has_step_cost = "step_safety_cost" in info
```

Add:
```python
            has_priv_obs = "privileged_obs" in info
```

Then, in the per-environment loop, add `privileged_obs` extraction before `replay.add()`:

```python
                step_cost = float(info["step_safety_cost"][i]) if has_step_cost else 0.0
                priv_obs = info["privileged_obs"][i] if has_priv_obs else None

                replay.add(
                    obs=obs[i],
                    action=action[i],
                    reward=float(reward[i]),
                    cost=step_cost,
                    next_obs=real_next_obs,
                    done=bool(terminated[i]),
                    privileged_obs=priv_obs,
                )
```

- [ ] **Step 6.5: Smoke test — baseline training still works**

```bash
python -m scripts.train_sac --total-steps 500 --random-steps 100 --update-after 100 \
  --num-envs 1 --eval-every 500 2>&1 | tail -5
```

Expected: runs without error, prints something like `[train] episode=... step=500`.

- [ ] **Step 6.6: Smoke test — DroQ variant**

```bash
python -m scripts.train_sac --total-steps 500 --random-steps 100 --update-after 100 \
  --num-envs 1 --eval-every 500 \
  --use-layernorm --dropout-rate 0.01 --updates-per-step 4 2>&1 | tail -5
```

Expected: runs without error.

- [ ] **Step 6.7: Smoke test — Asymmetric Critic**

```bash
python -m scripts.train_sac --total-steps 500 --random-steps 100 --update-after 100 \
  --num-envs 1 --eval-every 500 \
  --use-layernorm --use-asymmetric-critic 2>&1 | tail -5
```

Expected: runs without error.

- [ ] **Step 6.8: Commit**

```bash
git add scripts/train_sac.py
git commit -m "feat: wire privileged_obs and LayerNorm/DroQ/AsymmetricCritic through training loop"
```

---

### Task 7: Add ablation METHOD_SPECS and SUITE_PRESET to run_suite.py

**Files:**
- Modify: `scripts/run_suite.py`

- [ ] **Step 7.1: Add ablation METHOD_SPECS entries**

In `run_suite.py`, locate `METHOD_SPECS` (around L28). Add four new entries after `"sac"`:

```python
METHOD_SPECS: dict[str, MethodSpec] = {
    "sac": MethodSpec(
        key="sac",
        train_module="scripts.train_sac",
        extra_args=("--history-length", "1"),
        description="Baseline SAC (K=1, no improvements)",
        values={},
    ),
    "sac_ln_k16": MethodSpec(
        key="sac_ln_k16",
        train_module="scripts.train_sac",
        extra_args=("--use-layernorm", "--history-length", "16"),
        description="SAC + LayerNorm + history K=16",
        values={},
    ),
    "sac_droq": MethodSpec(
        key="sac_droq",
        train_module="scripts.train_sac",
        extra_args=(
            "--use-layernorm", "--dropout-rate", "0.01",
            "--updates-per-step", "4", "--history-length", "16",
        ),
        description="DroQ: SAC + LayerNorm + Dropout(0.01) + UTD=4 + K=16",
        values={},
    ),
    "sac_asym_k16": MethodSpec(
        key="sac_asym_k16",
        train_module="scripts.train_sac",
        extra_args=(
            "--use-layernorm", "--use-asymmetric-critic", "--history-length", "16",
        ),
        description="SAC + LayerNorm + Asymmetric Critic + K=16",
        values={},
    ),
    "sac_full": MethodSpec(
        key="sac_full",
        train_module="scripts.train_sac",
        extra_args=(
            "--use-layernorm", "--dropout-rate", "0.01",
            "--updates-per-step", "4",
            "--use-asymmetric-critic", "--history-length", "16",
        ),
        description="SAC + all improvements: LN + DroQ + AsymCritic + UTD=4 + K=16",
        values={},
    ),
}
```

- [ ] **Step 7.2: Add sac_ablation_v1 SUITE_PRESET**

In `SUITE_PRESETS`, add a new entry after the existing preset:

```python
    "sac_ablation_v1": SuitePreset(
        name="sac_ablation_v1",
        description=(
            "5-variant SAC ablation study: baseline vs LN+K16 vs DroQ "
            "vs AsymCritic vs Full. 3 seeds each."
        ),
        values={
            "methods": "sac,sac_ln_k16,sac_droq,sac_asym_k16,sac_full",
            "num_seeds": 3,
            "total_steps": 200_000,
            "difficulty": "medium",
            "task_geometry": "cross_stream",
            "num_envs": 8,
        },
    ),
```

- [ ] **Step 7.3: Smoke test — list presets**

```bash
python -m scripts.run_suite --help 2>&1 | grep -A5 "preset"
```

Expected output contains `sac_ablation_v1` in the choices list.

- [ ] **Step 7.4: Verify new specs are importable and correctly keyed**

```bash
python -c "
from scripts.run_suite import METHOD_SPECS, SUITE_PRESETS
print('Methods:', sorted(METHOD_SPECS.keys()))
print('Presets:', sorted(SUITE_PRESETS.keys()))
assert 'sac_ln_k16'   in METHOD_SPECS, 'missing sac_ln_k16'
assert 'sac_droq'     in METHOD_SPECS, 'missing sac_droq'
assert 'sac_asym_k16' in METHOD_SPECS, 'missing sac_asym_k16'
assert 'sac_full'     in METHOD_SPECS, 'missing sac_full'
assert 'sac_ablation_v1' in SUITE_PRESETS, 'missing suite preset'
print('All checks passed.')
"
```

Expected:
```
Methods: ['sac', 'sac_asym_k16', 'sac_droq', 'sac_full', 'sac_ln_k16']
Presets: ['medium_formal_v1', 'sac_ablation_v1']
All checks passed.
```

- [ ] **Step 7.5: Commit**

```bash
git add scripts/run_suite.py
git commit -m "feat: add SAC ablation METHOD_SPECS and sac_ablation_v1 suite preset"
```

---

## How to Run the Ablation Study

```bash
# Full ablation: 5 variants × 3 seeds (runs sequentially by default)
python -m scripts.run_suite --preset sac_ablation_v1

# Quick test: single method, 1 seed
python -m scripts.train_sac \
  --use-layernorm --dropout-rate 0.01 --updates-per-step 4 \
  --history-length 16 --total-steps 50000 --num-envs 8

# Asymmetric critic only
python -m scripts.train_sac \
  --use-layernorm --use-asymmetric-critic --history-length 16 \
  --total-steps 200000 --num-envs 8
```

## Ablation Variants Reference

| Variant key | LayerNorm | Dropout | UTD | Asym. Critic | K |
|-------------|-----------|---------|-----|--------------|---|
| `sac` | ✗ | ✗ | 1 | ✗ | 1 |
| `sac_ln_k16` | ✓ | ✗ | 1 | ✗ | 16 |
| `sac_droq` | ✓ | 0.01 | 4 | ✗ | 16 |
| `sac_asym_k16` | ✓ | ✗ | 1 | ✓ | 16 |
| `sac_full` | ✓ | 0.01 | 4 | ✓ | 16 |
