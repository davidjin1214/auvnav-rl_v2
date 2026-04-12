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


# --- Task 2: SACConfig, Actor, QNetwork ---

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


# --- Task 3: AsymmetricQNetwork, SACAgent ---

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
    """SACAgent.update() accepts current/next privileged critic inputs."""
    cfg = _cfg(privileged_obs_dim=2)
    agent = SACAgent(cfg, device="cpu")
    batch = {
        "obs":            torch.randn(4, 8),
        "actions":        torch.randn(4, 2),
        "rewards":        torch.randn(4),
        "next_obs":       torch.randn(4, 8),
        "dones":          torch.zeros(4),
        "privileged_obs": torch.randn(4, 2),
        "next_privileged_obs": torch.randn(4, 2),
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


# --- Task 4: env privileged_obs ---

def test_env_build_info_has_privileged_obs_key():
    """env.py _build_info() contains 'privileged_obs' key mapped from equivalent_body[:2]."""
    from pathlib import Path
    env_source = Path(__file__).parent.parent / "auv_nav" / "env.py"
    source = env_source.read_text()
    assert '"privileged_obs"' in source, "env.py missing 'privileged_obs' key"
    # Verify it slices equivalent_body[:2]
    assert "equivalent_body[:2].astype(np.float32)" in source, (
        "env.py does not slice equivalent_body[:2] for privileged_obs"
    )


# --- Task 5: TransitionReplay privileged_obs ---

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
    """TransitionReplay stores current/next privileged_obs and returns both."""
    cfg = TransitionReplayConfig(capacity=100, privileged_obs_dim=2)
    replay = TransitionReplay(obs_dim=4, action_dim=2, config=cfg)
    priv = np.array([1.5, -0.3], dtype=np.float32)
    next_priv = np.array([-0.2, 0.7], dtype=np.float32)
    replay.add(
        obs=np.zeros(4), action=np.zeros(2),
        reward=0.0, cost=0.0, next_obs=np.zeros(4), done=False,
        privileged_obs=priv,
        next_privileged_obs=next_priv,
    )
    assert len(replay) == 1
    batch = replay.sample_batch(1, device=torch.device("cpu"))
    assert "privileged_obs" in batch
    assert "next_privileged_obs" in batch
    assert batch["privileged_obs"].shape == (1, 2)
    assert batch["next_privileged_obs"].shape == (1, 2)
    assert float(batch["privileged_obs"][0, 0]) == pytest.approx(1.5)
    assert float(batch["next_privileged_obs"][0, 1]) == pytest.approx(0.7)


def test_replay_privileged_obs_none_not_stored():
    """If add() is called without privileged_obs, buffers stay at zeros."""
    cfg = TransitionReplayConfig(capacity=100, privileged_obs_dim=2)
    replay = TransitionReplay(obs_dim=4, action_dim=2, config=cfg)
    replay.add(
        obs=np.zeros(4), action=np.zeros(2),
        reward=0.0, cost=0.0, next_obs=np.zeros(4), done=False,
        # No privileged_obs
    )
    batch = replay.sample_batch(1, device=torch.device("cpu"))
    assert "privileged_obs" in batch   # key still present (zeros)
    assert "next_privileged_obs" in batch
    assert torch.allclose(batch["privileged_obs"], torch.zeros_like(batch["privileged_obs"]))
    assert torch.allclose(
        batch["next_privileged_obs"],
        torch.zeros_like(batch["next_privileged_obs"]),
    )


def test_replay_from_npz_loads_privileged_obs(tmp_path: Path):
    """TransitionReplay.from_npz() restores optional privileged observation arrays."""
    path = tmp_path / "offline.npz"
    np.savez_compressed(
        path,
        obs=np.zeros((3, 4), dtype=np.float32),
        actions=np.zeros((3, 2), dtype=np.float32),
        rewards=np.zeros(3, dtype=np.float32),
        costs=np.zeros(3, dtype=np.float32),
        next_obs=np.zeros((3, 4), dtype=np.float32),
        dones=np.zeros(3, dtype=np.float32),
        privileged_obs=np.ones((3, 2), dtype=np.float32),
        next_privileged_obs=2.0 * np.ones((3, 2), dtype=np.float32),
    )
    replay = TransitionReplay.from_npz(path)
    batch = replay.sample_batch(2, device=torch.device("cpu"))
    assert replay.config.privileged_obs_dim == 2
    assert "privileged_obs" in batch
    assert "next_privileged_obs" in batch


def test_dual_buffer_sampler_fills_missing_optional_keys():
    """DualBufferSampler zero-fills optional keys when only one buffer has them."""
    offline = TransitionReplay(obs_dim=4, action_dim=2, config=TransitionReplayConfig(capacity=10))
    online = TransitionReplay(
        obs_dim=4,
        action_dim=2,
        config=TransitionReplayConfig(capacity=10, privileged_obs_dim=2),
    )
    offline.add(
        obs=np.zeros(4), action=np.zeros(2),
        reward=0.0, cost=0.0, next_obs=np.zeros(4), done=False,
    )
    online.add(
        obs=np.zeros(4), action=np.zeros(2),
        reward=0.0, cost=0.0, next_obs=np.zeros(4), done=False,
        privileged_obs=np.ones(2, dtype=np.float32),
        next_privileged_obs=np.ones(2, dtype=np.float32),
    )
    from auv_nav.replay import DualBufferSampler
    sampler = DualBufferSampler(offline, online, offline_ratio=0.5)
    batch = sampler.sample_batch(2, device=torch.device("cpu"))
    assert "privileged_obs" in batch
    assert "next_privileged_obs" in batch
    assert batch["privileged_obs"].shape == (2, 2)
    assert batch["next_privileged_obs"].shape == (2, 2)
