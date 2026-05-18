"""Unit tests for the FQL agent (``auv_nav/fql.py``).

Test plan anchored in ``docs/fql_pytorch_port_design.md`` §13 (12 tests).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from auv_nav.fql import (
    DistilledStudent,
    FlowMatchingTeacher,
    FQLAgent,
    FQLConfig,
    FQLPolicy,
    TimeEmbed,
)
from auv_nav.offline_registry import (
    make_agent,
    make_agent_config,
    policy_from_payload,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_batch(
    batch_size: int = 32,
    obs_dim: int = 10,
    action_dim: int = 2,
    seed: int = 0,
) -> tuple[dict, np.ndarray]:
    """Synthetic transition batch suitable for FQLAgent.update()."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(batch_size, obs_dim)).astype(np.float32)
    next_obs = rng.normal(size=(batch_size, obs_dim)).astype(np.float32)
    actions = np.clip(
        rng.normal(size=(batch_size, action_dim)), -1.0, 1.0
    ).astype(np.float32)
    rewards = rng.normal(size=(batch_size,)).astype(np.float32)
    dones = rng.integers(
        0, 2, size=(batch_size,), endpoint=False
    ).astype(np.float32)
    batch = {
        "obs": torch.as_tensor(obs),
        "next_obs": torch.as_tensor(next_obs),
        "actions": torch.as_tensor(actions),
        "rewards": torch.as_tensor(rewards),
        "dones": torch.as_tensor(dones),
    }
    return batch, obs


# ---------------------------------------------------------------------------
# Section 1: config defaults
# ---------------------------------------------------------------------------


def test_fql_config_defaults() -> None:
    cfg = FQLConfig(obs_dim=10, action_dim=2)
    assert cfg.flow_steps == 10
    assert cfg.flow_time_embed_dim == 32
    assert cfg.distill_alpha_bc == pytest.approx(1.0)
    assert cfg.critic_use_layernorm is True
    assert cfg.actor_use_layernorm is False
    assert cfg.normalize_q is True
    assert cfg.policy_freq == 2
    assert cfg.gamma == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Section 2: TimeEmbed
# ---------------------------------------------------------------------------


def test_time_embed_shape() -> None:
    embed = TimeEmbed(32)
    # shape [B, 1]
    t = torch.linspace(0.0, 1.0, 8).unsqueeze(-1)
    out = embed(t)
    assert out.shape == (8, 32)
    # shape [B] (no trailing 1) — the layer should auto-unsqueeze.
    t1 = torch.linspace(0.0, 1.0, 8)
    out1 = embed(t1)
    assert out1.shape == (8, 32)
    # No NaN.
    assert torch.isfinite(out).all()


def test_time_embed_t0_t1_distinct() -> None:
    embed = TimeEmbed(32)
    e0 = embed(torch.tensor([[0.0]]))
    e1 = embed(torch.tensor([[1.0]]))
    # Two embeddings must differ by a non-trivial L2 distance.
    distance = (e0 - e1).norm().item()
    assert distance > 0.1, (
        f"TimeEmbed at t=0 and t=1 nearly identical (||diff||={distance:.4f})"
    )


def test_time_embed_invalid_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        TimeEmbed(31)


# ---------------------------------------------------------------------------
# Section 3 + 4: FlowMatchingTeacher forward / integrate
# ---------------------------------------------------------------------------


def test_flow_teacher_shapes() -> None:
    cfg = FQLConfig(obs_dim=10, action_dim=2, hidden_dim=64)
    teacher = FlowMatchingTeacher(cfg)
    B = 16
    x_t = torch.randn(B, cfg.action_dim)
    t = torch.rand(B, 1)
    obs = torch.randn(B, cfg.obs_dim)
    out = teacher(x_t, t, obs)
    assert out.shape == (B, cfg.action_dim)
    assert torch.isfinite(out).all()


def test_flow_integration_no_nan() -> None:
    cfg = FQLConfig(obs_dim=10, action_dim=2, hidden_dim=64, flow_steps=5)
    teacher = FlowMatchingTeacher(cfg)
    B = 8
    obs = torch.randn(B, cfg.obs_dim)
    a = teacher.integrate(obs)
    assert a.shape == (B, cfg.action_dim)
    assert torch.isfinite(a).all()


def test_flow_teacher_overfits_one_action() -> None:
    """Single (obs, action) sample: teacher's loss must decrease substantially.

    Flow-matching MSE is intrinsically noisy: ``v_target = a - x_0`` is
    sampled fresh per step, and as ``t -> 1`` the conditional variance
    diverges (``x_0 = (x_t - t*a) / (1 - t)``). Most flow-matching repos
    work around this by clipping ``t`` or biasing toward small ``t``; we
    keep the unbiased ``U(0, 1)`` sampling and instead test that the loss
    has a clear downward trend over many steps (a sanity check that the
    network is fitting the velocity field, not just noise).
    """
    torch.manual_seed(0)
    cfg = FQLConfig(
        obs_dim=10,
        action_dim=2,
        hidden_dim=64,
        num_hidden_layers=2,
        teacher_lr=3e-3,
    )
    teacher = FlowMatchingTeacher(cfg)
    opt = torch.optim.Adam(teacher.parameters(), lr=cfg.teacher_lr)

    obs = torch.randn(1, cfg.obs_dim)
    action = torch.tensor([[0.7, -0.3]], dtype=torch.float32)

    losses: list[float] = []
    for _ in range(1500):
        t = torch.rand(1, 1)
        x_0 = torch.randn_like(action)
        x_t = (1.0 - t) * x_0 + t * action
        v_target = action - x_0
        v_pred = teacher(x_t, t, obs)
        loss = ((v_pred - v_target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # Robust check: median drops by at least 5x over the run, and the tail
    # median is below a generous threshold (1.0, well above the converged
    # value but far below initialisation).
    initial_median = float(np.median(losses[:100]))
    tail_median = float(np.median(losses[-200:]))
    assert tail_median < 1.0, (
        f"Teacher tail median {tail_median:.3f} suggests it failed to learn"
    )
    assert tail_median < initial_median / 5.0, (
        "Teacher loss did not decrease enough: "
        f"initial median {initial_median:.3f} → tail median {tail_median:.4f}"
    )


# ---------------------------------------------------------------------------
# Section 5: agent update metrics + critic + actor smoke
# ---------------------------------------------------------------------------


def test_fql_update_metrics_keys() -> None:
    cfg = FQLConfig(obs_dim=10, action_dim=2, batch_size=32, policy_freq=1)
    agent = FQLAgent(cfg, device="cpu")
    batch, _ = _make_batch()
    metrics = agent.update(batch)
    expected = {
        "loss_flow",
        "q1_loss",
        "q2_loss",
        "critic_loss",
        "target_q",
        "td_abs_error",
        "actor_loss",
        "bc_loss",
        "mean_q",
        "lambda",
        "teacher_grad_norm",
        "q1_grad_norm",
        "q2_grad_norm",
        "student_grad_norm",
        "policy_updated",
    }
    missing = expected - set(metrics.keys())
    assert not missing, f"Missing FQL metrics keys: {sorted(missing)}"
    for key, value in metrics.items():
        assert np.isfinite(value), f"Metric {key} non-finite: {value}"


def test_fql_critic_update_smoke() -> None:
    """Run 100 updates and confirm Q losses remain bounded."""
    torch.manual_seed(1)
    cfg = FQLConfig(
        obs_dim=10,
        action_dim=2,
        batch_size=32,
        policy_freq=2,
        hidden_dim=64,
    )
    agent = FQLAgent(cfg, device="cpu")
    target_qs: list[float] = []
    for step in range(100):
        batch, _ = _make_batch(seed=step)
        m = agent.update(batch)
        target_qs.append(m["target_q"])
        assert np.isfinite(m["q1_loss"]) and np.isfinite(m["q2_loss"])
        assert abs(m["target_q"]) < 1e4
    # Target Q sequence should not diverge.
    assert max(abs(q) for q in target_qs) < 1e3


def test_fql_actor_update_smoke() -> None:
    """Run 60 updates with policy_freq=1 so actor updates every step; check
    actor / bc losses remain finite and the BC loss does not blow up."""
    torch.manual_seed(2)
    cfg = FQLConfig(
        obs_dim=10,
        action_dim=2,
        batch_size=32,
        policy_freq=1,
        hidden_dim=64,
        flow_steps=4,
    )
    agent = FQLAgent(cfg, device="cpu")
    bc_losses: list[float] = []
    for step in range(60):
        batch, _ = _make_batch(seed=step + 100)
        m = agent.update(batch)
        assert np.isfinite(m["actor_loss"])
        assert np.isfinite(m["bc_loss"])
        bc_losses.append(m["bc_loss"])
    # BC loss starts non-trivial then settles within an order of magnitude.
    assert max(bc_losses) < 1e3


def test_fql_policy_freq_gating() -> None:
    """policy_updated flag must follow ``update_count % policy_freq == 0``."""
    cfg = FQLConfig(obs_dim=10, action_dim=2, policy_freq=3, hidden_dim=64)
    agent = FQLAgent(cfg, device="cpu")
    flags: list[int] = []
    for step in range(6):
        batch, _ = _make_batch(seed=step)
        m = agent.update(batch)
        flags.append(int(m["policy_updated"]))
    # update_count starts at 0, increments inside update; so first time
    # update_count==3 is at step index 2 (0-based).
    assert flags == [0, 0, 1, 0, 0, 1], flags


# ---------------------------------------------------------------------------
# Section 6: registry roundtrip
# ---------------------------------------------------------------------------


def test_fql_registry_roundtrip() -> None:
    cfg = make_agent_config("fql", obs_dim=10, action_dim=2)
    assert isinstance(cfg, FQLConfig)
    agent = make_agent("fql", cfg, device="cpu")
    assert isinstance(agent, FQLAgent)


# ---------------------------------------------------------------------------
# Section 7: checkpoint save/load + policy export
# ---------------------------------------------------------------------------


def test_fql_checkpoint_save_load(tmp_path) -> None:
    torch.manual_seed(3)
    cfg = FQLConfig(
        obs_dim=10, action_dim=2, batch_size=32, policy_freq=1, hidden_dim=64
    )
    agent = FQLAgent(cfg, device="cpu")
    # warm up so weights diverge from initialisation
    for step in range(5):
        batch, _ = _make_batch(seed=step + 500)
        agent.update(batch)

    sample_obs = np.random.RandomState(0).randn(10).astype(np.float32)
    action_before, _ = agent.act(sample_obs, deterministic=True)

    ckpt = tmp_path / "fql_ckpt.pt"
    agent.save(str(ckpt))

    fresh = FQLAgent(
        FQLConfig(obs_dim=10, action_dim=2, hidden_dim=64), device="cpu"
    )
    fresh.load(str(ckpt))
    action_after, _ = fresh.act(sample_obs, deterministic=True)
    np.testing.assert_allclose(action_before, action_after, atol=1e-6)


def test_fql_policy_act_shape() -> None:
    """FQLPolicy.act should return correct shape on both single-obs and
    batched-obs inputs, and round-trip via export_policy_payload +
    policy_from_payload."""
    cfg = FQLConfig(obs_dim=10, action_dim=2, hidden_dim=64)
    agent = FQLAgent(cfg, device="cpu")
    payload = agent.export_policy_payload()
    policy = policy_from_payload(payload, device="cpu")
    assert isinstance(policy, FQLPolicy)

    obs_single = np.random.RandomState(1).randn(10).astype(np.float32)
    a_single, _ = policy.act(obs_single)
    assert a_single.shape == (2,)
    assert np.all(np.abs(a_single) <= 1.0)

    obs_batch = np.random.RandomState(2).randn(7, 10).astype(np.float32)
    a_batch, _ = policy.act(obs_batch)
    assert a_batch.shape == (7, 2)
    assert np.all(np.abs(a_batch) <= 1.0)


def test_fql_student_bounded() -> None:
    """DistilledStudent output should always be in [-1, 1]^A (tanh head)."""
    cfg = FQLConfig(obs_dim=10, action_dim=2, hidden_dim=64)
    student = DistilledStudent(cfg)
    obs = torch.randn(64, cfg.obs_dim) * 5.0  # large inputs
    a = student(obs)
    assert a.shape == (64, cfg.action_dim)
    assert torch.all(a.abs() <= 1.0 + 1e-6)


def test_fql_determinism_same_seed() -> None:
    """Two runs with the same torch seed must produce byte-identical loss
    sequences (sanity check for RNG hygiene per design doc §11)."""

    def _run() -> list[float]:
        torch.manual_seed(123)
        cfg = FQLConfig(
            obs_dim=10,
            action_dim=2,
            batch_size=32,
            policy_freq=1,
            hidden_dim=32,
            flow_steps=2,
        )
        agent = FQLAgent(cfg, device="cpu")
        losses = []
        for step in range(5):
            batch, _ = _make_batch(seed=step)
            m = agent.update(batch)
            losses.append(m["loss_flow"])
        return losses

    a = _run()
    b = _run()
    assert a == b, f"FQL update is non-deterministic across seeds: {a} != {b}"
