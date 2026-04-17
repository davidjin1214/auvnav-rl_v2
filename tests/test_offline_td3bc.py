from __future__ import annotations

import numpy as np
import torch

from auv_nav.td3bc import ObservationNormalizer, TD3BCAgent, TD3BCConfig


def _make_batch(batch_size: int, obs_dim: int, action_dim: int):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(batch_size, obs_dim)).astype(np.float32)
    next_obs = rng.normal(size=(batch_size, obs_dim)).astype(np.float32)
    actions = np.clip(rng.normal(size=(batch_size, action_dim)), -1.0, 1.0).astype(np.float32)
    rewards = rng.normal(size=(batch_size,)).astype(np.float32)
    dones = rng.integers(0, 2, size=(batch_size,), endpoint=False).astype(np.float32)
    return {
        "obs": torch.as_tensor(obs),
        "actions": torch.as_tensor(actions),
        "rewards": torch.as_tensor(rewards),
        "next_obs": torch.as_tensor(next_obs),
        "dones": torch.as_tensor(dones),
    }, obs


def test_td3bc_update_and_save_roundtrip(tmp_path):
    obs_dim = 10
    action_dim = 2
    batch, obs = _make_batch(batch_size=32, obs_dim=obs_dim, action_dim=action_dim)
    normalizer = ObservationNormalizer.from_observations(obs, device="cpu")
    agent = TD3BCAgent(
        TD3BCConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            batch_size=32,
            policy_freq=2,
        ),
        obs_normalizer=normalizer,
        device="cpu",
    )

    metrics = agent.update(batch)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["bc_loss"])
    assert np.isfinite(metrics["mean_q"])

    sample_obs = obs[0]
    action_before, _ = agent.act(sample_obs, deterministic=True)
    checkpoint = tmp_path / "agent.pt"
    agent.save(str(checkpoint))

    loaded = TD3BCAgent(
        TD3BCConfig(obs_dim=obs_dim, action_dim=action_dim),
        device="cpu",
    )
    loaded.load(str(checkpoint))
    action_after, _ = loaded.act(sample_obs, deterministic=True)
    assert np.allclose(action_before, action_after, atol=1e-6)
