from __future__ import annotations

import numpy as np
import torch

from auv_nav.replay import TransitionReplay, TransitionReplayConfig
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
            policy_freq=1,
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


def test_transition_replay_tensor_cache_and_invalidation():
    replay = TransitionReplay(
        obs_dim=4,
        action_dim=2,
        config=TransitionReplayConfig(capacity=8, privileged_obs_dim=3),
    )
    for idx in range(5):
        replay.add(
            obs=np.full(4, idx, dtype=np.float32),
            action=np.full(2, idx, dtype=np.float32),
            reward=float(idx),
            cost=float(idx) * 0.1,
            next_obs=np.full(4, idx + 1, dtype=np.float32),
            done=bool(idx % 2),
            privileged_obs=np.full(3, idx, dtype=np.float32),
            next_privileged_obs=np.full(3, idx + 1, dtype=np.float32),
        )

    replay.enable_tensor_cache("cpu")
    assert replay.has_tensor_cache("cpu")
    batch = replay.sample_batch(3, device=torch.device("cpu"))
    assert batch["obs"].shape == (3, 4)
    assert batch["actions"].shape == (3, 2)
    assert batch["privileged_obs"].shape == (3, 3)
    assert batch["obs"].device.type == "cpu"

    replay.add(
        obs=np.zeros(4, dtype=np.float32),
        action=np.zeros(2, dtype=np.float32),
        reward=0.0,
        cost=0.0,
        next_obs=np.zeros(4, dtype=np.float32),
        done=False,
    )
    assert not replay.has_tensor_cache()


def test_transition_replay_iter_batches_without_replacement_covers_dataset():
    replay = TransitionReplay(
        obs_dim=3,
        action_dim=1,
        config=TransitionReplayConfig(capacity=8),
    )
    for idx in range(5):
        replay.add(
            obs=np.full(3, idx, dtype=np.float32),
            action=np.asarray([idx], dtype=np.float32),
            reward=float(idx),
            cost=0.0,
            next_obs=np.full(3, idx + 1, dtype=np.float32),
            done=False,
        )

    batches = list(
        replay.iter_batches(
            batch_size=2,
            device=torch.device("cpu"),
            shuffle=False,
            drop_last=False,
        )
    )

    assert [batch["obs"].shape[0] for batch in batches] == [2, 2, 1]
    recovered = np.concatenate([batch["obs"][:, 0].cpu().numpy() for batch in batches])
    assert recovered.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
