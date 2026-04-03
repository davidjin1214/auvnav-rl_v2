from __future__ import annotations

import csv
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

from auv_nav.networks import require_torch
from auv_nav.env import ObservationHistoryWrapper, PlanarRemusEnv, PlanarRemusEnvConfig
from auv_nav.baselines import (
    CrossCurrentCompensationPolicy,
    CrossCurrentResidualPrior,
    GoalSeekPolicy,
    PrivilegedCorridorPolicy,
    ResidualPriorConfig,
    StillWaterStraightLine,
    WorldFrameCurrentCompensationPolicy,
)

try:
    import torch
except ImportError:
    torch = None


OFFLINE_POLICY_REGISTRY = {
    "goal": GoalSeekPolicy,
    "compensate": CrossCurrentCompensationPolicy,
    "still_water": StillWaterStraightLine,
    "world_compensate": WorldFrameCurrentCompensationPolicy,
    "corridor": PrivilegedCorridorPolicy,
}


def discover_flow_path() -> Path:
    data_dir = Path("wake_data")
    candidates = sorted(data_dir.glob("wake_*_roi.npy"))
    if not candidates:
        raise FileNotFoundError(
            "No wake-field data found under ./wake_data/. "
            "Generate data with `python -m scripts.generate_wake` first."
        )
    return candidates[0]


def make_planar_env(
    flow_path: str | Path,
    *,
    history_length: int = 1,
) -> PlanarRemusEnv | ObservationHistoryWrapper:
    env: PlanarRemusEnv | ObservationHistoryWrapper = PlanarRemusEnv(
        PlanarRemusEnvConfig(flow_path=flow_path)
    )
    if history_length > 1:
        env = ObservationHistoryWrapper(env, history_length)
    return env


def make_reset_options(args: Any) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if getattr(args, "difficulty", None) is not None:
        options["task_difficulty"] = args.difficulty
    if getattr(args, "task_geometry", None) is not None:
        options["task_geometry"] = args.task_geometry
    if getattr(args, "action_mode", None) is not None:
        options["action_mode"] = args.action_mode
    if getattr(args, "target_speed", None) is not None:
        options["target_auv_max_speed_mps"] = args.target_speed
    elif getattr(args, "speed_ratio", None) is not None:
        options["target_speed_ratio"] = args.speed_ratio
    return options


def build_residual_prior(env: PlanarRemusEnv, action_mode: str) -> CrossCurrentResidualPrior:
    layout = env.get_observation_layout()
    return CrossCurrentResidualPrior(
        ResidualPriorConfig(
            action_mode=action_mode,
            heading_offset_limit_rad=env.config.heading_offset_limit_rad,
            heading_cos_index=layout.heading_cos_index,
            heading_sin_index=layout.heading_sin_index,
            goal_x_index=layout.goal_x_index,
            goal_y_index=layout.goal_y_index,
            center_probe_v_index=layout.center_probe_v_index,
        )
    )


def load_trainer_state(path: str) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        meta_path = checkpoint_path / "trainer_state.json"
    else:
        meta_path = checkpoint_path
    if not meta_path.exists():
        raise FileNotFoundError(f"Resume metadata not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if torch is not None:
        torch_state = state.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        torch_cuda_state = state.get("torch_cuda")
        if torch_cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda_state)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_training_state(
    save_dir: Path,
    agent: Any,
    replay: Any,
    *,
    train_config: dict[str, Any],
    agent_config: dict[str, Any],
    reset_options: dict[str, Any],
    flow_path: str,
    env_step: int,
    episode_idx: int,
    extra_state: dict[str, Any] | None = None,
) -> None:
    agent_path = save_dir / "agent_latest.pt"
    meta_path = save_dir / "trainer_state.json"
    replay_path = save_dir / "replay_latest.pkl"
    rng_state_path = save_dir / "rng_state.pkl"
    agent.save(str(agent_path))
    with replay_path.open("wb") as fp:
        pickle.dump(replay.state_dict(), fp)
    with rng_state_path.open("wb") as fp:
        pickle.dump(capture_rng_state(), fp)
    meta = {
        "env_step": env_step,
        "episode_idx": episode_idx,
        "train_config": train_config,
        "agent_config": agent_config,
        "reset_options": reset_options,
        "flow_path": flow_path,
        "agent_path": agent_path.name,
        "replay_path": replay_path.name,
        "rng_state_path": rng_state_path.name,
    }
    if extra_state:
        meta.update(extra_state)
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)


def maybe_resume(
    agent: Any,
    replay: Any,
    resume_path: str | None,
) -> tuple[int, int]:
    if resume_path is None:
        return 0, 0

    meta = load_trainer_state(resume_path)
    checkpoint_path = Path(resume_path)
    meta_root = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    agent_path = meta_root / meta["agent_path"]
    replay_path = meta_root / meta["replay_path"]
    rng_state_path = meta_root / meta.get("rng_state_path", "rng_state.pkl")
    agent.load(str(agent_path))
    if replay_path.exists():
        with replay_path.open("rb") as fp:
            replay.load_state_dict(pickle.load(fp))
    if rng_state_path.exists():
        with rng_state_path.open("rb") as fp:
            restore_rng_state(pickle.load(fp))

    env_step = int(meta.get("env_step", 0))
    episode_idx = int(meta.get("episode_idx", 0))
    print(f"[resume] step={env_step} episode={episode_idx} from {resume_path}")
    return env_step, episode_idx


def evaluate_agent(
    env: Any,
    agent: Any,
    reset_options: dict[str, Any],
    seed: int,
    num_episodes: int,
) -> dict[str, float]:
    returns = []
    costs = []
    successes = 0
    times = []
    for idx in range(num_episodes):
        obs, info = env.reset(seed=seed + idx, options=reset_options)
        policy_state = agent.reset_policy_state()
        done = False
        total_reward = 0.0
        total_cost = 0.0
        while not done:
            action, policy_state = agent.act(obs, policy_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_cost += float(info["step_safety_cost"])
            done = terminated or truncated
        returns.append(total_reward)
        costs.append(total_cost)
        successes += int(info["success"])
        times.append(float(info["elapsed_time_s"]))

    return {
        "eval_return": float(np.mean(returns)),
        "eval_cost": float(np.mean(costs)),
        "eval_success_rate": float(successes / max(1, num_episodes)),
        "eval_time_s": float(np.mean(times)),
    }


def collect_offline_episodes(
    env: PlanarRemusEnv,
    replay: Any,
    reset_options: dict[str, Any],
    num_episodes: int,
    policy_name: str,
    seed: int,
) -> None:
    if policy_name not in OFFLINE_POLICY_REGISTRY:
        raise ValueError(f"Unknown offline policy: {policy_name}")
    policy = OFFLINE_POLICY_REGISTRY[policy_name]()
    for idx in range(num_episodes):
        obs, _ = env.reset(seed=seed + idx, options=reset_options)
        observations = [obs.copy()]
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        costs: list[float] = []
        dones: list[float] = []
        done = False
        while not done:
            action = policy.act(env, obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            actions.append(np.asarray(action, dtype=np.float32))
            rewards.append(float(reward))
            costs.append(float(info["step_safety_cost"]))
            dones.append(float(done))
            observations.append(next_obs.copy())
            obs = next_obs
        replay.add_episode(
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            costs=np.asarray(costs, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )


def pretrain_from_replay(
    agent: Any,
    replay: Any,
    updates: int,
    batch_size: int,
) -> dict[str, float]:
    require_torch()
    last_metrics: dict[str, float] = {}
    if updates <= 0:
        return last_metrics
    if not replay.ready(batch_size):
        return last_metrics
    for _ in range(updates):
        batch = replay.sample_batch(batch_size, agent.device)
        last_metrics = agent.update(batch)
    return last_metrics
