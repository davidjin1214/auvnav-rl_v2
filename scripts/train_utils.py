from __future__ import annotations

import csv
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

from auv_nav.env import ObservationHistoryWrapper, PlanarRemusEnv, PlanarRemusEnvConfig
from auv_nav.flow import make_probe_offsets
from auv_nav.reward import (
    REWARD_CONFIG_FIELDS,
    canonical_reward_objective,
    reward_objective_config,
)
from .benchmark_utils import BenchmarkManifest, load_benchmark_manifest

try:
    import torch
except ImportError:
    torch = None


def default_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


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
    probe_layout: str = "s0",
    env_config_overrides: dict[str, Any] | None = None,
) -> PlanarRemusEnv | ObservationHistoryWrapper:
    config_kwargs: dict[str, Any] = {
        "flow_path": flow_path,
        "probe_offsets_body": make_probe_offsets(probe_layout),
        "probe_channels": "velocity",
    }
    if env_config_overrides:
        config_kwargs.update(env_config_overrides)
    env: PlanarRemusEnv | ObservationHistoryWrapper = PlanarRemusEnv(
        PlanarRemusEnvConfig(**config_kwargs)
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


def make_env_config_overrides(
    args: Any,
    *,
    base_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = dict(base_overrides or {})

    # Keep environment defaults aligned with CLI episode settings so that
    # auto-resets in vectorized environments do not silently drift back to the
    # config defaults (for example downstream geometry).
    if getattr(args, "difficulty", None) is not None:
        overrides["task_difficulty"] = args.difficulty
    if getattr(args, "task_geometry", None) is not None:
        overrides["task_geometry"] = args.task_geometry
    if getattr(args, "action_mode", None) is not None:
        overrides["action_mode"] = args.action_mode
    if getattr(args, "target_speed", None) is not None:
        overrides["target_auv_max_speed_mps"] = args.target_speed
    elif getattr(args, "speed_ratio", None) is not None:
        overrides["target_speed_ratio"] = args.speed_ratio

    objective = getattr(args, "objective", None)
    if objective is not None:
        objective = canonical_reward_objective(objective)
        overrides.update(reward_objective_config(objective))
        overrides["reward_objective"] = objective

    for field in REWARD_CONFIG_FIELDS:
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    return overrides


def extract_env_config_overrides(trainer_state: dict[str, Any]) -> dict[str, Any]:
    saved = trainer_state.get("env_config_overrides", {})
    return dict(saved) if isinstance(saved, dict) else {}


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


def maybe_load_benchmark_manifest(path: str | Path | None) -> BenchmarkManifest | None:
    if path is None:
        return None
    return load_benchmark_manifest(path)


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
    benchmark_manifest: BenchmarkManifest | None = None,
) -> dict[str, Any]:
    if benchmark_manifest is not None:
        episodes = benchmark_manifest.episodes
    else:
        episodes = None

    results: list[dict[str, Any]] = []
    n_rollouts = len(episodes) if episodes is not None else num_episodes
    for idx in range(n_rollouts):
        episode_reset_options = dict(reset_options)
        rollout_seed = seed + idx
        episode_id = f"seed_{rollout_seed}"
        if episodes is not None:
            spec = episodes[idx]
            episode_reset_options.update(spec.reset_options)
            rollout_seed = int(spec.seed)
            episode_id = spec.episode_id

        obs, info = env.reset(seed=rollout_seed, options=episode_reset_options)
        policy_state = agent.reset_policy_state()
        done = False
        total_reward = 0.0
        total_cost = 0.0
        total_energy = 0.0
        path_length = 0.0
        speed_samples: list[float] = []
        relative_speed_samples: list[float] = []
        prev_pos = np.asarray(info["position_xy_m"], dtype=np.float32)
        initial_distance = float(info["initial_distance_m"])

        while not done:
            action, policy_state = agent.act(obs, policy_state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_cost += float(info["step_safety_cost"])
            total_energy += float(info["energy_cost"])
            pos = np.asarray(info["position_xy_m"], dtype=np.float32)
            path_length += float(np.linalg.norm(pos - prev_pos))
            prev_pos = pos
            speed_samples.append(float(info["ground_speed_mps"]))
            relative_speed_samples.append(float(info["water_relative_speed_mps"]))
            done = terminated or truncated

        final_distance = float(info["distance_to_goal_m"])
        progress_m = initial_distance - final_distance
        progress_ratio = progress_m / max(initial_distance, 1e-8)
        path_efficiency = progress_m / max(path_length, 1e-8)
        results.append(
            {
                "episode_id": episode_id,
                "seed": rollout_seed,
                "success": bool(info["success"]),
                "reason": str(info["reason"]),
                "return": total_reward,
                "cost": total_cost,
                "safety_cost": total_cost,
                "energy": total_energy,
                "elapsed_time_s": float(info["elapsed_time_s"]),
                "path_length_m": path_length,
                "initial_distance_m": initial_distance,
                "final_distance_m": final_distance,
                "progress_m": progress_m,
                "progress_ratio": progress_ratio,
                "path_efficiency": path_efficiency,
                "mean_ground_speed_mps": float(np.mean(speed_samples)) if speed_samples else 0.0,
                "mean_water_relative_speed_mps": (
                    float(np.mean(relative_speed_samples)) if relative_speed_samples else 0.0
                ),
            }
        )

    returns = np.array([row["return"] for row in results], dtype=float)
    costs = np.array([row["cost"] for row in results], dtype=float)
    energies = np.array([row["energy"] for row in results], dtype=float)
    times = np.array([row["elapsed_time_s"] for row in results], dtype=float)
    path_lengths = np.array([row["path_length_m"] for row in results], dtype=float)
    progress_ratios = np.array([row["progress_ratio"] for row in results], dtype=float)
    path_efficiencies = np.array([row["path_efficiency"] for row in results], dtype=float)
    successes = np.array([row["success"] for row in results], dtype=bool)
    success_times = times[successes]
    success_energies = energies[successes]
    success_paths = path_lengths[successes]
    termination_counts: dict[str, int] = {}
    for row in results:
        termination_counts[row["reason"]] = termination_counts.get(row["reason"], 0) + 1

    return {
        "num_eval_episodes": float(len(results)),
        "eval_return": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_cost": float(np.mean(costs)),
        "eval_cost_std": float(np.std(costs)),
        "eval_safety_cost": float(np.mean(costs)),
        "eval_safety_cost_std": float(np.std(costs)),
        "eval_success_rate": float(np.mean(successes.astype(float))),
        "eval_time_s": float(np.mean(times)),
        "eval_time_s_std": float(np.std(times)),
        "eval_time_s_success": float(np.mean(success_times)) if len(success_times) else float("nan"),
        "eval_energy": float(np.mean(energies)),
        "eval_energy_std": float(np.std(energies)),
        "eval_energy_success": (
            float(np.mean(success_energies)) if len(success_energies) else float("nan")
        ),
        "eval_path_length_m": float(np.mean(path_lengths)),
        "eval_path_length_m_std": float(np.std(path_lengths)),
        "eval_path_length_success": (
            float(np.mean(success_paths)) if len(success_paths) else float("nan")
        ),
        "eval_progress_ratio": float(np.mean(progress_ratios)),
        "eval_progress_ratio_std": float(np.std(progress_ratios)),
        "eval_path_efficiency": float(np.mean(path_efficiencies)),
        "eval_path_efficiency_std": float(np.std(path_efficiencies)),
        "eval_termination_counts": termination_counts,
        "eval_episode_results": results,
        "reward_objective": env.unwrapped.config.reward_objective,
        "energy_cost_gain": float(env.unwrapped.config.energy_cost_gain),
        "safety_cost_gain": float(env.unwrapped.config.safety_cost_gain),
        "eval_manifest_flow_path": (
            benchmark_manifest.flow_path if benchmark_manifest is not None else None
        ),
    }
