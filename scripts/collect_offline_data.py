"""Collect offline transition data from baseline policies for RLPD training.

Usage:
    python -m scripts.collect_offline_data \
        --policy worldcomp \
        --flow wake_data/wake_v8_U1p50_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
        --probe-layout s0 \
        --difficulty hard \
        --target-speed 1.5 \
        --episodes 500 \
        --seed 0 \
        --output-dir offline_data/worldcomp_s0_hard_U1p50
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from auv_nav.baselines import (
    CrossCurrentCompensationPolicy,
    GoalSeekPolicy,
    PrivilegedCorridorPolicy,
    WorldFrameCurrentCompensationPolicy,
)
from auv_nav.env import ObservationHistoryWrapper
from auv_nav.reward import REWARD_OBJECTIVE_PRESETS

from .train_utils import (
    discover_flow_path,
    make_env_config_overrides,
    make_planar_env,
    make_reset_options,
)

POLICY_MAP = {
    "goalseek": GoalSeekPolicy,
    "crosscomp": CrossCurrentCompensationPolicy,
    "worldcomp": WorldFrameCurrentCompensationPolicy,
    "privileged": PrivilegedCorridorPolicy,
}

TERMINATION_REASON_TO_CODE = {
    "running": 0,
    "goal": 1,
    "timeout": 2,
    "out_of_bounds": 3,
}
BEHAVIOR_POLICY_TO_CODE = {
    policy_name: index for index, policy_name in enumerate(sorted(POLICY_MAP.keys()))
}


@dataclass(slots=True)
class CollectWorkerConfig:
    policy_name: str
    policy_mixture: tuple[tuple[str, float], ...]
    flow_path: str
    history_length: int
    probe_layout: str
    env_config_overrides: dict[str, Any]
    reset_options: dict[str, Any]
    base_seed: int
    action_noise_std: float
    action_noise_clip: float


@dataclass(slots=True)
class CollectChunkResult:
    start_episode: int
    end_episode: int
    payload: dict[str, np.ndarray]
    successes: int
    episode_returns: list[float]
    episode_lengths: list[int]
    episode_reasons: list[str]
    episode_policy_names: list[str]


def _make_policy(policy_name: str) -> Any:
    if policy_name not in POLICY_MAP:
        raise ValueError(f"Unsupported baseline policy: {policy_name!r}")
    return POLICY_MAP[policy_name]()


def _parse_policy_mixture(
    policy_name: str,
    policy_mixture: str | None,
) -> tuple[tuple[str, float], ...]:
    if policy_mixture is None or not str(policy_mixture).strip():
        return ((policy_name, 1.0),)

    entries: list[tuple[str, float]] = []
    for raw_entry in str(policy_mixture).split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                "Each --policy-mixture entry must have the form policy:weight."
            )
        item_policy, raw_weight = (part.strip() for part in entry.split(":", 1))
        if item_policy not in POLICY_MAP:
            raise ValueError(f"Unsupported policy in --policy-mixture: {item_policy!r}")
        weight = float(raw_weight)
        if weight <= 0.0:
            raise ValueError("Policy mixture weights must be positive.")
        entries.append((item_policy, weight))

    if not entries:
        raise ValueError("--policy-mixture produced no valid entries.")

    total_weight = sum(weight for _, weight in entries)
    return tuple((name, weight / total_weight) for name, weight in entries)


def _sample_policy_name(
    policy_mixture: tuple[tuple[str, float], ...],
    rng: np.random.Generator,
) -> str:
    if len(policy_mixture) == 1:
        return policy_mixture[0][0]
    names = [name for name, _ in policy_mixture]
    probs = np.asarray([weight for _, weight in policy_mixture], dtype=np.float64)
    index = int(rng.choice(len(names), p=probs))
    return names[index]


def _collect_episode(
    env: Any,
    base_env: Any,
    policy: Any,
    *,
    seed: int,
    reset_options: dict[str, Any],
    action_noise_std: float,
    action_noise_clip: float,
    episode_rng: np.random.Generator,
    behavior_policy_name: str,
) -> tuple[dict[str, list[Any]], bool, float, int, str]:
    obs, info = env.reset(seed=seed, options=reset_options)
    current_privileged_obs = info.get("privileged_obs")
    ep_return = 0.0
    ep_length = 0
    done = False
    transitions: dict[str, list[Any]] = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "costs": [],
        "next_obs": [],
        "dones": [],
        "terminateds": [],
        "truncateds": [],
        "terminal_reason_codes": [],
        "behavior_policy_codes": [],
        "privileged_obs": [],
        "next_privileged_obs": [],
    }

    while not done:
        # Get single-step obs for policies that call decode_observation.
        if isinstance(env, ObservationHistoryWrapper):
            single_obs = np.asarray(env._history[-1], dtype=np.float32)
        else:
            single_obs = np.asarray(obs, dtype=np.float32)

        action = np.asarray(policy.act(base_env, single_obs), dtype=np.float32)
        if action_noise_std > 0.0:
            noise = episode_rng.normal(
                loc=0.0,
                scale=action_noise_std,
                size=action.shape,
            ).astype(np.float32)
            if action_noise_clip > 0.0:
                noise = np.clip(noise, -action_noise_clip, action_noise_clip)
            action = np.clip(
                action + noise,
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        terminal_reason = str(info.get("reason", "running")) if done else "running"
        terminal_reason_code = TERMINATION_REASON_TO_CODE.get(
            terminal_reason,
            TERMINATION_REASON_TO_CODE["running"],
        )

        transitions["obs"].append(np.asarray(obs, dtype=np.float32))
        transitions["actions"].append(np.asarray(action, dtype=np.float32))
        transitions["rewards"].append(float(reward))
        transitions["costs"].append(float(info.get("step_safety_cost", 0.0)))
        transitions["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
        transitions["dones"].append(bool(done))
        transitions["terminateds"].append(bool(terminated))
        transitions["truncateds"].append(bool(truncated))
        transitions["terminal_reason_codes"].append(int(terminal_reason_code))
        transitions["behavior_policy_codes"].append(BEHAVIOR_POLICY_TO_CODE[behavior_policy_name])
        if current_privileged_obs is not None:
            transitions["privileged_obs"].append(
                np.asarray(current_privileged_obs, dtype=np.float32)
            )
        next_privileged_obs = info.get("privileged_obs")
        if next_privileged_obs is not None:
            transitions["next_privileged_obs"].append(
                np.asarray(next_privileged_obs, dtype=np.float32)
            )

        ep_return += float(reward)
        ep_length += 1
        obs = next_obs
        current_privileged_obs = next_privileged_obs

    return (
        transitions,
        bool(info.get("success", False)),
        ep_return,
        ep_length,
        str(info.get("reason", "running")),
    )


def _collect_episode_range(
    worker_config: CollectWorkerConfig,
    start_episode: int,
    end_episode: int,
) -> CollectChunkResult:
    env = make_planar_env(
        worker_config.flow_path,
        history_length=worker_config.history_length,
        probe_layout=worker_config.probe_layout,
        env_config_overrides=worker_config.env_config_overrides,
    )
    try:
        base_env = env.env if isinstance(env, ObservationHistoryWrapper) else env
        policy_cache = {
            name: _make_policy(name) for name, _ in worker_config.policy_mixture
        }
        transitions: dict[str, list[Any]] = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "costs": [],
            "next_obs": [],
            "dones": [],
            "terminateds": [],
            "truncateds": [],
            "terminal_reason_codes": [],
            "behavior_policy_codes": [],
            "privileged_obs": [],
            "next_privileged_obs": [],
        }
        successes = 0
        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        episode_reasons: list[str] = []
        episode_policy_names: list[str] = []

        for ep in range(start_episode, end_episode):
            episode_seed = worker_config.base_seed + ep
            episode_rng = np.random.default_rng(episode_seed)
            policy_name = _sample_policy_name(worker_config.policy_mixture, episode_rng)
            policy = policy_cache[policy_name]
            episode_transitions, success, ep_return, ep_length, ep_reason = _collect_episode(
                env,
                base_env,
                policy,
                seed=episode_seed,
                reset_options=worker_config.reset_options,
                action_noise_std=worker_config.action_noise_std,
                action_noise_clip=worker_config.action_noise_clip,
                episode_rng=episode_rng,
                behavior_policy_name=policy_name,
            )
            for key, values in episode_transitions.items():
                transitions[key].extend(values)
            successes += int(success)
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            episode_reasons.append(ep_reason)
            episode_policy_names.append(policy_name)

        payload = {
            "obs": np.asarray(transitions["obs"], dtype=np.float32),
            "actions": np.asarray(transitions["actions"], dtype=np.float32),
            "rewards": np.asarray(transitions["rewards"], dtype=np.float32),
            "costs": np.asarray(transitions["costs"], dtype=np.float32),
            "next_obs": np.asarray(transitions["next_obs"], dtype=np.float32),
            "dones": np.asarray(transitions["dones"], dtype=np.float32),
            "terminateds": np.asarray(transitions["terminateds"], dtype=np.float32),
            "truncateds": np.asarray(transitions["truncateds"], dtype=np.float32),
            "terminal_reason_codes": np.asarray(
                transitions["terminal_reason_codes"],
                dtype=np.int8,
            ),
            "behavior_policy_codes": np.asarray(
                transitions["behavior_policy_codes"],
                dtype=np.int8,
            ),
        }
        if transitions["privileged_obs"]:
            payload["privileged_obs"] = np.asarray(
                transitions["privileged_obs"],
                dtype=np.float32,
            )
        if transitions["next_privileged_obs"]:
            payload["next_privileged_obs"] = np.asarray(
                transitions["next_privileged_obs"],
                dtype=np.float32,
            )

        return CollectChunkResult(
            start_episode=start_episode,
            end_episode=end_episode,
            payload=payload,
            successes=successes,
            episode_returns=episode_returns,
            episode_lengths=episode_lengths,
            episode_reasons=episode_reasons,
            episode_policy_names=episode_policy_names,
        )
    finally:
        env.close()


def _episode_chunks(num_episodes: int, num_chunks: int) -> list[tuple[int, int]]:
    if num_episodes <= 0:
        return []
    num_chunks = max(1, min(int(num_chunks), num_episodes))
    chunk_size = (num_episodes + num_chunks - 1) // num_chunks
    return [
        (start, min(num_episodes, start + chunk_size))
        for start in range(0, num_episodes, chunk_size)
    ]


def _merge_chunk_payloads(results: list[CollectChunkResult]) -> dict[str, np.ndarray]:
    ordered_results = sorted(results, key=lambda result: result.start_episode)
    payload_keys = [
        "obs",
        "actions",
        "rewards",
        "costs",
        "next_obs",
        "dones",
        "terminateds",
        "truncateds",
        "terminal_reason_codes",
        "behavior_policy_codes",
        "privileged_obs",
        "next_privileged_obs",
    ]
    merged: dict[str, np.ndarray] = {}
    for key in payload_keys:
        arrays = [result.payload[key] for result in ordered_results if key in result.payload]
        if arrays:
            merged[key] = np.concatenate(arrays, axis=0)
    return merged


def _collect_parallel(
    worker_config: CollectWorkerConfig,
    *,
    num_episodes: int,
    num_workers: int,
) -> list[CollectChunkResult]:
    chunks = _episode_chunks(num_episodes, num_workers)
    if len(chunks) <= 1:
        start, end = chunks[0] if chunks else (0, 0)
        return [_collect_episode_range(worker_config, start, end)]

    try:
        ctx = mp.get_context("spawn")
        results: list[CollectChunkResult] = []
        completed_episodes = 0
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as executor:
            futures = [
                executor.submit(_collect_episode_range, worker_config, start, end)
                for start, end in chunks
            ]
            for future in futures:
                result = future.result()
                results.append(result)
                completed_episodes += result.end_episode - result.start_episode
                elapsed = time.time() - t0
                success_rate = sum(item.successes for item in results) / max(1, completed_episodes)
                transitions = sum(int(item.payload["obs"].shape[0]) for item in results)
                print(
                    f"[collect] ep={completed_episodes}/{num_episodes} "
                    f"transitions={transitions} "
                    f"success_rate={success_rate:.2%} "
                    f"elapsed={elapsed:.1f}s"
                )
        return results
    except PermissionError as exc:
        print(f"[collect] parallel workers unavailable ({exc}); falling back to serial.")
    except OSError as exc:
        print(f"[collect] parallel workers unavailable ({exc}); falling back to serial.")

    return [_collect_episode_range(worker_config, 0, num_episodes)]


def collect(args: argparse.Namespace) -> None:
    flow_path = args.flow or discover_flow_path()
    env_config_overrides = make_env_config_overrides(args)
    reset_options = make_reset_options(args)
    probe_layout = args.probe_layout
    history_length = int(args.history_length)

    env = make_planar_env(
        flow_path,
        history_length=history_length,
        probe_layout=probe_layout,
        env_config_overrides=env_config_overrides,
    )
    try:
        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
    finally:
        env.close()

    worker_config = CollectWorkerConfig(
        policy_name=args.policy,
        policy_mixture=_parse_policy_mixture(args.policy, args.policy_mixture),
        flow_path=str(flow_path),
        history_length=history_length,
        probe_layout=probe_layout,
        env_config_overrides=dict(env_config_overrides),
        reset_options=dict(reset_options),
        base_seed=int(args.seed),
        action_noise_std=float(args.action_noise_std),
        action_noise_clip=float(args.action_noise_clip),
    )

    t0 = time.time()
    if args.num_workers > 1:
        results = _collect_parallel(
            worker_config,
            num_episodes=int(args.episodes),
            num_workers=int(args.num_workers),
        )
    else:
        result = _collect_episode_range(worker_config, 0, int(args.episodes))
        results = [result]
        print(
            f"[collect] ep={args.episodes}/{args.episodes} "
            f"transitions={int(result.payload['obs'].shape[0])} "
            f"success_rate={result.successes / max(1, args.episodes):.2%} "
            f"elapsed={time.time() - t0:.1f}s"
        )

    payload = _merge_chunk_payloads(results)
    successes = sum(result.successes for result in results)
    episode_returns = [
        value
        for result in sorted(results, key=lambda item: item.start_episode)
        for value in result.episode_returns
    ]
    episode_lengths = [
        value
        for result in sorted(results, key=lambda item: item.start_episode)
        for value in result.episode_lengths
    ]
    episode_reasons = [
        value
        for result in sorted(results, key=lambda item: item.start_episode)
        for value in result.episode_reasons
    ]
    reason_counts = Counter(episode_reasons)
    policy_counts = Counter(
        value
        for result in sorted(results, key=lambda item: item.start_episode)
        for value in result.episode_policy_names
    )
    other_reasons = sum(
        count
        for reason, count in reason_counts.items()
        if reason not in {"goal", "timeout", "out_of_bounds"}
    )

    # Save transitions as compressed npz.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "transitions.npz"
    np.savez_compressed(str(npz_path), **payload)

    # Save metadata.
    n_transitions = int(payload["obs"].shape[0])
    metadata = {
        "policy": args.policy,
        "policy_mixture": [
            {"policy": name, "weight": weight} for name, weight in worker_config.policy_mixture
        ],
        "action_noise_std": float(args.action_noise_std),
        "action_noise_clip": float(args.action_noise_clip),
        "flow_path": str(flow_path),
        "probe_layout": args.probe_layout,
        "history_length": args.history_length,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "privileged_obs_dim": (
            int(payload["privileged_obs"].shape[1]) if "privileged_obs" in payload else 0
        ),
        "difficulty": args.difficulty,
        "target_speed": args.target_speed,
        "task_geometry": args.task_geometry,
        "objective": env_config_overrides.get("reward_objective"),
        "reward_config": env_config_overrides,
        "seed": args.seed,
        "num_episodes": args.episodes,
        "num_transitions": n_transitions,
        "success_rate": successes / max(1, args.episodes),
        "goal_rate": reason_counts.get("goal", 0) / max(1, args.episodes),
        "timeout_rate": reason_counts.get("timeout", 0) / max(1, args.episodes),
        "out_of_bounds_rate": reason_counts.get("out_of_bounds", 0) / max(1, args.episodes),
        "other_terminal_rate": other_reasons / max(1, args.episodes),
        "episode_outcome_counts": dict(sorted(reason_counts.items())),
        "policy_episode_counts": dict(sorted(policy_counts.items())),
        "terminal_reason_vocab": TERMINATION_REASON_TO_CODE,
        "behavior_policy_vocab": BEHAVIOR_POLICY_TO_CODE,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }
    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(
        f"\n[done] Saved {n_transitions} transitions to {npz_path}\n"
        f"  policy={args.policy}  obs_dim={obs_dim}  action_dim={action_dim}\n"
        f"  episodes={args.episodes}  success_rate={successes / max(1, args.episodes):.2%}\n"
        f"  mean_return={np.mean(episode_returns):.2f} +/- {np.std(episode_returns):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect offline transition data from baseline policies.",
    )
    parser.add_argument(
        "--policy",
        choices=sorted(POLICY_MAP.keys()),
        required=True,
        help="Baseline policy to run.",
    )
    parser.add_argument("--flow", type=Path, default=None, help="Path to wake ROI .npy file.")
    parser.add_argument(
        "--probe-layout",
        choices=["s0", "s1", "s2"],
        default="s0",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=1,
        help="Observation history stacking length (must match training config).",
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument(
        "--task-geometry",
        choices=["downstream", "cross_stream", "upstream"],
        default=None,
    )
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument(
        "--objective",
        choices=sorted(REWARD_OBJECTIVE_PRESETS.keys()),
        default="arrival_v1",
        help="Reward objective preset used when collecting offline transitions.",
    )
    parser.add_argument("--energy-cost-gain", type=float, default=None)
    parser.add_argument("--safety-cost-gain", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-mixture",
        type=str,
        default=None,
        help=(
            "Optional episode-level behavior mixture, e.g. "
            "'crosscomp:0.8,goalseek:0.2'. Overrides the single-policy collector."
        ),
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help="Gaussian action noise std applied to the baseline actions during collection.",
    )
    parser.add_argument(
        "--action-noise-clip",
        type=float,
        default=0.5,
        help="Per-dimension clip applied to the action noise before action-space clipping.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel collector workers. 1 keeps the original serial path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save transitions.npz and metadata.json.",
    )
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
