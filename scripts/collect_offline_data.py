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
from concurrent.futures.process import BrokenProcessPool
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
from auv_nav.reward import ARRIVAL_V2_OBJECTIVES, REWARD_OBJECTIVE_PRESETS
from auv_nav.sac_policy import SACCheckpointPolicy, resolve_trainer_state_path

from .train_utils import (
    discover_flow_path,
    load_trainer_state,
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

SAC_BEHAVIOR_POLICY_NAME = "sac_checkpoint"

TERMINATION_REASON_TO_CODE = {
    "running": 0,
    "goal": 1,
    "timeout": 2,
    "out_of_bounds": 3,
}
# Existing rule-based codes stay stable (sorted POLICY_MAP gives the same order
# regardless of insertion); SAC mode is appended as the next free code so that
# previously collected datasets stay byte-compatible.
BEHAVIOR_POLICY_TO_CODE = {
    policy_name: index for index, policy_name in enumerate(sorted(POLICY_MAP.keys()))
}
BEHAVIOR_POLICY_TO_CODE[SAC_BEHAVIOR_POLICY_NAME] = (
    max(BEHAVIOR_POLICY_TO_CODE.values()) + 1 if BEHAVIOR_POLICY_TO_CODE else 0
)


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
    # SAC-collector mode. When sac_ckpt_path is set, the worker loads a
    # SACCheckpointPolicy once and bypasses POLICY_MAP / policy_mixture.
    sac_ckpt_path: str | None = None
    sac_deterministic: bool = False
    sac_device: str = "cpu"


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
        "next_actions": [],
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
        # Rule-based baselines call env.decode_observation and want the most
        # recent frame; SAC actors were trained on the full stacked obs and
        # opt in via the uses_stacked_obs marker.
        if getattr(policy, "uses_stacked_obs", False):
            policy_obs = np.asarray(obs, dtype=np.float32)
        elif isinstance(env, ObservationHistoryWrapper):
            policy_obs = np.asarray(env._history[-1], dtype=np.float32)
        else:
            policy_obs = np.asarray(obs, dtype=np.float32)

        action = np.asarray(policy.act(base_env, policy_obs), dtype=np.float32)
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

    episode_actions = transitions["actions"]
    for idx, action in enumerate(episode_actions):
        if idx + 1 < len(episode_actions) and not transitions["dones"][idx]:
            next_action = episode_actions[idx + 1]
        else:
            next_action = np.zeros_like(action, dtype=np.float32)
        transitions["next_actions"].append(np.asarray(next_action, dtype=np.float32))

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

        if worker_config.sac_ckpt_path is not None:
            sac_policy = SACCheckpointPolicy.from_checkpoint(
                worker_config.sac_ckpt_path,
                device=worker_config.sac_device,
                deterministic=worker_config.sac_deterministic,
            )

            def _resolve_policy(_rng: np.random.Generator) -> tuple[str, Any]:
                return (SAC_BEHAVIOR_POLICY_NAME, sac_policy)
        else:
            policy_cache = {
                name: _make_policy(name) for name, _ in worker_config.policy_mixture
            }

            def _resolve_policy(rng: np.random.Generator) -> tuple[str, Any]:
                name = _sample_policy_name(worker_config.policy_mixture, rng)
                return (name, policy_cache[name])

        transitions: dict[str, list[Any]] = {
            "obs": [],
            "actions": [],
            "next_actions": [],
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
            policy_name, policy = _resolve_policy(episode_rng)
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
            "next_actions": np.asarray(transitions["next_actions"], dtype=np.float32),
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
        "next_actions",
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
    except BrokenProcessPool as exc:
        # A worker died mid-collection, usually because the OS refused memory to the
        # spawned interpreters (each re-imports torch and rebuilds the env). The
        # serial retry below needs one interpreter instead of `num_workers`. Note
        # BrokenProcessPool is a RuntimeError, so the handlers above do not catch it.
        print(f"[collect] parallel worker died ({exc}); falling back to serial.")

    return [_collect_episode_range(worker_config, 0, num_episodes)]


def _prepare_sac_mode(
    *,
    args: argparse.Namespace,
    ckpt_path: str,
    env_obs_dim: int,
    probe_layout: str,
    history_length: int,
    env_config_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Run SAC-mode preflight: Layer 1 obs_dim + Layer 2 trainer_state checks.

    Returns a dict of SAC-specific metadata fields to merge into the dataset
    metadata.json. Raises ``ValueError`` on protocol mismatch.
    """
    import torch  # local import: avoids cost in rule-based mode

    if args.policy is not None:
        raise ValueError(
            "--policy and --sac-ckpt are mutually exclusive; pick one."
        )
    if args.policy_mixture:
        raise ValueError(
            "--policy-mixture is not supported with --sac-ckpt."
        )

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"SAC checkpoint not found: {ckpt}")

    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if "config" not in payload or "actor" not in payload:
        raise ValueError(
            f"Checkpoint {ckpt} is missing 'config'/'actor' keys; not a SAC ckpt."
        )
    ckpt_obs_dim = int(payload["config"].get("obs_dim", -1))
    if ckpt_obs_dim != env_obs_dim:
        raise ValueError(
            f"SAC ckpt obs_dim={ckpt_obs_dim} does not match env obs_dim={env_obs_dim}. "
            f"Check --probe-layout / --history-length / --objective matches the "
            f"ckpt training config."
        )

    # Layer 2 — protocol cross-check via trainer_state.json.
    trainer_state_path: Path | None
    if args.sac_trainer_state:
        trainer_state_path = Path(args.sac_trainer_state)
        if not trainer_state_path.exists():
            raise FileNotFoundError(
                f"--sac-trainer-state path does not exist: {trainer_state_path}"
            )
    else:
        trainer_state_path = resolve_trainer_state_path(ckpt)

    sac_meta: dict[str, Any] = {
        "sac_ckpt_path": str(ckpt),
        "sac_deterministic": bool(args.sac_deterministic),
        "sac_device": str(args.sac_device),
        "sac_agent_config": dict(payload["config"]),
        "sac_trainer_state_path": (
            str(trainer_state_path) if trainer_state_path is not None else None
        ),
    }

    if trainer_state_path is None:
        if not args.sac_skip_trainer_state_check:
            raise ValueError(
                f"Could not autodetect trainer_state.json next to {ckpt} "
                f"(expected via checkpoints/<X> ↔ experiments/<X> mirror). "
                f"Pass --sac-trainer-state <path> or, only if you trust the CLI "
                f"protocol args, --sac-skip-trainer-state-check."
            )
        print(
            f"[collect][warning] trainer_state.json not found for {ckpt}; "
            f"skipping Layer 2 protocol check at user's explicit request."
        )
        return sac_meta

    trainer_state = load_trainer_state(str(trainer_state_path))
    expected = {
        "algorithm": "sac",
        "probe_layout": probe_layout,
        "history_length": history_length,
        "reward_objective": env_config_overrides.get("reward_objective"),
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual = trainer_state.get(key)
        if actual != expected_value:
            mismatches.append(f"{key}: ckpt={actual!r} cli={expected_value!r}")
    if mismatches:
        raise ValueError(
            "SAC ckpt trainer_state.json protocol mismatch:\n  "
            + "\n  ".join(mismatches)
            + f"\n  (trainer_state path: {trainer_state_path})"
        )

    sac_meta["sac_trainer_state_snapshot"] = {
        key: trainer_state.get(key) for key in expected
    }
    return sac_meta


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

    sac_ckpt_path = getattr(args, "sac_ckpt", None)
    sac_metadata: dict[str, Any] = {}
    if sac_ckpt_path is not None:
        sac_metadata = _prepare_sac_mode(
            args=args,
            ckpt_path=sac_ckpt_path,
            env_obs_dim=obs_dim,
            probe_layout=probe_layout,
            history_length=history_length,
            env_config_overrides=env_config_overrides,
        )

    num_workers = int(args.num_workers)
    sac_device = str(getattr(args, "sac_device", "cpu"))
    if sac_ckpt_path is not None and sac_device == "cuda" and num_workers > 1:
        print(
            "[collect] --sac-device cuda with --num-workers > 1 is unsafe under "
            "spawn; forcing --num-workers=1."
        )
        num_workers = 1

    if sac_ckpt_path is not None:
        policy_name_for_meta = SAC_BEHAVIOR_POLICY_NAME
        policy_mixture: tuple[tuple[str, float], ...] = ()
    else:
        if args.policy is None:
            raise ValueError("Either --policy or --sac-ckpt must be provided.")
        policy_name_for_meta = args.policy
        policy_mixture = _parse_policy_mixture(args.policy, args.policy_mixture)

    worker_config = CollectWorkerConfig(
        policy_name=policy_name_for_meta,
        policy_mixture=policy_mixture,
        flow_path=str(flow_path),
        history_length=history_length,
        probe_layout=probe_layout,
        env_config_overrides=dict(env_config_overrides),
        reset_options=dict(reset_options),
        base_seed=int(args.seed),
        action_noise_std=float(args.action_noise_std),
        action_noise_clip=float(args.action_noise_clip),
        sac_ckpt_path=sac_ckpt_path,
        sac_deterministic=bool(getattr(args, "sac_deterministic", False)),
        sac_device=sac_device,
    )

    t0 = time.time()
    if num_workers > 1:
        results = _collect_parallel(
            worker_config,
            num_episodes=int(args.episodes),
            num_workers=num_workers,
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
    behavior_source = "sac_checkpoint" if sac_ckpt_path is not None else "rule_based"
    metadata = {
        "policy": worker_config.policy_name,
        "policy_mixture": [
            {"policy": name, "weight": weight} for name, weight in worker_config.policy_mixture
        ],
        "behavior_source": behavior_source,
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
        "include_episode_context_obs": bool(
            env_config_overrides.get("reward_objective") in ARRIVAL_V2_OBJECTIVES
        ),
        "timeout_bootstrap_semantics": (
            "terminal"
            if env_config_overrides.get("reward_objective") in ARRIVAL_V2_OBJECTIVES
            else "bootstrap"
        ),
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
    if sac_metadata:
        metadata.update(sac_metadata)
    meta_path = output_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(
        f"\n[done] Saved {n_transitions} transitions to {npz_path}\n"
        f"  policy={worker_config.policy_name}  obs_dim={obs_dim}  action_dim={action_dim}\n"
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
        default=None,
        help="Baseline policy to run. Required unless --sac-ckpt is set.",
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
    # SAC-collector mode (mutually exclusive with --policy/--policy-mixture).
    parser.add_argument(
        "--sac-ckpt",
        type=str,
        default=None,
        help=(
            "Path to a SAC checkpoint .pt to use as the behavior policy. "
            "Mutually exclusive with --policy / --policy-mixture."
        ),
    )
    parser.add_argument(
        "--sac-deterministic",
        action="store_true",
        help=(
            "Sample the mean action (D4RL expert standard). Without the flag "
            "the actor samples from its Gaussian — the D4RL medium/replay style."
        ),
    )
    parser.add_argument(
        "--sac-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Torch device for SAC actor inference (default: cpu).",
    )
    parser.add_argument(
        "--sac-trainer-state",
        type=str,
        default=None,
        help=(
            "Explicit path to trainer_state.json for Layer-2 protocol sanity. "
            "If omitted, autodetected via the checkpoints/<X> ↔ experiments/<X> "
            "directory mirror."
        ),
    )
    parser.add_argument(
        "--sac-skip-trainer-state-check",
        action="store_true",
        help=(
            "Skip the Layer-2 trainer_state.json protocol check when it cannot "
            "be located (use only when you trust the CLI protocol args)."
        ),
    )
    args = parser.parse_args()
    if args.sac_ckpt is None and args.policy is None:
        parser.error("Either --policy or --sac-ckpt is required.")
    if args.sac_ckpt is not None and args.policy is not None:
        parser.error("--policy and --sac-ckpt are mutually exclusive.")
    if args.sac_ckpt is not None and args.policy_mixture:
        parser.error("--policy-mixture is not supported with --sac-ckpt.")
    collect(args)


if __name__ == "__main__":
    main()
