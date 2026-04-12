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
import time
from pathlib import Path

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


def collect(args: argparse.Namespace) -> None:
    flow_path = args.flow or discover_flow_path()
    env_config_overrides = make_env_config_overrides(args)
    env = make_planar_env(
        flow_path,
        history_length=args.history_length,
        probe_layout=args.probe_layout,
        env_config_overrides=env_config_overrides,
    )

    # Baseline policies need the unwrapped PlanarRemusEnv for env-specific calls.
    if isinstance(env, ObservationHistoryWrapper):
        base_env = env.env
    else:
        base_env = env

    policy = POLICY_MAP[args.policy]()
    reset_options = make_reset_options(args)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_rewards: list[float] = []
    all_costs: list[float] = []
    all_next_obs: list[np.ndarray] = []
    all_dones: list[bool] = []
    all_privileged_obs: list[np.ndarray] = []
    all_next_privileged_obs: list[np.ndarray] = []

    successes = 0
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    t0 = time.time()
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep, options=reset_options)
        current_privileged_obs = info.get("privileged_obs")
        ep_return = 0.0
        ep_length = 0
        done = False

        while not done:
            # Get single-step obs for policies that call decode_observation.
            if isinstance(env, ObservationHistoryWrapper):
                single_obs = np.asarray(env._history[-1], dtype=np.float32)
            else:
                single_obs = obs

            action = policy.act(base_env, single_obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            all_obs.append(np.asarray(obs, dtype=np.float32))
            all_actions.append(np.asarray(action, dtype=np.float32))
            all_rewards.append(float(reward))
            all_costs.append(float(info.get("step_safety_cost", 0.0)))
            all_next_obs.append(np.asarray(next_obs, dtype=np.float32))
            all_dones.append(bool(terminated))
            if current_privileged_obs is not None:
                all_privileged_obs.append(np.asarray(current_privileged_obs, dtype=np.float32))
            next_privileged_obs = info.get("privileged_obs")
            if next_privileged_obs is not None:
                all_next_privileged_obs.append(np.asarray(next_privileged_obs, dtype=np.float32))

            ep_return += float(reward)
            ep_length += 1
            obs = next_obs
            current_privileged_obs = next_privileged_obs
            done = terminated or truncated

        successes += int(info.get("success", False))
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)

        if (ep + 1) % max(1, args.episodes // 10) == 0:
            elapsed = time.time() - t0
            print(
                f"[collect] ep={ep + 1}/{args.episodes} "
                f"transitions={len(all_obs)} "
                f"success_rate={successes / (ep + 1):.2%} "
                f"elapsed={elapsed:.1f}s"
            )

    # Save transitions as compressed npz.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "transitions.npz"
    payload = {
        "obs": np.array(all_obs, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "costs": np.array(all_costs, dtype=np.float32),
        "next_obs": np.array(all_next_obs, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.float32),
    }
    if all_privileged_obs:
        payload["privileged_obs"] = np.array(all_privileged_obs, dtype=np.float32)
    if all_next_privileged_obs:
        payload["next_privileged_obs"] = np.array(all_next_privileged_obs, dtype=np.float32)
    np.savez_compressed(str(npz_path), **payload)

    # Save metadata.
    n_transitions = len(all_obs)
    metadata = {
        "policy": args.policy,
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
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save transitions.npz and metadata.json.",
    )
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
