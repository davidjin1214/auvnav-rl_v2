"""Evaluate a saved SAC checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from auv_nav.sac import SACAgent, SACConfig
from .train_utils import load_trainer_state, make_planar_env, make_reset_options


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved SAC checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Run directory or trainer_state.json path.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--flow", type=str, default=None)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--task-geometry",
                        choices=["downstream", "cross_stream", "upstream"], default=None)
    parser.add_argument("--action-mode",
                        choices=["auto", "goal_relative_offset", "absolute_heading"],
                        default=None)
    parser.add_argument("--speed-ratio", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--history-length", type=int, default=None,
                        help="Override stacked-observation history length.")
    args = parser.parse_args()

    trainer_state = load_trainer_state(args.checkpoint)
    agent_cfg_dict = trainer_state["agent_config"]

    reset_options = {**trainer_state.get("reset_options", {}), **make_reset_options(args)}

    flow_path = args.flow or trainer_state.get("flow_path", "wake_data/wake_test_roi.npy")
    history_length = (
        args.history_length
        if args.history_length is not None
        else int(trainer_state.get("history_length", 1))
    )
    env = make_planar_env(flow_path, history_length=history_length)
    agent = SACAgent(SACConfig(**agent_cfg_dict), device=args.device)

    checkpoint_root = Path(args.checkpoint)
    agent_file = (
        checkpoint_root / trainer_state["agent_path"]
        if checkpoint_root.is_dir()
        else checkpoint_root.parent / trainer_state["agent_path"]
    )
    agent.load(str(agent_file))

    returns = []
    costs = []
    successes = 0
    times = []
    reasons: dict[str, int] = {}
    for idx in range(args.episodes):
        obs, info = env.reset(seed=args.seed + idx, options=reset_options)
        policy_state = agent.reset_policy_state()
        total_reward = 0.0
        total_cost = 0.0
        done = False
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
        reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1

    print(f"episodes          : {args.episodes}")
    print(
        f"success_rate      : {successes}/{args.episodes} "
        f"({100.0 * successes / max(1, args.episodes):.1f}%)"
    )
    print(f"avg_return        : {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    print(f"avg_cost          : {np.mean(costs):.3f} +/- {np.std(costs):.3f}")
    print(f"avg_time_s        : {np.mean(times):.2f} +/- {np.std(times):.2f}")
    print(f"termination       : {reasons}")
    print(f"task_geometry     : {info['task_geometry']}")
    print(f"target_u_max_mps  : {info['target_auv_max_speed_mps']:.2f}")
    print(f"max_rpm_command   : {info['max_rpm_command']:.0f}")


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
