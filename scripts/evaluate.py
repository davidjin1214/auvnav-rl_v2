"""Evaluate a saved SAC or GRU-Residual-SAC checkpoint.

Auto-detects the algorithm from the saved trainer_state.json.
SAC checkpoints contain a flat agent_config (obs_dim, hidden_dim, ...).
GRU-SAC checkpoints additionally contain encoder_dim in agent_config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from auv_nav.networks import require_torch
from .train_utils import build_residual_prior, load_trainer_state, make_planar_env


def _is_gru_checkpoint(trainer_state: dict) -> bool:
    return "encoder_dim" in trainer_state.get("agent_config", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved SAC or GRU-SAC checkpoint.")
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
                        help="Override for stacked-obs evaluation (SAC only).")
    args = parser.parse_args()

    require_torch()
    trainer_state = load_trainer_state(args.checkpoint)
    agent_cfg_dict = trainer_state["agent_config"]

    reset_options = dict(trainer_state.get("reset_options", {}))
    if args.difficulty is not None:
        reset_options["task_difficulty"] = args.difficulty
    if args.task_geometry is not None:
        reset_options["task_geometry"] = args.task_geometry
    if args.action_mode is not None:
        reset_options["action_mode"] = args.action_mode
    if args.target_speed is not None:
        reset_options["target_auv_max_speed_mps"] = args.target_speed
    elif args.speed_ratio is not None:
        reset_options["target_speed_ratio"] = args.speed_ratio

    flow_path = args.flow or trainer_state.get("flow_path", "wake_data/wake_test_roi.npy")
    checkpoint_root = Path(args.checkpoint)
    agent_file = (
        checkpoint_root / trainer_state["agent_path"]
        if checkpoint_root.is_dir()
        else checkpoint_root.parent / trainer_state["agent_path"]
    )

    if _is_gru_checkpoint(trainer_state):
        from auv_nav.env import PlanarRemusEnv, PlanarRemusEnvConfig
        from auv_nav.gru_sac import GRUResidualSACAgent, GRUResidualSACConfig
        env = PlanarRemusEnv(PlanarRemusEnvConfig(flow_path=flow_path))
        resolved_geometry = env.task_sampler.resolve_task_geometry(reset_options)
        resolved_mode = env.task_sampler.resolve_action_mode(reset_options, resolved_geometry)
        agent = GRUResidualSACAgent(
            GRUResidualSACConfig(**agent_cfg_dict),
            prior=build_residual_prior(env, resolved_mode),
            device=args.device,
        )
        gru_mode = True
    else:
        history_length = (
            args.history_length
            if args.history_length is not None
            else int(trainer_state.get("history_length", 1))
        )
        env = make_planar_env(flow_path, history_length=history_length)
        from auv_nav.sac import SACAgent, SACConfig
        agent = SACAgent(SACConfig(**agent_cfg_dict), device=args.device)
        gru_mode = False

    agent.load(str(agent_file))

    returns = []
    costs = []
    successes = 0
    times = []
    reasons: dict[str, int] = {}
    for idx in range(args.episodes):
        obs, info = env.reset(seed=args.seed + idx, options=reset_options)
        policy_state = agent.reset_policy_state() if not gru_mode else agent.reset_hidden()
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
