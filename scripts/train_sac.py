from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from auv_nav.sac import SACAgent, SACConfig
from auv_nav.replay import TransitionReplay, TransitionReplayConfig
from auv_nav.networks import require_torch

try:
    import torch
except ImportError:
    torch = None

from .train_utils import (
    append_csv,
    append_jsonl,
    discover_flow_path,
    evaluate_agent,
    load_trainer_state,
    make_planar_env,
    make_reset_options,
    maybe_resume,
    save_training_state,
)


@dataclass(slots=True)
class TrainConfig:
    total_env_steps: int = 50_000
    seed: int = 42
    random_steps: int = 2_000
    update_after: int = 2_000
    update_every: int = 1
    updates_per_step: int = 1
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    eval_every_steps: int = 5_000
    eval_episodes: int = 5
    log_every_episodes: int = 5
    save_dir: str = "checkpoints/sac"
    device: str = "cpu"
    checkpoint_every_steps: int = 5_000
    history_length: int = 1


def apply_resume_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.resume is None:
        return

    trainer_state = load_trainer_state(args.resume)
    train_state = trainer_state.get("train_config", {})
    agent_state = trainer_state.get("agent_config", {})
    reset_state = trainer_state.get("reset_options", {})

    def apply_default(arg_name: str, saved_value: Any) -> None:
        if saved_value is None:
            return
        if getattr(args, arg_name) == parser.get_default(arg_name):
            setattr(args, arg_name, saved_value)

    train_arg_map = {
        "total_steps": "total_env_steps",
        "seed": "seed",
        "random_steps": "random_steps",
        "update_after": "update_after",
        "update_every": "update_every",
        "updates_per_step": "updates_per_step",
        "batch_size": "batch_size",
        "replay_capacity": "replay_capacity",
        "eval_every": "eval_every_steps",
        "eval_episodes": "eval_episodes",
        "log_every_episodes": "log_every_episodes",
        "save_dir": "save_dir",
        "device": "device",
        "checkpoint_every": "checkpoint_every_steps",
        "history_length": "history_length",
    }
    for arg_name, state_name in train_arg_map.items():
        apply_default(arg_name, train_state.get(state_name))

    apply_default("flow", trainer_state.get("flow_path"))
    apply_default("difficulty", reset_state.get("task_difficulty"))
    apply_default("task_geometry", reset_state.get("task_geometry"))
    apply_default("action_mode", reset_state.get("action_mode"))
    apply_default("target_speed", reset_state.get("target_auv_max_speed_mps"))
    if args.target_speed == parser.get_default("target_speed"):
        apply_default("speed_ratio", reset_state.get("target_speed_ratio"))

    apply_default("hidden_dim", agent_state.get("hidden_dim"))


def train(args: argparse.Namespace) -> None:
    require_torch()

    train_cfg = TrainConfig(
        total_env_steps=args.total_steps,
        seed=args.seed,
        random_steps=args.random_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        updates_per_step=args.updates_per_step,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        eval_every_steps=args.eval_every,
        eval_episodes=args.eval_episodes,
        log_every_episodes=args.log_every_episodes,
        save_dir=args.save_dir,
        device=args.device,
        checkpoint_every_steps=args.checkpoint_every,
        history_length=args.history_length,
    )

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    if torch is not None:
        torch.manual_seed(train_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_cfg.seed)

    flow_path = args.flow or discover_flow_path()
    env = make_planar_env(flow_path, history_length=train_cfg.history_length)
    eval_env = make_planar_env(flow_path, history_length=train_cfg.history_length)
    reset_options = make_reset_options(args)

    agent_cfg = SACConfig(
        obs_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        hidden_dim=args.hidden_dim,
        batch_size=train_cfg.batch_size,
        updates_per_step=train_cfg.updates_per_step,
    )
    agent = SACAgent(config=agent_cfg, device=train_cfg.device)
    replay = TransitionReplay(
        obs_dim=agent_cfg.obs_dim,
        action_dim=agent_cfg.action_dim,
        config=TransitionReplayConfig(capacity=train_cfg.replay_capacity),
    )

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = save_dir / "train_log.jsonl"
    eval_log_path = save_dir / "eval_log.csv"

    start_step, start_episode = maybe_resume(
        agent=agent,
        replay=replay,
        resume_path=args.resume,
    )

    obs, info = env.reset(seed=train_cfg.seed + start_episode, options=reset_options)
    policy_state = agent.reset_policy_state()
    episode_return = 0.0
    episode_cost = 0.0
    episode_length = 0
    episode_idx = start_episode
    last_update: dict[str, float] = {}

    for env_step in range(start_step + 1, train_cfg.total_env_steps + 1):
        if env_step <= train_cfg.random_steps:
            action = env.action_space.sample()
        else:
            action, policy_state = agent.act(obs, policy_state, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay.add(
            obs=obs,
            action=action,
            reward=float(reward),
            cost=float(info["step_safety_cost"]),
            next_obs=next_obs,
            # Time-limit truncation should still bootstrap in the critic target.
            done=terminated,
        )

        episode_return += float(reward)
        episode_cost += float(info["step_safety_cost"])
        episode_length += 1
        obs = next_obs

        if (
            env_step >= train_cfg.update_after
            and replay.ready(train_cfg.batch_size)
            and env_step % train_cfg.update_every == 0
        ):
            for _ in range(train_cfg.updates_per_step):
                batch = replay.sample_batch(train_cfg.batch_size, agent.device)
                last_update = agent.update(batch)

        if done:
            episode_idx += 1
            append_jsonl(
                train_log_path,
                {
                    "episode": episode_idx,
                    "env_step": env_step,
                    "return": episode_return,
                    "episode_cost": episode_cost,
                    "success": bool(info["success"]),
                    "elapsed_time_s": float(info["elapsed_time_s"]),
                    "episode_length": episode_length,
                    "task_geometry": info["task_geometry"],
                    "target_auv_max_speed_mps": float(info["target_auv_max_speed_mps"]),
                    **last_update,
                },
            )
            if episode_idx % train_cfg.log_every_episodes == 0:
                update_msg = ""
                if last_update:
                    update_msg = (
                        f" | q1={last_update['q1_loss']:.3f}"
                        f" actor={last_update['actor_loss']:.3f}"
                        f" alpha={last_update['alpha']:.3f}"
                    )
                print(
                    f"[train] episode={episode_idx} step={env_step} "
                    f"return={episode_return:.2f} success={info['success']} "
                    f"time={info['elapsed_time_s']:.1f}s "
                    f"geometry={info['task_geometry']} "
                    f"history={train_cfg.history_length}"
                    f"{update_msg}"
                )

            obs, info = env.reset(seed=train_cfg.seed + episode_idx, options=reset_options)
            policy_state = agent.reset_policy_state()
            episode_return = 0.0
            episode_cost = 0.0
            episode_length = 0

        if env_step % train_cfg.eval_every_steps == 0:
            metrics = evaluate_agent(
                env=eval_env,
                agent=agent,
                reset_options=reset_options,
                seed=train_cfg.seed + 10_000 + env_step,
                num_episodes=train_cfg.eval_episodes,
            )
            print(
                f"[eval] step={env_step} "
                f"return={metrics['eval_return']:.2f} "
                f"cost={metrics['eval_cost']:.3f} "
                f"success={metrics['eval_success_rate']:.2%} "
                f"time={metrics['eval_time_s']:.1f}s"
            )
            append_csv(
                eval_log_path,
                {
                    "env_step": env_step,
                    "eval_return": metrics["eval_return"],
                    "eval_cost": metrics["eval_cost"],
                    "eval_success_rate": metrics["eval_success_rate"],
                    "eval_time_s": metrics["eval_time_s"],
                },
            )
            agent.save(str(save_dir / f"agent_step_{env_step}.pt"))
            save_training_state(
                save_dir=save_dir,
                agent=agent,
                replay=replay,
                train_config=asdict(train_cfg),
                agent_config=asdict(agent_cfg),
                reset_options=reset_options,
                flow_path=str(flow_path),
                env_step=env_step,
                episode_idx=episode_idx,
                extra_state={
                    "algorithm": "sac",
                    "history_length": train_cfg.history_length,
                },
            )

        if env_step % train_cfg.checkpoint_every_steps == 0:
            save_training_state(
                save_dir=save_dir,
                agent=agent,
                replay=replay,
                train_config=asdict(train_cfg),
                agent_config=asdict(agent_cfg),
                reset_options=reset_options,
                flow_path=str(flow_path),
                env_step=env_step,
                episode_idx=episode_idx,
                extra_state={
                    "algorithm": "sac",
                    "history_length": train_cfg.history_length,
                },
            )

    agent.save(str(save_dir / "agent_final.pt"))
    save_training_state(
        save_dir=save_dir,
        agent=agent,
        replay=replay,
        train_config=asdict(train_cfg),
        agent_config=asdict(agent_cfg),
        reset_options=reset_options,
        flow_path=str(flow_path),
        env_step=train_cfg.total_env_steps,
        episode_idx=episode_idx,
        extra_state={
            "algorithm": "sac",
            "history_length": train_cfg.history_length,
        },
    )
    final_metrics = evaluate_agent(
        env=eval_env,
        agent=agent,
        reset_options=reset_options,
        seed=train_cfg.seed + 20_000,
        num_episodes=train_cfg.eval_episodes,
    )
    with (save_dir / "final_eval.json").open("w", encoding="utf-8") as fp:
        json.dump(final_metrics, fp, indent=2)
    with (save_dir / "train_config.txt").open("w", encoding="utf-8") as fp:
        fp.write("TrainConfig\n")
        for key, value in asdict(train_cfg).items():
            fp.write(f"{key}={value}\n")
        fp.write("AgentConfig\n")
        for key, value in asdict(agent_cfg).items():
            fp.write(f"{key}={value}\n")
        fp.write("ResetOptions\n")
        for key, value in reset_options.items():
            fp.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a standard SAC baseline.")
    parser.add_argument("--flow", type=Path, default=None, help="Path to wake ROI .npy file.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument(
        "--task-geometry",
        choices=["downstream", "cross_stream", "upstream"],
        default=None,
    )
    parser.add_argument(
        "--action-mode",
        choices=["auto", "goal_relative_offset", "absolute_heading"],
        default=None,
    )
    parser.add_argument("--speed-ratio", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--random-steps", type=int, default=2_000)
    parser.add_argument("--update-after", type=int, default=2_000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-every-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints/sac")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--history-length",
        type=int,
        default=1,
        help="Number of recent observations to stack for feedforward history baselines.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    apply_resume_defaults(args, parser)
    train(args)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
