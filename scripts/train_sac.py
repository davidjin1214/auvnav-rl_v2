from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym

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
    num_envs: int = 1


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
        "num_envs": "num_envs",
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
    
    if train_cfg.num_envs > 1:
        def make_env_fn(seed_offset):
            def _thunk():
                env = make_planar_env(flow_path, history_length=train_cfg.history_length)
                env.action_space.seed(train_cfg.seed + seed_offset)
                return env
            return _thunk
        env = gym.vector.AsyncVectorEnv([make_env_fn(i) for i in range(train_cfg.num_envs)])
        # AsyncVectorEnv spaces are Batched. We need the unbatched shape for Agent
        obs_dim = int(env.single_observation_space.shape[0])
        action_dim = int(env.single_action_space.shape[0])
    else:
        env = make_planar_env(flow_path, history_length=train_cfg.history_length)
        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
        
    eval_env = make_planar_env(flow_path, history_length=train_cfg.history_length)
    reset_options = make_reset_options(args)

    agent_cfg = SACConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        batch_size=train_cfg.batch_size,
        updates_per_step=train_cfg.updates_per_step,
        use_layernorm=args.use_layernorm,
        dropout_rate=args.dropout_rate,
        privileged_obs_dim=2 if args.use_asymmetric_critic else 0,
    )
    agent = SACAgent(config=agent_cfg, device=train_cfg.device)
    replay = TransitionReplay(
        obs_dim=agent_cfg.obs_dim,
        action_dim=agent_cfg.action_dim,
        config=TransitionReplayConfig(
            capacity=train_cfg.replay_capacity,
            privileged_obs_dim=agent_cfg.privileged_obs_dim,
        ),
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
    
    is_vector_env = train_cfg.num_envs > 1
    num_envs = train_cfg.num_envs if is_vector_env else 1
    
    episode_return = np.zeros(num_envs, dtype=float)
    episode_cost = np.zeros(num_envs, dtype=float)
    episode_length = np.zeros(num_envs, dtype=int)
    episode_idx = start_episode
    last_update: dict[str, float] = {}

    for env_step in range(start_step + 1, (train_cfg.total_env_steps // num_envs) + 1):
        global_step = env_step * num_envs
        if global_step <= train_cfg.random_steps:
            action = env.action_space.sample()
        else:
            action, policy_state = agent.act(obs, policy_state, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)

        if is_vector_env:
            has_final_obs = "final_observation" in info
            has_final_info = "final_info" in info
            has_step_cost = "step_safety_cost" in info
            has_priv_obs = "privileged_obs" in info
            has_success = "success" in info
            has_elapsed = "elapsed_time_s" in info
            has_geom = "task_geometry" in info
            has_target = "target_auv_max_speed_mps" in info
            _false_sentinel = info.get("_final_observation", None)
            _final_info_mask = info.get("_final_info", None)

            for i in range(num_envs):
                done_i = terminated[i] or truncated[i]
                if has_final_obs and _false_sentinel is not None and _false_sentinel[i]:
                    real_next_obs = info["final_observation"][i]
                else:
                    real_next_obs = next_obs[i]

                step_cost = float(info["step_safety_cost"][i]) if has_step_cost else 0.0
                priv_obs = info["privileged_obs"][i] if has_priv_obs else None

                replay.add(
                    obs=obs[i],
                    action=action[i],
                    reward=float(reward[i]),
                    cost=step_cost,
                    next_obs=real_next_obs,
                    done=bool(terminated[i]),
                    privileged_obs=priv_obs,
                )
                episode_return[i] += float(reward[i])
                episode_cost[i] += step_cost
                episode_length[i] += 1

                if done_i:
                    episode_idx += 1

                    success = bool(info["success"][i]) if has_success else False
                    elapsed_time_s = float(info["elapsed_time_s"][i]) if has_elapsed else 0.0
                    task_geom = info["task_geometry"][i] if has_geom else ""
                    target_auv = float(info["target_auv_max_speed_mps"][i]) if has_target else 0.0

                    if has_final_info and _final_info_mask is not None and _final_info_mask[i]:
                        f_info = info["final_info"][i]
                        success = bool(f_info.get("success", success))
                        elapsed_time_s = float(f_info.get("elapsed_time_s", elapsed_time_s))
                        task_geom = f_info.get("task_geometry", task_geom)
                        target_auv = float(f_info.get("target_auv_max_speed_mps", target_auv))

                    append_jsonl(
                        train_log_path,
                        {
                            "episode": episode_idx,
                            "env_step": global_step,
                            "return": float(episode_return[i]),
                            "episode_cost": float(episode_cost[i]),
                            "success": success,
                            "elapsed_time_s": elapsed_time_s,
                            "episode_length": int(episode_length[i]),
                            "task_geometry": task_geom,
                            "target_auv_max_speed_mps": target_auv,
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
                            f"[train] episode={episode_idx} step={global_step} "
                            f"return={episode_return[i]:.2f} success={success} "
                            f"time={elapsed_time_s:.1f}s "
                            f"geometry={task_geom} "
                            f"history={train_cfg.history_length}"
                            f"{update_msg}"
                        )
                        
                    episode_return[i] = 0.0
                    episode_cost[i] = 0.0
                    episode_length[i] = 0
        else:
            done = terminated or truncated
            replay.add(
                obs=obs,
                action=action,
                reward=float(reward),
                cost=float(info["step_safety_cost"]),
                next_obs=next_obs,
                done=terminated,
                privileged_obs=info.get("privileged_obs"),
            )

            episode_return[0] += float(reward)
            episode_cost[0] += float(info["step_safety_cost"])
            episode_length[0] += 1

            if done:
                episode_idx += 1
                append_jsonl(
                    train_log_path,
                    {
                        "episode": episode_idx,
                        "env_step": global_step,
                        "return": float(episode_return[0]),
                        "episode_cost": float(episode_cost[0]),
                        "success": bool(info["success"]),
                        "elapsed_time_s": float(info["elapsed_time_s"]),
                        "episode_length": int(episode_length[0]),
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
                        f"[train] episode={episode_idx} step={global_step} "
                        f"return={episode_return[0]:.2f} success={info['success']} "
                        f"time={info['elapsed_time_s']:.1f}s "
                        f"geometry={info['task_geometry']} "
                        f"history={train_cfg.history_length}"
                        f"{update_msg}"
                    )

                obs, info = env.reset(seed=train_cfg.seed + episode_idx, options=reset_options)
                episode_return[0] = 0.0
                episode_cost[0] = 0.0
                episode_length[0] = 0

        obs = next_obs

        if (
            global_step >= train_cfg.update_after
            and replay.ready(train_cfg.batch_size)
            and global_step % train_cfg.update_every == 0
        ):
            for _ in range(train_cfg.updates_per_step):
                batch = replay.sample_batch(train_cfg.batch_size, agent.device)
                last_update = agent.update(batch)

        if global_step % train_cfg.eval_every_steps == 0:
            metrics = evaluate_agent(
                env=eval_env,
                agent=agent,
                reset_options=reset_options,
                seed=train_cfg.seed + 10_000 + global_step,
                num_episodes=train_cfg.eval_episodes,
            )
            print(
                f"[eval] step={global_step} "
                f"return={metrics['eval_return']:.2f} "
                f"cost={metrics['eval_cost']:.3f} "
                f"success={metrics['eval_success_rate']:.2%} "
                f"time={metrics['eval_time_s']:.1f}s"
            )
            append_csv(
                eval_log_path,
                {
                    "env_step": global_step,
                    "eval_return": metrics["eval_return"],
                    "eval_cost": metrics["eval_cost"],
                    "eval_success_rate": metrics["eval_success_rate"],
                    "eval_time_s": metrics["eval_time_s"],
                },
            )
            agent.save(str(save_dir / f"agent_step_{global_step}.pt"))
            save_training_state(
                save_dir=save_dir,
                agent=agent,
                replay=replay,
                train_config=asdict(train_cfg),
                agent_config=asdict(agent_cfg),
                reset_options=reset_options,
                flow_path=str(flow_path),
                env_step=global_step,
                episode_idx=episode_idx,
                extra_state={
                    "algorithm": "sac",
                    "history_length": train_cfg.history_length,
                },
            )

        if global_step % train_cfg.checkpoint_every_steps == 0:
            save_training_state(
                save_dir=save_dir,
                agent=agent,
                replay=replay,
                train_config=asdict(train_cfg),
                agent_config=asdict(agent_cfg),
                reset_options=reset_options,
                flow_path=str(flow_path),
                env_step=global_step,
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
        "--use-layernorm", action="store_true", default=False,
        help="Enable LayerNorm in Actor and Critic networks (DroQ/improved SAC).",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0, metavar="P",
        help="Dropout rate for hidden layers (0.0 = off). DroQ recommends 0.01.",
    )
    parser.add_argument(
        "--use-asymmetric-critic", action="store_true", default=False,
        help="Enable Asymmetric Critic: Critic sees privileged_obs from env info.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=16)
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
