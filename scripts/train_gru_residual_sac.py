from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from auv_nav.env import PlanarRemusEnv, PlanarRemusEnvConfig
from auv_nav.gru_sac import GRUResidualSACAgent, GRUResidualSACConfig
from auv_nav.replay import EpisodeSequenceReplay, SequenceReplayConfig
from auv_nav.networks import require_torch

try:
    import torch
except ImportError:
    torch = None

from .train_utils import (
    append_csv,
    append_jsonl,
    build_residual_prior,
    collect_offline_episodes,
    discover_flow_path,
    evaluate_agent,
    load_trainer_state,
    make_reset_options,
    maybe_resume,
    pretrain_from_replay,
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
    batch_size: int = 32
    burn_in: int = 8
    train_seq_len: int = 32
    replay_max_episodes: int = 5_000
    eval_every_steps: int = 5_000
    eval_episodes: int = 5
    log_every_episodes: int = 5
    save_dir: str = "checkpoints/gru_residual_sac"
    device: str = "cpu"
    checkpoint_every_steps: int = 5_000
    offline_episodes: int = 0
    pretrain_updates: int = 0


OFFLINE_POLICY_REGISTRY = {
    "goal": "goal",
    "compensate": "compensate",
    "still_water": "still_water",
    "world_compensate": "world_compensate",
    "corridor": "corridor",
}


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
        "burn_in": "burn_in",
        "train_seq_len": "train_seq_len",
        "replay_max_episodes": "replay_max_episodes",
        "eval_every": "eval_every_steps",
        "eval_episodes": "eval_episodes",
        "log_every_episodes": "log_every_episodes",
        "save_dir": "save_dir",
        "device": "device",
        "checkpoint_every": "checkpoint_every_steps",
        "offline_episodes": "offline_episodes",
        "pretrain_updates": "pretrain_updates",
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

    apply_default("residual_l2_weight", agent_state.get("residual_l2_weight"))
    apply_default("bc_weight", agent_state.get("behavior_cloning_weight"))
    apply_default("bc_decay_steps", agent_state.get("behavior_cloning_decay_steps"))
    if args.disable_residual_prior == parser.get_default("disable_residual_prior"):
        saved_use_prior = agent_state.get("use_residual_prior")
        if saved_use_prior is not None:
            args.disable_residual_prior = not bool(saved_use_prior)
    apply_default("cost_limit", agent_state.get("cost_limit"))
    if args.disable_safety_critic == parser.get_default("disable_safety_critic"):
        saved_use_safety_critic = agent_state.get("use_safety_critic")
        if saved_use_safety_critic is not None:
            args.disable_safety_critic = not bool(saved_use_safety_critic)


def train(args: argparse.Namespace) -> None:
    require_torch()
    # TODO: Add vectorized / parallel environment collection once the
    # simulator is upgraded to a GPU-friendly implementation.

    train_cfg = TrainConfig(
        total_env_steps=args.total_steps,
        seed=args.seed,
        random_steps=args.random_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        updates_per_step=args.updates_per_step,
        batch_size=args.batch_size,
        burn_in=args.burn_in,
        train_seq_len=args.train_seq_len,
        replay_max_episodes=args.replay_max_episodes,
        eval_every_steps=args.eval_every,
        eval_episodes=args.eval_episodes,
        log_every_episodes=args.log_every_episodes,
        save_dir=args.save_dir,
        device=args.device,
        checkpoint_every_steps=args.checkpoint_every,
        offline_episodes=args.offline_episodes,
        pretrain_updates=args.pretrain_updates,
    )

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    if torch is not None:
        torch.manual_seed(train_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_cfg.seed)

    flow_path = args.flow or discover_flow_path()
    env_cfg = PlanarRemusEnvConfig(flow_path=flow_path)
    env = PlanarRemusEnv(env_cfg)
    eval_env = PlanarRemusEnv(env_cfg)
    reset_options = make_reset_options(args)
    resolved_task_geometry = env.task_sampler.resolve_task_geometry(reset_options)
    resolved_action_mode = env.task_sampler.resolve_action_mode(
        reset_options, resolved_task_geometry
    )

    agent_cfg = GRUResidualSACConfig(
        obs_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        burn_in=train_cfg.burn_in,
        train_seq_len=train_cfg.train_seq_len,
        batch_size=train_cfg.batch_size,
        updates_per_step=train_cfg.updates_per_step,
        residual_l2_weight=args.residual_l2_weight,
        behavior_cloning_weight=args.bc_weight,
        behavior_cloning_decay_steps=args.bc_decay_steps,
        use_residual_prior=not args.disable_residual_prior,
        use_safety_critic=not args.disable_safety_critic,
        cost_limit=args.cost_limit,
    )
    agent = GRUResidualSACAgent(
        config=agent_cfg,
        prior=build_residual_prior(env, resolved_action_mode),
        device=train_cfg.device,
    )

    replay = EpisodeSequenceReplay(
        SequenceReplayConfig(
            max_episodes=train_cfg.replay_max_episodes,
            burn_in=train_cfg.burn_in,
            train_seq_len=train_cfg.train_seq_len,
        )
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

    if start_step == 0 and train_cfg.offline_episodes > 0:
        # TODO: Split offline pretraining and online fine-tuning into separate
        # stages with their own logging and checkpoint boundaries.
        collect_offline_episodes(
            env=env,
            replay=replay,
            reset_options=reset_options,
            num_episodes=train_cfg.offline_episodes,
            policy_name=args.offline_policy,
            seed=train_cfg.seed + 50_000,
        )
        print(
            f"[offline] collected episodes={train_cfg.offline_episodes} "
            f"policy={args.offline_policy} transitions={len(replay)}"
        )
        warmstart_metrics = pretrain_from_replay(
            agent=agent,
            replay=replay,
            updates=train_cfg.pretrain_updates,
            batch_size=train_cfg.batch_size,
        )
        if warmstart_metrics:
            print(
                f"[offline] pretrain_updates={train_cfg.pretrain_updates} "
                f"actor={warmstart_metrics['actor_total_loss']:.3f} "
                f"q1={warmstart_metrics['q1_loss']:.3f} "
                f"cost={warmstart_metrics['qc_loss']:.3f}"
            )

    obs, info = env.reset(seed=train_cfg.seed + start_episode, options=reset_options)
    hidden = agent.reset_hidden()
    episode_obs = [obs.copy()]
    episode_actions: list[np.ndarray] = []
    episode_rewards: list[float] = []
    episode_costs: list[float] = []
    episode_dones: list[float] = []
    episode_return = 0.0
    episode_idx = start_episode
    last_update: dict[str, float] = {}

    for env_step in range(start_step + 1, train_cfg.total_env_steps + 1):
        if env_step <= train_cfg.random_steps:
            hidden = agent.act(obs, hidden, deterministic=True)[1]
            action = env.action_space.sample()
        else:
            action, hidden = agent.act(obs, hidden, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_actions.append(np.asarray(action, dtype=np.float32))
        episode_rewards.append(float(reward))
        episode_costs.append(float(info["step_safety_cost"]))
        # Time-limit truncation should still bootstrap in the critic target.
        episode_dones.append(float(terminated))
        episode_obs.append(next_obs.copy())
        episode_return += float(reward)
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
            replay.add_episode(
                observations=np.asarray(episode_obs, dtype=np.float32),
                actions=np.asarray(episode_actions, dtype=np.float32),
                rewards=np.asarray(episode_rewards, dtype=np.float32),
                costs=np.asarray(episode_costs, dtype=np.float32),
                dones=np.asarray(episode_dones, dtype=np.float32),
            )

            episode_idx += 1
            append_jsonl(
                train_log_path,
                {
                    "episode": episode_idx,
                    "env_step": env_step,
                    "return": episode_return,
                    "episode_cost": float(np.sum(episode_costs)),
                    "success": bool(info["success"]),
                    "elapsed_time_s": float(info["elapsed_time_s"]),
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
                        f" actor={last_update['actor_total_loss']:.3f}"
                        f" alpha={last_update['alpha']:.3f}"
                    )
                print(
                    f"[train] episode={episode_idx} step={env_step} "
                    f"return={episode_return:.2f} success={info['success']} "
                    f"time={info['elapsed_time_s']:.1f}s "
                    f"geometry={info['task_geometry']} "
                    f"u_max={info['target_auv_max_speed_mps']:.2f}"
                    f"{update_msg}"
                )

            obs, info = env.reset(seed=train_cfg.seed + episode_idx, options=reset_options)
            hidden = agent.reset_hidden()
            episode_obs = [obs.copy()]
            episode_actions = []
            episode_rewards = []
            episode_costs = []
            episode_dones = []
            episode_return = 0.0

        if env_step % train_cfg.eval_every_steps == 0:
            # TODO: Extend evaluation to sweep multiple benchmark geometries
            # and speed ratios in one pass, then store per-split metrics.
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
                extra_state={"algorithm": "gru_residual_sac"},
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
                extra_state={"algorithm": "gru_residual_sac"},
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
        extra_state={"algorithm": "gru_residual_sac"},
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
    parser = argparse.ArgumentParser(description="Train a GRU Residual SAC agent.")
    # TODO: Add config-file support so full experiment settings can be stored
    # and reproduced without passing long CLI argument lists.
    parser.add_argument("--flow", type=Path, default=None,
                        help="Path to wake ROI .npy file.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None,
                        help="Benchmark difficulty preset.")
    parser.add_argument("--task-geometry",
                        choices=["downstream", "cross_stream", "upstream"],
                        default=None,
                        help="Explicit benchmark geometry override.")
    parser.add_argument("--action-mode",
                        choices=["auto", "goal_relative_offset", "absolute_heading"],
                        default=None,
                        help="Explicit action encoding override.")
    parser.add_argument("--speed-ratio", type=float, default=None,
                        help="Target AUV max-speed / reference-flow-speed ratio.")
    parser.add_argument("--target-speed", type=float, default=None,
                        help="Explicit target AUV max speed in m/s.")
    parser.add_argument("--total-steps", type=int, default=50_000,
                        help="Total environment interaction steps.")
    parser.add_argument("--random-steps", type=int, default=2_000,
                        help="Initial random exploration steps.")
    parser.add_argument("--update-after", type=int, default=2_000,
                        help="Start gradient updates after this many env steps.")
    parser.add_argument("--update-every", type=int, default=1,
                        help="Gradient update frequency in env steps.")
    parser.add_argument("--updates-per-step", type=int, default=1,
                        help="Number of SGD updates each update step.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Sequence batch size.")
    parser.add_argument("--burn-in", type=int, default=8,
                        help="Burn-in length for recurrent replay.")
    parser.add_argument("--train-seq-len", type=int, default=32,
                        help="Optimized sequence length after burn-in.")
    parser.add_argument("--replay-max-episodes", type=int, default=5_000,
                        help="Maximum number of episodes stored in replay.")
    parser.add_argument("--eval-every", type=int, default=5_000,
                        help="Run deterministic evaluation every N env steps.")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of deterministic evaluation episodes.")
    parser.add_argument("--log-every-episodes", type=int, default=5,
                        help="Print training progress every N completed episodes.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device string.")
    parser.add_argument("--save-dir", type=str, default="checkpoints/gru_residual_sac",
                        help="Directory for checkpoints and config dump.")
    parser.add_argument("--residual-l2-weight", type=float, default=1e-3,
                        help="L2 penalty on residual actions.")
    parser.add_argument("--bc-weight", type=float, default=0.0,
                        help="Initial behavior-cloning weight on replay actions.")
    parser.add_argument("--bc-decay-steps", type=int, default=50_000,
                        help="Linear decay horizon for behavior-cloning weight.")
    parser.add_argument("--offline-policy",
                        choices=list(OFFLINE_POLICY_REGISTRY.keys()),
                        default="world_compensate",
                        help="Baseline policy used for offline warm-start data collection.")
    parser.add_argument("--offline-episodes", type=int, default=0,
                        help="Number of baseline episodes to pre-collect into replay.")
    parser.add_argument("--pretrain-updates", type=int, default=0,
                        help="Number of gradient steps to run on offline replay before online training.")
    parser.add_argument("--disable-residual-prior", action="store_true",
                        help="Train without the analytic residual prior.")
    parser.add_argument("--disable-safety-critic", action="store_true",
                        help="Disable cost critic and Lagrangian safety penalty.")
    parser.add_argument("--cost-limit", type=float, default=0.02,
                        help="Average cost target for Lagrangian safety constraint.")
    parser.add_argument("--checkpoint-every", type=int, default=5_000,
                        help="Persist model, replay, and trainer metadata every N steps.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a saved run directory or trainer_state.json path.")
    args = parser.parse_args()
    apply_resume_defaults(args, parser)
    train(args)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
