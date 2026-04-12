from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym

from auv_nav.reward import REWARD_OBJECTIVE_PRESETS
from auv_nav.sac import SACAgent, SACConfig
from auv_nav.replay import DualBufferSampler, TransitionReplay, TransitionReplayConfig
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
    extract_env_config_overrides,
    load_trainer_state,
    make_env_config_overrides,
    make_planar_env,
    make_reset_options,
    maybe_load_benchmark_manifest,
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


def detect_physical_cpu_count() -> int | None:
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                check=True,
                capture_output=True,
                text=True,
            )
            return max(1, int(result.stdout.strip()))
        if sys.platform.startswith("linux"):
            cpuinfo = Path("/proc/cpuinfo")
            if not cpuinfo.exists():
                return None
            physical_cores: set[tuple[str, str]] = set()
            current_physical_id = "0"
            current_core_id = None
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    if current_core_id is not None:
                        physical_cores.add((current_physical_id, current_core_id))
                    current_physical_id = "0"
                    current_core_id = None
                    continue
                if ":" not in line:
                    continue
                key, value = [part.strip() for part in line.split(":", 1)]
                if key == "physical id":
                    current_physical_id = value
                elif key == "core id":
                    current_core_id = value
            if current_core_id is not None:
                physical_cores.add((current_physical_id, current_core_id))
            if physical_cores:
                return len(physical_cores)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def recommended_num_envs(
    logical_cpus: int | None = None,
    physical_cpus: int | None = None,
) -> int:
    logical = max(1, int(logical_cpus or os.cpu_count() or 1))
    physical = physical_cpus if physical_cpus is not None else detect_physical_cpu_count()
    if physical is not None:
        return max(1, min(8, int(physical)))
    if logical <= 4:
        return logical
    return max(1, min(8, logical // 2))


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
    apply_default("eval_manifest", trainer_state.get("eval_manifest"))
    apply_default("probe_layout", trainer_state.get("probe_layout"))
    apply_default("objective", trainer_state.get("reward_objective"))
    apply_default("difficulty", reset_state.get("task_difficulty"))
    apply_default("task_geometry", reset_state.get("task_geometry"))
    apply_default("action_mode", reset_state.get("action_mode"))
    apply_default("target_speed", reset_state.get("target_auv_max_speed_mps"))
    if args.target_speed == parser.get_default("target_speed"):
        apply_default("speed_ratio", reset_state.get("target_speed_ratio"))
    env_state = extract_env_config_overrides(trainer_state)
    for reward_field in ("energy_cost_gain", "safety_cost_gain"):
        apply_default(reward_field, env_state.get(reward_field))

    apply_default("hidden_dim", agent_state.get("hidden_dim"))


def _build_extra_state(
    args: argparse.Namespace,
    train_cfg: TrainConfig,
    probe_layout: str,
    offline_replay: TransitionReplay | None,
    env_config_overrides: dict[str, Any],
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "algorithm": "rlpd" if offline_replay is not None else "sac",
        "history_length": train_cfg.history_length,
        "probe_layout": probe_layout,
        "reward_objective": env_config_overrides.get("reward_objective"),
        "env_config_overrides": env_config_overrides,
    }
    if args.eval_manifest is not None:
        state["eval_manifest"] = str(args.eval_manifest)
    if offline_replay is not None:
        state["offline_data_path"] = str(args.offline_data)
        state["offline_ratio"] = args.offline_ratio
        state["offline_transitions"] = len(offline_replay)
    return state


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
        num_envs=args.num_envs,
    )

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    if torch is not None:
        torch.manual_seed(train_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_cfg.seed)

    flow_path = args.flow or discover_flow_path()
    benchmark_manifest = maybe_load_benchmark_manifest(args.eval_manifest)
    env_config_overrides = make_env_config_overrides(args)

    probe_layout = args.probe_layout

    if train_cfg.num_envs > 1:
        def make_env_fn(seed_offset):
            def _thunk():
                env = make_planar_env(
                    flow_path,
                    history_length=train_cfg.history_length,
                    probe_layout=probe_layout,
                    env_config_overrides=env_config_overrides,
                )
                env.action_space.seed(train_cfg.seed + seed_offset)
                return env
            return _thunk
        env = gym.vector.AsyncVectorEnv([make_env_fn(i) for i in range(train_cfg.num_envs)])
        # AsyncVectorEnv spaces are Batched. We need the unbatched shape for Agent
        obs_dim = int(env.single_observation_space.shape[0])
        action_dim = int(env.single_action_space.shape[0])
    else:
        env = make_planar_env(
            flow_path,
            history_length=train_cfg.history_length,
            probe_layout=probe_layout,
            env_config_overrides=env_config_overrides,
        )
        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])

    if benchmark_manifest is not None:
        if (
            benchmark_manifest.probe_layout is not None
            and benchmark_manifest.probe_layout != probe_layout
        ):
            raise ValueError(
                f"Eval manifest probe_layout={benchmark_manifest.probe_layout} "
                f"!= training probe_layout={probe_layout}."
            )
        if (
            benchmark_manifest.history_length is not None
            and benchmark_manifest.history_length != train_cfg.history_length
        ):
            raise ValueError(
                f"Eval manifest history_length={benchmark_manifest.history_length} "
                f"!= training history_length={train_cfg.history_length}."
            )
    eval_flow_path = benchmark_manifest.flow_path if benchmark_manifest is not None else flow_path
    eval_env = make_planar_env(
        eval_flow_path,
        history_length=train_cfg.history_length,
        probe_layout=probe_layout,
        env_config_overrides=env_config_overrides,
    )
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

    # RLPD: load offline data and create dual-buffer sampler.
    if args.offline_data is not None:
        offline_replay = TransitionReplay.from_npz(args.offline_data)
        offline_metadata_path = args.offline_data.parent / "metadata.json"
        if offline_metadata_path.exists():
            with offline_metadata_path.open("r", encoding="utf-8") as fp:
                offline_metadata = json.load(fp)
            offline_objective = offline_metadata.get("objective")
            if offline_objective is not None and offline_objective != env_config_overrides.get(
                "reward_objective"
            ):
                raise ValueError(
                    f"Offline objective={offline_objective} != "
                    f"training objective={env_config_overrides.get('reward_objective')}."
                )
            offline_reward_config = offline_metadata.get("reward_config")
            if isinstance(offline_reward_config, dict):
                mismatched_reward_keys = []
                for key, value in env_config_overrides.items():
                    if key not in offline_reward_config:
                        continue
                    if offline_reward_config[key] != value:
                        mismatched_reward_keys.append(key)
                if mismatched_reward_keys:
                    raise ValueError(
                        "Offline reward_config does not match training reward_config for keys: "
                        + ", ".join(sorted(mismatched_reward_keys))
                    )
        if offline_replay.obs_dim != obs_dim:
            raise ValueError(
                f"Offline obs_dim={offline_replay.obs_dim} != env obs_dim={obs_dim}. "
                "Offline data must match the target probe layout and history length."
            )
        if offline_replay.action_dim != action_dim:
            raise ValueError(
                f"Offline action_dim={offline_replay.action_dim} != env action_dim={action_dim}."
            )
        if args.use_asymmetric_critic:
            if offline_replay.privileged_obs is None or offline_replay.next_privileged_obs is None:
                raise ValueError(
                    "Offline data is missing privileged_obs / next_privileged_obs, "
                    "which are required when --use-asymmetric-critic is enabled."
                )
        dual_sampler = DualBufferSampler(
            offline_replay, replay, args.offline_ratio,
        )
        print(
            f"[rlpd] loaded {len(offline_replay)} offline transitions "
            f"from {args.offline_data}, ratio={args.offline_ratio}"
        )
    else:
        offline_replay = None
        dual_sampler = None

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
    current_privileged_obs = (
        info.get("privileged_obs")
        if not is_vector_env
        else (info["privileged_obs"] if "privileged_obs" in info else None)
    )

    for env_step in range(start_step + 1, (train_cfg.total_env_steps // num_envs) + 1):
        global_step = env_step * num_envs
        if global_step <= train_cfg.random_steps:
            action = env.action_space.sample()
        else:
            action, policy_state = agent.act(obs, policy_state, deterministic=False)

        next_obs, reward, terminated, truncated, step_info = env.step(action)

        if is_vector_env:
            has_final_obs = "final_observation" in step_info
            has_final_info = "final_info" in step_info
            has_step_cost = "step_safety_cost" in step_info
            has_priv_obs = "privileged_obs" in step_info
            has_success = "success" in step_info
            has_elapsed = "elapsed_time_s" in step_info
            has_geom = "task_geometry" in step_info
            has_target = "target_auv_max_speed_mps" in step_info
            _false_sentinel = step_info.get("_final_observation", None)
            _final_info_mask = step_info.get("_final_info", None)

            for i in range(num_envs):
                done_i = terminated[i] or truncated[i]
                if has_final_obs and _false_sentinel is not None and _false_sentinel[i]:
                    real_next_obs = step_info["final_observation"][i]
                else:
                    real_next_obs = next_obs[i]

                step_cost = float(step_info["step_safety_cost"][i]) if has_step_cost else 0.0
                priv_obs = (
                    np.asarray(current_privileged_obs[i], dtype=np.float32)
                    if current_privileged_obs is not None
                    else None
                )
                next_priv_obs = (
                    np.asarray(step_info["privileged_obs"][i], dtype=np.float32)
                    if has_priv_obs
                    else None
                )
                if done_i and has_final_info and _final_info_mask is not None and _final_info_mask[i]:
                    f_info = step_info["final_info"][i]
                    if "privileged_obs" in f_info:
                        next_priv_obs = np.asarray(f_info["privileged_obs"], dtype=np.float32)

                replay.add(
                    obs=obs[i],
                    action=action[i],
                    reward=float(reward[i]),
                    cost=step_cost,
                    next_obs=real_next_obs,
                    done=bool(terminated[i]),
                    privileged_obs=priv_obs,
                    next_privileged_obs=next_priv_obs,
                )
                episode_return[i] += float(reward[i])
                episode_cost[i] += step_cost
                episode_length[i] += 1

                if done_i:
                    episode_idx += 1

                    success = bool(step_info["success"][i]) if has_success else False
                    elapsed_time_s = float(step_info["elapsed_time_s"][i]) if has_elapsed else 0.0
                    task_geom = step_info["task_geometry"][i] if has_geom else ""
                    target_auv = (
                        float(step_info["target_auv_max_speed_mps"][i]) if has_target else 0.0
                    )

                    if has_final_info and _final_info_mask is not None and _final_info_mask[i]:
                        f_info = step_info["final_info"][i]
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

            obs = next_obs
            info = step_info
            current_privileged_obs = step_info["privileged_obs"] if has_priv_obs else None
        else:
            done = terminated or truncated
            next_priv_obs = step_info.get("privileged_obs")
            replay.add(
                obs=obs,
                action=action,
                reward=float(reward),
                cost=float(step_info["step_safety_cost"]),
                next_obs=next_obs,
                done=terminated,
                privileged_obs=current_privileged_obs,
                next_privileged_obs=next_priv_obs,
            )

            episode_return[0] += float(reward)
            episode_cost[0] += float(step_info["step_safety_cost"])
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
                        "success": bool(step_info["success"]),
                        "elapsed_time_s": float(step_info["elapsed_time_s"]),
                        "episode_length": int(episode_length[0]),
                        "task_geometry": step_info["task_geometry"],
                        "target_auv_max_speed_mps": float(step_info["target_auv_max_speed_mps"]),
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
                        f"return={episode_return[0]:.2f} success={step_info['success']} "
                        f"time={step_info['elapsed_time_s']:.1f}s "
                        f"geometry={step_info['task_geometry']} "
                        f"history={train_cfg.history_length}"
                        f"{update_msg}"
                    )

                obs, info = env.reset(seed=train_cfg.seed + episode_idx, options=reset_options)
                current_privileged_obs = info.get("privileged_obs")
                episode_return[0] = 0.0
                episode_cost[0] = 0.0
                episode_length[0] = 0
            else:
                obs = next_obs
                info = step_info
                current_privileged_obs = next_priv_obs

        if (
            global_step >= train_cfg.update_after
            and replay.ready(train_cfg.batch_size)
            and global_step % train_cfg.update_every == 0
        ):
            for _ in range(train_cfg.updates_per_step):
                if dual_sampler is not None and dual_sampler.ready(train_cfg.batch_size):
                    batch = dual_sampler.sample_batch(train_cfg.batch_size, agent.device)
                else:
                    batch = replay.sample_batch(train_cfg.batch_size, agent.device)
                last_update = agent.update(batch)

        if global_step % train_cfg.eval_every_steps == 0:
            metrics = evaluate_agent(
                env=eval_env,
                agent=agent,
                reset_options=reset_options,
                seed=train_cfg.seed + 10_000 + global_step,
                num_episodes=train_cfg.eval_episodes,
                benchmark_manifest=benchmark_manifest,
            )
            print(
                f"[eval] step={global_step} "
                f"objective={metrics['reward_objective']} "
                f"return={metrics['eval_return']:.2f} "
                f"safety={metrics['eval_safety_cost']:.3f} "
                f"success={metrics['eval_success_rate']:.2%} "
                f"time={metrics['eval_time_s']:.1f}s "
                f"energy={metrics['eval_energy']:.1f} "
                f"path={metrics['eval_path_length_m']:.1f} "
                f"prog={metrics['eval_progress_ratio']:.2f}"
            )
            append_csv(
                eval_log_path,
                {
                    "env_step": global_step,
                    "reward_objective": metrics["reward_objective"],
                    "eval_return": metrics["eval_return"],
                    "eval_cost": metrics["eval_cost"],
                    "eval_safety_cost": metrics["eval_safety_cost"],
                    "eval_success_rate": metrics["eval_success_rate"],
                    "eval_time_s": metrics["eval_time_s"],
                    "eval_energy": metrics["eval_energy"],
                    "eval_path_length_m": metrics["eval_path_length_m"],
                    "eval_progress_ratio": metrics["eval_progress_ratio"],
                    "eval_path_efficiency": metrics["eval_path_efficiency"],
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
                extra_state=_build_extra_state(
                    args,
                    train_cfg,
                    probe_layout,
                    offline_replay,
                    env_config_overrides,
                ),
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
                extra_state=_build_extra_state(
                    args,
                    train_cfg,
                    probe_layout,
                    offline_replay,
                    env_config_overrides,
                ),
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
        extra_state=_build_extra_state(
            args,
            train_cfg,
            probe_layout,
            offline_replay,
            env_config_overrides,
        ),
    )
    final_metrics = evaluate_agent(
        env=eval_env,
        agent=agent,
        reset_options=reset_options,
        seed=train_cfg.seed + 20_000,
        num_episodes=train_cfg.eval_episodes,
        benchmark_manifest=benchmark_manifest,
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
        fp.write("EnvConfigOverrides\n")
        for key, value in env_config_overrides.items():
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
    parser.add_argument(
        "--objective",
        choices=sorted(REWARD_OBJECTIVE_PRESETS.keys()),
        default="arrival_v1",
        help="Reward objective preset. Use efficiency_v1 for efficiency-aware training.",
    )
    parser.add_argument("--energy-cost-gain", type=float, default=None)
    parser.add_argument("--safety-cost-gain", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--random-steps", type=int, default=2_000)
    parser.add_argument("--update-after", type=int, default=2_000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--eval-manifest", type=Path, default=None,
        help="Optional fixed benchmark manifest used for periodic/final evaluation.",
    )
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
        "--num-envs", type=int, default=recommended_num_envs(),
        help=(
            "Number of parallel environments. Defaults to a conservative auto-tuned value "
            "based on available CPU cores. Use 1 for single-env debugging."
        ),
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=1,
        help="Number of recent observations to stack for feedforward history baselines.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument(
        "--offline-data", type=Path, default=None,
        help="Path to offline transitions.npz for RLPD mode.",
    )
    parser.add_argument(
        "--offline-ratio", type=float, default=0.5,
        help="Fraction of each batch drawn from offline buffer (default: 0.5).",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--probe-layout",
        choices=["s0", "s1", "s2"],
        default="s0",
        help=(
            "Flow sensing layout: "
            "s0=centre only (DVL, obs+2), "
            "s1=short-range forward ADCP 2-probe (obs+4), "
            "s2=forward ADCP 4-probe (obs+8)."
        ),
    )
    args = parser.parse_args()
    apply_resume_defaults(args, parser)
    train(args)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
