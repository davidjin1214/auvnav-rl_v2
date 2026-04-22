from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from auv_nav.offline_registry import (
    default_offline_save_dir,
    make_agent,
    make_agent_config,
    normalize_offline_algo,
)
from auv_nav.replay import TransitionReplay
from auv_nav.reward import REWARD_OBJECTIVE_PRESETS
from auv_nav.networks import require_torch
from auv_nav.td3bc import ObservationNormalizer

try:
    import torch
except ImportError:
    torch = None

from .train_utils import (
    append_csv,
    append_jsonl,
    capture_rng_state,
    default_device,
    discover_flow_path,
    evaluate_agent,
    evaluate_offline_policy_parallel,
    extract_env_config_overrides,
    make_env_config_overrides,
    make_planar_env,
    make_reset_options,
    maybe_load_benchmark_manifest,
)


@dataclass(slots=True)
class OfflineTrainConfig:
    total_steps: int = 200_000
    seed: int = 42
    batch_size: int = 256
    eval_every_steps: int = 10_000
    eval_episodes: int = 30
    log_every_steps: int = 1_000
    save_dir: str = "checkpoints/offline/td3bc"
    device: str = "cuda:0"
    checkpoint_every_steps: int = 10_000
    history_length: int = 1
    tensor_replay: bool = True
    skip_final_eval: bool = False
    eval_workers: int = 1
    eval_worker_device: str = "cpu"
    sampling_mode: str = "uniform"
    num_epochs: int | None = None
    drop_last_batch: bool = False


def _resolve_num_hidden_layers(args: argparse.Namespace, algo: str) -> int:
    if args.num_hidden_layers is not None:
        return int(args.num_hidden_layers)
    return 3 if algo == "rebrac" else 2


def _resolve_rebrac_actor_layernorm(args: argparse.Namespace) -> bool:
    if args.actor_layernorm is not None:
        return bool(args.actor_layernorm)
    return bool(args.use_layernorm)


def _resolve_rebrac_critic_layernorm(args: argparse.Namespace) -> bool:
    if args.critic_layernorm is not None:
        return bool(args.critic_layernorm)
    if args.use_layernorm:
        return True
    return True


def _build_agent_config(
    algo: str,
    args: argparse.Namespace,
    *,
    obs_dim: int,
    action_dim: int,
    batch_size: int,
    privileged_obs_dim: int,
) -> Any:
    common = dict(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=_resolve_num_hidden_layers(args, algo),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        batch_size=batch_size,
        grad_clip_norm=args.grad_clip_norm,
        dropout_rate=args.dropout_rate,
        privileged_obs_dim=privileged_obs_dim,
        privileged_actor_update_mode=args.privileged_actor_update_mode,
        normalizer_eps=args.normalizer_eps,
    )
    if algo == "td3bc":
        return make_agent_config(
            algo,
            **common,
            alpha=args.alpha,
            use_layernorm=bool(args.use_layernorm),
        )

    return make_agent_config(
        algo,
        **common,
        actor_bc_coef=args.actor_penalty_coef,
        critic_bc_coef=args.critic_penalty_coef,
        actor_use_layernorm=_resolve_rebrac_actor_layernorm(args),
        critic_use_layernorm=_resolve_rebrac_critic_layernorm(args),
        normalize_q=not args.disable_q_normalization,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _load_offline_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data if isinstance(data, dict) else {}


def _validate_offline_protocol(
    offline_replay: TransitionReplay,
    offline_metadata: dict[str, Any],
    *,
    obs_dim: int,
    action_dim: int,
    probe_layout: str,
    history_length: int,
    env_config_overrides: dict[str, Any],
    use_asymmetric_critic: bool,
) -> None:
    if offline_replay.obs_dim != obs_dim:
        raise ValueError(
            f"Offline obs_dim={offline_replay.obs_dim} != env obs_dim={obs_dim}. "
            "Offline data must match probe layout and history length."
        )
    if offline_replay.action_dim != action_dim:
        raise ValueError(
            f"Offline action_dim={offline_replay.action_dim} != env action_dim={action_dim}."
        )
    metadata_probe_layout = offline_metadata.get("probe_layout")
    if metadata_probe_layout is not None and str(metadata_probe_layout) != probe_layout:
        raise ValueError(
            f"Offline data probe_layout={metadata_probe_layout} != requested {probe_layout}."
        )
    metadata_history_length = offline_metadata.get("history_length")
    if (
        metadata_history_length is not None
        and int(metadata_history_length) != int(history_length)
    ):
        raise ValueError(
            "Offline data history_length="
            f"{metadata_history_length} != requested {history_length}."
        )
    offline_objective = offline_metadata.get("objective")
    train_objective = env_config_overrides.get("reward_objective")
    if offline_objective is not None and offline_objective != train_objective:
        raise ValueError(
            f"Offline objective={offline_objective} != training objective={train_objective}."
        )
    offline_reward_config = offline_metadata.get("reward_config")
    if isinstance(offline_reward_config, dict):
        mismatched = []
        for key, value in env_config_overrides.items():
            if key not in offline_reward_config:
                continue
            if offline_reward_config[key] != value:
                mismatched.append(key)
        if mismatched:
            raise ValueError(
                "Offline reward_config does not match training reward_config for keys: "
                + ", ".join(sorted(mismatched))
            )
    if use_asymmetric_critic:
        if offline_replay.privileged_obs is None or offline_replay.next_privileged_obs is None:
            raise ValueError(
                "Offline data is missing privileged_obs / next_privileged_obs, "
                "which are required when --use-asymmetric-critic is enabled."
            )


def _resolve_eval_flow_path(
    args: argparse.Namespace,
    offline_metadata: dict[str, Any],
    benchmark_flow_path: str | None,
) -> str:
    if benchmark_flow_path is not None:
        return benchmark_flow_path
    if args.flow is not None:
        return str(args.flow)
    metadata_flow_path = offline_metadata.get("flow_path")
    if metadata_flow_path is not None:
        return str(metadata_flow_path)
    return str(discover_flow_path())


def _metric_score(metrics: dict[str, Any]) -> tuple[float, float]:
    return (
        float(metrics["eval_success_rate"]),
        float(metrics["eval_return"]),
    )


def _steps_per_epoch(
    num_transitions: int,
    batch_size: int,
    *,
    drop_last: bool,
) -> int:
    if num_transitions <= 0:
        raise ValueError("Offline dataset is empty.")
    batch_size = max(1, int(batch_size))
    if drop_last:
        steps = num_transitions // batch_size
        if steps <= 0:
            raise ValueError(
                "drop_last_batch=True requires num_transitions >= batch_size."
            )
        return steps
    return max(1, math.ceil(num_transitions / batch_size))


def _training_batches(
    offline_replay: TransitionReplay,
    train_cfg: OfflineTrainConfig,
    device: "torch.device | str",
) -> Iterator[tuple[int | None, int | None, dict[str, "torch.Tensor"]]]:
    if train_cfg.sampling_mode == "uniform":
        for _ in range(train_cfg.total_steps):
            yield None, None, offline_replay.sample_batch(train_cfg.batch_size, device)
        return

    if train_cfg.sampling_mode != "shuffle_no_replacement":
        raise ValueError(f"Unsupported sampling mode: {train_cfg.sampling_mode!r}")
    if train_cfg.num_epochs is None or train_cfg.num_epochs <= 0:
        raise ValueError(
            "--num-epochs must be set to a positive integer when "
            "--sampling-mode=shuffle_no_replacement."
        )

    for epoch_idx in range(1, train_cfg.num_epochs + 1):
        for epoch_step, batch in enumerate(
            offline_replay.iter_batches(
                train_cfg.batch_size,
                device,
                shuffle=True,
                drop_last=train_cfg.drop_last_batch,
            ),
            start=1,
        ):
            yield epoch_idx, epoch_step, batch


def _evaluate_current_policy(
    *,
    eval_env: Any,
    agent: Any,
    eval_flow_path: str,
    train_cfg: OfflineTrainConfig,
    reset_options: dict[str, Any],
    probe_layout: str,
    env_config_overrides: dict[str, Any],
    seed: int,
    num_episodes: int,
    benchmark_manifest: Any,
) -> dict[str, Any]:
    if train_cfg.eval_workers <= 1:
        return evaluate_agent(
            env=eval_env,
            agent=agent,
            reset_options=reset_options,
            seed=seed,
            num_episodes=num_episodes,
            benchmark_manifest=benchmark_manifest,
        )
    return evaluate_offline_policy_parallel(
        policy_payload=agent.export_policy_payload(),
        flow_path=eval_flow_path,
        history_length=train_cfg.history_length,
        probe_layout=probe_layout,
        env_config_overrides=env_config_overrides,
        reset_options=reset_options,
        seed=seed,
        num_episodes=num_episodes,
        benchmark_manifest=benchmark_manifest,
        num_workers=train_cfg.eval_workers,
        worker_device=train_cfg.eval_worker_device,
    )


def _save_offline_training_state(
    save_dir: Path,
    agent: Any,
    *,
    algo: str,
    train_cfg: OfflineTrainConfig,
    agent_cfg: Any,
    reset_options: dict[str, Any],
    env_config_overrides: dict[str, Any],
    flow_path: str,
    manifest_path: str | None,
    offline_data_path: str,
    offline_metadata: dict[str, Any],
    step: int,
    probe_layout: str,
    protocol_label: str,
    best_metrics: dict[str, Any] | None,
    best_step: int | None,
    latest_agent_name: str = "agent_latest.pt",
    best_agent_name: str | None = None,
    final_agent_name: str | None = None,
) -> None:
    rng_state_path = save_dir / "rng_state.pkl"
    agent_path = save_dir / latest_agent_name
    agent.save(str(agent_path))
    with rng_state_path.open("wb") as fp:
        import pickle

        pickle.dump(capture_rng_state(), fp)

    metadata = {
        "algo": algo,
        "algorithm": algo,
        "train_step": int(step),
        "train_config": asdict(train_cfg),
        "agent_config": asdict(agent_cfg),
        "reset_options": reset_options,
        "env_config_overrides": env_config_overrides,
        "reward_objective": env_config_overrides.get("reward_objective"),
        "flow_path": str(flow_path),
        "manifest": manifest_path,
        "offline_data_path": str(offline_data_path),
        "offline_metadata": offline_metadata,
        "probe_layout": probe_layout,
        "history_length": int(train_cfg.history_length),
        "protocol": protocol_label,
        "obs_normalizer": agent.obs_normalizer.json_dict(),
        "agent_path": best_agent_name or latest_agent_name,
        "latest_agent_path": latest_agent_name,
        "best_agent_path": best_agent_name,
        "final_agent_path": final_agent_name,
        "rng_state_path": rng_state_path.name,
        "best_eval_metrics": best_metrics,
        "best_eval_step": best_step,
    }
    with (save_dir / "trainer_state.json").open("w", encoding="utf-8") as fp:
        json.dump(_json_ready(metadata), fp, indent=2)


def train(args: argparse.Namespace) -> None:
    require_torch()
    algo = normalize_offline_algo(args.algo)
    train_cfg = OfflineTrainConfig(
        total_steps=args.total_steps,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_every_steps=args.eval_every,
        eval_episodes=args.eval_episodes,
        log_every_steps=args.log_every,
        save_dir=args.save_dir or default_offline_save_dir(algo),
        device=args.device,
        checkpoint_every_steps=args.checkpoint_every,
        history_length=args.history_length,
        tensor_replay=not args.disable_tensor_replay,
        skip_final_eval=args.skip_final_eval,
        eval_workers=max(1, int(args.eval_workers)),
        eval_worker_device=args.eval_worker_device,
        sampling_mode=args.sampling_mode,
        num_epochs=args.num_epochs,
        drop_last_batch=args.drop_last_batch,
    )

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    benchmark_manifest = maybe_load_benchmark_manifest(args.manifest)
    env_config_overrides = make_env_config_overrides(args)
    reset_options = make_reset_options(args)
    probe_layout = args.probe_layout

    if benchmark_manifest is not None:
        if (
            benchmark_manifest.probe_layout is not None
            and benchmark_manifest.probe_layout != probe_layout
        ):
            raise ValueError(
                f"Manifest probe_layout={benchmark_manifest.probe_layout} "
                f"!= requested probe_layout={probe_layout}."
            )
        if (
            benchmark_manifest.history_length is not None
            and benchmark_manifest.history_length != train_cfg.history_length
        ):
            raise ValueError(
                f"Manifest history_length={benchmark_manifest.history_length} "
                f"!= requested history_length={train_cfg.history_length}."
            )

    offline_data_path = Path(args.offline_data)
    offline_replay = TransitionReplay.from_npz(offline_data_path)
    offline_metadata = _load_offline_metadata(offline_data_path)
    if train_cfg.sampling_mode == "shuffle_no_replacement":
        steps_per_epoch = _steps_per_epoch(
            len(offline_replay),
            train_cfg.batch_size,
            drop_last=train_cfg.drop_last_batch,
        )
        if train_cfg.num_epochs is None or train_cfg.num_epochs <= 0:
            raise ValueError(
                "--num-epochs must be set to a positive integer when "
                "--sampling-mode=shuffle_no_replacement."
            )
        train_cfg.total_steps = steps_per_epoch * train_cfg.num_epochs
    eval_flow_path = _resolve_eval_flow_path(
        args,
        offline_metadata,
        benchmark_manifest.flow_path if benchmark_manifest is not None else None,
    )
    eval_env = make_planar_env(
        eval_flow_path,
        history_length=train_cfg.history_length,
        probe_layout=probe_layout,
        env_config_overrides=env_config_overrides,
    )
    obs_dim = int(eval_env.observation_space.shape[0])
    action_dim = int(eval_env.action_space.shape[0])

    _validate_offline_protocol(
        offline_replay,
        offline_metadata,
        obs_dim=obs_dim,
        action_dim=action_dim,
        probe_layout=probe_layout,
        history_length=train_cfg.history_length,
        env_config_overrides=env_config_overrides,
        use_asymmetric_critic=args.use_asymmetric_critic,
    )

    obs_normalizer = ObservationNormalizer.from_observations(
        offline_replay.observations[: len(offline_replay)],
        eps=args.normalizer_eps,
        device=train_cfg.device,
    )
    if train_cfg.tensor_replay:
        offline_replay.enable_tensor_cache(train_cfg.device)
    agent_cfg = _build_agent_config(
        algo,
        args,
        obs_dim=obs_dim,
        action_dim=action_dim,
        batch_size=train_cfg.batch_size,
        privileged_obs_dim=(
            offline_replay.config.privileged_obs_dim if args.use_asymmetric_critic else 0
        ),
    )
    agent = make_agent(
        algo,
        agent_cfg,
        obs_normalizer=obs_normalizer,
        device=train_cfg.device,
    )

    save_dir = Path(train_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = save_dir / "train_log.jsonl"
    eval_log_path = save_dir / "eval_log.csv"
    protocol_label = "privileged-critic" if args.use_asymmetric_critic else "deployable"

    best_metrics: dict[str, Any] | None = None
    best_step: int | None = None
    best_score = (-np.inf, -np.inf)
    next_eval_step = train_cfg.eval_every_steps if train_cfg.eval_every_steps > 0 else None
    next_checkpoint_step = (
        train_cfg.checkpoint_every_steps if train_cfg.checkpoint_every_steps > 0 else None
    )

    print(
        f"[offline] algo={algo} transitions={len(offline_replay)} "
        f"obs_dim={offline_replay.obs_dim} "
        f"action_dim={offline_replay.action_dim} protocol={protocol_label} "
        f"tensor_replay={'on' if offline_replay.has_tensor_cache(agent.device) else 'off'} "
        f"eval_workers={train_cfg.eval_workers} "
        f"sampling={train_cfg.sampling_mode} total_steps={train_cfg.total_steps}"
    )

    for step, (epoch_idx, epoch_step, batch) in enumerate(
        _training_batches(offline_replay, train_cfg, agent.device),
        start=1,
    ):
        metrics = agent.update(batch)

        if step % train_cfg.log_every_steps == 0 or step == 1:
            log_row = {
                "train_step": step,
                "epoch": epoch_idx,
                "epoch_step": epoch_step,
                **metrics,
            }
            append_jsonl(train_log_path, log_row)
            train_prefix = f"[train] step={step}"
            if epoch_idx is not None and epoch_step is not None:
                train_prefix += f" epoch={epoch_idx} epoch_step={epoch_step}"
            critic_penalty_suffix = ""
            if "critic_penalty" in metrics:
                critic_penalty_suffix = f" critic_pen={metrics['critic_penalty']:.3f}"
            print(
                f"{train_prefix} q={metrics['mean_q']:.3f} "
                f"critic={metrics['critic_loss']:.3f} actor={metrics['actor_loss']:.3f} "
                f"bc={metrics['bc_loss']:.3f} lambda={metrics['lambda']:.3f}"
                f"{critic_penalty_suffix}"
            )

        if next_eval_step is not None and step >= next_eval_step:
            eval_metrics = _evaluate_current_policy(
                eval_env=eval_env,
                agent=agent,
                eval_flow_path=eval_flow_path,
                train_cfg=train_cfg,
                reset_options=reset_options,
                probe_layout=probe_layout,
                env_config_overrides=env_config_overrides,
                seed=train_cfg.seed + 10_000 + step,
                num_episodes=train_cfg.eval_episodes,
                benchmark_manifest=benchmark_manifest,
            )
            append_csv(
                eval_log_path,
                {
                    "train_step": step,
                    "reward_objective": eval_metrics["reward_objective"],
                    "eval_return": eval_metrics["eval_return"],
                    "eval_safety_cost": eval_metrics["eval_safety_cost"],
                    "eval_success_rate": eval_metrics["eval_success_rate"],
                    "eval_time_s": eval_metrics["eval_time_s"],
                    "eval_energy": eval_metrics["eval_energy"],
                    "eval_path_length_m": eval_metrics["eval_path_length_m"],
                    "eval_progress_ratio": eval_metrics["eval_progress_ratio"],
                    "eval_path_efficiency": eval_metrics["eval_path_efficiency"],
                },
            )
            print(
                f"[eval] step={step} success={eval_metrics['eval_success_rate']:.2%} "
                f"return={eval_metrics['eval_return']:.2f} "
                f"time={eval_metrics['eval_time_s']:.2f}s "
                f"energy={eval_metrics['eval_energy']:.2f} "
                f"path={eval_metrics['eval_path_length_m']:.2f} "
                f"prog={eval_metrics['eval_progress_ratio']:.3f}"
            )

            score = _metric_score(eval_metrics)
            if score > best_score:
                best_score = score
                best_metrics = eval_metrics
                best_step = step
                agent.save(str(save_dir / "agent_best.pt"))

            _save_offline_training_state(
                save_dir,
                agent,
                algo=algo,
                train_cfg=train_cfg,
                agent_cfg=agent_cfg,
                reset_options=reset_options,
                env_config_overrides=env_config_overrides,
                flow_path=eval_flow_path,
                manifest_path=str(args.manifest) if args.manifest is not None else None,
                offline_data_path=str(offline_data_path),
                offline_metadata=offline_metadata,
                step=step,
                probe_layout=probe_layout,
                protocol_label=protocol_label,
                best_metrics=best_metrics,
                best_step=best_step,
                best_agent_name="agent_best.pt" if best_metrics is not None else None,
            )
            next_eval_step += train_cfg.eval_every_steps

        if next_checkpoint_step is not None and step >= next_checkpoint_step:
            agent.save(str(save_dir / f"agent_step_{step:08d}.pt"))
            _save_offline_training_state(
                save_dir,
                agent,
                algo=algo,
                train_cfg=train_cfg,
                agent_cfg=agent_cfg,
                reset_options=reset_options,
                env_config_overrides=env_config_overrides,
                flow_path=eval_flow_path,
                manifest_path=str(args.manifest) if args.manifest is not None else None,
                offline_data_path=str(offline_data_path),
                offline_metadata=offline_metadata,
                step=step,
                probe_layout=probe_layout,
                protocol_label=protocol_label,
                best_metrics=best_metrics,
                best_step=best_step,
                best_agent_name="agent_best.pt" if best_metrics is not None else None,
            )
            next_checkpoint_step += train_cfg.checkpoint_every_steps

    agent.save(str(save_dir / "agent_final.pt"))
    if not train_cfg.skip_final_eval:
        final_metrics = _evaluate_current_policy(
            eval_env=eval_env,
            agent=agent,
            eval_flow_path=eval_flow_path,
            train_cfg=train_cfg,
            reset_options=reset_options,
            probe_layout=probe_layout,
            env_config_overrides=env_config_overrides,
            seed=train_cfg.seed + 20_000,
            num_episodes=train_cfg.eval_episodes,
            benchmark_manifest=benchmark_manifest,
        )
        with (save_dir / "final_eval.json").open("w", encoding="utf-8") as fp:
            json.dump(_json_ready(final_metrics), fp, indent=2)

        if _metric_score(final_metrics) > best_score:
            best_metrics = final_metrics
            best_step = train_cfg.total_steps
            agent.save(str(save_dir / "agent_best.pt"))

    _save_offline_training_state(
        save_dir,
        agent,
        algo=algo,
        train_cfg=train_cfg,
        agent_cfg=agent_cfg,
        reset_options=reset_options,
        env_config_overrides=env_config_overrides,
        flow_path=eval_flow_path,
        manifest_path=str(args.manifest) if args.manifest is not None else None,
        offline_data_path=str(offline_data_path),
        offline_metadata=offline_metadata,
        step=train_cfg.total_steps,
        probe_layout=probe_layout,
        protocol_label=protocol_label,
        best_metrics=best_metrics,
        best_step=best_step,
        best_agent_name="agent_best.pt" if best_metrics is not None else None,
        final_agent_name="agent_final.pt",
    )
    with (save_dir / "train_config.txt").open("w", encoding="utf-8") as fp:
        fp.write("OfflineTrainConfig\n")
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
    eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an offline RL agent on a fixed dataset.")
    parser.add_argument(
        "--algo",
        choices=["td3bc", "rebrac"],
        default="td3bc",
        help="Offline RL algorithm to train.",
    )
    parser.add_argument("--offline-data", type=Path, required=True)
    parser.add_argument("--flow", type=Path, default=None, help="Optional evaluation flow path.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional fixed benchmark manifest for periodic/final evaluation.")
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
    )
    parser.add_argument("--energy-cost-gain", type=float, default=None)
    parser.add_argument("--safety-cost-gain", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--sampling-mode",
        choices=["uniform", "shuffle_no_replacement"],
        default="uniform",
        help=(
            "Offline mini-batch sampling protocol. "
            "'uniform' matches standard replay sampling with replacement. "
            "'shuffle_no_replacement' iterates over the dataset once per epoch."
        ),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help=(
            "Number of offline epochs when --sampling-mode=shuffle_no_replacement. "
            "Ignored for uniform replay sampling."
        ),
    )
    parser.add_argument(
        "--drop-last-batch",
        action="store_true",
        default=False,
        help="Drop the final partial batch in shuffle_no_replacement mode.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10_000,
        help="Periodic evaluation interval in gradient steps. Set to 0 to disable.",
    )
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Number of worker processes for periodic/final evaluation inside training.",
    )
    parser.add_argument(
        "--eval-worker-device",
        type=str,
        default="cpu",
        help="Torch device used by in-training evaluation workers when --eval-workers > 1.",
    )
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Torch device. Defaults to cuda:0 when CUDA is available, otherwise cpu.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Checkpoint directory. Defaults to checkpoints/offline/<algo>.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
        help="Shared hidden-layer depth. Defaults to 2 for TD3BC and 3 for ReBRAC.",
    )
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument(
        "--actor-penalty-coef",
        type=float,
        default=1.0,
        help="ReBRAC actor behavior-cloning penalty coefficient.",
    )
    parser.add_argument(
        "--critic-penalty-coef",
        type=float,
        default=1.0,
        help="ReBRAC critic target behavior penalty coefficient.",
    )
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-freq", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--normalizer-eps", type=float, default=1e-3)
    parser.add_argument("--use-layernorm", action="store_true", default=False)
    parser.add_argument(
        "--actor-layernorm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override actor LayerNorm. ReBRAC defaults to off unless --use-layernorm is set.",
    )
    parser.add_argument(
        "--critic-layernorm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override critic LayerNorm. ReBRAC defaults to on.",
    )
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument(
        "--disable-q-normalization",
        action="store_true",
        default=False,
        help="Disable ReBRAC Q normalization in the actor loss.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Checkpoint interval in gradient steps. Set to 0 to disable periodic checkpoints.",
    )
    parser.add_argument(
        "--disable-tensor-replay",
        action="store_true",
        default=False,
        help="Keep offline data in numpy form and copy each batch on demand.",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        default=False,
        help="Skip the final evaluation pass. Useful when a separate evaluator job will run.",
    )
    parser.add_argument(
        "--probe-layout",
        choices=["s0", "s1", "s2"],
        default="s0",
    )
    parser.add_argument("--history-length", type=int, default=1)
    parser.add_argument(
        "--use-asymmetric-critic",
        action="store_true",
        default=False,
        help="Enable privileged critic using offline privileged_obs channels.",
    )
    parser.add_argument(
        "--privileged-actor-update-mode",
        choices=["zeros", "batch"],
        default="zeros",
        help=(
            "How actor loss queries a privileged critic. "
            "'zeros' zero-pads the privileged channels; 'batch' uses dataset privileged_obs."
        ),
    )
    args = parser.parse_args()
    train(args)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
