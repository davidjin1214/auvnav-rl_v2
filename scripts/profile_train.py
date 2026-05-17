"""Profile the inner train_sac loop into per-stage latencies.

The goal is to expose which stage actually dominates wallclock — env.step,
agent.act, replay.add, replay.sample_batch, or agent.update — before deciding
whether to invest in a GPU-resident environment. The loop here mirrors the
one in ``scripts/train_sac.py`` (random_steps → update_after → updates_per_step
inner loop, AsyncVectorEnv batched obs, privileged_obs threading, auto-reset
final_observation handling) so the numbers are apples-to-apples with real
training.

Run as a module from the repo root::

    # CPU baseline, current default configuration (6 envs, async, UTD=4).
    python -m scripts.profile_train --num-envs 6 --device cpu --vector-mode async

    # Isolate AsyncVectorEnv IPC by running the same N envs synchronously.
    python -m scripts.profile_train --num-envs 6 --vector-mode sync

    # On Colab L4 — see how much wallclock the SAC update end actually costs.
    python -m scripts.profile_train --device cuda --num-envs 6 --updates-per-step 4

    # Sweep num_envs to see how env.step / IPC scale.
    for n in 1 2 4 6 8 10; do
        python -m scripts.profile_train --num-envs $n --vector-mode async \
            --output-json experiments/profile/run_n${n}.json
    done
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from auv_nav.networks import require_torch
from auv_nav.replay import TransitionReplay, TransitionReplayConfig
from auv_nav.reward import ARRIVAL_V2_OBJECTIVES
from auv_nav.sac import SACAgent, SACConfig

from .train_utils import (
    discover_flow_path,
    make_env_config_overrides,
    make_planar_env,
    make_reset_options,
)


def _now() -> float:
    return time.perf_counter()


def _replay_done(reward_objective: str | None, *, terminated: bool, truncated: bool) -> bool:
    if reward_objective in ARRIVAL_V2_OBJECTIVES:
        return bool(terminated or truncated)
    return bool(terminated)


@dataclass
class Bucket:
    """Latency samples for a single profiled stage, recorded in milliseconds."""

    name: str
    samples_ms: list[float] = field(default_factory=list)

    def add(self, dt_s: float) -> None:
        self.samples_ms.append(dt_s * 1000.0)

    def summary(self, wallclock_total_s: float) -> dict[str, Any]:
        if not self.samples_ms:
            return {"name": self.name, "count": 0}
        arr = np.asarray(self.samples_ms, dtype=np.float64)
        total_ms = float(arr.sum())
        return {
            "name": self.name,
            "count": int(arr.size),
            "total_ms": total_ms,
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
            "share_of_wallclock_pct": 100.0 * total_ms / (wallclock_total_s * 1000.0)
            if wallclock_total_s > 0.0
            else 0.0,
        }


class CudaSync:
    """Force CUDA work to drain before reading perf_counter; no-op on CPU."""

    def __init__(self, device: torch.device) -> None:
        self.enabled = device.type == "cuda"
        self.device = device

    def sync(self) -> None:
        if self.enabled:
            torch.cuda.synchronize(self.device)


@contextmanager
def time_into(bucket: Bucket, cuda: CudaSync):
    cuda.sync()
    t0 = _now()
    try:
        yield
    finally:
        cuda.sync()
        bucket.add(_now() - t0)


def _build_overrides_args(args: argparse.Namespace) -> argparse.Namespace:
    """Project the profile CLI namespace down to the fields read by train_utils helpers."""
    ns = argparse.Namespace()
    for key in (
        "task_geometry",
        "target_speed",
        "objective",
        "action_mode",
        "difficulty",
        "speed_ratio",
    ):
        setattr(ns, key, getattr(args, key, None))
    return ns


def _build_env(args: argparse.Namespace, env_config_overrides: dict[str, Any]):
    flow_path = args.flow or discover_flow_path()

    def make_env_fn(seed_offset: int):
        def _thunk():
            env = make_planar_env(
                flow_path,
                history_length=args.history_length,
                probe_layout=args.probe_layout,
                env_config_overrides=env_config_overrides,
            )
            env.action_space.seed(args.seed + seed_offset)
            return env
        return _thunk

    if args.num_envs > 1:
        thunks = [make_env_fn(i) for i in range(args.num_envs)]
        if args.vector_mode == "async":
            env = gym.vector.AsyncVectorEnv(thunks)
        else:
            env = gym.vector.SyncVectorEnv(thunks)
        obs_dim = int(env.single_observation_space.shape[0])
        action_dim = int(env.single_action_space.shape[0])
    else:
        env = make_planar_env(
            flow_path,
            history_length=args.history_length,
            probe_layout=args.probe_layout,
            env_config_overrides=env_config_overrides,
        )
        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
    return env, obs_dim, action_dim


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    require_torch()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    overrides_args = _build_overrides_args(args)
    env_config_overrides = make_env_config_overrides(overrides_args)
    reset_options = make_reset_options(overrides_args)
    reward_objective = env_config_overrides.get("reward_objective")

    is_vector = args.num_envs > 1
    num_envs = args.num_envs if is_vector else 1
    env, obs_dim, action_dim = _build_env(args, env_config_overrides)

    resolved_device = args.device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        print("[profile] CUDA requested but unavailable; falling back to CPU.")
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    cuda = CudaSync(device)

    agent_cfg = SACConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        updates_per_step=args.updates_per_step,
        use_layernorm=args.use_layernorm,
        dropout_rate=args.dropout_rate,
        privileged_obs_dim=2 if args.use_asymmetric_critic else 0,
    )
    agent = SACAgent(config=agent_cfg, device=str(device))
    replay = TransitionReplay(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=TransitionReplayConfig(
            capacity=args.replay_capacity,
            privileged_obs_dim=agent_cfg.privileged_obs_dim,
        ),
    )

    obs, info = env.reset(seed=args.seed, options=reset_options)
    current_priv = info.get("privileged_obs") if isinstance(info, dict) else None

    buckets = {
        name: Bucket(name)
        for name in ("agent_act", "env_step", "replay_add", "replay_sample", "sac_update")
    }

    transitions = 0
    updates = 0
    measure_start_transitions = 0
    measure_start_updates = 0
    measure_wallclock_t0: float | None = None
    warmup_done = False
    policy_state = None
    target_transitions = args.warmup_steps + args.measure_steps

    print(
        f"[profile] device={device}  num_envs={args.num_envs} ({args.vector_mode})  "
        f"obs_dim={obs_dim}  action_dim={action_dim}  priv_dim={agent_cfg.privileged_obs_dim}  "
        f"target={target_transitions} transitions"
    )

    while transitions < target_transitions:
        if not warmup_done and transitions >= args.warmup_steps:
            for b in buckets.values():
                b.samples_ms.clear()
            measure_start_transitions = transitions
            measure_start_updates = updates
            measure_wallclock_t0 = _now()
            warmup_done = True
            print(f"[profile] warmup complete at {transitions} transitions; measuring...")

        if transitions < args.random_steps:
            action = env.action_space.sample()
        else:
            with time_into(buckets["agent_act"], cuda):
                action, policy_state = agent.act(obs, policy_state, deterministic=False)

        with time_into(buckets["env_step"], cuda):
            next_obs, reward, terminated, truncated, step_info = env.step(action)

        with time_into(buckets["replay_add"], cuda):
            if is_vector:
                has_final_obs = "final_observation" in step_info
                final_mask = step_info.get("_final_observation", None)
                has_step_cost = "step_safety_cost" in step_info
                has_priv = "privileged_obs" in step_info
                for i in range(num_envs):
                    if has_final_obs and final_mask is not None and final_mask[i]:
                        real_next = step_info["final_observation"][i]
                    else:
                        real_next = next_obs[i]
                    step_cost = float(step_info["step_safety_cost"][i]) if has_step_cost else 0.0
                    priv_i = (
                        np.asarray(current_priv[i], dtype=np.float32)
                        if current_priv is not None
                        else None
                    )
                    next_priv_i = (
                        np.asarray(step_info["privileged_obs"][i], dtype=np.float32)
                        if has_priv
                        else None
                    )
                    replay.add(
                        obs=obs[i],
                        action=action[i],
                        reward=float(reward[i]),
                        cost=step_cost,
                        next_obs=real_next,
                        done=_replay_done(
                            reward_objective,
                            terminated=bool(terminated[i]),
                            truncated=bool(truncated[i]),
                        ),
                        privileged_obs=priv_i,
                        next_privileged_obs=next_priv_i,
                    )
            else:
                next_priv = step_info.get("privileged_obs")
                replay.add(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    cost=float(step_info.get("step_safety_cost", 0.0)),
                    next_obs=next_obs,
                    done=_replay_done(
                        reward_objective,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    ),
                    privileged_obs=current_priv,
                    next_privileged_obs=next_priv,
                )

        transitions += num_envs

        if transitions >= args.update_after and replay.ready(args.batch_size):
            for _ in range(args.updates_per_step):
                with time_into(buckets["replay_sample"], cuda):
                    batch = replay.sample_batch(args.batch_size, device)
                with time_into(buckets["sac_update"], cuda):
                    _ = agent.update(batch)
                updates += 1

        if is_vector:
            obs = next_obs
            current_priv = (
                step_info["privileged_obs"] if "privileged_obs" in step_info else None
            )
        else:
            done = bool(terminated) or bool(truncated)
            if done:
                obs, info = env.reset(seed=args.seed + transitions, options=reset_options)
                current_priv = info.get("privileged_obs") if isinstance(info, dict) else None
            else:
                obs = next_obs
                current_priv = step_info.get("privileged_obs")

    if measure_wallclock_t0 is None:
        raise RuntimeError(
            "Measurement window never started. Reduce --warmup-steps or "
            "increase --measure-steps."
        )
    cuda.sync()
    wallclock_s = _now() - measure_wallclock_t0
    env.close()

    measure_transitions = transitions - measure_start_transitions
    measure_updates = updates - measure_start_updates

    summary: dict[str, Any] = {
        "config": {
            "num_envs": args.num_envs,
            "vector_mode": args.vector_mode if is_vector else "single",
            "device": str(device),
            "batch_size": args.batch_size,
            "updates_per_step": args.updates_per_step,
            "use_asymmetric_critic": args.use_asymmetric_critic,
            "use_layernorm": args.use_layernorm,
            "dropout_rate": args.dropout_rate,
            "probe_layout": args.probe_layout,
            "history_length": args.history_length,
            "task_geometry": args.task_geometry,
            "target_speed": args.target_speed,
            "objective": args.objective,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "privileged_obs_dim": agent_cfg.privileged_obs_dim,
        },
        "throughput": {
            "measure_wallclock_s": wallclock_s,
            "measure_transitions": measure_transitions,
            "measure_updates": measure_updates,
            "transitions_per_s": measure_transitions / wallclock_s if wallclock_s > 0 else 0.0,
            "env_step_calls_per_s": (measure_transitions / num_envs) / wallclock_s
            if wallclock_s > 0
            else 0.0,
            "updates_per_s": measure_updates / wallclock_s if wallclock_s > 0 else 0.0,
        },
        "buckets": {name: b.summary(wallclock_s) for name, b in buckets.items()},
    }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    cfg = summary["config"]
    thr = summary["throughput"]
    print("=" * 88)
    print(
        f"  device={cfg['device']}  num_envs={cfg['num_envs']} ({cfg['vector_mode']})  "
        f"UTD={cfg['updates_per_step']}  batch={cfg['batch_size']}  "
        f"asym={cfg['use_asymmetric_critic']}  ln={cfg['use_layernorm']}"
    )
    print(
        f"  probe={cfg['probe_layout']}  history={cfg['history_length']}  "
        f"geom={cfg['task_geometry']}  tgt={cfg['target_speed']}  obj={cfg['objective']}"
    )
    print("-" * 88)
    print(
        f"  measured {thr['measure_transitions']} transitions in "
        f"{thr['measure_wallclock_s']:.2f}s  ->  "
        f"{thr['transitions_per_s']:.1f} transitions/s   "
        f"{thr['env_step_calls_per_s']:.1f} env.step/s   "
        f"{thr['updates_per_s']:.1f} updates/s"
    )
    print("-" * 88)
    header = (
        f"  {'bucket':<16}{'count':>8}{'total_ms':>12}{'mean_ms':>10}"
        f"{'p50_ms':>10}{'p95_ms':>10}{'% wallclock':>14}"
    )
    print(header)
    for name in ("agent_act", "env_step", "replay_add", "replay_sample", "sac_update"):
        s = summary["buckets"][name]
        if s.get("count", 0) == 0:
            print(f"  {name:<16}{'-':>8}{'-':>12}{'-':>10}{'-':>10}{'-':>10}{'-':>14}")
            continue
        print(
            f"  {name:<16}{s['count']:>8}{s['total_ms']:>12.1f}"
            f"{s['mean_ms']:>10.3f}{s['p50_ms']:>10.3f}{s['p95_ms']:>10.3f}"
            f"{s.get('share_of_wallclock_pct', 0.0):>13.1f}%"
        )
    print("=" * 88)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Profile the inner train_sac loop into per-stage latencies."
    )
    p.add_argument("--flow", type=Path, default=None)
    p.add_argument("--num-envs", type=int, default=6)
    p.add_argument("--vector-mode", choices=("async", "sync"), default="async")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=1500,
        help="Transitions used to warm up before measurement (default 1500).",
    )
    p.add_argument(
        "--measure-steps",
        type=int,
        default=3000,
        help="Transitions to measure after warmup (default 3000).",
    )
    p.add_argument("--random-steps", type=int, default=500)
    p.add_argument("--update-after", type=int, default=1000)
    p.add_argument("--updates-per-step", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--replay-capacity", type=int, default=200_000)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--use-asymmetric-critic", action="store_true")
    p.add_argument("--use-layernorm", action="store_true")
    p.add_argument("--dropout-rate", type=float, default=0.0)
    p.add_argument("--probe-layout", default="s0")
    p.add_argument("--history-length", type=int, default=4)
    p.add_argument("--task-geometry", default="upstream")
    p.add_argument("--target-speed", type=float, default=1.5)
    p.add_argument("--objective", default="efficiency_v2")
    p.add_argument("--action-mode", default=None)
    p.add_argument("--difficulty", default=None)
    p.add_argument("--speed-ratio", type=float, default=None)
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the structured summary JSON.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary = run_profile(args)
    _print_summary(summary)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"[profile] wrote {args.output_json}")


if __name__ == "__main__":
    main()
