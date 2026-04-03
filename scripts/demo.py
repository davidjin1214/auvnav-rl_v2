"""
Run baseline rollouts on the planar wake-navigation environment.

Supports batch statistics, single-episode detail, and trajectory visualisation
— all from one entry point.

Usage:
    # Text-only: run all baselines, 5 episodes each
    python -m scripts.run_planar_env_example

    # 20-episode statistical comparison
    python -m scripts.run_planar_env_example --episodes 20

    # Single policy with trajectory plot
    python -m scripts.run_planar_env_example --policy goal --plot

    # Compare all policies visually and save figure
    python -m scripts.run_planar_env_example --policy all --plot --save trajectory.png

    # Specify flow file
    python -m scripts.run_planar_env_example --flow wake_data/wake_nav_roi.npy

    # Medium benchmark task with matched speed ratio
    python -m scripts.run_planar_env_example --difficulty medium --speed-ratio 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from auv_nav.policies import (
    CrossCurrentCompensationPolicy,
    GoalSeekPolicy,
    PrivilegedCorridorPolicy,
    StillWaterStraightLine,
    WorldFrameCurrentCompensationPolicy,
)
from auv_nav.envs import PlanarRemusEnv, PlanarRemusEnvConfig


# ── registry ──────────────────────────────────────────────────────────────

POLICIES: dict[str, type] = {
    "goal": GoalSeekPolicy,
    "compensate": CrossCurrentCompensationPolicy,
    "still_water": StillWaterStraightLine,
    "world_compensate": WorldFrameCurrentCompensationPolicy,
    "corridor": PrivilegedCorridorPolicy,
}


# ── helpers ───────────────────────────────────────────────────────────────

def discover_flow_path() -> Path:
    """Find the first wake .npy file under ./wake_data."""
    data_dir = Path("wake_data")
    candidates = sorted(data_dir.glob("wake_*_roi.npy"))
    if not candidates:
        raise FileNotFoundError(
            "No wake-field data found under ./wake_data/. "
            "Generate data with `python -m scripts.generate_wake_v7` first."
        )
    return candidates[0]


# ── episode runner ────────────────────────────────────────────────────────

def run_episode(
    env: PlanarRemusEnv,
    policy_name: str,
    seed: int | None = None,
    *,
    collect_trajectory: bool = False,
    reset_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one episode and return a summary dict.

    When *collect_trajectory* is True, extra per-step arrays (positions,
    headings, distances, rewards, speeds, currents) are included for
    plotting.
    """
    policy = POLICIES[policy_name]()
    obs, info = env.reset(seed=seed, options=reset_options)

    done = False
    total_reward = 0.0
    steps = 0

    # lightweight trajectory (always)
    trajectory = [info["position_xy_m"].copy()]

    # heavy per-step data (only when plotting)
    if collect_trajectory:
        headings = [info["psi_rad"]]
        distances = [info["distance_to_goal_m"]]
        rewards: list[float] = []
        times = [0.0]
        speeds = [float(np.hypot(info["body_velocity"][0], info["body_velocity"][1]))]
        currents = [info["equivalent_current_world"].copy()]

    while not done:
        action = policy.act(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        trajectory.append(info["position_xy_m"].copy())

        if collect_trajectory:
            headings.append(info["psi_rad"])
            distances.append(info["distance_to_goal_m"])
            rewards.append(reward)
            times.append(info["elapsed_time_s"])
            speeds.append(float(np.hypot(info["body_velocity"][0], info["body_velocity"][1])))
            currents.append(info["equivalent_current_world"].copy())

    result: dict[str, Any] = {
        "policy": policy_name,
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "elapsed_time_s": info["elapsed_time_s"],
        "distance_to_goal_m": info["distance_to_goal_m"],
        "initial_distance_m": info["initial_distance_m"],
        "success": info["success"],
        "reason": info["reason"],
        "start_xy": info["start_xy_m"],
        "goal_xy": info["goal_xy_m"],
        "final_xy": info["position_xy_m"],
        "cumulative_rpm": info["cumulative_rpm"],
        "task_geometry": info["task_geometry"],
        "action_mode": info["action_mode"],
        "reference_flow_speed_mps": info["reference_flow_speed_mps"],
        "reference_flow_heading_rad": info["reference_flow_heading_rad"],
        "start_flow_time_s": float(info["flow_time_s"]),
        "target_auv_max_speed_mps": info["target_auv_max_speed_mps"],
        "max_rpm_command": info["max_rpm_command"],
        "trajectory": np.array(trajectory, dtype=np.float32),
    }

    if collect_trajectory:
        result.update({
            "positions": np.array(trajectory, dtype=np.float32),
            "headings": np.array(headings),
            "distances": np.array(distances),
            "rewards": np.array(rewards),
            "times": np.array(times),
            "speeds": np.array(speeds),
            "currents_world": np.array(currents),
        })

    return result


# ── text output ───────────────────────────────────────────────────────────

def print_episode_summary(result: dict) -> None:
    print(f"  policy           : {result['policy']}")
    print(f"  task_geometry    : {result['task_geometry']}")
    print(f"  action_mode      : {result['action_mode']}")
    print(f"  success          : {result['success']}")
    print(f"  reason           : {result['reason']}")
    print(f"  steps            : {result['steps']}")
    print(f"  elapsed_time_s   : {result['elapsed_time_s']:.2f}")
    print(f"  total_reward     : {result['total_reward']:.2f}")
    print(f"  init_distance_m  : {result['initial_distance_m']:.1f}")
    print(f"  final_distance_m : {result['distance_to_goal_m']:.1f}")
    print(f"  start_xy         : {np.round(result['start_xy'], 1)}")
    print(f"  goal_xy          : {np.round(result['goal_xy'], 1)}")
    print(f"  final_xy         : {np.round(result['final_xy'], 1)}")
    print(f"  ref_flow_mps     : {result['reference_flow_speed_mps']:.2f}")
    print(f"  target_u_max_mps : {result['target_auv_max_speed_mps']:.2f}")
    print(f"  max_rpm_command  : {result['max_rpm_command']:.0f}")
    print(f"  cumulative_rpm   : {result['cumulative_rpm']:.0f}")


def print_batch_summary(results: list[dict], policy_name: str) -> None:
    successes = [r for r in results if r["success"]]
    n = len(results)
    n_succ = len(successes)
    times = [r["elapsed_time_s"] for r in successes] if successes else [0.0]
    rewards_list = [r["total_reward"] for r in results]

    print(f"\n{'='*60}")
    print(f"  {policy_name.upper()} — {n} episodes")
    print(f"{'='*60}")
    print(f"  task_geometry    : {results[0]['task_geometry']}")
    print(f"  action_mode      : {results[0]['action_mode']}")
    print(f"  ref_flow_mps     : {results[0]['reference_flow_speed_mps']:.2f}")
    print(f"  target_u_max_mps : {results[0]['target_auv_max_speed_mps']:.2f}")
    print(f"  max_rpm_command  : {results[0]['max_rpm_command']:.0f}")
    print(f"  success_rate     : {n_succ}/{n} ({100*n_succ/n:.1f}%)")
    if successes:
        print(f"  avg_time (succ)  : {np.mean(times):.2f} +/- {np.std(times):.2f} s")
        print(f"  min/max time     : {np.min(times):.2f} / {np.max(times):.2f} s")
    print(f"  avg_reward       : {np.mean(rewards_list):.2f} +/- {np.std(rewards_list):.2f}")
    reasons: dict[str, int] = {}
    for r in results:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print(f"  termination      : {reasons}")


# ── plotting (lazy import) ────────────────────────────────────────────────

def _plot_flow_snapshot(ax: Any, env: PlanarRemusEnv, t: float) -> None:
    """Contour + quiver of flow field at time *t*."""
    import matplotlib.pyplot as plt  # noqa: F811

    wf = env.wake_field
    skip_x = max(1, wf.flow.shape[1] // 40)
    skip_y = max(1, wf.flow.shape[2] // 20)

    x_coords = wf.x_coords[::skip_x]
    y_coords = wf.y_coords[::skip_y]
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")

    U = np.zeros_like(X)
    V = np.zeros_like(X)
    for i, xi in enumerate(x_coords):
        for j, yj in enumerate(y_coords):
            sample = env.flow_sampler.sample_world(float(xi), float(yj), t)
            U[i, j] = sample[0]
            V[i, j] = sample[1]

    speed = np.sqrt(U**2 + V**2)
    ax.contourf(X, Y, speed, levels=20, cmap="Blues", alpha=0.4)
    ax.quiver(X, Y, U, V, speed, cmap="Blues", scale=30, alpha=0.5, width=0.002)


def plot_episode(
    episode: dict, env: PlanarRemusEnv, save_path: str | None = None
) -> None:
    """Four-panel figure for a single episode."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1 — trajectory on flow field
    ax = axes[0, 0]
    _plot_flow_snapshot(ax, env, t=float(episode["start_flow_time_s"]))
    pos = episode["positions"]
    ax.plot(pos[:, 0], pos[:, 1], "k-", lw=1.5, label="trajectory")
    n_q = min(30, len(pos))
    step = max(1, len(pos) // n_q)
    idx = np.arange(0, len(pos), step)
    h = episode["headings"][idx]
    ax.quiver(pos[idx, 0], pos[idx, 1], np.cos(h), np.sin(h),
              color="red", scale=25, width=0.004, zorder=5)
    ax.plot(*episode["start_xy"], "go", ms=12, label="start", zorder=10)
    ax.plot(*episode["goal_xy"], "r*", ms=15, label="goal", zorder=10)
    ax.add_patch(plt.Circle(episode["goal_xy"], env.config.goal_radius_m,
                            fill=False, color="red", ls="--", lw=1))
    ax.set(xlabel="x (North) [m]", ylabel="y (East) [m]")
    ax.set_title(
        f"Trajectory — {episode['policy']} | "
        f"{episode['task_geometry']} | "
        f"{'SUCCESS' if episode['success'] else episode['reason']} | "
        f"{episode['elapsed_time_s']:.1f}s"
    )
    ax.legend(loc="upper left")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # Panel 2 — distance
    ax = axes[0, 1]
    ax.plot(episode["times"], episode["distances"], "b-", lw=1.2)
    ax.axhline(env.config.goal_radius_m, color="r", ls="--", alpha=0.5, label="goal radius")
    ax.set(xlabel="Time [s]", ylabel="Distance to goal [m]", title="Distance to Goal")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3 — cumulative reward
    ax = axes[1, 0]
    if len(episode["rewards"]) > 0:
        ax.plot(episode["times"][1:], np.cumsum(episode["rewards"]), "g-", lw=1.2)
    ax.set(xlabel="Time [s]", ylabel="Cumulative reward", title="Cumulative Reward")
    ax.grid(True, alpha=0.3)

    # Panel 4 — speed
    ax = axes[1, 1]
    ax.plot(episode["times"], episode["speeds"], "b-", lw=1.0, label="AUV speed")
    flow_spd = np.linalg.norm(episode["currents_world"][:, :2], axis=1)
    ax.plot(episode["times"], flow_spd, "r-", lw=1.0, alpha=0.7, label="flow speed")
    ax.set(xlabel="Time [s]", ylabel="Speed [m/s]", title="AUV Speed vs Flow Speed")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_comparison(
    episodes: list[dict], env: PlanarRemusEnv, save_path: str | None = None
) -> None:
    """Three-panel comparison of multiple policies (same start/goal)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ["blue", "orange", "green", "red", "purple"]

    # Panel 1 — trajectories
    ax = axes[0]
    _plot_flow_snapshot(ax, env, t=float(episodes[0]["start_flow_time_s"]))
    for i, ep in enumerate(episodes):
        c = colors[i % len(colors)]
        tag = "OK" if ep["success"] else ep["reason"]
        ax.plot(ep["positions"][:, 0], ep["positions"][:, 1], "-",
                color=c, lw=1.5, label=f"{ep['policy']} ({tag}, {ep['elapsed_time_s']:.1f}s)")
    ax.plot(*episodes[0]["start_xy"], "go", ms=12, zorder=10)
    ax.plot(*episodes[0]["goal_xy"], "r*", ms=15, zorder=10)
    ax.add_patch(plt.Circle(episodes[0]["goal_xy"], env.config.goal_radius_m,
                            fill=False, color="red", ls="--", lw=1))
    ax.set(xlabel="x (North) [m]", ylabel="y (East) [m]", title="Trajectory Comparison")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # Panel 2 — distance
    ax = axes[1]
    for i, ep in enumerate(episodes):
        ax.plot(ep["times"], ep["distances"], "-", color=colors[i % len(colors)],
                lw=1.2, label=ep["policy"])
    ax.axhline(env.config.goal_radius_m, color="r", ls="--", alpha=0.5)
    ax.set(xlabel="Time [s]", ylabel="Distance to goal [m]", title="Distance to Goal")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3 — cumulative reward
    ax = axes[2]
    for i, ep in enumerate(episodes):
        if len(ep["rewards"]) > 0:
            ax.plot(ep["times"][1:], np.cumsum(ep["rewards"]), "-",
                    color=colors[i % len(colors)], lw=1.2, label=ep["policy"])
    ax.set(xlabel="Time [s]", ylabel="Cumulative reward", title="Cumulative Reward")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline rollouts on the planar wake environment."
    )
    parser.add_argument("--flow", type=Path, default=None,
                        help="Path to a wake .npy file.")
    parser.add_argument("--policy",
                        choices=list(POLICIES.keys()) + ["all"],
                        default="all",
                        help="Baseline policy to execute.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes per policy.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed.")
    parser.add_argument("--plot", action="store_true",
                        help="Show trajectory plot (requires matplotlib).")
    parser.add_argument("--save", type=str, default=None,
                        help="Save trajectory plot to file (implies --plot).")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None,
                        help="Benchmark difficulty preset: easy=downstream, medium=cross_stream, hard=upstream.")
    parser.add_argument("--task-geometry",
                        choices=["downstream", "cross_stream", "upstream"],
                        default=None,
                        help="Explicit benchmark geometry override.")
    parser.add_argument("--action-mode",
                        choices=["auto", "goal_relative_offset", "absolute_heading"],
                        default=None,
                        help="Explicit action encoding override.")
    parser.add_argument("--speed-ratio", type=float, default=None,
                        help="Target AUV max-speed / reference-flow-speed ratio for benchmark mode.")
    parser.add_argument("--target-speed", type=float, default=None,
                        help="Explicit target AUV max speed in m/s. Overrides --speed-ratio.")
    args = parser.parse_args()

    want_plot = args.plot or args.save is not None

    flow_path = args.flow or discover_flow_path()
    config = PlanarRemusEnvConfig(flow_path=flow_path)
    env = PlanarRemusEnv(config)
    reset_options: dict[str, Any] = {}
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

    policies = list(POLICIES.keys()) if args.policy == "all" else [args.policy]

    # ── text-mode batch runs ──
    for policy_name in policies:
        results = []
        for i in range(args.episodes):
            seed = args.seed + i
            result = run_episode(env, policy_name, seed=seed, reset_options=reset_options)
            results.append(result)

            if args.episodes <= 5:
                print(f"\n--- Episode {i+1} ---")
                print_episode_summary(result)

        if args.episodes > 1:
            print_batch_summary(results, policy_name)

    # ── optional plot (one seed only) ──
    if want_plot:
        if len(policies) > 1:
            episodes = [
                run_episode(
                    env, p, seed=args.seed,
                    collect_trajectory=True,
                    reset_options=reset_options,
                )
                for p in policies
            ]
            plot_comparison(episodes, env, save_path=args.save)
        else:
            ep = run_episode(env, policies[0], seed=args.seed,
                             collect_trajectory=True,
                             reset_options=reset_options)
            plot_episode(ep, env, save_path=args.save)


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
