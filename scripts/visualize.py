"""
Wake-environment rollout animation.

Renders a single episode as a four-panel matplotlib animation:

  Top panel        — vorticity field (omega_1ps) heat-map.  The AUV is drawn
                     as a filled triangle (nose = heading direction).  The
                     trajectory trail is coloured by the flow's contribution to
                     goal-directed progress: green = flow helping, red = flow
                     opposing.  Start, goal and its acceptance radius are
                     marked.

  Middle-left      — distance to goal over time.
  Middle-right     — cumulative reward over time.
  Bottom (full)    — goal-direction speed decomposition:
                       • blue  solid  : ground speed toward goal (v_goal_ground)
                       • green solid  : flow contribution  (v_goal_flow)
                       • orange dashed: AUV's own rowing contribution
                                        (v_goal_water_rel = ground − flow)
                     The gap between ground and own-rowing lines is the flow's
                     share; when the green region is positive the AUV is
                     riding the current.

Output is saved as .mp4 (ffmpeg) or .gif (Pillow fallback), or displayed
interactively when --save is omitted.

Usage
-----
    python -m scripts.visualize_wake_env
    python -m scripts.visualize_wake_env --policy compensate --fps 15
    python -m scripts.visualize_wake_env --save episode.mp4
    python -m scripts.visualize_wake_env --difficulty medium --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from auv_nav.envs import PlanarRemusEnv, PlanarRemusEnvConfig
from auv_nav.policies import (
    CrossCurrentCompensationPolicy,
    GoalSeekPolicy,
    PrivilegedCorridorPolicy,
    StillWaterStraightLine,
    WorldFrameCurrentCompensationPolicy,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICIES: dict[str, type] = {
    "goal":             GoalSeekPolicy,
    "compensate":       CrossCurrentCompensationPolicy,
    "still_water":      StillWaterStraightLine,
    "world_compensate": WorldFrameCurrentCompensationPolicy,
    "corridor":         PrivilegedCorridorPolicy,
}

# AUV marker dimensions in physical metres (exaggerated for readability).
_AUV_LENGTH_M = 8.0
_AUV_WIDTH_M  = 4.0

# Past control steps shown as a coloured trajectory trail.
_TRAIL_STEPS = 60


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def discover_flow_path() -> Path:
    """Return the first wake .npy file found under ./wake_data/."""
    candidates = sorted(Path("wake_data").glob("wake_*_roi.npy"))
    if not candidates:
        raise FileNotFoundError(
            "No wake-field data found under ./wake_data/. "
            "Generate data with `python -m scripts.generate_wake_v8` first."
        )
    return candidates[0]


def _auv_triangle(x: float, y: float, psi: float) -> np.ndarray:
    """Return a (3, 2) array of triangle vertices for the AUV at (x, y, psi).

    psi = 0 means the nose points in the +x (North) direction.
    """
    fwd  = np.array([ np.cos(psi),  np.sin(psi)])
    side = np.array([-np.sin(psi),  np.cos(psi)])
    origin = np.array([x, y])
    nose  = origin + fwd  * (_AUV_LENGTH_M * 0.6)
    left  = origin - fwd  * (_AUV_LENGTH_M * 0.4) + side * (_AUV_WIDTH_M * 0.5)
    right = origin - fwd  * (_AUV_LENGTH_M * 0.4) - side * (_AUV_WIDTH_M * 0.5)
    return np.array([nose, left, right])


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def run_and_collect(
    env: PlanarRemusEnv,
    policy_name: str,
    seed: int,
    reset_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Roll out one episode and collect all per-step data for animation.

    Returns a dict with arrays of length N (number of control steps + 1 for
    the initial state):

      positions        (N, 2)  — (x_North, y_East) in metres
      headings         (N,)    — psi in radians
      distances        (N,)    — distance to goal in metres
      times            (N,)    — elapsed simulation time in seconds
      flow_times       (N,)    — wake-field time pointer in seconds
      body_velocities  (N, 2)  — [u, v] body-frame ground velocity (m/s)
      currents_world   (N, 2)  — [u_c, v_c] equivalent current world frame (m/s)
      rewards          (N-1,)  — per-step reward
      cum_rewards      (N,)    — cumulative reward (starts at 0)
    """
    policy = POLICIES[policy_name]()
    obs, info = env.reset(seed=seed, options=reset_options or {})

    positions       = [info["position_xy_m"].copy()]
    headings        = [float(info["psi_rad"])]
    distances       = [float(info["distance_to_goal_m"])]
    times           = [0.0]
    flow_times      = [float(info["flow_time_s"])]
    body_velocities = [info["body_velocity"][:2].copy()]
    currents_world  = [info["equivalent_current_world"][:2].copy()]
    rewards: list[float] = []

    done = False
    while not done:
        action = policy.act(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        positions.append(info["position_xy_m"].copy())
        headings.append(float(info["psi_rad"]))
        distances.append(float(info["distance_to_goal_m"]))
        times.append(float(info["elapsed_time_s"]))
        flow_times.append(float(info["flow_time_s"]))
        body_velocities.append(info["body_velocity"][:2].copy())
        currents_world.append(info["equivalent_current_world"][:2].copy())
        rewards.append(float(reward))

    cum_rewards = np.concatenate([[0.0], np.cumsum(rewards)]).astype(np.float32)

    return {
        "policy":           policy_name,
        "seed":             seed,
        "success":          bool(info["success"]),
        "reason":           str(info["reason"]),
        "start_xy":         info["start_xy_m"].copy(),
        "goal_xy":          info["goal_xy_m"].copy(),
        "positions":        np.array(positions,       dtype=np.float32),
        "headings":         np.array(headings,        dtype=np.float32),
        "distances":        np.array(distances,       dtype=np.float32),
        "times":            np.array(times,           dtype=np.float32),
        "flow_times":       np.array(flow_times,      dtype=np.float32),
        "body_velocities":  np.array(body_velocities, dtype=np.float32),
        "currents_world":   np.array(currents_world,  dtype=np.float32),
        "rewards":          np.array(rewards,         dtype=np.float32),
        "cum_rewards":      cum_rewards,
    }


# ---------------------------------------------------------------------------
# Speed decomposition
# ---------------------------------------------------------------------------

def _compute_goal_speed_decomposition(episode: dict[str, Any]) -> dict[str, np.ndarray]:
    """Project velocity components onto the instantaneous goal direction.

    All output arrays have shape (N,) matching the other episode arrays.

    Returns
    -------
    v_goal_ground    — ground speed toward goal  (positive = approaching)
    v_goal_flow      — flow's contribution toward goal
    v_goal_water_rel — AUV's own rowing contribution (= ground − flow)

    Identity: v_goal_ground = v_goal_water_rel + v_goal_flow
    """
    positions      = episode["positions"]        # (N, 2)
    headings       = episode["headings"]         # (N,)
    body_vels      = episode["body_velocities"]  # (N, 2)
    currents       = episode["currents_world"]   # (N, 2)
    goal_xy        = episode["goal_xy"]          # (2,)

    # Rotate body-frame ground velocity into world frame.
    cos_p = np.cos(headings)
    sin_p = np.sin(headings)
    gnd_x = body_vels[:, 0] * cos_p - body_vels[:, 1] * sin_p
    gnd_y = body_vels[:, 0] * sin_p + body_vels[:, 1] * cos_p

    # Unit vector from current position toward goal.
    d = goal_xy - positions                                  # (N, 2)
    dist = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-3)
    goal_dir = d / dist                                      # (N, 2)

    v_goal_ground    = gnd_x * goal_dir[:, 0] + gnd_y * goal_dir[:, 1]
    v_goal_flow      = (currents[:, 0] * goal_dir[:, 0]
                        + currents[:, 1] * goal_dir[:, 1])
    v_goal_water_rel = v_goal_ground - v_goal_flow

    return {
        "v_goal_ground":    v_goal_ground.astype(np.float32),
        "v_goal_flow":      v_goal_flow.astype(np.float32),
        "v_goal_water_rel": v_goal_water_rel.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def _preload_omega(env: PlanarRemusEnv, flow_times: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract vorticity frames for every animation step.

    Returns
    -------
    omega : (N, Ny, Nx) float32 — transposed for ``imshow`` with origin='lower'.
    vmax  : float — robust colour-scale limit (99th percentile of |omega|).
    """
    wf = env.wake_field
    frame_indices = (flow_times / wf.dt).astype(np.int32) % wf.total_frames
    omega_nx_ny = wf.flow[frame_indices, :, :, 2].astype(np.float32)
    omega = omega_nx_ny.transpose(0, 2, 1)           # (N, Ny, Nx)
    vmax = max(float(np.percentile(np.abs(omega), 99)), 1e-4)
    return omega, vmax


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def build_and_save_animation(
    episode: dict[str, Any],
    env: PlanarRemusEnv,
    *,
    fps: int = 10,
    trail_steps: int = _TRAIL_STEPS,
    save_path: str | None = None,
) -> None:
    """Build the four-panel animation and either display or save it."""

    wf          = env.wake_field
    positions   = episode["positions"]    # (N, 2)
    headings    = episode["headings"]     # (N,)
    distances   = episode["distances"]    # (N,)
    times       = episode["times"]        # (N,)
    flow_times  = episode["flow_times"]   # (N,)
    cum_rewards = episode["cum_rewards"]  # (N,)
    N = len(positions)

    print("Pre-loading vorticity frames...", end=" ", flush=True)
    omega_frames, vmax = _preload_omega(env, flow_times)
    print(f"done  ({N} frames, |ω|_99 = {vmax:.4f} 1/s)")

    print("Computing goal-direction speed decomposition...", end=" ", flush=True)
    spd = _compute_goal_speed_decomposition(episode)
    v_goal_ground    = spd["v_goal_ground"]
    v_goal_flow      = spd["v_goal_flow"]
    v_goal_water_rel = spd["v_goal_water_rel"]
    print("done")

    # Colour scale for the trail: symmetric around zero.
    flow_scale = max(float(np.percentile(np.abs(v_goal_flow), 95)), 0.1)
    trail_norm = Normalize(vmin=-flow_scale, vmax=flow_scale)

    # Physical extent of the ROI (half-cell pad for pixel alignment).
    dx = wf.dx
    x0, x1 = wf.x_min - dx / 2, wf.x_max + dx / 2
    y0, y1 = wf.y_min - dx / 2, wf.y_max + dx / 2
    img_extent = [x0, x1, y0, y1]

    # -----------------------------------------------------------------------
    # Figure layout  (4 rows, 2 columns)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(
        3, 2,
        height_ratios=[4, 2, 2],
        hspace=0.50, wspace=0.32,
        left=0.07, right=0.97, top=0.92, bottom=0.06,
    )
    ax_flow   = fig.add_subplot(gs[0, :])   # full width — flow field
    ax_dist   = fig.add_subplot(gs[1, 0])   # distance to goal
    ax_reward = fig.add_subplot(gs[1, 1])   # cumulative reward
    ax_speed  = fig.add_subplot(gs[2, :])   # goal-direction speed decomposition

    # -----------------------------------------------------------------------
    # Static elements — flow panel
    # -----------------------------------------------------------------------
    ax_flow.set_xlim(x0, x1)
    ax_flow.set_ylim(y0, y1)
    ax_flow.set_xlabel("x / North [m]", fontsize=9)
    ax_flow.set_ylabel("y / East [m]", fontsize=9)
    ax_flow.set_aspect("equal", adjustable="box")
    ax_flow.tick_params(labelsize=8)

    goal_xy  = episode["goal_xy"]
    start_xy = episode["start_xy"]
    goal_r   = env.config.goal_radius_m

    ax_flow.add_patch(plt.Circle(goal_xy, goal_r, fill=False,
                                 color="lime", ls="--", lw=1.5, zorder=5))
    ax_flow.plot(*goal_xy,  "*", color="lime", ms=10, zorder=6, label="goal")
    ax_flow.plot(*start_xy, "o", color="cyan", ms=8,  zorder=6, label="start")

    # -----------------------------------------------------------------------
    # Animated artists — flow panel
    # -----------------------------------------------------------------------

    # Vorticity heat-map.
    im = ax_flow.imshow(
        omega_frames[0],
        origin="lower",
        extent=img_extent,
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        aspect="auto",
        interpolation="bilinear",
        zorder=1,
    )
    cbar_omega = fig.colorbar(im, ax=ax_flow, pad=0.01, fraction=0.018,
                              label="ω [1/s]")
    cbar_omega.ax.tick_params(labelsize=7)

    # Coloured trajectory trail (LineCollection: v_goal_flow → RdYlGn).
    lc_trail = LineCollection(
        [], cmap="RdYlGn", norm=trail_norm,
        lw=2.2, zorder=3, capstyle="round",
    )
    ax_flow.add_collection(lc_trail)

    # AUV triangle.
    auv_patch = mpatches.Polygon(
        _auv_triangle(positions[0, 0], positions[0, 1], headings[0]),
        closed=True, fc="yellow", ec="black", lw=0.8, zorder=7,
    )
    ax_flow.add_patch(auv_patch)

    ax_flow.legend(loc="upper right", fontsize=8, framealpha=0.55)
    status_str = "SUCCESS" if episode["success"] else episode["reason"]
    fig.suptitle(
        f"Policy: {episode['policy']}  |  seed = {episode['seed']}  |  {status_str}",
        fontsize=11,
    )
    title_text = ax_flow.set_title("", fontsize=9)

    # -----------------------------------------------------------------------
    # Static elements — distance panel
    # -----------------------------------------------------------------------
    t_end = float(times[-1]) * 1.05 or 1.0

    ax_dist.set_xlim(0, t_end)
    ax_dist.set_ylim(0, float(distances[0]) * 1.1 or 1.0)
    ax_dist.axhline(goal_r, color="lime", ls="--", lw=1.0, alpha=0.8,
                    label=f"r = {goal_r:.1f} m")
    ax_dist.set_xlabel("Time [s]", fontsize=8)
    ax_dist.set_ylabel("Distance [m]", fontsize=8)
    ax_dist.set_title("Distance to Goal", fontsize=9)
    ax_dist.tick_params(labelsize=7)
    ax_dist.legend(fontsize=7, loc="upper right")
    ax_dist.grid(True, alpha=0.3)
    (line_dist,) = ax_dist.plot([], [], "b-", lw=1.2)

    # -----------------------------------------------------------------------
    # Static elements — reward panel
    # -----------------------------------------------------------------------
    r_lo = float(cum_rewards.min()) * 1.1
    r_hi = float(cum_rewards.max()) * 1.1
    if r_lo >= r_hi:
        r_lo, r_hi = -1.0, 1.0
    ax_reward.set_xlim(0, t_end)
    ax_reward.set_ylim(r_lo, r_hi)
    ax_reward.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax_reward.set_xlabel("Time [s]", fontsize=8)
    ax_reward.set_ylabel("Cumulative reward", fontsize=8)
    ax_reward.set_title("Cumulative Reward", fontsize=9)
    ax_reward.tick_params(labelsize=7)
    ax_reward.grid(True, alpha=0.3)
    (line_reward,) = ax_reward.plot([], [], "g-", lw=1.2)

    # -----------------------------------------------------------------------
    # Static elements — speed decomposition panel
    # -----------------------------------------------------------------------
    all_speeds = np.concatenate([v_goal_ground, v_goal_flow, v_goal_water_rel])
    s_lo = float(all_speeds.min()) * 1.15
    s_hi = float(all_speeds.max()) * 1.15
    if s_lo >= s_hi:
        s_lo, s_hi = -0.5, 2.0

    ax_speed.set_xlim(0, t_end)
    ax_speed.set_ylim(s_lo, s_hi)
    ax_speed.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax_speed.set_xlabel("Time [s]", fontsize=8)
    ax_speed.set_ylabel("Speed toward goal [m/s]", fontsize=8)
    ax_speed.set_title(
        "Goal-Direction Speed Decomposition  "
        "(ground = own-rowing + flow contribution)",
        fontsize=9,
    )
    ax_speed.tick_params(labelsize=7)
    ax_speed.grid(True, alpha=0.3)

    # Three animated lines + static legend proxies.
    (line_spd_ground,)    = ax_speed.plot([], [], color="steelblue",  lw=1.6,
                                          label="ground speed")
    (line_spd_flow,)      = ax_speed.plot([], [], color="forestgreen", lw=1.6,
                                          label="flow contribution")
    (line_spd_water_rel,) = ax_speed.plot([], [], color="darkorange",  lw=1.2,
                                          ls="--", label="AUV rowing")
    ax_speed.legend(fontsize=7, loc="upper right", framealpha=0.6)

    # Trail colour scale: attach to ax_speed (v_goal_flow is the same quantity).
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=trail_norm)
    sm.set_array([])
    cbar_trail = fig.colorbar(sm, ax=ax_speed, pad=0.01, fraction=0.018,
                              label="flow→goal [m/s]")
    cbar_trail.ax.tick_params(labelsize=7)

    # Vertical cursor line shared across all three time-series axes.
    cursor_dist   = ax_dist.axvline(0, color="red", lw=0.8, alpha=0.5)
    cursor_reward = ax_reward.axvline(0, color="red", lw=0.8, alpha=0.5)
    cursor_speed  = ax_speed.axvline(0, color="red", lw=0.8, alpha=0.5)

    # -----------------------------------------------------------------------
    # Animation update
    # -----------------------------------------------------------------------

    def update(i: int) -> list:
        # --- vorticity frame ---
        im.set_data(omega_frames[i])

        # --- coloured trail (LineCollection) ---
        t0 = max(0, i - trail_steps)
        pts = positions[t0 : i + 1]          # (k+1, 2)
        if len(pts) >= 2:
            segs = np.stack([pts[:-1], pts[1:]], axis=1)   # (k, 2, 2)
            vals = v_goal_flow[t0 : i]                      # (k,)
            lc_trail.set_segments(segs)
            lc_trail.set_array(vals)
        else:
            lc_trail.set_segments([])

        # --- AUV triangle ---
        auv_patch.set_xy(_auv_triangle(positions[i, 0], positions[i, 1], headings[i]))

        # --- frame label ---
        f_idx = int(flow_times[i] / wf.dt) % wf.total_frames
        title_text.set_text(
            f"t = {times[i]:.1f} s  |  dist = {distances[i]:.1f} m  "
            f"|  flow frame {f_idx}/{wf.total_frames}"
        )

        # --- time-series lines ---
        t_slice = times[: i + 1]
        line_dist.set_data(t_slice, distances[: i + 1])
        line_reward.set_data(t_slice, cum_rewards[: i + 1])
        line_spd_ground.set_data(t_slice,    v_goal_ground[: i + 1])
        line_spd_flow.set_data(t_slice,      v_goal_flow[: i + 1])
        line_spd_water_rel.set_data(t_slice, v_goal_water_rel[: i + 1])

        # --- cursors ---
        t_now = float(times[i])
        cursor_dist.set_xdata([t_now, t_now])
        cursor_reward.set_xdata([t_now, t_now])
        cursor_speed.set_xdata([t_now, t_now])

        return [
            im, lc_trail, auv_patch, title_text,
            line_dist, line_reward,
            line_spd_ground, line_spd_flow, line_spd_water_rel,
            cursor_dist, cursor_reward, cursor_speed,
        ]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=N,
        interval=max(1, 1000 // fps),
        blit=True,
    )

    if save_path is not None:
        _save_animation(ani, save_path, fps)
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_animation(
    ani: animation.FuncAnimation, save_path: str, fps: int
) -> None:
    out = Path(save_path)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        dest   = out.with_suffix(".mp4")
        ani.save(str(dest), writer=writer)
        print(f"Saved → {dest}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ffmpeg unavailable ({exc}); falling back to Pillow GIF.")
        writer = animation.PillowWriter(fps=fps)
        dest   = out.with_suffix(".gif")
        ani.save(str(dest), writer=writer)
        print(f"Saved → {dest}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a single wake-environment rollout as an animation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--flow", type=Path, default=None,
        help="Path to a wake .npy file.  Auto-discovers under ./wake_data/ if omitted.",
    )
    parser.add_argument(
        "--policy", choices=list(POLICIES), default="goal",
        help="Baseline policy to visualise.",
    )
    parser.add_argument("--seed",  type=int, default=42, help="Random seed.")
    parser.add_argument("--fps",   type=int, default=10, help="Animation frame rate.")
    parser.add_argument(
        "--trail", type=int, default=_TRAIL_STEPS,
        help="Number of past steps shown as trajectory trail.",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Output file path (.mp4 via ffmpeg, or .gif via Pillow). "
             "Displays interactively if omitted.",
    )
    parser.add_argument(
        "--task-geometry", choices=["downstream", "cross_stream", "upstream"],
        default=None,
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], default=None,
    )
    args = parser.parse_args()

    flow_path = args.flow or discover_flow_path()
    print(f"Flow file : {flow_path}")

    env = PlanarRemusEnv(PlanarRemusEnvConfig(flow_path=flow_path))

    reset_options: dict[str, Any] = {}
    if args.task_geometry is not None:
        reset_options["task_geometry"] = args.task_geometry
    if args.difficulty is not None:
        reset_options["task_difficulty"] = args.difficulty

    print(f"Policy    : {args.policy}  (seed={args.seed})")
    episode = run_and_collect(env, args.policy, args.seed, reset_options)

    status = "SUCCESS" if episode["success"] else episode["reason"]
    print(
        f"Episode   : {status}  |  {len(episode['positions'])} steps  "
        f"|  final dist = {episode['distances'][-1]:.1f} m  "
        f"|  total reward = {episode['cum_rewards'][-1]:.1f}"
    )

    build_and_save_animation(
        episode, env,
        fps=args.fps,
        trail_steps=args.trail,
        save_path=args.save,
    )


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
