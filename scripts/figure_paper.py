"""
Static paper figures for the wake-navigation RL paper.

Produces two publication-ready PDF figures:

  figure1_comparison.pdf
    Top    — vorticity snapshot with complete trajectories for every policy.
             Each policy gets a distinct colour; trajectory width encodes
             instantaneous flow contribution toward the goal (v_goal_flow).
    Bottom — left:  distance-to-goal curves (all policies overlaid).
             right: cumulative flow contribution ∫v_goal_flow dt (all
                    policies overlaid) — the quantitative "who rides the
                    current" metric.

  figure2_speed_decomposition.pdf  (optional, requires --checkpoint)
    Single policy (RL or best baseline) speed decomposition:
      blue  solid  — ground speed toward goal
      green filled — flow contribution (fill_between with zero axis)
      orange dashed — AUV own rowing contribution
    Shows that RL actively reduces self-propulsion when the flow is helpful.

Usage
-----
  # baselines only
  python -m scripts.figure_paper --seed 42

  # include RL from a checkpoint directory
  python -m scripts.figure_paper --checkpoint runs/my_run --seed 42

  # downstream task, medium difficulty
  python -m scripts.figure_paper --task-geometry downstream --difficulty medium

  # specify output directory
  python -m scripts.figure_paper --outdir figs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from auv_nav.envs import PlanarRemusEnv, PlanarRemusEnvConfig
from auv_nav.policies import (
    CrossCurrentCompensationPolicy,
    GoalSeekPolicy,
    PrivilegedCorridorPolicy,
    WorldFrameCurrentCompensationPolicy,
)


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

# Publication colours — colour-blind-friendly (Wong 2011 palette)
_POLICY_STYLES: dict[str, dict] = {
    "goal":             {"color": "#E69F00", "lw": 1.6, "ls": "-",  "label": "Goal-Seek"},
    "world_compensate": {"color": "#56B4E9", "lw": 1.6, "ls": "--", "label": "World-Compensate"},
    "corridor":         {"color": "#009E73", "lw": 1.6, "ls": "-.", "label": "Privileged-Corridor"},
    "rl":               {"color": "#CC79A7", "lw": 2.4, "ls": "-",  "label": "GRU-Res-SAC (ours)"},
}

_BASELINE_POLICIES = ["goal", "world_compensate", "corridor"]

_AUV_LENGTH_M = 8.0
_AUV_WIDTH_M  = 4.0

# Figure widths in inches — double-column IEEE ≈ 7.16"
_FIG1_SIZE = (10.0, 7.5)
_FIG2_SIZE = (7.0,  3.5)


# ---------------------------------------------------------------------------
# Helpers shared with visualize_wake_env (duplicated to keep file independent)
# ---------------------------------------------------------------------------

def _discover_flow_path() -> Path:
    candidates = sorted(Path("wake_data").glob("wake_*_roi.npy"))
    if not candidates:
        raise FileNotFoundError(
            "No wake .npy found under ./wake_data/.  "
            "Run `python -m scripts.generate_wake_v8` first."
        )
    return candidates[0]


def _auv_triangle(x: float, y: float, psi: float) -> np.ndarray:
    fwd  = np.array([ np.cos(psi),  np.sin(psi)])
    side = np.array([-np.sin(psi),  np.cos(psi)])
    origin = np.array([x, y])
    nose  = origin + fwd  * (_AUV_LENGTH_M * 0.6)
    left  = origin - fwd  * (_AUV_LENGTH_M * 0.4) + side * (_AUV_WIDTH_M * 0.5)
    right = origin - fwd  * (_AUV_LENGTH_M * 0.4) - side * (_AUV_WIDTH_M * 0.5)
    return np.array([nose, left, right])


# ---------------------------------------------------------------------------
# Episode collection — baselines
# ---------------------------------------------------------------------------

_POLICY_CLS = {
    "goal":             GoalSeekPolicy,
    "world_compensate": WorldFrameCurrentCompensationPolicy,
    "corridor":         PrivilegedCorridorPolicy,
}


def run_baseline(
    env: PlanarRemusEnv,
    policy_name: str,
    seed: int,
    reset_options: dict[str, Any],
) -> dict[str, Any]:
    policy = _POLICY_CLS[policy_name]()
    obs, info = env.reset(seed=seed, options=reset_options)

    positions      = [info["position_xy_m"].copy()]
    headings       = [float(info["psi_rad"])]
    distances      = [float(info["distance_to_goal_m"])]
    times          = [0.0]
    flow_times     = [float(info["flow_time_s"])]
    body_vels      = [info["body_velocity"][:2].copy()]
    currents_world = [info["equivalent_current_world"][:2].copy()]
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
        body_vels.append(info["body_velocity"][:2].copy())
        currents_world.append(info["equivalent_current_world"][:2].copy())
        rewards.append(float(reward))

    cum_rewards = np.concatenate([[0.0], np.cumsum(rewards)]).astype(np.float32)
    return dict(
        policy_name    = policy_name,
        seed           = seed,
        success        = bool(info["success"]),
        reason         = str(info["reason"]),
        start_xy       = info["start_xy_m"].copy(),
        goal_xy        = info["goal_xy_m"].copy(),
        positions      = np.array(positions,      dtype=np.float32),
        headings       = np.array(headings,       dtype=np.float32),
        distances      = np.array(distances,      dtype=np.float32),
        times          = np.array(times,          dtype=np.float32),
        flow_times     = np.array(flow_times,     dtype=np.float32),
        body_velocities= np.array(body_vels,      dtype=np.float32),
        currents_world = np.array(currents_world, dtype=np.float32),
        rewards        = np.array(rewards,        dtype=np.float32),
        cum_rewards    = cum_rewards,
    )


# ---------------------------------------------------------------------------
# Episode collection — RL agent
# ---------------------------------------------------------------------------

def run_rl_agent(
    env: PlanarRemusEnv,
    checkpoint_dir: str,
    seed: int,
    reset_options: dict[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a GRU-Residual-SAC checkpoint and collect one episode."""
    from auv_nav.algorithms import GRUResidualSACAgent, GRUResidualSACConfig, require_torch
    from auv_nav.policies import CrossCurrentResidualPrior, ResidualPriorConfig

    require_torch()

    ckpt_root = Path(checkpoint_dir)
    meta_path = ckpt_root / "trainer_state.json" if ckpt_root.is_dir() else ckpt_root
    with meta_path.open() as fp:
        state = json.load(fp)

    agent_cfg = GRUResidualSACConfig(**state["agent_config"])

    # Build residual prior (mirrors train script).
    merged_options = dict(state.get("reset_options", {}))
    merged_options.update(reset_options)
    resolved_geometry   = env.task_sampler.resolve_task_geometry(merged_options)
    resolved_action_mode = env.task_sampler.resolve_action_mode(merged_options, resolved_geometry)

    layout = env.get_observation_layout()
    prior = CrossCurrentResidualPrior(ResidualPriorConfig(
        action_mode             = resolved_action_mode,
        heading_offset_limit_rad= env.config.heading_offset_limit_rad,
        heading_cos_index       = layout.heading_cos_index,
        heading_sin_index       = layout.heading_sin_index,
        goal_x_index            = layout.goal_x_index,
        goal_y_index            = layout.goal_y_index,
        center_probe_v_index    = layout.center_probe_v_index,
    ))
    agent = GRUResidualSACAgent(agent_cfg, prior=prior, device=device)
    agent_path = (
        ckpt_root / state["agent_path"] if ckpt_root.is_dir()
        else ckpt_root.parent / state["agent_path"]
    )
    agent.load(str(agent_path))

    obs, info = env.reset(seed=seed, options=merged_options)
    hidden = agent.reset_hidden()

    positions      = [info["position_xy_m"].copy()]
    headings       = [float(info["psi_rad"])]
    distances      = [float(info["distance_to_goal_m"])]
    times          = [0.0]
    flow_times     = [float(info["flow_time_s"])]
    body_vels      = [info["body_velocity"][:2].copy()]
    currents_world = [info["equivalent_current_world"][:2].copy()]
    rewards: list[float] = []

    done = False
    while not done:
        action, hidden = agent.act(obs, hidden, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        positions.append(info["position_xy_m"].copy())
        headings.append(float(info["psi_rad"]))
        distances.append(float(info["distance_to_goal_m"]))
        times.append(float(info["elapsed_time_s"]))
        flow_times.append(float(info["flow_time_s"]))
        body_vels.append(info["body_velocity"][:2].copy())
        currents_world.append(info["equivalent_current_world"][:2].copy())
        rewards.append(float(reward))

    cum_rewards = np.concatenate([[0.0], np.cumsum(rewards)]).astype(np.float32)
    return dict(
        policy_name    = "rl",
        seed           = seed,
        success        = bool(info["success"]),
        reason         = str(info["reason"]),
        start_xy       = info["start_xy_m"].copy(),
        goal_xy        = info["goal_xy_m"].copy(),
        positions      = np.array(positions,      dtype=np.float32),
        headings       = np.array(headings,       dtype=np.float32),
        distances      = np.array(distances,      dtype=np.float32),
        times          = np.array(times,          dtype=np.float32),
        flow_times     = np.array(flow_times,     dtype=np.float32),
        body_velocities= np.array(body_vels,      dtype=np.float32),
        currents_world = np.array(currents_world, dtype=np.float32),
        rewards        = np.array(rewards,        dtype=np.float32),
        cum_rewards    = cum_rewards,
    )


# ---------------------------------------------------------------------------
# Physics: goal-direction speed decomposition
# ---------------------------------------------------------------------------

def _speed_decomposition(ep: dict[str, Any]) -> dict[str, np.ndarray]:
    positions = ep["positions"]        # (N, 2)
    headings  = ep["headings"]         # (N,)
    body_vels = ep["body_velocities"]  # (N, 2)
    currents  = ep["currents_world"]   # (N, 2)
    goal_xy   = ep["goal_xy"]          # (2,)

    cos_p = np.cos(headings)
    sin_p = np.sin(headings)
    gnd_x = body_vels[:, 0] * cos_p - body_vels[:, 1] * sin_p
    gnd_y = body_vels[:, 0] * sin_p + body_vels[:, 1] * cos_p

    d = goal_xy - positions
    dist = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-3)
    goal_dir = d / dist

    v_ground    = gnd_x * goal_dir[:, 0] + gnd_y * goal_dir[:, 1]
    v_flow      = currents[:, 0] * goal_dir[:, 0] + currents[:, 1] * goal_dir[:, 1]
    v_water_rel = v_ground - v_flow

    return dict(
        v_goal_ground    = v_ground.astype(np.float32),
        v_goal_flow      = v_flow.astype(np.float32),
        v_goal_water_rel = v_water_rel.astype(np.float32),
    )


def _cumulative_flow_contribution(ep: dict[str, Any], spd: dict[str, np.ndarray]) -> np.ndarray:
    """∫ v_goal_flow dt, starting at 0.  Shape (N,)."""
    times  = ep["times"]                   # (N,)
    v_flow = spd["v_goal_flow"]            # (N,)
    dt = np.diff(times, prepend=times[0])  # (N,)
    return np.cumsum(v_flow * dt).astype(np.float32)


# ---------------------------------------------------------------------------
# Figure 1 — environment + trajectory comparison
# ---------------------------------------------------------------------------

def _get_vorticity_snapshot(env: PlanarRemusEnv, episodes: list[dict]) -> np.ndarray:
    """Return a (Ny, Nx) vorticity frame at a representative mid-episode time."""
    wf = env.wake_field
    # Use the mid-point of the shortest episode to find a meaningful flow frame.
    mid_t = float(min(ep["flow_times"][len(ep["flow_times"]) // 2] for ep in episodes))
    frame_idx = int(mid_t / wf.dt) % wf.total_frames
    omega_nx_ny = wf.flow[frame_idx, :, :, 2].astype(np.float32)
    return omega_nx_ny.T  # (Ny, Nx) for imshow origin='lower'


def _add_trajectory(
    ax: plt.Axes,
    ep: dict[str, Any],
    spd: dict[str, np.ndarray],
    style: dict,
    flow_norm: Normalize,
) -> None:
    """Draw the full trajectory on ax.

    Uses a single LineCollection per policy for clean rendering.
    A direction arrow is added at the 40% point along the path.
    """
    positions = ep["positions"]      # (N, 2)
    v_flow    = spd["v_goal_flow"]   # (N,)

    segs = np.stack([positions[:-1], positions[1:]], axis=1)  # (N-1, 2, 2)
    alpha_vals = np.clip(0.45 + 0.55 * np.abs(flow_norm(v_flow[:-1])), 0.0, 1.0)

    # Draw as individual segments to preserve per-segment alpha.
    for seg, alpha in zip(segs, alpha_vals):
        ax.plot(
            seg[:, 0], seg[:, 1],
            color=style["color"],
            lw=style["lw"],
            ls=style["ls"],
            alpha=float(alpha),
            solid_capstyle="round",
        )

    # Direction arrow at ~40% of the trajectory.
    n = len(positions)
    idx = max(1, int(0.40 * n))
    ax.annotate(
        "",
        xy=(float(positions[idx, 0]),     float(positions[idx, 1])),
        xytext=(float(positions[idx-1, 0]), float(positions[idx-1, 1])),
        arrowprops=dict(
            arrowstyle="-|>",
            color=style["color"],
            lw=1.6,
            mutation_scale=10,
        ),
        zorder=9,
    )

    # AUV triangle at final position.
    x_end, y_end = float(positions[-1, 0]), float(positions[-1, 1])
    psi_end = float(ep["headings"][-1])
    tri = mpatches.Polygon(
        _auv_triangle(x_end, y_end, psi_end),
        closed=True,
        fc=style["color"],
        ec="white",
        lw=0.8,
        zorder=8,
    )
    ax.add_patch(tri)


def build_figure1(
    episodes: list[dict[str, Any]],
    env: PlanarRemusEnv,
    outpath: Path,
) -> None:
    print("Building Figure 1 …")

    wf = env.wake_field
    dx = wf.dx
    x0, x1 = wf.x_min - dx / 2, wf.x_max + dx / 2
    y0, y1 = wf.y_min - dx / 2, wf.y_max + dx / 2
    img_extent = [x0, x1, y0, y1]

    # Precompute speed decomposition for every episode.
    spds = [_speed_decomposition(ep) for ep in episodes]

    # Shared colour scale for flow contribution (used in alpha modulation).
    all_flow = np.concatenate([s["v_goal_flow"] for s in spds])
    flow_scale = max(float(np.percentile(np.abs(all_flow), 95)), 0.05)
    flow_norm  = Normalize(vmin=-flow_scale, vmax=flow_scale)

    # Vorticity snapshot.
    omega = _get_vorticity_snapshot(env, episodes)
    vmax = max(float(np.percentile(np.abs(omega), 99)), 1e-4)

    # -----------------------------------------------------------------------
    # Layout: 2 rows × 2 cols; top row spans both cols
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=_FIG1_SIZE)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3, 1.6],
        hspace=0.42, wspace=0.30,
        left=0.07, right=0.97, top=0.93, bottom=0.08,
    )
    ax_flow  = fig.add_subplot(gs[0, :])
    ax_dist  = fig.add_subplot(gs[1, 0])
    ax_cumfl = fig.add_subplot(gs[1, 1])

    # -----------------------------------------------------------------------
    # Top panel — vorticity + trajectories
    # -----------------------------------------------------------------------
    ax_flow.imshow(
        omega,
        origin="lower",
        extent=img_extent,
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        aspect="auto",
        interpolation="bilinear",
        zorder=1,
        alpha=0.85,
    )

    # Start / goal (same for all episodes — use first).
    ep0      = episodes[0]
    goal_xy  = ep0["goal_xy"]
    start_xy = ep0["start_xy"]
    goal_r   = env.config.goal_radius_m

    ax_flow.add_patch(plt.Circle(
        goal_xy, goal_r, fill=False, color="#FFD700", ls="--", lw=1.4, zorder=5,
    ))
    ax_flow.plot(*goal_xy,  "*", color="#FFD700", ms=12, zorder=6)
    ax_flow.plot(*start_xy, "o", color="#00E5FF", ms=9,  zorder=6,
                 mfc="none", mew=2.0)

    for ep, spd in zip(episodes, spds):
        style = _POLICY_STYLES[ep["policy_name"]]
        _add_trajectory(ax_flow, ep, spd, style, flow_norm)

    # Show the full ROI with no aspect constraint — imshow fills the panel
    # cleanly and axis labels + tick spacing convey the physical scale.
    # (Equal-aspect would shrink the axes box and leave white gaps.)
    ax_flow.set_xlim(x0, x1)
    ax_flow.set_ylim(y0, y1)
    ax_flow.set_xlabel("x / North [m]", fontsize=9)
    ax_flow.set_ylabel("y / East [m]",  fontsize=9)
    ax_flow.tick_params(labelsize=8)

    # Colourbar for vorticity.
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=Normalize(vmin=-vmax, vmax=vmax),
        ),
        ax=ax_flow, pad=0.01, fraction=0.018,
        label="ω [1/s]",
    )
    cbar.ax.tick_params(labelsize=7)

    # Legend — use explicit handles so markers are always visible.
    legend_handles = [
        Line2D([0], [0], color="#FFD700", marker="*", ms=9,  ls="none", label="Goal"),
        Line2D([0], [0], color="#00E5FF", marker="o", ms=7,  ls="none",
               mfc="none", mew=1.8, label="Start"),
    ] + [
        Line2D(
            [0], [0],
            color=_POLICY_STYLES[ep["policy_name"]]["color"],
            lw=_POLICY_STYLES[ep["policy_name"]]["lw"],
            ls=_POLICY_STYLES[ep["policy_name"]]["ls"],
            label=_POLICY_STYLES[ep["policy_name"]]["label"]
            + (" ✓" if ep["success"] else " ✗"),
        )
        for ep in episodes
    ]
    ax_flow.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.75,
        edgecolor="#555555",
        facecolor="#f5f5f5",
    )

    ax_flow.set_title(
        "Wake-field vorticity and navigation trajectories",
        fontsize=10, pad=4,
    )

    # -----------------------------------------------------------------------
    # Bottom-left — distance to goal
    # -----------------------------------------------------------------------
    ax_dist.axhline(goal_r, color="gray", ls=":", lw=0.9, alpha=0.7,
                    label=f"acceptance r = {goal_r:.0f} m")
    for ep in episodes:
        sty = _POLICY_STYLES[ep["policy_name"]]
        ax_dist.plot(
            ep["times"], ep["distances"],
            color=sty["color"], lw=sty["lw"], ls=sty["ls"],
            label=sty["label"],
        )
    ax_dist.set_xlabel("Time [s]", fontsize=9)
    ax_dist.set_ylabel("Distance to goal [m]", fontsize=9)
    ax_dist.set_title("Distance to Goal", fontsize=9)
    ax_dist.tick_params(labelsize=8)
    ax_dist.grid(True, alpha=0.25)
    ax_dist.legend(fontsize=7, loc="upper right", framealpha=0.5)

    # -----------------------------------------------------------------------
    # Bottom-right — cumulative flow contribution
    # -----------------------------------------------------------------------
    ax_cumfl.axhline(0, color="gray", ls=":", lw=0.9, alpha=0.6)
    for ep, spd in zip(episodes, spds):
        sty     = _POLICY_STYLES[ep["policy_name"]]
        cum_fl  = _cumulative_flow_contribution(ep, spd)
        ax_cumfl.plot(
            ep["times"], cum_fl,
            color=sty["color"], lw=sty["lw"], ls=sty["ls"],
            label=sty["label"],
        )
    ax_cumfl.set_xlabel("Time [s]", fontsize=9)
    ax_cumfl.set_ylabel("∫v_flow→goal dt [m]", fontsize=9)
    ax_cumfl.set_title("Cumulative Flow Contribution", fontsize=9)
    ax_cumfl.tick_params(labelsize=8)
    ax_cumfl.grid(True, alpha=0.25)
    ax_cumfl.legend(fontsize=7, loc="upper left", framealpha=0.5)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


# ---------------------------------------------------------------------------
# Figure 2 — speed decomposition for a single policy
# ---------------------------------------------------------------------------

def build_figure2(
    ep: dict[str, Any],
    outpath: Path,
) -> None:
    print("Building Figure 2 …")

    spd          = _speed_decomposition(ep)
    v_ground     = spd["v_goal_ground"]
    v_flow       = spd["v_goal_flow"]
    v_water_rel  = spd["v_goal_water_rel"]
    times        = ep["times"]
    sty          = _POLICY_STYLES[ep["policy_name"]]

    # Skip the first few steps to avoid AUV spin-up transient.
    skip = min(5, len(times) - 1)
    t_plot  = times[skip:]
    vg_plot = v_ground[skip:]
    vf_plot = v_flow[skip:]
    vw_plot = v_water_rel[skip:]

    fig, ax = plt.subplots(figsize=_FIG2_SIZE)

    ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.6)

    # Flow contribution — filled region (green = helping, red = opposing).
    ax.fill_between(t_plot, 0, vf_plot,
                    where=vf_plot >= 0,
                    color="forestgreen", alpha=0.20, label="_nolegend_")
    ax.fill_between(t_plot, 0, vf_plot,
                    where=vf_plot < 0,
                    color="#CC4444", alpha=0.20, label="_nolegend_")

    ax.plot(t_plot, vg_plot, color="steelblue",   lw=1.8, ls="-",
            label="Ground speed toward goal")
    ax.plot(t_plot, vf_plot, color="forestgreen", lw=1.4, ls="-",
            label="Flow contribution")
    ax.plot(t_plot, vw_plot, color="darkorange",  lw=1.4, ls="--",
            label="AUV own rowing")

    # Annotation: "flow helping" in green fill region.
    pos_mask = vf_plot > 0
    if pos_mask.any():
        idx_c = int(np.argmax(pos_mask)) + max(1, pos_mask.sum() // 3)
        idx_c = min(idx_c, len(t_plot) - 1)
        ax.text(
            float(t_plot[idx_c]),
            max(float(vf_plot[idx_c]) * 0.5, 0.05),
            "flow helping",
            fontsize=7, color="forestgreen", ha="center", va="bottom",
            style="italic",
        )

    # Annotation: "flow opposing" in red fill region.
    neg_mask = vf_plot < 0
    if neg_mask.any():
        # Find the centre of the negative region by median index.
        neg_indices = np.where(neg_mask)[0]
        idx_n = int(neg_indices[len(neg_indices) // 2])
        ax.text(
            float(t_plot[idx_n]),
            min(float(vf_plot[idx_n]) * 0.5, -0.05),
            "flow opposing",
            fontsize=7, color="#AA2222", ha="center", va="top",
            style="italic",
        )

    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("Speed toward goal [m/s]", fontsize=9)
    ax.set_title(
        f"Goal-direction speed decomposition — "
        f"{sty['label']} "
        f"({'SUCCESS' if ep['success'] else ep['reason']})",
        fontsize=9,
    )
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.6)

    # Matching x-axis limits (from skip onward).
    ax.set_xlim(float(t_plot[0]), float(t_plot[-1]) * 1.02)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate static paper figures for the wake-navigation RL study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--flow", type=Path, default=None,
                        help="Wake .npy file.  Auto-discovers under ./wake_data/ if omitted.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-geometry",
                        choices=["downstream", "cross_stream", "upstream"],
                        default=None)
    parser.add_argument("--difficulty",
                        choices=["easy", "medium", "hard"],
                        default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="GRU-Res-SAC checkpoint dir/file.  Adds RL to the comparison.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=Path, default=Path("."),
                        help="Output directory for PDF figures.")
    parser.add_argument("--decomp-policy", default=None,
                        choices=list(_POLICY_STYLES),
                        help="Which policy to use for Figure 2.  "
                             "Defaults to 'rl' if checkpoint given, else 'world_compensate'.")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    flow_path = args.flow or _discover_flow_path()
    print(f"Flow file : {flow_path}")
    env = PlanarRemusEnv(PlanarRemusEnvConfig(flow_path=flow_path))

    reset_options: dict[str, Any] = {}
    if args.task_geometry:
        reset_options["task_geometry"] = args.task_geometry
    if args.difficulty:
        reset_options["task_difficulty"] = args.difficulty

    # --- collect baseline episodes ---
    episodes: list[dict] = []
    for name in _BASELINE_POLICIES:
        print(f"Rolling out '{name}' …", end=" ", flush=True)
        ep = run_baseline(env, name, args.seed, reset_options)
        status = "SUCCESS" if ep["success"] else ep["reason"]
        print(f"{status}  ({len(ep['positions'])} steps)")
        episodes.append(ep)

    # --- collect RL episode (optional) ---
    if args.checkpoint:
        print("Rolling out RL agent …", end=" ", flush=True)
        try:
            ep_rl = run_rl_agent(
                env, args.checkpoint, args.seed, reset_options, device=args.device
            )
            status = "SUCCESS" if ep_rl["success"] else ep_rl["reason"]
            print(f"{status}  ({len(ep_rl['positions'])} steps)")
            episodes.append(ep_rl)
        except Exception as exc:
            print(f"FAILED ({exc}) — skipping RL episode.")

    # --- Figure 1 ---
    build_figure1(episodes, env, outdir / "figure1_comparison.pdf")

    # --- Figure 2 ---
    decomp_name = args.decomp_policy or ("rl" if args.checkpoint else "goal")
    ep2 = next((e for e in episodes if e["policy_name"] == decomp_name), episodes[-1])
    build_figure2(ep2, outdir / "figure2_speed_decomposition.pdf")

    print("Done.")


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
