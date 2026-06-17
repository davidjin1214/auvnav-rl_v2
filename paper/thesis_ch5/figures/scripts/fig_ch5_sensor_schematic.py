"""
paper/thesis_ch5/figures/scripts/fig_ch5_sensor_schematic.py

Figure (§5.3.3): task geometry + the deployable/privileged information boundary
— the chapter's core experimental lever. Replaces the paper-1-style schematic
with the unified Chapter-5 visual style.

Panel (a) Kármán wake field + REMUS-100 cross-stream task:
    one vorticity frame (subcritical Re=150, U_inf=1.0 m/s) with the AUV body,
    heading, the cross-stream goal, the 4 m success radius, and the +/-20 deg
    cross-stream cone. Start-goal range (40-90 m) is stated in the caption.

Panel (b) Deployable vs. privileged sampling, body frame to scale:
    single-point DVL at the hull origin (deployable; read by the policy in both
    protocols) vs. the five along-hull points whose average is the privileged
    flow (read by the value network only in the privileged protocol).

Design rules (shared with the chapter's other figures):
  - English words, minimal in-figure text; protocol semantics live in the
    caption (which keeps the o_priv / u_eq,v_eq symbols).
  - Orthogonal/clean guides, width <= 138 mm, shared _ch5_style palette/fonts.
  - NO result numbers (spec §0.4 red line).

Source data:
    wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy (+ _meta)

Usage (needs the wake .npy locally; pure numpy/matplotlib, no auv_nav):
    python paper/thesis_ch5/figures/scripts/fig_ch5_sensor_schematic.py

Output:
    paper/thesis_ch5/figures/fig_ch5_sensor_schematic.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch5_style import (  # noqa: E402
    COLORS,
    MAX_WIDTH_MM,
    apply_style,
    in_from_mm,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
WAKE_NPY = REPO_ROOT / "wake_data" / "wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
WAKE_META = REPO_ROOT / "wake_data" / "wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi_meta.json"

GOAL_GREEN = "#2E7D32"        # success / goal accent (low-saturation)
FRAME_IDX = 600
AUV_POS = (210.0, 78.0)
AUV_HEADING_DEG = 74.0        # cross-stream, ~16 deg off +y
GOAL_POS = (220.0, 113.0)
PANEL_A_XLIM = (170.0, 285.0)
PANEL_A_YLIM = (62.0, 118.0)
SAMPLE_XI = (-0.4, -0.2, 0.0, 0.2, 0.4)
GOAL_RADIUS_M = 4.0
CROSS_STREAM_CONE_DEG = 20.0


def load_wake_frame(npy_path: Path, meta_path: Path, frame_idx: int):
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    arr = np.load(npy_path, mmap_mode="r")
    frame = np.asarray(arr[frame_idx]).astype(np.float32)
    omega = frame[..., 2]
    extent = (
        float(meta["roi_x0_phys_m"]), float(meta["roi_x1_phys_m"]),
        float(meta["roi_y0_phys_m"]), float(meta["roi_y1_phys_m"]),
    )
    return omega, extent


# ---------------------------------------------------------------------------
# Panel (a)
# ---------------------------------------------------------------------------

def panel_a(ax: plt.Axes) -> None:
    omega, extent = load_wake_frame(WAKE_NPY, WAKE_META, FRAME_IDX)
    vmax = float(np.percentile(np.abs(omega.T), 98))
    im = ax.imshow(omega.T, extent=extent, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="equal", zorder=1)

    cx, cy = AUV_POS
    gx, gy = GOAL_POS

    # AUV -> goal guide
    ax.plot([cx, gx], [cy, gy], color=GOAL_GREEN, lw=0.9,
            linestyle=(0, (4, 3)), zorder=4)

    # cross-stream +/-20 deg cone around +y (perpendicular to free stream)
    reach = 30.0
    for sign in (-1.0, 1.0):
        ang = np.deg2rad(90.0 + sign * CROSS_STREAM_CONE_DEG)
        ax.plot([cx, cx + reach * np.cos(ang)], [cy, cy + reach * np.sin(ang)],
                color=GOAL_GREEN, lw=0.7, linestyle=(0, (2, 2)), alpha=0.7, zorder=3)
    ax.text(cx - 2.0, cy + reach + 1.0, "cross-stream  ±20°", ha="center",
            va="bottom", fontsize=7.0, color=GOAL_GREEN, zorder=7)

    # AUV body (white, dark edge — legible over the blue/red field)
    ax.add_patch(Ellipse(xy=(cx, cy), width=14.0, height=4.5,
                         angle=AUV_HEADING_DEG, edgecolor=COLORS["ink"],
                         facecolor="white", linewidth=1.2, zorder=5))
    rad = np.deg2rad(AUV_HEADING_DEG)
    ax.annotate("", xy=(cx + 10.0 * np.cos(rad), cy + 10.0 * np.sin(rad)),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["ink"], lw=1.2,
                                mutation_scale=11), zorder=6)
    ax.text(cx - 13.0, cy - 5.0, "AUV", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=COLORS["ink"], zorder=7)

    # goal: star + 4 m success radius
    ax.add_patch(Circle((gx, gy), GOAL_RADIUS_M, facecolor="none",
                        edgecolor=GOAL_GREEN, linewidth=1.0,
                        linestyle=(0, (3, 2)), zorder=5))
    ax.scatter([gx], [gy], marker="*", s=150, c=GOAL_GREEN,
               edgecolor=COLORS["ink"], linewidth=0.6, zorder=6)
    ax.text(gx + 5.5, gy, "goal\n(4 m radius)", ha="left", va="center",
            fontsize=7.0, color=GOAL_GREEN, zorder=7)

    # free-stream arrow (top-left)
    x0, _ = PANEL_A_XLIM
    _, y1 = PANEL_A_YLIM
    ax.annotate("", xy=(x0 + 30.0, y1 - 7.0), xytext=(x0 + 5.0, y1 - 7.0),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["ink"], lw=1.2,
                                mutation_scale=11), zorder=8)
    ax.text(x0 + 5.0, y1 - 4.5, "free stream  1.0 m/s", ha="left", va="center",
            fontsize=7.5, color=COLORS["ink"], zorder=8)

    ax.set_xlim(*PANEL_A_XLIM)
    ax.set_ylim(*PANEL_A_YLIM)
    ax.set_xlabel("x  [m]", fontsize=8.5)
    ax.set_ylabel("y  [m]", fontsize=8.5)
    ax.tick_params(labelsize=7.5, width=0.75, length=2.8, colors=COLORS["ink"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(0.0, 1.03, "(a)  Kármán wake field and cross-stream task",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2,
            fontweight="bold", color=COLORS["ink"])

    cbar = plt.colorbar(im, ax=ax, fraction=0.0235, pad=0.015, aspect=22)
    cbar.set_label("vorticity  [1/s]", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.8, width=0.6)
    cbar.outline.set_linewidth(0.6)


# ---------------------------------------------------------------------------
# Panel (b)
# ---------------------------------------------------------------------------

def panel_b(ax: plt.Axes) -> None:
    L = 1.0
    ax.set_xlim(-1.05, 1.45)
    ax.set_ylim(-0.92, 0.66)

    # AUV body in body frame (to scale)
    ax.add_patch(Ellipse(xy=(0.0, 0.0), width=L, height=L * 0.12, angle=0.0,
                         edgecolor=COLORS["ink"], facecolor=COLORS["backbone_fill"],
                         linewidth=1.4, zorder=3))
    ax.annotate("", xy=(L * 0.72, 0.0), xytext=(L * 0.40, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], lw=1.2,
                                mutation_scale=10), zorder=4)
    ax.text(L * 0.78, 0.0, "body axis", ha="left", va="center",
            fontsize=7.0, color=COLORS["muted"], zorder=5)

    # s0: single-point DVL above the body (deployable)
    s0_y = 0.34
    ax.plot([0.0, 0.0], [0.03, s0_y - 0.03], color=COLORS["ink"], lw=0.9,
            linestyle=(0, (2, 2)), zorder=4)
    ax.scatter([0.0], [s0_y], marker="o", s=70, c=COLORS["ink"],
               edgecolor=COLORS["ink"], linewidth=1.0, zorder=6)
    ax.text(0.10, s0_y, "single-point  (deployable)", ha="left", va="center",
            fontsize=7.5, color=COLORS["ink"], fontweight="bold", zorder=7)

    # 5-point hull average below the body (privileged)
    samples_y = -0.40
    mc = COLORS["mainline_edge"]
    for xi in SAMPLE_XI:
        ax.plot([xi * L, xi * L], [-0.03, samples_y + 0.03], color=mc, lw=0.8,
                linestyle=(0, (1, 2)), zorder=4)
        ax.scatter([xi * L], [samples_y], marker="D", s=42, c="white",
                   edgecolor=mc, linewidth=1.1, zorder=6)
    # bracket spanning the 5 points
    by = samples_y - 0.13
    ax.plot([-0.4 * L, 0.4 * L], [by, by], color=mc, lw=1.1, zorder=5)
    for xend in (-0.4 * L, 0.4 * L):
        ax.plot([xend, xend], [by, by + 0.04], color=mc, lw=1.1, zorder=5)
    ax.text(0.0, by - 0.07, "5-point hull average  →  privileged flow",
            ha="center", va="top", fontsize=7.5, color=mc, fontweight="bold",
            zorder=7)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, 1.03, "(b)  Deployable vs. privileged sampling  (body frame)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2,
            fontweight="bold", color=COLORS["ink"])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_figure(width_mm: float = MAX_WIDTH_MM) -> plt.Figure:
    apply_style()
    width_in = in_from_mm(width_mm)
    height_in = in_from_mm(118.0)
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(width_in, height_in),
        gridspec_kw=dict(height_ratios=[1.0, 0.62]),
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.06, top=0.945, hspace=0.30)
    panel_a(ax_a)
    panel_b(ax_b)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 5 task/sensing schematic.")
    default_out = Path(__file__).resolve().parents[1] / "fig_ch5_sensor_schematic.pdf"
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--width-mm", type=float, default=MAX_WIDTH_MM)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(width_mm=args.width_mm)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved task/sensing schematic to {args.output}")


if __name__ == "__main__":
    main()
