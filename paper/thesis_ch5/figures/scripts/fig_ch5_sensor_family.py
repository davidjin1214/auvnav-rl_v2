"""
paper/thesis_ch5/figures/scripts/fig_ch5_sensor_family.py

Figure (§5.3.2): Sensing family s0 / s1 / s2 — body-frame probe geometry.

Three panels share the AUV body and axes; they differ only in probe layout:
    s0 — single-point DVL water-track at the hull origin (deployable main axis)
    s1 — s0 + one short-range forward ADCP cell at 4.5 m ahead
    s2 — s0 + a near forward cell at 5 m and two lateral cells at (8, +/-4) m

Probe coordinates mirror auv_nav/flow.py `make_probe_offsets` (body frame, m);
they are hard-coded here so the figure renders with pure matplotlib (no
auv_nav / gymnasium import). The panel titles fix a long-standing labelling
bug in scripts/plot_probe_layouts.py, where s1/s2 were mislabelled
("Cross array" / "Forward ADCP") against the chapter's semantics.

Design rules (shared with the chapter's method figure):
  - English words only, minimal text; sensing roles live in the caption.
  - Width <= 138 mm, vector PDF, shared _ch5_style palette/fonts.
  - NO result numbers (spec §0.4 red line).

Usage:
    python paper/thesis_ch5/figures/scripts/fig_ch5_sensor_family.py

Output:
    paper/thesis_ch5/figures/fig_ch5_sensor_family.{pdf,png}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ch5_style import (  # noqa: E402
    COLORS,
    MAX_WIDTH_MM,
    apply_style,
    in_from_mm,
)

# Body-frame probe offsets (m) — mirror auv_nav/flow.py make_probe_offsets.
PROBES = {
    "s0": np.array([[0.0, 0.0]]),
    "s1": np.array([[0.0, 0.0], [4.5, 0.0]]),
    "s2": np.array([[0.0, 0.0], [5.0, 0.0], [8.0, 4.0], [8.0, -4.0]]),
}
TITLES = {
    "s0": "S0  ·  single-point DVL",
    "s1": "S1  ·  + forward ADCP",
    "s2": "S2  ·  + long-range ADCP",
}
PANEL = {"s0": "(a)", "s1": "(b)", "s2": "(c)"}


def _auv_outline() -> np.ndarray:
    """Streamlined AUV body silhouette in the body frame (bow at +x)."""
    return np.array(
        [
            [-1.45, -0.18], [-1.25, -0.28], [-0.95, -0.34], [-0.25, -0.34],
            [0.25, -0.30], [0.58, -0.20], [0.85, -0.08], [1.00, 0.00],
            [0.85, 0.08], [0.58, 0.20], [0.25, 0.30], [-0.25, 0.34],
            [-0.95, 0.34], [-1.25, 0.28], [-1.45, 0.18],
        ],
        dtype=float,
    )


def _draw_auv(ax: plt.Axes) -> None:
    ax.add_patch(
        Polygon(_auv_outline(), closed=True, facecolor=COLORS["backbone_fill"],
                edgecolor=COLORS["backbone_edge"], linewidth=1.1,
                joinstyle="round", zorder=1)
    )
    ax.plot([-1.2, 0.75], [0.0, 0.0], color=COLORS["backbone_edge"],
            linewidth=0.7, alpha=0.7, zorder=2)


def _draw_guides(ax: plt.Axes, offsets: np.ndarray) -> None:
    """Dashed guide from the hull origin to each auxiliary probe."""
    for x, y in offsets[1:]:
        ax.plot([0.0, x], [0.0, y], color=COLORS["grid"], linewidth=0.8,
                linestyle=(0, (3, 2)), zorder=1.5)


def _draw_probe(ax: plt.Axes, x: float, y: float, idx: int) -> None:
    is_center = idx == 0
    color = COLORS["ink"] if is_center else COLORS["offline_edge"]
    ax.add_patch(
        Circle((x, y), radius=0.30,
               facecolor=color if is_center else "white",
               edgecolor=color, linewidth=1.2, zorder=4)
    )
    if is_center:
        ax.add_patch(Circle((x, y), radius=0.10, facecolor="white",
                            edgecolor="white", linewidth=0.0, zorder=5))
    dy, va = (9, "bottom") if y >= -0.05 else (-10, "top")
    ax.annotate(f"p{idx}", (x, y), xytext=(0, dy), textcoords="offset points",
                ha="center", va=va, fontsize=7.0, color=color, zorder=6)


def _style_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 10.8)
    ax.set_ylim(-5.4, 5.8)
    ax.set_xticks(np.arange(-2, 11, 2))
    ax.set_yticks(np.arange(-4, 5, 2))
    ax.grid(True, linestyle="-", linewidth=0.5, color=COLORS["grid"])
    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.6, zorder=0)
    ax.axvline(0.0, color=COLORS["grid"], linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out", length=2.8, width=0.75)


def _add_title(ax: plt.Axes, layout: str) -> None:
    ax.text(0.02, 1.02, f"{PANEL[layout]}  {TITLES[layout]}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.2, fontweight="bold", color=COLORS["ink"], clip_on=False)


def draw_layout(ax: plt.Axes, layout: str) -> None:
    offsets = PROBES[layout]
    _draw_auv(ax)
    _draw_guides(ax, offsets)
    for idx, (x, y) in enumerate(offsets):
        _draw_probe(ax, float(x), float(y), idx)
    _style_axes(ax)
    _add_title(ax, layout)


def build_figure(width_mm: float = MAX_WIDTH_MM) -> plt.Figure:
    apply_style()
    width_in = in_from_mm(width_mm)
    height_in = width_in / 2.32
    fig, axes = plt.subplots(1, 3, figsize=(width_in, height_in),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.165, top=0.90, wspace=0.06)
    for ax, layout in zip(axes, ("s0", "s1", "s2")):
        draw_layout(ax, layout)
    fig.text(0.535, 0.045, "Body-frame x  [m]", ha="center", va="center",
             fontsize=8.5)
    fig.text(0.016, 0.535, "Body-frame y  [m]", ha="center", va="center",
             rotation=90, fontsize=8.5)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 5 sensing-family figure.")
    default_out = Path(__file__).resolve().parents[1] / "fig_ch5_sensor_family.pdf"
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--width-mm", type=float, default=MAX_WIDTH_MM)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(width_mm=args.width_mm)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sensing-family figure to {args.output}")


if __name__ == "__main__":
    main()
