from __future__ import annotations

import argparse
import os
from pathlib import Path

_MPLCONFIGDIR = Path(".tmp/matplotlib")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from auv_nav.flow import make_probe_offsets


LAYOUT_TITLES = {
    "s0": "S0  Baseline DVL",
    "s1": "S1  Cross array",
    "s2": "S2  Forward ADCP",
}

PANEL_LABELS = {
    "s0": "(a)",
    "s1": "(b)",
    "s2": "(c)",
}

COLORS = {
    "ink": "#1F2933",
    "muted": "#52606D",
    "grid": "#E5E7EB",
    "guide": "#9AA5B1",
    "hull_fill": "#F5F7FA",
    "hull_edge": "#66788A",
    "probe_ref": "#1F2933",
    "probe_aux": "#5F6B76",
}

MM_PER_INCH = 25.4
DEFAULT_WIDTH_MM = 140.0


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _auv_outline() -> np.ndarray:
    """Return a streamlined AUV body silhouette in the body frame."""
    return np.array(
        [
            [-1.45, -0.18],
            [-1.25, -0.28],
            [-0.95, -0.34],
            [-0.25, -0.34],
            [0.25, -0.30],
            [0.58, -0.20],
            [0.85, -0.08],
            [1.00, 0.00],
            [0.85, 0.08],
            [0.58, 0.20],
            [0.25, 0.30],
            [-0.25, 0.34],
            [-0.95, 0.34],
            [-1.25, 0.28],
            [-1.45, 0.18],
        ],
        dtype=float,
    )


def _draw_auv(ax: plt.Axes, show_label: bool = False) -> None:
    hull = Polygon(
        _auv_outline(),
        closed=True,
        facecolor=COLORS["hull_fill"],
        edgecolor=COLORS["hull_edge"],
        linewidth=1.2,
        joinstyle="round",
        zorder=1,
    )
    ax.add_patch(hull)
    ax.plot(
        [-1.25, 0.75],
        [0.0, 0.0],
        color=COLORS["hull_edge"],
        linewidth=0.8,
        alpha=0.7,
        zorder=2,
    )
    ax.plot(
        [-1.33, -1.33],
        [-0.10, 0.10],
        color=COLORS["hull_edge"],
        linewidth=0.7,
        alpha=0.7,
        zorder=2,
    )

    if show_label:
        ax.annotate(
            "AUV hull",
            xy=(-0.55, -0.22),
            xytext=(-1.95, -1.00),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=7.5,
            color=COLORS["muted"],
            arrowprops=dict(
                arrowstyle="-",
                linewidth=0.75,
                color=COLORS["guide"],
                shrinkA=0.0,
                shrinkB=4.0,
            ),
            zorder=6,
        )


def _draw_reference_frame_icon(ax: plt.Axes) -> None:
    origin = (-2.55, -4.15)
    arrow_kw = dict(
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.85,
        color=COLORS["muted"],
        shrinkA=0.0,
        shrinkB=0.0,
    )
    ax.add_patch(FancyArrowPatch(origin, (-1.35, -4.15), **arrow_kw))
    ax.add_patch(FancyArrowPatch(origin, (-2.55, -2.95), **arrow_kw))
    ax.text(-1.22, -4.15, r"$x_b$", va="center", ha="left", color=COLORS["muted"])
    ax.text(-2.55, -2.80, r"$y_b$", va="bottom", ha="center", color=COLORS["muted"])


def _draw_probe(ax: plt.Axes, x: float, y: float, index: int, is_center: bool) -> None:
    color = COLORS["probe_ref"] if is_center else COLORS["probe_aux"]
    ax.add_patch(
        Circle(
            (x, y),
            radius=0.24,
            facecolor=color if is_center else "white",
            edgecolor=color,
            linewidth=1.2,
            zorder=4,
        )
    )
    if is_center:
        ax.add_patch(
            Circle(
                (x, y),
                radius=0.08,
                facecolor="white",
                edgecolor="white",
                linewidth=0.0,
                zorder=5,
            )
        )

    if index == 0:
        dx, dy, va = 0, 7, "bottom"
    elif y > 0.05:
        dx, dy, va = 0, 7, "bottom"
    elif y < -0.05:
        dx, dy, va = 0, -8, "top"
    else:
        dx, dy, va = 0, 7, "bottom"

    ax.annotate(
        f"p{index}",
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=7.5,
        color=color,
        zorder=6,
    )


def _draw_layout_guides(ax: plt.Axes, offsets: np.ndarray, layout: str) -> None:
    if layout == "s1":
        for x0, y0, x1, y1 in (
            (-2.0, 0.0, 2.0, 0.0),
            (0.0, -2.0, 0.0, 2.0),
        ):
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=COLORS["guide"],
                linewidth=0.8,
                linestyle=(0, (3, 2)),
                zorder=2,
            )
    elif layout == "s2":
        for x, y in offsets[1:]:
            ax.plot(
                [0.0, x],
                [0.0, y],
                color=COLORS["guide"],
                linewidth=0.8,
                linestyle=(0, (3, 2)),
                zorder=2,
            )


def _style_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-3.3, 10.6)
    ax.set_ylim(-5.2, 5.8)
    ax.set_xticks(np.arange(-2, 11, 2))
    ax.set_yticks(np.arange(-4, 5, 2))
    ax.grid(True, linestyle="-", linewidth=0.5, color=COLORS["grid"])
    ax.axhline(0.0, color=COLORS["grid"], linewidth=0.65, zorder=0)
    ax.axvline(0.0, color=COLORS["grid"], linewidth=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["ink"], direction="out")


def _add_panel_title(ax: plt.Axes, layout: str) -> None:
    ax.text(
        0.02,
        1.015,
        f"{PANEL_LABELS[layout]} {LAYOUT_TITLES[layout]}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def _draw_layout(ax: plt.Axes, layout: str) -> None:
    offsets = np.asarray(make_probe_offsets(layout), dtype=float)
    _draw_auv(ax, show_label=(layout == "s0"))
    _draw_layout_guides(ax, offsets, layout)
    _draw_reference_frame_icon(ax)

    for idx, (x, y) in enumerate(offsets):
        _draw_probe(ax, float(x), float(y), idx, is_center=(idx == 0))

    _style_axes(ax)
    _add_panel_title(ax, layout)


def build_figure(width_mm: float = DEFAULT_WIDTH_MM) -> plt.Figure:
    _apply_publication_style()
    width_in = width_mm / MM_PER_INCH
    height_in = width_in / 2.25
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(width_in, height_in),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.072, right=0.998, bottom=0.14, top=0.915, wspace=0.045)

    for ax, layout in zip(axes, ("s0", "s1", "s2"), strict=True):
        _draw_layout(ax, layout)

    fig.text(0.5, 0.08, r"Body-frame $x_b$ [m]", ha="center", va="center", fontsize=8.5)
    fig.text(
        0.022,
        0.53,
        r"Body-frame $y_b$ [m]",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.5,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AUV flow-probe layouts.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/auv_probe_layouts.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output DPI.",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also save an SVG next to the raster image.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save a PDF next to the raster image.",
    )
    parser.add_argument(
        "--width-mm",
        type=float,
        default=DEFAULT_WIDTH_MM,
        help="Figure width in millimetres.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(width_mm=args.width_mm)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    if args.svg:
        fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved probe layout figure to {args.output}")


if __name__ == "__main__":
    main()
