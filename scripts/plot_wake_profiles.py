from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

_MPLCONFIGDIR = Path(".tmp/matplotlib")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR.resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator

try:
    from scripts.generate_wake import (
        LatticeConfig,
        PhysicsConfig,
        make_side_by_side_physics_config,
        make_tandem_physics_config,
        make_training_configs,
    )
except ModuleNotFoundError:
    from generate_wake import (  # type: ignore
        LatticeConfig,
        PhysicsConfig,
        make_side_by_side_physics_config,
        make_tandem_physics_config,
        make_training_configs,
    )


@dataclass(frozen=True)
class ProfileScene:
    key: str
    title: str
    subtitle: str
    config: PhysicsConfig


FONT_FAMILY = "DejaVu Sans"
INK = "#1F2933"
MUTED = "#5B6571"
BORDER = "#4A5565"
FLOW = "#5A7DA0"
ROI_FILL = "#DCE8F3"
ROI_EDGE = "#6E8FB1"
CYLINDER = "#2F2F2F"
DIMENSION = "#4B4B4B"
PANEL_LABELS = ("a", "b", "c")
MM_TO_INCH = 1.0 / 25.4


def _navigation_config() -> PhysicsConfig:
    return make_training_configs(
        profile="navigation",
        re_values=[150.0],
        u_values=[1.0],
    )[0]


def build_scenes() -> list[ProfileScene]:
    return [
        ProfileScene(
            key="navigation",
            title="Single cylinder",
            subtitle="Baseline wake configuration used for agent training",
            config=_navigation_config(),
        ),
        ProfileScene(
            key="tandem_G35_nav",
            title="Tandem cylinders",
            subtitle="Streamwise arrangement with gap ratio G/D = 3.5",
            config=make_tandem_physics_config(Re=150.0, U_phys=1.0),
        ),
        ProfileScene(
            key="side_by_side_G35_nav",
            title="Side-by-side cylinders",
            subtitle="Lateral arrangement with symmetric ROI and G/D = 3.5",
            config=make_side_by_side_physics_config(Re=150.0, U_phys=1.0),
        ),
    ]


def _apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _roi_bounds(pc: PhysicsConfig) -> tuple[float, float, float, float]:
    lc = LatticeConfig.from_physics(pc)
    return (
        lc.roi_x0 * pc.dx,
        lc.roi_x1 * pc.dx,
        lc.roi_y0 * pc.dx,
        lc.roi_y1 * pc.dx,
    )


def _cylinders(pc: PhysicsConfig) -> list[tuple[float, float, float]]:
    cylinders = [(pc.cyl_x_phys, pc.cyl_y_center + pc.cyl_y_jitter, pc.D_phys / 2.0)]
    for x, y, d in pc.extra_cylinders:
        cylinders.append((x, y, d / 2.0))
    return cylinders


def _flow_arrows(ax: plt.Axes, domain_x: float, domain_y: float) -> None:
    y_positions = np.linspace(0.25 * domain_y, 0.75 * domain_y, 3)
    for y in y_positions:
        ax.add_patch(
            FancyArrowPatch(
                (0.04 * domain_x, y),
                (0.15 * domain_x, y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.1,
                color=FLOW,
                zorder=2,
            )
        )
    ax.text(
        0.043 * domain_x,
        0.84 * domain_y,
        r"$U_\infty$",
        color=FLOW,
        fontsize=9,
        ha="left",
        va="bottom",
    )


def _annotate_geometry(ax: plt.Axes, scene: ProfileScene) -> None:
    pc = scene.config
    cylinders = _cylinders(pc)

    if scene.key == "navigation":
        cx, cy, r = cylinders[0]
        ax.text(
            cx,
            cy - 1.75 * r,
            "D = 12 m",
            ha="center",
            va="top",
            fontsize=8,
            color=MUTED,
            bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.9),
        )
        return

    if scene.key == "tandem_G35_nav":
        (x1, y1, r1), (x2, y2, r2) = cylinders
        gap_start = x1 + r1
        gap_end = x2 - r2
        y = y1 - 1.6 * r1
        ax.annotate(
            "",
            xy=(gap_start, y),
            xytext=(gap_end, y),
            arrowprops=dict(arrowstyle="<->", color=DIMENSION, linewidth=1.0),
        )
        ax.text(
            0.5 * (gap_start + gap_end),
            y - 8.0,
            "G/D = 3.5",
            ha="center",
            va="top",
            fontsize=8,
            color=DIMENSION,
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9),
        )
        return

    if scene.key == "side_by_side_G35_nav":
        (x1, y1, r1), (x2, y2, r2) = cylinders
        gap_low = y2 + r2
        gap_high = y1 - r1
        x = x1 + 0.95 * r1
        ax.annotate(
            "",
            xy=(x, gap_low),
            xytext=(x, gap_high),
            arrowprops=dict(arrowstyle="<->", color=DIMENSION, linewidth=1.0),
        )
        ax.text(
            x + 18.0,
            gap_low - 8.0,
            "G/D = 3.5",
            ha="left",
            va="top",
            fontsize=8,
            color=DIMENSION,
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9),
        )


def _draw_scene(ax: plt.Axes, scene: ProfileScene, panel_label: str, show_xlabel: bool) -> None:
    pc = scene.config
    domain_x = pc.Lx_phys
    domain_y = pc.Ly_phys
    roi_x0, roi_x1, roi_y0, roi_y1 = _roi_bounds(pc)
    roi_w = roi_x1 - roi_x0
    roi_h = roi_y1 - roi_y0

    domain = Rectangle(
        (0.0, 0.0),
        domain_x,
        domain_y,
        facecolor="white",
        edgecolor=BORDER,
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(domain)

    roi = Rectangle(
        (roi_x0, roi_y0),
        roi_w,
        roi_h,
        facecolor=ROI_FILL,
        edgecolor=ROI_EDGE,
        linewidth=1.0,
        alpha=0.65,
        hatch="////",
        zorder=1,
    )
    ax.add_patch(roi)

    ax.text(
        roi_x0 + 0.04 * roi_w,
        roi_y1 - 0.12 * roi_h,
        "ROI",
        ha="left",
        va="top",
        fontsize=8,
        color=ROI_EDGE,
        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
    )

    for x, y, r in _cylinders(pc):
        cylinder = Circle(
            (x, y),
            radius=r,
            facecolor=CYLINDER,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.add_patch(cylinder)

    _flow_arrows(ax, domain_x, domain_y)
    _annotate_geometry(ax, scene)

    ax.text(
        0.985 * domain_x,
        0.955 * domain_y,
        f"Domain {domain_x:.0f} × {domain_y:.0f} m\nROI {roi_w:.0f} × {roi_h:.0f} m",
        ha="right",
        va="top",
        fontsize=7.6,
        color=MUTED,
        linespacing=1.35,
        bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9),
    )

    ax.text(
        0.0,
        1.09,
        f"({panel_label}) {scene.title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.02 * domain_x, 1.02 * domain_x)
    ax.set_ylim(-0.04 * domain_y, 1.06 * domain_y)
    ax.set_xlabel("x (m)" if show_xlabel else "")
    ax.set_ylabel("y (m)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=INK, pad=2)
    ax.grid(False)


def build_figure() -> plt.Figure:
    _apply_publication_style()
    scenes = build_scenes()
    fig_width = 140.0 * MM_TO_INCH
    fig_height = 150.0 * MM_TO_INCH
    fig, axes = plt.subplots(len(scenes), 1, figsize=(fig_width, fig_height), constrained_layout=False)
    axes = np.atleast_1d(axes)
    for idx, (ax, scene) in enumerate(zip(axes, scenes, strict=True)):
        _draw_scene(ax, scene, PANEL_LABELS[idx], show_xlabel=idx == len(scenes) - 1)

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.11, right=0.985, top=0.98, bottom=0.065, hspace=0.46)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot wake-generation profile schematics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/wake_profile_schematics.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
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
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    if args.svg:
        fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved wake profile schematic to {args.output}")


if __name__ == "__main__":
    main()
