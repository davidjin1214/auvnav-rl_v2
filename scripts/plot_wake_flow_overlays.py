from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

_MPLCONFIGDIR = Path(".tmp/matplotlib")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator

from scripts.generate_wake import make_side_by_side_physics_config, make_tandem_physics_config, make_training_configs


WAKE_DIR = Path("wake_data")

MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 140.0

PANEL_LABEL_FS = 8.0
PANEL_TITLE_FS = 7.1
AXIS_LABEL_FS = 8.0
TICK_FS = 7.0
INSET_LABEL_FS = 6.3
CBAR_LABEL_FS = 7.9
MIN_PANEL_HEIGHT_IN = 1.70

TEXT_COLOR = "#202124"
MUTED_TEXT_COLOR = "#5F6C78"
SPINE_COLOR = "#A8B0B8"
STREAMLINE_COLOR = (0.17, 0.18, 0.19, 0.30)
FLOW_ARROW_COLOR = "#4E7DB2"
CYLINDER_FACE = "#2F2F31"
CYLINDER_EDGE = "#111214"
ROI_EDGE = "#D8C8AF"
INSET_EDGE = "#B8C5D3"
INSET_BG = (1.0, 1.0, 1.0, 0.94)
PANEL_BG = "#FFFFFF"
FIELD_ALPHA = 0.96

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "axes.linewidth": 0.8,
    }
)


def _paper_diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "paper_blue_orange",
        [
            (0.00, "#2A5C91"),
            (0.18, "#6FA0CA"),
            (0.48, "#F7F6F2"),
            (0.82, "#F2B38F"),
            (1.00, "#B24745"),
        ],
        N=256,
    )


PAPER_CMAP = _paper_diverging_cmap()
SCENE_EXPORT_STEMS = {
    "navigation": "wake_profile_single_cylinder",
    "tandem_G35_nav": "wake_profile_tandem_cylinders",
    "side_by_side_G35_nav": "wake_profile_side_by_side_cylinders",
}
SCENE_STREAM_DENSITY = {
    "navigation": 0.34,
    "tandem_G35_nav": 0.38,
    "side_by_side_G35_nav": 0.34,
}


@dataclass(frozen=True)
class FlowScene:
    key: str
    title: str
    repo_name: str
    data_path: Path
    meta_path: Path
    config: object


def _navigation_config():
    return make_training_configs(profile="navigation", re_values=[150.0], u_values=[1.0])[0]


def build_scenes() -> list[FlowScene]:
    return [
        FlowScene(
            key="navigation",
            title="Single cylinder",
            repo_name="navigation",
            data_path=WAKE_DIR / "wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
            meta_path=WAKE_DIR / "wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi_meta.json",
            config=_navigation_config(),
        ),
        FlowScene(
            key="tandem_G35_nav",
            title="Tandem cylinders",
            repo_name="tandem_G35_nav",
            data_path=WAKE_DIR / "wake_tandem_G35_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
            meta_path=WAKE_DIR / "wake_tandem_G35_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi_meta.json",
            config=make_tandem_physics_config(Re=150.0, U_phys=1.0),
        ),
        FlowScene(
            key="side_by_side_G35_nav",
            title="Side-by-side cylinders",
            repo_name="side_by_side_G35_nav",
            data_path=WAKE_DIR / "wake_sbs_G35_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
            meta_path=WAKE_DIR / "wake_sbs_G35_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi_meta.json",
            config=make_side_by_side_physics_config(Re=150.0, U_phys=1.0),
        ),
    ]


def _load_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _representative_frame(flow: np.ndarray) -> int:
    omega = flow[..., 2].astype(np.float32)
    rms = np.sqrt(np.mean(omega * omega, axis=(1, 2)))
    return int(np.argmax(rms))


def _cylinders(pc) -> list[tuple[float, float, float]]:
    cylinders = [(pc.cyl_x_phys, pc.cyl_y_center + pc.cyl_y_jitter, pc.D_phys / 2.0)]
    for x, y, d in pc.extra_cylinders:
        cylinders.append((x, y, d / 2.0))
    return cylinders


def _plot_limits(meta: dict, pc) -> tuple[float, float, float, float]:
    cylinders = _cylinders(pc)
    x_min = min(meta["roi_x0_phys_m"], min(x - r for x, _, r in cylinders)) - 18.0
    x_max = meta["roi_x1_phys_m"] + 12.0
    y_min = min(meta["roi_y0_phys_m"], min(y - r for _, y, r in cylinders)) - 14.0
    y_max = max(meta["roi_y1_phys_m"], max(y + r for _, y, r in cylinders)) + 14.0
    return x_min, x_max, y_min, y_max


def _focus_limits(meta: dict, pc) -> tuple[float, float, float, float]:
    x0 = float(meta["roi_x0_phys_m"])
    x1 = float(meta["roi_x1_phys_m"])
    y0 = float(meta["roi_y0_phys_m"])
    y1 = float(meta["roi_y1_phys_m"])

    cylinders = _cylinders(pc)
    cyl_x_min = min(x - r for x, _, r in cylinders)
    cyl_y_min = min(y - r for _, y, r in cylinders)
    cyl_y_max = max(y + r for _, y, r in cylinders)
    diameter = float(pc.D_phys)

    left = min(cyl_x_min - 1.0 * diameter, x0 - 0.35 * diameter)
    right = x1 + 0.45 * diameter
    bottom = min(y0 - 0.20 * diameter, cyl_y_min - 1.0 * diameter)
    top = max(y1 + 0.20 * diameter, cyl_y_max + 1.0 * diameter)
    return left, right, bottom, top


def _global_x_limits(scenes: list[FlowScene]) -> tuple[float, float]:
    limits = []
    for scene in scenes:
        meta = _load_meta(scene.meta_path)
        limits.append(_focus_limits(meta, scene.config))
    x_min = min(v[0] for v in limits)
    x_max = max(v[1] for v in limits)
    return x_min, x_max


def _add_flow_annotation(ax: plt.Axes, plot_x0: float, plot_x1: float, y0: float, y1: float) -> None:
    span_x = plot_x1 - plot_x0
    span_y = y1 - y0
    x_start = plot_x0 + 0.055 * span_x
    x_end = plot_x0 + 0.130 * span_x
    y = y0 + 0.845 * span_y
    ax.add_patch(
        FancyArrowPatch(
            (x_start, y),
            (x_end, y),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=FLOW_ARROW_COLOR,
            alpha=0.78,
            zorder=6,
        )
    )
    ax.text(
        0.5 * (x_start + x_end),
        y + 0.035 * span_y,
        r"$U_\infty$",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=FLOW_ARROW_COLOR,
        zorder=7,
    )


def _draw_inset(ax: plt.Axes, scene: FlowScene, meta: dict) -> None:
    pc = scene.config
    inset = ax.inset_axes([0.76, 0.64, 0.22, 0.28], zorder=9)
    inset.set_facecolor(INSET_BG)

    inset.add_patch(
        Rectangle(
            (0.0, 0.0),
            pc.Lx_phys,
            pc.Ly_phys,
            facecolor="none",
            edgecolor=INSET_EDGE,
            linewidth=0.8,
            zorder=0,
        )
    )
    inset.add_patch(
        Rectangle(
            (meta["roi_x0_phys_m"], meta["roi_y0_phys_m"]),
            meta["roi_x1_phys_m"] - meta["roi_x0_phys_m"],
            meta["roi_y1_phys_m"] - meta["roi_y0_phys_m"],
            facecolor="#E6D3B7",
            edgecolor=ROI_EDGE,
            linewidth=0.65,
            alpha=0.18,
            zorder=1,
        )
    )
    for x, y, r in _cylinders(pc):
        inset.add_patch(
            Circle(
                (x, y),
                radius=r,
                facecolor=CYLINDER_FACE,
                edgecolor=CYLINDER_EDGE,
                linewidth=0.55,
                zorder=2,
            )
        )

    inset.set_xlim(-0.02 * pc.Lx_phys, 1.02 * pc.Lx_phys)
    inset.set_ylim(-0.03 * pc.Ly_phys, 1.03 * pc.Ly_phys)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor(INSET_EDGE)
        spine.set_linewidth(0.7)
    inset.text(
        0.06,
        0.93,
        "Full domain",
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=INSET_LABEL_FS,
        color=MUTED_TEXT_COLOR,
    )


def _draw_scene(
    ax: plt.Axes,
    scene: FlowScene,
    norm: TwoSlopeNorm,
    panel_label: str = "",
    plot_limits: tuple[float, float, float, float] | None = None,
    show_flow_annotation: bool = False,
) -> None:
    flow = np.load(scene.data_path, mmap_mode="r")
    meta = _load_meta(scene.meta_path)
    frame_idx = _representative_frame(flow)
    frame = np.asarray(flow[frame_idx], dtype=np.float32)

    dx = float(meta["dx_m"])
    x0 = float(meta["roi_x0_phys_m"])
    x1 = float(meta["roi_x1_phys_m"])
    y0 = float(meta["roi_y0_phys_m"])
    y1 = float(meta["roi_y1_phys_m"])
    extent = (x0, x1, y0, y1)

    if plot_limits is None:
        plot_x0, plot_x1, plot_y0, plot_y1 = _focus_limits(meta, scene.config)
    else:
        plot_x0, plot_x1, plot_y0, plot_y1 = plot_limits
    ax.set_facecolor(PANEL_BG)

    img = ax.imshow(
        frame[:, :, 2].T,
        origin="lower",
        extent=extent,
        cmap=PAPER_CMAP,
        norm=norm,
        interpolation="bilinear",
        alpha=FIELD_ALPHA,
        zorder=1,
    )

    x_coords = x0 + dx * np.arange(frame.shape[0], dtype=float)
    y_coords = y0 + dx * np.arange(frame.shape[1], dtype=float)
    step_x = max(1, frame.shape[0] // 70)
    step_y = max(1, frame.shape[1] // 36)
    xs = x_coords[::step_x]
    ys = y_coords[::step_y]
    u = frame[::step_x, ::step_y, 0].T
    v = frame[::step_x, ::step_y, 1].T
    ax.streamplot(
        xs,
        ys,
        u,
        v,
        density=SCENE_STREAM_DENSITY.get(scene.key, 0.36),
        color=STREAMLINE_COLOR,
        linewidth=0.38,
        arrowsize=0.55,
        arrowstyle="-",
        minlength=0.20,
        maxlength=3.8,
        integration_direction="forward",
        broken_streamlines=False,
        zorder=4,
    )

    roi = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        facecolor="none",
        edgecolor=ROI_EDGE,
        linewidth=0.60,
        linestyle="-",
        alpha=0.65,
        zorder=7,
    )
    ax.add_patch(roi)

    for x, y, r in _cylinders(scene.config):
        ax.add_patch(
            Circle(
                (x, y),
                radius=r,
                facecolor=CYLINDER_FACE,
                edgecolor=CYLINDER_EDGE,
                linewidth=0.9,
                zorder=8,
            )
        )

    if show_flow_annotation:
        _add_flow_annotation(ax, plot_x0, plot_x1, plot_y0, plot_y1)
    _draw_inset(ax, scene, meta)

    title_x = 0.005
    if panel_label:
        ax.text(
            0.005,
            1.018,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_FS,
            weight="bold",
            color=TEXT_COLOR,
            zorder=10,
            clip_on=False,
        )
        title_x = 0.040
    ax.text(
        title_x,
        1.018,
        scene.title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_TITLE_FS,
        weight="regular",
        color=TEXT_COLOR,
        zorder=10,
        clip_on=False,
    )

    ax.set_xlim(plot_x0, plot_x1)
    ax.set_ylim(plot_y0, plot_y1)
    ax.grid(False)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.tick_params(direction="out", length=2.8, width=0.8, colors=TEXT_COLOR, labelsize=TICK_FS, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(SPINE_COLOR)
    return img


def _global_norm(scenes: list[FlowScene]) -> TwoSlopeNorm:
    max_abs = 0.0
    for scene in scenes:
        flow = np.load(scene.data_path, mmap_mode="r")
        idx = _representative_frame(flow)
        omega = np.asarray(flow[idx, :, :, 2], dtype=np.float32)
        vmax = float(np.percentile(np.abs(omega), 99.5))
        max_abs = max(max_abs, vmax)
    max_abs = max(max_abs, 1e-6)
    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def _scene_y_span(scene: FlowScene) -> float:
    meta = _load_meta(scene.meta_path)
    _, _, y0, y1 = _focus_limits(meta, scene.config)
    return y1 - y0


def _scene_plot_limits(scene: FlowScene) -> tuple[float, float, float, float]:
    meta = _load_meta(scene.meta_path)
    return _focus_limits(meta, scene.config)


def _single_panel_height(fig_width_in: float, plot_limits: tuple[float, float, float, float]) -> float:
    x0, x1, y0, y1 = plot_limits
    axis_width_in = fig_width_in * 0.79
    data_ratio = (y1 - y0) / max(x1 - x0, 1e-6)
    return max(MIN_PANEL_HEIGHT_IN, axis_width_in * data_ratio)


def _panel_height_from_axis_width(axis_width_in: float, plot_limits: tuple[float, float, float, float]) -> float:
    x0, x1, y0, y1 = plot_limits
    data_ratio = (y1 - y0) / max(x1 - x0, 1e-6)
    return max(MIN_PANEL_HEIGHT_IN, axis_width_in * data_ratio)


def build_single_figure(scene: FlowScene, width_mm: float = FIGURE_WIDTH_MM) -> plt.Figure:
    all_scenes = build_scenes()
    norm = _global_norm(all_scenes)
    plot_limits = _scene_plot_limits(scene)
    fig_width = width_mm / MM_PER_INCH
    fig_height = _single_panel_height(fig_width, plot_limits) + 0.32

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    ax = fig.add_gridspec(
        nrows=1,
        ncols=1,
        left=0.085,
        right=0.888,
        top=0.925,
        bottom=0.15,
    ).subplots()

    img = _draw_scene(
        ax,
        scene,
        norm,
        panel_label="",
        plot_limits=plot_limits,
        show_flow_annotation=True,
    )
    ax.set_xlabel("x (m)", fontsize=AXIS_LABEL_FS, color=TEXT_COLOR, labelpad=2)
    ax.set_ylabel("y (m)", fontsize=AXIS_LABEL_FS, color=TEXT_COLOR, labelpad=2)

    cbar = fig.colorbar(img, ax=ax, shrink=0.94, pad=0.014, aspect=34, fraction=0.040)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    cbar.set_label(r"Vorticity, $\omega$ (s$^{-1}$)", fontsize=CBAR_LABEL_FS, color=TEXT_COLOR)
    cbar.ax.tick_params(labelsize=TICK_FS, length=2.8, width=0.8, colors=TEXT_COLOR)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor(SPINE_COLOR)
    return fig


def build_composite_figure(width_mm: float = FIGURE_WIDTH_MM) -> plt.Figure:
    scenes = build_scenes()
    norm = _global_norm(scenes)

    fig_width = width_mm / MM_PER_INCH
    left = 0.10
    right = 0.885
    top = 0.965
    bottom = 0.075
    cbar_pad = 0.015
    cbar_width = 0.020
    panel_gap_in = 0.20

    axis_width_in = fig_width * (right - left)
    plot_limits = [_scene_plot_limits(scene) for scene in scenes]
    panel_heights_in = [_panel_height_from_axis_width(axis_width_in, limits) for limits in plot_limits]
    fig_height = top + bottom + sum(panel_heights_in) + panel_gap_in * (len(scenes) - 1)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)

    top_cursor = 1.0 - top / fig_height
    axes: list[plt.Axes] = []
    img = None
    for idx, (scene, limits, panel_height_in) in enumerate(zip(scenes, plot_limits, panel_heights_in)):
        panel_height = panel_height_in / fig_height
        bottom_pos = top_cursor - panel_height
        ax = fig.add_axes([left, bottom_pos, right - left, panel_height])
        img = _draw_scene(
            ax,
            scene,
            norm,
            panel_label=f"({chr(97 + idx)})",
            plot_limits=limits,
            show_flow_annotation=(idx == 0),
        )
        ax.set_ylabel("y (m)", fontsize=AXIS_LABEL_FS, color=TEXT_COLOR, labelpad=2)
        if idx == len(scenes) - 1:
            ax.set_xlabel("x (m)", fontsize=AXIS_LABEL_FS, color=TEXT_COLOR, labelpad=2)
        else:
            ax.set_xlabel("")
        axes.append(ax)
        top_cursor = bottom_pos - panel_gap_in / fig_height

    cbar_bottom = bottom / fig_height
    cbar_top = 1.0 - top / fig_height
    cax = fig.add_axes([right + cbar_pad, cbar_bottom, cbar_width, cbar_top - cbar_bottom])
    cbar = fig.colorbar(img, cax=cax)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    cbar.set_label(r"Vorticity, $\omega$ (s$^{-1}$)", fontsize=CBAR_LABEL_FS, color=TEXT_COLOR)
    cbar.ax.tick_params(labelsize=TICK_FS, length=2.8, width=0.8, colors=TEXT_COLOR)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor(SPINE_COLOR)
    return fig


def _save_figure(fig: plt.Figure, output: Path, dpi: int, save_pdf: bool, save_svg: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    if save_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def export_separate_figures(output_dir: Path, dpi: int, save_pdf: bool, save_svg: bool, width_mm: float) -> None:
    for scene in build_scenes():
        output = output_dir / f"{SCENE_EXPORT_STEMS[scene.key]}.png"
        fig = build_single_figure(scene, width_mm=width_mm)
        _save_figure(fig, output, dpi=dpi, save_pdf=save_pdf, save_svg=save_svg)


def export_composite_figure(output: Path, dpi: int, save_pdf: bool, save_svg: bool, width_mm: float) -> None:
    fig = build_composite_figure(width_mm=width_mm)
    _save_figure(fig, output, dpi=dpi, save_pdf=save_pdf, save_svg=save_svg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real wake snapshots with geometry overlays.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/wake_profile_real_flow.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster output DPI.",
    )
    parser.add_argument(
        "--width-mm",
        type=float,
        default=FIGURE_WIDTH_MM,
        help="Figure width in millimeters.",
    )
    parser.add_argument(
        "--scene",
        choices=("navigation", "tandem_G35_nav", "side_by_side_G35_nav"),
        help="Render a single scene only.",
    )
    parser.add_argument(
        "--export-separate-set",
        action="store_true",
        help="Export three separate figures, one for each scene.",
    )
    parser.add_argument(
        "--composite",
        action="store_true",
        help="Also export a three-panel composite figure to --output.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save a PDF next to the raster image.",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also save an SVG next to the raster image.",
    )
    args = parser.parse_args()

    if args.scene:
        scene = next(scene for scene in build_scenes() if scene.key == args.scene)
        fig = build_single_figure(scene, width_mm=args.width_mm)
        _save_figure(fig, args.output, dpi=args.dpi, save_pdf=args.pdf, save_svg=args.svg)
    else:
        if args.export_separate_set or not args.composite:
            export_separate_figures(
                output_dir=args.output.parent,
                dpi=args.dpi,
                save_pdf=args.pdf,
                save_svg=args.svg,
                width_mm=args.width_mm,
            )
        if args.composite:
            export_composite_figure(
                output=args.output,
                dpi=args.dpi,
                save_pdf=args.pdf,
                save_svg=args.svg,
                width_mm=args.width_mm,
            )
    print(f"Saved real-flow wake profile figure to {args.output}")


if __name__ == "__main__":
    main()
