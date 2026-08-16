"""
paper/figures/scripts/fig3_q_drift.py

Figure 3: β2 ablation 下 mean_target_q 的跨 dataset 漂移方向。
4 根条形对应 (dataset, β2) ∈ {worldcomp, crosscomp} × {2, 0}；同 dataset 下 β2=2 与 β2=0
之间用箭头 + 数值连接，强调两个 dataset baseline Q 的符号相反 (worldcomp +15 / crosscomp -8)，
但移除 β2 后两者都朝 "更不保守" 方向漂 +7~+8 个绝对单位。

数字源 (rev.8)：
- worldcomp β2=2 winner:   docs/rebrac_experiment_report.md §7.10.4   (mean_target_q +15.22)
- worldcomp β2=0 probe:    docs/rebrac_experiment_report.md §7.13.3   (mean_target_q +22.30)
- crosscomp β2=2 winner:   docs/rebrac_experiment_report.md §7.7      (mean_target_q -8.25)
- crosscomp β2=0 stage E:  docs/rebrac_experiment_report.md §7.14.3   (mean_target_q -0.13)

使用方法：
    python paper/figures/scripts/fig3_q_drift.py
输出：
    paper/figures/output/fig3_q_drift.pdf  (与 fig3_q_drift.png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# (label_x, mean_target_q, color, hatch)
BARS: list[tuple[str, float, str, str | None]] = [
    ("worldcomp\n$\\beta_2{=}2$",  +15.22, "#1f77b4", None),
    ("worldcomp\n$\\beta_2{=}0$",  +22.30, "#1f77b4", "//"),
    ("crosscomp\n$\\beta_2{=}2$",   -8.25, "#d62728", None),
    ("crosscomp\n$\\beta_2{=}0$",   -0.13, "#d62728", "//"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    x = np.arange(len(BARS))
    heights = [b[1] for b in BARS]
    colors = [b[2] for b in BARS]
    hatches = [b[3] for b in BARS]

    bars = ax.bar(
        x,
        heights,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        width=0.55,
    )
    for bar, hatch in zip(bars, hatches):
        if hatch is not None:
            bar.set_hatch(hatch)

    for i, h in enumerate(heights):
        va = "bottom" if h >= 0 else "top"
        offset = 0.6 if h >= 0 else -0.6
        ax.text(x[i], h + offset, f"{h:+.2f}", ha="center", va=va, fontsize=9)

    arrow_kwargs = dict(arrowstyle="->", color="black", lw=1.2)

    ax.annotate(
        "",
        xy=(1, 22.30),
        xytext=(0, 15.22),
        arrowprops=arrow_kwargs,
    )
    ax.text(
        0.5,
        (15.22 + 22.30) / 2 + 1.0,
        "$+7.08$ ($+46\\%$)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )

    ax.annotate(
        "",
        xy=(3, -0.13),
        xytext=(2, -8.25),
        arrowprops=arrow_kwargs,
    )
    ax.text(
        2.5,
        (-8.25 - 0.13) / 2 + 1.0,
        "$+8.12$ ($+98\\%$)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )

    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in BARS], fontsize=9)
    ax.set_ylabel(r"$\hat{Q}_{\mathrm{target}}$ (mean target Q)")
    ax.set_ylim(-13, 28)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_title(
        r"$\beta_2$ ablation: $\hat{Q}_{\mathrm{target}}$ drifts to less-conservative direction"
        " on both datasets",
        fontsize=10,
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                      linewidth=0.7, label="$\\beta_2{=}2$ (winner)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                      linewidth=0.7, hatch="//", label="$\\beta_2{=}0$ (ablation)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=True)

    plt.tight_layout()

    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig3_q_drift.pdf"
    png_path = output_dir / "fig3_q_drift.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
