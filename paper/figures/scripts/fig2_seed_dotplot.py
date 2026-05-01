"""
paper/figures/scripts/fig2_seed_dotplot.py

Figure 2: seed-level success rate dot plot for worldcomp-1000.
对照 3 个协议（TD3+BC privileged-critic / ReBRAC-Q deployable / ReBRAC-Q privileged-critic）
在 5 个 random seeds (42–46) 上的 success rate；同 seed 之间用细虚线连接。

数字源 (rev.8)：
- TD3+BC privileged α=0.1: docs/td3bc_phase0c_*  (5-seed mean 0.922 ± 0.086)
- ReBRAC-Q deployable: docs/rebrac_experiment_report.md §7.10.4  (Stage D Phase 2 5-seed)
- ReBRAC-Q privileged-critic: docs/rebrac_experiment_report.md §7.12.4  (Stage D Phase 2 priv 5-seed)

使用方法：
    python paper/figures/scripts/fig2_seed_dotplot.py
输出：
    paper/figures/output/fig2_seed_dotplot.pdf  (与 fig2_seed_dotplot.png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS: list[int] = [42, 43, 44, 45, 46]


PROTOCOLS: dict[str, dict] = {
    "TD3+BC priv ($\\alpha{=}0.1$)": {
        "values": [0.97, 0.98, 0.91, 0.99, 0.76],
        "color": "#7f7f7f",
        "marker": "^",
        "size": 80,
        "x_offset": -0.18,
    },
    "ReBRAC-Q deployable (ours)": {
        "values": [0.99, 0.93, 0.78, 0.98, 0.96],
        "color": "#1f77b4",
        "marker": "o",
        "size": 80,
        "x_offset": 0.0,
    },
    "ReBRAC-Q priv (ours)": {
        "values": [0.96, 0.92, 0.90, 0.93, 0.96],
        "color": "#d62728",
        "marker": "s",
        "size": 80,
        "x_offset": 0.18,
    },
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    seed_means_dep = np.array(PROTOCOLS["ReBRAC-Q deployable (ours)"]["values"])
    seed_means_priv = np.array(PROTOCOLS["ReBRAC-Q priv (ours)"]["values"])

    for seed_idx, _ in enumerate(SEEDS):
        ax.plot(
            [seed_idx + PROTOCOLS["ReBRAC-Q deployable (ours)"]["x_offset"],
             seed_idx + PROTOCOLS["ReBRAC-Q priv (ours)"]["x_offset"]],
            [seed_means_dep[seed_idx], seed_means_priv[seed_idx]],
            color="#999999",
            linestyle=":",
            linewidth=0.9,
            zorder=1,
        )

    for protocol_name, spec in PROTOCOLS.items():
        x_positions = np.arange(len(SEEDS)) + spec["x_offset"]
        ax.scatter(
            x_positions,
            spec["values"],
            color=spec["color"],
            marker=spec["marker"],
            s=spec["size"],
            edgecolor="black",
            linewidth=0.6,
            label=protocol_name,
            zorder=3,
        )

    seed_44_idx = SEEDS.index(44)
    ax.annotate(
        "seed 44 rescued by\nprivileged critic (+12pp)",
        xy=(seed_44_idx + PROTOCOLS["ReBRAC-Q priv (ours)"]["x_offset"], 0.90),
        xytext=(seed_44_idx + 0.55, 0.84),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.7),
    )

    teacher_y = 0.99
    ax.axhline(teacher_y, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(
        len(SEEDS) - 0.4,
        teacher_y + 0.005,
        "online teacher = 0.99",
        fontsize=8,
        color="#2ca02c",
        ha="right",
        va="bottom",
    )

    ax.set_xticks(np.arange(len(SEEDS)))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("Success rate (test=100 episodes)")
    ax.set_xlabel("Random seed")
    ax.set_ylim(0.70, 1.02)
    ax.set_xlim(-0.6, len(SEEDS) - 0.4)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.9)
    ax.set_title("worldcomp-1000: seed-level dispersion across 3 protocols", fontsize=11)

    plt.tight_layout()

    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig2_seed_dotplot.pdf"
    png_path = output_dir / "fig2_seed_dotplot.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"saved: {pdf_path}")
    print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
