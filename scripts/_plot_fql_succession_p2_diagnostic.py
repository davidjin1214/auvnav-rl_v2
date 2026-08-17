"""Plot diagnostic training curves for FQL Succession P2 (3 cells × 2 algos × 2 seeds).

Reads ``results/fql_succession/p2/<cell>/training_curves/<algo>_seed<seed>/eval_log.csv``
and produces a 1×3 panel figure (one subplot per cell) overlaying eval_success_rate
vs train_step for all (algo, seed) combinations.

Also writes a CSV-style markdown table with the test-eval primary metric for the
diagnostic doc.

Output:
- docs/assets/fql_succession_p2/training_curves_3cells.png
- docs/assets/fql_succession_p2/test_eval_summary.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "fql_succession" / "p2"
OUT = ROOT / "docs" / "assets" / "fql_succession_p2"

CELLS = [
    ("e_uni", "E-uni  (clean priv, σ=0.0)"),
    ("m_uni_noise", "M-uni-noise  (priv + ε, σ=0.5)"),
    ("m_multi_mix", "M-multi-mix  (priv 50% + goalseek 50%, σ=0.1)"),
]
ALGOS = ["rebrac", "fql"]
SEEDS = [42, 0]

COLORS = {"rebrac": "tab:blue", "fql": "tab:orange"}
LINESTYLES = {42: "-", 0: "--"}


def read_eval_log(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    srs: list[float] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            steps.append(int(row["train_step"]))
            srs.append(float(row["eval_success_rate"]))
    return steps, srs


def read_test_sr(cell: str, algo: str, seed: int) -> float:
    f = RESULTS / cell / "test" / f"{algo}_seed{seed}.json"
    with f.open(encoding="utf-8") as fh:
        return float(json.load(fh)["eval_success_rate"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (cell, title) in zip(axes, CELLS, strict=True):
        for algo in ALGOS:
            for seed in SEEDS:
                p = RESULTS / cell / "training_curves" / f"{algo}_seed{seed}" / "eval_log.csv"
                steps, srs = read_eval_log(p)
                ax.plot(
                    steps,
                    srs,
                    color=COLORS[algo],
                    linestyle=LINESTYLES[seed],
                    linewidth=1.6,
                    alpha=0.9,
                    label=f"{algo} seed={seed}",
                )
                # mark test eval (100 ep) as a star at final step
                test_sr = read_test_sr(cell, algo, seed)
                ax.scatter(
                    [200_000],
                    [test_sr],
                    color=COLORS[algo],
                    marker="*",
                    s=110,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=10,
                )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("train step")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.02)
        ax.axhline(1.0, color="gray", linewidth=0.4, alpha=0.5)
    axes[0].set_ylabel("eval success rate")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "FQL Succession P2 — training-time eval (50 ep) + test eval (★, 100 ep)",
        fontsize=11,
    )
    fig.tight_layout()

    out_png = OUT / "training_curves_3cells.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

    # also dump a quick numeric summary
    summary_md = OUT / "test_eval_summary.md"
    lines = ["# FQL Succession P2 — test-eval primary metric", "", "n=2 seeds, 100 ep / seed, primary = `eval_success_rate`.", "", "| cell | rebrac s42 | rebrac s0 | rebrac μ | fql s42 | fql s0 | fql μ | δ=F−R | spec verdict |"]
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for cell, _ in CELLS:
        rs = [read_test_sr(cell, "rebrac", s) for s in SEEDS]
        fs = [read_test_sr(cell, "fql", s) for s in SEEDS]
        rmu = sum(rs) / 2
        fmu = sum(fs) / 2
        delta = fmu - rmu
        per_seed = [fs[i] - rs[i] for i in range(2)]
        signs = [1 if x > 0 else -1 for x in per_seed]
        consistent = signs[0] == signs[1]
        ad = abs(delta)
        if ad <= 0.03:
            verdict = "NULL"
        elif ad >= 0.05 and consistent:
            verdict = "POSITIVE" if delta > 0 else "NEGATIVE"
        elif 0.03 < ad < 0.05:
            verdict = "GRAY"
        else:
            verdict = "INCONSISTENT"
        lines.append(
            f"| {cell} | {rs[0]:.3f} | {rs[1]:.3f} | **{rmu:.3f}** | "
            f"{fs[0]:.3f} | {fs[1]:.3f} | **{fmu:.3f}** | "
            f"**{delta:+.3f}** | {verdict} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {summary_md}")


if __name__ == "__main__":
    main()
