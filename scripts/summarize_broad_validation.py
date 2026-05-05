"""Aggregate broad-validation P1/P2 results and apply the trigger gate.

Spec §3.1 anchor reference: success = 0.902 +/- 0.021.
Spec §6.1 trigger judgment:
  - |mean(spoke 2-seed) - 0.902| > 0.05  (bidirectional mean shift)
  - std(spoke 2-seed) > 2 * 0.021 = 0.042 (std blow-up)
  - For A2 only: |mean_ReBRAC - mean_TD3BC| < 0.05 (gap collapse)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ANCHOR_MEAN = 0.902
ANCHOR_STD = 0.021
MEAN_SHIFT_THRESHOLD = 0.05
STD_BLOW_UP_THRESHOLD = 2 * ANCHOR_STD
TD3BC_GAP_THRESHOLD = 0.05
WINNER_DRIFT_THRESHOLD = 0.03  # Spec §6.2


def _safe_mean(values: list[float]) -> float | None:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(cleaned) if cleaned else None


def _safe_std(values: list[float]) -> float:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0


def collect_p1_results(broad_root: Path) -> list[dict[str, Any]]:
    """Walk results/offline/rebrac/broad_validation/<spoke>/<pair>/test/seed_*.json."""
    rows: list[dict[str, Any]] = []
    if not broad_root.exists():
        return rows
    for spoke_dir in sorted(p for p in broad_root.iterdir() if p.is_dir()):
        if spoke_dir.name == "summaries":
            continue
        for pair_dir in sorted(p for p in spoke_dir.iterdir() if p.is_dir()):
            test_dir = pair_dir / "test"
            if not test_dir.is_dir():
                continue
            for json_path in sorted(test_dir.glob("seed_*.json")):
                metrics = json.loads(json_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "spoke_id": spoke_dir.name,
                        "pair": pair_dir.name,
                        "seed": int(json_path.stem.replace("seed_", "")),
                        "eval_success_rate": float(metrics["eval_success_rate"]),
                        "eval_return": float(metrics["eval_return"]),
                        "eval_safety_cost": float(metrics["eval_safety_cost"]),
                        "eval_time_s": float(metrics["eval_time_s"]),
                    }
                )
    return rows


def summarize_p1(broad_root: Path) -> list[dict[str, Any]]:
    rows = collect_p1_results(broad_root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["spoke_id"], row["pair"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (spoke_id, pair), seeds in sorted(grouped.items()):
        success = [r["eval_success_rate"] for r in seeds]
        returns = [r["eval_return"] for r in seeds]
        out.append(
            {
                "spoke_id": spoke_id,
                "pair": pair,
                "num_seeds": len(seeds),
                "seeds": sorted(r["seed"] for r in seeds),
                "mean_test_success_rate": _safe_mean(success),
                "std_test_success_rate": _safe_std(success),
                "mean_test_return": _safe_mean(returns),
                "std_test_return": _safe_std(returns),
            }
        )
    return out


def apply_trigger_gate(p1_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per spoke, decide whether P2 deepening is required.

    Returns a list of decisions, one per spoke (A2-td3bc folded into A2).
    """
    by_spoke: dict[str, list[dict[str, Any]]] = {}
    for row in p1_summary:
        by_spoke.setdefault(row["spoke_id"], []).append(row)

    decisions: list[dict[str, Any]] = []
    rebrac_spokes = sorted(s for s in by_spoke if s != "A2-td3bc")
    for spoke_id in rebrac_spokes:
        # The spoke's primary anchor pair is the one with the most seeds.
        rebrac_rows = sorted(
            by_spoke[spoke_id], key=lambda r: r["num_seeds"], reverse=True
        )
        primary = rebrac_rows[0]
        mean_val = primary["mean_test_success_rate"] or 0.0
        std_val = primary["std_test_success_rate"] or 0.0
        delta_pp = mean_val - ANCHOR_MEAN
        reasons: list[str] = []
        if abs(delta_pp) > MEAN_SHIFT_THRESHOLD:
            reasons.append("mean_shift")
        if std_val > STD_BLOW_UP_THRESHOLD:
            reasons.append("std_blow_up")
        if spoke_id == "A2" and "A2-td3bc" in by_spoke:
            td3bc_primary = sorted(
                by_spoke["A2-td3bc"],
                key=lambda r: r["num_seeds"],
                reverse=True,
            )[0]
            td3bc_mean = td3bc_primary["mean_test_success_rate"] or 0.0
            if abs(mean_val - td3bc_mean) < TD3BC_GAP_THRESHOLD:
                reasons.append("td3bc_gap_collapse")
        decisions.append(
            {
                "spoke_id": spoke_id,
                "pair": primary["pair"],
                "num_seeds": primary["num_seeds"],
                "mean_test_success_rate": mean_val,
                "std_test_success_rate": std_val,
                "delta_pp": delta_pp,
                "triggered": bool(reasons),
                "reasons": reasons,
            }
        )
    return decisions


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = {
                k: (",".join(str(v) for v in val) if isinstance(val, list) else val)
                for k, val in row.items()
            }
            writer.writerow(row_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize broad validation.")
    parser.add_argument(
        "--broad-root",
        type=Path,
        default=Path("results/offline/rebrac/broad_validation"),
    )
    parser.add_argument(
        "--summaries-dir",
        type=Path,
        default=None,
        help="Override summaries output dir (defaults to <broad-root>/summaries).",
    )
    args = parser.parse_args()

    summaries_dir = args.summaries_dir or (args.broad_root / "summaries")
    summaries_dir.mkdir(parents=True, exist_ok=True)

    p1_summary = summarize_p1(args.broad_root)
    write_csv(p1_summary, summaries_dir / "p1_overview.csv")
    (summaries_dir / "p1_overview.json").write_text(
        json.dumps(p1_summary, indent=2), encoding="utf-8"
    )

    decisions = apply_trigger_gate(p1_summary)
    (summaries_dir / "trigger_decisions.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8"
    )

    triggered = [d for d in decisions if d["triggered"]]
    print(f"[summary] P1 spokes summarized: {len(p1_summary)}")
    print(f"[summary] triggered for P2: {len(triggered)}")
    for d in triggered:
        print(
            f"  - {d['spoke_id']}: mean={d['mean_test_success_rate']:.3f} "
            f"std={d['std_test_success_rate']:.3f} "
            f"delta_pp={d['delta_pp']:+.3f} "
            f"reasons={','.join(d['reasons'])}"
        )


if __name__ == "__main__":
    main()
