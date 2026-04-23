"""Compare deterministic and noisy-support offline TD3BC screening results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_dataset_diagnostics(path: Path) -> dict[int, dict[str, float | str | bool]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: dict[int, dict[str, float | str | bool]] = {}
        for row in reader:
            episodes = int(row["episodes"])
            rows[episodes] = {
                "dataset": row["dataset"],
                "best_alpha_tag": row["best_alpha_tag"],
                "best_alpha": float(row["best_alpha"]),
                "best_alpha_hits_boundary": row["best_alpha_hits_boundary"] == "True",
                "td3bc_test_success_rate": float(row["td3bc_test_success_rate"]),
                "td3bc_test_success_rate_std": float(row["td3bc_test_success_rate_std"]),
                "bc_test_success_rate": float(row["bc_test_success_rate"]),
                "bc_test_success_rate_std": float(row["bc_test_success_rate_std"]),
                "td3bc_test_return": float(row["td3bc_test_return"]),
                "td3bc_test_return_std": float(row["td3bc_test_return_std"]),
                "bc_test_return": float(row["bc_test_return"]),
                "bc_test_return_std": float(row["bc_test_return_std"]),
                "td3bc_minus_bc_success_rate": float(row["td3bc_minus_bc_success_rate"]),
                "td3bc_minus_baseline_success_rate": float(row["td3bc_minus_baseline_success_rate"]),
                "mean_selected_ckpt_frac": float(row["mean_selected_ckpt_frac"]),
            }
        return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_conclusions(rows: list[dict[str, object]]) -> list[str]:
    conclusions: list[str] = []
    by_episodes = {int(row["episodes"]): row for row in rows}

    row_1000 = by_episodes.get(1000)
    row_2000 = by_episodes.get(2000)
    if row_1000 is None or row_2000 is None:
        conclusions.append("comparison is incomplete: expected both 1000 and 2000 episode rows.")
        return conclusions

    delta_bc_1000 = float(row_1000["delta_bc_success"])
    delta_bc_2000 = float(row_2000["delta_bc_success"])
    delta_td3bc_1000 = float(row_1000["delta_td3bc_success"])
    delta_td3bc_2000 = float(row_2000["delta_td3bc_success"])

    conclusions.append(
        "1000 deterministic vs noisy TD3BC success delta: "
        f"{delta_td3bc_1000:+.3f}; BC success delta: {delta_bc_1000:+.3f}."
    )
    conclusions.append(
        "2000 deterministic vs noisy TD3BC success delta: "
        f"{delta_td3bc_2000:+.3f}; BC success delta: {delta_bc_2000:+.3f}."
    )

    if delta_bc_2000 >= 0.03:
        conclusions.append(
            "2000 noisy BC improves materially over deterministic BC; this supports a behavior-support explanation."
        )
    elif delta_td3bc_2000 >= 0.03 and delta_bc_2000 < 0.03:
        conclusions.append(
            "2000 noisy TD3BC improves while BC stays similar; this suggests support/Q interaction rather than pure cloning limits."
        )
    elif abs(delta_td3bc_2000) < 0.02 and abs(delta_bc_2000) < 0.02:
        conclusions.append(
            "2000 noisy and deterministic results are close for both TD3BC and BC; this weakens the narrow-support hypothesis."
        )
    else:
        conclusions.append(
            "2000 noisy-support effects are mixed; inspect per-seed variance before drawing a mechanistic conclusion."
        )

    noisy_2000 = float(row_2000["noisy_td3bc_success"])
    det_1000 = float(row_1000["deterministic_td3bc_success"])
    if noisy_2000 >= det_1000:
        conclusions.append(
            "2000 noisy TD3BC matches or exceeds 1000 deterministic TD3BC; support broadening may explain much of the original 1000 > 2000 gap."
        )
    else:
        conclusions.append(
            "2000 noisy TD3BC still does not reach 1000 deterministic TD3BC; support broadening alone is unlikely to fully explain the gap."
        )

    return conclusions


def analyze(det_results_root: Path, noisy_results_root: Path, output_dir: Path) -> None:
    det_path = det_results_root / "analysis" / "dataset_diagnostics.csv"
    noisy_path = noisy_results_root / "analysis" / "dataset_diagnostics.csv"
    if not det_path.exists():
        raise FileNotFoundError(f"Missing deterministic diagnostics: {det_path}")
    if not noisy_path.exists():
        raise FileNotFoundError(f"Missing noisy diagnostics: {noisy_path}")

    det_rows = _load_dataset_diagnostics(det_path)
    noisy_rows = _load_dataset_diagnostics(noisy_path)
    shared_episodes = sorted(set(det_rows) & set(noisy_rows))
    if not shared_episodes:
        raise ValueError("No shared episode counts found between deterministic and noisy results.")

    comparison_rows: list[dict[str, object]] = []
    for episodes in shared_episodes:
        det = det_rows[episodes]
        noisy = noisy_rows[episodes]
        comparison_rows.append(
            {
                "episodes": episodes,
                "deterministic_dataset": det["dataset"],
                "noisy_dataset": noisy["dataset"],
                "deterministic_best_alpha": det["best_alpha"],
                "noisy_best_alpha": noisy["best_alpha"],
                "deterministic_td3bc_success": det["td3bc_test_success_rate"],
                "noisy_td3bc_success": noisy["td3bc_test_success_rate"],
                "delta_td3bc_success": float(noisy["td3bc_test_success_rate"]) - float(det["td3bc_test_success_rate"]),
                "deterministic_bc_success": det["bc_test_success_rate"],
                "noisy_bc_success": noisy["bc_test_success_rate"],
                "delta_bc_success": float(noisy["bc_test_success_rate"]) - float(det["bc_test_success_rate"]),
                "deterministic_td3bc_return": det["td3bc_test_return"],
                "noisy_td3bc_return": noisy["td3bc_test_return"],
                "delta_td3bc_return": float(noisy["td3bc_test_return"]) - float(det["td3bc_test_return"]),
                "deterministic_bc_return": det["bc_test_return"],
                "noisy_bc_return": noisy["bc_test_return"],
                "delta_bc_return": float(noisy["bc_test_return"]) - float(det["bc_test_return"]),
                "deterministic_td3bc_minus_bc": det["td3bc_minus_bc_success_rate"],
                "noisy_td3bc_minus_bc": noisy["td3bc_minus_bc_success_rate"],
                "delta_td3bc_minus_bc": float(noisy["td3bc_minus_bc_success_rate"]) - float(det["td3bc_minus_bc_success_rate"]),
                "deterministic_ckpt_frac": det["mean_selected_ckpt_frac"],
                "noisy_ckpt_frac": noisy["mean_selected_ckpt_frac"],
                "delta_ckpt_frac": float(noisy["mean_selected_ckpt_frac"]) - float(det["mean_selected_ckpt_frac"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "noisy_support_comparison.csv"
    json_path = output_dir / "noisy_support_comparison.json"
    conclusions_path = output_dir / "conclusions.txt"

    _write_csv(csv_path, comparison_rows)
    json_path.write_text(json.dumps(comparison_rows, indent=2), encoding="utf-8")

    conclusions = _build_conclusions(comparison_rows)
    conclusions_path.write_text("\n".join(conclusions) + "\n", encoding="utf-8")

    print(f"[write] comparison csv: {csv_path}")
    print(f"[write] comparison json: {json_path}")
    print(f"[write] conclusions: {conclusions_path}")
    for line in conclusions:
        print(f"[conclusion] {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-results-root", type=Path, required=True)
    parser.add_argument("--noisy-results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(
        det_results_root=args.det_results_root,
        noisy_results_root=args.noisy_results_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
