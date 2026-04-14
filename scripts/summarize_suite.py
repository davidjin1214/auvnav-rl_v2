from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from auv_nav.reward import REWARD_OBJECTIVE_PRESETS


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_train_log_tail(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_line = None
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return None
    return json.loads(last_line)


def _looks_like_gain_label(value: str) -> bool:
    return value.startswith("e") and "_s" in value


def infer_runs_from_layout(suite_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for trainer_state in suite_root.glob("**/trainer_state.json"):
        run_dir = trainer_state.parent
        relative_parts = run_dir.relative_to(suite_root).parts
        if len(relative_parts) == 2:
            benchmark = None
            objective = None
            gain_label = None
            method, seed_part = relative_parts
        elif len(relative_parts) == 3:
            part0, part1, seed_part = relative_parts
            if part0 in REWARD_OBJECTIVE_PRESETS:
                benchmark = None
                objective = part0
                gain_label = None
                method = part1
            else:
                benchmark = part0
                objective = None
                gain_label = None
                method = part1
        elif len(relative_parts) == 4:
            part0, part1, part2, seed_part = relative_parts
            if part0 in REWARD_OBJECTIVE_PRESETS:
                benchmark = None
                objective = part0
                gain_label = part1 if _looks_like_gain_label(part1) else None
                method = part2 if gain_label is not None else part1
            elif _looks_like_gain_label(part1):
                benchmark = part0
                objective = None
                gain_label = part1
                method = part2
            else:
                benchmark = part0
                objective = part1
                gain_label = None
                method = part2
        elif len(relative_parts) == 5:
            benchmark, objective, gain_label, method, seed_part = relative_parts
        else:
            continue
        seed_text = seed_part.removeprefix("seed_")
        try:
            seed = int(seed_text)
        except ValueError:
            seed = -1
        runs.append(
            {
                "benchmark": benchmark,
                "objective": objective,
                "gain_label": gain_label,
                "method": method,
                "seed": seed,
                "run_dir": str(run_dir),
            }
        )
    return sorted(
        runs,
        key=lambda item: (
            (item["benchmark"] or ""),
            (item["objective"] or ""),
            (item["gain_label"] or ""),
            item["method"],
            item["seed"],
        ),
    )


def format_mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=float)
    return f"{np.mean(arr):.4f} +/- {np.std(arr):.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(payload: dict[str, Any], key: str) -> float:
    return float(payload.get(key, np.nan))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a benchmark-aware RL suite.")
    parser.add_argument(
        "--suite-root",
        type=str,
        required=True,
        help="Root directory containing benchmark/method/seed run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for summary CSV/JSON/Markdown files. Defaults to suite root.",
    )
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    output_dir = Path(args.output_dir) if args.output_dir is not None else suite_root
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = suite_root / "suite_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        runs = list(manifest.get("runs", []))
        benchmark_specs = {
            item["key"]: item for item in manifest.get("benchmarks", [])
            if item.get("key") is not None
        }
    else:
        manifest = None
        runs = infer_runs_from_layout(suite_root)
        benchmark_specs = {}

    detailed_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for run in runs:
        run_dir = Path(run["run_dir"])
        trainer_state_path = run_dir / "trainer_state.json"
        final_eval_path = run_dir / "final_eval.json"
        if not trainer_state_path.exists() or not final_eval_path.exists():
            continue

        trainer_state = load_json(trainer_state_path)
        final_eval = load_json(final_eval_path)
        train_tail = load_train_log_tail(run_dir / "train_log.jsonl") or {}
        benchmark_key = run.get("benchmark")
        benchmark_meta = benchmark_specs.get(benchmark_key, {})
        objective = str(
            run.get("objective")
            or trainer_state.get("reward_objective", "")
        )
        gain_label = str(run.get("gain_label") or "")
        reward_overrides = trainer_state.get("env_config_overrides", {})

        row = {
            "benchmark": benchmark_key or "",
            "objective": objective,
            "gain_label": gain_label,
            "method": run["method"],
            "seed": int(run.get("seed", -1)),
            "run_dir": str(run_dir),
            "algorithm": trainer_state.get("algorithm", ""),
            "reward_objective": trainer_state.get("reward_objective", ""),
            "energy_cost_gain": float(
                run.get("energy_cost_gain")
                if run.get("energy_cost_gain") is not None
                else final_eval.get("energy_cost_gain", reward_overrides.get("energy_cost_gain", np.nan))
            ),
            "safety_cost_gain": float(
                run.get("safety_cost_gain")
                if run.get("safety_cost_gain") is not None
                else final_eval.get("safety_cost_gain", reward_overrides.get("safety_cost_gain", np.nan))
            ),
            "env_step": int(trainer_state.get("env_step", 0)),
            "eval_return": _safe_float(final_eval, "eval_return"),
            "eval_cost": _safe_float(final_eval, "eval_cost"),
            "eval_safety_cost": _safe_float(final_eval, "eval_safety_cost"),
            "eval_success_rate": _safe_float(final_eval, "eval_success_rate"),
            "eval_time_s": _safe_float(final_eval, "eval_time_s"),
            "eval_energy": _safe_float(final_eval, "eval_energy"),
            "eval_path_length_m": _safe_float(final_eval, "eval_path_length_m"),
            "eval_progress_ratio": _safe_float(final_eval, "eval_progress_ratio"),
            "eval_path_efficiency": _safe_float(final_eval, "eval_path_efficiency"),
            "last_train_return": _safe_float(train_tail, "return"),
            "last_train_cost": _safe_float(train_tail, "episode_cost"),
            "last_train_success": _safe_float(train_tail, "success"),
            "flow_path": benchmark_meta.get("flow_path", trainer_state.get("flow_path", "")),
            "task_geometry": benchmark_meta.get(
                "task_geometry",
                trainer_state.get("reset_options", {}).get("task_geometry", ""),
            ),
            "target_speed": benchmark_meta.get(
                "target_speed",
                trainer_state.get("reset_options", {}).get("target_auv_max_speed_mps", np.nan),
            ),
        }
        detailed_rows.append(row)
        grouped[(row["benchmark"], row["objective"], row["gain_label"], row["method"])].append(row)

    detailed_rows.sort(
        key=lambda item: (
            item["benchmark"],
            item["objective"],
            item["gain_label"],
            item["method"],
            item["seed"],
        )
    )
    write_csv(output_dir / "ablation_runs.csv", detailed_rows)

    summary_rows: list[dict[str, Any]] = []
    for (benchmark, objective, gain_label, method), rows in sorted(grouped.items()):
        returns = [float(row["eval_return"]) for row in rows]
        costs = [float(row["eval_cost"]) for row in rows]
        safety_costs = [float(row["eval_safety_cost"]) for row in rows]
        success = [float(row["eval_success_rate"]) for row in rows]
        times = [float(row["eval_time_s"]) for row in rows]
        energy = [float(row["eval_energy"]) for row in rows]
        path_length = [float(row["eval_path_length_m"]) for row in rows]
        progress = [float(row["eval_progress_ratio"]) for row in rows]
        efficiency = [float(row["eval_path_efficiency"]) for row in rows]
        exemplar = rows[0]
        summary_rows.append(
            {
                "benchmark": benchmark,
                "objective": objective,
                "gain_label": gain_label,
                "method": method,
                "num_runs": len(rows),
                "reward_objective": exemplar["reward_objective"],
                "energy_cost_gain": exemplar["energy_cost_gain"],
                "safety_cost_gain": exemplar["safety_cost_gain"],
                "flow_path": exemplar["flow_path"],
                "task_geometry": exemplar["task_geometry"],
                "target_speed": exemplar["target_speed"],
                "eval_return_mean": float(np.mean(returns)),
                "eval_return_std": float(np.std(returns)),
                "eval_cost_mean": float(np.mean(costs)),
                "eval_cost_std": float(np.std(costs)),
                "eval_safety_cost_mean": float(np.mean(safety_costs)),
                "eval_safety_cost_std": float(np.std(safety_costs)),
                "eval_success_rate_mean": float(np.mean(success)),
                "eval_success_rate_std": float(np.std(success)),
                "eval_time_s_mean": float(np.mean(times)),
                "eval_time_s_std": float(np.std(times)),
                "eval_energy_mean": float(np.mean(energy)),
                "eval_energy_std": float(np.std(energy)),
                "eval_path_length_m_mean": float(np.mean(path_length)),
                "eval_path_length_m_std": float(np.std(path_length)),
                "eval_progress_ratio_mean": float(np.mean(progress)),
                "eval_progress_ratio_std": float(np.std(progress)),
                "eval_path_efficiency_mean": float(np.mean(efficiency)),
                "eval_path_efficiency_std": float(np.std(efficiency)),
            }
        )

    summary_rows.sort(
        key=lambda item: (
            item["benchmark"],
            item["objective"],
            item["gain_label"],
            -item["eval_success_rate_mean"],
            -item["eval_return_mean"],
            item["eval_cost_mean"],
        )
    )
    write_csv(output_dir / "ablation_summary.csv", summary_rows)

    report = {
        "suite_root": str(suite_root),
        "num_runs": len(detailed_rows),
        "summaries": summary_rows,
    }
    if manifest is not None:
        report["manifest"] = manifest
    with (output_dir / "ablation_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    markdown_lines = ["# Ablation Summary", ""]
    markdown_lines.append(f"Suite root: `{suite_root}`")
    markdown_lines.append(f"Runs summarized: {len(detailed_rows)}")
    markdown_lines.append("")
    markdown_lines.append(
        "| Benchmark | Objective | Gain | Method | N | Eval Return | Success | Time (s) | Energy | Path (m) | Progress | Efficiency |"
    )
    markdown_lines.append(
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in summary_rows:
        method_rows = grouped[(row["benchmark"], row["objective"], row["gain_label"], row["method"])]
        markdown_lines.append(
            "| "
            f"{row['benchmark'] or '-'} | "
            f"{row['objective'] or '-'} | "
            f"{row['gain_label'] or '-'} | "
            f"{row['method']} | "
            f"{row['num_runs']} | "
            f"{format_mean_std([float(item['eval_return']) for item in method_rows])} | "
            f"{format_mean_std([float(item['eval_success_rate']) for item in method_rows])} | "
            f"{format_mean_std([float(item['eval_time_s']) for item in method_rows])} | "
            f"{format_mean_std([float(item['eval_energy']) for item in method_rows])} | "
            f"{format_mean_std([float(item['eval_path_length_m']) for item in method_rows])} | "
            f"{format_mean_std([float(item['eval_progress_ratio']) for item in method_rows])} | "
            f"{format_mean_std([float(item['eval_path_efficiency']) for item in method_rows])} |"
        )
    markdown_lines.append("")
    with (output_dir / "ablation_summary.md").open("w", encoding="utf-8") as fp:
        fp.write("\n".join(markdown_lines) + "\n")

    print(f"wrote: {output_dir / 'ablation_runs.csv'}")
    print(f"wrote: {output_dir / 'ablation_summary.csv'}")
    print(f"wrote: {output_dir / 'ablation_summary.json'}")
    print(f"wrote: {output_dir / 'ablation_summary.md'}")


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
