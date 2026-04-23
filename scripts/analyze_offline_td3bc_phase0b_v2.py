from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _column_width(rows: list[dict[str, str]], key: str, header: str) -> int:
    return max([len(header), *(len(str(row.get(key, ""))) for row in rows)])


def _print_table(title: str, columns: list[tuple[str, str]], rows: list[dict[str, str]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("(empty)")
        return
    widths = {key: _column_width(rows, key, header) for key, header in columns}
    print("  ".join(header.ljust(widths[key]) for key, header in columns))
    print("  ".join("-" * widths[key] for key, _ in columns))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns))


def _parse_alpha_tag(alpha_tag: str) -> float:
    tag = alpha_tag[len("alpha_") :] if alpha_tag.startswith("alpha_") else alpha_tag
    try:
        return float(tag.replace("p", "."))
    except ValueError:
        return math.nan


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    mean_value = statistics.fmean(values)
    return mean_value, statistics.pstdev(values)


def _safe_sub(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs) - float(rhs)


def _monotonic_drop(values: list[float]) -> bool:
    return all(values[idx] >= values[idx + 1] for idx in range(len(values) - 1))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_dataset_baselines(results_root: Path, dataset_name: str) -> dict[str, dict[str, Any]]:
    baselines_dir = results_root / dataset_name / "baselines"
    baselines: dict[str, dict[str, Any]] = {}
    if not baselines_dir.exists():
        return baselines
    for path in sorted(baselines_dir.glob("*_test_eval.json")):
        baselines[path.name.removesuffix("_test_eval.json")] = _load_json(path)
    return baselines


def _select_reference_baseline(
    baselines: dict[str, dict[str, Any]],
    offline_metadata: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    if not baselines:
        return None, None
    if len(baselines) == 1:
        baseline_name = next(iter(baselines))
        return baseline_name, baselines[baseline_name]
    policy_name = str(offline_metadata.get("policy", "")) if offline_metadata else ""
    if policy_name and policy_name in baselines:
        return policy_name, baselines[policy_name]
    baseline_name = sorted(baselines)[0]
    return baseline_name, baselines[baseline_name]


def _mean_metric(records: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    values = [float(record[key]) for record in records if key in record and record[key] is not None]
    return _mean_std(values)


def _selected_run_total_steps(trainer_state: dict[str, Any]) -> int:
    train_cfg = _as_dict(trainer_state.get("train_config"))
    total_steps = trainer_state.get("train_step")
    if total_steps is None:
        total_steps = train_cfg.get("total_steps", 0)
    return int(total_steps or 0)


def run_bc_test_evals(
    *,
    results_root: Path,
    checkpoints_root: Path,
    test_manifest: Path,
    python_bin: str,
    device: str,
    num_workers: int,
    worker_device: str,
    test_seed: int,
    force_reeval: bool,
    dataset_filter: str | None,
) -> list[str]:
    if not test_manifest.exists():
        raise FileNotFoundError(f"Missing test manifest: {test_manifest}")

    outputs: list[str] = []
    dataset_dirs = [
        path
        for path in sorted(results_root.iterdir())
        if path.is_dir() and path.name not in {"summaries", "analysis"}
    ]
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        if dataset_filter and dataset_filter not in dataset_name:
            continue
        bc_selection_root = dataset_dir / "selection" / "alpha_0p0"
        if not bc_selection_root.exists():
            continue
        for selection_path in sorted(bc_selection_root.glob("seed_*/selected_checkpoint.json")):
            payload = _load_json(selection_path)
            best = _as_dict(payload.get("best"))
            if not best:
                continue
            seed_dir_name = selection_path.parent.name
            run_dir = checkpoints_root / dataset_name / "alpha_0p0" / seed_dir_name
            output_dir = dataset_dir / "test_bc_selected" / "alpha_0p0"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_json = output_dir / f"{seed_dir_name}.json"
            if output_json.exists() and not force_reeval:
                outputs.append(str(output_json))
                continue
            cmd = [
                python_bin,
                "-m",
                "scripts.evaluate_offline",
                "--checkpoint",
                str(run_dir),
                "--agent-file",
                str(best["agent_file"]),
                "--manifest",
                str(test_manifest),
                "--device",
                device,
                "--num-workers",
                str(max(1, int(num_workers))),
                "--worker-device",
                worker_device,
                "--seed",
                str(int(test_seed)),
                "--output-json",
                str(output_json),
            ]
            print(f"[run] {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            outputs.append(str(output_json))
    return outputs


def _collect_validation_curve_rows(
    *,
    dataset_name: str,
    results_root: Path,
    checkpoints_root: Path,
) -> list[dict[str, Any]]:
    validation_root = results_root / dataset_name / "validation"
    rows: list[dict[str, Any]] = []
    if not validation_root.exists():
        return rows

    for json_path in sorted(validation_root.glob("alpha_*/seed_*/agent*.json")):
        alpha_tag = json_path.parents[1].name
        seed_name = json_path.parent.name
        if json_path.stem == "agent_final":
            trainer_state_path = checkpoints_root / dataset_name / alpha_tag / seed_name / "trainer_state.json"
            trainer_state = _load_json(trainer_state_path)
            train_step = _selected_run_total_steps(trainer_state)
        else:
            try:
                train_step = int(json_path.stem.split("_")[-1])
            except ValueError:
                continue
        metrics = _load_json(json_path)
        rows.append(
            {
                "dataset": dataset_name,
                "alpha_tag": alpha_tag,
                "alpha": _parse_alpha_tag(alpha_tag),
                "seed": seed_name,
                "train_step": train_step,
                "val_success_rate": metrics.get("eval_success_rate"),
                "val_return": metrics.get("eval_return"),
            }
        )

    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["alpha"]), int(row["train_step"]))].append(row)

    curve_rows: list[dict[str, Any]] = []
    for (alpha, train_step), entries in sorted(grouped.items()):
        success_values = [float(entry["val_success_rate"]) for entry in entries if entry["val_success_rate"] is not None]
        return_values = [float(entry["val_return"]) for entry in entries if entry["val_return"] is not None]
        mean_success, std_success = _mean_std(success_values)
        mean_return, _ = _mean_std(return_values)
        curve_rows.append(
            {
                "dataset": dataset_name,
                "alpha": alpha,
                "train_step": train_step,
                "num_seeds": len(entries),
                "mean_val_success_rate": mean_success,
                "std_val_success_rate": std_success,
                "mean_val_return": mean_return,
            }
        )
    return curve_rows


def analyze_phase0b_v2(
    *,
    results_root: Path,
    checkpoints_root: Path,
    dataset_filter: str | None = None,
) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    alpha_validation_rows: list[dict[str, Any]] = []

    dataset_dirs = [
        path
        for path in sorted(results_root.iterdir())
        if path.is_dir() and path.name not in {"summaries", "analysis"}
    ]
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        if dataset_filter and dataset_filter not in dataset_name:
            continue
        selection_path = dataset_dir / "selection" / "best_alpha.json"
        if not selection_path.exists():
            continue

        selection = _load_json(selection_path)
        best_alpha_tag = str(selection.get("best_alpha_tag", ""))
        per_alpha = selection.get("per_alpha", [])
        selected_runs = selection.get("selected_runs", [])
        if not best_alpha_tag or not selected_runs:
            continue

        probe_seed = str(selected_runs[0]["seed"])
        probe_state_path = checkpoints_root / dataset_name / best_alpha_tag / probe_seed / "trainer_state.json"
        trainer_state = _load_json(probe_state_path)
        offline_metadata = _as_dict(trainer_state.get("offline_metadata"))

        baselines = _load_dataset_baselines(results_root, dataset_name)
        baseline_name, baseline_metrics = _select_reference_baseline(baselines, offline_metadata)

        td3bc_test_dir = dataset_dir / "test_selected" / best_alpha_tag
        td3bc_test_records = [
            _load_json(path) for path in sorted(td3bc_test_dir.glob("*.json"))
        ] if td3bc_test_dir.exists() else []
        bc_test_dir = dataset_dir / "test_bc_selected" / "alpha_0p0"
        bc_test_records = [
            _load_json(path) for path in sorted(bc_test_dir.glob("*.json"))
        ] if bc_test_dir.exists() else []

        td3bc_test_success_mean, td3bc_test_success_std = _mean_metric(
            td3bc_test_records, "eval_success_rate"
        )
        td3bc_test_return_mean, td3bc_test_return_std = _mean_metric(
            td3bc_test_records, "eval_return"
        )
        bc_test_success_mean, bc_test_success_std = _mean_metric(
            bc_test_records, "eval_success_rate"
        )
        bc_test_return_mean, bc_test_return_std = _mean_metric(
            bc_test_records, "eval_return"
        )

        bc_record = next(
            (
                item for item in per_alpha
                if item.get("alpha") is not None and abs(float(item["alpha"])) < 1e-12
            ),
            None,
        )

        max_alpha = max((float(item["alpha"]) for item in per_alpha), default=math.nan)
        selected_step_fracs: list[float] = []
        selected_steps: list[int] = []
        total_steps_list: list[int] = []
        for run in selected_runs:
            alpha_tag = str(run.get("alpha_tag") or best_alpha_tag)
            seed_name = str(run["seed"])
            trainer_state_path = checkpoints_root / dataset_name / alpha_tag / seed_name / "trainer_state.json"
            run_state = _load_json(trainer_state_path)
            total_steps = _selected_run_total_steps(run_state)
            chosen_step = int(run.get("train_step") or total_steps)
            total_steps_list.append(total_steps)
            selected_steps.append(chosen_step)
            if total_steps > 0:
                selected_step_fracs.append(chosen_step / total_steps)

        for alpha_item in per_alpha:
            alpha_validation_rows.append(
                {
                    "dataset": dataset_name,
                    "episodes": offline_metadata.get("num_episodes"),
                    "alpha_tag": alpha_item.get("alpha_tag"),
                    "alpha": alpha_item.get("alpha"),
                    "num_seeds": alpha_item.get("num_seeds"),
                    "mean_val_success_rate": alpha_item.get("mean_val_success_rate"),
                    "std_val_success_rate": alpha_item.get("std_val_success_rate"),
                    "mean_val_return": alpha_item.get("mean_val_return"),
                    "mean_val_safety_cost": alpha_item.get("mean_val_safety_cost"),
                    "mean_val_time_s": alpha_item.get("mean_val_time_s"),
                }
            )

        dataset_rows.append(
            {
                "dataset": dataset_name,
                "episodes": offline_metadata.get("num_episodes"),
                "best_alpha_tag": best_alpha_tag,
                "best_alpha": selection.get("best_alpha"),
                "max_alpha_in_sweep": max_alpha,
                "best_alpha_hits_boundary": (
                    bool(math.isfinite(float(max_alpha)))
                    and math.isclose(float(selection.get("best_alpha", math.nan)), float(max_alpha))
                ),
                "bc_val_success_rate": None if bc_record is None else bc_record.get("mean_val_success_rate"),
                "best_val_success_rate": max(
                    (float(item["mean_val_success_rate"]) for item in per_alpha),
                    default=math.nan,
                ),
                "td3bc_test_success_rate": td3bc_test_success_mean,
                "td3bc_test_success_rate_std": td3bc_test_success_std,
                "bc_test_success_rate": bc_test_success_mean,
                "bc_test_success_rate_std": bc_test_success_std,
                "td3bc_test_return": td3bc_test_return_mean,
                "td3bc_test_return_std": td3bc_test_return_std,
                "bc_test_return": bc_test_return_mean,
                "bc_test_return_std": bc_test_return_std,
                "baseline_name": baseline_name,
                "baseline_test_success_rate": (
                    None if baseline_metrics is None else baseline_metrics.get("eval_success_rate")
                ),
                "baseline_test_return": (
                    None if baseline_metrics is None else baseline_metrics.get("eval_return")
                ),
                "td3bc_minus_bc_success_rate": _safe_sub(
                    td3bc_test_success_mean,
                    bc_test_success_mean,
                ),
                "td3bc_minus_baseline_success_rate": _safe_sub(
                    td3bc_test_success_mean,
                    None if baseline_metrics is None else baseline_metrics.get("eval_success_rate"),
                ),
                "mean_selected_ckpt_frac": (
                    statistics.fmean(selected_step_fracs) if selected_step_fracs else None
                ),
                "min_selected_ckpt_frac": min(selected_step_fracs) if selected_step_fracs else None,
                "max_selected_ckpt_frac": max(selected_step_fracs) if selected_step_fracs else None,
                "selected_steps": selected_steps,
                "total_steps": total_steps_list,
                "selected_checkpoint_fractions": selected_step_fracs,
            }
        )

        comparison_rows.append(
            {
                "dataset": dataset_name,
                "episodes": offline_metadata.get("num_episodes"),
                "best_alpha": selection.get("best_alpha"),
                "bc_test_success_mean": bc_test_success_mean,
                "bc_test_success_std": bc_test_success_std,
                "bc_test_return_mean": bc_test_return_mean,
                "bc_test_return_std": bc_test_return_std,
                "td3bc_test_success_mean": td3bc_test_success_mean,
                "td3bc_test_success_std": td3bc_test_success_std,
                "td3bc_test_return_mean": td3bc_test_return_mean,
                "td3bc_test_return_std": td3bc_test_return_std,
                "td3bc_minus_bc_success": _safe_sub(
                    td3bc_test_success_mean,
                    bc_test_success_mean,
                ),
                "td3bc_minus_bc_return": _safe_sub(
                    td3bc_test_return_mean,
                    bc_test_return_mean,
                ),
                "baseline_test_success": (
                    None if baseline_metrics is None else baseline_metrics.get("eval_success_rate")
                ),
                "td3bc_minus_baseline": _safe_sub(
                    td3bc_test_success_mean,
                    None if baseline_metrics is None else baseline_metrics.get("eval_success_rate"),
                ),
            }
        )

    dataset_rows.sort(key=lambda row: (int(row["episodes"] or -1), str(row["dataset"])))
    comparison_rows.sort(key=lambda row: (int(row["episodes"] or -1), str(row["dataset"])))
    alpha_validation_rows.sort(
        key=lambda row: (
            int(row["episodes"] or -1),
            math.nan if row["alpha"] is None else float(row["alpha"]),
            str(row["dataset"]),
        )
    )

    largest_dataset_name = dataset_rows[-1]["dataset"] if dataset_rows else None
    largest_dataset_curve_rows = (
        _collect_validation_curve_rows(
            dataset_name=str(largest_dataset_name),
            results_root=results_root,
            checkpoints_root=checkpoints_root,
        )
        if largest_dataset_name is not None
        else []
    )

    conclusions: list[str] = []
    bc_val_values = [
        float(row["bc_val_success_rate"])
        for row in dataset_rows
        if row["bc_val_success_rate"] is not None
    ]
    if dataset_rows and len(bc_val_values) == len(dataset_rows):
        if _monotonic_drop(bc_val_values):
            conclusions.append(
                "BC validation success decreases monotonically with dataset size. "
                "The dominant bottleneck is still protocol or optimization budget, not only TD3BC's Q term."
            )
        else:
            conclusions.append(
                "BC validation success does not show a clean monotonic drop. "
                "The size effect may be more TD3BC-specific than pure cloning difficulty."
            )

    bc_test_values = [
        float(row["bc_test_success_mean"])
        for row in comparison_rows
        if row["bc_test_success_mean"] is not None
    ]
    if comparison_rows and len(bc_test_values) == len(comparison_rows):
        if _monotonic_drop(bc_test_values):
            conclusions.append(
                "BC test success also decreases monotonically with dataset size. "
                "This strengthens the case that the current optimization protocol remains the main issue."
            )
        else:
            conclusions.append(
                "BC test success does not decrease monotonically with dataset size. "
                "The degradation is not explained solely by cloning difficulty."
            )
    elif comparison_rows:
        conclusions.append(
            "BC test comparison is incomplete. Run this analysis with --run-bc-test to populate "
            "held-out test evaluations for alpha=0."
        )

    if dataset_rows:
        largest = dataset_rows[-1]
        if bool(largest["best_alpha_hits_boundary"]):
            conclusions.append(
                f"Largest dataset ({largest['episodes']} episodes) selects alpha={largest['best_alpha']}, "
                "which hits the current sweep boundary. Expand the alpha range in the next round."
            )
        else:
            conclusions.append(
                f"Largest dataset ({largest['episodes']} episodes) does not hit the current alpha boundary."
            )
        mean_frac = largest["mean_selected_ckpt_frac"]
        if mean_frac is not None and float(mean_frac) >= 0.8:
            conclusions.append(
                f"Largest dataset best checkpoints are selected late in training "
                f"(mean fraction={float(mean_frac):.3f}). Increase TRAIN_EPOCHS."
            )
        elif mean_frac is not None:
            conclusions.append(
                f"Largest dataset best checkpoints are not concentrated near the end "
                f"(mean fraction={float(mean_frac):.3f}). More epochs may not be the main lever."
            )

    td3bc_vs_bc_values = [
        float(row["td3bc_minus_bc_success"])
        for row in comparison_rows
        if row["td3bc_minus_bc_success"] is not None
    ]
    if comparison_rows and len(td3bc_vs_bc_values) == len(comparison_rows):
        if all(value >= 0.0 for value in td3bc_vs_bc_values):
            conclusions.append("TD3BC consistently improves over BC on the held-out test manifest.")
        elif all(value <= 0.0 for value in td3bc_vs_bc_values):
            conclusions.append(
                "TD3BC consistently underperforms BC on the held-out test manifest. "
                "Re-check alpha range and critic regime before trusting TD3BC."
            )
        else:
            conclusions.append(
                "TD3BC vs BC is mixed across dataset sizes. "
                "The Q term helps in some regimes but hurts in others."
            )

    return {
        "num_datasets": len(dataset_rows),
        "dataset_diagnostics": dataset_rows,
        "alpha_validation": alpha_validation_rows,
        "td3bc_vs_bc_test": comparison_rows,
        "largest_dataset": largest_dataset_name,
        "largest_dataset_validation_curve": largest_dataset_curve_rows,
        "conclusions": conclusions,
    }


def _plot_success_vs_dataset_size(
    dataset_rows: list[dict[str, Any]],
    output_dir: Path,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not dataset_rows:
        return None

    episodes = [int(row["episodes"]) for row in dataset_rows]
    td3bc = [row["td3bc_test_success_rate"] for row in dataset_rows]
    bc = [row["bc_test_success_rate"] for row in dataset_rows]
    baseline = [row["baseline_test_success_rate"] for row in dataset_rows]

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(episodes, td3bc, marker="o", label="TD3BC selected")
    if any(value is not None for value in bc):
        plt.plot(episodes, bc, marker="o", label="BC selected")
    if any(value is not None for value in baseline):
        plt.plot(episodes, baseline, marker="o", label="baseline")
    plt.xlabel("dataset episodes")
    plt.ylabel("test success rate")
    plt.title("Phase0b v2 test success vs dataset size")
    plt.grid(alpha=0.3)
    plt.legend()
    output_path = output_dir / "success_vs_dataset_size.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return str(output_path)


def _plot_validation_heatmap(
    alpha_rows: list[dict[str, Any]],
    output_dir: Path,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not alpha_rows:
        return None

    episodes = sorted({int(row["episodes"]) for row in alpha_rows if row["episodes"] is not None})
    alphas = sorted({float(row["alpha"]) for row in alpha_rows if row["alpha"] is not None})
    if not episodes or not alphas:
        return None

    grid = [[math.nan for _ in alphas] for _ in episodes]
    for row in alpha_rows:
        if row["episodes"] is None or row["alpha"] is None or row["mean_val_success_rate"] is None:
            continue
        epi_idx = episodes.index(int(row["episodes"]))
        alpha_idx = alphas.index(float(row["alpha"]))
        grid[epi_idx][alpha_idx] = float(row["mean_val_success_rate"])

    plt.figure(figsize=(max(6.0, 0.9 * len(alphas)), max(3.5, 0.8 * len(episodes))))
    image = plt.imshow(grid, aspect="auto", interpolation="nearest", cmap="YlGn")
    plt.colorbar(image, label="mean validation success")
    plt.xticks(range(len(alphas)), [f"{alpha:g}" for alpha in alphas])
    plt.yticks(range(len(episodes)), [str(episode) for episode in episodes])
    plt.xlabel("alpha")
    plt.ylabel("dataset episodes")
    plt.title("Phase0b v2 validation success heatmap")
    output_path = output_dir / "validation_success_heatmap.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return str(output_path)


def _plot_largest_dataset_curves(
    curve_rows: list[dict[str, Any]],
    largest_dataset_name: str | None,
    output_dir: Path,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not curve_rows or largest_dataset_name is None:
        return None

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[float(row["alpha"])].append(row)

    plt.figure(figsize=(8.0, 4.8))
    for alpha in sorted(grouped):
        entries = sorted(grouped[alpha], key=lambda item: int(item["train_step"]))
        steps = [int(item["train_step"]) for item in entries]
        values = [float(item["mean_val_success_rate"]) for item in entries]
        plt.plot(steps, values, marker="o", label=f"alpha={alpha:g}")
    plt.xlabel("train step")
    plt.ylabel("mean validation success")
    plt.title(f"Validation success vs checkpoint step | {largest_dataset_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    output_path = output_dir / "largest_dataset_validation_curves.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return str(output_path)


def write_analysis_outputs(
    *,
    analysis: dict[str, Any],
    output_dir: Path,
    skip_plots: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = list(analysis["dataset_diagnostics"])
    alpha_rows = list(analysis["alpha_validation"])
    comparison_rows = list(analysis["td3bc_vs_bc_test"])
    curve_rows = list(analysis["largest_dataset_validation_curve"])

    _write_json(output_dir / "analysis_summary.json", analysis)
    _write_csv(output_dir / "dataset_diagnostics.csv", dataset_rows)
    _write_csv(output_dir / "alpha_validation.csv", alpha_rows)
    _write_csv(output_dir / "td3bc_vs_bc_test.csv", comparison_rows)
    _write_csv(output_dir / "largest_dataset_validation_curve.csv", curve_rows)
    (output_dir / "conclusions.txt").write_text(
        "\n".join(f"- {line}" for line in analysis["conclusions"]),
        encoding="utf-8",
    )

    plot_outputs: dict[str, str] = {}
    if not skip_plots:
        success_plot = _plot_success_vs_dataset_size(dataset_rows, output_dir)
        if success_plot is not None:
            plot_outputs["success_vs_dataset_size"] = success_plot
        heatmap_plot = _plot_validation_heatmap(alpha_rows, output_dir)
        if heatmap_plot is not None:
            plot_outputs["validation_success_heatmap"] = heatmap_plot
        curve_plot = _plot_largest_dataset_curves(
            curve_rows,
            analysis.get("largest_dataset"),
            output_dir,
        )
        if curve_plot is not None:
            plot_outputs["largest_dataset_validation_curves"] = curve_plot

    return plot_outputs


def _print_report(analysis: dict[str, Any], plot_outputs: dict[str, str]) -> None:
    dataset_rows = analysis["dataset_diagnostics"]
    comparison_rows = analysis["td3bc_vs_bc_test"]
    alpha_rows = analysis["alpha_validation"]

    dataset_table = [
        {
            "episodes": _fmt(row["episodes"]),
            "best_alpha": _fmt(row["best_alpha"], digits=3),
            "alpha_edge": "yes" if row["best_alpha_hits_boundary"] else "no",
            "bc_val": _fmt(row["bc_val_success_rate"], digits=3),
            "best_val": _fmt(row["best_val_success_rate"], digits=3),
            "td3bc_test": _fmt(row["td3bc_test_success_rate"], digits=3),
            "bc_test": _fmt(row["bc_test_success_rate"], digits=3),
            "baseline": _fmt(row["baseline_test_success_rate"], digits=3),
            "ckpt_frac": _fmt(row["mean_selected_ckpt_frac"], digits=3),
        }
        for row in dataset_rows
    ]
    _print_table(
        "Dataset Diagnostics",
        [
            ("episodes", "episodes"),
            ("best_alpha", "best_alpha"),
            ("alpha_edge", "edge"),
            ("bc_val", "bc_val"),
            ("best_val", "best_val"),
            ("td3bc_test", "td3bc_test"),
            ("bc_test", "bc_test"),
            ("baseline", "baseline"),
            ("ckpt_frac", "ckpt_frac"),
        ],
        dataset_table,
    )

    alpha_preview = [
        {
            "episodes": _fmt(row["episodes"]),
            "alpha": _fmt(row["alpha"], digits=3),
            "val_success": _fmt(row["mean_val_success_rate"], digits=3),
            "val_std": _fmt(row["std_val_success_rate"], digits=3),
        }
        for row in alpha_rows
    ]
    _print_table(
        "Validation by Alpha",
        [
            ("episodes", "episodes"),
            ("alpha", "alpha"),
            ("val_success", "val_success"),
            ("val_std", "val_std"),
        ],
        alpha_preview,
    )

    comparison_table = [
        {
            "episodes": _fmt(row["episodes"]),
            "best_alpha": _fmt(row["best_alpha"], digits=3),
            "bc_test": _fmt(row["bc_test_success_mean"], digits=3),
            "td3bc_test": _fmt(row["td3bc_test_success_mean"], digits=3),
            "delta": _fmt(row["td3bc_minus_bc_success"], digits=3),
            "baseline": _fmt(row["baseline_test_success"], digits=3),
        }
        for row in comparison_rows
    ]
    _print_table(
        "TD3BC vs BC Test",
        [
            ("episodes", "episodes"),
            ("best_alpha", "best_alpha"),
            ("bc_test", "bc_test"),
            ("td3bc_test", "td3bc_test"),
            ("delta", "td3bc-bc"),
            ("baseline", "baseline"),
        ],
        comparison_table,
    )

    print("\nConclusions")
    if analysis["conclusions"]:
        for line in analysis["conclusions"]:
            print(f"- {line}")
    else:
        print("(empty)")

    if plot_outputs:
        print("\nPlots")
        for name, path in sorted(plot_outputs.items()):
            print(f"- {name}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze offline TD3BC phase0b_v2 outputs.")
    parser.add_argument("--results-root", type=Path, default=Path("results/offline/td3bc/phase0b_v2"))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints/offline/td3bc/phase0b_v2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory. Defaults to <results-root>/analysis.",
    )
    parser.add_argument("--dataset-filter", type=str, default=None)
    parser.add_argument(
        "--run-bc-test",
        action="store_true",
        default=False,
        help="Evaluate the selected alpha=0 checkpoints on the held-out test manifest before analysis.",
    )
    parser.add_argument("--test-manifest", type=Path, default=None, help="Required when --run-bc-test is set.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--worker-device", type=str, default="cpu")
    parser.add_argument("--test-seed", type=int, default=456)
    parser.add_argument("--force-reeval", action="store_true", default=False)
    parser.add_argument("--skip-plots", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_root / "analysis")

    if args.run_bc_test:
        if args.test_manifest is None:
            raise ValueError("--test-manifest is required when --run-bc-test is set.")
        run_bc_test_evals(
            results_root=args.results_root,
            checkpoints_root=args.checkpoints_root,
            test_manifest=args.test_manifest,
            python_bin=args.python_bin,
            device=args.device,
            num_workers=args.num_workers,
            worker_device=args.worker_device,
            test_seed=args.test_seed,
            force_reeval=args.force_reeval,
            dataset_filter=args.dataset_filter,
        )

    analysis = analyze_phase0b_v2(
        results_root=args.results_root,
        checkpoints_root=args.checkpoints_root,
        dataset_filter=args.dataset_filter,
    )
    plot_outputs = write_analysis_outputs(
        analysis=analysis,
        output_dir=output_dir,
        skip_plots=args.skip_plots,
    )
    _print_report(analysis, plot_outputs)
    print(f"\n[write] analysis: {output_dir}")


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
