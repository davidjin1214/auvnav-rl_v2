from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return data if isinstance(data, dict) else {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_alpha(alpha_tag: str) -> float:
    try:
        return float(alpha_tag.replace("p", "."))
    except ValueError:
        return math.nan


def _parse_seed(seed_tag: str) -> int | None:
    if not seed_tag.startswith("seed_"):
        return None
    try:
        return int(seed_tag[len("seed_") :])
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float], mean_value: float | None = None) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    mu = _mean(values) if mean_value is None else mean_value
    assert mu is not None
    var = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var)


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _display_metric(
    value: float | int | None,
    *,
    status: str = "available",
    digits: int = 3,
) -> str:
    if value is not None:
        return _fmt(value, digits=digits)
    status_labels = {
        "disabled": "disabled",
        "missing": "missing",
        "missing_baseline": "missing baseline",
        "missing_final_eval": "missing final eval",
        "not_applicable": "n/a",
        "mixed": "mixed",
        "available": "-",
    }
    return status_labels.get(status, status)


def _column_width(rows: list[dict[str, str]], key: str, header: str) -> int:
    return max([len(header), *(len(str(row.get(key, ""))) for row in rows)])


def _print_table(title: str, columns: list[tuple[str, str]], rows: list[dict[str, str]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("(empty)")
        return
    widths = {key: _column_width(rows, key, header) for key, header in columns}
    header_line = "  ".join(header.ljust(widths[key]) for key, header in columns)
    print(header_line)
    print("  ".join("-" * widths[key] for key, _ in columns))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns))


def _load_dataset_baselines(results_root: Path, dataset_name: str) -> dict[str, dict[str, Any]]:
    baselines_dir = results_root / dataset_name / "baselines"
    baselines: dict[str, dict[str, Any]] = {}
    if not baselines_dir.exists():
        return baselines
    for path in sorted(baselines_dir.glob("*_final_eval.json")):
        baseline_name = path.name.removesuffix("_final_eval.json")
        baselines[baseline_name] = _load_json(path)
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


def _best_eval_status(train_cfg: dict[str, Any], best_eval_metrics: dict[str, Any]) -> str:
    if best_eval_metrics:
        return "available"
    eval_every = train_cfg.get("eval_every_steps")
    if eval_every is not None:
        try:
            if int(eval_every) <= 0:
                return "disabled"
        except (TypeError, ValueError):
            pass
    return "missing"


def _baseline_status(
    baselines: dict[str, dict[str, Any]],
    baseline_name: str | None,
) -> str:
    if baseline_name is not None:
        return "available"
    if not baselines:
        return "missing_baseline"
    return "missing"


def _final_eval_status(final_eval: dict[str, Any], final_eval_source: str | None) -> str:
    if final_eval and final_eval_source is not None:
        return "available"
    return "missing_final_eval"


def _aggregate_status(values: list[str]) -> str:
    unique = {value for value in values if value}
    if not unique:
        return "missing"
    if len(unique) == 1:
        return next(iter(unique))
    return "mixed"


def _collect_run_records(
    checkpoints_root: Path,
    results_root: Path,
    dataset_filter: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trainer_state_path in sorted(checkpoints_root.glob("*/*/seed_*/trainer_state.json")):
        seed_dir = trainer_state_path.parent
        alpha_dir = seed_dir.parent
        dataset_dir = alpha_dir.parent
        dataset_name = dataset_dir.name
        if dataset_filter and dataset_filter not in dataset_name:
            continue
        alpha_dir_name = alpha_dir.name
        if not alpha_dir_name.startswith("alpha_"):
            continue
        alpha_tag = alpha_dir_name[len("alpha_") :]
        seed = _parse_seed(seed_dir.name)

        trainer_state = _load_json(trainer_state_path)
        offline_metadata = _as_dict(trainer_state.get("offline_metadata"))
        final_eval_path = seed_dir / "final_eval.json"
        result_eval_path = results_root / dataset_name / alpha_dir_name / f"{seed_dir.name}_final_eval.json"
        if result_eval_path.exists():
            final_eval = _load_json(result_eval_path)
            final_eval_source = str(result_eval_path)
        elif final_eval_path.exists():
            final_eval = _load_json(final_eval_path)
            final_eval_source = str(final_eval_path)
        else:
            final_eval = {}
            final_eval_source = None

        baselines = _load_dataset_baselines(results_root, dataset_name)
        baseline_name, baseline_metrics = _select_reference_baseline(baselines, offline_metadata)

        best_eval_metrics = _as_dict(trainer_state.get("best_eval_metrics"))
        train_cfg = _as_dict(trainer_state.get("train_config"))
        agent_cfg = _as_dict(trainer_state.get("agent_config"))
        best_eval_status = _best_eval_status(train_cfg, best_eval_metrics)
        reference_baseline_status = _baseline_status(baselines, baseline_name)
        run_final_eval_status = _final_eval_status(final_eval, final_eval_source)

        record = {
            "dataset": dataset_name,
            "alpha_tag": alpha_tag,
            "alpha": _parse_alpha(alpha_tag),
            "seed": seed,
            "protocol": trainer_state.get("protocol"),
            "policy": offline_metadata.get("policy"),
            "policy_mixture": offline_metadata.get("policy_mixture"),
            "action_noise_std": offline_metadata.get("action_noise_std"),
            "num_episodes": offline_metadata.get("num_episodes"),
            "num_transitions": offline_metadata.get("num_transitions"),
            "dataset_success_rate": offline_metadata.get("success_rate"),
            "goal_rate": offline_metadata.get("goal_rate"),
            "timeout_rate": offline_metadata.get("timeout_rate"),
            "out_of_bounds_rate": offline_metadata.get("out_of_bounds_rate"),
            "train_steps": trainer_state.get("train_step"),
            "total_steps": train_cfg.get("total_steps"),
            "batch_size": train_cfg.get("batch_size"),
            "gamma": agent_cfg.get("gamma"),
            "config_alpha": agent_cfg.get("alpha"),
            "best_eval_step": trainer_state.get("best_eval_step"),
            "best_eval_status": best_eval_status,
            "best_eval_success_rate": best_eval_metrics.get("eval_success_rate"),
            "best_eval_return": best_eval_metrics.get("eval_return"),
            "final_eval_status": run_final_eval_status,
            "final_eval_success_rate": final_eval.get("eval_success_rate"),
            "final_eval_return": final_eval.get("eval_return"),
            "final_eval_cost": final_eval.get("eval_cost"),
            "final_eval_time_s": final_eval.get("eval_time_s"),
            "final_eval_termination_counts": final_eval.get("eval_termination_counts"),
            "reference_baseline_name": baseline_name,
            "reference_baseline_status": reference_baseline_status,
            "reference_baseline_success_rate": (
                baseline_metrics.get("eval_success_rate") if baseline_metrics else None
            ),
            "reference_baseline_return": (
                baseline_metrics.get("eval_return") if baseline_metrics else None
            ),
            "trainer_state_path": str(trainer_state_path),
            "final_eval_path": final_eval_source,
        }
        if record["final_eval_success_rate"] is not None and record["reference_baseline_success_rate"] is not None:
            record["final_vs_baseline_success_gap"] = (
                float(record["final_eval_success_rate"])
                - float(record["reference_baseline_success_rate"])
            )
        else:
            record["final_vs_baseline_success_gap"] = None
        records.append(record)
    return records


def _aggregate_by_alpha(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        grouped[(record["dataset"], record["alpha_tag"])].append(record)

    summaries: list[dict[str, Any]] = []
    for (dataset, alpha_tag), records in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], _parse_alpha(item[0][1])),
    ):
        final_success = [
            float(record["final_eval_success_rate"])
            for record in records
            if record["final_eval_success_rate"] is not None
        ]
        final_return = [
            float(record["final_eval_return"])
            for record in records
            if record["final_eval_return"] is not None
        ]
        best_eval_success = [
            float(record["best_eval_success_rate"])
            for record in records
            if record["best_eval_success_rate"] is not None
        ]
        baseline_success = [
            float(record["reference_baseline_success_rate"])
            for record in records
            if record["reference_baseline_success_rate"] is not None
        ]
        mean_final_success = _mean(final_success)
        mean_final_return = _mean(final_return)
        mean_best_eval_success = _mean(best_eval_success)
        mean_baseline_success = _mean(baseline_success)
        summary = {
            "dataset": dataset,
            "alpha_tag": alpha_tag,
            "alpha": _parse_alpha(alpha_tag),
            "num_runs": len(records),
            "num_final_eval_runs": len(final_success),
            "best_eval_status": _aggregate_status(
                [str(record.get("best_eval_status", "missing")) for record in records]
            ),
            "final_eval_status": _aggregate_status(
                [str(record.get("final_eval_status", "missing_final_eval")) for record in records]
            ),
            "reference_baseline_status": _aggregate_status(
                [str(record.get("reference_baseline_status", "missing_baseline")) for record in records]
            ),
            "mean_final_eval_success_rate": mean_final_success,
            "std_final_eval_success_rate": _std(final_success, mean_final_success),
            "mean_final_eval_return": mean_final_return,
            "std_final_eval_return": _std(final_return, mean_final_return),
            "mean_best_eval_success_rate": mean_best_eval_success,
            "std_best_eval_success_rate": _std(best_eval_success, mean_best_eval_success),
            "mean_reference_baseline_success_rate": mean_baseline_success,
            "mean_final_vs_baseline_success_gap": (
                None
                if mean_final_success is None or mean_baseline_success is None
                else mean_final_success - mean_baseline_success
            ),
            "seeds": sorted(record["seed"] for record in records if record["seed"] is not None),
        }
        summaries.append(summary)
    return summaries


def _aggregate_by_dataset(alpha_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in alpha_summaries:
        grouped[summary["dataset"]].append(summary)

    dataset_summaries: list[dict[str, Any]] = []
    for dataset, summaries in sorted(grouped.items()):
        ordered = sorted(
            summaries,
            key=lambda item: (
                item["mean_final_eval_success_rate"] is None,
                -(item["mean_final_eval_success_rate"] or float("-inf")),
                -(item["mean_final_eval_return"] or float("-inf")),
            ),
        )
        best = ordered[0] if ordered else None
        bc_summary = next((item for item in summaries if item["alpha_tag"] == "0p0"), None)
        baseline_success = None if best is None else best.get("mean_reference_baseline_success_rate")
        dataset_summary = {
            "dataset": dataset,
            "best_alpha_tag": None if best is None else best["alpha_tag"],
            "best_alpha": None if best is None else best["alpha"],
            "best_eval_status": (
                "missing" if best is None else str(best.get("best_eval_status", "missing"))
            ),
            "final_eval_status": (
                "missing_final_eval"
                if best is None
                else str(best.get("final_eval_status", "missing_final_eval"))
            ),
            "reference_baseline_status": (
                "missing_baseline"
                if best is None
                else str(best.get("reference_baseline_status", "missing_baseline"))
            ),
            "best_mean_final_eval_success_rate": (
                None if best is None else best["mean_final_eval_success_rate"]
            ),
            "best_mean_final_eval_return": None if best is None else best["mean_final_eval_return"],
            "reference_baseline_success_rate": baseline_success,
            "bc_mean_final_eval_success_rate": (
                None if bc_summary is None else bc_summary["mean_final_eval_success_rate"]
            ),
            "best_minus_bc_success_rate": (
                None
                if best is None
                or bc_summary is None
                or best["mean_final_eval_success_rate"] is None
                or bc_summary["mean_final_eval_success_rate"] is None
                else best["mean_final_eval_success_rate"] - bc_summary["mean_final_eval_success_rate"]
            ),
            "best_minus_baseline_success_rate": (
                None
                if best is None
                or baseline_success is None
                or best["mean_final_eval_success_rate"] is None
                else best["mean_final_eval_success_rate"] - baseline_success
            ),
        }
        dataset_summaries.append(dataset_summary)
    return dataset_summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(
    checkpoints_root: Path,
    results_root: Path,
    dataset_filter: str | None = None,
) -> dict[str, Any]:
    run_records = _collect_run_records(checkpoints_root, results_root, dataset_filter)
    alpha_summaries = _aggregate_by_alpha(run_records)
    dataset_summaries = _aggregate_by_dataset(alpha_summaries)
    return {
        "checkpoints_root": str(checkpoints_root),
        "results_root": str(results_root),
        "num_runs": len(run_records),
        "per_run": run_records,
        "per_alpha": alpha_summaries,
        "per_dataset": dataset_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize offline Phase-0/0b TD3+BC runs by dataset, alpha, and seed.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("checkpoints/offline/td3bc/phase0"),
        help="Root directory that contains dataset/alpha_*/seed_*/trainer_state.json.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/offline/td3bc/phase0"),
        help="Root directory that contains dataset/alpha_*/seed_*_final_eval.json.",
    )
    parser.add_argument(
        "--dataset-filter",
        type=str,
        default=None,
        help="Optional substring filter applied to dataset names.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON summary.",
    )
    parser.add_argument(
        "--output-csv-prefix",
        type=Path,
        default=None,
        help="Optional prefix path. Writes <prefix>_per_run.csv, <prefix>_per_alpha.csv, <prefix>_per_dataset.csv.",
    )
    args = parser.parse_args()

    summary = summarize(
        checkpoints_root=args.checkpoints_root,
        results_root=args.results_root,
        dataset_filter=args.dataset_filter,
    )

    per_run_rows = [
        {
            "dataset": record["dataset"],
            "alpha": _fmt(record["alpha"], digits=3),
            "seed": record["seed"],
            "final_succ": _display_metric(
                record["final_eval_success_rate"],
                status=str(record.get("final_eval_status", "available")),
                digits=3,
            ),
            "best_succ": _display_metric(
                record["best_eval_success_rate"],
                status=str(record.get("best_eval_status", "available")),
                digits=3,
            ),
            "best_step": (
                record["best_eval_step"]
                if record["best_eval_step"] is not None
                else _display_metric(None, status=str(record.get("best_eval_status", "available")))
            ),
            "baseline": _display_metric(
                record["reference_baseline_success_rate"],
                status=str(record.get("reference_baseline_status", "available")),
                digits=3,
            ),
            "gap": _display_metric(
                record["final_vs_baseline_success_gap"],
                status=(
                    "not_applicable"
                    if record.get("reference_baseline_status") != "available"
                    else "available"
                ),
                digits=3,
            ),
            "n_trans": record["num_transitions"] if record["num_transitions"] is not None else "-",
            "policy": record["policy"] or "-",
            "noise": _fmt(record["action_noise_std"], digits=3),
        }
        for record in sorted(
            summary["per_run"],
            key=lambda item: (item["dataset"], item["alpha"], item["seed"] if item["seed"] is not None else -1),
        )
    ]
    _print_table(
        "Per-Run Summary",
        [
            ("dataset", "dataset"),
            ("alpha", "alpha"),
            ("seed", "seed"),
            ("final_succ", "final_succ"),
            ("best_succ", "best_eval"),
            ("best_step", "best_step"),
            ("baseline", "baseline"),
            ("gap", "gap"),
            ("n_trans", "n_trans"),
            ("policy", "policy"),
            ("noise", "noise"),
        ],
        per_run_rows,
    )

    per_alpha_rows = [
        {
            "dataset": record["dataset"],
            "alpha": _fmt(record["alpha"], digits=3),
            "runs": record["num_runs"],
            "final_mean": _display_metric(
                record["mean_final_eval_success_rate"],
                status=str(record.get("final_eval_status", "available")),
                digits=3,
            ),
            "final_std": _fmt(record["std_final_eval_success_rate"], digits=3),
            "best_mean": _display_metric(
                record["mean_best_eval_success_rate"],
                status=str(record.get("best_eval_status", "available")),
                digits=3,
            ),
            "baseline": _display_metric(
                record["mean_reference_baseline_success_rate"],
                status=str(record.get("reference_baseline_status", "available")),
                digits=3,
            ),
            "gap": _display_metric(
                record["mean_final_vs_baseline_success_gap"],
                status=(
                    "not_applicable"
                    if record.get("reference_baseline_status") != "available"
                    else "available"
                ),
                digits=3,
            ),
            "seeds": ",".join(str(seed) for seed in record["seeds"]),
        }
        for record in summary["per_alpha"]
    ]
    _print_table(
        "Per-Alpha Summary",
        [
            ("dataset", "dataset"),
            ("alpha", "alpha"),
            ("runs", "runs"),
            ("final_mean", "final_mean"),
            ("final_std", "final_std"),
            ("best_mean", "best_mean"),
            ("baseline", "baseline"),
            ("gap", "gap"),
            ("seeds", "seeds"),
        ],
        per_alpha_rows,
    )

    per_dataset_rows = [
        {
            "dataset": record["dataset"],
            "best_alpha": _fmt(record["best_alpha"], digits=3),
            "best_final": _display_metric(
                record["best_mean_final_eval_success_rate"],
                status=str(record.get("final_eval_status", "available")),
                digits=3,
            ),
            "bc_final": _fmt(record["bc_mean_final_eval_success_rate"], digits=3),
            "baseline": _display_metric(
                record["reference_baseline_success_rate"],
                status=str(record.get("reference_baseline_status", "available")),
                digits=3,
            ),
            "best_minus_bc": _fmt(record["best_minus_bc_success_rate"], digits=3),
            "best_minus_baseline": _display_metric(
                record["best_minus_baseline_success_rate"],
                status=(
                    "not_applicable"
                    if record.get("reference_baseline_status") != "available"
                    else "available"
                ),
                digits=3,
            ),
        }
        for record in summary["per_dataset"]
    ]
    _print_table(
        "Per-Dataset Summary",
        [
            ("dataset", "dataset"),
            ("best_alpha", "best_alpha"),
            ("best_final", "best_final"),
            ("bc_final", "bc_final"),
            ("baseline", "baseline"),
            ("best_minus_bc", "best-bc"),
            ("best_minus_baseline", "best-base"),
        ],
        per_dataset_rows,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        print(f"\n[write] json: {args.output_json}")

    if args.output_csv_prefix is not None:
        prefix = args.output_csv_prefix
        _write_csv(Path(f"{prefix}_per_run.csv"), summary["per_run"])
        _write_csv(Path(f"{prefix}_per_alpha.csv"), summary["per_alpha"])
        _write_csv(Path(f"{prefix}_per_dataset.csv"), summary["per_dataset"])
        print(f"[write] csv: {prefix}_per_run.csv")
        print(f"[write] csv: {prefix}_per_alpha.csv")
        print(f"[write] csv: {prefix}_per_dataset.csv")


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
