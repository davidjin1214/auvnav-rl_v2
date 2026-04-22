from __future__ import annotations

import argparse
import csv
import json
import math
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


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _read_dataset_diagnostics(results_root: Path) -> dict[str, dict[str, Any]]:
    csv_path = results_root / "analysis" / "dataset_diagnostics.csv"
    rows = _read_csv_rows(csv_path)
    payload: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        if not dataset:
            continue
        payload[dataset] = row
    return payload


def _read_overview(results_root: Path) -> dict[str, dict[str, Any]]:
    csv_path = results_root / "summaries" / "phase0b_v2_overview.csv"
    rows = _read_csv_rows(csv_path)
    payload: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        if not dataset:
            continue
        payload[dataset] = row
    return payload


def _compare_rows(
    *,
    deployable_rows: dict[str, dict[str, Any]],
    privileged_rows: dict[str, dict[str, Any]],
    deployable_overview: dict[str, dict[str, Any]],
    privileged_overview: dict[str, dict[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in sorted(set(deployable_rows) | set(privileged_rows)):
        dep = deployable_rows.get(dataset, {})
        priv = privileged_rows.get(dataset, {})
        dep_over = deployable_overview.get(dataset, {})
        priv_over = privileged_overview.get(dataset, {})

        dep_success = _safe_float(dep.get("td3bc_test_success_rate")) or _safe_float(
            dep_over.get("mean_test_success_rate")
        )
        dep_bc_success = _safe_float(dep.get("bc_test_success_rate"))
        dep_baseline_success = _safe_float(dep.get("baseline_test_success_rate")) or _safe_float(
            dep_over.get("baseline_test_success_rate")
        )
        priv_success = _safe_float(priv.get("td3bc_test_success_rate")) or _safe_float(
            priv_over.get("mean_test_success_rate")
        )
        priv_baseline_success = _safe_float(priv.get("baseline_test_success_rate")) or _safe_float(
            priv_over.get("baseline_test_success_rate")
        )
        baseline_success = dep_baseline_success if dep_baseline_success is not None else priv_baseline_success

        dep_gap = None
        priv_gap = None
        gap_closed = None
        if baseline_success is not None and dep_success is not None:
            dep_gap = baseline_success - dep_success
        if baseline_success is not None and priv_success is not None:
            priv_gap = baseline_success - priv_success
        if dep_gap is not None and priv_gap is not None:
            gap_closed = dep_gap - priv_gap

        rows.append(
            {
                "comparison_label": label,
                "dataset": dataset,
                "episodes": int(float(dep.get("episodes") or priv.get("episodes") or 0)),
                "deployable_best_alpha": _safe_float(dep.get("best_alpha")) or _safe_float(dep_over.get("best_alpha")),
                "privileged_best_alpha": _safe_float(priv.get("best_alpha")) or _safe_float(priv_over.get("best_alpha")),
                "deployable_success": dep_success,
                "deployable_bc_success": dep_bc_success,
                "privileged_success": priv_success,
                "baseline_success": baseline_success,
                "privileged_minus_deployable_success": (
                    None if dep_success is None or priv_success is None else priv_success - dep_success
                ),
                "deployable_minus_bc_success": (
                    None if dep_success is None or dep_bc_success is None else dep_success - dep_bc_success
                ),
                "privileged_minus_bc_success": (
                    None if priv_success is None or dep_bc_success is None else priv_success - dep_bc_success
                ),
                "deployable_gap_to_baseline": dep_gap,
                "privileged_gap_to_baseline": priv_gap,
                "teacher_gap_closed_by_privileged": gap_closed,
                "deployable_ckpt_frac": _safe_float(dep.get("mean_selected_ckpt_frac")),
                "privileged_ckpt_frac": _safe_float(priv.get("mean_selected_ckpt_frac")),
            }
        )
    return rows


def _build_conclusions(
    *,
    screen_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    for row in final_rows:
        dataset = row["dataset"]
        dep = _safe_float(row.get("deployable_success"))
        priv = _safe_float(row.get("privileged_success"))
        base = _safe_float(row.get("baseline_success"))
        gap_closed = _safe_float(row.get("teacher_gap_closed_by_privileged"))
        dep_bc = _safe_float(row.get("deployable_bc_success"))

        lines.append(
            f"{dataset} final deployable vs privileged success: "
            f"{dep if dep is not None else 'NA'} -> {priv if priv is not None else 'NA'}."
        )
        if dep is not None and priv is not None:
            delta = priv - dep
            if delta >= 0.05:
                lines.append(
                    f"{dataset} privileged critic improves materially over deployable (+{delta:.3f}); "
                    "local teacher information is likely an important bottleneck."
                )
            elif delta >= 0.02:
                lines.append(
                    f"{dataset} privileged critic provides a modest but non-trivial improvement (+{delta:.3f})."
                )
            else:
                lines.append(
                    f"{dataset} privileged critic does not materially improve over deployable (+{delta:.3f}); "
                    "teacher information gap is likely not the dominant bottleneck."
                )
        if base is not None and dep is not None and priv is not None:
            lines.append(
                f"{dataset} baseline success={base:.3f}, deployable gap={base - dep:.3f}, "
                f"privileged gap={base - priv:.3f}."
            )
        if gap_closed is not None:
            lines.append(
                f"{dataset} privileged critic closes {gap_closed:.3f} success points of the deployable teacher gap."
            )
        if dep_bc is not None and dep is not None:
            lines.append(
                f"{dataset} deployable TD3BC minus BC success: {dep - dep_bc:.3f}."
            )

    for row in screen_rows:
        dataset = row["dataset"]
        dep = _safe_float(row.get("deployable_success"))
        priv = _safe_float(row.get("privileged_success"))
        if dep is not None and priv is not None:
            lines.append(
                f"{dataset} screening deployable vs privileged delta: {priv - dep:+.3f}."
            )

    return lines


def analyze_worldcomp_teacher_gap(
    *,
    deployable_screen_root: Path,
    privileged_screen_root: Path,
    deployable_final_root: Path,
    privileged_final_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    screen_rows = _compare_rows(
        deployable_rows=_read_dataset_diagnostics(deployable_screen_root),
        privileged_rows=_read_dataset_diagnostics(privileged_screen_root),
        deployable_overview=_read_overview(deployable_screen_root),
        privileged_overview=_read_overview(privileged_screen_root),
        label="screen",
    )
    final_rows = _compare_rows(
        deployable_rows=_read_dataset_diagnostics(deployable_final_root),
        privileged_rows=_read_dataset_diagnostics(privileged_final_root),
        deployable_overview=_read_overview(deployable_final_root),
        privileged_overview=_read_overview(privileged_final_root),
        label="final",
    )

    conclusions = _build_conclusions(screen_rows=screen_rows, final_rows=final_rows)

    payload = {
        "screen_comparison_rows": screen_rows,
        "final_comparison_rows": final_rows,
        "conclusions": conclusions,
        "deployable_screen_root": str(deployable_screen_root),
        "privileged_screen_root": str(privileged_screen_root),
        "deployable_final_root": str(deployable_final_root),
        "privileged_final_root": str(privileged_final_root),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "screen_comparison.csv", screen_rows)
    _write_csv(output_dir / "final_comparison.csv", final_rows)
    _write_json(output_dir / "teacher_gap_comparison.json", payload)
    (output_dir / "conclusions.txt").write_text("\n".join(conclusions) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deployable and privileged-critic worldcomp runs.")
    parser.add_argument("--deployable-screen-root", type=Path, required=True)
    parser.add_argument("--privileged-screen-root", type=Path, required=True)
    parser.add_argument("--deployable-final-root", type=Path, required=True)
    parser.add_argument("--privileged-final-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    analyze_worldcomp_teacher_gap(
        deployable_screen_root=args.deployable_screen_root,
        privileged_screen_root=args.privileged_screen_root,
        deployable_final_root=args.deployable_final_root,
        privileged_final_root=args.privileged_final_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
