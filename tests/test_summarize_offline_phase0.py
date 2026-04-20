from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_offline_phase0 import summarize


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _trainer_state(alpha: float, final_success_hint: float) -> dict:
    return {
        "protocol": "deployable",
        "train_step": 100000,
        "best_eval_step": 80000,
        "best_eval_metrics": {
            "eval_success_rate": final_success_hint + 0.05,
            "eval_return": -10.0,
        },
        "train_config": {
            "total_steps": 100000,
            "batch_size": 256,
            "eval_every_steps": 10000,
        },
        "agent_config": {
            "gamma": 0.99,
            "alpha": alpha,
        },
        "offline_metadata": {
            "policy": "crosscomp",
            "num_episodes": 500,
            "num_transitions": 75000,
            "success_rate": 0.86,
            "goal_rate": 0.86,
            "timeout_rate": 0.10,
            "out_of_bounds_rate": 0.04,
            "action_noise_std": 0.0,
        },
    }


def _final_eval(success_rate: float) -> dict:
    return {
        "eval_success_rate": success_rate,
        "eval_return": -50.0,
        "eval_cost": 10.0,
        "eval_time_s": 80.0,
        "eval_termination_counts": {"goal": int(round(100 * success_rate))},
    }


def test_summarize_offline_phase0_selects_best_alpha(tmp_path: Path) -> None:
    checkpoints_root = tmp_path / "checkpoints"
    results_root = tmp_path / "results"
    dataset = "dataset_a"

    _write_json(
        checkpoints_root / dataset / "alpha_0p0" / "seed_42" / "trainer_state.json",
        _trainer_state(alpha=0.0, final_success_hint=0.60),
    )
    _write_json(
        checkpoints_root / dataset / "alpha_0p1" / "seed_42" / "trainer_state.json",
        _trainer_state(alpha=0.1, final_success_hint=0.72),
    )
    _write_json(
        results_root / dataset / "alpha_0p0" / "seed_42_final_eval.json",
        _final_eval(0.60),
    )
    _write_json(
        results_root / dataset / "alpha_0p1" / "seed_42_final_eval.json",
        _final_eval(0.72),
    )
    _write_json(
        results_root / dataset / "baselines" / "crosscomp_final_eval.json",
        _final_eval(0.95),
    )

    summary = summarize(checkpoints_root=checkpoints_root, results_root=results_root)

    assert summary["num_runs"] == 2
    assert len(summary["per_run"]) == 2
    assert len(summary["per_alpha"]) == 2
    assert len(summary["per_dataset"]) == 1

    dataset_summary = summary["per_dataset"][0]
    assert dataset_summary["dataset"] == dataset
    assert dataset_summary["best_alpha_tag"] == "0p1"
    assert dataset_summary["best_eval_status"] == "available"
    assert dataset_summary["reference_baseline_status"] == "available"
    assert abs(dataset_summary["best_mean_final_eval_success_rate"] - 0.72) < 1e-9
    assert abs(dataset_summary["bc_mean_final_eval_success_rate"] - 0.60) < 1e-9
    assert abs(dataset_summary["best_minus_bc_success_rate"] - 0.12) < 1e-9
    assert abs(dataset_summary["best_minus_baseline_success_rate"] + 0.23) < 1e-9

    per_run_by_alpha = {record["alpha_tag"]: record for record in summary["per_run"]}
    assert per_run_by_alpha["0p0"]["best_eval_status"] == "available"
    assert per_run_by_alpha["0p0"]["reference_baseline_status"] == "available"
    assert abs(per_run_by_alpha["0p0"]["final_vs_baseline_success_gap"] + 0.35) < 1e-9
    assert abs(per_run_by_alpha["0p1"]["final_vs_baseline_success_gap"] + 0.23) < 1e-9


def test_summarize_offline_phase0_handles_null_best_eval_metrics(tmp_path: Path) -> None:
    checkpoints_root = tmp_path / "checkpoints"
    results_root = tmp_path / "results"
    dataset = "dataset_null_eval"

    trainer_state = _trainer_state(alpha=0.0, final_success_hint=0.50)
    trainer_state["best_eval_metrics"] = None
    trainer_state["train_config"]["eval_every_steps"] = 0
    _write_json(
        checkpoints_root / dataset / "alpha_0p0" / "seed_42" / "trainer_state.json",
        trainer_state,
    )
    _write_json(
        results_root / dataset / "alpha_0p0" / "seed_42_final_eval.json",
        _final_eval(0.50),
    )

    summary = summarize(checkpoints_root=checkpoints_root, results_root=results_root)

    assert summary["num_runs"] == 1
    record = summary["per_run"][0]
    assert record["best_eval_status"] == "disabled"
    assert record["reference_baseline_status"] == "missing_baseline"
    assert record["best_eval_success_rate"] is None
    assert record["best_eval_return"] is None
    assert abs(record["final_eval_success_rate"] - 0.50) < 1e-9

    alpha_summary = summary["per_alpha"][0]
    assert alpha_summary["best_eval_status"] == "disabled"
    assert alpha_summary["reference_baseline_status"] == "missing_baseline"

    dataset_summary = summary["per_dataset"][0]
    assert dataset_summary["best_eval_status"] == "disabled"
    assert dataset_summary["reference_baseline_status"] == "missing_baseline"
