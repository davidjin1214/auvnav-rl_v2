"""
paper/thesis_ch5/figures/scripts/_ch5_data.py

Shared eval-log loader for §5.5 online learning-curve figures (Fig A-E).

These figures plot REAL periodic-evaluation trajectories from the repository's
training logs (eval_log.csv), unlike the earlier §5.5 bar charts that carried
hand-entered scalar means. Keeping the loader in one place keeps every curve
figure on the same alignment / aggregation convention (DRY).

Convention (cross-checked against the source reports, do not edit from memory):
  - eval_log.csv = periodic evaluation, 30 episodes/point. Columns:
    env_step, reward_objective, eval_return, eval_cost, eval_safety_cost,
    eval_success_rate, eval_time_s, eval_energy, eval_path_length_m,
    eval_progress_ratio, eval_path_efficiency.
  - final_eval.json = a dedicated final-checkpoint re-evaluation (30 episodes);
    this is the "final" reported in the locked bottleneck table and differs
    from the last periodic point (which is noisier). Curve trajectories come
    from eval_log.csv; scalar endpoints in the table come from final_eval.json.
  - Seeds are aligned BY ROW INDEX (all critical runs share the same 39-point
    eval schedule; A0 shares a 60-point schedule), so no interpolation is used.

Pure numpy/csv; no auv_nav import (renders in any env).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# Repo root: this file is at paper/thesis_ch5/figures/scripts/_ch5_data.py
REPO_ROOT = Path(__file__).resolve().parents[4]

COLS = {
    "env_step": 0,
    "eval_return": 2,
    "eval_safety_cost": 4,
    "eval_success_rate": 5,
    "eval_progress_ratio": 9,
    "eval_path_efficiency": 10,
}

CRIT_BASE = (
    REPO_ROOT
    / "experiments/arrival_v2_prototype/single_u15_cross_tgt15/arrival_v2"
)
A0_BASE = (
    REPO_ROOT
    / "experiments/protocol_screen_v2/A0_single_u10_cross_tgt15"
)


def _read_series(path: Path, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (env_step, values[col]) from one eval_log.csv."""
    steps: list[float] = []
    vals: list[float] = []
    ci = COLS[col]
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            steps.append(float(row[COLS["env_step"]]))
            vals.append(float(row[ci]))
    return np.asarray(steps), np.asarray(vals)


def seed_aggregate(
    paths: list[Path], col: str = "eval_success_rate"
) -> dict:
    """Row-index aligned seed mean/std for one cell.

    Returns dict with steps (mean env_step across seeds), mean, std (ddof=1 when
    n>1 else 0), n, and the per-seed matrix (n_seeds, n_pts) for optional
    representative-trace overlays.
    """
    series = [_read_series(p, col) for p in paths if p.exists()]
    if not series:
        raise FileNotFoundError(f"no eval logs found among: {paths}")
    n_pts = min(len(v) for _, v in series)
    steps_stack = np.vstack([s[:n_pts] for s, _ in series])
    val_stack = np.vstack([v[:n_pts] for _, v in series])
    mean = val_stack.mean(axis=0)
    std = val_stack.std(axis=0, ddof=1) if val_stack.shape[0] > 1 else np.zeros(n_pts)
    return {
        "steps": steps_stack.mean(axis=0),
        "mean": mean,
        "std": std,
        "n": val_stack.shape[0],
        "per_seed": val_stack,
    }


def crit_cell(algo: str, k: int, seeds: list[int]) -> list[Path]:
    """Critical-regime (single_u15_cross_tgt15) eval_log paths.

    algo in {"sac_vanilla", "sac_asym"}; k in {4, 8, 12}; sensor fixed to s0.
    """
    return [
        CRIT_BASE / algo / f"s0_k{k}" / f"seed_{s}" / "results" / "eval_log.csv"
        for s in seeds
    ]


def crit_s1(seeds: list[int]) -> list[Path]:
    return [
        CRIT_BASE / "sac_vanilla" / "s1_k4" / f"seed_{s}" / "results" / "eval_log.csv"
        for s in seeds
    ]


def a0_cell(objective: str, sensor: str, seeds: list[int]) -> list[Path]:
    """Subcritical A0 eval_log paths. objective in {efficiency_v2, arrival_v1}."""
    return [
        A0_BASE / objective / f"{sensor}_k4" / f"seed_{s}" / "eval_log.csv"
        for s in seeds
    ]


def final_eval(algo: str, k: int, seed: int, sensor: str = "s0") -> dict:
    """Read one final_eval.json (the locked-table 'final' convention)."""
    p = CRIT_BASE / algo / f"{sensor}_k{k}" / f"seed_{seed}" / "results" / "final_eval.json"
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def episode_fail_matrix(runs: list[tuple[int, int]]) -> np.ndarray:
    """Per-episode failure matrix (n_runs, n_episodes) for vanilla s0 runs.

    runs: list of (k, seed). Cell = 1 if that final-checkpoint episode did NOT
    reach the goal (out_of_bounds / timeout), else 0. Used to reconstruct the
    manifest universal floor (episodes that fail across every run).
    """
    rows = []
    for k, s in runs:
        d = final_eval("sac_vanilla", k, s)
        rows.append([0 if e["success"] else 1 for e in d["eval_episode_results"]])
    return np.asarray(rows, dtype=int)
