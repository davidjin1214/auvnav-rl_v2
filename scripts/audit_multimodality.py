"""Multimodality audit for offline RL dataset comparison.

FQL Succession Plan v1 §3.4, Gate A.2 (see
``docs/fql_audit_multimodality_design.md`` v1.0).

Given two offline datasets — one expected to be **unimodal** (e.g. a single
expert policy) and one expected to be **multimodal** (e.g. a mixture of
multiple collector policies) — quantify the conditional action multimodality
strength by:

1. Sampling anchor states from each dataset.
2. Building a k-NN index over each dataset's ``obs`` and pulling the k
   nearest neighbours' actions per anchor (conditional ``p(a | s ≈ s_anchor)``).
3. Fitting a GMM with ``n ∈ {1, ..., max_components}`` to those neighbour
   actions, picking the best by BIC, and applying a mixture-weight floor.
4. Aggregating per-anchor mode counts → ``p_{≥2}`` per dataset.
5. Paired bootstrap CI for ``Δp_{≥2}`` and Welch's t one-sided test.
6. Emitting Gate A.2 verdict (PASS / FAIL) + JSON / CSV / PNG artefacts.

Dependency: ``scikit-learn`` (not in the repo's main training requirements;
install via ``pip install scikit-learn`` before running this script). Audit
is a separate dev-time tool isolated from the training pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.concat_offline_datasets import _load_dataset

try:
    from joblib import Parallel, delayed
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:  # pragma: no cover - explicit guard
    raise ImportError(
        "scripts/audit_multimodality.py requires scikit-learn (joblib comes "
        "with it). Install with: pip install scikit-learn\n"
        "(audit is a separate dev-time tool; sklearn is not in the repo's "
        "main training requirements.)"
    ) from exc


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AuditConfig:
    """Parameter bundle aligning with the CLI."""

    dataset_a: Path
    dataset_b: Path
    output_dir: Path
    knn_k: int = 50
    gmm_max_components: int = 3
    gmm_n_init: int = 3
    mode_weight_floor: float = 0.10
    n_anchor_states: int = 500
    n_bootstrap: int = 1000
    seed: int = 0
    label_a: str = "A"
    label_b: str = "B"
    n_jobs: int = -1  # -1 = use all cores; 1 = sequential (for tests/debug)


# ---------------------------------------------------------------------------
# I/O + validation
# ---------------------------------------------------------------------------


def _load_obs_actions(
    dataset_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load (obs, actions, metadata) from a dataset directory.

    Delegates I/O to :func:`scripts.concat_offline_datasets._load_dataset`
    (canonical npz+metadata loader; uses a context-managed ``np.load`` so the
    file handle is released).  We then project to the two arrays the audit
    cares about, with shape validation.
    """
    arrays, metadata = _load_dataset(dataset_dir)
    if "obs" not in arrays:
        raise KeyError(f"transitions.npz at {dataset_dir} is missing 'obs'")
    if "actions" not in arrays:
        raise KeyError(
            f"transitions.npz at {dataset_dir} is missing 'actions'"
        )
    obs = np.asarray(arrays["obs"], dtype=np.float32)
    actions = np.asarray(arrays["actions"], dtype=np.float32)
    if obs.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            "Expected obs/actions shape [N, D]; got "
            f"obs={obs.shape}, actions={actions.shape} at {dataset_dir}"
        )
    if obs.shape[0] != actions.shape[0]:
        raise ValueError(
            f"obs/actions length mismatch in {dataset_dir}: "
            f"{obs.shape[0]} vs {actions.shape[0]}"
        )
    return obs, actions, metadata


def _validate_compatibility(
    meta_a: dict[str, Any],
    meta_b: dict[str, Any],
    obs_a: np.ndarray,
    obs_b: np.ndarray,
    actions_a: np.ndarray,
    actions_b: np.ndarray,
) -> None:
    """Ensure both datasets share obs_dim and action_dim."""
    # Prefer metadata when present; fall back to array shapes.
    def _resolve(key: str, meta: dict[str, Any], fallback: int) -> int:
        if key in meta:
            return int(meta[key])
        return int(fallback)

    obs_dim_a = _resolve("obs_dim", meta_a, obs_a.shape[1])
    obs_dim_b = _resolve("obs_dim", meta_b, obs_b.shape[1])
    if obs_dim_a != obs_dim_b:
        raise ValueError(
            f"obs_dim mismatch: a={obs_dim_a} vs b={obs_dim_b}"
        )
    action_dim_a = _resolve("action_dim", meta_a, actions_a.shape[1])
    action_dim_b = _resolve("action_dim", meta_b, actions_b.shape[1])
    if action_dim_a != action_dim_b:
        raise ValueError(
            f"action_dim mismatch: a={action_dim_a} vs b={action_dim_b}"
        )


# ---------------------------------------------------------------------------
# anchor sampling + k-NN
# ---------------------------------------------------------------------------


def _sample_anchors(
    obs: np.ndarray,
    n_anchors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = obs.shape[0]
    if n <= n_anchors:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=n_anchors, replace=False).astype(np.int64)


def _build_knn_index(obs: np.ndarray, k: int) -> NearestNeighbors:
    n = obs.shape[0]
    if k >= n:
        raise ValueError(
            f"knn_k={k} must be less than dataset size N={n}"
        )
    # k+1 to account for the anchor matching itself.
    nbrs = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    )
    nbrs.fit(obs)
    return nbrs


def _query_neighbor_actions(
    nbrs: NearestNeighbors,
    anchor_obs: np.ndarray,
    all_actions: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return neighbour actions of shape ``[n_anchor, k, A]``."""
    _, indices = nbrs.kneighbors(anchor_obs, n_neighbors=k + 1)
    neighbor_idx = indices[:, 1 : k + 1]
    return all_actions[neighbor_idx]


# ---------------------------------------------------------------------------
# GMM mode count
# ---------------------------------------------------------------------------


def _gmm_mode_count(
    actions_k: np.ndarray,
    max_components: int,
    n_init: int,
    weight_floor: float,
    rng_seed: int,
) -> int:
    """Fit GMM with BIC selection + weight floor, return effective mode count."""
    best_n = 1
    best_bic = float("inf")
    best_gmm: GaussianMixture | None = None
    for n in range(1, max_components + 1):
        if n >= actions_k.shape[0]:
            break
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            n_init=n_init,
            random_state=rng_seed,
            max_iter=200,
            reg_covar=1e-4,
        )
        try:
            gmm.fit(actions_k)
        except ValueError:
            continue
        bic = float(gmm.bic(actions_k))
        if bic < best_bic:
            best_bic = bic
            best_n = n
            best_gmm = gmm

    if best_gmm is None:
        return 1
    weights = best_gmm.weights_
    n_effective = int(np.sum(weights >= weight_floor))
    return max(1, n_effective)


def _gmm_mode_count_or_fail(
    actions_k: np.ndarray,
    max_components: int,
    n_init: int,
    weight_floor: float,
    rng_seed: int,
) -> tuple[int, str | None]:
    """Worker variant of :func:`_gmm_mode_count` for ``joblib.Parallel``.

    Returns ``(mode_count, error_repr_or_None)``.  Failures (degenerate
    covariance, cholesky breakdown) are downgraded to ``mode_count=1`` with
    the type+message captured so the caller can surface a single warning.
    """
    try:
        n = _gmm_mode_count(
            actions_k, max_components, n_init, weight_floor, rng_seed
        )
        return n, None
    except (ValueError, np.linalg.LinAlgError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# audit one dataset
# ---------------------------------------------------------------------------


def _audit_dataset(
    obs: np.ndarray,
    actions: np.ndarray,
    config: AuditConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = obs.shape[0]
    if config.n_anchor_states > n:
        print(
            f"[audit] warning: n_anchor_states={config.n_anchor_states} "
            f"> dataset size {n}; using all {n} transitions as anchors.",
            file=sys.stderr,
        )
    anchor_idx = _sample_anchors(obs, config.n_anchor_states, rng)
    anchor_obs = obs[anchor_idx]

    nbrs = _build_knn_index(obs, config.knn_k)
    neighbor_actions = _query_neighbor_actions(
        nbrs, anchor_obs, actions, k=config.knn_k
    )

    # 500 anchors × up to 9 GMM fits each → dominant dev-time cost.  joblib
    # backend defaults to ``loky`` (process pool), which keeps each worker's
    # numpy RNG deterministic per-anchor since we pass an explicit
    # ``rng_seed`` derived from ``config.seed + i``.
    results = Parallel(n_jobs=int(config.n_jobs))(
        delayed(_gmm_mode_count_or_fail)(
            neighbor_actions[i],
            config.gmm_max_components,
            config.gmm_n_init,
            config.mode_weight_floor,
            int(config.seed) + i,
        )
        for i in range(len(anchor_idx))
    )
    mode_counts = np.fromiter(
        (mc for mc, _ in results), dtype=np.int32, count=len(results)
    )
    fail_reprs = [err for _, err in results if err is not None]
    fail_count = len(fail_reprs)
    first_fail_repr = fail_reprs[0] if fail_reprs else None

    if fail_count == len(anchor_idx):
        raise RuntimeError(
            "All GMM fits failed — audit invalid; verify dataset integrity. "
            f"First failure: {first_fail_repr}"
        )
    if first_fail_repr is not None:
        print(
            f"[audit] warning: {fail_count} / {len(anchor_idx)} anchors had "
            f"GMM fit failures (first: {first_fail_repr}); those were "
            "counted as mode_count=1.",
            file=sys.stderr,
        )

    n_anchor = int(len(mode_counts))
    p_distribution = {
        f"p_{j}": float(np.mean(mode_counts == j))
        for j in range(1, config.gmm_max_components + 1)
    }
    p_ge_2 = float(np.mean(mode_counts >= 2))
    std_mode = float(np.std(mode_counts, ddof=1)) if n_anchor > 1 else 0.0
    return {
        "n_anchor": n_anchor,
        "anchor_indices": anchor_idx.tolist(),
        "mode_counts": mode_counts.tolist(),
        "p_distribution": p_distribution,
        "p_ge_2": p_ge_2,
        "mean_mode_count": float(np.mean(mode_counts)),
        "std_mode_count": std_mode,
        "gmm_fit_failures": int(fail_count),
    }


# ---------------------------------------------------------------------------
# paired bootstrap + Welch's t
# ---------------------------------------------------------------------------


def _paired_bootstrap_delta_p_ge_2(
    mode_counts_a: np.ndarray,
    mode_counts_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    n_a = int(len(mode_counts_a))
    n_b = int(len(mode_counts_b))
    if n_a != n_b:
        raise ValueError(
            "Paired bootstrap requires matched anchor counts: "
            f"a={n_a} vs b={n_b}"
        )
    # Vectorised: draw all bootstrap indices in one (B, N) integer matrix,
    # broadcast-index the boolean ``>=2`` masks once, then average per row.
    idx = rng.integers(0, n_a, size=(n_bootstrap, n_a))
    mask_a = (mode_counts_a >= 2)[idx]   # [B, N] bool
    mask_b = (mode_counts_b >= 2)[idx]   # [B, N] bool
    p_a = mask_a.mean(axis=1)            # [B] float
    p_b = mask_b.mean(axis=1)            # [B] float
    deltas = p_b - p_a
    return {
        "delta_mean": float(np.mean(deltas)),
        "delta_ci_2p5": float(np.percentile(deltas, 2.5)),
        "delta_ci_97p5": float(np.percentile(deltas, 97.5)),
        "delta_std": float(np.std(deltas, ddof=1)) if n_bootstrap > 1 else 0.0,
    }


def _welch_t_one_sided(
    mode_counts_a: np.ndarray,
    mode_counts_b: np.ndarray,
) -> dict[str, float]:
    """One-sided Welch's t-test: H1: mean(B) > mean(A)."""
    from scipy import stats

    t_stat, p_one_sided = stats.ttest_ind(
        mode_counts_b,
        mode_counts_a,
        equal_var=False,
        alternative="greater",
    )
    return {
        "welch_t": float(t_stat),
        "welch_p_one_sided": float(p_one_sided),
    }


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def _verdict(
    audit_a: dict[str, Any],
    audit_b: dict[str, Any],
    bootstrap: dict[str, float],
    welch: dict[str, float],
) -> dict[str, Any]:
    """Apply the 4 Gate A.2 criteria from P0+P1 spec §1.3."""
    c1 = bootstrap["delta_ci_2p5"] > 0.10
    c2 = welch["welch_p_one_sided"] < 0.07
    c3 = audit_a["p_ge_2"] < 0.20
    c4 = audit_b["p_ge_2"] > 0.30
    criteria = {
        "c1_delta_p_ge2_ci_lower_above_0p10": {
            "value": float(bootstrap["delta_ci_2p5"]),
            "threshold": 0.10,
            "pass": bool(c1),
        },
        "c2_welch_p_below_0p07": {
            "value": float(welch["welch_p_one_sided"]),
            "threshold": 0.07,
            "pass": bool(c2),
        },
        "c3_a_unimodal_p_ge2_below_0p20": {
            "value": float(audit_a["p_ge_2"]),
            "threshold": 0.20,
            "pass": bool(c3),
        },
        "c4_b_multimodal_p_ge2_above_0p30": {
            "value": float(audit_b["p_ge_2"]),
            "threshold": 0.30,
            "pass": bool(c4),
        },
    }
    overall_pass = all(item["pass"] for item in criteria.values())
    return {
        "criteria": criteria,
        "overall_pass": bool(overall_pass),
        "verdict": "Gate A.2 PASS" if overall_pass else "Gate A.2 FAIL",
    }


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _save_summary_json(
    summary: dict[str, Any],
    out_path: Path,
) -> None:
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=False))


def _save_per_anchor_csv(
    audit_a: dict[str, Any],
    audit_b: dict[str, Any],
    label_a: str,
    label_b: str,
    out_path: Path,
) -> None:
    lines = ["dataset_label,anchor_idx,mode_count"]
    for idx, mc in zip(
        audit_a["anchor_indices"],
        audit_a["mode_counts"],
        strict=True,
    ):
        lines.append(f"{label_a},{int(idx)},{int(mc)}")
    for idx, mc in zip(
        audit_b["anchor_indices"],
        audit_b["mode_counts"],
        strict=True,
    ):
        lines.append(f"{label_b},{int(idx)},{int(mc)}")
    out_path.write_text("\n".join(lines) + "\n")


_COLOR_A = "#1f77b4"
_COLOR_B = "#d62728"


def _plot_mode_count_histogram(
    ax,
    audit_a: dict[str, Any],
    audit_b: dict[str, Any],
    label_a: str,
    label_b: str,
    max_components: int,
) -> None:
    counts_a = np.asarray(audit_a["mode_counts"], dtype=np.int32)
    counts_b = np.asarray(audit_b["mode_counts"], dtype=np.int32)
    bins = np.arange(0.5, max_components + 1.5, 1.0)
    ax.hist(counts_a, bins=bins, alpha=0.55, label=label_a, color=_COLOR_A)
    ax.hist(counts_b, bins=bins, alpha=0.55, label=label_b, color=_COLOR_B)
    ax.set_xlabel("per-anchor GMM mode count")
    ax.set_ylabel("# anchors")
    ax.set_xticks(range(1, max_components + 1))
    ax.legend(loc="best")


def _plot_p_distribution_bars(
    ax,
    audit_a: dict[str, Any],
    audit_b: dict[str, Any],
    label_a: str,
    label_b: str,
    max_components: int,
) -> None:
    width = 0.35
    x = np.arange(1, max_components + 1)
    pa = [audit_a["p_distribution"][f"p_{j}"] for j in x]
    pb = [audit_b["p_distribution"][f"p_{j}"] for j in x]
    ax.bar(x - width / 2, pa, width=width, color=_COLOR_A, label=label_a)
    ax.bar(x + width / 2, pb, width=width, color=_COLOR_B, label=label_b)
    ax.set_xlabel("mode count")
    ax.set_ylabel("fraction of anchors")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")


def _plot_mode_count_distribution(
    audit_a: dict[str, Any],
    audit_b: dict[str, Any],
    bootstrap: dict[str, float],
    welch: dict[str, float],
    verdict: dict[str, Any],
    label_a: str,
    label_b: str,
    max_components: int,
    out_path: Path,
) -> None:
    """Two-panel matplotlib summary figure (histogram + p-distribution bars)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - audit allowed to skip plot
        print(
            "[audit] matplotlib unavailable; skipping distribution plot.",
            file=sys.stderr,
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    _plot_mode_count_histogram(
        axes[0], audit_a, audit_b, label_a, label_b, max_components
    )
    _plot_p_distribution_bars(
        axes[1], audit_a, audit_b, label_a, label_b, max_components
    )
    fig.suptitle(f"Multimodality audit — {label_a} vs {label_b}")
    fig.text(
        0.5,
        -0.02,
        (
            f"Δp(≥2) = {bootstrap['delta_mean']:+.3f} "
            f"(95% CI [{bootstrap['delta_ci_2p5']:+.3f}, "
            f"{bootstrap['delta_ci_97p5']:+.3f}]), "
            f"Welch p = {welch['welch_p_one_sided']:.2e}, "
            f"{verdict['verdict']}"
        ),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main orchestration
# ---------------------------------------------------------------------------


def _config_from_args(args: argparse.Namespace) -> AuditConfig:
    return AuditConfig(
        dataset_a=Path(args.dataset_a).expanduser().resolve(),
        dataset_b=Path(args.dataset_b).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        knn_k=int(args.knn_k),
        gmm_max_components=int(args.gmm_max_components),
        gmm_n_init=int(args.gmm_n_init),
        mode_weight_floor=float(args.mode_weight_floor),
        n_anchor_states=int(args.n_anchor_states),
        n_bootstrap=int(args.n_bootstrap),
        seed=int(args.seed),
        label_a=str(args.label_a),
        label_b=str(args.label_b),
        n_jobs=int(args.n_jobs),
    )


def _build_summary(
    config: AuditConfig,
    meta_a: dict[str, Any],
    meta_b: dict[str, Any],
    audit_a: dict[str, Any],
    audit_b: dict[str, Any],
    bootstrap: dict[str, float],
    welch: dict[str, float],
    verdict: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    config_dict = asdict(config)
    config_dict["dataset_a"] = str(config.dataset_a)
    config_dict["dataset_b"] = str(config.dataset_b)
    config_dict["output_dir"] = str(config.output_dir)

    # Trim heavy per-anchor lists out of the audit blocks for the summary
    # JSON to keep it readable; the per-anchor CSV holds the full record.
    def _trim(audit_block: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in audit_block.items()
            if key not in {"anchor_indices", "mode_counts"}
        }

    return {
        "version": "1.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config_dict,
        "metadata_a": meta_a,
        "metadata_b": meta_b,
        "audit_a": _trim(audit_a),
        "audit_b": _trim(audit_b),
        "bootstrap": bootstrap,
        "welch": welch,
        "verdict": verdict,
        "wallclock_seconds": float(elapsed_sec),
    }


def run_audit(config: AuditConfig) -> dict[str, Any]:
    """End-to-end audit. Used both by ``main()`` and by tests."""
    rng = np.random.default_rng(int(config.seed))

    obs_a, actions_a, meta_a = _load_obs_actions(config.dataset_a)
    obs_b, actions_b, meta_b = _load_obs_actions(config.dataset_b)
    _validate_compatibility(meta_a, meta_b, obs_a, obs_b, actions_a, actions_b)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    audit_a = _audit_dataset(obs_a, actions_a, config, rng)
    audit_b = _audit_dataset(obs_b, actions_b, config, rng)

    # Truncate to the shorter array for paired bootstrap; both
    # n_anchor_states samples should normally have the same length.
    mc_a = np.asarray(audit_a["mode_counts"], dtype=np.int32)
    mc_b = np.asarray(audit_b["mode_counts"], dtype=np.int32)
    n_paired = min(len(mc_a), len(mc_b))
    mc_a_paired = mc_a[:n_paired]
    mc_b_paired = mc_b[:n_paired]

    bootstrap = _paired_bootstrap_delta_p_ge_2(
        mc_a_paired, mc_b_paired, config.n_bootstrap, rng
    )
    welch = _welch_t_one_sided(mc_a_paired, mc_b_paired)
    verdict = _verdict(audit_a, audit_b, bootstrap, welch)
    elapsed = time.perf_counter() - t0

    summary = _build_summary(
        config, meta_a, meta_b, audit_a, audit_b,
        bootstrap, welch, verdict, elapsed,
    )

    _save_summary_json(
        summary, config.output_dir / "audit_summary.json"
    )
    _save_per_anchor_csv(
        audit_a,
        audit_b,
        config.label_a,
        config.label_b,
        config.output_dir / "mode_count_per_anchor.csv",
    )
    _plot_mode_count_distribution(
        audit_a,
        audit_b,
        bootstrap,
        welch,
        verdict,
        config.label_a,
        config.label_b,
        config.gmm_max_components,
        config.output_dir / "mode_count_distribution.png",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Multimodality audit for offline RL datasets "
            "(FQL succession Gate A.2)."
        )
    )
    parser.add_argument("--dataset-a", required=True, type=Path)
    parser.add_argument("--dataset-b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--gmm-max-components", type=int, default=3)
    parser.add_argument("--gmm-n-init", type=int, default=3)
    parser.add_argument("--mode-weight-floor", type=float, default=0.10)
    parser.add_argument("--n-anchor-states", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-a", type=str, default="A")
    parser.add_argument("--label-b", type=str, default="B")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "joblib worker count for GMM fits. -1 uses all cores; "
            "1 forces sequential (useful for debugging or deterministic tests)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry. Returns exit code (0 = PASS, 2 = FAIL, 1 = error)."""
    parser = _parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    summary = run_audit(config)
    verdict_str = summary["verdict"]["verdict"]
    overall_pass = bool(summary["verdict"]["overall_pass"])
    print(f"[audit] {verdict_str}")
    print(
        f"[audit] p_ge_2(A)={summary['audit_a']['p_ge_2']:.3f}, "
        f"p_ge_2(B)={summary['audit_b']['p_ge_2']:.3f}, "
        f"Δ={summary['bootstrap']['delta_mean']:+.3f} "
        f"[{summary['bootstrap']['delta_ci_2p5']:+.3f}, "
        f"{summary['bootstrap']['delta_ci_97p5']:+.3f}]"
    )
    return 0 if overall_pass else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
