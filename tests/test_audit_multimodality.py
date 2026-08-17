"""Unit tests for ``scripts/audit_multimodality.py``.

Test plan anchored in ``docs/fql_audit_multimodality_design.md`` §9
(14 tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_multimodality import (
    AuditConfig,
    _audit_dataset,
    _build_knn_index,
    _gmm_mode_count,
    _load_obs_actions,
    _paired_bootstrap_delta_p_ge_2,
    _query_neighbor_actions,
    _sample_anchors,
    _validate_compatibility,
    _verdict,
    _welch_t_one_sided,
    main,
    run_audit,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_dataset(
    out_dir: Path,
    obs: np.ndarray,
    actions: np.ndarray,
    extra_meta: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "transitions.npz",
        obs=obs.astype(np.float32),
        actions=actions.astype(np.float32),
    )
    metadata = {
        "obs_dim": int(obs.shape[1]),
        "action_dim": int(actions.shape[1]),
        "num_episodes": 1,
        "num_transitions": int(obs.shape[0]),
    }
    if extra_meta:
        metadata.update(extra_meta)
    (out_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return out_dir


def _make_unimodal_dataset(
    n: int = 300, obs_dim: int = 4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, obs_dim)).astype(np.float32)
    # Action: deterministic function of obs + small noise → uni-modal cond.
    base = np.stack(
        [np.tanh(obs[:, 0]), np.tanh(0.5 * obs[:, 1])], axis=-1
    )
    actions = base + 0.01 * rng.normal(size=(n, 2))
    return obs.astype(np.float32), actions.astype(np.float32)


def _make_multimodal_dataset(
    n: int = 300, obs_dim: int = 4, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, obs_dim)).astype(np.float32)
    actions = np.empty((n, 2), dtype=np.float32)
    # Half of the rows take action ≈ +1, half take action ≈ -1, regardless of obs.
    flips = rng.integers(0, 2, size=n)
    for i in range(n):
        if flips[i] == 0:
            actions[i] = np.array([0.8, -0.5]) + 0.02 * rng.normal(size=2)
        else:
            actions[i] = np.array([-0.8, 0.5]) + 0.02 * rng.normal(size=2)
    return obs, actions


# ---------------------------------------------------------------------------
# 1. load / validate
# ---------------------------------------------------------------------------


def test_load_obs_actions_roundtrip(tmp_path: Path) -> None:
    obs = np.random.RandomState(0).randn(20, 6).astype(np.float32)
    actions = np.random.RandomState(1).randn(20, 2).astype(np.float32)
    ds = _write_dataset(tmp_path / "ds", obs, actions)
    obs_loaded, actions_loaded, meta = _load_obs_actions(ds)
    np.testing.assert_array_equal(obs_loaded, obs)
    np.testing.assert_array_equal(actions_loaded, actions)
    assert meta["obs_dim"] == 6
    assert meta["action_dim"] == 2


def test_validate_compatibility_mismatch_raises() -> None:
    obs_a = np.zeros((5, 4), dtype=np.float32)
    obs_b = np.zeros((5, 6), dtype=np.float32)
    actions_a = np.zeros((5, 2), dtype=np.float32)
    actions_b = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="obs_dim"):
        _validate_compatibility(
            {"obs_dim": 4, "action_dim": 2},
            {"obs_dim": 6, "action_dim": 2},
            obs_a,
            obs_b,
            actions_a,
            actions_b,
        )

    actions_b_wide = np.zeros((5, 3), dtype=np.float32)
    obs_b_same = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="action_dim"):
        _validate_compatibility(
            {"obs_dim": 4, "action_dim": 2},
            {"obs_dim": 4, "action_dim": 3},
            obs_a,
            obs_b_same,
            actions_a,
            actions_b_wide,
        )


# ---------------------------------------------------------------------------
# 2. anchor sampling
# ---------------------------------------------------------------------------


def test_sample_anchors_no_replacement() -> None:
    obs = np.zeros((100, 4), dtype=np.float32)
    rng = np.random.default_rng(0)
    idx = _sample_anchors(obs, n_anchors=20, rng=rng)
    assert idx.shape == (20,)
    assert len(np.unique(idx)) == 20  # no duplicates


def test_sample_anchors_full_when_small() -> None:
    obs = np.zeros((10, 4), dtype=np.float32)
    rng = np.random.default_rng(0)
    idx = _sample_anchors(obs, n_anchors=50, rng=rng)
    assert np.array_equal(idx, np.arange(10))


# ---------------------------------------------------------------------------
# 3. GMM mode count
# ---------------------------------------------------------------------------


def test_gmm_mode_count_unimodal_data() -> None:
    rng = np.random.default_rng(42)
    actions = rng.normal(loc=(0.5, -0.2), scale=0.05, size=(50, 2))
    n = _gmm_mode_count(
        actions.astype(np.float32),
        max_components=3,
        n_init=3,
        weight_floor=0.10,
        rng_seed=0,
    )
    assert n == 1, f"Expected single mode for tight gaussian, got {n}"


def test_gmm_mode_count_bimodal_data() -> None:
    rng = np.random.default_rng(43)
    half = 25
    cluster1 = rng.normal(loc=(0.8, -0.5), scale=0.05, size=(half, 2))
    cluster2 = rng.normal(loc=(-0.8, 0.5), scale=0.05, size=(half, 2))
    actions = np.concatenate([cluster1, cluster2], axis=0).astype(np.float32)
    n = _gmm_mode_count(
        actions,
        max_components=3,
        n_init=3,
        weight_floor=0.10,
        rng_seed=0,
    )
    assert n == 2, f"Expected 2 modes for well-separated bimodal, got {n}"


def test_gmm_mode_count_weight_floor() -> None:
    """A 95/5 mix should drop the tiny mode under weight_floor=0.10."""
    rng = np.random.default_rng(44)
    cluster1 = rng.normal(loc=(0.7, -0.3), scale=0.05, size=(95, 2))
    cluster2 = rng.normal(loc=(-0.7, 0.3), scale=0.05, size=(5, 2))
    actions = np.concatenate([cluster1, cluster2], axis=0).astype(np.float32)
    n = _gmm_mode_count(
        actions,
        max_components=3,
        n_init=3,
        weight_floor=0.10,
        rng_seed=0,
    )
    assert n == 1, (
        f"Expected weight_floor to suppress 5% minority cluster, got {n}"
    )


# ---------------------------------------------------------------------------
# 4. audit_dataset smoke
# ---------------------------------------------------------------------------


def test_audit_dataset_smoke(tmp_path: Path) -> None:
    obs, actions = _make_unimodal_dataset(n=200, obs_dim=4, seed=10)
    rng = np.random.default_rng(0)
    cfg = AuditConfig(
        dataset_a=tmp_path / "a",
        dataset_b=tmp_path / "b",
        output_dir=tmp_path / "out",
        knn_k=20,
        gmm_max_components=3,
        gmm_n_init=2,
        mode_weight_floor=0.10,
        n_anchor_states=50,
        n_bootstrap=100,
        seed=0,
    )
    result = _audit_dataset(obs, actions, cfg, rng)
    assert set(result.keys()) >= {
        "n_anchor",
        "anchor_indices",
        "mode_counts",
        "p_distribution",
        "p_ge_2",
        "mean_mode_count",
        "std_mode_count",
    }
    assert 0.0 <= result["p_ge_2"] <= 1.0
    assert result["n_anchor"] == 50
    assert len(result["mode_counts"]) == 50


# ---------------------------------------------------------------------------
# 5. paired bootstrap
# ---------------------------------------------------------------------------


def test_paired_bootstrap_ci_symmetric() -> None:
    mc = np.array([1, 2, 1, 2, 1, 1, 2, 1], dtype=np.int32)
    rng = np.random.default_rng(0)
    result = _paired_bootstrap_delta_p_ge_2(mc, mc, n_bootstrap=500, rng=rng)
    # With identical inputs, delta_mean should be ~0 and 0 within CI.
    assert abs(result["delta_mean"]) < 0.05
    assert result["delta_ci_2p5"] <= 0.0 <= result["delta_ci_97p5"]


def test_paired_bootstrap_ci_skewed() -> None:
    mc_a = np.ones(100, dtype=np.int32)
    mc_b = np.full(100, 2, dtype=np.int32)
    rng = np.random.default_rng(0)
    result = _paired_bootstrap_delta_p_ge_2(
        mc_a, mc_b, n_bootstrap=500, rng=rng
    )
    assert result["delta_mean"] == pytest.approx(1.0)
    assert result["delta_ci_2p5"] == pytest.approx(1.0)
    assert result["delta_ci_97p5"] == pytest.approx(1.0)


def test_paired_bootstrap_mismatched_lengths() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="matched"):
        _paired_bootstrap_delta_p_ge_2(
            np.ones(5, dtype=np.int32),
            np.ones(7, dtype=np.int32),
            n_bootstrap=10,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# 6. Welch's t
# ---------------------------------------------------------------------------


def test_welch_t_one_sided_directional() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=1.0, scale=1.0, size=200)
    b = rng.normal(loc=2.0, scale=1.0, size=200)
    out = _welch_t_one_sided(a, b)  # H1: mean(b) > mean(a)
    assert out["welch_p_one_sided"] < 0.05

    out_rev = _welch_t_one_sided(b, a)  # H1: mean(a) > mean(b) → fail
    assert out_rev["welch_p_one_sided"] > 0.95


# ---------------------------------------------------------------------------
# 7. verdict
# ---------------------------------------------------------------------------


def test_verdict_all_pass_pathway() -> None:
    audit_a = {"p_ge_2": 0.10}
    audit_b = {"p_ge_2": 0.45}
    bootstrap = {
        "delta_mean": 0.35,
        "delta_ci_2p5": 0.20,
        "delta_ci_97p5": 0.42,
        "delta_std": 0.05,
    }
    welch = {"welch_t": 8.0, "welch_p_one_sided": 1e-12}
    out = _verdict(audit_a, audit_b, bootstrap, welch)
    assert out["overall_pass"] is True
    assert out["verdict"] == "Gate A.2 PASS"
    for entry in out["criteria"].values():
        assert entry["pass"] is True


def test_verdict_any_fail_pathway() -> None:
    audit_a = {"p_ge_2": 0.10}
    audit_b = {"p_ge_2": 0.45}
    # Break c1: CI lower bound too low.
    bootstrap = {
        "delta_mean": 0.05,
        "delta_ci_2p5": 0.01,
        "delta_ci_97p5": 0.10,
        "delta_std": 0.02,
    }
    welch = {"welch_t": 1.0, "welch_p_one_sided": 0.5}
    out = _verdict(audit_a, audit_b, bootstrap, welch)
    assert out["overall_pass"] is False
    assert out["verdict"] == "Gate A.2 FAIL"
    # Specifically c1 + c2 must fail.
    assert not out["criteria"]["c1_delta_p_ge2_ci_lower_above_0p10"]["pass"]
    assert not out["criteria"]["c2_welch_p_below_0p07"]["pass"]
    # c3 + c4 should still pass given the audit blocks.
    assert out["criteria"]["c3_a_unimodal_p_ge2_below_0p20"]["pass"]
    assert out["criteria"]["c4_b_multimodal_p_ge2_above_0p30"]["pass"]


# ---------------------------------------------------------------------------
# 8. end-to-end (synthetic uni vs bi datasets)
# ---------------------------------------------------------------------------


def test_main_e2e_synthetic(tmp_path: Path) -> None:
    obs_a, act_a = _make_unimodal_dataset(n=200, obs_dim=4, seed=10)
    obs_b, act_b = _make_multimodal_dataset(n=200, obs_dim=4, seed=11)
    ds_a = _write_dataset(tmp_path / "uni", obs_a, act_a)
    ds_b = _write_dataset(tmp_path / "multi", obs_b, act_b)
    out_dir = tmp_path / "audit_out"

    rc = main(
        [
            "--dataset-a",
            str(ds_a),
            "--dataset-b",
            str(ds_b),
            "--output-dir",
            str(out_dir),
            "--knn-k",
            "20",
            "--gmm-max-components",
            "3",
            "--gmm-n-init",
            "2",
            "--n-anchor-states",
            "60",
            "--n-bootstrap",
            "100",
            "--seed",
            "0",
            "--label-a",
            "uni",
            "--label-b",
            "multi",
        ]
    )
    # Must be PASS (0) or FAIL (2) — never error (1).
    assert rc in (0, 2)

    # Artefacts must all exist.
    assert (out_dir / "audit_summary.json").exists()
    assert (out_dir / "mode_count_per_anchor.csv").exists()
    assert (out_dir / "mode_count_distribution.png").exists()

    summary = json.loads((out_dir / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["audit_a"]["p_ge_2"] <= summary["audit_b"]["p_ge_2"], (
        "Multimodal dataset should have higher p_ge_2 than unimodal"
    )


# ---------------------------------------------------------------------------
# 9. run_audit determinism
# ---------------------------------------------------------------------------


def test_run_audit_deterministic(tmp_path: Path) -> None:
    obs_a, act_a = _make_unimodal_dataset(n=120, obs_dim=4, seed=20)
    obs_b, act_b = _make_multimodal_dataset(n=120, obs_dim=4, seed=21)
    ds_a = _write_dataset(tmp_path / "u", obs_a, act_a)
    ds_b = _write_dataset(tmp_path / "m", obs_b, act_b)

    def _go(target: Path) -> dict:
        cfg = AuditConfig(
            dataset_a=ds_a,
            dataset_b=ds_b,
            output_dir=target,
            knn_k=15,
            gmm_max_components=2,
            gmm_n_init=2,
            mode_weight_floor=0.10,
            n_anchor_states=40,
            n_bootstrap=50,
            seed=7,
            label_a="u",
            label_b="m",
        )
        return run_audit(cfg)

    a = _go(tmp_path / "o1")
    b = _go(tmp_path / "o2")
    assert a["audit_a"]["p_ge_2"] == b["audit_a"]["p_ge_2"]
    assert a["audit_b"]["p_ge_2"] == b["audit_b"]["p_ge_2"]
    assert a["bootstrap"]["delta_mean"] == b["bootstrap"]["delta_mean"]
    assert a["welch"]["welch_p_one_sided"] == b["welch"]["welch_p_one_sided"]


# ---------------------------------------------------------------------------
# 10. knn boundary
# ---------------------------------------------------------------------------


def test_knn_index_invalid_k() -> None:
    obs = np.random.RandomState(0).randn(8, 4).astype(np.float32)
    with pytest.raises(ValueError, match="less than"):
        _build_knn_index(obs, k=10)


def test_query_neighbor_actions_shape() -> None:
    obs = np.random.RandomState(0).randn(40, 4).astype(np.float32)
    actions = np.random.RandomState(1).randn(40, 2).astype(np.float32)
    nbrs = _build_knn_index(obs, k=5)
    anchor_obs = obs[:7]
    result = _query_neighbor_actions(nbrs, anchor_obs, actions, k=5)
    assert result.shape == (7, 5, 2)
