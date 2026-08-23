"""Tests for the two tools that recompute a published readout from `results/`.

`ch5_holdout_split_audit.py` splits every published 100-episode readout into the 40 episodes
that fed checkpoint selection and the 60 that did not -- `docs/data_integrity_open_items.md`
item (3). `ch5_clean_probe_readout.py` re-evaluates the already-selected checkpoints on a
manifest disjoint from the training range and reports the difference-in-differences -- item
(1). Both feed disclosed numbers in the chapter, and both read trees that are gitignored,
so every source here is a fixture.

The split boundary is the load-bearing constant: the validation manifest is `manifest_seed`
1250 plus offsets 0..39, so seeds 1250..1289 took part in selection and 1290..1349 did not.
Move that boundary by one and the "held-out" readout quietly stops being held out.
"""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "paper" / "thesis_ch5" / "tools"

if str(TOOLS) not in sys.path:
    sys.path.append(str(TOOLS))

holdout = importlib.import_module("ch5_holdout_split_audit")
probe = importlib.import_module("ch5_clean_probe_readout")


def episodes(first: int, count: int, successes: int) -> list[dict]:
    """`count` episodes from manifest seed `first`, the first `successes` of them won."""
    return [{"seed": first + i, "success": i < successes} for i in range(count)]


def readout(path: Path, rate: float, eps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"eval_success_rate": rate, "eval_episode_results": eps}),
                    encoding="utf-8")


# =========================================================================== #
# ch5_holdout_split_audit -- the sel40 / hold60 split
# =========================================================================== #
def test_the_split_falls_on_the_documented_manifest_seed_boundary(tmp_path):
    # 40 selection episodes, 30 of them won; 60 held-out, 30 of them won.
    eps = episodes(1250, 40, 30) + episodes(1290, 60, 30)
    readout(tmp_path / "seed_0.json", 0.60, eps)
    got = holdout.unit_split(tmp_path)
    assert got["n_seeds"] == 1
    assert got["sel"] == [0.75]
    assert got["hold"] == [0.50]


def test_the_two_boundary_seeds_land_on_the_sides_the_manifest_puts_them(tmp_path):
    # 1289 is the last selection episode, 1290 the first held out. Off by one here and a
    # "held-out" readout silently includes an episode selection already saw.
    readout(tmp_path / "seed_0.json", 0.5,
            [{"seed": 1289, "success": True}, {"seed": 1290, "success": False}])
    got = holdout.unit_split(tmp_path)
    assert got["sel"] == [1.0] and got["hold"] == [0.0]


def test_a_readout_with_nothing_on_one_side_is_skipped(tmp_path):
    # A 40-episode validation readout has no hold60 half; averaging it in would report a
    # held-out rate computed from selection episodes.
    readout(tmp_path / "seed_0.json", 0.75, episodes(1250, 40, 30))
    assert holdout.unit_split(tmp_path) is None


def test_the_full_rate_is_the_published_one_not_a_recomputation(tmp_path):
    # The point of the split is to compare against the published figure, so `full` must
    # come from the readout's own field. Recomputing it would compare a number to itself.
    readout(tmp_path / "seed_0.json", 0.61,
            episodes(1250, 40, 30) + episodes(1290, 60, 30))
    assert holdout.unit_split(tmp_path)["full"] == [0.61]


def test_a_directory_with_no_readouts_is_reported_as_not_synced(tmp_path):
    # `results/` is gitignored: absent data is "not synced locally", never a failure.
    assert holdout.unit_split(tmp_path) is None


def test_the_alternate_eval_json_naming_is_picked_up(tmp_path):
    readout(tmp_path / "crosscomp_eval.json", 0.6,
            episodes(1250, 40, 30) + episodes(1290, 60, 30))
    assert holdout.unit_split(tmp_path)["n_seeds"] == 1


def test_seed_files_take_precedence_over_the_fallback_naming(tmp_path):
    eps = episodes(1250, 40, 40) + episodes(1290, 60, 60)
    readout(tmp_path / "seed_0.json", 0.11, eps)
    readout(tmp_path / "other_eval.json", 0.99, eps)
    assert holdout.unit_split(tmp_path)["full"] == [0.11]


def test_multiple_seed_files_are_read_in_sorted_order(tmp_path):
    eps = episodes(1250, 40, 40) + episodes(1290, 60, 60)
    for i, rate in enumerate([0.1, 0.2, 0.3]):
        readout(tmp_path / f"seed_{i}.json", rate, eps)
    assert holdout.unit_split(tmp_path)["full"] == [0.1, 0.2, 0.3]


# --------------------------------------------------------------------------- #
# the statistics the readouts are reported with
# --------------------------------------------------------------------------- #
def test_the_two_dispersion_conventions_differ_by_the_bessel_correction():
    xs = [0.9, 0.8, 1.0]
    assert holdout.std_pop(xs) == pytest.approx(math.sqrt(0.02 / 3))
    assert holdout.std_sample(xs) == pytest.approx(math.sqrt(0.02 / 2))


def test_paired_t_is_computed_on_the_per_seed_differences():
    a, b = [0.9, 0.8, 1.0], [0.8, 0.8, 0.8]
    m, t, hw, sd = holdout.paired_t(a, b)
    assert m == pytest.approx(0.1)  # diffs 0.1, 0.0, 0.2
    assert sd == pytest.approx(0.1)
    assert t == pytest.approx(m / (0.1 / math.sqrt(3)))
    assert hw == pytest.approx(4.303 * 0.1 / math.sqrt(3))  # t crit at n=3


def test_paired_t_on_identical_samples_does_not_divide_by_zero():
    m, t, hw, sd = holdout.paired_t([0.9, 0.9, 0.9], [0.9, 0.9, 0.9])
    assert (m, sd, hw) == (0.0, 0.0, 0.0)
    assert math.isnan(t)


def test_welch_uses_the_satterthwaite_degrees_of_freedom():
    # The two groups must differ in spread. At equal variance and equal n, Satterthwaite
    # returns exactly n1 + n2 - 2, so a pooled-df bug is invisible on such a pair.
    a, b = [0.9, 0.8, 1.0], [0.5, 0.55, 0.45]
    diff, t, df = holdout.welch(a, b)
    va = holdout.std_sample(a) ** 2 / 3
    vb = holdout.std_sample(b) ** 2 / 3
    assert diff == pytest.approx(0.4)
    assert df == pytest.approx((va + vb) ** 2 / (va ** 2 / 2 + vb ** 2 / 2))
    assert t == pytest.approx(0.4 / math.sqrt(va + vb))


# =========================================================================== #
# ch5_clean_probe_readout -- the difference-in-differences
# =========================================================================== #
# Means 0.90 / 0.94 / 0.86 / 0.87, so the flip is still +4 pp published, +1 pp clean and
# the DiD 3 pp. The spreads are deliberately all different: the cells used to carry
# identical per-seed values, which made every dispersion zero and left the sd guard --
# and the choice of ddof inside it -- unobservable.
PUB_1000 = [0.88, 0.90, 0.92]   # population sd 0.016330
PUB_2000 = [0.93, 0.94, 0.95]   # population sd 0.008165
CLN_1000 = [0.83, 0.86, 0.89]   # population sd 0.024495
CLN_2000 = [0.86, 0.87, 0.88]   # population sd 0.008165


@pytest.fixture
def probe_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(probe, "SEEDS", [42, 43, 44])
    monkeypatch.setattr(probe, "PUBLISHED_DIRS",
                        {"cross-1000": "pub/1000", "cross-2000": "pub/2000"})
    monkeypatch.setattr(probe, "CLEAN_DIRS",
                        {"cross-1000": "cln/1000", "cross-2000": "cln/2000"})
    monkeypatch.setattr(probe, "BASELINES", {})
    monkeypatch.setattr(probe, "EXPECTED", {
        ("published", "cross-1000"): 0.900, ("published", "cross-2000"): 0.940,
        ("clean", "cross-1000"): 0.860, ("clean", "cross-2000"): 0.870,
    })
    monkeypatch.setattr(probe, "EXPECTED_SD", {
        ("published", "cross-1000"): 0.016330, ("published", "cross-2000"): 0.008165,
        ("clean", "cross-1000"): 0.024495, ("clean", "cross-2000"): 0.008165,
    })
    for rel, rates in (("pub/1000", PUB_1000), ("pub/2000", PUB_2000),
                       ("cln/1000", CLN_1000), ("cln/2000", CLN_2000)):
        for seed, rate in zip(probe.SEEDS, rates):
            path = tmp_path / rel / f"seed_{seed}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"eval_success_rate": rate}), encoding="utf-8")
    return tmp_path


def test_the_reported_quantity_is_the_difference_in_differences(probe_tree, capsys):
    probe.main()
    out = capsys.readouterr().out
    assert "difference-in-differences" in out
    # Instance difficulty and selection generalisation are common-mode across the two
    # cells; only how much the flip shrinks is attributable to item (1).
    assert "+3.00 pp" in out


def test_a_readout_that_is_not_the_one_this_analysis_was_written_against_aborts(probe_tree):
    # The frozen expectation is the tool's guard against being pointed at a different
    # study and quietly reporting its numbers under this analysis's headings.
    path = probe_tree / "pub" / "1000" / "seed_42.json"
    path.write_text(json.dumps({"eval_success_rate": 0.5}), encoding="utf-8")
    with pytest.raises(SystemExit, match="published cross-1000"):
        probe.main()


def test_the_frozen_expectation_tolerates_only_fourth_decimal_drift(probe_tree, monkeypatch):
    # 5e-4 is a rounding allowance, not a slack budget: it accepts a re-serialised value,
    # not a different run.
    def expect(value):
        return {**probe.EXPECTED, ("published", "cross-1000"): value}

    monkeypatch.setattr(probe, "EXPECTED", expect(0.9004))
    probe.main()
    monkeypatch.setattr(probe, "EXPECTED", expect(0.9010))
    with pytest.raises(SystemExit):
        probe.main()


def test_a_drifted_dispersion_is_caught_even_when_the_mean_is_right(probe_tree, monkeypatch):
    """The guard that was missing until 2026-08-24.

    `docs/data_integrity_open_items.md` printed the clean cross-2000 dispersion as 0.025
    where the per-seed values give 0.024495; the mean was correct, so a mean-only
    expectation saw nothing and the doc sat one digit away from the chapter.
    """
    monkeypatch.setattr(probe, "EXPECTED_SD",
                        {**probe.EXPECTED_SD, ("clean", "cross-2000"): 0.0300})

    with pytest.raises(SystemExit, match="clean cross-2000: sd"):
        probe.main()


def test_the_dispersion_guard_reads_the_population_convention(probe_tree, monkeypatch):
    """Sample sd is the natural slip, and on three seeds it is 22% larger.

    The fixture's cells carry three different spreads so this cannot be satisfied by a
    coincidence in one of them -- with the old all-zero spreads it could not fail at all.
    """
    monkeypatch.setattr(probe, "EXPECTED_SD",
                        {("published", "cross-1000"): 0.020,
                         ("published", "cross-2000"): 0.010,
                         ("clean", "cross-1000"): 0.030,
                         ("clean", "cross-2000"): 0.010})

    with pytest.raises(SystemExit, match="sd"):
        probe.main()


def test_a_missing_seed_readout_aborts_and_names_the_path(probe_tree):
    (probe_tree / "cln" / "2000" / "seed_44.json").unlink()
    with pytest.raises(SystemExit, match="missing readout"):
        probe.main()


def test_a_paired_test_with_no_spread_reports_nan_rather_than_dividing_by_zero():
    mean, t, lo, hi = probe._paired([0.03, 0.03, 0.03])
    assert mean == pytest.approx(0.03)
    assert math.isnan(t)
    assert (lo, hi) == (mean, mean)


def test_the_paired_ci_uses_the_two_sided_95_percent_critical_value():
    mean, t, lo, hi = probe._paired([0.02, 0.03, 0.04])
    sd = math.sqrt(((0.01) ** 2 + 0 + (0.01) ** 2) / 2)
    se = sd / math.sqrt(3)
    assert mean == pytest.approx(0.03)
    assert hi - mean == pytest.approx(4.303 * se)
