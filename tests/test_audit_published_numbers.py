"""Tests for the published-number traceback sweep.

The sweep exists to make four hand audits repeatable, and its whole value is that it
goes red when a report and its source data disagree. So every case here is run against
a tree that has the disagreement, not only against a healthy one -- a checker that
passes on clean input has proved nothing about the case it was written for.

The tree under test is a fixture, never this repo. One test deliberately reads the
shipped specs, and it is written so it needs no `results/`: anchors and captures are
resolved before any data is looked up, so a clone can still tell whether a report was
edited out from under its traceback spec.
"""
from __future__ import annotations

import json
import posixpath
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_published_numbers as apn

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relpath: str, text: str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _seed_files(root: Path, relpath: str, values: dict[str, float],
                metric: str = "eval_success_rate") -> None:
    for name, value in values.items():
        _write(root, f"{relpath}/{name}.json",
               json.dumps({metric: value, "num_eval_episodes": 100}))


def _spec(tmp_path: Path, claims: list[dict], **overrides) -> dict:
    spec = {
        "chain": "demo",
        "doc": "docs/report.md",
        "root": "results/demo",
        "metric": "eval_success_rate",
        "claims": claims,
        "_path": "docs/tracebacks/demo.json",
    }
    spec.update(overrides)
    return spec


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """A report citing two seeds, and the two seeds."""
    monkeypatch.setattr(apn, "ROOT", str(tmp_path))
    _seed_files(tmp_path, "results/demo/cell/test", {"seed_0": 0.88, "seed_42": 0.89})
    _write(tmp_path, "docs/report.md", "| cell | 0.885 |\n")
    return tmp_path


MEAN_CLAIM = {
    "label": "cell mean",
    "anchor": r"^\| cell \|",
    "capture": r"\| ([0-9.]+) \|",
    "stat": "mean",
    "sources": ["cell/test/seed_*.json"],
}


def run(tree, claims, **kw) -> dict[str, list]:
    return apn.run_spec(_spec(tree, claims), kw.pop("data_root", None),
                        kw.pop("require_data", False))


# ------------------------------------------------------------------ the comparison


def test_a_published_mean_that_matches_its_seeds_is_clean(tree):
    buckets = run(tree, [MEAN_CLAIM])

    assert buckets["value-mismatch"] == []
    assert len(buckets["ok"]) == 1


def test_a_published_mean_that_does_not_match_is_flagged(tree):
    _write(tree, "docs/report.md", "| cell | 0.925 |\n")

    mismatch = run(tree, [MEAN_CLAIM])["value-mismatch"]

    assert len(mismatch) == 1
    assert mismatch[0][3] == "0.925", "the published figure"
    assert mismatch[0][4].startswith("0.885"), "and what the seeds actually give"


def test_a_rounding_that_could_go_either_way_is_accepted(tree):
    """`-0.0275` published as `-0.027`, which the FQL report does.

    Which way a half rounds is a property of whatever formatted it, not a defect, so
    the criterion is "within half a unit of the last printed place" and not string
    equality. String equality would fail an honest row.
    """
    _seed_files(tree, "results/demo/cell/test", {"seed_0": 0.885, "seed_42": 0.830})
    for published in ("0.857", "0.858"):
        _write(tree, "docs/report.md", f"| cell | {published} |\n")

        assert run(tree, [MEAN_CLAIM])["value-mismatch"] == [], published


def test_a_value_outside_half_a_unit_is_not_a_rounding(tree):
    _write(tree, "docs/report.md", "| cell | 0.886 |\n")

    assert run(tree, [MEAN_CLAIM])["value-mismatch"] != []


def test_a_typeset_minus_sign_is_still_a_number(tree):
    """Reports are typeset prose: their minus is U+2212, which `float()` rejects."""
    _seed_files(tree, "results/demo/other/test", {"seed_0": 0.80, "seed_42": 0.83})
    _write(tree, "docs/report.md", "| delta | −0.070 |\n")

    buckets = run(tree, [{
        "label": "delta", "anchor": r"^\| delta \|",
        "capture": r"\| (−?[0-9.]+) \|", "stat": "delta",
        "sources": ["cell/test/seed_*.json", "other/test/seed_*.json"],
    }])

    assert buckets["value-mismatch"] == []
    assert len(buckets["ok"]) == 1


def test_a_thousands_separator_is_still_a_number(tree):
    """`152,683` is how the transition counts are printed, and `float()` rejects it.

    The negative control is the pair: with the comma stripped the figure must reproduce,
    and one thousand away from the true value it must not. Only asserting the first half
    would pass just as well if the comparison had quietly stopped happening.
    """
    _seed_files(tree, "results/demo/big/test", {"seed_0": 152683}, metric="num_transitions")
    claim = {
        "label": "transitions", "anchor": r"^\| transitions \|",
        "capture": r"\*\*([0-9,]+)\*\*", "stat": "mean",
        "sources": ["big/test/seed_*.json"], "metric": "num_transitions",
    }

    _write(tree, "docs/report.md", "| transitions | **152,683** |\n")
    assert run(tree, [claim])["value-mismatch"] == []
    assert len(run(tree, [claim])["ok"]) == 1

    _write(tree, "docs/report.md", "| transitions | **153,683** |\n")
    assert run(tree, [claim])["value-mismatch"] != []


def test_delta_is_the_second_source_minus_the_first(tree):
    _seed_files(tree, "results/demo/other/test", {"seed_0": 0.80, "seed_42": 0.83})
    _write(tree, "docs/report.md", "| delta | +0.070 |\n")

    assert run(tree, [{
        "label": "delta", "anchor": r"^\| delta \|",
        "capture": r"\| ([-+0-9.]+) \|", "stat": "delta",
        "sources": ["cell/test/seed_*.json", "other/test/seed_*.json"],
    }])["value-mismatch"] != [], "sign must follow sources order, not absolute value"


# ------------------------------------------------------------------ per-seed lists


def test_per_seed_lists_are_compared_as_a_multiset(tree):
    """Reports order per-seed values for reading, not by seed; the FQL one does."""
    _write(tree, "docs/report.md", "| cell | 0.89, 0.88 |\n")

    buckets = run(tree, [dict(MEAN_CLAIM, stat="seeds",
                              capture=r"\| ([0-9., ]+) \|")])

    assert buckets["value-mismatch"] == []


def test_a_per_seed_list_with_a_wrong_member_is_flagged(tree):
    _write(tree, "docs/report.md", "| cell | 0.88, 0.91 |\n")

    mismatch = run(tree, [dict(MEAN_CLAIM, stat="seeds",
                               capture=r"\| ([0-9., ]+) \|")])["value-mismatch"]

    assert len(mismatch) == 1
    assert "0.91" in mismatch[0][5]


def test_a_per_seed_list_of_the_wrong_length_is_flagged(tree):
    """A dropped seed changes the mean's meaning even when the survivors are right."""
    _write(tree, "docs/report.md", "| cell | 0.88 |\n")

    mismatch = run(tree, [dict(MEAN_CLAIM, stat="seeds",
                               capture=r"\| ([0-9., ]+) \|")])["value-mismatch"]

    assert len(mismatch) == 1
    assert "source has 2" in mismatch[0][5]


# ------------------------------------------------------------------ the ddof verdict


def test_the_dispersion_verdict_names_the_convention(tree):
    """What `ch5_dispersion_audit.py` cannot do for rows that print no per-seed values.

    Both conventions are computed from the source seeds, and the published figure is
    told which one it is written in -- that is the whole answer to the open question
    those four hand audits kept running into.
    """
    _seed_files(tree, "results/demo/five/test",
                {"seed_0": 0.74, "seed_1": 0.58, "seed_2": 0.65,
                 "seed_3": 0.61, "seed_4": 0.38})
    sd_claim = dict(MEAN_CLAIM, stat="sd", capture=r"± ([0-9.]+)",
                    sources=["five/test/seed_*.json"])

    _write(tree, "docs/report.md", "| cell | 0.592 ± 0.119 |\n")
    assert run(tree, [sd_claim])["ok"][0][4] == "ddof=0"

    _write(tree, "docs/report.md", "| cell | 0.592 ± 0.133 |\n")
    assert run(tree, [sd_claim])["ok"][0][4] == "ddof=1"


def test_a_dispersion_matching_neither_convention_is_flagged(tree):
    _seed_files(tree, "results/demo/three/test",
                {"seed_0": 0.74, "seed_1": 0.58, "seed_2": 0.65})
    _write(tree, "docs/report.md", "| cell | 0.657 ± 0.400 |\n")

    mismatch = run(tree, [dict(MEAN_CLAIM, stat="sd", capture=r"± ([0-9.]+)",
                               sources=["three/test/seed_*.json"])])["value-mismatch"]

    assert len(mismatch) == 1
    assert mismatch[0][4] == "NEITHER"


def test_a_single_seed_cannot_settle_a_dispersion(tree):
    _seed_files(tree, "results/demo/one/test", {"seed_0": 0.74})
    _write(tree, "docs/report.md", "| cell | 0.740 ± 0.000 |\n")

    mismatch = run(tree, [dict(MEAN_CLAIM, stat="sd", capture=r"± ([0-9.]+)",
                               sources=["one/test/seed_*.json"])])["value-mismatch"]

    assert len(mismatch) == 1


# ------------------------------------------------------------------ the locator


def test_an_anchor_that_no_longer_matches_is_a_defect(tree):
    """A rewritten sentence is exactly when the claim needs re-checking, so it fails
    rather than quietly checking nothing."""
    _write(tree, "docs/report.md", "| the cell in question | 0.885 |\n")

    buckets = run(tree, [MEAN_CLAIM])

    assert len(buckets["anchor-missing"]) == 1
    assert buckets["ok"] == []


def test_an_anchor_matching_two_lines_is_a_defect(tree):
    _write(tree, "docs/report.md", "| cell | 0.885 |\n| cell | 0.885 |\n")

    ambiguous = run(tree, [MEAN_CLAIM])["anchor-ambiguous"]

    assert len(ambiguous) == 1
    assert ambiguous[0][2] == [1, 2], "say which lines, so the anchor can be tightened"


def test_a_section_bound_separates_two_identical_rows(tree):
    """The ReBRAC screening grids: the same row under two dataset headings.

    Nothing in the row says which grid it belongs to; only the heading above it does.
    Without the bound the anchor matches both and the sweep refuses to pick.
    """
    _seed_files(tree, "results/demo/other/test", {"seed_0": 0.80, "seed_42": 0.82})
    _write(tree, "docs/report.md", "\n".join([
        "**dataset A**", "| cell | 0.885 |",
        "**dataset B**", "| cell | 0.810 |", "",
    ]))

    assert run(tree, [MEAN_CLAIM])["anchor-ambiguous"] != [], "unscoped, it is ambiguous"

    buckets = run(tree, [
        dict(MEAN_CLAIM, label="A", after=r"^\*\*dataset A\*\*", before=r"^\*\*dataset B\*\*"),
        dict(MEAN_CLAIM, label="B", after=r"^\*\*dataset B\*\*",
             sources=["other/test/seed_*.json"]),
    ])

    assert buckets["anchor-ambiguous"] == []
    assert [r[1] for r in buckets["ok"]] == ["docs/report.md:2", "docs/report.md:4"]


def test_a_section_runs_to_the_next_heading_of_the_same_level(tree):
    """The ReBRAC shape: the same bold dataset marker under two different sections.

    Scoping to the section first is what lets `after` stay unambiguous; a subsection
    inside must not end the section, or the rows below it fall out of scope.
    """
    _seed_files(tree, "results/demo/other/test", {"seed_0": 0.80, "seed_42": 0.82})
    _write(tree, "docs/report.md", "\n".join([
        "### 6.3 screen", "**dataset A**", "#### 6.3.1 aside", "| cell | 0.885 |",
        "### 6.4 per-seed", "**dataset A**", "| cell | 0.810 |", "",
    ]))
    scoped = dict(MEAN_CLAIM, section=r"^### 6\.3 ", after=r"^\*\*dataset A\*\*")

    assert run(tree, [MEAN_CLAIM])["anchor-ambiguous"] != [], "unscoped, it is ambiguous"

    buckets = run(tree, [scoped])

    assert buckets["spec-error"] == []
    assert [r[1] for r in buckets["ok"]] == ["docs/report.md:4"]


def test_a_section_that_is_not_a_heading_is_rejected(tree):
    _write(tree, "docs/report.md", "**dataset A**\n| cell | 0.885 |\n")

    errors = run(tree, [dict(MEAN_CLAIM, section=r"^\*\*dataset A\*\*")])["spec-error"]

    assert len(errors) == 1
    assert "no heading" in errors[0][1]


def test_a_section_bound_that_is_itself_ambiguous_is_an_error(tree):
    _write(tree, "docs/report.md", "**heading**\n| cell | 0.885 |\n**heading**\n")

    errors = run(tree, [dict(MEAN_CLAIM, after=r"^\*\*heading\*\*")])["spec-error"]

    assert len(errors) == 1
    assert "matches 2 lines" in errors[0][1]


def test_a_capture_that_misses_on_the_anchored_line_is_a_defect(tree):
    _write(tree, "docs/report.md", "| cell | n/a |\n")

    assert len(run(tree, [MEAN_CLAIM])["capture-failed"]) == 1


def test_the_report_names_the_line_the_figure_was_read_from(tree):
    _write(tree, "docs/report.md", "intro\n\n| cell | 0.885 |\n")

    assert run(tree, [MEAN_CLAIM])["ok"][0][1] == "docs/report.md:3"


# ------------------------------------------------------------------ missing data


def test_missing_results_are_skipped_not_failed(tree):
    """`results/` is gitignored, so a clone has nothing to recompute from. "Cannot
    check here" must not read the same as "the number is wrong"."""
    buckets = run(tree, [MEAN_CLAIM], data_root=str(tree / "nowhere"))

    assert len(buckets["no-data"]) == 1
    assert buckets["value-mismatch"] == []
    assert all(buckets[b] == [] for b in apn.DEFECT_BUCKETS)


def test_require_data_turns_a_skip_into_a_failure(tree):
    buckets = run(tree, [MEAN_CLAIM], data_root=str(tree / "nowhere"),
                  require_data=True)

    assert buckets["no-data"] == []
    assert len(buckets["spec-error"]) == 1


def test_a_missing_metric_key_in_the_json_is_reported(tree):
    _seed_files(tree, "results/demo/cell/test", {"seed_0": 0.88}, metric="other_key")

    errors = run(tree, [MEAN_CLAIM])["spec-error"]

    assert len(errors) == 1
    assert "eval_success_rate" in errors[0][1]


# ------------------------------------------------------------------ the spec itself


def test_an_unknown_spec_key_is_rejected(tree):
    """A misspelt key would leave its claims unchecked, which looks like passing."""
    spec = _spec(tree, [MEAN_CLAIM])
    spec["metrics"] = "eval_success_rate"        # note the plural

    errors = apn.run_spec(spec, None, False)["spec-error"]

    assert len(errors) == 1
    assert "metrics" in errors[0][1]


def test_an_unknown_claim_key_is_rejected(tree):
    errors = run(tree, [dict(MEAN_CLAIM, source=["cell/test/seed_*.json"])])["spec-error"]

    assert len(errors) == 1
    assert "source" in errors[0][1]


def test_an_unknown_statistic_is_rejected(tree):
    """Rejected by the whitelist, at load, before any doc or data is touched.

    `evaluate` also refuses an unknown stat, so asserting only "an error happened"
    stays green when the whitelist is removed -- two mechanisms, one of them then
    untested. The assertion names the validator's message on purpose.
    """
    errors = run(tree, [dict(MEAN_CLAIM, stat="median")])["spec-error"]

    assert len(errors) == 1
    assert "not one of" in errors[0][1] and "median" in errors[0][1]


def test_a_capture_with_two_groups_is_rejected(tree):
    """Which of the two is the published figure has no answer, so it is not guessed."""
    errors = run(tree, [dict(MEAN_CLAIM, capture=r"\| ([0-9]+)\.([0-9]+) \|")])["spec-error"]

    assert len(errors) == 1
    assert "exactly one group" in errors[0][1]


def test_a_delta_needs_exactly_two_sources(tree):
    errors = run(tree, [dict(MEAN_CLAIM, stat="delta")])["spec-error"]

    assert len(errors) == 1
    assert "two sources" in errors[0][1]


def test_a_mean_takes_only_one_source(tree):
    errors = run(tree, [dict(MEAN_CLAIM,
                             sources=["cell/test/seed_*.json", "cell/test/seed_*.json"])])

    assert len(errors["spec-error"]) == 1


# ------------------------------------------------------------------ CSV sources

EVAL_LOG = (
    "env_step,eval_success_rate\n"
    "100,0.2\n"
    "200,0.9\n"
    "300,0.5\n"
    "400,0.9\n"
)


def _csv_claim(**kw) -> dict:
    claim = dict(MEAN_CLAIM, sources=["cell/test/run.csv"],
                 metric="mean(eval_success_rate)")
    claim.update(kw)
    return claim


def assert_reproduced(buckets: dict[str, list]) -> None:
    """One claim, recomputed and matching -- and nothing routed elsewhere.

    `value-mismatch == []` alone is not that assertion: a claim that died in
    `spec-error` also leaves it empty, so a broken aggregation dispatch reads as a
    pass. One mutation run's worth of evidence for writing it out.
    """
    assert buckets["ok"] and len(buckets["ok"]) == 1, buckets
    for name in apn.DEFECT_BUCKETS + ("no-data",):
        assert buckets[name] == [], f"{name} -> {buckets[name]}"


@pytest.fixture()
def csv_tree(tree):
    _write(tree, "results/demo/cell/test/run.csv", EVAL_LOG)
    return tree


def test_a_csv_column_is_reduced_to_one_number_per_file(csv_tree):
    _write(csv_tree, "docs/report.md", "| cell | 0.625 |\n")

    assert_reproduced(run(csv_tree, [_csv_claim()]))


def test_a_csv_mean_that_does_not_match_is_flagged(csv_tree):
    _write(csv_tree, "docs/report.md", "| cell | 0.900 |\n")

    assert len(run(csv_tree, [_csv_claim()])["value-mismatch"]) == 1


def test_peak_is_the_column_max(csv_tree):
    _write(csv_tree, "docs/report.md", "| cell | 0.900 |\n")

    assert_reproduced(run(csv_tree, [_csv_claim(metric="max(eval_success_rate)")]))


def test_argmax_reports_first_attainment_not_the_last(csv_tree):
    """`peak @ 475k` is when the run got there; a plateau ties every later row."""
    _write(csv_tree, "docs/report.md", "| cell | 200 |\n")

    claim = _csv_claim(metric="argmax(eval_success_rate, env_step)")
    buckets = run(csv_tree, [claim])

    assert_reproduced(buckets)  # 400 would mean it took the last tie


def test_count_gt_counts_the_rows_above_the_threshold(csv_tree):
    _write(csv_tree, "docs/report.md", "| cell | 4 |\n")

    claim = _csv_claim(metric="count_gt(eval_success_rate, 0)")
    assert_reproduced(run(csv_tree, [claim]))

    _write(csv_tree, "docs/report.md", "| cell | 2 |\n")
    claim = _csv_claim(metric="count_gt(eval_success_rate, 0.5)")
    assert_reproduced(run(csv_tree, [claim]))


def test_nrows_counts_the_evaluations(csv_tree):
    _write(csv_tree, "docs/report.md", "| cell | 4 |\n")

    assert_reproduced(run(csv_tree, [_csv_claim(metric="nrows(eval_success_rate)")]))


def test_a_csv_source_read_as_a_plain_key_is_an_error(csv_tree):
    """Silently reading a `.csv` as JSON would surface as a parse error somewhere far
    from the spec line that caused it."""
    errors = run(csv_tree, [_csv_claim(metric="eval_success_rate")])["spec-error"]

    assert len(errors) == 1
    assert "aggregation" in errors[0][1]


def test_a_json_source_read_as_an_aggregation_is_an_error(tree):
    errors = run(tree, [dict(MEAN_CLAIM, metric="mean(eval_success_rate)")])["spec-error"]

    assert len(errors) == 1
    assert "JSON key" in errors[0][1]


def test_a_missing_csv_column_is_reported(csv_tree):
    errors = run(csv_tree, [_csv_claim(metric="mean(no_such_column)")])["spec-error"]

    assert len(errors) == 1
    assert "no_such_column" in errors[0][1]


def test_argmax_without_a_second_column_is_rejected(csv_tree):
    errors = run(csv_tree, [_csv_claim(metric="argmax(eval_success_rate)")])["spec-error"]

    assert len(errors) == 1


def test_count_gt_without_a_threshold_is_rejected(csv_tree):
    errors = run(csv_tree, [_csv_claim(metric="count_gt(eval_success_rate)")])["spec-error"]

    assert len(errors) == 1


# ------------------------------------------------------------------ scale

def test_scale_converts_the_unit_before_comparison(csv_tree):
    """`peak @ 475k` against 475002 steps on disk."""
    _write(csv_tree, "results/demo/cell/test/run.csv",
           "env_step,eval_success_rate\n100,0.2\n475002,0.9\n")
    _write(csv_tree, "docs/report.md", "| cell | 475 |\n")

    claim = _csv_claim(metric="argmax(eval_success_rate, env_step)", scale=1000)
    assert_reproduced(run(csv_tree, [claim]))


def test_without_the_scale_the_same_claim_is_a_mismatch(csv_tree):
    _write(csv_tree, "results/demo/cell/test/run.csv",
           "env_step,eval_success_rate\n100,0.2\n475002,0.9\n")
    _write(csv_tree, "docs/report.md", "| cell | 475 |\n")

    claim = _csv_claim(metric="argmax(eval_success_rate, env_step)")
    assert len(run(csv_tree, [claim])["value-mismatch"]) == 1


def test_scale_is_applied_to_every_seed_before_the_statistic(tree):
    _seed_files(tree, "results/demo/scaled/test", {"seed_0": 880.0, "seed_42": 890.0})
    _write(tree, "docs/report.md", "| cell | 0.885 |\n")

    claim = dict(MEAN_CLAIM, sources=["scaled/test/seed_*.json"], scale=1000)
    assert_reproduced(run(tree, [claim]))


def test_a_non_positive_scale_is_rejected(tree):
    errors = run(tree, [dict(MEAN_CLAIM, scale=0)])["spec-error"]

    assert len(errors) == 1


# ------------------------------------------------------------------ declared errata

ERRATUM = {
    "label": "a figure the report keeps and annotates as wrong",
    "anchor": r"^\| cell \|",
    "capture": r"\| ([0-9.]+) \|",
    "stat": "mean",
    "sources": ["cell/test/seed_*.json"],
    "expect": "mismatch",
    "note": "pins the ⚠ note under the table",
}


def test_a_declared_erratum_passes_while_it_stays_unreproducible(tree):
    _write(tree, "docs/report.md", "| cell | 0.925 |\n")

    buckets = run(tree, [ERRATUM])

    assert buckets["value-mismatch"] == [], "an erratum is not a value defect"
    assert buckets["erratum-stale"] == []
    assert len(buckets["ok"]) == 1


def test_an_erratum_that_starts_reproducing_is_a_defect(tree):
    """The figure now matches its source, so whatever the note says about it is stale.

    Without this the disclosure could be silently outlived by an edit to the table.
    """
    buckets = run(tree, [ERRATUM])          # the fixture doc prints the true 0.885

    assert len(buckets["erratum-stale"]) == 1
    assert "erratum-stale" in apn.DEFECT_BUCKETS, "so it fails --strict"


def test_an_erratum_claim_without_a_note_is_rejected(tree):
    claim = {k: v for k, v in ERRATUM.items() if k != "note"}

    errors = run(tree, [claim])["spec-error"]

    assert len(errors) == 1
    assert "note" in errors[0][1]


def test_an_unknown_expect_value_is_rejected(tree):
    errors = run(tree, [dict(MEAN_CLAIM, expect="maybe")])["spec-error"]

    assert len(errors) == 1


# ------------------------------------------------------------------ CLI and shipped specs


def test_cli_exits_nonzero_only_under_strict(tmp_path):
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    (spec_dir / "demo.json").write_text(json.dumps({
        "chain": "demo", "doc": "docs/nope.md", "root": "results/demo",
        "metric": "eval_success_rate", "claims": [],
    }), encoding="utf-8")

    def run_cli(*extra):
        return subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "scripts.audit_published_numbers",
             "--spec-dir", str(spec_dir), *extra],
            capture_output=True, text=True, encoding="utf-8", cwd=REPO_ROOT)

    assert run_cli().returncode == 0
    assert run_cli("--strict").returncode == 1


def test_the_shipped_specs_still_anchor_to_their_reports():
    """The one case that reads the real repo, and it needs no `results/`.

    Anchors and captures are resolved before any data lookup, so this runs in a clone.
    It fails when a report is edited out from under its spec -- which is the moment the
    traceback stops covering the number it claims to cover.
    """
    specs = apn.load_specs(apn.SPEC_DIR)
    assert specs, "no traceback specs found"

    for spec in specs:
        buckets = apn.run_spec(spec, str(REPO_ROOT / "no-such-data-root"), False)
        for name in ("spec-error", "anchor-missing", "anchor-ambiguous", "capture-failed"):
            assert buckets[name] == [], f"{spec['chain']}: {name} -> {buckets[name]}"


# ------------------------------------------------------------------ scope bounds

def test_before_alone_bounds_the_search_from_above(tree):
    """`after` on its own had a test; `before` on its own did not.

    It is the half the *first* of two identical tables depends on -- the last table in a
    document needs no lower bound, the first needs no upper one, and only one of those
    two cases was covered.
    """
    _write(tree, "docs/report.md", "\n".join([
        "| cell | 0.885 |", "**second table**", "| cell | 0.810 |", "",
    ]))

    assert run(tree, [MEAN_CLAIM])["anchor-ambiguous"] != [], "unscoped, it is ambiguous"

    buckets = run(tree, [dict(MEAN_CLAIM, before=r"^\*\*second table\*\*")])

    assert buckets["anchor-ambiguous"] == []
    assert [r[1] for r in buckets["ok"]] == ["docs/report.md:1"]


def test_a_bound_that_matches_nothing_in_scope_is_a_spec_error(tree):
    """Zero matches and two matches are both "cannot resolve", and only two was tested.

    A bound naming a heading that has since been renamed must not degrade into
    `anchor-missing`: that bucket reads as a defect against the report, when the thing
    that broke is the spec.
    """
    _write(tree, "docs/report.md", "**top**\n| cell | 0.885 |\n")

    errors = run(tree, [dict(MEAN_CLAIM, after=r"^\*\*renamed\*\*")])["spec-error"]

    assert len(errors) == 1
    assert "matches 0 lines" in errors[0][1]


# ---------------------------------------------------- the shipped online A0 chain

@pytest.fixture()
def online_a0() -> dict:
    specs = [s for s in apn.load_specs(apn.SPEC_DIR) if s["chain"] == "online_a0"]
    assert len(specs) == 1, "the online A0 traceback spec is not installed"
    return specs[0]


def _resolved(spec: dict) -> list[tuple[dict, str]]:
    """Every claim paired with the figure it captured, resolved without any `results/`.

    Anchoring and capture happen before the data lookup, so a clone gets the same
    answers here as the machine that holds `experiments/`.
    """
    buckets = apn.run_spec(spec, str(REPO_ROOT / "no-such-data-root"), False)
    records = buckets["no-data"]
    assert len(records) == len(spec["claims"]), "some claim did not reach the data lookup"
    for claim, record in zip(spec["claims"], records):
        assert record[2] == claim["sources"], "claim order and record order diverged"
    return [(c, r[1]) for c, r in zip(spec["claims"], records)]


def test_the_online_a0_chain_pins_the_terminal_evaluation(online_a0):
    """The provenance rule is `final_eval.json`, and that is the whole finding.

    Three reductions over the same run's `eval_log.csv` were tried when this chain was
    built and all three miss -- `max` included, which is the one the summary's own
    argument list invites by saying "best per-cell success >= 70%". A source repointed
    at the training curve would still recompute *something*, so the shape of the source
    is the only place that mistake can be caught mechanically.
    """
    for claim in online_a0["claims"]:
        for source in claim["sources"]:
            assert source.endswith("/final_eval.json"), source
            assert "eval_log" not in source, source
        assert "(" not in claim["metric"], "an aggregate belongs to a CSV, not to this chain"


def test_the_online_a0_chain_reads_each_cell_once(online_a0):
    """Both result tables carry the same header and the same two row labels.

    So the failure to guard against is not a missing number -- it is the same cell read
    twice under two labels, which leaves the count right and half the grid unchecked.
    Grouping is taken from `metric` and `sources`, never from the label, because the
    label is the one field a wrong claim would still describe correctly.
    """
    seen: dict[tuple[str, str, str], list[str]] = {}
    for claim, published in _resolved(online_a0):
        objective, column = claim["sources"][0].split("/")[:2]
        seen.setdefault((claim["metric"], objective, claim["stat"]), []).append(published)
        assert column in ("s0_k4", "s1_k4", "s2_k4"), column

    assert len(seen) == 2 * 2 * 2, f"expected four rows x two statistics, got {len(seen)}"
    for key, values in seen.items():
        assert len(values) == 3, f"{key}: a row has {len(values)} cells, not three"
        assert len(set(values)) == 3, f"{key}: two columns captured the same figure {values}"

    for objective in ("efficiency_v2", "arrival_v1"):
        for stat in ("mean", "sd"):
            success = set(seen[("eval_success_rate", objective, stat)])
            efficiency = set(seen[("eval_path_efficiency", objective, stat)])
            assert not (success & efficiency), (
                f"{objective} {stat}: the two tables captured a shared figure "
                f"{success & efficiency} -- one of them is being read twice")


def test_the_online_a0_chain_is_ambiguous_without_its_table_scope(online_a0):
    """The negative control for the scope: strip it and the chain must stop resolving.

    Without this, a claim that had lost its `after`/`before` would still anchor -- onto
    whichever of the two identical tables comes first -- and every other test here would
    keep passing.
    """
    stripped = dict(online_a0, claims=[
        {k: v for k, v in claim.items() if k not in ("after", "before")}
        for claim in online_a0["claims"]
    ])

    buckets = apn.run_spec(stripped, str(REPO_ROOT / "no-such-data-root"), False)

    assert len(buckets["anchor-ambiguous"]) == len(online_a0["claims"])


# ------------------------------------------------ the shipped chapter-5 chains

CH5_CHAINS = ("ch5_online", "ch5_boundary", "ch5_rebrac", "ch5_td3bc")

# One published `mean \pm sd` cell in a `.tex` line. Digits on both sides are what tells
# it from the bare `$\pm$` a table caption writes when it says "mean $\pm$ sd" in prose.
PM_CELL = re.compile(r"[0-9.]+ \\pm [0-9.]+")

# The prefix a capture repeats once per cell it has to walk past, verbatim as generated.
CELL_SKIP = r"(?:.*?[0-9.]+ \\pm [0-9.]+)"

# Chapter sources with no report-side counterpart, and why. Every other source must be a
# file some report chain also reads; adding to this list is how a new uncovered cell gets
# declared rather than slipping in.
#
# Empty as of 2026-08-24, and both ways it emptied are worth keeping. The two clean-probe
# entries went when `data_integrity_open_items` was built, which is what a declaration
# retiring correctly looks like. The two k-ladder entries were instead found *dead* by
# `covered` below: the arrival_v2 chain reads those exact `final_eval.json` globs for the
# section 7.9 dispersions, so the `or` never reached the declaration. What they recorded --
# that no report publishes the ladder's own success-rate cells -- is true and load-bearing,
# but it is a statement about a published cell, and this table keys on a file. That fact
# lives in `test_the_chapter_ladder_is_pinned_to_the_terminal_evaluation`, which is the
# test it actually justifies.
UNSHARED_WITH_THE_REPORTS: dict[str, str] = {}


@pytest.fixture()
def chapter_chains() -> list[dict]:
    specs = [s for s in apn.load_specs(apn.SPEC_DIR) if s["chain"] in CH5_CHAINS]
    assert len(specs) == len(CH5_CHAINS), "the chapter-5 traceback specs are not installed"
    return specs


def _anchored(spec: dict) -> list[tuple[dict, str, str]]:
    """(claim, its one anchored line, the figure it captured) -- no `results/` needed.

    The anchor is resolved inside the claim's own `section`/`after`/`before` scope, via
    the tool's own `_region`, because that is what the tool does. Searching the whole
    document instead makes an anchor that is unique in its section look ambiguous -- the
    worldcomp report has two screening tables whose `| 0.1 |` rows are identical, and only
    the heading above them says which is which.
    """
    lines = (REPO_ROOT / spec["doc"]).read_text(encoding="utf-8").split("\n")
    out = []
    for claim in spec["claims"]:
        lo, hi = apn._region(lines, claim)
        hits = [ln for i, ln in enumerate(lines, 1)
                if lo < i < hi and re.search(claim["anchor"], ln)]
        assert len(hits) == 1, f"{spec['chain']} :: {claim['label']}: {len(hits)} lines"
        found = re.search(claim["capture"], hits[0])
        assert found, f"{spec['chain']} :: {claim['label']}: capture missed"
        out.append((claim, hits[0], found))
    return out


def _repo_relative_sources(spec: dict) -> list[str]:
    return [posixpath.normpath(f"{spec['root']}/{source}")
            for claim in spec["claims"] for source in claim["sources"]]


def test_the_chapter_chains_capture_one_whole_published_cell(chapter_chains):
    r"""A capture must land inside one `mean \pm sd` cell, and its pair on the same one.

    The chapter puts up to four cells on a line -- `rebrac.tex` section 5.6 has a
    sentence carrying all four figures of a two-by-two comparison -- so the failure to
    guard is not a missing number but a mean read off one cell and its dispersion off the
    next. Anchoring on the character offset rather than on the captured text is
    deliberate: two cells on a line can print the same figure, and matching by text would
    call that agreement.

    `rebrac.tex`'s per-seed table caption is what makes this load-bearing: it says "mean
    $\pm$ cross-seed sd" in prose before quoting a figure, so any scheme that counted
    `\pm` occurrences instead of complete numeric cells would be off by one there.
    """
    for spec in chapter_chains:
        by_anchor: dict[tuple[str, str], dict[str, set[int]]] = {}
        for claim, line, found in _anchored(spec):
            spans = [m.span() for m in PM_CELL.finditer(line)]
            index = [i for i, (lo, hi) in enumerate(spans) if lo <= found.start(1) < hi]
            where = f"{spec['chain']} :: {claim['label']}"
            assert len(index) == 1, f"{where}: capture landed outside a published cell"
            seen = by_anchor.setdefault((claim["anchor"], claim.get("metric", "")), {})
            taken = seen.setdefault(claim["stat"], set())
            assert index[0] not in taken, f"{where}: two claims read cell {index[0]}"
            taken.add(index[0])

        for (anchor, _metric), stats in by_anchor.items():
            assert stats.get("mean") == stats.get("sd"), (
                f"{spec['chain']}: {anchor} reads means from cells {stats.get('mean')} "
                f"but dispersions from {stats.get('sd')}")


def test_the_chapter_captures_depend_on_the_cell_they_count_to(chapter_chains):
    """The negative control for the cell index: drop it and the figure must change.

    Every capture past the first cell is a prefix repeated once per cell walked over.
    Strip the prefix and the claim reads the line's first cell instead; if that came back
    with the same figure, the index would be carrying no weight and a slip in it would be
    invisible to every other test here.
    """
    checked = 0
    for spec in chapter_chains:
        for claim, line, found in _anchored(spec):
            stripped = claim["capture"]
            while stripped.startswith(CELL_SKIP):
                stripped = stripped[len(CELL_SKIP):]
            if stripped == claim["capture"]:
                continue
            checked += 1
            first = re.search(stripped, line)
            assert first, f"{claim['label']}: the index-0 form matched nothing"
            assert first.group(1) != found.group(1), (
                f"{spec['chain']} :: {claim['label']}: reading the first cell gives the "
                f"same figure {found.group(1)}, so the cell index proves nothing")
    assert checked >= 8, f"only {checked} claims walk past a cell; the control is thin"


def test_the_chapter_chains_read_the_files_the_reports_read(chapter_chains):
    """The two sides must resolve to the same per-seed files, not merely to equal figures.

    That is the property that makes a cross-side comparison mean anything: a chapter
    reading pinned to a different run would still recompute, and would still agree
    whenever the two runs happen to round the same way. It is checked as path arithmetic
    so it holds in a clone, where none of those files exist.

    Roots differ by a prefix -- `results/offline` against `results/offline/rebrac` -- so
    the comparison is on the repo-relative path, not on the glob as written.
    """
    reports = {source
               for spec in apn.load_specs(apn.SPEC_DIR) if spec["chain"] not in CH5_CHAINS
               for source in _repo_relative_sources(spec)}
    assert reports, "no report-side chains found to compare against"

    for spec in chapter_chains:
        for source in _repo_relative_sources(spec):
            assert source in reports or source in UNSHARED_WITH_THE_REPORTS, (
                f"{spec['chain']}: {source} is read by no report chain and is not "
                f"declared in UNSHARED_WITH_THE_REPORTS")

    used = {source for spec in chapter_chains for source in _repo_relative_sources(spec)}
    stale = set(UNSHARED_WITH_THE_REPORTS) - used
    assert not stale, f"declared as unshared but no longer cited: {sorted(stale)}"

    # The other direction, and the one a new report chain makes false: a source declared
    # unshared that some report chain now does read. The `or` above short-circuits on the
    # declaration, so nothing else would ever revisit it -- `data_integrity_open_items`
    # is exactly the chain that turned two of these entries false.
    covered = set(UNSHARED_WITH_THE_REPORTS) & reports
    assert not covered, (
        f"declared as read by no report chain, but one reads them: {sorted(covered)}")


def test_the_chapter_ladder_is_pinned_to_the_terminal_evaluation(chapter_chains):
    """The ladder's rule was recomputed, not read: it is `final_eval.json`, per seed.

    No report publishes the k=8 and k=12 rows, so nothing else records what those figures
    mean. Every reduction over the same runs' `eval_log.csv` was tried and all of them
    miss -- `max` gives 0.90 where 0.88 is published for k=12 -- but a source repointed at
    the training curve would still recompute *something*, so the shape of the source is
    where that mistake has to be caught.
    """
    ladder = [source
              for spec in chapter_chains for source in _repo_relative_sources(spec)
              if "sac_vanilla/" in source]
    assert len(ladder) >= 14, f"only {len(ladder)} ladder sources found"
    for source in ladder:
        assert source.endswith("/results/final_eval.json"), source
        assert "eval_log" not in source, source


# ------------------------------------------------------- the specs and their generators
#
# Every spec here was emitted by a script, because hand-typing positional capture regexes
# is how a table ends up quietly checking the wrong column. Until 2026-08-24 those scripts
# lived only in the session scratchpad that produced them, so the tables were unextendable
# the moment the session ended. They now sit in `docs/tracebacks/_gen/`, and these two
# tests are what keeps them honest: the first proves each one still emits exactly what is
# committed, the second stops a new spec from arriving without one.

GEN_DIR = REPO_ROOT / "docs/tracebacks/_gen"

# spec file -> the generator that writes it. One generator may write several specs.
SPEC_GENERATORS = {
    "arrival_v2.json": "gen_arrival_spec.py",
    "online_a0.json": "gen_online_a0_spec.py",
    "rebrac.json": "gen_rebrac_spec.py",
    "td3bc_phase0c.json": "gen_td3bc_spec.py",
    "td3bc_worldcomp_teacher_gap.json": "gen_worldcomp_spec.py",
    "data_integrity_open_items.json": "gen_data_integrity_spec.py",
    "benchmarks_and_datasets.json": "gen_benchmarks_datasets_spec.py",
    "ch5_online.json": "gen_ch5_specs.py",
    "ch5_boundary.json": "gen_ch5_specs.py",
    "ch5_rebrac.json": "gen_ch5_specs.py",
    "ch5_td3bc.json": "gen_ch5_specs.py",
}

# A spec with no generator must be declared, with the reason -- the same discipline
# UNSHARED_WITH_THE_REPORTS applies to a source no report reads.
UNGENERATED_SPECS = {
    "fql_succession_p2.json":
        "emitted in the session that landed 2a8c311 and not recovered when the other five "
        "generators were; its 35 claims stand, but extending this one still means editing "
        "the JSON by hand.",
}


def test_every_committed_spec_can_be_regenerated(tmp_path):
    """Byte for byte, from the generator alone.

    This is what makes the generator load-bearing rather than a historical note: edit a
    spec by hand and this goes red, so the edit has to go into the generator instead.

    Compared with line endings normalised on purpose. `core.autocrlf` is true on the
    Windows machine, so a checked-out spec is CRLF while a freshly written one is LF --
    `git status` hides that difference and a raw byte comparison would fail on it alone.
    """
    for name, generator in sorted(SPEC_GENERATORS.items()):
        proc = subprocess.run([sys.executable, "-X", "utf8", str(GEN_DIR / generator),
                               str(tmp_path)],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=REPO_ROOT)
        assert proc.returncode == 0, f"{generator}: {proc.stderr[-800:]}"

        regenerated = tmp_path / name
        assert regenerated.exists(), f"{generator} did not write {name}"
        committed = (Path(apn.SPEC_DIR) / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert regenerated.read_text(encoding="utf-8").replace("\r\n", "\n") == committed, (
            f"{name} differs from what {generator} emits -- if the spec was hand-edited, "
            f"move the edit into the generator")


# ------------------------------------- the shipped report-side markdown-table chains

# Two report chains read markdown tables by counting cells, the way the chapter chains
# count `mean \pm sd` cells in a `.tex` line. Nothing but the anchor sweep looked at them.
REPORT_TABLE_CHAINS = ("td3bc_worldcomp_teacher_gap", "data_integrity_open_items")

# The prefix such a capture opens with, verbatim as generated: `(?:[^|]*\|){N}`, where N
# counts the pipes walked past -- the empty cell before the leading pipe, the anchor's own
# cell, and then one per column.
MD_CELL_SKIP = re.compile(r"^\(\?:\[\^\|\]\*\\\|\)\{(\d+)\}")

MD_NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")


@pytest.fixture()
def table_chains() -> list[dict]:
    specs = [s for s in apn.load_specs(apn.SPEC_DIR) if s["chain"] in REPORT_TABLE_CHAINS]
    assert len(specs) == len(REPORT_TABLE_CHAINS), "the table-reading chains are not installed"
    return specs


def _md_cells(claim: dict, line: str) -> tuple[int, list[str]] | None:
    """(index of the cell this claim's skip count names, the line split on pipes)."""
    skip = MD_CELL_SKIP.match(claim["capture"])
    if skip is None:
        return None
    return int(skip.group(1)), line.split("|")


def test_the_report_table_chains_index_a_cell_that_carries_weight(table_chains):
    """The negative control for the cell index: a neighbour must not print the same figure.

    A slipped count is caught by the recomputation, but only where `results/` is present.
    In a clone the count is unchecked, and the case that makes it uncheckable *anywhere*
    is a row whose adjacent cells hold the same number -- then the index proves nothing
    and a slip is invisible on both sides. This is the test that says the rows these two
    chains read are not like that.

    `checked` is asserted from below for the same reason the chapter version is: a claim
    that loses its skip prefix silently drops out of the control, so hollowing it out has
    to fail as loudly as breaking it.
    """
    checked = 0
    for spec in table_chains:
        for claim, line, found in _anchored(spec):
            split = _md_cells(claim, line)
            if split is None:
                continue
            index, cells = split
            assert 0 <= index < len(cells), (
                f"{spec['chain']} :: {claim['label']}: cell {index} is off the row")
            checked += 1
            figure = float(apn._norm(found.group(1)))
            for step in (-1, 1):
                near = index + step
                if not 0 <= near < len(cells):
                    continue
                clash = [t for t in MD_NUMBER.findall(cells[near])
                         if abs(float(t) - figure) < 1e-12]
                assert not clash, (
                    f"{spec['chain']} :: {claim['label']}: cell {near} also prints "
                    f"{clash[0]}, so the index {index} proves nothing")
    assert checked >= 45, f"only {checked} claims count cells; the control is thin"


def test_no_two_report_table_claims_read_the_same_cell_of_a_row(table_chains):
    """One published cell, one claim. Two claims on one cell is an index that slipped.

    Keyed by the metric as well as the statistic, because a row legitimately reads two
    different metrics out of two different columns -- the worldcomp screening rows take a
    success rate and a return off the same anchor, and those are not duplicates.

    The boundary this shares with the chapter chains, measured rather than assumed: a
    *pair* of indices sliding together onto a cell no claim pins stays green here, and
    only the recomputation sees it. The two layers are not redundant -- a clone has only
    this one.
    """
    for spec in table_chains:
        seen: dict[tuple[str, str, str, str], set[int]] = {}
        for claim, line, _found in _anchored(spec):
            split = _md_cells(claim, line)
            if split is None:
                continue
            index, _cells = split
            # The section is part of the key: the worldcomp report's two screening
            # tables carry the same `| 0.1 |` anchor, and only the heading separates them.
            key = (claim.get("section", ""), claim["anchor"],
                   claim.get("metric", ""), claim["stat"])
            taken = seen.setdefault(key, set())
            assert index not in taken, (
                f"{spec['chain']} :: {claim['label']}: cell {index} is already read by "
                f"another {claim['stat']} claim on this row")
            taken.add(index)


def test_every_spec_declares_whether_it_has_a_generator(tmp_path):
    """A new spec arriving with no generator is the failure this line already had once.

    Nothing else notices: the sweep passes, `--strict` passes, and the gap only surfaces
    the next time somebody needs to extend the table and finds no way in but the JSON.
    """
    shipped = {p.name for p in Path(apn.SPEC_DIR).glob("*.json")}
    accounted = set(SPEC_GENERATORS) | set(UNGENERATED_SPECS)

    assert shipped == accounted, (
        f"undeclared: {sorted(shipped - accounted)} | "
        f"declared but absent: {sorted(accounted - shipped)}")
    assert not (set(SPEC_GENERATORS) & set(UNGENERATED_SPECS))
    for name, generator in SPEC_GENERATORS.items():
        assert (GEN_DIR / generator).exists(), f"{name} names a missing generator"


def test_a_spec_declared_ungenerated_is_not_emitted_by_any_generator(tmp_path):
    """A false `UNGENERATED_SPECS` entry silences its own alarm, so it is cross-checked.

    The same failure `check_doc_pointers` has with 自述缺席 declarations: the escape hatch
    is only as good as the claim it rests on, and nothing re-reads that claim. Move a spec
    into this table by mistake -- or leave it there after writing its generator -- and
    `test_every_spec_declares_whether_it_has_a_generator` goes on passing while the spec is
    no longer hand-maintained at all.
    """
    # Every generator in the tree, not the declared ones: a false declaration is exactly
    # the case where the generator has been dropped from SPEC_GENERATORS, so grading the
    # declared set is grading the mutation's own story. Measured -- with the declared set
    # the injection that moves a spec into UNGENERATED_SPECS stays green.
    for generator in sorted(p.name for p in GEN_DIR.glob("gen_*.py")):
        proc = subprocess.run([sys.executable, "-X", "utf8", str(GEN_DIR / generator),
                               str(tmp_path)],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=REPO_ROOT)
        assert proc.returncode == 0, f"{generator}: {proc.stderr[-800:]}"

    emitted = {p.name for p in tmp_path.glob("*.json")}
    overlap = emitted & set(UNGENERATED_SPECS)
    assert not overlap, (
        f"declared as having no generator, but one writes it: {sorted(overlap)}")
