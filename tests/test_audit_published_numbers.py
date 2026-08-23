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
