"""Tests for the chapter's dispersion-convention classifier.

`ch5_dispersion_audit.py` answers one question about each ``\\pm`` in the chapter: is that
sd a population (ddof=0) or a sample (ddof=1) figure? It answers only where the row prints
its own per-seed values, and says "needs the ground-truth report" everywhere else. Both
halves matter: the classification, and the refusal.

The refusal is the part that is easy to erode. Two of the chapter's tables put per-seed
cells on opposite sides of the mean, and other columns (beta values, seed counts) are
decimals too -- so the candidate seeds are accepted only when they reproduce the published
mean at the mean's own printed precision. Drop that guard and the tool starts classifying
rows from whatever decimals happened to sit in them, which is worse than reporting nothing
because it looks like an answer.

The corpus is a fixture, never the real chapter: pointed at `sections/`, every assertion
here would restate a count that expires on the next edit.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "paper" / "thesis_ch5" / "tools"

if str(TOOLS) not in sys.path:
    sys.path.append(str(TOOLS))

audit = importlib.import_module("ch5_dispersion_audit")

# {0.900, 0.867, 0.933}: mean 0.900, ddof0 0.027, ddof1 0.033. Three-digit printing keeps
# the two conventions apart, which is what makes the row classifiable at all.
SEEDS_BEFORE = r"ReBRAC & 0.900 & 0.867 & 0.933 & 0.900 \pm 0.027 \\"
SEEDS_AFTER = r"ReBRAC & 0.900 \pm 0.033 & 5 & 0.900 & 0.867 & 0.933 \\"


@pytest.fixture
def sections(tmp_path, monkeypatch):
    root = tmp_path / "sections"
    root.mkdir()
    monkeypatch.setattr(audit, "SECTIONS", root)
    return root


def write(root: Path, name: str, *lines: str) -> None:
    (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def only(readings: list) -> object:
    assert len(readings) == 1, readings
    return readings[0]


# --------------------------------------------------------------------------- #
# the two table layouts
# --------------------------------------------------------------------------- #
def test_per_seed_cells_before_the_mean_settle_the_row(sections):
    write(sections, "a.tex", SEEDS_BEFORE)
    r = only(audit.collect())
    assert r.resolved
    assert r.seeds == [0.900, 0.867, 0.933]
    assert r.verdict()[2] == "ddof=0"


def test_per_seed_cells_after_the_mean_settle_the_row(sections):
    write(sections, "a.tex", SEEDS_AFTER)
    r = only(audit.collect())
    assert r.resolved
    assert r.seeds == [0.900, 0.867, 0.933]
    assert r.verdict()[2] == "ddof=1"


def test_a_seed_count_column_is_not_mistaken_for_a_seed(sections):
    # `5` in the seed-count column is not a decimal, so the NUM pattern skips it. If it
    # were read as a seed the mean check would fail and the row would go unresolved --
    # a silent loss, not an error.
    write(sections, "a.tex", SEEDS_AFTER)
    assert only(audit.collect()).seeds == [0.900, 0.867, 0.933]


# --------------------------------------------------------------------------- #
# the refusal
# --------------------------------------------------------------------------- #
def test_decimals_that_do_not_reproduce_the_mean_are_not_taken_as_seeds(sections):
    # beta values in the leading columns: decimals, but not seeds.
    write(sections, "a.tex", r"ReBRAC & 0.5 & 0.1 & 0.750 \pm 0.020 \\")
    r = only(audit.collect())
    assert not r.resolved
    assert r.seeds == []


def test_a_single_printed_value_cannot_settle_a_row(sections):
    # Two thresholds guard this, and only one of them shows up in `resolved`: the
    # candidate scan also requires more than one value before it accepts the cells as
    # seeds at all. Asserting `not resolved` alone leaves that one untested -- it is
    # shadowed by the identical check on the property.
    write(sections, "a.tex", r"ReBRAC & 0.900 & 0.900 \pm 0.000 \\")
    r = only(audit.collect())
    assert not r.resolved
    assert r.seeds == []


def test_a_row_with_no_per_seed_cells_at_all_is_unresolved(sections):
    write(sections, "a.tex", r"ReBRAC & 0.900 \pm 0.027 \\")
    assert not only(audit.collect()).resolved


def test_the_mean_must_reproduce_at_its_own_printed_precision(sections):
    # {0.900, 0.866, 0.933} averages to 0.899666..., which prints as 0.900 at three
    # digits. The guard compares at the mean's precision, not the sd's.
    write(sections, "a.tex", r"ReBRAC & 0.900 & 0.866 & 0.933 & 0.900 \pm 0.027 \\")
    assert only(audit.collect()).resolved


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #
def test_a_dispersion_matching_both_conventions_is_reported_as_either(sections):
    # Identical seeds: both conventions give exactly 0. The tool must say so rather than
    # pick one -- this is the shape of the chapter's own near-zero cells.
    write(sections, "a.tex", r"零 & 0.900 & 0.900 & 0.900 & 0.900 \pm 0.000 \\")
    assert only(audit.collect()).verdict()[2] == "either"


def test_a_dispersion_matching_neither_convention_is_reported_as_neither(sections):
    # The arrival_v2 §7.9.4 shape: a sd that is neither sd of the seeds beside it,
    # because it was transposed from another row. NEITHER is the finding, not an error.
    write(sections, "a.tex", r"k=12 & 0.900 & 0.867 & 0.933 & 0.900 \pm 0.064 \\")
    sd0, sd1, tag = only(audit.collect()).verdict()
    assert tag == "NEITHER"
    assert round(sd0, 3) == 0.027 and round(sd1, 3) == 0.033


def test_coarse_printing_can_make_the_two_conventions_indistinguishable(sections):
    """0.027 and 0.033 both print as 0.03, so a two-digit cell settles nothing.

    The same seeds classify as ddof=0 at three digits. Reporting `either` here rather
    than picking the nearer one is the point: a convention read off a rounded cell is a
    guess, and the whole tally exists to stop guesses being quoted as findings.
    """
    write(sections, "a.tex", r"ReBRAC & 0.900 & 0.867 & 0.933 & 0.900 \pm 0.03 \\")
    assert only(audit.collect()).verdict()[2] == "either"


# --------------------------------------------------------------------------- #
# corpus rules
# --------------------------------------------------------------------------- #
def test_a_commented_out_row_is_not_a_published_reading(sections):
    # The rev headers quote superseded numbers freely; counting them would inflate every
    # tally and could classify a figure that was removed from the chapter.
    write(sections, "a.tex",
          r"% 旧稿曾印 0.900 \pm 0.099，已改",
          SEEDS_BEFORE)
    assert len(audit.collect()) == 1


def test_every_tex_file_in_sections_is_scanned(sections):
    write(sections, "a.tex", SEEDS_BEFORE)
    write(sections, "b.tex", SEEDS_AFTER)
    assert {r.file for r in audit.collect()} == {"a.tex", "b.tex"}


def test_two_readings_on_one_line_are_both_collected(sections):
    write(sections, "a.tex",
          r"两格 & 0.900 \pm 0.027 & 0.800 \pm 0.011 \\")
    assert len(audit.collect()) == 2


def test_the_line_number_is_reported(sections):
    write(sections, "a.tex", r"\begin{tabular}{lc}", SEEDS_BEFORE)
    assert only(audit.collect()).line == 2


# --------------------------------------------------------------------------- #
# --convert
# --------------------------------------------------------------------------- #
def test_convert_prints_the_value_under_the_other_convention(sections, monkeypatch,
                                                             capsys):
    write(sections, "a.tex", SEEDS_BEFORE)
    monkeypatch.setattr(sys, "argv", ["tool", "--convert", "ddof1"])
    audit.main()
    assert "->  0.900 +/- 0.033" in capsys.readouterr().out


def test_convert_says_unchanged_when_the_row_already_uses_that_convention(
        sections, monkeypatch, capsys):
    write(sections, "a.tex", SEEDS_BEFORE)
    monkeypatch.setattr(sys, "argv", ["tool", "--convert", "ddof0"])
    audit.main()
    assert "unchanged" in capsys.readouterr().out


def test_the_summary_counts_resolved_and_unresolved_separately(sections, monkeypatch,
                                                               capsys):
    write(sections, "a.tex", SEEDS_BEFORE, r"其他 & 0.750 \pm 0.020 \\")
    monkeypatch.setattr(sys, "argv", ["tool"])
    audit.main()
    out = capsys.readouterr().out
    assert "2 pm readings in the chapter body; 1 settle themselves" in out
    assert "ddof=0=1" in out
    assert re.search(r"^  a\.tex\s+1$", out, re.M), out
