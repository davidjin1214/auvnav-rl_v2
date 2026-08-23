"""Tests for the §5.9 SAC-ladder dispersion check.

This is the tool that answers what `ch5_dispersion_audit.py` refuses to: the ladder table
prints no per-seed values, so the convention has to come from the per-run records under
`results/`. It is also the existing precedent for reading the published figure out of the
`.tex` at run time instead of transcribing it -- a transcribed figure keeps agreeing with
itself after the manuscript changes, which is the one failure the check exists to prevent.

Two things it enforces beyond the convention verdict, both tested here: the published mean
must be reproducible from the records (that is the exit-code criterion; the sd verdicts are
reported, not enforced), and the two readings quoted in the running text must be character-
identical to the table cells they quote.

`results/` is gitignored, so every source tree here is a fixture. Pointed at the real one,
these assertions would be restating today's numbers.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "paper" / "thesis_ch5" / "tools"

if str(TOOLS) not in sys.path:
    sys.path.append(str(TOOLS))

ladder = importlib.import_module("ch5_sac_ladder_dispersion_check")

# mean 0.900, ddof=0 0.027, ddof=1 0.033 -- three digits keep the conventions apart.
R1 = [0.900, 0.867, 0.933]
# mean 0.850, ddof=0 0.041, ddof=1 0.050.
R2 = [0.850, 0.800, 0.900]

ROWS = {"随机": "random", "中等": "medium", "中等偏专家": "mexp", "专家": "expert"}
COLUMNS = ["fql", "rebrac/b1_1p0"]

TABLE = "\n".join([
    r"随机 & 0.900 \pm 0.027 & 0.900 \pm 0.027 \\",
    r"中等 & 0.900 \pm 0.027 & 0.900 \pm 0.027 \\",
    r"中等偏专家 & 0.900 \pm 0.027 & 0.850 \pm 0.041 \\",
    r"专家 & 0.900 \pm 0.027 & 0.900 \pm 0.027 \\",
])
QUOTE = r"正文里 FQL 达 $0.900 \pm 0.027$，高于 ReBRAC 的 $0.850 \pm 0.041$。"
SUPPLEMENT = r"跨源补充的成功率 $0.900 \pm 0.027$ 不在表内。"


def seeds(root: Path, rel: str, rates: list[float]) -> None:
    for seed, rate in zip(ladder.SEEDS, rates):
        d = root / rel / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "test_result.json").write_text(json.dumps({"eval_success_rate": rate}),
                                            encoding="utf-8")


@pytest.fixture
def ladder_tree(tmp_path, monkeypatch):
    """A miniature ladder: four tiers, two columns, plus the cross-source supplement."""
    h2h = tmp_path / "results"
    monkeypatch.setattr(ladder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ladder, "H2H", h2h)
    monkeypatch.setattr(ladder, "ROWS", ROWS)
    monkeypatch.setattr(ladder, "COLUMNS", COLUMNS)
    monkeypatch.setattr(ladder, "TEX", tmp_path / "algo_compare.tex")

    for tier in ROWS.values():
        for column in COLUMNS:
            rates = R2 if (tier == "mexp" and column == "rebrac/b1_1p0") else R1
            seeds(h2h, f"{column}/{tier}", rates)
    # Spelt out rather than taken from ladder.XSOURCE: which run backs the sentence
    # in the running text is a binding, not a fixture parameter. Reading the constant
    # here would make the fixture follow it anywhere it was repointed, and a figure
    # quietly validated against a different experiment is the failure this tool is for.
    seeds(h2h, "xsource_supplement/b1_1p0/m_multi_mix", R1)
    tex(tmp_path, TABLE, QUOTE, SUPPLEMENT)
    return tmp_path


def tex(root: Path, *blocks: str) -> None:
    (root / "algo_compare.tex").write_text("\n".join(blocks) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# the exit-code criterion: every published mean must be reproducible
# --------------------------------------------------------------------------- #
def test_a_ladder_whose_means_all_reproduce_passes(ladder_tree, capsys):
    assert ladder.main() == 0
    out = capsys.readouterr().out
    assert "all published means reproduced" in out
    # 8 table cells + the cross-source supplement quoted in the running text.
    assert "verdict tally: ddof=0=9" in out


def test_a_drifted_published_mean_fails(ladder_tree, capsys):
    # The manuscript moved and the records did not: exactly the drift that transcribing
    # the figure into the tool would hide.
    tex(ladder_tree, TABLE.replace(r"随机 & 0.900", r"随机 & 0.910"), QUOTE, SUPPLEMENT)
    assert ladder.main() == 1
    out = capsys.readouterr().out
    assert "MEAN MISMATCH" in out and "0.900" in out


def test_a_missing_source_directory_fails_and_names_it(ladder_tree, capsys):
    (ladder_tree / "results" / "fql" / "random" / "seed_7" / "test_result.json").unlink()
    assert ladder.main() == 1
    out = capsys.readouterr().out
    assert "SOURCE MISSING" in out
    assert "seed_7" in out


def test_the_sd_verdict_is_reported_but_does_not_change_the_exit_code(ladder_tree, capsys):
    # A sd matching neither convention is a finding to read, not a gate: the gate is the
    # mean. Conflating them would make every rounding quirk block the chapter.
    tex(ladder_tree, TABLE.replace(r"随机 & 0.900 \pm 0.027", r"随机 & 0.900 \pm 0.099"),
        QUOTE, SUPPLEMENT)
    assert ladder.main() == 0
    assert "NEITHER" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the convention verdict
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("published, expected", [
    ("0.027", "ddof=0"),
    ("0.033", "ddof=1"),
    ("0.099", "NEITHER"),
])
def test_the_verdict_names_the_convention(published, expected):
    assert ladder.verdict(published, R1, 3)[2] == expected


def test_identical_seeds_settle_nothing_and_are_reported_as_either():
    assert ladder.verdict("0.000", [0.9, 0.9, 0.9], 3)[2] == "either"


def test_the_verdict_is_taken_at_the_published_precision():
    # 0.027 and 0.033 both print as 0.03: at two digits the row settles nothing.
    assert ladder.verdict("0.03", R1, 2)[2] == "either"


# --------------------------------------------------------------------------- #
# reading the table out of the .tex
# --------------------------------------------------------------------------- #
def test_a_tier_row_that_is_not_in_the_table_is_an_error(ladder_tree):
    # Silently skipping would shrink the audit without saying so.
    tex(ladder_tree, TABLE.replace(r"专家 & 0.900 \pm 0.027 & 0.900 \pm 0.027 \\", ""),
        QUOTE, SUPPLEMENT)
    with pytest.raises(LookupError, match="row not found"):
        ladder.main()


def test_a_row_with_the_wrong_number_of_cells_is_an_error(ladder_tree):
    tex(ladder_tree, TABLE.replace(r"随机 & 0.900 \pm 0.027 & 0.900 \pm 0.027 \\",
                                   r"随机 & 0.900 \pm 0.027 \\"),
        QUOTE, SUPPLEMENT)
    with pytest.raises(LookupError, match="expected 2"):
        ladder.main()


def test_the_cross_source_supplement_in_the_running_text_is_checked_too(ladder_tree,
                                                                       capsys):
    assert ladder.main() == 0
    assert "m_multi_mix" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the in-text quotations
# --------------------------------------------------------------------------- #
def test_a_quotation_that_is_not_a_cell_of_its_row_fails(ladder_tree, capsys):
    tex(ladder_tree, TABLE, QUOTE.replace("0.850 \\pm 0.041", "0.860 \\pm 0.041"),
        SUPPLEMENT)
    assert ladder.main() == 1
    assert "NOT IN TABLE ROW" in capsys.readouterr().out


def test_a_quotation_spaced_differently_from_its_cell_still_matches(ladder_tree, capsys):
    # `$0.900\pm0.027$` and `$0.900 \pm 0.027$` typeset identically; only the whitespace
    # is normalised, so a changed digit still fails.
    tex(ladder_tree, TABLE, QUOTE.replace(r"$0.900 \pm 0.027$", r"$0.900\pm0.027$"),
        SUPPLEMENT)
    assert ladder.main() == 0
    assert "matches table" in capsys.readouterr().out


def test_quotations_that_have_gone_missing_are_a_failure(ladder_tree, capsys):
    # The sentence was rewritten and the check silently stopped covering anything.
    tex(ladder_tree, TABLE, SUPPLEMENT)
    assert ladder.main() == 1
    assert "quotations of the mexp cells were not found" in capsys.readouterr().out
