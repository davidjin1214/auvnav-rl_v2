"""Tests for the Chapter 5 gate checks that read the LaTeX build products.

`ch5_floats.py`, `ch5_order.py`, `ch5_refs.py` and the compile gate inside
`ch5_check_all.py` are what `ch5_check_all.py` reports PASS/FAIL from, and that verdict
is quoted into review rounds. All four read `main.aux` / `main.log` / `main.pdf`, none of
which is in git -- so the fixtures here synthesise those products rather than requiring a
build. A test that needs `latexmk` is a test that does not run on a clone.

Three exit codes matter and are distinguished throughout: 0 = pass, 1 = a defect in the
chapter, 2 = no current build. Collapsing 2 into 1 would report a missing artefact as a
chapter defect, which is the same mistake as scoring a gitignored `results/` tree as a
failed audit.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "paper" / "thesis_ch5" / "tools"

if str(TOOLS) not in sys.path:
    sys.path.append(str(TOOLS))

floats = importlib.import_module("ch5_floats")
order = importlib.import_module("ch5_order")
refs = importlib.import_module("ch5_refs")
check_all = importlib.import_module("ch5_check_all")

SECTIONS = ["alpha", "beta"]
SECTION_NO = {"alpha": "5.1", "beta": "5.2"}


def aux(*entries: tuple[str, str, int]) -> str:
    """`\\newlabel` lines in the shape latexmk writes them."""
    return "".join(
        f"\\newlabel{{{key}}}{{{{{num}}}{{{page}}}{{}}{{}}{{}}}}\n"
        for key, num, page in entries
    )


@pytest.fixture
def chapter(tmp_path, monkeypatch):
    """A chapter directory whose build products the test writes itself."""
    root = tmp_path / "thesis_ch5"
    (root / "sections").mkdir(parents=True)
    for module in (floats, order, refs, check_all):
        monkeypatch.setattr(module, "CHAPTER_DIR", str(root), raising=False)
    for module in (order, refs):
        monkeypatch.setattr(module, "SECTION_ORDER", SECTIONS)
        monkeypatch.setattr(module, "SECTION_NO", SECTION_NO)
    for stem in SECTIONS:
        (root / "sections" / f"{stem}.tex").write_text("", encoding="utf-8")
    return root


def write(root: Path, stem: str, text: str) -> None:
    (root / "sections" / f"{stem}.tex").write_text(text, encoding="utf-8")


def argv(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["tool", *args])


# =========================================================================== #
# ch5_order -- float numbering vs first-citation order
# =========================================================================== #
def test_numbering_that_follows_first_citation_order_passes(chapter, monkeypatch, capsys):
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 3), ("tab:b", "5.2", 5)),
                                      encoding="utf-8")
    write(chapter, "alpha", "先引 \\ref{tab:a} 再引 \\ref{tab:b}。\n")
    argv(monkeypatch)
    assert order.main() == 0
    assert "PASS" in capsys.readouterr().out


def test_a_float_numbered_before_one_cited_earlier_is_an_inversion(chapter, monkeypatch,
                                                                   capsys):
    # The defect four review rounds missed (2026-07-28): tab:ch5_rebrac_perseed was cited
    # in §5.7.2 but placed in §5.7.3, so it was numbered after a table cited later.
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 3), ("tab:b", "5.2", 5)),
                                      encoding="utf-8")
    write(chapter, "alpha", "先引 \\ref{tab:b} 后引 \\ref{tab:a}。\n")
    argv(monkeypatch)
    assert order.main() == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "tab:a" in out


def test_tables_and_figures_are_ordered_independently(chapter, monkeypatch):
    # A 图 cited after a 表 does not invert anything -- the two series are numbered
    # separately, so sharing one highwater mark would invent failures.
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 3), ("fig:a", "5.1", 4)),
                                      encoding="utf-8")
    write(chapter, "alpha", "先引 \\ref{fig:a} 后引 \\ref{tab:a}。\n")
    argv(monkeypatch)
    assert order.main() == 0


def test_a_float_the_text_never_cites_is_reported_but_is_not_an_inversion(
        chapter, monkeypatch, capsys):
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 3), ("tab:b", "5.2", 5)),
                                      encoding="utf-8")
    write(chapter, "alpha", "只引 \\ref{tab:a}。\n")
    argv(monkeypatch)
    assert order.main() == 0
    assert "正文无" in capsys.readouterr().out


def test_a_citation_inside_a_comment_does_not_set_the_first_citation(chapter, monkeypatch):
    # The rev blocks at the head of each section quote \ref{} freely. Counting those
    # would date a float's first citation to a commentary line no reader ever sees.
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 3), ("tab:b", "5.2", 5)),
                                      encoding="utf-8")
    write(chapter, "alpha", "% rev: 这里提到 \\ref{tab:b} 只是修订说明\n"
                            "先引 \\ref{tab:a} 再引 \\ref{tab:b}。\n")
    argv(monkeypatch)
    assert order.main() == 0


def test_the_first_citation_counts_not_the_last(chapter, monkeypatch):
    # "First-citation order" is the whole criterion: a float cited early and again
    # late must be ordered by the early mention, or a recap sentence at the end of a
    # section silently re-dates it.
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 3), ("tab:b", "5.2", 5)),
                                      encoding="utf-8")
    write(chapter, "alpha", "先引 \\ref{tab:a}，再引 \\ref{tab:b}。\n"
                            "小结里又提一次 \\ref{tab:a}。\n")
    argv(monkeypatch)
    assert order.main() == 0


def test_order_without_a_build_exits_two_not_one(chapter, monkeypatch):
    argv(monkeypatch)
    assert order.main() == 2


# =========================================================================== #
# ch5_refs -- cross-reference resolution
# =========================================================================== #
def test_every_reference_resolving_to_a_heading_passes(chapter, monkeypatch, capsys):
    (chapter / "main.aux").write_text(aux(("subsec:x", "5.1.1", 3)), encoding="utf-8")
    write(chapter, "alpha", "\\subsection{某小节}\\label{subsec:x}\n")
    write(chapter, "beta", "见 §\\ref{subsec:x} 的论证。\n")
    argv(monkeypatch)
    assert refs.main() == 0
    assert "PASS" in capsys.readouterr().out


def test_a_reference_with_no_heading_behind_it_fails(chapter, monkeypatch, capsys):
    (chapter / "main.aux").write_text(aux(("subsec:x", "5.1.1", 3)), encoding="utf-8")
    write(chapter, "beta", "见 §\\ref{subsec:ghost} 的论证。\n")
    argv(monkeypatch)
    assert refs.main() == 1
    assert "subsec:ghost" in capsys.readouterr().out


def test_a_heading_nobody_cites_is_reported_but_is_not_a_failure(chapter, monkeypatch,
                                                                 capsys):
    (chapter / "main.aux").write_text(aux(("subsec:x", "5.1.1", 3)), encoding="utf-8")
    write(chapter, "alpha", "\\subsection{无人引用}\\label{subsec:x}\n")
    argv(monkeypatch)
    assert refs.main() == 0
    assert "从未被引用 1 个" in capsys.readouterr().out


def test_a_reference_inside_a_comment_is_not_counted(chapter, monkeypatch):
    # A rev block quoting a label the draft no longer defines would otherwise be
    # reported as an unresolved cross-reference in text nobody sees.
    (chapter / "main.aux").write_text(aux(("subsec:x", "5.1.1", 3)), encoding="utf-8")
    write(chapter, "alpha", "\\subsection{某小节}\\label{subsec:x}\n")
    write(chapter, "beta", "% rev: 旧稿曾引 \\ref{subsec:ghost}，已删\n"
                           "见 §\\ref{subsec:x} 的论证。\n")
    argv(monkeypatch)
    assert refs.main() == 0


def test_refs_without_a_build_exits_two_not_one(chapter, monkeypatch):
    argv(monkeypatch)
    assert refs.main() == 2


# =========================================================================== #
# ch5_floats -- placement distance
# =========================================================================== #
def pages(monkeypatch, *page_texts: str) -> None:
    monkeypatch.setattr(floats, "read_pages", lambda: list(page_texts))


def test_a_float_within_two_pages_of_its_first_citation_passes(chapter, monkeypatch,
                                                               capsys):
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 3)), encoding="utf-8")
    pages(monkeypatch, "见表 5.1。", "", "表 5.1 的标题在这一页。")
    argv(monkeypatch)
    assert floats.main() == 0
    assert "PASS" in capsys.readouterr().out


def test_the_two_page_criterion_is_inclusive(chapter, monkeypatch):
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 3)), encoding="utf-8")
    pages(monkeypatch, "见表 5.1。", "", "标题页")
    argv(monkeypatch)
    assert floats.main() == 0  # distance is exactly 2


def test_a_float_further_than_two_pages_from_its_first_citation_fails(chapter, monkeypatch,
                                                                      capsys):
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 4)), encoding="utf-8")
    pages(monkeypatch, "见表 5.1。", "", "", "标题页")
    argv(monkeypatch)
    assert floats.main() == 1
    assert "越线" in capsys.readouterr().out


def test_a_number_that_never_appears_in_the_text_is_not_a_violation(chapter, monkeypatch):
    # No citation means no distance to measure. ch5_order is what reports this case.
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 9)), encoding="utf-8")
    pages(monkeypatch, "", "", "")
    argv(monkeypatch)
    assert floats.main() == 0


def test_a_float_number_is_not_matched_by_a_longer_one_sharing_its_prefix(
        chapter, monkeypatch):
    """表 5.1 must not be found inside 表 5.10.

    This chapter carries 21 tables and 15 figures, so 表 5.1 is a prefix of 表 5.10 through
    表 5.19 and 表 5.2 of 表 5.20 / 表 5.21. A substring test dates 表 5.1's first citation
    to whichever of those eleven pages comes first, and `min()` can only move it earlier --
    inflating the distance into a false violation, or, when the caption precedes the true
    citation, hiding a real one.
    """
    # Laid out so that only the collision can fail it: 表 5.10 is cited on page 1 and
    # captioned on page 3 (distance 2, inside the criterion), 表 5.1 is cited on page 4
    # and captioned on page 5 (distance 1). A substring test pulls 表 5.1's first
    # citation back to page 1 and reports distance 4.
    (chapter / "main.aux").write_text(aux(("tab:a", "5.1", 5), ("tab:j", "5.10", 3)),
                                      encoding="utf-8")
    pages(monkeypatch, "先出现的是表 5.10。", "", "表 5.10 的标题页。", "见表 5.1。",
          "表 5.1 的标题页。")
    argv(monkeypatch)
    assert floats.main() == 0


def test_floats_without_a_build_exits_two_not_one(chapter, monkeypatch):
    argv(monkeypatch)
    assert floats.main() == 2


# =========================================================================== #
# ch5_check_all.compile_gate -- the log/blg half of the gate
# =========================================================================== #
CLEAN_LOG = "Output written on main.pdf (63 pages, 1234 bytes).\n"


def test_a_clean_log_passes_and_reports_the_page_count(chapter):
    (chapter / "main.log").write_text(CLEAN_LOG, encoding="utf-8")
    ok, notes = check_all.compile_gate()
    assert ok
    assert "页数: 63" in notes


def test_an_undefined_reference_fails(chapter):
    (chapter / "main.log").write_text(
        CLEAN_LOG + "LaTeX Warning: Reference `tab:x' on page 3 undefined.\n",
        encoding="utf-8")
    ok, _ = check_all.compile_gate()
    assert not ok


def test_a_multiply_defined_label_fails(chapter):
    (chapter / "main.log").write_text(
        CLEAN_LOG + "LaTeX Warning: Label `sec:x' multiply defined.\n", encoding="utf-8")
    ok, _ = check_all.compile_gate()
    assert not ok


def test_an_overfull_box_over_the_limit_fails(chapter):
    (chapter / "main.log").write_text(
        CLEAN_LOG + "Overfull \\hbox (12.50pt too wide) in paragraph at lines 10--12\n",
        encoding="utf-8")
    ok, notes = check_all.compile_gate()
    assert not ok
    assert "12.50pt" in " ".join(notes)


def test_an_overfull_box_under_the_limit_is_reported_but_passes(chapter):
    # The limit is 10pt, not zero: a hairline overfull is not worth buying with a rewrite.
    (chapter / "main.log").write_text(
        CLEAN_LOG + "Overfull \\hbox (3.20pt too wide) in paragraph at lines 10--12\n",
        encoding="utf-8")
    ok, notes = check_all.compile_gate()
    assert ok
    assert "3.20pt" in " ".join(notes)


def test_a_bibtex_warning_fails(chapter):
    (chapter / "main.log").write_text(CLEAN_LOG, encoding="utf-8")
    (chapter / "main.blg").write_text("Warning--empty journal in smith2020\n",
                                      encoding="utf-8")
    ok, _ = check_all.compile_gate()
    assert not ok


def test_the_bst_warning_counter_line_is_not_a_warning(chapter):
    # Every .blg ends with a `warning$` function-call tally. Reading it as a warning makes
    # the gate fail on every clean build, and a gate that always fails gets switched off.
    (chapter / "main.log").write_text(CLEAN_LOG, encoding="utf-8")
    (chapter / "main.blg").write_text(
        "(There was 1 error message)\n  warning$ -- 0\n", encoding="utf-8")
    ok, notes = check_all.compile_gate()
    assert ok
    assert "bibtex warning: 0" in notes


def test_compile_gate_without_a_build_is_a_failure_not_a_pass(chapter):
    ok, notes = check_all.compile_gate()
    assert not ok
    assert "main.log" in notes[0]
