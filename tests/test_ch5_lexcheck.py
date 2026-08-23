"""Tests for the Chapter 5 collation search (spec §0.5.11 layers 2 and 4).

Two design decisions in this tool came out of failures it now exists to prevent, and both
are pinned here.

Counting is by **occurrence**, not by matching line: reporting per line under-counted
「无差异」as 6 when it is 8 (定点复核 L-2, 2026-07-28). A per-line count passes every test
built from one-hit-per-line fixtures, so the fixture below puts two hits on one line.

The **frozen baseline** separates "already adjudicated" from "new". Most hits in this
chapter are legitimately compliant, and re-arguing all of them each round is what let a
real one hide: the claim「§5.7.1 是全章唯一的规则②漏网」was written into four documents on
the strength of a search that had enumerated one of the five rule-② words (定点复核 H-1).
So the gate reports deltas, and `--freeze` is an assertion that every current hit has been
adjudicated -- never a way to silence a warning.

The baseline also carries a provenance stamp, because it once sat unrefreshed across five
整改 batches with nothing in the file recording when it was frozen: it could not go stale
visibly. That stamp is data about the baseline, not a watched term, and must stay out of
the diff.
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

lexcheck = importlib.import_module("ch5_lexcheck")

SECTIONS = ["alpha", "beta"]
SECTION_NO = {"alpha": "5.1", "beta": "5.2"}
WATCHED = [("测试组", ["无差异", "追平"])]


@pytest.fixture
def chapter(tmp_path, monkeypatch):
    root = tmp_path / "thesis_ch5"
    (root / "sections").mkdir(parents=True)
    monkeypatch.setattr(lexcheck, "CHAPTER_DIR", str(root))
    monkeypatch.setattr(lexcheck, "SECTION_ORDER", SECTIONS)
    monkeypatch.setattr(lexcheck, "SECTION_NO", SECTION_NO)
    monkeypatch.setattr(lexcheck, "GROUPS", WATCHED)
    monkeypatch.setattr(lexcheck, "BASELINE", str(tmp_path / "baseline.json"))
    for stem in SECTIONS:
        (root / "sections" / f"{stem}.tex").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["tool"])
    return root


def write(root: Path, stem: str, text: str) -> None:
    (root / "sections" / f"{stem}.tex").write_text(text, encoding="utf-8")


def baseline(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def run(monkeypatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["tool", *args])
    return lexcheck.main()


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #
def test_hits_are_counted_by_occurrence_not_by_line(chapter):
    # Both hits are on one line. A per-line count reports 1 and reads as a smaller
    # problem than it is -- the miscount that produced 6-instead-of-8 for 「无差异」.
    write(chapter, "alpha", "此处无差异，彼处也无差异。\n")
    total, per_file, where = lexcheck.count_hits(lexcheck.body_lines(), "无差异")
    assert total == 2
    assert per_file == {"alpha": 2}
    assert len(where) == 1  # one context line, two occurrences


def test_a_hit_inside_a_comment_is_not_a_hit(chapter):
    # The rev blocks quote superseded wording freely; counting it would make the gate
    # fail on text no reader sees, and a gate that always fails gets switched off.
    write(chapter, "alpha", "% 旧稿写过「无差异」，已改\n正文没有该词。\n")
    total, _, _ = lexcheck.count_hits(lexcheck.body_lines(), "无差异")
    assert total == 0


def test_hits_are_attributed_to_the_section_they_are_in(chapter):
    write(chapter, "alpha", "此处无差异。\n")
    write(chapter, "beta", "彼处追平。\n")
    doc = lexcheck.body_lines()
    assert lexcheck.count_hits(doc, "无差异")[1] == {"alpha": 1}
    assert lexcheck.count_hits(doc, "追平")[1] == {"beta": 1}


# --------------------------------------------------------------------------- #
# the frozen baseline
# --------------------------------------------------------------------------- #
def test_a_run_matching_the_baseline_passes(chapter, monkeypatch, capsys):
    write(chapter, "alpha", "此处无差异。\n")
    assert run(monkeypatch, "--freeze") == 0
    assert run(monkeypatch) == 0
    assert "PASS" in capsys.readouterr().out


def test_a_new_hit_is_a_delta_and_fails(chapter, monkeypatch, capsys):
    write(chapter, "alpha", "此处无差异。\n")
    run(monkeypatch, "--freeze")
    write(chapter, "alpha", "此处无差异，那处也无差异。\n")
    assert run(monkeypatch) == 1
    out = capsys.readouterr().out
    assert "无差异" in out and "1 -> 2" in out


def test_a_hit_that_has_gone_away_is_also_a_delta(chapter, monkeypatch, capsys):
    # Deleting an adjudicated hit is a change to the adjudicated state too: the findings
    # entry that exempted it now describes text that is no longer there.
    write(chapter, "alpha", "此处无差异。\n")
    run(monkeypatch, "--freeze")
    write(chapter, "alpha", "此处已改写。\n")
    assert run(monkeypatch) == 1
    assert "1 -> 0" in capsys.readouterr().out


def test_a_hit_moving_between_sections_is_a_delta_even_at_the_same_total(chapter,
                                                                        monkeypatch,
                                                                        capsys):
    # The baseline is per-file, so a chapter-wide total cannot hide a relocation. Which
    # section a locked phrase sits in is the whole point of the per-section adjudication.
    write(chapter, "alpha", "此处无差异。\n")
    run(monkeypatch, "--freeze")
    write(chapter, "alpha", "此处已改写。\n")
    write(chapter, "beta", "此处无差异。\n")
    assert run(monkeypatch) == 1
    out = capsys.readouterr().out
    assert "§5.1: 1 -> 0" in out and "§5.2: 0 -> 1" in out


def test_a_word_dropped_from_the_watch_list_is_still_reported(chapter, monkeypatch,
                                                              capsys):
    # Otherwise removing a word from GROUPS silently retires whatever it was catching.
    write(chapter, "alpha", "此处无差异。\n")
    baseline(lexcheck.BASELINE, {"已退役的词": {"alpha": 3}})
    assert run(monkeypatch) == 1
    assert "已退役的词" in capsys.readouterr().out


def test_with_no_baseline_the_tool_asks_to_freeze_rather_than_failing(chapter,
                                                                      monkeypatch,
                                                                      capsys):
    write(chapter, "alpha", "此处无差异。\n")
    assert run(monkeypatch) == 0
    assert "先跑 --freeze" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the provenance stamp
# --------------------------------------------------------------------------- #
def test_freezing_stamps_when_and_against_which_tree(chapter, monkeypatch):
    write(chapter, "alpha", "此处无差异。\n")
    run(monkeypatch, "--freeze")
    payload = json.loads(Path(lexcheck.BASELINE).read_text(encoding="utf-8"))
    assert set(payload[lexcheck.META_KEY]) == {"frozen", "commit", "entries"}


def test_the_stamp_is_not_diffed_as_if_it_were_a_watched_term(chapter, monkeypatch):
    write(chapter, "alpha", "此处无差异。\n")
    run(monkeypatch, "--freeze")
    # It is a dict of provenance, not a per-section count. Left in the diff it would
    # report a delta on every run.
    assert run(monkeypatch) == 0


def test_a_baseline_older_than_the_sections_it_covers_is_flagged_even_under_quiet(
        chapter, monkeypatch, capsys):
    monkeypatch.setattr(lexcheck, "_git", lambda *a: "7" if a[0] == "rev-list" else "abc1234")
    lexcheck._report_baseline_age({"frozen": "2026-07-28", "commit": "abc1234",
                                   "entries": 88}, quiet=True)
    out = capsys.readouterr().out
    assert "已前进 7 个提交" in out


def test_a_current_baseline_stays_silent_under_quiet(chapter, monkeypatch, capsys):
    monkeypatch.setattr(lexcheck, "_git", lambda *a: "0" if a[0] == "rev-list" else "abc1234")
    lexcheck._report_baseline_age({"frozen": "2026-08-23", "commit": "abc1234",
                                   "entries": 88}, quiet=True)
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# the naming lock and the parenthetical glosses
# --------------------------------------------------------------------------- #
def test_bare_rebrac_is_counted_but_the_chapter_s_own_name_is_not(chapter, monkeypatch,
                                                                  capsys):
    # §0.5.10 (A) keeps the original method name as-is; the lock is only that THIS
    # chapter's method is written ReBRAC-Q. So a bare hit is a prompt to adjudicate,
    # which is why it goes into the baseline rather than straight to FAIL.
    write(chapter, "alpha", "原始 ReBRAC 与本章的 ReBRAC-Q 不同。\n")
    run(monkeypatch, "--freeze")
    payload = json.loads(Path(lexcheck.BASELINE).read_text(encoding="utf-8"))
    assert payload["<裸写 ReBRAC>"] == {"alpha": 1}


def test_the_typeset_hyphen_form_also_counts_as_the_locked_name(chapter, monkeypatch):
    write(chapter, "alpha", r"本章的 ReBRAC\text{-}Q 写法。" + "\n")
    run(monkeypatch, "--freeze")
    payload = json.loads(Path(lexcheck.BASELINE).read_text(encoding="utf-8"))
    assert payload["<裸写 ReBRAC>"] == {}


def test_an_english_gloss_repeated_in_two_sections_fails(chapter, monkeypatch, capsys):
    write(chapter, "alpha", "行为克隆（behaviour cloning）首现。\n")
    run(monkeypatch, "--freeze")
    write(chapter, "beta", "又一次行为克隆（behaviour cloning）。\n")
    assert run(monkeypatch) == 1
    out = capsys.readouterr().out
    assert "重复者 1 个" in out
    assert "英文括注重复出现" in out


def test_a_gloss_appearing_once_passes(chapter, monkeypatch, capsys):
    write(chapter, "alpha", "行为克隆（behaviour cloning）首现。\n")
    write(chapter, "beta", "此后只写中文。\n")
    run(monkeypatch, "--freeze")
    assert run(monkeypatch) == 0
    assert "重复者 0 个" in capsys.readouterr().out
