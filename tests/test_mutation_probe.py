"""Tests for the negative-control engine and for the row tables it runs.

Two halves, and the second is the reason this file exists.

The engine half checks `apply_edit` on fixtures under `tmp_path`: it is the piece that has
been rewritten five times, and every rewrite re-learned the same two lessons -- an injection
point that occurs twice must refuse rather than pick one, and a CRLF working copy must come
back byte-identical or the restore silently rewrites the file it was protecting.

The table half checks the shipped rows *without running them*. A batch takes minutes and
edits the real tree, so nothing runs it casually and a row that has quietly stopped
addressing anything can sit there looking fine. These cases are data-free and take
milliseconds: they assert every row is well-formed, that every file a row names still
exists, that no row targets the evidence directories, and that every sweep row names a tool
the engine has registered -- an unregistered prefix makes self-check A fail for a reason
that has nothing to do with the mechanism under test.

What they deliberately do NOT check: that a row's `old` text still occurs in its target.
That is a real question, but answering it means reading the tool sources on every test run,
and the engine already reports it per row as "注入点出现 0 次". Keeping it there keeps the
signal attached to the row that owns it.
"""
from __future__ import annotations

import json
import sys

import pytest

from scripts import mutation_probe as mp

BATCHES = mp.available_batches()


# --------------------------------------------------------------------------- engine

def test_a_text_edit_that_matches_once_is_applied(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("WINDOW = 40\nother = 1\n", encoding="utf-8")
    mp.apply_edit(("text", "t.py", "WINDOW = 40", "WINDOW = 500"), tmp_path)
    assert p.read_text(encoding="utf-8") == "WINDOW = 500\nother = 1\n"


def test_an_injection_point_that_occurs_twice_is_refused(tmp_path):
    """Picking the first of two would mutate a line the row never named."""
    p = tmp_path / "t.py"
    p.write_text("x = 1\nx = 1\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="出现 2 次"):
        mp.apply_edit(("text", "t.py", "x = 1", "x = 2"), tmp_path)
    assert p.read_text(encoding="utf-8") == "x = 1\nx = 1\n"


def test_an_injection_point_that_is_gone_is_refused(tmp_path):
    """The signal that a mechanism moved and its negative control has to move with it."""
    p = tmp_path / "t.py"
    p.write_text("WINDOW = 40\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="出现 0 次"):
        mp.apply_edit(("text", "t.py", "WINDOW = 41", "WINDOW = 500"), tmp_path)


def test_a_crlf_file_stays_crlf(tmp_path):
    """The working copy is CRLF; an edit that normalises it corrupts the restore."""
    p = tmp_path / "t.py"
    p.write_bytes(b"WINDOW = 40\r\nother = 1\r\n")
    mp.apply_edit(("text", "t.py", "WINDOW = 40", "WINDOW = 500"), tmp_path)
    assert p.read_bytes() == b"WINDOW = 500\r\nother = 1\r\n"


def test_an_lf_file_stays_lf(tmp_path):
    p = tmp_path / "t.py"
    p.write_bytes(b"WINDOW = 40\nother = 1\n")
    mp.apply_edit(("text", "t.py", "WINDOW = 40", "WINDOW = 500"), tmp_path)
    assert p.read_bytes() == b"WINDOW = 500\nother = 1\n"


def test_a_multiline_needle_matches_across_line_endings(tmp_path):
    """A row quotes source with `\\n`; the file on disk has `\\r\\n`."""
    p = tmp_path / "t.py"
    p.write_bytes(b"if guard:\r\n    return None\r\nrest = 1\r\n")
    mp.apply_edit(("text", "t.py", "if guard:\n    return None\n", ""), tmp_path)
    assert p.read_bytes() == b"rest = 1\r\n"


def test_the_digest_ignores_line_endings(tmp_path):
    """Restore is verified by digest; a raw hash would call a rewritten file 'restored'."""
    (tmp_path / "crlf").write_bytes(b"a\r\nb\r\n")
    (tmp_path / "lf").write_bytes(b"a\nb\n")
    assert mp.digest(tmp_path / "crlf") == mp.digest(tmp_path / "lf")


def test_a_case_that_cannot_be_collected_is_reported_as_such(tmp_path):
    """Self-check C's own control. It runs a real pytest, hence the second or so.

    Both directions matter: C is only worth having if it says no to a missing case *and*
    yes to a present one -- a `pytest_collects` stuck at False would fail every row with a
    message about broken modules and nothing would look wrong about the engine.
    """
    here = "tests/test_mutation_probe.py"
    bt = tmp_path / "bt"
    assert not mp.pytest_collects(f"{here}::test_no_such_case_exists",
                                  mp.ROOT, sys.executable, bt)
    assert mp.pytest_collects(f"{here}::test_the_digest_ignores_line_endings",
                              mp.ROOT, sys.executable, bt)


def _spec(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"claims": [{"label": "A", "value": 1, "unit": "pp"},
                                        {"label": "B", "value": 2}]}, ensure_ascii=False),
                 encoding="utf-8")
    return p


def test_a_claim_edit_sets_one_key_of_one_claim(tmp_path):
    p = _spec(tmp_path)
    mp.apply_edit(("claim", "s.json", "A", "value", 99), tmp_path)
    claims = json.loads(p.read_text(encoding="utf-8"))["claims"]
    assert [c["value"] for c in claims] == [99, 2]
    assert claims[0]["unit"] == "pp"


def test_a_claim_edit_with_none_deletes_the_key(tmp_path):
    p = _spec(tmp_path)
    mp.apply_edit(("claim", "s.json", "A", "unit", None), tmp_path)
    claims = json.loads(p.read_text(encoding="utf-8"))["claims"]
    assert "unit" not in claims[0]


def test_a_claim_label_that_is_not_unique_is_refused(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"claims": [{"label": "A"}, {"label": "A"}]}), encoding="utf-8")
    with pytest.raises(AssertionError, match="命中 2 条"):
        mp.apply_edit(("claim", "s.json", "A", "value", 1), tmp_path)


def test_the_evidence_files_are_never_an_injection_target(tmp_path):
    """`benchmarks/` and friends are the records under audit, not code to be disabled."""
    for rel in ("benchmarks/x.json", "offline_data/y/metadata.json", "results/z.json"):
        with pytest.raises(AssertionError, match="证据文件"):
            mp.apply_edit(("text", rel, "a", "b"), tmp_path)


def test_markdown_under_an_evidence_tree_is_still_a_document(tmp_path):
    """The other half of the rule, asserted positively.

    `benchmarks/README.md` prints the counts and seed ranges the `cbm:doc-mismatch` rows are
    written against. Refusing it would make those rows unrunnable while the suite stayed
    green -- "报了该报的" is not "查了该查的".
    """
    assert not mp.is_evidence("benchmarks/README.md")
    assert mp.is_evidence("benchmarks/offline_rebrac_broad/test_100/x.json")
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "README.md").write_text("| 40 | 1250..1289 |\n",
                                                       encoding="utf-8")
    mp.apply_edit(("text", "benchmarks/README.md", "1250..1289", "1250..1290"), tmp_path)
    assert "1250..1290" in (tmp_path / "benchmarks" / "README.md").read_text(encoding="utf-8")


def test_an_unknown_edit_kind_is_refused(tmp_path):
    (tmp_path / "t.py").write_text("x\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="未知的 edit 类型"):
        mp.apply_edit(("patch", "t.py", "a", "b"), tmp_path)


# --------------------------------------------------------------------------- row tables

def test_there_is_at_least_one_batch():
    """A rename or a moved package would otherwise make every table test vacuous."""
    assert BATCHES, "scripts/mutations/ 下一个批次都没有"


@pytest.mark.parametrize("batch", BATCHES)
def test_every_row_is_well_formed(batch):
    rows, headline = mp.load_batch(batch)
    assert rows, f"{batch} 的 ROWS 是空的"
    assert headline, f"{batch} 缺 docstring 首行（--list 靠它）"
    for label, kind, check, edits in rows:
        assert kind in ("pytest", "sweep"), f"{label}: 未知的判据层 {kind!r}"
        assert edits, f"{label}: 没有任何 edit"
        for edit in edits:
            assert edit[0] in ("text", "claim"), f"{label}: 未知的 edit 类型 {edit[0]!r}"
            assert len(edit) == (4 if edit[0] == "text" else 5), f"{label}: edit 元数不对"


@pytest.mark.parametrize("batch", BATCHES)
def test_every_pytest_row_names_one_test_id(batch):
    """A suite selector lets an unrelated failure stand in for the graded mechanism."""
    rows, _ = mp.load_batch(batch)
    for label, kind, check, _ in rows:
        if kind == "pytest":
            assert check.count("::") == 1, f"{label}: {check!r} 不是单个用例 id"
            assert check.split("::")[0].endswith(".py"), f"{label}: {check!r} 缺测试文件"


@pytest.mark.parametrize("batch", BATCHES)
def test_every_sweep_row_names_a_registered_tool(batch):
    """An unregistered prefix fails self-check A for a reason unrelated to the mechanism."""
    rows, _ = mp.load_batch(batch)
    for label, kind, check, _ in rows:
        if kind == "sweep":
            tool = check.split(":")[0]
            assert ":" in check and tool in mp.SWEEPS, \
                f"{label}: 桶名 {check!r} 的工具前缀不在 SWEEPS（{', '.join(mp.SWEEPS)}）"


@pytest.mark.parametrize("batch", BATCHES)
def test_every_file_a_row_names_still_exists(batch):
    """The cheap half of rot detection: a renamed file is caught without running anything."""
    rows, _ = mp.load_batch(batch)
    for label, kind, check, edits in rows:
        for edit in edits:
            assert (mp.ROOT / edit[1]).is_file(), f"{label}: 注入目标 {edit[1]} 不存在"
        if kind == "pytest":
            assert (mp.ROOT / check.split("::")[0]).is_file(), \
                f"{label}: 测试文件 {check.split('::')[0]} 不存在"


@pytest.mark.parametrize("batch", BATCHES)
def test_no_row_targets_an_evidence_file(batch):
    """Stated as a property of the tables, not only as a guard inside `apply_edit`."""
    rows, _ = mp.load_batch(batch)
    for label, _, _, edits in rows:
        for edit in edits:
            assert not mp.is_evidence(edit[1]), f"{label}: {edit[1]} 是证据文件"


@pytest.mark.parametrize("batch", BATCHES)
def test_row_labels_start_with_a_unique_number(batch):
    """`--only` selects on that first token, so a duplicate silently runs two rows."""
    rows, _ = mp.load_batch(batch)
    heads = [label.split()[0] for label, *_ in rows]
    dupes = {h for h in heads if heads.count(h) > 1}
    assert not dupes, f"{batch} 的行号重复：{sorted(dupes)}"
