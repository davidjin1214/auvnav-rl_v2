"""Tests for the generated document index.

The index exists because a hand-written map goes stale silently, and `--check` is what
makes the staleness visible. That makes `--check` itself the thing worth protecting: if
it stops being able to fail, the index goes back to being a hand-written map that merely
looks generated.

`status_of` gets the rest of the attention. It reads a doc's own banner, and it has
misfired in both directions already -- tagging CLAUDE.md PAUSED because it *describes*
the paused line in running prose, and tagging a live dataset card ARCHIVE because a path
in its banner contained the word. Under-reporting is the safe direction: a doc with no
banner reads `—`, which means "did not label itself", not "still current".

Every test runs against a fixture tree, never this repo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_doc_index as bdi

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relpath: str, text: str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(bdi, "REPO", tmp_path)
    monkeypatch.setattr(bdi, "OUT", tmp_path / "docs/DOC_INDEX.md")
    return tmp_path


def indexed(root: Path) -> set[str]:
    return {rel for rel, _title, _status in bdi.collect()}


def status(head: str) -> str:
    return bdi.status_of(head.split("\n")[: bdi.HEAD_LINES])


# --------------------------------------------------------------------------- corpus


def test_agent_worktrees_are_not_indexed(repo):
    """Indexing one lists every doc twice, and `--check` then fails purely because a
    worktree happens to exist right now (2d4ba97)."""
    _write(repo, "docs/a.md", "# A\n")
    _write(repo, ".claude/worktrees/w/docs/a.md", "# A\n")

    assert indexed(repo) == {"docs/a.md"}


def test_tool_caches_are_not_indexed(repo):
    """A cache ships its own README, which would enter the index the first time anyone
    runs pytest here -- making the index depend on what has been run on this machine."""
    _write(repo, "docs/a.md", "# A\n")
    _write(repo, ".pytest_cache/README.md", "# cache\n")

    assert indexed(repo) == {"docs/a.md"}


def test_gitignored_data_directories_are_skipped_at_the_top_level(repo):
    """They may be junctions to another volume; walking one means ~40 GB of run output,
    which carries markdown of its own. Whether it shows up at all depends on how a given
    machine mounts its data, which would make the index machine-dependent."""
    _write(repo, "results/archived/report.md", "# 旧报告\n")
    _write(repo, "docs/a.md", "# A\n")

    assert indexed(repo) == {"docs/a.md"}


def test_a_data_directory_name_deeper_in_the_tree_is_not_skipped(repo):
    """Top level only -- `docs/results_note.md` and `paper/.../figures/` are content."""
    _write(repo, "docs/results/note.md", "# 说明\n")

    assert indexed(repo) == {"docs/results/note.md"}


def test_the_index_does_not_index_itself(repo):
    _write(repo, "docs/DOC_INDEX.md", "# 全仓文档索引\n")
    _write(repo, "docs/a.md", "# A\n")

    assert indexed(repo) == {"docs/a.md"}


# --------------------------------------------------------------------------- banners


def test_a_banner_in_the_blockquote_after_the_h1_is_read():
    assert status("# 标题\n\n> **DEPRECATED** 2026-05-06 起停用\n") == "DEPRECATED 2026-05-06"


def test_a_banner_without_a_date_keeps_just_the_label():
    assert status("# 标题\n\n> **CLOSED** —— 本线收口\n") == "CLOSED"


def test_prose_that_merely_mentions_another_lines_state_is_not_a_banner():
    """CLAUDE.md and README.md both got tagged PAUSED this way.

    They describe the paused AUVHamNODE line in running prose; neither is paused.
    """
    assert status("# CLAUDE.md\n\n本仓有一条 AUVHamNODE 线，已 PAUSED 2026-05-13。\n") == "—"


def test_a_keyword_inside_a_path_is_not_a_banner():
    """Once the pre-P2 FQL records moved under `docs/archive/`, every banner citing one
    carried a literal "archive" whose neighbouring `**Spec**:` label satisfied the
    emphasis test -- tagging a live dataset card ARCHIVE, the opposite of its state."""
    head = "# 数据集卡\n\n> **Spec**: `docs/archive/fql_succession/spec.md`\n"

    assert status(head) == "—"


def test_a_plain_mention_inside_the_banner_without_emphasis_does_not_count():
    """The blockquote scoping above and this emphasis rule are different mechanisms.

    A test whose keyword sits outside any blockquote passes with the emphasis rule
    removed, so it pins the scoping and nothing else. This one keeps the keyword inside
    the banner, where only the missing `**`/`⚠` can decide it.
    """
    assert status("# 标题\n\n> 本文 deprecated 与否见下文讨论\n") == "—"


def test_the_first_matching_pattern_wins():
    """Ordered so SUPERSEDED beats a passing mention of "archive" in the same banner."""
    head = "# 标题\n\n> **SUPERSEDED** 2026-05-18，原件见 ⚠ 归档\n"

    assert status(head) == "SUPERSEDED 2026-05-18"


def test_a_doc_with_no_banner_reads_as_unlabelled_not_as_active(repo):
    """`—` means "did not label itself". Deciding a doc is still current is human work."""
    _write(repo, "docs/a.md", "# 标题\n\n正文。\n")

    assert bdi.collect() == [("docs/a.md", "标题", "—")]


def test_a_long_title_is_truncated(repo):
    _write(repo, "docs/a.md", "# " + "长" * 100 + "\n")

    _rel, title, _status = bdi.collect()[0]

    assert len(title) == 88 and title.endswith("…")


def test_a_doc_without_an_h1_falls_back_to_its_filename(repo):
    _write(repo, "docs/notes.md", "正文，没有标题。\n")

    assert bdi.collect() == [("docs/notes.md", "notes", "—")]


# ----------------------------------------------------------------------- the --check


def test_check_passes_on_a_current_index(repo, monkeypatch, capsys):
    _write(repo, "docs/a.md", "# A\n")
    monkeypatch.setattr(sys, "argv", ["build_doc_index"])
    assert bdi.main() == 0

    monkeypatch.setattr(sys, "argv", ["build_doc_index", "--check"])

    assert bdi.main() == 0
    assert "is current" in capsys.readouterr().out


def test_check_fails_once_a_doc_is_added(repo, monkeypatch, capsys):
    """The whole point: drift becomes visible instead of being quoted as fact."""
    _write(repo, "docs/a.md", "# A\n")
    monkeypatch.setattr(sys, "argv", ["build_doc_index"])
    bdi.main()
    _write(repo, "docs/b.md", "# B\n")

    monkeypatch.setattr(sys, "argv", ["build_doc_index", "--check"])

    assert bdi.main() == 1
    assert "out of date" in capsys.readouterr().out


def test_check_fails_when_a_banner_changes(repo, monkeypatch, capsys):
    """A doc that flips from live to DEPRECATED must move the index, not just the doc."""
    _write(repo, "docs/a.md", "# A\n")
    monkeypatch.setattr(sys, "argv", ["build_doc_index"])
    bdi.main()
    _write(repo, "docs/a.md", "# A\n\n> **DEPRECATED** 2026-08-23\n")

    monkeypatch.setattr(sys, "argv", ["build_doc_index", "--check"])

    assert bdi.main() == 1


def test_check_fails_when_the_index_has_never_been_written(repo, monkeypatch, capsys):
    _write(repo, "docs/a.md", "# A\n")
    monkeypatch.setattr(sys, "argv", ["build_doc_index", "--check"])

    assert bdi.main() == 1
    assert "does not exist" in capsys.readouterr().out


def test_line_endings_do_not_make_the_index_look_stale(repo, monkeypatch):
    """This repo is edited on Windows and on macOS; a CRLF round-trip is not drift.

    Two mechanisms deliver this -- the explicit normalisation in `main`, and Python's
    universal-newline translation on `read_text` -- so removing either one leaves the
    contract intact. Pinned as the observable behaviour rather than as one of the two.
    """
    _write(repo, "docs/a.md", "# A\n")
    monkeypatch.setattr(sys, "argv", ["build_doc_index"])
    bdi.main()
    text = bdi.OUT.read_text(encoding="utf-8")
    bdi.OUT.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))

    monkeypatch.setattr(sys, "argv", ["build_doc_index", "--check"])

    assert bdi.main() == 0


def test_the_shipped_index_is_current():
    """The one test that does look at this repo, because that is the claim being made.

    Unlike the pointer sweeps, a stale index here is a defect and not a finding: the
    fix is to rerun the generator, which is what the failure message says.
    """
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "scripts.build_doc_index", "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8")

    assert proc.returncode == 0, proc.stdout + proc.stderr
