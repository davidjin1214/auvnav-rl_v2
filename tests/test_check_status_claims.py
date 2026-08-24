"""Tests for the table status-cell checker.

The checker replaces a hand pass (`cce2b2d`) that walked one 13-row routing table and
found 3 rows disagreeing with the banner of the document they cite. Two things about it
are worth protecting, and they pull in opposite directions:

  it must fire      -- a status cell that contradicts the doc it is about, and a cell
                       holding a forecast where a state belongs, are the two failures
                       the audit actually found;
  it must stay quiet -- the repo holds 701 status cells and the overwhelming majority
                       are honest dated snapshots. A checker that reports those buries
                       the handful that are wrong, which is how the hand pass was
                       justified in the first place.

So every "does not fire" test below is a real shape measured in this repo on 2026-08-24,
not a hypothetical: a ❌ column meaning "holds no numbers", a `TBD` meaning "the
literature has no figure", a `TBD` meaning "the notebook is unnamed", and a table a doc
freezes on purpose and says so.

Every test runs against a fixture tree, never this repo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_status_claims as csc

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relpath: str, text: str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(csc, "ROOT", tmp_path)
    return tmp_path


def dead_doc(root: Path, name: str = "docs/dead.md", banner: str = "SUPERSEDED 2026-05-18") -> None:
    _write(root, name, f"# Dead\n\n> **⚠ {banner}** — 已被后继文档取代。\n")


def routing_table(rows: str, header: str = "| 文档 | 状态 | 角色 |") -> str:
    return f"# Index\n\n{header}\n|---|---|---|\n{rows}\n"


def buckets(root: Path) -> dict[str, list[str]]:
    return csc.scan()


# ----------------------------------------------------------------- R1: it must fire


def test_a_status_cell_contradicting_the_cited_banner_is_a_defect(repo):
    """`rlpd_design.md` was carried as "active reference" in one table for months while
    its own banner said otherwise -- the shape `cce2b2d` found three of."""
    dead_doc(repo)
    _write(repo, "docs/index.md", routing_table(
        "| [`dead.md`](dead.md) | active reference | 设计档 |"))

    found = buckets(repo)
    assert len(found["contradict"]) == 1
    assert "docs/index.md:5" in found["contradict"][0]


def test_a_cell_naming_a_different_state_than_the_banner_is_a_defect(repo):
    dead_doc(repo, banner="PAUSED 2026-05-13")
    _write(repo, "docs/index.md", routing_table(
        "| [`dead.md`](dead.md) | **CLOSED** | 计划 |"))

    assert len(buckets(repo)["contradict"]) == 1


def test_a_matching_keyword_with_a_different_date_is_a_defect(repo):
    """The version and the date both went stale on `online_sac_reward_redesign.md`; the
    keyword agreeing is exactly what makes a wrong date easy to read past."""
    dead_doc(repo, banner="SUPERSEDED 2026-05-18")
    _write(repo, "docs/index.md", routing_table(
        "| [`dead.md`](dead.md) | **SUPERSEDED 2026-04-27** | 旧版 |"))

    found = buckets(repo)
    assert not found["contradict"]
    assert len(found["date"]) == 1


def test_the_same_keyword_and_date_agree(repo):
    dead_doc(repo, banner="SUPERSEDED 2026-05-18")
    _write(repo, "docs/index.md", routing_table(
        "| [`dead.md`](dead.md) | **SUPERSEDED 2026-05-18 (archive)** | 旧版 |"))

    found = buckets(repo)
    assert len(found["agree"]) == 1
    assert not found["contradict"] and not found["date"]


# ------------------------------------------------------------ R1: it must stay quiet


def test_a_cell_that_omits_the_banner_keyword_is_advisory_not_a_defect(repo):
    """"已被 ReBRAC 主线取代；保留作历史" says what DEPRECATED says without the word.
    Silence is under-reporting, which is the safe direction -- it must not fail --strict."""
    dead_doc(repo, banner="DEPRECATED 2026-05-08")
    _write(repo, "docs/index.md", routing_table(
        "| [`dead.md`](dead.md) | 已被主线取代；保留作历史 | 旧计划 |"))

    found = buckets(repo)
    assert len(found["silent"]) == 1
    assert not found["contradict"]
    assert csc.main([]) == 0 and csc.main(["--strict"]) == 0


def test_a_column_that_is_not_a_status_column_is_not_graded(repo):
    """`持数字 ground truth？` is a column of ✅/❌ whose ❌ means "holds no numbers".
    Detecting the status column from its cells instead of its header grades it, and then
    every plan document in the repo reads as contradicting itself."""
    dead_doc(repo, banner="DEPRECATED 2026-05-08")
    _write(repo, "docs/index.md", routing_table(
        "| [`dead.md`](dead.md) | ❌ | 计划 |", header="| 文档 | 持数字 ground truth？ | 角色 |"))

    found = buckets(repo)
    assert not found["contradict"] and not found["silent"] and not found["agree"]


def test_a_row_whose_subject_merely_mentions_a_doc_is_not_graded(repo):
    """A ✅ against "六份文档头注仍写 spec rev.4（…[`outline.md`](…)…）" grades the *fix*,
    not the document. Those rows are how a naive scan floods."""
    dead_doc(repo)
    _write(repo, "docs/index.md", routing_table(
        "| 六份文档头注仍写旧节号，含 [`dead.md`](dead.md) 与另外五份 | ✅ 每份加一行对照 | 整改 |"))

    assert not any(buckets(repo)[b] for b in ("contradict", "date", "silent", "agree"))


def test_a_row_citing_two_docs_is_not_graded(repo):
    """Which of the two the status cell is about is not decidable from the row."""
    dead_doc(repo, "docs/dead.md")
    dead_doc(repo, "docs/dead2.md", banner="PAUSED 2026-05-13")
    _write(repo, "docs/index.md", routing_table(
        "| [`dead.md`](dead.md) + [`dead2.md`](dead2.md) | active | 两份 |"))

    assert not buckets(repo)["contradict"]


def test_a_doc_without_a_banner_is_not_graded(repo):
    """The absence of a banner is absence of evidence, exactly as the index says."""
    _write(repo, "docs/plain.md", "# Plain\n\n> 文档版本：rev.8\n")
    _write(repo, "docs/index.md", routing_table(
        "| [`plain.md`](plain.md) | active | 计划 |"))

    assert not any(buckets(repo)[b] for b in ("contradict", "date", "silent", "agree"))


def test_the_generated_index_is_not_graded(repo):
    """DOC_INDEX.md is generated FROM these banners; grading it restates --check."""
    dead_doc(repo)
    _write(repo, "docs/DOC_INDEX.md", routing_table(
        "| [`dead.md`](dead.md) | active | 自动生成 |", header="| 文档 | 自我标注状态 | 角色 |"))

    assert not buckets(repo)["contradict"]


# ----------------------------------------------------------------- R2: the forecast


def test_a_bare_forecast_in_a_progress_column_is_a_defect(repo):
    """`⬜ 第三批` stood in this repo's own audit ledger after the third batch closed
    without it. A forecast never becomes false on its own, so nobody revisits it."""
    _write(repo, "docs/ledger.md",
           "# Ledger\n\n| # | 审计 | 本轮 |\n|---|---|---|\n| 12 | benchmarks 落库核 | ⬜ 第三批 |\n")

    found = buckets(repo)
    assert len(found["forecast"]) == 1
    assert csc.main(["--strict"]) == 1


def test_tbd_in_a_status_column_is_a_defect(repo):
    _write(repo, "docs/report.md",
           "# R\n\n| 文档 | 修改 | 状态 |\n|---|---|---|\n| `a.md` | 加入口 | **TBD** |\n")

    assert len(buckets(repo)["forecast"]) == 1


def test_a_cell_ending_in_a_batch_name_is_not_a_defect(repo):
    """`✅ 已并入第三批` states a fact and happens to end on a batch name. This is the
    end the start anchor holds: searching instead of matching flags it."""
    _write(repo, "docs/ledger.md",
           "# Ledger\n\n| # | 审计 | 本轮 |\n|---|---|---|\n"
           "| 11 | manifest 归属 | ✅ 已并入第三批 |\n")

    assert not buckets(repo)["forecast"]


def test_a_cell_beginning_with_a_batch_name_is_not_a_defect(repo):
    """`第三批已收尾（925d9c4）` opens on a batch name and then says what happened. This
    is the end the trailing `$` holds."""
    _write(repo, "docs/ledger.md",
           "# Ledger\n\n| # | 审计 | 本轮 |\n|---|---|---|\n"
           "| 11 | manifest 归属 | 第三批已收尾（`925d9c4`）|\n")

    assert not buckets(repo)["forecast"]


def test_a_settled_cell_that_merely_mentions_a_batch_is_not_a_defect(repo):
    """The shape both anchors together must survive, from the ledger itself."""
    _write(repo, "docs/ledger.md",
           "# Ledger\n\n| # | 审计 | 本轮 |\n|---|---|---|\n"
           "| 11 | manifest 归属 | ✅ 已并入第三批（`db0c235`，19 项用例）|\n"
           "| 12 | benchmarks 落库核 | ⬜ 未做，**未分配批次**（第三批收尾时未含它）|\n")

    assert not buckets(repo)["forecast"]


def test_a_forecast_outside_a_progress_column_is_not_graded(repo):
    """A bare `TBD` under `Expected σ_final ↓` means the literature has no number for
    it, and under `Notebook` means the file is unnamed. Both are in this repo today and
    neither is a stale status."""
    _write(repo, "docs/design.md",
           "# D\n\n| Technique | Compute cost | Expected σ_final ↓ |\n|---|---|---|\n"
           "| CrossQ BatchNorm | 1.0x | TBD |\n\n"
           "| Step | Notebook | Duration |\n|---|---|---|\n"
           "| P0b/c (conditional) | TBD | 2.5-5h |\n")

    assert not buckets(repo)["forecast"]


def test_a_row_about_a_to_be_named_file_is_not_graded(repo):
    """`| TBD | 待建 |` names what the row is about, not where it stands. It is the
    column scoping that decides this, which is why the row survives with the subject
    column no longer skipped explicitly -- a skip that never fired, because a subject
    column named 文件 / Path / 审计 is never also a progress column."""
    _write(repo, "docs/plan.md",
           "# P\n\n| 文件 | 状态 |\n|---|---|\n| TBD | 待建 |\n")

    assert not buckets(repo)["forecast"]


# ------------------------------------------------------------ R2: the freeze escape


def test_a_table_declared_frozen_is_excused_but_still_printed(repo, capsys):
    """A plan may state that its deliverables table stays at planning-time values on
    purpose; `rebrac_broad_validation_v2_plan.md` does, in eleven rows."""
    _write(repo, "docs/plan.md",
           "# P\n\n## 11. 产物\n\n> 本注只记录落地实况，不改本表任何一行。\n\n"
           "| Path | Role | Status |\n|---|---|---|\n| `scripts/x.py` | driver | TBD |\n")

    found = buckets(repo)
    assert not found["forecast"] and len(found["declared"]) == 1
    assert csc.main([]) == 0
    out = capsys.readouterr().out
    assert "本注只记录落地实况" in out, "a declaration nobody can read cannot be checked"
    assert "声明，不是事实" in out


def test_the_freeze_declaration_must_sit_under_the_same_heading(repo):
    """Scoped like `check_doc_pointers`'s self-declared absences: the declaration is the
    prose the reader meets before the table. One left behind under an earlier heading
    would go on excusing a table it no longer describes."""
    _write(repo, "docs/plan.md",
           "# P\n\n## 11. 产物\n\n> 本注只记录落地实况，不改本表任何一行。\n\n"
           "## 12. 另一节\n\n| Path | Role | Status |\n|---|---|---|\n| `scripts/x.py` | driver | TBD |\n")

    found = buckets(repo)
    assert len(found["forecast"]) == 1 and not found["declared"]


# ----------------------------------------------------------------------- mechanics


def test_pipes_inside_inline_code_do_not_split_cells(repo):
    """A cell may legitimately hold `a|b`; splitting on it shifts every later column and
    silently grades the wrong one."""
    assert csc.split_row("| `a|b` | active | x |") == ["`a|b`", "active", "x"]


def test_strict_exits_1_only_on_defect_buckets(repo, capsys):
    """The grading contract: advisory buckets print and pass, defect buckets fail."""
    dead_doc(repo, banner="DEPRECATED 2026-05-08")
    _write(repo, "docs/index.md", routing_table(
        "| [`dead.md`](dead.md) | 已被主线取代 | 旧计划 |"))
    assert csc.main(["--strict"]) == 0
    assert "未提 1" in capsys.readouterr().out

    _write(repo, "docs/index2.md", routing_table(
        "| [`dead.md`](../docs/dead.md) | active | 旧计划 |"))
    assert csc.main(["--strict"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_the_report_survives_a_non_utf8_stdout_pipe():
    """The one case here that runs against the live tree, and the reason to.

    The PostToolUse hook captures this sweep through a pipe, and on Windows a pipe
    defaults to cp936, where `⚠` is not encodable. Running the script by hand never
    shows it -- a terminal here is UTF-8. Left unfixed the sweep dies mid-report, the
    hook reads the crash as a finding, and every markdown edit in the repo is blocked.
    """
    proc = subprocess.run([sys.executable, "-m", "scripts.check_status_claims", "--strict"],
                          cwd=REPO_ROOT, capture_output=True)
    stderr = proc.stderr.decode("utf-8", "replace")

    assert "UnicodeEncodeError" not in stderr, stderr[-600:]
    assert proc.returncode == 0, proc.stdout.decode("utf-8", "replace")[-800:]


def test_gitignored_data_trees_are_not_walked(repo):
    """`results/` may hold 40 GB and its own markdown; it is not part of the corpus."""
    dead_doc(repo)
    _write(repo, "results/archived/index.md", routing_table(
        "| [`../../docs/dead.md`](../../docs/dead.md) | active | x |"))

    assert not buckets(repo)["contradict"]
