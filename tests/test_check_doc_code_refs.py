"""Tests for the doc-to-source reference sweep.

Every check here is exercised against a tree that has the defect, not only against a
clean one: a sweep that passes on healthy input proves nothing about the case it exists
for. Both defect classes below are reconstructions of rot this repo actually shipped --
`collect_offline_data.py:315` after 5f8228c pushed the statement to 316, and CLAUDE.md's
`get_probe_positions()`, a function name that has never existed anywhere but in
CLAUDE.md.

The tree under test is always a fixture, never this repo. Pointed at the real one, a
true finding would show up as a failing test, which inverts what the assertion means.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_doc_code_refs as refs

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relpath: str, text: str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def tree(tmp_path):
    """A miniature repo: one module, one doc citing into it."""
    _write(tmp_path, "auv_nav/flow.py", "\n".join([
        "import numpy as np",
        "",
        "",
        "def make_probe_offsets(layout):",
        "    return np.zeros(3)",
        "",
        "",
        "def unrelated():",
        "    return None",
    ]))
    return tmp_path


def bucket(root: Path, name: str, **kw) -> list[tuple]:
    return refs.scan(str(root), **kw)[name]


def test_a_reference_whose_fragment_is_on_the_cited_line_is_clean(tree):
    _write(tree, "docs/a.md",
           "采集：`auv_nav/flow.py:4` → `def make_probe_offsets(layout)`")

    buckets = refs.scan(str(tree))

    assert buckets["drift"] == []
    assert buckets["drift-near"] == []
    assert any(row[1] == "auv_nav/flow.py" for row in buckets["clean"])


def test_line_number_past_the_end_of_the_file(tree):
    _write(tree, "docs/a.md", "见 `auv_nav/flow.py:400`")

    rows = bucket(tree, "out-of-range")

    assert len(rows) == 1
    assert rows[0][1:] == ("auv_nav/flow.py", 400, 9)


def test_a_citation_landing_on_a_blank_line(tree):
    """`online_rl_thesis_plan.md` cites `env.py:895` for a definition and gets nothing.

    A blank line is the visible half of an insertion above it: the path still resolves,
    so the pointer sweep stays quiet, and the reader is sent to whitespace.
    """
    _write(tree, "docs/a.md", "`privileged_obs` 的定义在 `auv_nav/flow.py:3`")

    rows = bucket(tree, "blank-line")

    assert len(rows) == 1
    assert rows[0][2] == 3


def test_a_fragment_that_moved_far_away_is_rot(tree):
    _write(tree, "docs/a.md", "采集：`auv_nav/flow.py:1` → `make_probe_offsets`")

    rows = bucket(tree, "drift")

    assert len(rows) == 1
    where, raw, start, target, tokens, near = rows[0]
    assert (start, near) == (1, 4), "the report must name the line that does carry it"
    assert tokens == ["make_probe_offsets"]


def test_a_fragment_one_line_off_is_an_offset_not_rot(tree):
    """The 5f8228c shape, and the reason it is not a build-breaking defect.

    A doc may cite the line an effect lands on while quoting the condition just above it.
    Both readings are honest, so the offset is printed and `--strict` ignores it; what
    fails is a fragment that is far away or gone.
    """
    _write(tree, "docs/a.md", "采集：`auv_nav/flow.py:5` → `def make_probe_offsets`")

    buckets = refs.scan(str(tree))

    assert [r[2] for r in buckets["drift-near"]] == [5]
    assert buckets["drift"] == []
    assert "drift-near" not in refs.DEFECT_BUCKETS


def test_the_offset_window_is_adjustable(tree):
    _write(tree, "docs/a.md", "采集：`auv_nav/flow.py:5` → `def make_probe_offsets`")

    assert bucket(tree, "drift", near_window=0) != []
    assert bucket(tree, "drift-near", near_window=0) == []


def test_one_shared_token_next_door_does_not_downgrade_a_real_drift(tree):
    """`arrival_v2_experiment_report.md:285`, which hid behind the offset grade.

    The row quotes four identifiers; all four are on the loop it describes, and only the
    commonest of them, `num_envs`, happens to appear a line from the stale citation.
    Nearest-by-any-token answered with the neighbour, and a 13-line drift was filed as a
    one-line offset -- out of `--strict`, and out of sight.
    """
    _write(tree, "scripts/train_sac.py", "\n".join([
        "def train():",                                  # 1
        "    episode_length = np.zeros(num_envs)",       # 2  one token, adjacent
        "    episode_idx = start_episode",               # 3  the stale citation
        "    pad = 0",
        "    pad = 0",
        "    pad = 0",
        "    for env_step in range(start_step // num_envs + 1, total):",  # 7  all four
    ]))
    _write(tree, "scripts/x.md",
           "| 单位混淆 | `train_sac.py:3` | `env_step`、`num_envs`、`range`、`start_step` |")

    drift = bucket(tree, "drift")

    assert [r[2] for r in drift] == [3]
    assert drift[0][5] == 7, "report the line carrying the whole fragment"
    assert bucket(tree, "drift-near") == []


def test_a_function_name_that_exists_nowhere(tree):
    """CLAUDE.md's `get_probe_positions()`, reconstructed.

    The file resolved, the docstring description matched, and only the name was invented
    -- so nothing that checks paths could have caught it.
    """
    _write(tree, "docs/a.md",
           "探针坐标规定在 `auv_nav/flow.py` 的 `get_probe_positions()` docstring 里")

    missing = bucket(tree, "symbol-missing")

    assert len(missing) == 1
    assert missing[0][1] == "get_probe_positions"
    assert missing[0][3] == ["auv_nav/flow.py"], "name the file that put it in scope"


def test_a_function_that_lives_in_a_different_file_is_a_known_blind_spot(tree):
    """The documented limit, asserted so it cannot be lost or re-added by accident.

    Judging *where* a name is defined was built first and then measured: every hit it
    produced against this repo was a false positive, because a `.py` path beside a
    `Foo()` is a topic and not an attribution. Existence is what survives the evidence,
    and this case is the price.
    """
    _write(tree, "auv_nav/env.py", "class PlanarRemusEnv:\n    pass\n")
    _write(tree, "docs/a.md", "`PlanarRemusEnv()` 在 `auv_nav/flow.py` 里")

    assert bucket(tree, "symbol-missing") == []


def test_a_symbol_named_beside_a_file_is_not_a_claim_about_that_file(tree):
    """`fql_succession_p0p1_spec.md:479`, the false positive that forced the rewrite.

    The sentence says the metrics dict must stay compatible with that file's logger.
    File attribution read it as "FQLAgent lives in train_offline.py" and failed it.
    """
    _write(tree, "auv_nav/fql.py", "class FQLAgent:\n    def update(self):\n        pass\n")
    _write(tree, "scripts/train_offline.py", "def main():\n    pass\n")
    _write(tree, "docs/a.md",
           "`FQLAgent.update()` 返回的 key 要与 `scripts/train_offline.py` 的 logger 兼容")

    assert bucket(tree, "symbol-missing") == []


def test_a_dotted_symbol_is_judged_by_its_leaf_once_the_head_is_ours(tree):
    """`PlanarRemusEnv.compute_flow_at_position()` -- real class, invented method."""
    _write(tree, "auv_nav/env.py",
           "class PlanarRemusEnv:\n    def step(self):\n        pass\n")
    _write(tree, "docs/a.md",
           "由 `PlanarRemusEnv.compute_flow_at_position()` 采样，喂给 `auv_nav/flow.py`")

    missing = bucket(tree, "symbol-missing")

    assert len(missing) == 1
    assert missing[0][1] == "PlanarRemusEnv.compute_flow_at_position"
    assert missing[0][2] == "compute_flow_at_position", "the leaf is what was tested"


def test_a_dotted_symbol_whose_head_is_unknown_is_skipped(tree):
    """An unrecognised head is another package, or an instance attribute.

    `sampler.interpolate()` sitting next to a repo path says nothing about the repo,
    and reporting `interpolate` as missing would be a guess dressed as a finding.
    """
    _write(tree, "docs/a.md", "`auv_nav/flow.py` 里调用 `sampler.interpolate()`")

    assert bucket(tree, "symbol-missing") == []


def test_a_module_name_counts_as_a_known_head(tree):
    _write(tree, "scripts/train_utils.py", "def resolve_episodes():\n    pass\n")
    _write(tree, "docs/a.md", "`auv_nav/flow.py` 的调用方是 `train_utils.pick_episodes()`")

    assert [r[2] for r in bucket(tree, "symbol-missing")] == ["pick_episodes"]


def test_a_deeper_dotted_chain_is_beyond_what_can_be_attributed(tree):
    _write(tree, "auv_nav/env.py", "class PlanarRemusEnv:\n    pass\n")
    _write(tree, "docs/a.md",
           "`auv_nav/flow.py`：`PlanarRemusEnv.backend.last_delta_r_cmd()`")

    assert bucket(tree, "symbol-missing") == []


def test_a_symbol_with_no_repo_file_on_the_line_is_not_judged(tree):
    """The gate. A `name()` in prose naming no source file is as likely to be a shell
    builtin or a cited paper's notation as it is to be ours."""
    _write(tree, "docs/a.md", "编译用 `latexmk -silent`，失败再跑一次 `rerunfilecheck()`")

    assert bucket(tree, "symbol-missing") == []


def test_third_party_calls_are_not_read_as_claims_about_the_file(tree):
    """Covered by the unknown-head rule, not by FOREIGN_PREFIX -- `np` is nothing here.

    Kept as the shape a reader expects to see tested; the case that actually needs the
    prefix list is the collision below.
    """
    _write(tree, "docs/a.md", "`auv_nav/flow.py` 用的是 `np.zeros()`")

    assert bucket(tree, "symbol-missing") == []


def test_a_third_party_alias_beats_a_repo_class_of_the_same_name(tree):
    """What FOREIGN_PREFIX is for, once the unknown-head rule has taken the easy cases.

    `F.relu()` is torch, whatever else in the repo happens to be called `F`. Without the
    prefix list the head resolves to that class and `relu` gets reported as missing.
    """
    _write(tree, "auv_nav/f.py", "class F:\n    pass\n")
    _write(tree, "docs/a.md", "`auv_nav/flow.py` 的激活用 `F.relu()`")

    assert bucket(tree, "symbol-missing") == []


def test_a_renamed_symbol_may_be_declared_in_the_same_sentence(tree):
    """The HISTORICAL branch, which nothing exercised until now.

    A plan doc names what the implementer was told to write; the code shipped under a
    different name. Correcting the plan would rewrite what it planned, so it declares
    instead -- the same contract `check_doc_pointers` gives an absent path, and it only
    counts when the real name is given alongside.
    """
    _write(tree, "docs/a.md",
           "让实现者去 `auv_nav/flow.py` 找 `get_probe_positions()`"
           "（原计划名，落地时定名为 `make_probe_offsets`）")

    assert bucket(tree, "symbol-missing") == []
    assert [r[1] for r in bucket(tree, "declared")] == ["get_probe_positions"]


def test_a_backticked_path_is_not_read_as_the_fragment(tree):
    """A neighbouring path in backticks is not a quotation of the target's code.

    Isolated from the multi-citation rule below on purpose: the two mechanisms mask
    each other, and a mutation test showed that a single case covering both stays green
    when either one is removed.
    """
    _write(tree, "docs/a.md", "见 `auv_nav/flow.py:4`，判据见 `docs/policy.md`")

    assert bucket(tree, "drift") == []
    assert bucket(tree, "drift-near") == []


def test_several_citations_on_one_line_disable_fragment_attribution(tree):
    """The impact-assessment review cites six sections on a single line.

    Asking which of them the neighbouring fragment belongs to has no answer, so no
    fragment is attributed to any of them -- twenty such lines were false findings on
    the first calibration run.
    """
    _write(tree, "auv_nav/env.py", "x = 1\n" * 20)
    _write(tree, "docs/a.md",
           "`auv_nav/flow.py:9` 与 `auv_nav/env.py:1` 都用到 `make_probe_offsets`")

    assert bucket(tree, "drift") == []
    assert bucket(tree, "drift-near") == []


def test_a_fragment_far_along_the_line_is_not_attributed(tree):
    """Findings tables here run past a thousand characters per row.

    One row cited a script at one end and named a figure at the other; whole-line
    attribution read the figure name as code that should appear at the cited line.
    """
    filler = "。".join(["说明"] * 120)
    _write(tree, "docs/a.md",
           f"见 `auv_nav/flow.py:4`{filler}，另有图 `algo_interaction` 待重出")

    assert bucket(tree, "drift") == []
    assert any(row[4] == "not-attributable" or row[4] == "no-fragment"
               for row in bucket(tree, "clean"))


def test_a_basename_collision_prefers_the_citing_files_own_subtree(tree):
    """`setup.tex` exists under both the live chapter and the archived paper."""
    _write(tree, "paper/thesis_ch5/sections/setup.tex", "\\section{live}\n" * 5)
    _write(tree, "paper/archive/old/sections/setup.tex", "\\section{archived}\n" * 5)
    _write(tree, "paper/thesis_ch5/notes/n.md", "见 `setup.tex:3`")

    assert bucket(tree, "ambiguous") == []
    assert any(r[3] == "paper/thesis_ch5/sections/setup.tex"
               for r in bucket(tree, "clean"))


def test_a_basename_collision_with_no_winner_stays_ambiguous(tree):
    _write(tree, "paper/thesis_ch5/sections/setup.tex", "\\section{live}\n" * 5)
    _write(tree, "paper/archive/old/sections/setup.tex", "\\section{archived}\n" * 5)
    _write(tree, "docs/a.md", "见 `setup.tex:3`")

    rows = bucket(tree, "ambiguous")

    assert len(rows) == 1
    assert len(rows[0][3]) == 2


def test_a_prose_anchor_is_checked_for_existence_only(tree):
    """A `.tex` line anchor in a review note means "the passage around here".

    The reviewer wrote it against a draft that has been edited above the anchor since, so
    it lands a few lines off -- on a blank line as often as not -- and correcting the
    number would falsify what the record says was read. Five of this sweep's first ten
    blank-line findings were review notes of exactly that shape. Out-of-range still
    applies: that one is unambiguous whatever the target is.
    """
    _write(tree, "paper/s.tex", "\\section{a}\n\\label{sec:x}\n\ntext\n")
    _write(tree, "docs/a.md", "见 `paper/s.tex:3` 的 `sec:x`")
    _write(tree, "docs/b.md", "见 `paper/s.tex:99` 的 `sec:x`")

    assert bucket(tree, "drift") == []
    assert bucket(tree, "blank-line") == []
    assert [r[0] for r in bucket(tree, "out-of-range")] == ["docs/b.md:1"]


def test_agent_worktrees_are_not_scanned(tree):
    """A worktree is a full second checkout living inside the repo.

    Walking one doubled this repo's markdown corpus from 128 to 240 and reported that
    copy's pointers as the working tree's (2d4ba97). The exemption is shared with
    `check_doc_pointers` rather than restated, so the two cannot drift apart.
    """
    _write(tree, ".claude/worktrees/w/docs/a.md", "见 `auv_nav/flow.py:400`")

    assert refs.scan(str(tree))["out-of-range"] == []
    assert "worktrees" in refs.SKIP_DIRS


def test_dot_claude_itself_is_still_scanned(tree):
    """Only the worktree root is exempt. Project tooling under .claude/ cites real paths."""
    _write(tree, ".claude/agents/a.md", "见 `auv_nav/flow.py:400`")

    assert len(refs.scan(str(tree))["out-of-range"]) == 1


def test_a_range_citation_is_resolved_from_its_first_line(tree):
    _write(tree, "docs/a.md", "见 `auv_nav/flow.py:4-5` 的 `make_probe_offsets`")
    _write(tree, "docs/b.md", "见 `auv_nav/flow.py:4–5` 的 `make_probe_offsets`")

    buckets = refs.scan(str(tree))

    assert buckets["drift"] == []
    assert buckets["out-of-range"] == []


def test_cli_exits_nonzero_only_under_strict(tmp_path):
    """`--strict` is what a hook would run; without it the sweep is a report."""
    _write(tmp_path, "auv_nav/flow.py", "a = 1\n")
    _write(tmp_path, "docs/a.md", "见 `auv_nav/flow.py:99`")

    def run(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "scripts.check_doc_code_refs",
             "--root", str(tmp_path), *extra],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8")

    plain, strict = run(), run("--strict")

    assert plain.returncode == 0, plain.stdout + plain.stderr
    assert strict.returncode == 1
    assert "行号越界" in strict.stdout


def test_cli_is_silent_on_a_clean_tree(tmp_path):
    _write(tmp_path, "auv_nav/flow.py", "def f():\n    return 1\n")
    _write(tmp_path, "docs/a.md", "见 `auv_nav/flow.py:1` 的 `def f`")

    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "scripts.check_doc_code_refs",
         "--root", str(tmp_path), "--strict"],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8")

    assert proc.returncode == 0, proc.stdout + proc.stderr
