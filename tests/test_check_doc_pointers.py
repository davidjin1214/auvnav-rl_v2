"""Tests for the repo-wide markdown pointer sweep.

This sweep runs unattended -- a PostToolUse hook fires it in `--strict` mode after every
markdown edit -- so a regression in it is silent by construction: pointers stop being
checked and nothing says so. Each test below pins one behaviour to the incident that
produced it, and each was confirmed to go red when that behaviour is removed.

The `never` bucket gets the most attention because it is the only one holding a *claim*
rather than an observation: the citing doc says a path was planned and never built, and
the sweep takes its word. A false declaration silences its own alarm permanently. That
cross-check was fault-injected by hand once, on the day it was written (a507eb3); this
makes the injection permanent.

Every test runs against a fixture tree. Pointed at this repo, a genuine new dead link
would show up as a failing test, which inverts what the assertion means.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_doc_pointers as cdp

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, relpath: str, text: str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """An empty tree standing in for the repo root."""
    monkeypatch.setattr(cdp, "ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture()
def git_repo(repo):
    """The same tree, with history -- `verify_declarations` adjudicates against git."""
    env = ["-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    return lambda *args: subprocess.run(
        ["git", "-C", str(repo), *env, *args], check=True, capture_output=True)


def run_sweep(argv: list[str], capsys) -> tuple[int, str]:
    old = sys.argv
    sys.argv = ["check_doc_pointers", *argv]
    try:
        code = cdp.main()
    finally:
        sys.argv = old
    return code, capsys.readouterr().out


def real_misses(out: str) -> int:
    for line in out.splitlines():
        if line.startswith("未解析 "):
            return int(line.split("真失效 ")[1].split(" ")[0])
    raise AssertionError(f"no summary line in:\n{out}")


# --------------------------------------------------------------------------- corpus


def test_agent_worktrees_are_not_part_of_the_corpus(repo):
    """One run read 240 markdown files and 67 dead pointers against the real 128 and 0.

    An agent worktree is a full second checkout at `.claude/worktrees/<name>/`, so
    walking it counts every doc twice and resolves that copy's pointers against the main
    tree's root (2d4ba97).
    """
    _write(repo, "docs/a.md", "x")
    _write(repo, ".claude/worktrees/w/docs/a.md", "x")

    assert [Path(p).name for p in cdp.md_files()] == ["a.md"]


def test_dot_claude_itself_is_still_walked(repo):
    """Only the worktree root is exempt.

    `.claude/` used to be skipped wholesale, on the assumption that agent and skill
    definitions were all illustration. That stopped being true once the CLI reference
    moved out of CLAUDE.md and into `.claude/skills/`, and the exempt area was then
    holding real script paths that nothing checked.
    """
    _write(repo, ".claude/skills/s/SKILL.md", "x")

    assert len(cdp.md_files()) == 1


# ----------------------------------------------------------------------- extraction


def test_a_jsonl_path_is_not_truncated_to_a_json_path():
    """`train_log.jsonl` was once read as `train_log.json` and reported missing.

    Two separate mechanisms in one regex keep it whole. This one is the alternation
    order -- `jsonl` is listed before `json`, so it wins the match outright.
    """
    hits = [m.group(1) for m in cdp.BARE.finditer("见 `results/x/train_log.jsonl`")]

    assert hits == ["results/x/train_log.jsonl"]


@pytest.mark.parametrize("text", ["docs/notes.mdx", "scripts/a.python"])
def test_an_extension_may_not_match_a_prefix_of_a_longer_one(text):
    """The other mechanism: the trailing guard.

    Measured rather than assumed -- the alternation order alone already keeps `.jsonl`
    whole, so the guard is what stops `.mdx` and `.python` from being read as `.md` and
    `.py`. A test using the `.jsonl` example passes with the guard removed.
    """
    assert cdp.BARE.findall(f"见 `{text}`") == []


def test_a_footnote_definition_is_not_a_link_reference():
    """`[^1]: Denis ...` is prose whose first word looked like a path.

    A bibliography written that way reported six broken pointers named after its
    authors.
    """
    assert cdp.REFDEF.findall("[^1]: Denis Tarasov, Vladislav Kurenkov\n") == []
    assert cdp.REFDEF.findall("[spec]: docs/a.md\n") == [("spec", "docs/a.md")]


def test_links_inside_a_fenced_block_are_not_pointers(repo, capsys):
    """A plan doc embeds the future report's header as a template.

    Its nine cross-references are written relative to where that report would live, not
    to the plan citing it, so resolving them against the plan reported nine misses for
    links that were never links.
    """
    _write(repo, "docs/a.md",
           "```\n[gone](nowhere/at/all.md)\n```\n")

    _, out = run_sweep([], capsys)

    assert real_misses(out) == 0


def test_bare_paths_inside_a_fence_are_still_checked(repo, capsys):
    """A path in a ```bash block is still a claim about the repo.

    Run commands cite scripts from exactly there.
    """
    _write(repo, "docs/a.md",
           "```bash\npython -m scripts.nope\n```\n"
           "```bash\ncat scripts/does_not_exist.py\n```\n")

    _, out = run_sweep([], capsys)

    assert real_misses(out) == 1


# ------------------------------------------------------------------------ resolving


def test_a_line_anchor_does_not_stop_the_file_from_resolving(repo):
    """`scripts/train_utils.py:185` still points at the file.

    Whether the line number itself is still right is `check_doc_code_refs`'s job.
    """
    _write(repo, "scripts/train_utils.py", "x = 1\n")

    target = cdp.resolve("scripts/train_utils.py:185", "bare", str(repo / "docs/a.md"))

    assert Path(target).name == "train_utils.py"


def test_a_bare_path_resolves_against_the_citing_files_ancestors(repo):
    """`figures/scripts/x.py` written inside paper/thesis_ch5/ means that copy.

    Root-anchoring alone false-positives on it.
    """
    src = _write(repo, "paper/thesis_ch5/notes/n.md", "见 figures/scripts/x.py")
    _write(repo, "paper/thesis_ch5/figures/scripts/x.py", "x = 1\n")

    target = cdp.resolve("figures/scripts/x.py", "bare", str(src))

    assert Path(target).exists()


# -------------------------------------------------------------------------- triage


@pytest.mark.parametrize("raw,line,heading,expected", [
    ("docs/gone.md", "普通一句 `docs/gone.md`", "# 标题", "real"),
    ("results/run/x.json", "见 `results/run/x.json`", "# 标题", "artifact"),
    ("docs/<name>.md", "模板 `docs/<name>.md`", "# 标题", "example"),
    ("/abs/path.md", "见 /abs/path.md", "# 标题", "abs"),
    ("docs/x.md", "| `docs/x.md` | TBD |", "# 标题", "plan"),
    ("docs/x.md", "| `docs/x.md` | 说明 |", "## 待创建文件", "plan"),
])
def test_triage_buckets(raw, line, heading, expected):
    assert cdp.triage("docs/src.md", raw, line, heading) == expected


def test_an_artifact_path_is_not_a_defect_on_a_fresh_clone(repo, capsys):
    """The five data directories are gitignored; absent here is by design."""
    _write(repo, "docs/a.md", "读数在 `results/offline/rebrac/test_result.json`")

    code, out = run_sweep(["--strict"], capsys)

    assert real_misses(out) == 0
    assert code == 0


# ------------------------------------------------------- the self-declaration bucket


def test_a_declaration_must_sit_on_the_same_line_as_the_path(repo):
    """The declaration IS the note a reader sees, not a machine-only list beside it.

    A separate list would drift away from the prose it excuses.
    """
    same = _write(repo, "docs/same.md", "`docs/planned.md` 从未产出。\n")
    apart = _write(repo, "docs/apart.md", "从未产出。\n\n见 `docs/planned.md`\n")

    assert cdp.declared_never(str(same), same.read_text(encoding="utf-8").split("\n"))
    assert not cdp.declared_never(str(apart),
                                  apart.read_text(encoding="utf-8").split("\n"))


@pytest.mark.parametrize("wording", [
    "`docs/x.md` 从未产出", "`docs/x.md` 用后即删",
    "`docs/x.md` 原计划文件名", "`docs/x.md` 未入库",
])
def test_all_four_accepted_reasons_are_recognised(repo, wording):
    """The list is closed on purpose: anything else is a defect, not a category."""
    src = _write(repo, "docs/a.md", wording + "\n")

    assert cdp.declared_never(str(src), [wording])


def test_an_undeclared_miss_is_not_excused(repo, capsys):
    _write(repo, "docs/a.md", "见 `docs/planned.md`\n")

    code, out = run_sweep(["--strict"], capsys)

    assert real_misses(out) == 1
    assert code == 1


def test_a_declared_miss_leaves_strict_clean(repo, capsys):
    _write(repo, "docs/a.md", "`docs/planned.md` 从未产出，本线取消时它还没写。\n")

    code, out = run_sweep(["--strict"], capsys)

    assert real_misses(out) == 0
    assert code == 0


def test_a_declaration_is_scoped_to_the_file_that_makes_it(repo, capsys):
    """Another doc citing the same path still reports `real`.

    There the miss may well be a genuine defect -- the excuse belongs to whoever wrote
    the sentence carrying it.
    """
    _write(repo, "docs/a.md", "`docs/planned.md` 从未产出。\n")
    _write(repo, "docs/b.md", "见 `docs/planned.md`\n")

    code, out = run_sweep(["--strict"], capsys)

    assert real_misses(out) == 1
    assert code == 1


# ------------------------------------------------ git adjudication of those claims


def test_a_false_never_built_claim_is_flagged_and_fails_strict(repo, git_repo, capsys):
    """The fault injection from a507eb3, made permanent.

    A doc declaring "planned and never built" about a file that did exist and was merely
    moved silences a real alarm, and nothing downstream would ever notice. Git can
    adjudicate: the path was added once, so the claim is false.
    """
    _write(repo, "docs/was_real.md", "早年真的有这份文档\n")
    git_repo("add", "-A")
    git_repo("commit", "-qm", "add it")
    (repo / "docs/was_real.md").unlink()
    _write(repo, "docs/a.md", "`docs/was_real.md` 从未产出。\n")

    code, out = run_sweep(["--strict", "--verify-declarations"], capsys)

    assert "[SUSPECT]" in out
    assert code == 1, "a false declaration must not pass --strict"


def test_a_truthful_never_built_claim_stays_consistent(repo, git_repo, capsys):
    _write(repo, "docs/a.md", "`docs/never_written.md` 从未产出。\n")
    git_repo("add", "-A")
    git_repo("commit", "-qm", "just the doc")

    code, out = run_sweep(["--strict", "--verify-declarations"], capsys)

    assert "[SUSPECT]" not in out
    assert code == 0


def test_declarations_are_only_adjudicated_when_asked(repo, git_repo, capsys):
    """The cross-check shells out to git per declaration; the sweep itself stays fast."""
    _write(repo, "docs/was_real.md", "x\n")
    git_repo("add", "-A")
    git_repo("commit", "-qm", "add it")
    (repo / "docs/was_real.md").unlink()
    _write(repo, "docs/a.md", "`docs/was_real.md` 从未产出。\n")

    code, out = run_sweep(["--strict"], capsys)

    assert "SUSPECT" not in out
    assert code == 0


# ------------------------------------------------------------------------ the hook


def test_the_hook_only_fires_on_markdown(tmp_path):
    """A hook must never be the reason an edit of something else fails."""
    hook = REPO_ROOT / ".claude/hooks/doc_pointers.py"
    payload = '{"tool_input": {"file_path": "auv_nav/env.py"}}'

    proc = subprocess.run([sys.executable, "-X", "utf8", str(hook)],
                          input=payload, capture_output=True, text=True,
                          encoding="utf-8", cwd=REPO_ROOT)

    assert proc.returncode == 0
    assert proc.stderr == ""


def test_the_hook_survives_junk_on_stdin():
    hook = REPO_ROOT / ".claude/hooks/doc_pointers.py"

    proc = subprocess.run([sys.executable, "-X", "utf8", str(hook)],
                          input="not json at all", capture_output=True, text=True,
                          encoding="utf-8", cwd=REPO_ROOT)

    assert proc.returncode == 0


def _hook_module():
    """`.claude/hooks/` is not a package, so load the file directly."""
    import importlib.util

    path = REPO_ROOT / ".claude/hooks/doc_pointers.py"
    spec = importlib.util.spec_from_file_location("doc_pointers_hook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_hook_runs_every_sweep_not_only_the_path_one():
    """Wiring, asserted separately from behaviour.

    The three sweeps check disjoint things -- whether the path exists, whether the line
    and the name behind it still do, and whether a published figure still recomputes --
    so dropping any one leaves the others passing and nothing else says so.
    """
    modules = [module for module, _marker, _hint in _hook_module().SWEEPS]

    assert modules == ["scripts.check_doc_pointers",
                       "scripts.check_doc_code_refs",
                       "scripts.audit_published_numbers"]


def test_the_defect_filter_keeps_only_the_failing_sections():
    """Both sweeps print every bucket; only the graded-defect ones belong in stderr."""
    report = "\n".join([
        "--- ★ 行号越界（文件没有那么多行）：1 ---",
        "  docs/a.md:3  -> auv_nav/env.py:9999",
        "--- 行号偏移（引用方式使然，不判缺陷）：1 ---",
        "  docs/b.md:4  -> auv_nav/env.py:377",
    ])

    kept = _hook_module()._defect_section(report, "★")

    assert "env.py:9999" in kept
    assert "env.py:377" not in kept


def test_the_hook_is_quiet_on_this_repo_as_it_stands():
    """The one case here that reads the live tree, and the point of doing so.

    Both sweeps are at zero right now. If that stops being true the hook starts
    rejecting every markdown edit in the repo, including edits that have nothing to do
    with the finding -- better to learn that from a test than from a blocked turn.
    """
    hook = REPO_ROOT / ".claude/hooks/doc_pointers.py"
    payload = '{"tool_input": {"file_path": "docs/DOC_INDEX.md"}}'

    proc = subprocess.run([sys.executable, "-X", "utf8", str(hook)],
                          input=payload, capture_output=True, text=True,
                          encoding="utf-8", cwd=REPO_ROOT)

    assert proc.returncode == 0, proc.stderr
