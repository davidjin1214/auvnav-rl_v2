"""Generate (or verify) docs/DOC_INDEX.md -- a map of every markdown file in the repo.

Why this exists: markdown lives in 21 different directories here, and an audit on
2026-08-16 found 11 of the 52 top-level docs/ files reachable from no entry point at
all, plus 20 whose validity could not be told without opening them.

Why it is generated rather than hand-written: CLAUDE.md names pointer rot as this
index's standing failure mode. A hand-written map goes stale silently. This one is
rebuilt from the files themselves, and `--check` fails when it has drifted, so the
staleness becomes visible instead of being quoted as fact.

What is extracted, and nothing more:
  title   the H1 line (all 54 docs/ files have one)
  status  a banner keyword in the first 15 lines, with its date when present

A file with no banner is reported as "--", NOT as active: the absence of a banner is
absence of evidence. Deciding that a doc is still current stays human work.

Usage:
    python -m scripts.build_doc_index            # write docs/DOC_INDEX.md
    python -m scripts.build_doc_index --check    # exit 1 if the file is out of date
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/DOC_INDEX.md"
HEAD_LINES = 15

# Tool caches ship their own README.md, which would otherwise enter the index the
# first time anyone runs pytest here -- making the index depend on what commands
# happened to have run on this machine.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints",
    ".venv", "venv", ".tmp", "tmp",
    # Agent worktrees are full second checkouts living at .claude/worktrees/<name>/.
    # Indexing one lists every doc twice, and --check then fails purely because a
    # worktree happens to exist right now.
    "worktrees",
}

# Gitignored data directories, skipped at the repo top level only. They may be
# junctions to another volume, so rglob otherwise walks into ~40 GB of run output --
# which carries markdown of its own (results/archived/, plus a stale pre-move copy of
# experiments/auvhamnode_spike/ that a68b4c2 relocated to docs/). None of it belongs in
# a document index, and whether it shows up at all depends on how a given machine
# mounts its data, which would make the index machine-dependent.
DATA_DIRS = {"results", "experiments", "checkpoints", "wake_data", "offline_data"}

# Ordered: first match wins, so SUPERSEDED beats a passing mention of "archive".
STATUS_PATTERNS: list[tuple[str, str]] = [
    ("SUPERSEDED", r"SUPERSEDED"),
    ("DEPRECATED", r"DEPRECATED|deprecated"),
    ("PAUSED", r"PAUSED|⏸"),
    ("CLOSED", r"\bCLOSED\b|已闭环|已收口"),
    ("CANCELLED", r"已撤销|CANCELLED"),
    ("ARCHIVE", r"\barchive\b|归档"),
]
DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")

# Section grouping: (heading, predicate on repo-relative posix path)
GROUPS: list[tuple[str, str]] = [
    ("入口与总纲", "^(README|CLAUDE|AGENTS)\\.md$"),
    ("docs/ — 研究与实现文档", "^docs/[^/]+\\.md$"),
    ("docs/archive/fql_succession/ — FQL succession P2 之前的施工记录（归档=移位+标注，非有效性判断）",
     "^docs/archive/fql_succession/"),
    ("docs/auvhamnode_spike/ — AUVHamNODE 预 spike 审计（⏸ 线已暂停）", "^docs/auvhamnode_spike/"),
    ("docs/offline_mbrl_plan/ — 已废弃的 MBRL 草案（未纳入 git）", "^docs/offline_mbrl_plan/"),
    ("docs/superpowers/ — 早期 plan/spec 存档", "^docs/superpowers/"),
    ("docs/assets/ — 图注", "^docs/assets/"),
    ("paper/thesis_ch5/ — 第 5 章活跃工作文件", "^paper/thesis_ch5/[^/]+\\.md$"),
    ("paper/thesis_ch5/notes/ — 第 5 章过程存档", "^paper/thesis_ch5/notes/"),
    ("paper/ — 其它（写作规范 / 工具说明 / 已归档论文）", "^paper/"),
    (".claude/ — 项目工具定义（须留在原位，Claude Code 按路径加载）", "^\\.claude/"),
    ("其它", ".*"),
]


def head_of(p: Path) -> list[str]:
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            return [next(fh, "") for _ in range(HEAD_LINES)]
    except OSError:
        return []


def title_of(lines: list[str], fallback: str) -> str:
    for ln in lines:
        if ln.startswith("# "):
            t = ln[2:].strip()
            return t if len(t) <= 88 else t[:87] + "…"
    return fallback


def status_of(lines: list[str]) -> str:
    """Read the doc's own banner only.

    Scoped deliberately tight: the banner is a blockquote right after the H1, and the
    keyword is emphasised (`**SUPERSEDED**`) or flagged (`⚠`). Scanning the whole head
    instead misreads a doc that merely *mentions* another line's state -- CLAUDE.md and
    README.md both got tagged PAUSED that way, because they describe the paused
    AUVHamNODE line in running prose. Under-reporting is the safe direction here.
    """
    seen_h1 = False
    quote: list[str] = []
    for ln in lines:
        if not seen_h1:
            if ln.startswith("# "):
                seen_h1 = True
            continue
        if ln.startswith(">"):
            quote.append(ln)
        elif quote and ln.strip():
            break  # blockquote ended at real prose
    head = "".join(quote)
    if not head:
        return "—"
    for label, pat in STATUS_PATTERNS:
        for m in re.finditer(rf"(\*\*[^*]{{0,12}}|⚠\s*)?({pat})", head):
            # A banner keyword is prose; inside a path it is just a directory name.
            # Once the pre-P2 FQL records moved under docs/archive/, every banner
            # citing one carried a literal "archive" whose neighbouring `**Spec**:`
            # label satisfied the emphasis test below -- tagging a live dataset card
            # ARCHIVE, the exact opposite of its state. A slash on either side means
            # path, not banner. Kept per-occurrence rather than per-doc so a real
            # banner further down the same blockquote still registers.
            # This was one misfire of its kind, not the only possible one: after moving
            # docs, rebuild and read the status column's diff line by line -- a matching
            # total hides a doc that flipped from live to ARCHIVE.
            kw_start, kw_end = m.start(2), m.end(2)
            tail = re.match(r"[^\s`)\]]*", head[kw_end:]).group(0)
            looks_like_path = head[kw_start - 1 : kw_start] == "/" or (
                tail.startswith("/")
                and (tail.count("/") >= 2 or re.search(r"\.\w+$", tail))
            )
            if looks_like_path:
                continue
            # require emphasis or a warning glyph adjacent -- plain prose mentions don't count
            ctx = head[max(0, m.start() - 4) : m.end() + 4]
            if "**" not in ctx and "⚠" not in ctx:
                continue
            d = DATE_RE.search(head[m.start() : m.start() + 120])
            return f"{label} {d.group(1)}" if d else label
    return "—"


def collect() -> list[tuple[str, str, str]]:
    rows = []
    for p in sorted(REPO.rglob("*.md")):
        if SKIP_DIRS & set(p.parts):
            continue
        rel = p.relative_to(REPO).as_posix()
        if rel.split("/", 1)[0] in DATA_DIRS:
            continue
        if rel == OUT.relative_to(REPO).as_posix():
            continue
        lines = head_of(p)
        rows.append((rel, title_of(lines, p.stem), status_of(lines)))
    return rows


def render(rows: list[tuple[str, str, str]]) -> str:
    used: set[str] = set()
    out: list[str] = [
        "# 全仓文档索引",
        "",
        "> **自动生成，请勿手改。** 由 `python -m scripts.build_doc_index` 重建；",
        "> `--check` 会在本文件落后于实际时报错。",
        ">",
        "> 只提取两样东西：H1 标题、以及文件开头 15 行内的状态横幅。",
        "> **状态栏的 `—` 表示「该文档未自我标注」，不表示「仍然有效」**——没有横幅是证据缺失，",
        "> 判断一份文档是否仍然作数，仍然是人的活。",
        ">",
        "> 研究线的叙事索引（阶段时间轴、结论路由、数字 ground truth）在",
        "> [`offline_rl_line_summary.md`](offline_rl_line_summary.md) 与",
        "> [`online_rl_line_summary.md`](online_rl_line_summary.md)；本文件是**文件地图**，两者角色不同。",
        "",
        f"共 {len(rows)} 个 markdown 文件。",
        "",
    ]
    for heading, pat in GROUPS:
        rx = re.compile(pat)
        sel = [r for r in rows if r[0] not in used and rx.search(r[0])]
        if not sel:
            continue
        used.update(r[0] for r in sel)
        out += [f"## {heading}（{len(sel)}）", "", "| 文档 | 标题 | 自我标注状态 |", "|---|---|---|"]
        for rel, title, st in sel:
            link = Path(rel).name if rel.startswith("docs/") and rel.count("/") == 1 else "../" + rel
            title = title.replace("|", "\\|")
            out.append(f"| [`{rel}`]({link}) | {title} | {st} |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    text = render(collect())
    if "--check" in sys.argv:
        if not OUT.exists():
            print(f"FAIL: {OUT.relative_to(REPO)} does not exist -- run without --check")
            return 1
        cur = OUT.read_text(encoding="utf-8")
        if cur.replace("\r\n", "\n") != text.replace("\r\n", "\n"):
            print(f"FAIL: {OUT.relative_to(REPO)} is out of date -- rerun without --check")
            return 1
        print(f"OK: {OUT.relative_to(REPO)} is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}  ({text.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
