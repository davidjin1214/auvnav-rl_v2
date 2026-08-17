"""Repo-wide markdown pointer sweep.

Motivation: this repo's docs are held together by cross-pointers, and the pointers
rot silently. Two stale claims found on 2026-07-28 (`paper/thesis_ch5/status.md`'s
2026-07-08 layout claim, `CLAUDE.md`'s spec rev number) were both stumbled upon
rather than searched for. This script is the search.

Resolves three pointer forms across every tracked-or-not markdown file:
  1. inline links      [text](path)  and  [text](path#anchor)
  2. reference defs    [id]: path
  3. bare prose paths  `docs/foo.md`  /  scripts/bar.py  (backticked or plain)

Bare paths are tried against repo root first, then every ancestor of the citing
file -- `figures/scripts/x.py` written inside paper/thesis_ch5/ means
paper/thesis_ch5/figures/scripts/x.py, and root-anchoring alone false-positives on it.

Unresolvable pointers are triaged, because most are not defects:
  real      a doc pointing at something that should exist  <- the actionable bucket
  plan      a forecast, not a pointer: a row in a 待创建 / 已删除 table, or a table
            row still marked TBD. `rebrac_broad_validation_v2_plan.md` §11 alone
            accounts for seven of these -- four infra files the notebook ended up
            doing inline, two M1-conditional files that correctly never existed
            because M1 never triggered, and one the plan itself records deleting.
  never     the citing doc explains, in the same sentence, why this path is absent:
            planned and never built / built then deliberately deleted / shipped
            under a different name / generated at run time and left untracked.
            Cancelled and paused plan docs are full of the first kind -- 45 of the
            67 `real` misses on 2026-08-17 were one paused line and two cancelled
            ones pointing at modules nobody ever wrote. Deleting those links would
            erase what the line planned, which is the whole reason the plan doc is
            kept; so the doc declares them instead, and the declaration IS the
            visible note a reader sees, not a separate machine-only list that could
            drift away from it. Verify before writing one -- `git log
            --diff-filter=A -- <path>` distinguishes never-built from moved.
  artifact  under a gitignored产物 dir; absent on this machine by design
  example   an <angle>/glob/foo stand-in rather than a real path. No source
            directory is exempt -- .claude/agents/ and .claude/skills/ are
            walked like any other doc.
  abs       absolute file:// style path baked in by an old tool

Anchors are checked with GitHub's slug rules. Note that CJK headings with
full-width punctuation (（）：) slug differently than hand-written TOC links
usually assume -- most anchor misses in this repo are that, and are cosmetic.

What this canNOT check, and stays human work:
  - whether a pointer aims at the *right* doc
  - whether a deprecated/superseded banner is consistent with who cites it
  - whether a version number or dated claim inside prose is still true

Usage:
    python -m scripts.check_doc_pointers            # triaged report
    python -m scripts.check_doc_pointers --anchors  # + anchor misses
    python -m scripts.check_doc_pointers --orphans  # + docs nobody links to
    python -m scripts.check_doc_pointers --strict   # exit 1 on any `real` miss
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {".git", "node_modules", ".pytest_cache", ".pytest_tmp", "__pycache__",
             "wake_data", "offline_data", "checkpoints", ".ipynb_checkpoints",
             # Agent worktrees land *inside* the repo at .claude/worktrees/<name>/.
             # Each is a full second checkout, so walking one doubles the corpus and
             # reports that copy's pointers as if they were the working tree's: a run
             # with one worktree present read 240 markdown files and 67 dead pointers
             # against the real 128 and 0. The tool deliberately scans .claude/ for
             # project tooling, so only the worktree root is exempt, not .claude itself.
             "worktrees"}

INLINE = re.compile(r"\[([^\]\[]*)\]\(([^)\s]+?)(?:\s+\"[^\"]*\")?\)")
# `[^id]: text` is a footnote definition, not a link reference definition -- its first
# word is prose. The bibliography in NODE_IQL_FQL_SORL_revised_roadmap_v3.md otherwise
# reports six broken pointers named Denis, Ilya, Seohong, Nicolas, Divyansh and Justin.
REFDEF = re.compile(r"^\s{0,3}\[(?!\^)([^\]]+)\]:\s*(\S+)", re.M)
# `jsonl` precedes `json`, and the trailing guard stops any extension from matching a
# prefix of a longer one: without it `.../train_log.jsonl` was read as `train_log.json`
# and reported missing -- a broken pointer the repo never had.
BARE = re.compile(
    r"`?((?:docs|paper|scripts|auv_nav|notebooks|benchmarks|experiments|figures|results|"
    r"offline_data|\.claude)/[A-Za-z0-9_./\-]+"
    r"\.(?:md|tex|py|ipynb|jsonl|json|sh|npz|csv|bib)(?![A-Za-z0-9]))`?"
)
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)

ARTIFACT_DIR = re.compile(
    r"(?:^|/)(results|offline_data|wake_data|checkpoints|figures)/")
# No source directory is exempt. `.claude/` used to be: agents and skills were
# assumed to be all illustration. That stopped being true once the CLI reference
# moved out of CLAUDE.md and into .claude/skills/ -- the exempt area was holding
# real script paths and manifest keys that nothing checked. Both agent definitions
# now write their examples in <angle>/glob form, so PLACEHOLDER alone covers every
# legitimate stand-in and every concrete path gets resolved.
PLACEHOLDER = re.compile(r"(foo|bar|baz|<[a-zA-Z]|\*|\.\.\.|^path$|^archive/$)")

# A heading that makes every table row beneath it a forecast rather than a claim.
PLAN_HEADING = re.compile(r"(待创建|待建|已删除|已弃用|计划创建|planned|to be created|deleted)",
                          re.I)
# ...or a row that still carries its planning-time status cell.
PLAN_ROW = re.compile(r"\|\s*(TBD|待定|计划中|planned)\s*\|?\s*$", re.I)

# A doc may declare that paths it cites were planned and never built. The trigger
# phrase and the paths must sit on one line, so the declaration is the same sentence
# the reader sees -- a hidden list would drift away from the prose beside it. Scoped
# to the declaring file only: another doc citing the same path still reports `real`,
# because there the miss may well be a genuine defect.
# Closed list, four families, extend only deliberately: (1) planned, never built;
# (2) built and deliberately deleted; (3) shipped under a different name; (4) generated
# at run time and deliberately not committed. Anything else is a defect, not a category.
NEVER_PRODUCED = re.compile(
    r"从未产出|从未创建|从未建成|never produced|never built"
    r"|用后即删|deleted by design"
    r"|原计划文件名|原计划名|renamed"
    r"|不入 ?git|不是 tracked|非 tracked|not tracked|未入库",
    re.I)
DECL_PATH = re.compile(r"`([A-Za-z0-9_./\-]+\.[A-Za-z0-9]{1,6})`")

# Inside a fenced block, `[text](path)` is literal text, not a link -- it renders as
# the characters themselves. The v1 broad-validation plan embeds the future report's
# header as a template, and its nine cross-references are written relative to where
# that report would live (docs/), not to the plan citing it; resolving them against
# the plan's own directory reported nine misses for links that were never links.
# Bare paths stay in scope: a path inside a ```bash block is still a claim about the
# repo, and that is exactly where run commands cite scripts.
FENCE = re.compile(r"^\s*(```|~~~)")


def md_files() -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        out += [os.path.join(dirpath, f) for f in filenames if f.endswith(".md")]
    return sorted(out)


def slug(text: str) -> str:
    """GitHub-flavoured anchor slug."""
    t = re.sub(r"`([^`]*)`", r"\1", text)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = t.strip().lower()
    t = re.sub(r"[^\w一-鿿 \-]", "", t)
    return t.replace(" ", "-")


def anchors_of(path: str) -> set[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            return {slug(m.group(1)) for m in HEADING.finditer(fh.read())}
    except (OSError, UnicodeDecodeError):
        return set()


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace("\\", "/")


def resolve(raw: str, kind: str, src: str) -> str:
    """Return the best-guess absolute target for one pointer."""
    srcdir = os.path.dirname(src)
    stem = raw.split("#")[0]
    # `scripts/train_utils.py:185` anchors a line; the pointer is still the file.
    stem = re.sub(r":\d+(?:-\d+)?$", "", stem)
    if not stem:
        return src
    target = (os.path.join(ROOT, stem) if kind == "bare"
              else os.path.join(srcdir, stem))
    target = os.path.normpath(target)
    if kind == "bare" and not os.path.exists(target):
        probe = srcdir
        while len(probe) >= len(ROOT):
            cand = os.path.normpath(os.path.join(probe, stem))
            if os.path.exists(cand):
                return cand
            probe = os.path.dirname(probe)
    return target


def declared_never(src: str, lines: list[str]) -> set[str]:
    """Absolute targets this file itself declares as planned-and-never-built."""
    srcdir = os.path.dirname(src)
    out: set[str] = set()
    for line in lines:
        if not NEVER_PRODUCED.search(line):
            continue
        cands = ([m.group(1) for m in DECL_PATH.finditer(line)]
                 + [m.group(2) for m in INLINE.finditer(line)])
        for c in cands:
            c = c.split("#")[0]
            if not c or re.match(r"^(https?|mailto|ftp):", c):
                continue
            # Both anchorings, because the same target is written `auv_nav/x.py`
            # in prose and ../auv_nav/x.py in a link, often in the same doc.
            out.add(os.path.normpath(os.path.join(ROOT, c)))
            out.add(os.path.normpath(os.path.join(srcdir, c)))
    return out


def triage(src: str, raw: str, line: str, heading: str) -> str:
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        return "abs"
    if PLACEHOLDER.search(raw):
        return "example"
    if line.lstrip().startswith("|") and (PLAN_ROW.search(line.rstrip())
                                          or PLAN_HEADING.search(heading)):
        return "plan"
    if ARTIFACT_DIR.search("/" + raw):
        return "artifact"
    return "real"


# `never` is the only bucket that holds a *claim* instead of an observation: the doc says
# the path was planned and never built, and this script takes its word.  The failure mode is
# a doc writing "never produced" about a file that did exist and was merely moved -- that
# silences a real alarm permanently, and nothing downstream would ever notice.  Git can
# adjudicate: `git log --all --diff-filter=A -- <path>` says whether the path was ever added.
DECL_FAMILY = (
    ("planned-never-built", re.compile(r"从未产出|从未创建|从未建成|never produced|never built")),
    ("deleted-by-design", re.compile(r"用后即删|deleted by design")),
    ("renamed", re.compile(r"原计划文件名|原计划名|renamed")),
    ("not-tracked", re.compile(r"不入 ?git|不是 tracked|非 tracked|not tracked|未入库")),
)


def _ever_added(relpath: str) -> int:
    """How many commits, anywhere in history, added this exact path."""
    out = subprocess.run(
        ["git", "-C", ROOT, "log", "--all", "--diff-filter=A", "--format=%h", "--", relpath],
        capture_output=True)
    return len([l for l in out.stdout.decode("utf-8", "replace").splitlines() if l.strip()])


def verify_declarations() -> int:
    """Cross-check each `never` declaration against git history.  Returns SUSPECT count."""
    print("\n" + "=" * 96)
    print("自述缺席桶的验真（该桶装的是声明，不是事实；此处拿 git 历史逐条对质）")
    print("=" * 96)
    rows: list[tuple[str, int, str, str, str, int, str]] = []
    occurrences = 0  # the bucket counts every citation; a declaration may be cited many times
    for src in md_files():
        with open(src, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        never = declared_never(src, lines)
        if not never:
            continue
        srcdir = os.path.dirname(src)
        seen: set[tuple[str, str]] = set()
        in_fence = False
        for lineno, line in enumerate(lines, 1):
            if FENCE.match(line):
                in_fence = not in_fence
                continue
            found = [(m.group(1), "bare") for m in BARE.finditer(line)]
            if not in_fence:
                found += ([(m.group(2), "link") for m in INLINE.finditer(line)]
                          + [(m.group(2), "refdef") for m in REFDEF.finditer(line)])
            for raw, kind in found:
                if re.match(r"^(https?|mailto|ftp):", raw) or raw.startswith("#"):
                    continue
                target = resolve(raw, kind, src)
                if os.path.exists(target) or target not in never:
                    continue
                try:
                    trel = os.path.relpath(target, ROOT).replace("\\", "/")
                except ValueError:
                    trel = target
                occurrences += 1
                if (rel(src), trel) in seen:
                    continue
                seen.add((rel(src), trel))
                fams = set()
                for dline in lines:
                    if not NEVER_PRODUCED.search(dline):
                        continue
                    cands = ([m.group(1) for m in DECL_PATH.finditer(dline)]
                             + [m.group(2) for m in INLINE.finditer(dline)])
                    hit = any(os.path.normpath(os.path.join(a, c.split("#")[0])) == target
                              for c in cands if c and not re.match(r"^(https?|mailto|ftp):", c)
                              for a in (ROOT, srcdir))
                    if hit:
                        fams |= {n for n, rx in DECL_FAMILY if rx.search(dline)}
                adds = _ever_added(trel)
                if adds and fams == {"planned-never-built"}:
                    verdict = "SUSPECT"
                elif adds and "planned-never-built" in fams:
                    verdict = "check"
                elif not adds and fams == {"deleted-by-design"}:
                    verdict = "weak"
                else:
                    verdict = "consistent"
                rows.append((rel(src), lineno, raw, trel, "+".join(sorted(fams)) or "?",
                             adds, verdict))

    tally: dict[str, int] = {}
    for r in rows:
        tally[r[-1]] = tally.get(r[-1], 0) + 1
    print(f"声明 {len(rows)} 条（对应桶内 {occurrences} 处引用；同一声明常被多处引用）："
          + "、".join(f"{k} {n}" for k, n in sorted(tally.items())))
    print("  SUSPECT = 声明「计划过没建成」，但 git 确实新增过该路径 —— 警报被错误掐掉")
    print("  check   = 同上但声明族不止一个（可能是改名，须人读）")
    print("  weak    = 声明「用后即删」而 git 无该路径记录（与「未提交即删」自洽，不算错）")
    for src, ln, raw, trel, fam, adds, verdict in sorted(rows):
        if verdict == "consistent":
            continue
        print(f"\n  [{verdict}] {src}:{ln}  -> {raw}")
        print(f"        解析 {trel} ｜ 声明族 {fam} ｜ git 新增该路径的提交数 {adds}")
    return tally.get("SUSPECT", 0) + tally.get("check", 0)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="repo-wide markdown pointer sweep")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any `real` miss is found")
    ap.add_argument("--anchors", action="store_true",
                    help="also list #anchor misses (mostly CJK slug drift)")
    ap.add_argument("--orphans", action="store_true",
                    help="also list docs/ and paper/ files nothing links to")
    ap.add_argument("--verify-declarations", action="store_true",
                    help="cross-check every `never` declaration against git history "
                         "(the one bucket holding claims rather than observed facts)")
    args = ap.parse_args()

    files = md_files()
    anchor_cache: dict[str, set[str]] = {}
    buckets: dict[str, list] = {"real": [], "never": [], "plan": [], "artifact": [],
                                "example": [], "abs": []}
    bad_anchor: list[tuple[str, int, str]] = []
    cited: set[str] = set()
    total = 0

    for src in files:
        with open(src, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        never = declared_never(src, lines)
        heading = ""
        in_fence = False
        for lineno, line in enumerate(lines, 1):
            if FENCE.match(line):
                in_fence = not in_fence
                continue
            if line.startswith("#") and not in_fence:
                heading = line
            found = [(m.group(1), "bare") for m in BARE.finditer(line)]
            if not in_fence:
                found += ([(m.group(2), "link") for m in INLINE.finditer(line)]
                          + [(m.group(2), "refdef") for m in REFDEF.finditer(line)])
            for raw, kind in found:
                if re.match(r"^(https?|mailto|ftp):", raw):
                    continue
                total += 1
                target = (src if raw.startswith("#") else resolve(raw, kind, src))
                if not os.path.exists(target):
                    bucket = ("never" if target in never
                              else triage(rel(src), raw, line, heading))
                    buckets[bucket].append((rel(src), lineno, raw, kind))
                    continue
                cited.add(rel(target))
                anchor = raw.partition("#")[2]
                if anchor and target.endswith(".md"):
                    if target not in anchor_cache:
                        anchor_cache[target] = anchors_of(target)
                    if slug(anchor) not in anchor_cache[target]:
                        bad_anchor.append((rel(src), lineno, raw))

    miss = sum(len(v) for v in buckets.values())
    print(f"扫描 {len(files)} 个 markdown；解析到 {total} 个仓内指针")
    print("=" * 96)
    print(f"未解析 {miss} = 真失效 {len(buckets['real'])} / 自述缺席 "
          f"{len(buckets['never'])} / 计划表预告 {len(buckets['plan'])} / 产物路径 "
          f"{len(buckets['artifact'])} / 示例占位 {len(buckets['example'])} / 绝对路径 "
          f"{len(buckets['abs'])}")
    print("=" * 96)
    for name, label in (("real", "★ 真失效（文档指向本应存在的东西）"),
                        ("never", "自述缺席（引用方在同一句正文里说明了为何不存在：计划过没"
                                  "建成，或用后即删；按源文件折叠）"),
                        ("plan", "计划表预告（待创建/已删除表内，或 Status 仍为 TBD 的行；"
                                 "是预测不是指针）"),
                        ("artifact", "产物路径（gitignored，本机缺席属正常；按源文件折叠）"),
                        ("abs", "绝对路径链接（旧工具烘进去的 file:// 式路径；按源文件折叠）"),
                        ("example", "示例占位符（<>/glob 模板；无目录豁免）")):
        rows = buckets[name]
        print(f"\n--- {label}：{len(rows)} ---")
        if name == "real" or len(rows) <= 12:
            for s, ln, raw, kind in rows:
                print(f"  {s}:{ln}  [{kind}]  -> {raw}")
        else:
            counts: dict[str, int] = {}
            for s, *_ in rows:
                counts[s] = counts.get(s, 0) + 1
            for f, n in sorted(counts.items()):
                print(f"  {f}  ({n} 处)")

    # Anchor misses in this repo are near-uniformly CJK slug drift (a heading's
    # full-width （）： vanish under GitHub's rules, hand-written TOC links assume
    # they don't), and the docs are read in editors, not on GitHub. Kept behind a
    # flag so a real finding is never buried under thirteen cosmetic ones.
    print(f"\n--- 锚点未命中：{len(bad_anchor)}"
          + ("（--anchors 展开）" if bad_anchor and not args.anchors else ""))
    if args.anchors:
        for s, ln, raw in bad_anchor:
            print(f"  {s}:{ln}  -> {raw}")

    orphans = [rel(f) for f in files
               if rel(f).startswith(("docs/", "paper/")) and rel(f) not in cited]
    print(f"--- docs/ 与 paper/ 下无人指向：{len(orphans)}（逐轮 prompt / 复审存档"
          f"天然无入链，未必是问题）"
          + ("（--orphans 展开）" if orphans and not args.orphans else ""))
    if args.orphans:
        for o in orphans:
            print(f"  {o}")

    suspect = verify_declarations() if args.verify_declarations else 0

    print("\n⚠ 本脚本只验「目标是否存在」。指向是否**恰当**、deprecated 横幅与引用方是否"
          "一致、prose 里的版本号与日期声明是否仍然为真——机器验不了，是人的活。")
    return 1 if (args.strict and (buckets["real"] or suspect)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
