"""Repo-wide sweep for doc references that point *into* source files.

`check_doc_pointers` resolves whether a cited path exists. It says so itself, in
`resolve()`: a trailing `:N` is stripped before the check, because the pointer is the
file. That leaves two reference forms unverified, and both have already rotted here:

  line anchors    `scripts/collect_offline_data.py:315` -> `episode_seed = base_seed + ep`
                  Inserting seven lines above it (5f8228c, a parallel-worker fallback)
                  moved that statement to 316. Three docs still cite 315, one of them
                  `data_integrity_open_items.md`, the ledger for the seed-overlap
                  finding. The path still resolves, so nothing complained.

  symbol claims   CLAUDE.md routed readers to `get_probe_positions()` in
                  `auv_nav/flow.py`. That name has never existed anywhere in the repo
                  -- the real one is `make_probe_offsets()`. The file resolved; the
                  function was invented (found 2026-08-17, a49fb1e).

Both classes were last checked by reading the lines by hand (2a8c311 verified five of
them in one report). This makes that mechanical.

What is reported, and why each bucket is separate:

  out-of-range    the file has fewer lines than the citation claims. Unambiguous.
  blank-line      the cited line exists but is empty. A citation that lands on
                  whitespace is telling the reader nothing; `docs/online_rl_thesis_plan.md`
                  cites `auv_nav/env.py:895` for a definition and gets a blank line.
  drift           the citing sentence quotes a code fragment, and none of that
                  fragment's distinctive identifiers appear on the cited line. The
                  report names the nearest line that does carry them, so the fix is
                  mechanical rather than another hand-read.
  symbol-missing  a `name()` written on a line that also names a repo `.py` file, with
                  no `def name` / `class name` anywhere in the repo. See below for why
                  the test is repo-wide rather than per-file.
  ambiguous       a bare basename (`fql.py:444`) matching more than one file. Reported,
                  not failed: which one was meant is a human call.

Why the symbol test is repo-wide. The first cut read "`Foo()` on the same line as
`bar.py`" as the claim "Foo is defined in bar.py". Measured against this repo, every
such hit was a false positive: `FQLAgent.update()` next to `train_offline.py` says the
metrics dict is compatible with that file's logger, and `PlanarRemusEnv.compute_...()`
next to `vehicle.py` says swapping that file changes nothing. Co-occurrence is a topic,
not an attribution, and prose carries no reliable attribution marker. So the `.py` path
survives only as a *gate* -- it says the sentence is talking about this codebase -- and
what gets tested is whether the name exists at all.

What this canNOT check, as a direct consequence: a function that *moved*. If the name is
defined anywhere in the repo, the citation passes even when the doc sends the reader to
the wrong file. Nor whether the cited line is the *right* line for the claim being made;
a fragment-free citation is only checked for existence and non-blankness.

Usage:
    python -m scripts.check_doc_code_refs           # triaged report
    python -m scripts.check_doc_code_refs --strict  # exit 1 on any defect bucket
    python -m scripts.check_doc_code_refs --all     # also list clean references
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# The one closed list of directories that are not project content -- imported rather
# than restated, because a second copy is what drifts (agent worktrees are full second
# checkouts and doubled this repo's corpus once already, 2d4ba97).
from scripts.check_doc_pointers import SKIP_DIRS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# `path.py:315`, `fql.py:472-474`, `fql.py:472–474` (en dash: docs are written in
# Chinese and the editor substitutes it). Backticks optional -- half these citations
# are plain prose.
LINE_REF = re.compile(
    r"`?((?:[A-Za-z0-9_.\-]+/)*[A-Za-z0-9_.\-]+"
    r"\.(?:py|sh|tex|md|json|jsonl|ipynb))"
    r":(\d+)(?:\s*[-–]\s*(\d+))?`?"
)

# A backticked `name()` or `mod.name()`. The trailing `()` is what marks it as a claim
# about a definition rather than a passing mention of a variable.
SYMBOL = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\(\s*\)`")

CODE_SPAN = re.compile(r"`([^`\n]+)`")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# A backticked span that is itself a path is not a quoted fragment of the target. Docs
# here routinely cite five sections on one line (`setup.tex:215`, `rebrac.tex:451`, ...),
# and reading the other four filenames as "the fragment that should appear at line 451"
# turned every such line into a false drift -- 20 of the first run's 34.
LOOKS_LIKE_PATH = re.compile(r"/|\.(?:py|sh|tex|md|json|jsonl|ipynb|npz|csv|bib)\b")

# Identifiers too common to distinguish one line from its neighbours. Keeping this
# short on purpose: a token wrongly kept costs a false `drift` that a human dismisses,
# a token wrongly dropped costs a missed rot that nobody sees.
STOPWORDS = {
    "def", "class", "self", "return", "import", "from", "for", "and", "not", "the",
    "int", "str", "float", "bool", "none", "true", "false", "list", "dict", "set",
    "value", "data", "path", "file", "line", "name", "type", "args", "kwargs",
}

# Prefixes belonging to third-party or stdlib namespaces: `np.std()` on a line that
# also cites a repo file is a claim about numpy, not about that file.
# A plan or an archived record may cite a name the code never carried, or carried once.
# Correcting those would rewrite what the line planned or what the reviewer read -- the
# same reason `check_doc_pointers` lets a doc declare an absent path rather than delete
# the link. So the citing sentence says so, on the line the reader sees, and this reads
# that sentence. Closed list, extend only deliberately; a declaration that does not name
# the real thing is just a way to switch the alarm off.
HISTORICAL = re.compile(
    r"当时的行号|当时的名字|当时叫|原计划名|原计划文件名|落地时定名|后改名为|实现时改名"
    r"|renamed|shipped as|as built", re.I)

FOREIGN_PREFIX = re.compile(
    r"^(np|numpy|torch|nn|F|pd|pandas|plt|os|sys|re|json|math|random|gym|gymnasium)\.")


def md_files(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        out += [os.path.join(dirpath, f) for f in filenames if f.endswith(".md")]
    return sorted(out)


def _source_index(root: str) -> dict[str, list[str]]:
    """basename -> every repo-relative source path carrying it."""
    index: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if not f.endswith((".py", ".sh", ".tex", ".md", ".json", ".jsonl", ".ipynb")):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), root).replace("\\", "/")
            index.setdefault(f, []).append(rel)
    return {k: sorted(v) for k, v in index.items()}


def resolve_ref(raw: str, src: str, root: str, index: dict[str, list[str]]) -> list[str]:
    """Repo-relative candidates for one cited path. Empty means nothing matched."""
    stem = raw.lstrip("./")
    direct = os.path.normpath(os.path.join(root, stem))
    if os.path.isfile(direct):
        return [os.path.relpath(direct, root).replace("\\", "/")]
    # Relative to the citing file, then to each ancestor -- `figures/scripts/x.py`
    # written inside paper/thesis_ch5/ means that directory's copy.
    probe = os.path.dirname(src)
    while len(probe) >= len(root):
        cand = os.path.normpath(os.path.join(probe, stem))
        if os.path.isfile(cand):
            return [os.path.relpath(cand, root).replace("\\", "/")]
        probe = os.path.dirname(probe)
    if "/" not in stem:
        cands = index.get(stem, [])
    else:
        tail = "/" + stem
        cands = [p for p in sum(index.values(), []) if p.endswith(tail)]
    return _prefer_nearest(cands, src, root)


def _prefer_nearest(cands: list[str], src: str, root: str) -> list[str]:
    """Collapse same-basename candidates using the citing file's own subtree.

    `setup.tex` exists twice -- once under the live chapter, once under the archived
    standalone paper. A note inside `paper/thesis_ch5/notes/` means its own chapter's
    copy, and treating that as undecidable buried 10 real citations in an `ambiguous`
    list nobody would read. Only a unique winner collapses; a genuine tie stays reported.
    """
    if len(cands) < 2:
        return cands
    srel = os.path.relpath(src, root).replace("\\", "/").split("/")
    scored = [(_shared_prefix(srel, c.split("/")), c) for c in cands]
    best = max(n for n, _ in scored)
    winners = [c for n, c in scored if n == best]
    return winners if len(winners) == 1 else sorted(cands)


def _shared_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def read_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().split("\n")
    except OSError:
        return []


# How far from the citation a backticked span may sit and still count as its quoted
# fragment. Findings tables here run past a thousand characters per row: one row cited
# `summarize_offline_phase0.py:241` at one end and named a figure `algo_interaction` at
# the other, and whole-line attribution read the figure name as code that should appear
# at line 241.
SPAN_PROXIMITY = 160


def distinctive_tokens(md_line: str, at: int = -1) -> set[str]:
    """Identifiers quoted next to the citation that are not themselves paths.

    `at` is the character offset of the citation; spans further than SPAN_PROXIMITY
    from it are somebody else's fragment. Pass -1 to consider the whole line.
    """
    stems = {os.path.basename(m.group(1)).split(".")[0]
             for m in LINE_REF.finditer(md_line)}
    stems |= {os.path.basename(m.group(1)).split(".")[0]
              for m in BARE_PY.finditer(md_line)}
    tokens: set[str] = set()
    for m in CODE_SPAN.finditer(md_line):
        if at >= 0 and min(abs(m.start() - at), abs(m.end() - at)) > SPAN_PROXIMITY:
            continue
        span = m.group(1)
        if LOOKS_LIKE_PATH.search(span):
            continue
        for tok in IDENT.findall(span):
            if tok.lower() in STOPWORDS or tok in stems:
                continue
            tokens.add(tok)
    return tokens


DEFINITION = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


def definition_index(root: str) -> dict[str, list[str]]:
    """Every `def`/`class` name in the repo -> the `.py` files defining it.

    Methods land here alongside module-level names, which is what makes the dotted case
    below decidable: `FQLAgent.update()` is cleared by *some* `def update`, because the
    only thing this tool can honestly test about it is that the method exists. Fixtures
    under `tests/` count too -- "exists in the repo" is the claim being tested, and
    carving out directories would be a second closed list to keep in step with SKIP_DIRS.
    """
    out: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), root).replace("\\", "/")
            for ln in read_lines(os.path.join(dirpath, f)):
                m = DEFINITION.match(ln)
                if m and rel not in out.setdefault(m.group(1), []):
                    out[m.group(1)].append(rel)
    return {k: sorted(v) for k, v in out.items()}


def judged_name(symbol: str, defs: dict[str, list[str]],
                modules: set[str]) -> str | None:
    """Which part of `symbol` this tool is entitled to test, or None to skip.

    A bare name tests itself. A dotted `A.b` tests the leaf `b`, but only once `A` is
    known to be ours -- a class/def by that name, or a module `A.py`. When the head is
    unrecognised the whole thing is somebody else's namespace or an instance variable,
    and reporting the leaf would be guessing; FOREIGN_PREFIX catches only the handful of
    third-party roots common enough to be worth naming.
    """
    if "." not in symbol:
        return symbol
    head, leaf = symbol.split(".", 1)
    if "." in leaf:  # `a.b.c()` -- more nesting than this tool can attribute
        return None
    if head in defs or f"{head}.py" in modules:
        return leaf
    return None


def scan(root: str, near_window: int = 2) -> dict[str, list[tuple]]:
    index = _source_index(root)
    defs = definition_index(root)
    modules = {k for k in index if k.endswith(".py")}
    source_cache: dict[str, list[str]] = {}
    buckets: dict[str, list[tuple]] = {
        "out-of-range": [], "blank-line": [], "drift": [], "drift-near": [],
        "symbol-missing": [], "ambiguous": [],
        "declared": [], "unresolved": [], "clean": [],
    }

    def lines_of(relpath: str) -> list[str]:
        if relpath not in source_cache:
            source_cache[relpath] = read_lines(os.path.join(root, relpath))
        return source_cache[relpath]

    for src in md_files(root):
        srel = os.path.relpath(src, root).replace("\\", "/")
        for lineno, md_line in enumerate(read_lines(src), 1):
            seen_paths: set[str] = set()
            # A quoted fragment can only be attributed when the sentence cites one file.
            # The impact-assessment review cites six sections on a single line; asking
            # which of them the neighbouring `alpha_0p05` belongs to has no answer.
            distinct = {c for m in LINE_REF.finditer(md_line)
                        for c in resolve_ref(m.group(1), src, root, index)}
            attributable = len(distinct) == 1

            for m in LINE_REF.finditer(md_line):
                raw, start_s, end_s = m.group(1), m.group(2), m.group(3)
                start = int(start_s)
                end = int(end_s) if end_s else start
                cands = resolve_ref(raw, src, root, index)
                seen_paths.update(cands)
                where = f"{srel}:{lineno}"
                if not cands:
                    # A path that does not resolve at all is check_doc_pointers' bucket,
                    # not this one; recording it keeps the two reports reconcilable.
                    buckets["unresolved"].append((where, raw, start))
                    continue
                if len(cands) > 1:
                    buckets["ambiguous"].append((where, raw, start, cands))
                    continue
                target = cands[0]
                lines = lines_of(target)
                if start > len(lines) or end > len(lines):
                    buckets["out-of-range"].append((where, raw, start, len(lines)))
                    continue
                cited = lines[start - 1:end]
                is_code = target.endswith((".py", ".sh"))
                if is_code and not any(ln.strip() for ln in cited):
                    if HISTORICAL.search(md_line):
                        buckets["declared"].append((where, raw, start, target))
                        continue
                    buckets["blank-line"].append((where, raw, start, target))
                    continue
                # Line anchors into prose are checked for existence only. A `.tex`/`.md`
                # anchor in a review note means "the passage around here" -- the reviewer
                # wrote it against a draft that has been edited above the anchor since,
                # so it lands a few lines off, blank lines included, and correcting the
                # number would falsify what the record says was read. Out-of-range still
                # applies: that one is unambiguous whatever the target is.
                if not is_code or not attributable:
                    buckets["clean"].append((where, raw, start, target, "not-attributable"))
                    continue
                tokens = distinctive_tokens(md_line, m.start())
                if not tokens:
                    buckets["clean"].append((where, raw, start, target, "no-fragment"))
                    continue
                if any(tok in ln for tok in tokens for ln in cited):
                    buckets["clean"].append((where, raw, start, target, "anchored"))
                    continue
                near = _nearest(lines, tokens, start)
                # Off by a line or two is how citations are honestly written: a doc may
                # cite the line an effect lands on while quoting the condition above it
                # (`environment_design.md` -> `env.py:377`, quoting the test on 376).
                # Rot is when the fragment is far away or gone. Both are printed; only
                # the latter fails --strict, so a hook stays quiet about cosmetics.
                bucket = ("drift-near"
                          if near is not None and abs(near - start) <= near_window
                          else "drift")
                buckets[bucket].append(
                    (where, raw, start, target, sorted(tokens), near))

            for m in SYMBOL.finditer(md_line):
                symbol = m.group(1)
                if FOREIGN_PREFIX.match(symbol):
                    continue
                # The gate, not an attribution: a `name()` on a line that names no repo
                # source file is as likely to be a shell builtin or a cited paper's
                # notation as it is to be ours.
                context = sorted(seen_paths | set(_bare_py_paths(md_line, src, root, index)))
                context = [p for p in context if p.endswith(".py")]
                if not context:
                    continue
                name = judged_name(symbol, defs, modules)
                if name is None:
                    continue
                where = f"{srel}:{lineno}"
                if name in defs:
                    buckets["clean"].append((where, symbol, 0, defs[name][0], "symbol"))
                    continue
                if HISTORICAL.search(md_line):
                    buckets["declared"].append((where, symbol, 0, context[0]))
                    continue
                buckets["symbol-missing"].append((where, symbol, name, context))

    return buckets


BARE_PY = re.compile(r"`?((?:[A-Za-z0-9_.\-]+/)*[A-Za-z0-9_\-]+\.py)`?")


def _bare_py_paths(md_line: str, src: str, root: str,
                   index: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for m in BARE_PY.finditer(md_line):
        out += resolve_ref(m.group(1), src, root, index)
    return sorted(set(out))


def _nearest(lines: list[str], tokens: set[str], start: int) -> int | None:
    """Where the quoted fragment actually is: most tokens matched, then closest.

    Nearest-by-any-token is not good enough, and the failure is not hypothetical. A
    table row quoting `env_step`, `num_envs`, `range` and `start_step` cited
    `train_sac.py:467`; the loop it describes is on 480, but `num_envs` alone happens to
    appear on 466, so "any token" answered 466 and a 13-line drift was filed as a
    one-line offset -- graded down out of `--strict` and never seen again.
    """
    best: int | None = None
    best_score = 0
    for i, ln in enumerate(lines, 1):
        score = sum(1 for tok in tokens if tok in ln)
        if score == 0:
            continue
        if score > best_score or (score == best_score and best is not None
                                  and abs(i - start) < abs(best - start)):
            best, best_score = i, score
    return best


DEFECT_BUCKETS = ("out-of-range", "blank-line", "drift", "symbol-missing")

LABELS = {
    "out-of-range": "★ 行号越界（文件没有那么多行）",
    "blank-line": "★ 引到空行（目标行存在但是空的）",
    "drift": "★ 行号漂移（同句引用的代码片段远在别处，或已不存在）",
    "symbol-missing": "★ 函数名全仓不存在（同句提到了仓内 .py，但这个名字没有任何定义）",
    "drift-near": "行号偏移（片段就在邻近几行；引用方式使然，不判缺陷）",
    "declared": "自述历史引用（引用方在同一句里写明这是当时的名字/行号，并给出真名）",
    "ambiguous": "同名多份（裸文件名匹配到多个路径；哪一份是人的活）",
    "unresolved": "路径本身解析不了（属 check_doc_pointers 的桶，此处仅登记）",
}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description="verify doc references that point into source files")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any defect bucket is non-empty")
    ap.add_argument("--all", action="store_true", help="also list clean references")
    ap.add_argument("--root", default=ROOT, help="repo root to scan")
    ap.add_argument("--near-window", type=int, default=2, metavar="N",
                    help="a fragment found within N lines of the citation is reported "
                         "as an offset rather than as rot (default 2)")
    args = ap.parse_args()

    buckets = scan(os.path.abspath(args.root), near_window=args.near_window)
    total = sum(len(v) for v in buckets.values())
    defects = sum(len(buckets[b]) for b in DEFECT_BUCKETS)
    print(f"解析到 {total} 处指向源码内部的引用"
          f"（行锚 + 函数名）；缺陷 {defects}，正常 {len(buckets['clean'])}")
    print("=" * 96)

    for name in ("out-of-range", "blank-line", "drift", "symbol-missing",
                 "drift-near", "declared", "ambiguous", "unresolved"):
        rows = buckets[name]
        print(f"\n--- {LABELS[name]}：{len(rows)} ---")
        if name == "declared":
            print("    接受的措辞，闭列表（必须同时写出真名/真行号，否则只是消音）：")
            print("      当时的名字 / 当时叫 / 当时的行号     引用的是彼时的状态")
            print("      原计划名 / 落地时定名 / 后改名为     计划与落地不同名")
        for row in rows:
            if name in ("drift", "drift-near"):
                where, raw, start, target, tokens, near = row
                near_s = f"最近命中 {target}:{near}" if near else "全文件均无该片段"
                print(f"  {where}  -> {raw}:{start}")
                print(f"        片段 {', '.join(tokens)} ｜ {near_s}")
            elif name == "blank-line":
                where, raw, start, target = row
                print(f"  {where}  -> {raw}:{start}  （{target} 第 {start} 行为空）")
            elif name == "out-of-range":
                where, raw, start, n = row
                print(f"  {where}  -> {raw}:{start}  （该文件共 {n} 行）")
            elif name == "symbol-missing":
                where, symbol, judged, context = row
                tail = f"｜实测的名字 {judged}" if judged != symbol else ""
                print(f"  {where}  -> {symbol}() ｜同句提到 {', '.join(context)} {tail}")
            elif name == "declared":
                where, what, start, target = row
                anchor = f"{what}:{start}" if start else f"{what}()"
                print(f"  {where}  -> {anchor}  （自述为历史引用；真名见同句）")
            elif name == "ambiguous":
                where, raw, start, cands = row
                print(f"  {where}  -> {raw}:{start}  候选 {', '.join(cands)}")
            else:
                where, raw, start = row
                print(f"  {where}  -> {raw}:{start}")

    if args.all:
        print(f"\n--- 正常：{len(buckets['clean'])} ---")
        for where, raw, start, target, how in buckets["clean"]:
            print(f"  {where}  -> {raw}:{start}  [{how}]  {target}")

    print("\n⚠ 本脚本只验「被引的行/符号是否还在那里」。那一行是否**恰当**支撑该句论断，"
          "机器验不了，是人的活。")
    return 1 if (args.strict and defects) else 0


if __name__ == "__main__":
    raise SystemExit(main())
