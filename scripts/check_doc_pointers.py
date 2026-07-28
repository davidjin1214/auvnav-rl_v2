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
  artifact  under a gitignored产物 dir; absent on this machine by design
  example   placeholder inside an agent/skill definition or a spec template
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
    python -m scripts.check_doc_pointers --strict   # exit 1 on any `real` miss
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DIRS = {".git", "node_modules", ".pytest_cache", ".pytest_tmp", "__pycache__",
             "wake_data", "offline_data", "checkpoints", ".ipynb_checkpoints"}

INLINE = re.compile(r"\[([^\]\[]*)\]\(([^)\s]+?)(?:\s+\"[^\"]*\")?\)")
REFDEF = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*(\S+)", re.M)
BARE = re.compile(
    r"`?((?:docs|paper|scripts|auv_nav|notebooks|benchmarks|experiments|figures|results|"
    r"offline_data|\.claude)/[A-Za-z0-9_./\-]+\.(?:md|tex|py|ipynb|json|sh|npz|csv|bib))`?"
)
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)

ARTIFACT_DIR = re.compile(
    r"(?:^|/)(results|offline_data|wake_data|checkpoints|figures)/")
EXAMPLE_SRC = (".claude/agents/", ".claude/skills/")
PLACEHOLDER = re.compile(r"(foo|bar|baz|<[a-zA-Z]|\.\.\.|^path$|^archive/$)")


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


def triage(src: str, raw: str) -> str:
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        return "abs"
    if src.startswith(EXAMPLE_SRC) or PLACEHOLDER.search(raw):
        return "example"
    if ARTIFACT_DIR.search("/" + raw):
        return "artifact"
    return "real"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="repo-wide markdown pointer sweep")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any `real` miss is found")
    args = ap.parse_args()

    files = md_files()
    anchor_cache: dict[str, set[str]] = {}
    buckets: dict[str, list] = {"real": [], "artifact": [], "example": [], "abs": []}
    bad_anchor: list[tuple[str, int, str]] = []
    cited: set[str] = set()
    total = 0

    for src in files:
        with open(src, encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        for lineno, line in enumerate(lines, 1):
            found = ([(m.group(2), "link") for m in INLINE.finditer(line)]
                     + [(m.group(2), "refdef") for m in REFDEF.finditer(line)]
                     + [(m.group(1), "bare") for m in BARE.finditer(line)])
            for raw, kind in found:
                if re.match(r"^(https?|mailto|ftp):", raw):
                    continue
                total += 1
                target = (src if raw.startswith("#") else resolve(raw, kind, src))
                if not os.path.exists(target):
                    buckets[triage(rel(src), raw)].append((rel(src), lineno, raw, kind))
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
    print(f"未解析 {miss} = 真失效 {len(buckets['real'])} / 产物路径 "
          f"{len(buckets['artifact'])} / 示例占位 {len(buckets['example'])} / "
          f"绝对路径 {len(buckets['abs'])}")
    print("=" * 96)
    for name, label in (("real", "★ 真失效（文档指向本应存在的东西）"),
                        ("artifact", "产物路径（gitignored，本机缺席属正常；按源文件折叠）"),
                        ("abs", "绝对路径链接（旧工具烘进去的 file:// 式路径；按源文件折叠）"),
                        ("example", "示例占位符（agent/skill 定义或模板内的假路径）")):
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

    print(f"\n--- 锚点未命中：{len(bad_anchor)}（目标文件在，#anchor 无对应标题；"
          f"CJK 全角标点导致的偏差多为观感问题）---")
    for s, ln, raw in bad_anchor:
        print(f"  {s}:{ln}  -> {raw}")

    orphans = [rel(f) for f in files
               if rel(f).startswith(("docs/", "paper/")) and rel(f) not in cited]
    print(f"\n--- docs/ 与 paper/ 下无人指向：{len(orphans)}（未必是问题，"
          f"逐轮 prompt / 复审存档天然无入链）---")
    for o in orphans:
        print(f"  {o}")

    print("\n⚠ 本脚本只验「目标是否存在」。指向是否**恰当**、deprecated 横幅与引用方是否"
          "一致、prose 里的版本号与日期声明是否仍然为真——机器验不了，是人的活。")
    return 1 if (args.strict and buckets["real"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
