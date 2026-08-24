"""Check the status cells of markdown tables against what they claim to know.

Why this exists: a 2026-08-17 audit (`cce2b2d`) walked one 13-row routing table by hand
and found 3 rows whose 状态 cell disagreed with the banner of the document that row
cites. Nothing re-ran it. This makes that pass one command, and adds the second failure
mode the same audit's own ledger later exhibited: a status cell that names a plan
instead of stating a fact, and so never becomes false because the plan changed
underneath it.

Two rules, deliberately narrow. Scanning every ✅/⬜/⚠ in the repo was measured first and
rejected: 701 status cells, of which the overwhelming majority are honest dated
snapshots ("300 passed when the two batches closed"), and reporting those buries the
handful that are actually wrong.

  R1 交叉声称  In a table whose header names a status column, a row whose *subject* is a
              repo markdown document must not contradict that document's own banner.
              The banner is read by `build_doc_index.status_of` -- the same reading the
              generated index publishes, so the two can never drift apart.

  R2 预告占位  A table cell holding nothing but a forecast token (`TBD`, `第三批`, `待定`)
              is a plan standing where a state belongs. It was true when written and
              silently stops being true; unlike a wrong status, nobody ever has cause
              to revisit it.

What R1 deliberately does NOT do: compare a status cell to prose elsewhere in the same
document ("⚠ 判据待改，见 §四" when §三 records the disposition). That failure is real --
it is how this repo's own audit ledger drifted -- but it was measured on 2026-08-24 and
found unmechanisable here: 61 status cells carry a `§` pointer and all 61 resolve,
because most of them point at *thesis* sections that live in `.tex` files, not at
headings of the citing document. The rule stays a human criterion, written down in
`docs/handoff/2026-08-23-integrity-audit-pytest.md` §一.

Buckets and grading follow `check_doc_pointers`: only the defect buckets fail --strict.
A doc may freeze a table deliberately and say so in its own prose above it; that is the
`declared` bucket, and like the `自述缺席` bucket next door, **a false declaration
silences its own alarm permanently** -- the printed trigger line is what makes one
readable.

Usage:
    python -m scripts.check_status_claims            # report, always exit 0
    python -m scripts.check_status_claims --strict   # exit 1 if any defect bucket is non-empty
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from scripts.build_doc_index import (
    DATA_DIRS,
    SKIP_DIRS,
    STATUS_PATTERNS,
    head_of,
    status_of,
)

ROOT = Path(__file__).resolve().parents[1]

# The index is generated FROM the banners this checker compares against, so grading it
# would only ever restate `build_doc_index --check`.
GENERATED = {"docs/DOC_INDEX.md"}

# A status column is identified by its header, from a closed vocabulary. Detecting it
# from the cells instead (a column that is mostly ✅/❌) was tried and withdrawn: it
# picks up `持数字 ground truth？`, whose ❌ means "holds no numbers", not "dead doc".
STATUS_HEADER = re.compile(r"状态|\bstatus\b|\bstate\b", re.I)

# R2 grades a wider set of columns than R1 does, and the difference is deliberate. R1
# compares against a *document's* banner, so it needs a column that claims a document's
# state. R2 asks only "does this cell state where the thing stands", which is equally
# the job of a 本轮 / 处置 column -- and 本轮 is the header the one historical instance
# of this failure sat under (`⬜ 第三批`, in this repo's own audit ledger). Scoping is
# still by header, not by cell: a bare `TBD` under `Expected σ_final ↓` means the
# literature has no number, and under `Notebook` means the file is unnamed. Neither is a
# stale status, and both are in this repo today.
PROGRESS_HEADER = re.compile(r"状态|\bstatus\b|\bstate\b|本轮|本次|进度|处置|落地|\bprogress\b", re.I)

SEPARATOR = re.compile(r"^\|?[\s:|-]+\|[\s:|-]*$")
ORDINAL = re.compile(r"^[\d.#*\s]*$")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+?)(?:#[^)]*)?\)")
DATE = re.compile(r"(20\d{2}-\d{2}-\d{2})")

# Words by which a cell asserts the cited document is still in force. Needed because the
# absence of a banner keyword is not itself a contradiction -- "已被主线取代；保留作历史"
# says the same thing as DEPRECATED without using the word -- but "active" against a
# DEPRECATED banner is a straight disagreement.
ALIVE = re.compile(r"\bactive\b|\bcurrent\b|\blive\b|仍有效|仍然有效|现行|在用|活跃", re.I)

# Two conditions for "this row is about that document" rather than "this sentence
# mentions it". The cell must open with the link, and what remains after removing the
# link must be short. Measured on 2026-08-24 over the 20 rows this rule admits: all 20
# open with the link, and the largest residue is 19 characters -- `(rev.5, 2026-04-22)`,
# with the next at 7. The rows it must exclude open with prose instead ("六份文档头注仍写
# spec rev.4（…）", whose ✅ grades the fix, not any of the six documents).
SUBJECT_OPENER = re.compile(r"^[*`~\s]*\[")
SUBJECT_RESIDUE = 24

# A cell that is ONLY a forecast. `✅ 已并入第三批（db0c235，19 项用例）` states a fact and
# happens to name a batch; an unanchored match turns every honest record of which batch
# did the work into a finding, and the ledger this rule was written for holds six of
# them. Both ends are anchored and each end earns its keep separately: `.match` supplies
# the start (without it `✅ 已并入第三批` matches), the trailing `$` the end (without it
# `第三批已收尾（925d9c4）` matches). No leading `^` -- `.match` already means that, and a
# redundant guard is one no negative control can grade.
FORECAST_ONLY = re.compile(
    r"(TBD|TODO|待定|待补|计划中|待排期|planned|pending"
    r"|第[一二三四五六七八九十\d]+批|待第[一二三四五六七八九十\d]+批|下一批|后续批次)$",
    re.I)
CELL_NOISE = re.compile(r"[*`~\s]|⚠️|[⚠⬜✅❌⏸🟡🔴📦📍]")

# A doc may state that a table is frozen at its planning-time values on purpose. The
# declaration must sit between the table and the heading above it, so the reader meets
# it before the table it excuses.
FROZEN = re.compile(
    r"停在计划期|未随落地更新|不改本表任何一行|不改本表|只记录落地实况"
    r"|frozen at planning|left at planning-time")
HEADING = re.compile(r"^#{1,6}\s")


def md_files() -> list[Path]:
    out = []
    for p in sorted(ROOT.rglob("*.md")):
        if SKIP_DIRS & set(p.parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.split("/", 1)[0] in DATA_DIRS or rel in GENERATED:
            continue
        out.append(p)
    return out


def split_row(line: str) -> list[str]:
    """Split a table row on unescaped pipes that are not inside inline code."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells, cur, in_tick, i = [], [], False, 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            cur.append(s[i:i + 2])
            i += 2
            continue
        if c == "`":
            in_tick = not in_tick
        if c == "|" and not in_tick:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def tables(lines: list[str]):
    """Yield (header_lineno, header_cells, [(lineno, cells), ...]) for each table."""
    i = 0
    while i < len(lines):
        if (lines[i].lstrip().startswith("|") and i + 1 < len(lines)
                and SEPARATOR.match(lines[i + 1].strip())):
            header = split_row(lines[i])
            body, j = [], i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                body.append((j + 1, split_row(lines[j])))
                j += 1
            yield i + 1, header, body
            i = j
        else:
            i += 1


def status_column(header: list[str]) -> int | None:
    for k, cell in enumerate(header):
        if STATUS_HEADER.search(cell):
            return k
    return None


def progress_columns(header: list[str]) -> set[int]:
    return {k for k, cell in enumerate(header) if PROGRESS_HEADER.search(cell)}


def subject_column(cells: list[str]) -> int:
    for k, cell in enumerate(cells):
        if not ORDINAL.match(cell):
            return k
    return 0


def cited_doc(cell: str, src: Path, known: dict[str, str]) -> str | None:
    """The repo markdown doc this cell is *about*, or None if it is merely mentioning one."""
    hrefs = [h for h in LINK.findall(cell) if h.endswith(".md") and "://" not in h]
    if len(hrefs) != 1:
        return None
    if not SUBJECT_OPENER.match(cell) or len(LINK.sub("", cell).strip()) > SUBJECT_RESIDUE:
        return None  # a sentence that contains a link, not a row about that document
    try:
        rel = (src.parent / hrefs[0]).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None
    return rel if rel in known else None


def cell_labels(cell: str) -> set[str]:
    return {label for label, pat in STATUS_PATTERNS if re.search(pat, cell, re.I)}


def frozen_above(lines: list[str], table_lineno: int) -> str | None:
    """The doc's own declaration that this table is deliberately not maintained."""
    for i in range(table_lineno - 2, -1, -1):
        if HEADING.match(lines[i]):
            return None
        if FROZEN.search(lines[i]):
            return lines[i].strip()
    return None


def scan() -> dict[str, list[str]]:
    banners = {}
    for p in md_files():
        banners[p.relative_to(ROOT).as_posix()] = status_of(head_of(p))
    labelled = {rel: st for rel, st in banners.items() if st != "—"}

    buckets: dict[str, list[str]] = {k: [] for k in
                                     ("contradict", "date", "forecast",
                                      "silent", "declared", "agree")}
    for p in md_files():
        rel = p.relative_to(ROOT).as_posix()
        lines = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").split("\n")
        for table_lineno, header, body in tables(lines):
            k = status_column(header)
            graded = progress_columns(header)
            frozen = None
            for lineno, cells in body:
                # ---- R2: a forecast standing where a state belongs -------------------
                subj = subject_column(cells)
                for idx, cell in enumerate(cells):
                    if idx not in graded or not FORECAST_ONLY.match(CELL_NOISE.sub("", cell)):
                        continue
                    if frozen is None:
                        frozen = frozen_above(lines, table_lineno) or ""
                    where = f"{rel}:{lineno}  「{cell}」  ← {cells[subj][:60]}"
                    buckets["declared" if frozen else "forecast"].append(
                        f"{where}\n      本文声明：{frozen}" if frozen else where)

                # ---- R1: a status cell contradicting the doc it is about -------------
                if k is None or len(cells) <= k:
                    continue
                target = cited_doc(cells[subj], p, labelled)
                if target is None or target == rel:
                    continue
                cell = cells[k]
                banner = labelled[target].split()
                blabel, bdate = banner[0], (banner[1] if len(banner) > 1 else None)
                labels = cell_labels(cell)
                where = (f"{rel}:{lineno}  「{cell[:70]}」\n"
                         f"      {target} 自述 {labelled[target]}")
                if blabel in labels:
                    dates = set(DATE.findall(cell))
                    bucket = "date" if (bdate and dates and bdate not in dates) else "agree"
                elif labels or ALIVE.search(cell):
                    bucket = "contradict"
                else:
                    bucket = "silent"
                buckets[bucket].append(where)
    return buckets


# `★` marks a defect section. The PostToolUse hook greps for it to decide which part of
# this report to hand back, so the marker and the DEFECT tuple must stay in step.
DEFECT = ("contradict", "date", "forecast")
TITLES = {
    "contradict": "★ 状态栏与被引文档自述的横幅相左",
    "date": "★ 关键词一致但日期不一致",
    "forecast": "★ 状态栏里站着一个预告（批次名／计划名／TBD），不是既成事实",
    "silent": "被引文档自述了状态，引用它的状态栏未提（提示，不判失败）",
    "declared": "本文自述该表停在计划期，故意不维护（声明不是事实，见下）",
}


def main(argv: list[str] | None = None) -> int:
    # The PostToolUse hook captures stdout through a pipe, which on Windows defaults to
    # cp936 -- and `⚠` is not encodable there. Without this the sweep dies mid-report and
    # the hook reads the crash as a finding, blocking every markdown edit in the repo.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any defect bucket is non-empty")
    args = ap.parse_args(argv)

    buckets = scan()
    total = sum(len(v) for v in buckets.values())
    defects = sum(len(buckets[b]) for b in DEFECT)

    print(f"表格状态栏核查：受检 {total} 处 = 吻合 {len(buckets['agree'])} / "
          f"相左 {len(buckets['contradict'])} / 日期不符 {len(buckets['date'])} / "
          f"预告占位 {len(buckets['forecast'])} / 未提 {len(buckets['silent'])} / "
          f"自述冻结 {len(buckets['declared'])}")
    for name in ("contradict", "date", "forecast", "silent", "declared"):
        rows = buckets[name]
        if not rows:
            continue
        print(f"\n--- {TITLES[name]}：{len(rows)} ---")
        for row in rows:
            print(f"  {row}")
    if buckets["declared"]:
        print("\n  ⚠ 自述冻结桶装的是声明，不是事实。移动或更新那张表时要一并复核这条声明，"
              "否则它会永久静音自己的告警。")
    print()
    if defects and args.strict:
        print(f"FAIL: {defects} 处缺陷（--strict）")
        return 1
    print("OK: 无缺陷" if not defects else f"{defects} 处缺陷（未加 --strict，不判失败）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
