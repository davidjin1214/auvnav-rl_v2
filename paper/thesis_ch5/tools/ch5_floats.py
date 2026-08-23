"""
paper/thesis_ch5/tools/ch5_floats.py

浮动落位核查 (float placement) for Chapter 5.

For every float: the page its caption landed on (from main.aux \\newlabel) minus the
page where its number first appears in the text stream (from pdftotext). Criterion:
**distance <= 2 pages**.

Requires a current build -- run latexmk first, and do NOT run `latexmk -c` until
after this check (it deletes main.aux).

History worth knowing before you "fix" anything here (findings §12.4):
  - The 2026-07-08 claim「全部浮动距首引 0-2 页」was measured once and then quoted for
    three rounds; by 2026-07-28 it was wrong in 4 places. **Layout claims expire on any
    text edit.** Re-run this, never quote a past result.
  - The failure mode is a QUEUE BACKLOG, not a per-float placement option: captions in
    this chapter run 200-290 net chars, so table+caption exceeds the LaTeX default
    \\topfraction=0.7 and defers wholesale. Per-float [t]->[tp] tweaks cannot clear a
    queue; the fix that worked is the float parameter block in main.tex.
  - Do not buy layout by shortening a caption -- most of the long captions are 口径隔离
    声明 patching a CRITICAL finding (spec §0.5.9 (d)).
  - The first-citation scan was a substring test until 2026-08-23, so any float whose
    number prefixes a longer one (表 5.1 vs 表 5.10) could be measured against the
    wrong page. Re-measure before quoting a distance recorded before that date.

Usage:
    python ch5_floats.py            # table of all floats, exit 1 if any exceeds 2
    python ch5_floats.py --quiet    # only the verdict line
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

from _ch5_corpus import CHAPTER_DIR, use_utf8_stdout

MAX_DISTANCE = 2

NEWLABEL = re.compile(r"\\newlabel\{((?:tab|fig):[^}]+)\}\{\{([0-9.]+)\}\{(\d+)\}")


def read_pages() -> list[str]:
    pdf = os.path.join(CHAPTER_DIR, "main.pdf")
    out = subprocess.run(["pdftotext", "-enc", "UTF-8", pdf, "-"],
                         capture_output=True, check=True).stdout
    return out.decode("utf-8", "replace").split("\f")


def main() -> int:
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description="Chapter 5 float placement check")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    aux_path = os.path.join(CHAPTER_DIR, "main.aux")
    if not os.path.exists(aux_path):
        print("main.aux 不存在——先跑 latexmk（且在本检查之后再 latexmk -c）", file=sys.stderr)
        return 2

    with open(aux_path, encoding="utf-8") as fh:
        floats = {m.group(1): (m.group(2), int(m.group(3)))
                  for m in NEWLABEL.finditer(fh.read())}
    pages = read_pages()

    rows = []
    for key, (num, caption_page) in floats.items():
        kind = "表" if key.startswith("tab:") else "图"
        # 表 5.1 is a prefix of 表 5.10..5.19, and 表 5.2 of 表 5.20/5.21. A substring
        # test dates 表 5.1's first citation to whichever of those pages comes first,
        # and min() can only move it earlier -- inflating the distance into a false
        # violation, or hiding a real one when the caption precedes the true citation.
        cite = re.compile(rf"{kind} ?{re.escape(num)}(?![0-9])")
        hits = sorted({i for i, page in enumerate(pages, 1) if cite.search(page)})
        first = min(hits) if hits else None
        dist = abs(caption_page - first) if first else None
        rows.append((kind, num, key, caption_page, first, dist))

    rows.sort(key=lambda r: (r[0], [int(x) for x in r[1].split(".")]))
    bad = [r for r in rows if r[5] is not None and r[5] > MAX_DISTANCE]

    if not args.quiet:
        print(f"PDF {len(pages)} 页；浮动 {len(rows)} 个；判据 距首引 <= {MAX_DISTANCE} 页")
        print(f"{'浮动':<9}{'label':<34}{'落位':>5}{'首引':>5}{'距离':>5}")
        for kind, num, key, cp, first, dist in rows:
            flag = "  <== 越线" if dist is not None and dist > MAX_DISTANCE else ""
            shown = dist if dist is not None else -1
            print(f"{kind + num:<9}{key:<34}{cp:>5}{first if first else 0:>5}{shown:>5}{flag}")
        print()

    if bad:
        print(f"FAIL 浮动距首引 > {MAX_DISTANCE} 页：" +
              "、".join(f"{k}{n}（距 {d}）" for k, n, _, _, _, d in bad))
        return 1
    print(f"PASS 浮动距首引 <= {MAX_DISTANCE} 页：{len(rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
