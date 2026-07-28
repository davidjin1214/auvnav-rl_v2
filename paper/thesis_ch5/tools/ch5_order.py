"""
paper/thesis_ch5/tools/ch5_order.py

图表编号与首引序对齐核查 (numbering vs first-citation order).

Float numbers follow \\caption order in the source; readers expect them to follow the
order in which the text first cites them. A mismatch means a float is numbered before
one that is cited earlier. Checked separately for 表 and 图.

This check found a defect on 2026-07-28 that four prior review rounds had missed:
tab:ch5_rebrac_perseed was first cited in §5.7.2 but placed in §5.7.3, so it was
numbered after tab:ch5_rebrac_beta2 while being cited before it. Fix was to move the
table block to just after its first citation -- which also cut its float distance from
4 pages to 1. Numbering and placement are the same problem seen from two sides.

Requires a current main.aux (run latexmk first, `latexmk -c` after).

Usage:
    python ch5_order.py             # exit 1 on any inversion
    python ch5_order.py --quiet
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys

from _ch5_corpus import CHAPTER_DIR, SECTION_NO, SECTION_ORDER, use_utf8_stdout

NEWLABEL = re.compile(r"\\newlabel\{((?:tab|fig):[^}]+)\}\{\{([0-9.]+)\}")
REF = re.compile(r"\\ref\{((?:tab|fig):[^}]+)\}")


def main() -> int:
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description="Chapter 5 float numbering vs first-citation order")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    aux_path = os.path.join(CHAPTER_DIR, "main.aux")
    if not os.path.exists(aux_path):
        print("main.aux 不存在——先跑 latexmk", file=sys.stderr)
        return 2
    with open(aux_path, encoding="utf-8") as fh:
        number = {m.group(1): m.group(2) for m in NEWLABEL.finditer(fh.read())}

    first: dict[str, tuple[int, str]] = {}
    seq = 0
    for stem in SECTION_ORDER:
        path = os.path.join(CHAPTER_DIR, "sections", stem + ".tex")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.lstrip().startswith("%"):
                    continue
                for m in REF.finditer(line):
                    seq += 1
                    first.setdefault(m.group(1), (seq, stem))

    inversions: list[tuple[str, str]] = []
    for kind, label in (("tab", "表"), ("fig", "图")):
        keys = sorted((k for k in number if k.startswith(kind + ":")),
                      key=lambda k: float(number[k].split(".", 1)[1]))
        if not args.quiet:
            print(f"\n== {label} ==  编号序 vs 首引序")
        highwater = -1
        for k in keys:
            hit = first.get(k)
            order = hit[0] if hit else 10 ** 9
            flag = ""
            if hit is None:
                flag = "  (正文无 \\ref)"
            elif order < highwater:
                flag = "  <== 逆序"
                inversions.append((label + number[k], k))
            highwater = max(highwater, order)
            if not args.quiet:
                where = SECTION_NO[hit[1]] if hit else "-"
                print(f"  {label}{number[k]:<6}{k:<34}首引序 {order if hit else '-':<8}§{where}{flag}")

    print()
    if inversions:
        print("FAIL 编号序与首引序逆序：" + "、".join(f"{n}（{k}）" for n, k in inversions))
        return 1
    print(f"PASS 编号序与首引序零逆序（{len(number)} 个浮动）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
