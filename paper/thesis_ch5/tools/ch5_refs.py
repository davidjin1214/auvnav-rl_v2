"""
paper/thesis_ch5/tools/ch5_refs.py

跨节引用抽核 (cross-reference audit) for Chapter 5.

`latexmk` reporting 0 undefined references only proves every label EXISTS. It cannot
tell you a \\S\\ref points at the wrong subsection. This script resolves every
sec:/subsec: reference to its actual number and its actual heading text, and prints
the preceding clause of each citing sentence, so the pairing can be read off directly.

Also reports labels that are never referenced (usually fine -- section anchors) and
labels referenced but missing a heading (a real defect).

Requires a current main.aux (run latexmk first, `latexmk -c` after).

Usage:
    python ch5_refs.py            # full audit table
    python ch5_refs.py --quiet    # verdict + anomalies only
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys

from _ch5_corpus import CHAPTER_DIR, SECTION_NO, SECTION_ORDER, use_utf8_stdout

HEADING = re.compile(r"\\(sub)?section\{(.*?)\}\s*\\label\{([^}]+)\}")
REF = re.compile(r"\\ref\{((?:subsec|sec):[^}]+)\}")
NEWLABEL = re.compile(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}")


def main() -> int:
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description="Chapter 5 cross-reference audit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    aux_path = os.path.join(CHAPTER_DIR, "main.aux")
    if not os.path.exists(aux_path):
        print("main.aux 不存在——先跑 latexmk", file=sys.stderr)
        return 2
    with open(aux_path, encoding="utf-8") as fh:
        number = {m.group(1): m.group(2) for m in NEWLABEL.finditer(fh.read())}

    title: dict[str, str] = {}
    cites: dict[str, list] = collections.defaultdict(list)
    for stem in SECTION_ORDER:
        path = os.path.join(CHAPTER_DIR, "sections", stem + ".tex")
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if line.lstrip().startswith("%"):
                    continue
                m = HEADING.search(line)
                if m:
                    title[m.group(3)] = m.group(2)
                for r in REF.finditer(line):
                    cites[r.group(1)].append((stem, lineno, line.rstrip(), r.start()))

    def sort_key(label: str):
        try:
            return [int(p) for p in number.get(label, "99").split(".")]
        except ValueError:
            return [99]

    missing = [k for k in cites if k not in title]
    if not args.quiet:
        print("交叉引用逐条核对：label -> 实际编号 / 实际标题 / 各引用处前文语境")
        print("=" * 100)
        for key in sorted(cites, key=sort_key):
            heading = title.get(key, "?? 无标题")
            print(f"\n[{number.get(key, '??')}] {key}  <{heading}>  引用 {len(cites[key])} 处")
            for stem, lineno, line, start in cites[key]:
                print(f"   <- §{SECTION_NO[stem]:<5}{stem}:{lineno}  ...{line[max(0, start - 74):start].strip()}")

    unused = sorted(k for k in title if k not in cites)
    print()
    print(f"被引用的 sec/subsec label {len(cites)} 个；从未被引用 {len(unused)} 个：{unused}")
    if missing:
        print(f"FAIL 引用了但无对应标题的 label：{missing}")
        return 1
    print(f"PASS 全部 {len(cites)} 个交叉引用解析到真实小节标题"
          f"（⚠ 指向是否恰当仍须人读上表，机器只能验存在性）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
