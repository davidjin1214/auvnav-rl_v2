"""
paper/thesis_ch5/tools/ch5_check_all.py

送审门槛一键复核 (submission-threshold gate) for Chapter 5.

Runs the mechanical checks and prints a single PASS/FAIL table. Intended as the
last step of any round that touched the chapter, and as the thing to re-run before
quoting any layout or register number -- **layout claims expire on any text edit**
(the 2026-07-08 claim survived three rounds and was wrong in four places by
2026-07-28; see findings §12.4).

Checked here:
  compile      main.log: undefined refs / undefined citations / multiply-defined
               / Overfull > 10pt; main.blg: bibtex warnings; page count
  floats       every float within 2 pages of first citation      (ch5_floats.py)
  order        float numbering follows first-citation order       (ch5_order.py)
  lexicon      禁用词/术语命中与已裁定基线一致                     (ch5_lexcheck.py)
  refs         every sec/subsec \\ref resolves to a real heading   (ch5_refs.py)

NOT checked here, and not checkable by machine -- these stay human work:
  - whether a \\ref points at the *right* subsection (ch5_refs.py prints the pairing)
  - whether a banned-word hit is semantically compliant (spec §0.5.11 adjudication)
  - 数字忠实度 against ground-truth docs
  - 反向 churn: that no 口径声明 / 锁定措辞 / scope 句 / 种子与配置限定 was weakened

Prerequisite: a current build. Run
    latexmk -pdf -xelatex -interaction=nonstopmode main.tex
first, and `latexmk -c` only AFTER this script (it needs main.aux).

Usage:
    python ch5_check_all.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

from _ch5_corpus import CHAPTER_DIR, use_utf8_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
OVERFULL_LIMIT_PT = 10.0


def compile_gate() -> tuple[bool, list[str]]:
    log_path = os.path.join(CHAPTER_DIR, "main.log")
    blg_path = os.path.join(CHAPTER_DIR, "main.blg")
    if not os.path.exists(log_path):
        return False, ["main.log 不存在——先跑 latexmk"]
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        log = fh.read()

    notes, ok = [], True
    for label, pattern in (("undefined reference/citation", r"undefined"),
                           ("multiply-defined", r"multiply")):
        n = len(re.findall(pattern, log, re.I))
        notes.append(f"{label}: {n}")
        ok &= n == 0

    over = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log)]
    big = [x for x in over if x > OVERFULL_LIMIT_PT]
    notes.append(f"Overfull: {len(over)} 处"
                 + (f"（{', '.join(f'{x:.2f}pt' for x in over)}）" if over else "")
                 + (f"，其中 >{OVERFULL_LIMIT_PT:g}pt 者 {len(big)}" if big else ""))
    ok &= not big

    if os.path.exists(blg_path):
        with open(blg_path, encoding="utf-8", errors="replace") as fh:
            blg = fh.read()
        # `warning$ -- 0` is a bst function-call counter line, not a warning
        warns = [l for l in blg.splitlines()
                 if "warning" in l.lower() and not re.match(r"\s*warning\$", l)]
        notes.append(f"bibtex warning: {len(warns)}")
        ok &= not warns

    pages = re.findall(r"Output written on .*?\((\d+) pages", log)
    notes.append(f"页数: {pages[-1] if pages else '?'}")
    return ok, notes


def run(script: str) -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, os.path.join(HERE, script), "--quiet"],
                          capture_output=True, cwd=HERE)
    out = proc.stdout.decode("utf-8", "replace").strip().splitlines()
    tail = next((l for l in reversed(out) if l.strip()), "")
    return proc.returncode == 0, tail


def main() -> int:
    use_utf8_stdout()
    print("=" * 88)
    print("第 5 章 送审门槛复核")
    print("=" * 88)

    ok, notes = compile_gate()
    print(f"[{'PASS' if ok else 'FAIL'}] compile   " + " / ".join(notes))
    overall = ok

    for label, script in (("floats ", "ch5_floats.py"),
                          ("order  ", "ch5_order.py"),
                          ("lexicon", "ch5_lexcheck.py"),
                          ("refs   ", "ch5_refs.py")):
        good, tail = run(script)
        print(f"[{'PASS' if good else 'FAIL'}] {label}   {tail}")
        overall &= good

    print("=" * 88)
    print("总判定：" + ("PASS——机械门槛全过" if overall else "FAIL——见上"))
    print("⚠ 机械门槛不覆盖：\\ref 指向是否恰当、禁用词命中的语义裁定、数字忠实度、"
          "反向 churn（口径声明/锁定措辞/scope 句/种子与配置限定有无被削弱）。")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
