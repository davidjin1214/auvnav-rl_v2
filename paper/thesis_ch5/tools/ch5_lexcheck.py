"""
paper/thesis_ch5/tools/ch5_lexcheck.py

收口检索 (collation search) for Chapter 5 -- spec §0.5.11, layers 2 and 4.

Covers, over 正文 + caption + 子节标题 (comments excluded):
  - 历轮措辞锁禁用词表 (§0.5.11 layer 2), by category
  - 术语四档 (§0.5.10): 黑话 / 概念纠正 / 命名锁 / 首现括注
  - 报告体 marker (§0.5.9 (a)) and 引号规范
  - 英文括注首现一次 (each parenthetical gloss must appear exactly once)

Two design decisions come straight from failures this tooling exists to prevent:

1. **Counts are by OCCURRENCE, not by matching line.** Reporting per line under-counted
   「无差异」as 6 when it is 8 (定点复核 L-2, 2026-07-28).

2. **A frozen baseline separates "already adjudicated" from "new".** Most hits in this
   chapter are legitimately compliant -- 「相当比例」is a quantifier, 「不作等价判断」is a
   negation, the 「无差异」cluster is §5.9.1's pre-registered matrix verdict. Re-arguing
   all of them every round is what let a real hit hide among them: on 2026-07-28 the
   claim「§5.7.1 是全章唯一的规则②漏网」was written into four documents on the strength of
   a search that had穷举 exactly one of the five rule-② words (定点复核 H-1).
   So: `--freeze` records the adjudicated state; a later run flags only DELTAS.

   ⚠ Freezing is an assertion that every current hit has been adjudicated and recorded
   in findings + spec §0.5.11 豁免行. Do not freeze to silence a warning.

Rule-② adjudication criterion, written down so it need not be re-derived
(spec §0.5.11, 2026-07-28): the rule governs **幅度类零结果**. A **离散配置事实**
(which configuration the posterior selection landed on) is outside its scope -- but the
referent must then be written in directly observable form, not as an interpretive
phrase. This is why §5.7.2 says「最优配置退回纯行为克隆」and not「价值项无益于可部署策略」.

Usage:
    python ch5_lexcheck.py                # delta against the frozen baseline
    python ch5_lexcheck.py --full         # every hit with context
    python ch5_lexcheck.py --freeze       # re-adopt current state as adjudicated
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

from _ch5_corpus import (
    CHAPTER_DIR, SECTION_NO, SECTION_ORDER, strip_comments, use_utf8_stdout,
)

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ch5_lexicon_baseline.json")
META_KEY = "_meta"  # provenance stamp, not a watched term: skipped on read and diff

# spec §0.5.11 layer-2 禁用词表 + §0.5.10 术语四档 + §0.5.9 (a) marker
GROUPS: list[tuple[str, list[str]]] = [
    ("§0.5.11 ① 等价/非劣（第3批 H2）", [
        "追平", "持平", "相当", "至少相当", "同等水平", "一致的水平",
        "等价", "无差异", "都不低于", "均不低于", "不劣于", "不低于"]),
    ("§0.5.11 ② 零结果全称化（第3批 H2 连锁）", [
        "不再带来", "不再出现", "未再出现", "不再有", "消失", "抵消殆尽"]),
    ("§0.5.11 ③ 上限（第3批 N1）", [
        "上限", "经验上限", "性能上限", "天花板", "所能达到的上限"]),
    ("§0.5.11 ④ 选择性呈现（第4批 H7）", [
        "单调上升", "单调", "代表性训练", "典型训练", "单次训练内部", "代表性运行", "代表性"]),
    ("§0.5.11 ⑤ 口径反号（第4批 M1）", ["方向相反", "反号", "结论反转"]),
    ("§0.5.11 ⑥ 机制过度归属（第4批 H6）", ["特属于", "专属于"]),
    ("§0.5.11 ⑦ 过度断言（第4批 H4）", [
        "精确指认", "指认为", "证明", "证实", "本章发现", "增量不在于"]),
    ("§0.5.11 ⑧ 混杂/旁证（第4批 M12）", ["混杂因素", "排除了", "旁证"]),
    ("§0.5.11 ⑨ 计数与独立性（第1、2批 C1/C2）", [
        "两种子", "两个种子", "独立采样", "样本不重合", "彼此独立", "相互印证"]),
    ("§0.5.11 ⑩ 不可支撑的预测（第2批 M8）", ["不会翻转其方向", "必然收紧", "终将"]),
    ("§0.5.11 ⑪ 参照口径混称（第5批 M2）", [
        "采集器自身成功率", "自身轨迹的成功率", "采集成功率参照"]),
    ("§0.5.11 ⑫ 检查点口径（第5批 N13）", ["终检", "最终检查点"]),
    ("§0.5.11 ⑬ 非劣判定兜底（第5批 N7）", ["族内判定", "均未被超过", "未见可检出的差距"]),
    ("§0.5.10 (C) 黑话（应零命中）", [
        "旋钮", "可用性与匹配", "regime", "headline", "dispensability"]),
    ("§0.5.10 (D) 概念纠正与术语硬约束（应零命中）", [
        "teacher", "蒸馏师", "船体", "艇体", "导航", "评价器"]),
    ("§0.5.9 (a) 报告体 marker（应零命中）", [
        "本章提出一个", "本章的回答有", "需要强调", "值得注意的是", "本章的新意在于",
        "本章的贡献不是", "本章其余部分", "贯穿这", "相应地", "也是本章的重点"]),
    ("引号规范（须弯引号；应零命中）", ["「", "」", '"', "``", "''"]),
]

BARE_REBRAC = re.compile(r"ReBRAC(?!-Q|\\text\{-\}Q)")
PAREN_GLOSS = re.compile(r"（([A-Za-z][A-Za-z0-9 ,.\-\\{}]*?)）")


def body_lines() -> dict[str, list[tuple[int, str]]]:
    doc = {}
    for stem in SECTION_ORDER:
        path = os.path.join(CHAPTER_DIR, "sections", stem + ".tex")
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        lines = []
        for i, line in enumerate(strip_comments(raw).split("\n"), 1):
            if line.strip():
                lines.append((i, line))
        doc[stem] = lines
    return doc


def count_hits(doc, word) -> tuple[int, dict[str, int], list[tuple[str, int, str]]]:
    total, per_file, where = 0, {}, []
    for stem in SECTION_ORDER:
        n = 0
        for lineno, line in doc[stem]:
            c = line.count(word)
            if c:
                n += c
                idx = line.find(word)
                where.append((stem, lineno, line[max(0, idx - 34):idx + 44]))
        if n:
            per_file[stem] = n
            total += n
    return total, per_file, where


def _git(*args: str) -> str:
    try:
        out = subprocess.run(("git",) + args, cwd=CHAPTER_DIR, capture_output=True)
    except OSError:
        return ""
    return out.stdout.decode("utf-8", "replace").strip() if out.returncode == 0 else ""


def _freeze_meta(entries: int) -> dict[str, object]:
    """Stamp when this baseline was frozen and against which tree.

    The baseline sat unrefreshed across five 整改 batches before anyone noticed, which
    left the lexicon gate failing on backlog drift -- a genuine new violation would have
    been buried in it. Nothing in the file recorded when it was frozen, so it could not
    go stale *visibly*. These three fields are what makes staleness observable.
    """
    return {
        "frozen": _git("log", "-1", "--format=%ad", "--date=short") or "unknown",
        "commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "entries": entries,
    }


def _report_baseline_age(meta: dict[str, object], quiet: bool) -> None:
    """Print the stamp -- and print it even under --quiet once sections/ has moved on."""
    commit = str(meta.get("commit", "unknown"))
    behind = ""
    if commit != "unknown":
        n = _git("rev-list", "--count", f"{commit}..HEAD", "--", "sections")
        if n.isdigit() and int(n):
            behind = f"  ⚠ sections/ 已前进 {n} 个提交——基线未重冻，存量漂移会淹没新违例"
    if behind or not quiet:
        print(f"  基线：{meta.get('frozen', 'unknown')} @ {commit}，"
              f"{meta.get('entries', '?')} 个词条{behind}")


def main() -> int:
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description="Chapter 5 collation search (spec §0.5.11)")
    ap.add_argument("--full", action="store_true", help="print every hit with context")
    ap.add_argument("--quiet", action="store_true", help="verdict line only")
    ap.add_argument("--freeze", action="store_true",
                    help="record current state as the adjudicated baseline")
    args = ap.parse_args()

    doc = body_lines()
    current: dict[str, dict[str, int]] = {}

    for title, words in GROUPS:
        printed_header = False
        for word in words:
            total, per_file, where = count_hits(doc, word)
            current[word] = per_file
            if args.full:
                if not printed_header:
                    print("=" * 92); print(title); print("=" * 92)
                    printed_header = True
                if total == 0:
                    print(f"  「{word}」 零命中")
                else:
                    print(f"  「{word}」 {total} 次  {per_file}")
                    for stem, lineno, ctx in where:
                        print(f"      §{SECTION_NO[stem]:<5}{stem}:{lineno}  …{ctx}…")
        if args.full and printed_header:
            print()

    # naming lock: bare ReBRAC (not ReBRAC-Q)
    bare = {}
    for stem in SECTION_ORDER:
        n = sum(len(BARE_REBRAC.findall(line)) for _, line in doc[stem])
        if n:
            bare[stem] = n
    current["<裸写 ReBRAC>"] = bare

    # English parenthetical glosses: each must appear exactly once
    gloss: dict[str, list[str]] = {}
    for stem in SECTION_ORDER:
        for _, line in doc[stem]:
            for m in PAREN_GLOSS.finditer(line):
                gloss.setdefault(m.group(1).strip(), []).append(SECTION_NO[stem])
    repeated = {k: v for k, v in gloss.items() if len(v) > 1}

    if not args.quiet:
        print("=" * 92)
        print("汇总")
        print("=" * 92)
        print(f"  英文括注词条 {len(gloss)} 个；重复者 {len(repeated)} 个"
              + (f" -> {repeated}" if repeated else ""))
        # Bare "ReBRAC" is NOT automatically a violation: spec §0.5.10 (A) keeps the
        # original method名 as-is, and the lock is only that THIS chapter's method is
        # written ReBRAC-Q. Adjudicated 2026-07-28: the 2 hits in methodology.tex:133
        # both refer to the original method (「离线方法 ReBRAC~\citep{}」/「原始 ReBRAC」)
        # and are correct. Frozen into the baseline; a delta means re-adjudicate.
        print(f"  裸写 ReBRAC（指原始方法时合规，见 §0.5.10 (A)）：{bare if bare else '零命中'}")

    if args.freeze:
        stamped = dict(current)
        stamped[META_KEY] = _freeze_meta(len(current))
        with open(BASELINE, "w", encoding="utf-8") as fh:
            json.dump(stamped, fh, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"\n已冻结基线 -> {os.path.basename(BASELINE)}")
        print("⚠ 冻结即断言「当前每一处命中都已裁定并记入 findings ＋ spec §0.5.11 豁免行」。")
        return 0

    if not os.path.exists(BASELINE):
        print("\n无基线文件；先跑 --freeze（须确认当前每处命中都已裁定）")
        return 0

    with open(BASELINE, encoding="utf-8") as fh:
        base = json.load(fh)
    _report_baseline_age(base.pop(META_KEY, {}), args.quiet)

    deltas = []
    for word, per_file in current.items():
        old = base.get(word, {})
        if per_file != old:
            for stem in sorted(set(old) | set(per_file)):
                a, b = old.get(stem, 0), per_file.get(stem, 0)
                if a != b:
                    deltas.append((word, SECTION_NO[stem] if stem in SECTION_NO else stem, a, b))
    for word in base:
        if word not in current:
            deltas.append((word, "-", sum(base[word].values()), 0))

    print()
    if deltas:
        print("FAIL 相对已裁定基线有变动，逐条须重新裁定并记入 findings：")
        for word, sec, a, b in deltas:
            print(f"  「{word}」 §{sec}: {a} -> {b}")
        return 1
    print(f"PASS 禁用词与术语命中与已裁定基线完全一致（{len(current)} 个词条）")
    if repeated:
        print("FAIL 英文括注重复出现")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
