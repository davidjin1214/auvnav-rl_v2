"""
paper/thesis_ch5/tools/ch5_metrics.py

语体计量 (register metrics) for Chapter 5, per spec §0.5.9 (d).

Reports, under the caliber implemented in _ch5_corpus.py:
  A  破折号密度  -- per-section and chapter-wide, body-paragraph and with-caption
  B  段落句数    -- paragraphs over the §0.5.7 4-sentence diagnostic trigger
  C  最长句      -- top-N by net chars, with section / subsection location
  D  caption 长度 -- longest captions (layout input: long captions defer floats)

⚠ These are DIAGNOSTIC READINGS, not thresholds. spec §0.5.8 (用户拍板的最高写作原则)
   forbids preset quantitative targets for page count, cite count and paragraph
   length; §0.5.9 (d) says so again for these numbers specifically. In particular:
     - the later sections' higher dash density is an argument-type difference, not
       register decay -- §5.7-§5.10 are where 口径隔离 / 种子与配置限定 / scope 句
       live, and the 破折号 is the standard Chinese carrier for those;
     - NEVER improve any of these numbers by deleting or shortening a 口径声明,
       锁定措辞, scope 句, or 种子/配置限定. Rewrite, never delete.

Usage:
    python ch5_metrics.py                       # working tree
    python ch5_metrics.py --top 20              # longer top-sentence list
    python ch5_metrics.py --baseline 8cd9160 bbffb98
        # re-measure git baselines with THIS code and print a trend table.
        # Historical values quoted in review docs came from other implementations
        # and are not comparable -- see spec §0.5.9 (d).
"""

from __future__ import annotations

import argparse
import collections

from _ch5_corpus import (
    DASH, SECTION_NO, SECTION_ORDER, load, load_baseline, net_chars,
    sentences, use_utf8_stdout,
)

SENTENCE_TRIGGER = 4  # §0.5.7 diagnostic trigger (NOT a cap)


def summarise(doc: dict) -> dict:
    paras = [(s, p) for s in SECTION_ORDER for p in doc[s]["paras"]]
    caps = [(s, c) for s in SECTION_ORDER for c in doc[s]["caps"]]
    body_dash = sum(p["text"].count(DASH) for _, p in paras)
    cap_dash = sum(c.count(DASH) for _, c in caps)
    over = collections.Counter()
    longest = 0
    total_sent = 0
    for _, p in paras:
        ss = sentences(p["text"])
        total_sent += len(ss)
        if len(ss) > SENTENCE_TRIGGER:
            over[len(ss)] += 1
        for s in ss:
            longest = max(longest, net_chars(s))
    return {
        "paras": len(paras), "dash": body_dash,
        "cap_units": len(paras) + len(caps), "cap_dash": body_dash + cap_dash,
        "over": over, "sentences": total_sent, "longest": longest,
    }


def main() -> None:
    use_utf8_stdout()
    ap = argparse.ArgumentParser(description="Chapter 5 register metrics (spec §0.5.9 (d))")
    ap.add_argument("--top", type=int, default=15, help="longest-sentence list length")
    ap.add_argument("--baseline", nargs="*", metavar="COMMIT",
                    help="also re-measure these git commits with this code")
    args = ap.parse_args()

    doc = load()

    print("=" * 92)
    print("A. 破折号密度（一处 = 一个「——」；诊断读数，非门槛）")
    print("=" * 92)
    print(f"{'节':<6}{'文件':<14}{'正文段':>7}{'——':>5}{'密度':>8}   {'含cap单元':>9}{'——':>5}{'密度':>8}")
    tp = td = tc = tcd = 0
    for stem in SECTION_ORDER:
        paras, caps = doc[stem]["paras"], doc[stem]["caps"]
        d = sum(p["text"].count(DASH) for p in paras)
        cd = d + sum(c.count(DASH) for c in caps)
        cu = len(paras) + len(caps)
        tp += len(paras); td += d; tc += cu; tcd += cd
        print(f"{SECTION_NO[stem]:<6}{stem:<14}{len(paras):>7}{d:>5}{d / len(paras):>8.3f}"
              f"   {cu:>9}{cd:>5}{cd / cu:>8.3f}")
    print("-" * 92)
    print(f"{'全章':<20}{tp:>7}{td:>5}{td / tp:>8.3f}   {tc:>9}{tcd:>5}{tcd / tc:>8.3f}")

    print()
    print("=" * 92)
    print(f"B. 段落句数（> {SENTENCE_TRIGGER} 句触发诊断；§0.5.7 已降格为触发器，不是硬帽）")
    print("=" * 92)
    over = []
    total = 0
    for stem in SECTION_ORDER:
        for i, p in enumerate(doc[stem]["paras"], 1):
            total += 1
            n = len(sentences(p["text"]))
            if n > SENTENCE_TRIGGER:
                over.append((stem, i, n, net_chars(p["text"]), p["sub"], p["text"]))
    dist = collections.Counter(o[2] for o in over)
    print(f"全章正文段 {total}；触发 {len(over)} 段（{len(over) / total * 100:.1f}%）；"
          f"分布 {dict(sorted(dist.items()))}")
    print()
    for stem, i, n, nc, sub, text in over:
        print(f"  §{SECTION_NO[stem]:<5}{stem:<13}#{i:<3}{n:>3}句{nc:>5}字  [{sub}]")
        print(f"        {text[:76]}")

    print()
    print("=" * 92)
    print(f"C. 最长句 top{args.top}")
    print("=" * 92)
    allsent = []
    for stem in SECTION_ORDER:
        for i, p in enumerate(doc[stem]["paras"], 1):
            for s in sentences(p["text"]):
                allsent.append((net_chars(s), stem, i, p["sub"], s))
    allsent.sort(key=lambda x: -x[0])
    for rank, (n, stem, i, sub, s) in enumerate(allsent[:args.top], 1):
        print(f"{rank:>3}. {n:>4}字  §{SECTION_NO[stem]} {stem}#{i} [{sub}]")
        print(f"      {s.strip()[:98]}")
    print(f"\n总句数 {len(allsent)}；"
          f">=120字 {sum(1 for x in allsent if x[0] >= 120)}；"
          f">=100字 {sum(1 for x in allsent if x[0] >= 100)}；"
          f">=80字 {sum(1 for x in allsent if x[0] >= 80)}")

    print()
    print("=" * 92)
    print("D. caption 净字长度 top10（长 caption 会把浮动整体推迟，见 main.tex 浮动参数注）")
    print("=" * 92)
    caps = sorted(((net_chars(c), stem, c) for stem in SECTION_ORDER
                   for c in doc[stem]["caps"]), key=lambda x: -x[0])
    for n, stem, c in caps[:10]:
        print(f"  {n:>4}字  §{SECTION_NO[stem]:<5}{c[:64]}...")
    print(f"  共 {len(caps)} 条，均值 {sum(c[0] for c in caps) / len(caps):.0f} 字")

    if args.baseline:
        print()
        print("=" * 92)
        print("E. 跨基线趋势（全部用本脚本重测；历史复审所报数值不可直接相减，见 spec §0.5.9 (d)）")
        print("=" * 92)
        print(f"{'基线':<18}{'正文段':>7}{'——':>5}{'密度':>8}{'触发段':>7}{'最长句':>7}")
        rows = [(c, summarise(load_baseline(c)[0])) for c in args.baseline]
        rows.append(("(工作树)", summarise(doc)))
        for label, s in rows:
            print(f"{label:<18}{s['paras']:>7}{s['dash']:>5}"
                  f"{s['dash'] / s['paras']:>8.3f}{sum(s['over'].values()):>7}{s['longest']:>7}")


if __name__ == "__main__":
    main()
