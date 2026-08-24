"""Generate docs/tracebacks/data_integrity_open_items.json.

Why this document needed a chain of its own. It is the 娘家 of the clean-manifest probe:
the four cells `0.902/0.862` and `0.918/0.870` are published here and nowhere else on the
report side, so `tests/test_audit_published_numbers.py` had to declare both clean-probe
sources as read by no report chain. Everything else in the readout -- the displacements,
the flip, the collector baseline -- had no pin at all.

`0993832` is the reason the dispersions are pinned as hard as the means here. The tool the
document tells the reader to recompute with, `paper/thesis_ch5/tools/ch5_clean_probe_readout.py`,
froze only the means; the cell that had drifted was a standard deviation, and a mean-only
expectation cannot see that. The frozen dispersions were added to that tool in the same
fix, but a frozen expectation and a published figure are different objects: the expectation
says what the readout computes, this chain says what the document prints.

Provenance rule, recomputed on 2026-08-24 rather than read off `ch5_clean_probe_readout.py`
-- a chain that trusts that tool's paths inherits whatever the tool got wrong, and the tool
is exactly the thing that had already missed a drift here:

  已刊 column (1250..1349)   `rebrac/formal/<dataset>/actorb_4p0__criticb_2p0/test/seed_*.json`,
                             five seeds, dispersion **ddof=0** (ddof=1 gives 0.024 where
                             0.021 is printed, so the two conventions are distinguishable
                             on this row and the population one wins).
  干净集 column (3000..3099) `rebrac/clean_probe/cross-{1000,2000}/seed_*.json`, five seeds,
                             same convention.
  位移 / 翻转 columns        differences of those means, printed in percentage points --
                             `scale: 0.01`, unit conversion only.
  采集器基线 row             two single-file readouts under `rebrac/clean_probe/baselines/`;
                             the analytic collector was evaluated once per manifest, so
                             there are no seeds to disperse over.
  §5.7 quotes                the TD3+BC halves come from
                             `td3bc/phase0c/stage_c_final/<dataset>/test_selected/alpha_*/`,
                             the same files the chapter's §5.6 rows read.

Deliberately not covered, so the count is not mistaken for completeness:

  * every `t`, `SE` and 95% CI in the table and the prose. The tool computes means,
    dispersions, counts and differences; a paired t over per-seed differences is a
    statistic it does not have, and faking one through `scale` would be a fudge.
  * the difference-in-differences `+0.80` pp. It is a difference of two differences over
    four sources, and `delta` takes exactly two.
  * `0.000505` on the 订正 line, which is |sd0 - 0.025| -- a distance to a wrong value,
    not a recomputation of anything.
  * §1's dataset metadata (`num_transitions`, `success_rate` 0.8855) and §2/§3/§5, whose
    sources are `offline_data/` and `benchmarks/`, a different provenance family with a
    different root. C-class item 12 is the audit that covers `benchmarks/`.

One claim is an erratum. The 订正 blockquote keeps the superseded `\\pm0.025` standing and
says underneath that it is wrong; a plain claim on it would sit red forever. As an erratum
it asserts the disclosure instead -- it must keep failing to reproduce, and it fails by
0.000505 against a half-unit budget of 0.0005, which is as thin a margin as this repo has.
That thinness is the point: if the underlying per-seed readouts are ever regenerated and
move even slightly, the erratum flips and the 订正 needs re-reading.

Usage:
    python docs/tracebacks/_gen/gen_data_integrity_spec.py [out_dir]

Edit this file, never the JSON it writes: `test_every_committed_spec_can_be_regenerated`
reruns it and compares byte for byte, so a hand edit to the spec shows up as a failure.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "tracebacks"
DOC = "docs/data_integrity_open_items.md"

DATASET = "crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep%s"
FORMAL = "rebrac/formal/" + DATASET + "/actorb_4p0__criticb_2p0/test/seed_*.json"
CLEAN = "rebrac/clean_probe/cross-%s/seed_*.json"
BASE_PUBLISHED = "rebrac/clean_probe/baselines/crosscomp_published.json"
BASE_CLEAN = "rebrac/clean_probe/baselines/crosscomp_clean_s3000.json"
TD3BC = ("td3bc/phase0c/stage_c_final/" + DATASET
         + "/test_selected/alpha_%s/seed_*.json")

# The two headings the claims are scoped to. Both must match exactly one line: this
# document carries two `### ✅ 核实结论（2026-08-09）` headings, one per open item.
SEC_SEED = r"^### ✅ 核实结论（2026-08-09）：\*\*`base_seed`"
SEC_DELTA = r"^### ✅ 污染幅度"

claims: list[dict] = []


def claim(label: str, section: str, anchor: str, capture: str, stat: str, sources,
          expected: str, *, metric: str | None = None, scale: float | None = None,
          note: str | None = None, expect: str | None = None) -> None:
    entry: dict = {
        "label": label,
        "section": section,
        "anchor": anchor,
        "capture": capture,
        "stat": stat,
        "sources": sources if isinstance(sources, list) else [sources],
    }
    if metric is not None:
        entry["metric"] = metric
    if scale is not None:
        entry["scale"] = scale
    if expect is not None:
        entry["expect"] = expect
    if note is not None:
        entry["note"] = note
    claims.append(entry)
    _verify(section, anchor, capture, expected, label)


def _verify(section: str, anchor: str, capture: str, expected: str, label: str) -> None:
    """Resolve against the live document now, so a mis-pointed capture fails here.

    The expectation lives in this generator and never reaches the spec -- a spec that
    carried the figure would keep passing after the document changed, which is the one
    thing these tables exist to notice.

    `capture` is applied to the whole line, alone, because that is what the tool does.
    Verifying `anchor + capture` concatenated instead is how 28 claims in the worldcomp
    chain passed generation with their column two cells off: concatenation hides exactly
    the offset the tool then walks.
    """
    lines = (REPO / DOC).read_text(encoding="utf-8").split("\n")
    heads = [i for i, ln in enumerate(lines) if re.search(section, ln)]
    assert len(heads) == 1, f"{label}: section matched {len(heads)} lines"
    lo = heads[0]
    level = len(re.match(r"^(#{1,6}) ", lines[lo]).group(1))
    hi = next((i for i in range(lo + 1, len(lines))
               if re.match(r"^(#{1,6}) ", lines[i])
               and len(re.match(r"^(#{1,6}) ", lines[i]).group(1)) <= level), len(lines))
    hits = [i for i in range(lo, hi) if re.search(anchor, lines[i])]
    assert len(hits) == 1, f"{label}: anchor matched {len(hits)} lines in section"
    found = re.search(capture, lines[hits[0]])
    assert found, f"{label}: capture did not fire on {lines[hits[0]]!r}"
    assert found.group(1) == expected, \
        f"{label}: captured {found.group(1)!r}, expected {expected!r}"


def _cells(column: int) -> str:
    """Skip to the start of `column`, counting from 1 at the cell after the anchor's.

    Two extra cells are consumed: the empty one before the leading pipe, and the anchor's
    own cell. Counting characters instead is how a spec ends up checking the column next
    door -- reading the offset off the table header is what keeps it honest.
    """
    return r"(?:[^|]*\|){%d}" % (column + 1)


PM_MEAN = r"\s*\$([-+]?[0-9.]+)\\pm"
PM_SD = r"\s*\$[-+]?[0-9.]+\\pm([0-9.]+)\$"
PLAIN = r"\s*\$([-+]?[0-9.]+)\$"
BOLD = r"\s*\*\*\$([-+]?[0-9.]+)\$"

# ---------------------------------------------------------------- §5.7 影响面 quotations
IMPACT = r"^\*\*影响面（限本条）\*\*"
claim("影响面 | ReBRAC-Q cross-2000 mean", SEC_SEED, IMPACT,
      r"ReBRAC-Q \$([0-9.]+)\\pm", "mean", FORMAL % "2000", "0.918")
claim("影响面 | ReBRAC-Q cross-2000 sd", SEC_SEED, IMPACT,
      r"ReBRAC-Q \$[0-9.]+\\pm([0-9.]+)\$", "sd", FORMAL % "2000", "0.030")

QUOTE = r"^TD3\+BC .*数据规模退化的翻转"
claim("影响面 | TD3+BC cross-2000 mean", SEC_SEED, QUOTE,
      r"^TD3\+BC \$([0-9.]+)\\pm", "mean", TD3BC % ("2000", "0p15"), "0.596")
claim("影响面 | TD3+BC cross-2000 sd", SEC_SEED, QUOTE,
      r"^TD3\+BC \$[0-9.]+\\pm([0-9.]+)\$", "sd", TD3BC % ("2000", "0p15"), "0.036")
claim("影响面 | flip high end quoted", SEC_SEED, QUOTE,
      r"翻转」（\$([0-9.]+)\$ 对", "mean", FORMAL % "2000", "0.918")
claim("影响面 | flip low end quoted", SEC_SEED, QUOTE,
      r"回合的 \$([0-9.]+)\$", "mean", FORMAL % "1000", "0.902")
claim("影响面 | flip magnitude", SEC_SEED, r"^这一格上，\*\*最脆弱的是",
      r"只有 \$([-+]?[0-9.]+)\$pp", "delta", [FORMAL % "1000", FORMAL % "2000"],
      "+1.6", scale=0.01)

# ------------------------------------------------------------------- the δ readout table
ROWS = [
    ("cross-1000", r"^\| ReBRAC-Q cross-1000 \|", FORMAL % "1000", CLEAN % "1000",
     ("0.902", "0.021", "0.862", "0.019", "-4.0")),
    ("cross-2000", r"^\| ReBRAC-Q cross-2000 \|", FORMAL % "2000", CLEAN % "2000",
     ("0.918", "0.030", "0.870", "0.024", "-4.8")),
]
for row, anchor, published, clean, want in ROWS:
    p_mean, p_sd, c_mean, c_sd, shift = want
    claim(f"δ表 | {row} 已刊 mean", SEC_DELTA, anchor, _cells(1) + PM_MEAN,
          "mean", published, p_mean)
    claim(f"δ表 | {row} 已刊 sd", SEC_DELTA, anchor, _cells(1) + PM_SD,
          "sd", published, p_sd)
    claim(f"δ表 | {row} 干净集 mean", SEC_DELTA, anchor, _cells(2) + PM_MEAN,
          "mean", clean, c_mean)
    claim(f"δ表 | {row} 干净集 sd", SEC_DELTA, anchor, _cells(2) + PM_SD,
          "sd", clean, c_sd)
    claim(f"δ表 | {row} 位移 pp", SEC_DELTA, anchor, _cells(3) + PLAIN,
          "delta", [published, clean], shift, scale=0.01)

FLIP = r"^\| \*\*翻转（2000 "
claim("δ表 | 翻转 已刊 pp", SEC_DELTA, FLIP, _cells(1) + BOLD, "delta",
      [FORMAL % "1000", FORMAL % "2000"], "+1.6", scale=0.01)
claim("δ表 | 翻转 干净集 pp", SEC_DELTA, FLIP, _cells(2) + BOLD, "delta",
      [CLEAN % "1000", CLEAN % "2000"], "+0.8", scale=0.01)

BASELINE = r"^\| 采集器基线（解析式"
claim("δ表 | 采集器基线 已刊", SEC_DELTA, BASELINE, _cells(1) + PLAIN,
      "mean", BASE_PUBLISHED, "0.950")
claim("δ表 | 采集器基线 干净集", SEC_DELTA, BASELINE, _cells(2) + PLAIN,
      "mean", BASE_CLEAN, "0.930")
claim("δ表 | 采集器基线 位移 pp", SEC_DELTA, BASELINE, _cells(3) + PLAIN,
      "delta", [BASE_PUBLISHED, BASE_CLEAN], "-2.0", scale=0.01)
claim("δ表 | 采集器基线 回合数", SEC_DELTA, BASELINE, r"n=([0-9]+)",
      "mean", BASE_CLEAN, "100", metric="num_eval_episodes")

# ------------------------------------------------------------------- the 2026-08-24 订正
ERRATUM_NOTE = (
    "pins the 2026-08-24 订正 in this same blockquote: the clean cross-2000 dispersion was "
    "printed as 0.025 where the per-seed population sd is 0.024495, whose correct "
    "three-place rounding is 0.024. The gap is 0.000505 against a half-unit budget of "
    "0.0005, so this must keep failing; the day it reproduces, the 订正 is describing a "
    "readout that no longer exists.")
claim("订正 | 原印的错值（勘误）", SEC_DELTA, r"^> \*\*2026-08-24 订正\*\*",
      r"原印 \$\\pm([0-9.]+)\$", "sd0", CLEAN % "2000", "0.025",
      expect="mismatch", note=ERRATUM_NOTE)
claim("订正 | 错值复述（勘误）", SEC_DELTA, r"与真值相差.*可接受舍入范围",
      r"^> \$([0-9.]+)\$ 与真值相差", "sd0", CLEAN % "2000", "0.025",
      expect="mismatch",
      note=ERRATUM_NOTE + " Restated on its own line; a fix that touched only the first "
                          "mention would leave this one standing.")
claim("订正 | 逐种子值", SEC_DELTA, r"^> \$\[.*的总体标准差为",
      r"\$\[([0-9., \\]+)\]\$", "seeds", CLEAN % "2000",
      r"0.89,\ 0.87,\ 0.86,\ 0.83,\ 0.90")
claim("订正 | 总体标准差全精度", SEC_DELTA, r"^> \$\[.*的总体标准差为",
      r"总体标准差为 \$([0-9.]+)\$", "sd0", CLEAN % "2000", "0.024495")
claim("订正 | 总体标准差三位", SEC_DELTA, r"^> \$\[.*的总体标准差为",
      r"三位舍入为 \$([0-9.]+)\$", "sd0", CLEAN % "2000", "0.024")
claim("订正 | 论文侧一直印的值", SEC_DELTA, r"一直印的就是",
      r"一直印的就是 \$([0-9.]+)\$", "sd0", CLEAN % "2000", "0.024")

# ------------------------------------------------------- 三条不利事实 (same section)
claim("不利事实 2 | 抽样难度位移", SEC_DELTA, r"^\s+偏移，实测",
      r"实测 \$([-+]?[0-9.]+)\$ pp", "delta", [BASE_PUBLISHED, BASE_CLEAN],
      "-2.0", scale=0.01)
claim("不利事实 3 | cross-1000 基线差距", SEC_DELTA, r"^3\. \*\*本次只补评了",
      r"基线差距（\$([0-9.]+)\$", "delta", [TD3BC % ("1000", "0p25"), FORMAL % "1000"],
      "23.0", scale=0.01)
claim("不利事实 3 | cross-2000 基线差距", SEC_DELTA, r"pp）\*\*不能\*\*在干净集上重述",
      r"^\s+\$([0-9.]+)\$ pp）", "delta", [TD3BC % ("2000", "0p15"), FORMAL % "2000"],
      "32.2", scale=0.01)

spec = {
    "chain": "data_integrity_open_items",
    "doc": DOC,
    "root": "results/offline",
    "metric": "eval_success_rate",
    "note": (
        "Provenance rule recomputed on 2026-08-24, not read off "
        "paper/thesis_ch5/tools/ch5_clean_probe_readout.py, which is the tool this "
        "document tells the reader to recompute with and is also the tool whose "
        "mean-only frozen expectation let a dispersion drift through (0993832). 已刊 "
        "cells come from rebrac/formal/<dataset>/actorb_4p0__criticb_2p0/test/seed_*.json "
        "and 干净集 cells from rebrac/clean_probe/cross-*/seed_*.json, five seeds each, "
        "dispersion ddof=0 -- distinguishable on the cross-1000 row, where ddof=1 would "
        "print 0.024 against the published 0.021. 位移 and 翻转 cells are differences of "
        "those means in percentage points (scale 0.01); the 采集器基线 row is two "
        "single-file readouts of the analytic collector, which has no seeds. The §5.7 "
        "quotations read TD3+BC from td3bc/phase0c/stage_c_final/. Deliberately not "
        "covered: every t / SE / 95% CI (not statistics this tool computes), the "
        "difference-in-differences +0.80 pp (four sources, delta takes two), 0.000505 (a "
        "distance to a wrong value), and items 2/3/5, whose sources are offline_data/ "
        "and benchmarks/."),
    "claims": claims,
}

path = OUT_DIR / "data_integrity_open_items.json"
path.parent.mkdir(parents=True, exist_ok=True)
io.open(path, "w", encoding="utf-8", newline="\n").write(
    json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {path}: {len(claims)} claims")
