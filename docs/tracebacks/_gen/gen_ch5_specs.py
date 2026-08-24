"""Generate the four chapter-5 traceback specs under docs/tracebacks/.

Why the chapter needs chains of its own.  `ch5_dispersion_audit.py` classifies a `\\pm`
only when the same row prints its per-seed values; 31 readings across four section files
do not, and it says so rather than guessing.  The report-side chains already recompute
most of those cells -- but from the *report's* line, not the chapter's.  Nothing checked
that the two sides agree, and `0993832` showed they can drift apart: the report printed
0.870 +/- 0.025 where the chapter had the correct 0.870 +/- 0.024.

So each chapter reading is pinned to the same per-seed files the report side uses, and
independently: no verdict is inherited.  When a source tree moves, both sides break, which
is the point.

Why four specs and not one.  A spec has exactly one `doc` and one `root`.  Four section
files hold the 31 readings, so four is the floor; each spec's root is then the lowest
common ancestor of that file's sources, which for two of them sits one level above the
corresponding report chain's root (`results/offline` vs `results/offline/rebrac`,
`experiments` vs `experiments/arrival_v2_prototype`).  The source *strings* therefore
differ from the report side's by that prefix while resolving to the same files.

Three cells had no provenance rule anywhere and were established by recomputation:
  * the k-ladder k=8 and k=12 rows -- no report prints them.  `final_eval.json` reproduces
    both means and both sds; every reduction over `eval_log.csv` (mean, max, last) misses.
  * the TD3+BC worldcomp cell 0.858 +/- 0.080 -- quoted in the ReBRAC report as a
    comparison, but its own report has no chain.  Two candidate directories exist
    (`test_selected/alpha_0p0` and `test_bc_selected/alpha_0p0`) and the worldcomp report
    claims they agree; recomputation confirms it, and the privileged cell 0.922 +/- 0.086
    reproduces from the sibling directory as an independent check on the layout.
  * the clean-probe cells 0.862 / 0.870 -- only ever published in
    `docs/data_integrity_open_items.md`, which has no chain.

Usage:
    python docs/tracebacks/_gen/gen_ch5_specs.py [out_dir]

Edit this file, never the JSON it writes: `test_every_committed_spec_can_be_regenerated`
reruns it and compares byte for byte, so a hand edit to the spec shows up as a failure.
"""
import io
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SECTIONS = REPO / "paper" / "thesis_ch5" / "sections"
# Optional output directory, so the regression test can regenerate into a scratch
# tree instead of overwriting the committed specs.
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "tracebacks"

# ---------------------------------------------------------------- source globs

A0 = "protocol_screen_v2/A0_single_u10_cross_tgt15/efficiency_v2/%s_k4/seed_*/final_eval.json"
LADDER = "single_u15_cross_tgt15/arrival_v2/sac_vanilla/%s/seed_*/results/final_eval.json"
# ch5_online's root is one level up, so its ladder globs carry the extra prefix.
LADDER_E = "arrival_v2_prototype/" + LADDER
CLEAN = "rebrac/clean_probe/cross-%s/seed_*.json"
STEM = "crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone"
WSTEM = "worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone"

# Byte-identical to the td3bc_phase0c chain's globs, modulo each spec's root prefix.
TD3BC = "stage_c_final/%s/test_selected/alpha_%s/seed_*.json"
PUREBC = "stage_c_bc_final/%s/test_bc_selected/alpha_0p0/seed_*.json"
WORLD = ("worldcomp_teacher_gap/deployable_final/%s/test_selected/alpha_0p0/seed_*.json"
         % WSTEM)

# ---------------------------------------------------------------- capture scheme

# One published `mean \pm sd` cell, matched without capturing, so it can be used to walk
# past cells that come earlier on the same line.  Requiring digits on both sides is what
# makes it skip the bare `$\pm$` that the chapter's table captions write in prose.
PMCELL = r"[0-9.]+ \\pm [0-9.]+"


def capture(index: int, stat: str) -> str:
    """Read the `index`-th published `mean \\pm sd` cell on the anchored line.

    `\\mathbf{` is optional because which cell carries the emphasis is typography: the
    chapter bolds the best entry per column, and that moves when a table is re-sorted.
    """
    head = ("(?:.*?%s)" % PMCELL) * index + r".*?\$(?:\\mathbf\{)?"
    return head + (r"([0-9.]+) \\pm" if stat == "mean" else r"[0-9.]+ \\pm ([0-9.]+)")


# ---------------------------------------------------------------- the readings
#
# (line, anchor prefix, [(index, expected mean, expected sd, label, source, metric)])
# The expected figures live here and are NOT written into the spec: a spec that carries
# the figure keeps passing after the document changes, which is the one failure the tool
# exists to prevent.  Here they are a generation-time self-check that a cell index has
# not slipped by one.

SPECS = {}


def add(chain, doc, root, note, rows, metric="eval_success_rate"):
    SPECS[chain] = {"doc": doc, "root": root, "note": note, "rows": rows, "metric": metric}


add(
    "ch5_online",
    "paper/thesis_ch5/sections/online.tex",
    "experiments",
    "Chapter 5 section 5.5, both of its tables. The sensor-screen table is the same six "
    "cells the online_a0 chain pins on the line-summary side; the bottleneck ladder is "
    "not published in any report, so its rule was recomputed: per-seed final_eval.json, "
    "aggregated as mean and sd, and no reduction over eval_log.csv (mean, max, last) "
    "reproduces any of the three s0 rows. Root is `experiments` rather than either "
    "sub-tree because this one file cites both; the globs below resolve to the same files "
    "the online_a0 and arrival_v2 chains read.",
    [
        (270, "$s_0$（可部署，单点 DVL） &", [
            (0, "0.789", "0.211", "5.5 sensor | s0 | success", A0 % "s0", None),
            (1, "0.801", "0.057", "5.5 sensor | s0 | path efficiency", A0 % "s0",
             "eval_path_efficiency")]),
        (271, "$s_1$（$+$ 短程 ADCP，参照上界） &", [
            (0, "0.967", "0.027", "5.5 sensor | s1 | success", A0 % "s1", None),
            (1, "0.854", "0.024", "5.5 sensor | s1 | path efficiency", A0 % "s1",
             "eval_path_efficiency")]),
        (272, "$s_2$（$+$ 长程 ADCP，参照上界） &", [
            (0, "0.856", "0.204", "5.5 sensor | s2 | success", A0 % "s2", None),
            (1, "0.813", "0.100", "5.5 sensor | s2 | path efficiency", A0 % "s2",
             "eval_path_efficiency")]),
        (331, "可部署 $s_0$，$k=4$ &", [
            (0, "0.46", "0.39", "5.5 ladder | s0 k=4", LADDER_E % "s0_k4", None)]),
        (332, "可部署 $s_0$，$k=8$ &", [
            (0, "0.76", "0.22", "5.5 ladder | s0 k=8", LADDER_E % "s0_k8", None)]),
        (333, "可部署 $s_0$，$k=12$ &", [
            (0, "0.88", "0.04", "5.5 ladder | s0 k=12", LADDER_E % "s0_k12", None)]),
        (334, "空间参照 $s_1$，$k=4$（参照上界） &", [
            (0, "0.90", "0.00", "5.5 ladder | s1 k=4", LADDER_E % "s1_k4", None)]),
    ])

add(
    "ch5_boundary",
    "paper/thesis_ch5/sections/boundary.tex",
    "experiments/arrival_v2_prototype",
    "Chapter 5 section 5.8 quotes two cells of the section 5.5 ladder -- the cross-line "
    "comparison the section is built on. They are pinned here to the same files rather "
    "than to section 5.5's rendering of them, so that editing one place and not the other "
    "shows up. Root equals the arrival_v2 chain's root, so the globs are byte-identical "
    "to that chain's for the k=4 cell (section 7.10) and follow the same shape for k=12, "
    "which no report publishes.",
    [
        (226, "在线标准 SAC，$k=4$（三种子） &", [
            (0, "0.46", "0.39", "5.8 table | online SAC k=4", LADDER % "s0_k4", None)]),
        (232, "表中的次序本身构成一个发现", [
            (0, "0.46", "0.39", "5.8 prose | online SAC k=4", LADDER % "s0_k4", None)]),
        (234, "在线一侧的证据同时划定了这一边界的适用范围", [
            (0, "0.88", "0.04", "5.8 prose | online SAC k=12", LADDER % "s0_k12", None)]),
    ])

add(
    "ch5_rebrac",
    "paper/thesis_ch5/sections/rebrac.tex",
    "results/offline",
    "Chapter 5 section 5.6. Root is `results/offline` because this one file cites both "
    "the ReBRAC tree and the TD3+BC tree; the globs resolve to the same files the rebrac "
    "and td3bc_phase0c chains read. Two groups have no chain on the report side and were "
    "recomputed: the clean-probe supplementary evaluation (published only in "
    "docs/data_integrity_open_items.md, whose cross-2000 dispersion was found misprinted "
    "in 0993832 -- the chapter had it right) and the TD3+BC worldcomp cell, whose own "
    "report has no chain.",
    [
        (347, "\\S\\ref{subsec:ch5_td3bc_support} 把两千回合处的回落确立为可复现的真实效应", [
            (1, "0.672", "0.045", "5.6 prose | TD3+BC crosscomp-1000",
             "td3bc/phase0c/" + TD3BC % (STEM + "_ep1000", "0p25"), None),
            (3, "0.596", "0.036", "5.6 prose | TD3+BC crosscomp-2000",
             "td3bc/phase0c/" + TD3BC % (STEM + "_ep2000", "0p15"), None)]),
        (351, "这一翻转所依据的两个数据格中", [
            (0, "0.862", "0.019", "5.6 prose | clean probe cross-1000", CLEAN % "1000", None),
            (1, "0.870", "0.024", "5.6 prose | clean probe cross-2000", CLEAN % "2000", None)]),
        (371, "横流补偿参照（$1000$ 回合） & 可部署 &", [
            (0, "0.672", "0.045", "5.6 table | TD3+BC crosscomp-1000",
             "td3bc/phase0c/" + TD3BC % (STEM + "_ep1000", "0p25"), None)]),
        (372, "横流补偿参照（$2000$ 回合） & 可部署 &", [
            (0, "0.596", "0.036", "5.6 table | TD3+BC crosscomp-2000",
             "td3bc/phase0c/" + TD3BC % (STEM + "_ep2000", "0p15"), None)]),
        # The identical multirow head opens two rows of two different tables; what tells
        # them apart is the protocol cell that follows.
        (374, "\\multirow{2}{*}{\\makecell[l]{世界坐标系流速补偿参照"
              "\\\\（$1000$ 回合）}} & 可部署 &", [
            (0, "0.858", "0.080", "5.6 table | TD3+BC worldcomp deployable",
             "td3bc/phase0c/" + WORLD, None)]),
        (384, "世界坐标系流速补偿参照的一千回合数据把考察从数据规模转向价值网络的信息", [
            (0, "0.858", "0.080", "5.6 prose | TD3+BC worldcomp deployable",
             "td3bc/phase0c/" + WORLD, None)]),
        (396, "\\caption{正式协议下主线与消融各实验单元的逐种子终检成功率", [
            (0, "0.894", "0.048", "5.6 caption | ReBRAC crosscomp-2000 (4.0, 1.0)",
             "rebrac/formal/%s_ep2000/actorb_4p0__criticb_1p0/test/seed_*.json" % STEM, None)]),
        (573, "两千回合数据上训练与评估任务实例的重合", [
            (0, "0.862", "0.019", "5.6 impl | clean probe cross-1000", CLEAN % "1000", None),
            (1, "0.870", "0.024", "5.6 impl | clean probe cross-2000", CLEAN % "2000", None)]),
    ])

add(
    "ch5_td3bc",
    "paper/thesis_ch5/sections/td3bc.tex",
    "results/offline/td3bc/phase0c",
    "Chapter 5 section 5.4. Root equals the td3bc_phase0c chain's root, so the six data "
    "scale globs are byte-identical to that chain's. The worldcomp cell is the seventh "
    "and has no chain on the report side; its rule was recomputed, and it is the same "
    "cell section 5.6 quotes, pinned there too so the two chapter sites cannot drift "
    "apart silently.",
    [
        (207, "$500$  &", [
            (0, "0.592", "0.119", "5.4 table | 500 | TD3+BC",
             TD3BC % (STEM, "0p5"), None),
            (1, "0.524", "0.053", "5.4 table | 500 | pure BC", PUREBC % STEM, None)]),
        (208, "$1000$ &", [
            (0, "0.672", "0.045", "5.4 table | 1000 | TD3+BC",
             TD3BC % (STEM + "_ep1000", "0p25"), None),
            (1, "0.588", "0.070", "5.4 table | 1000 | pure BC",
             PUREBC % (STEM + "_ep1000"), None)]),
        (209, "$2000$ &", [
            (0, "0.596", "0.036", "5.4 table | 2000 | TD3+BC",
             TD3BC % (STEM + "_ep2000", "0p15"), None),
            (1, "0.534", "0.079", "5.4 table | 2000 | pure BC",
             PUREBC % (STEM + "_ep2000"), None)]),
        (240, "可部署协议 & 单点观测 &", [
            (0, "0.858", "0.080", "5.4 table | worldcomp deployable", WORLD, None)]),
    ])

# ---------------------------------------------------------------- build + self-check

total = 0
for chain, meta in SPECS.items():
    lines = (REPO / meta["doc"]).read_text(encoding="utf-8").split("\n")
    claims = []
    for lineno, prefix, cells in meta["rows"]:
        anchor = "^" + re.escape(prefix)
        hits = [i for i, ln in enumerate(lines, 1) if re.search(anchor, ln)]
        assert hits == [lineno], "%s anchor %r hit %s, expected line %d" % (
            chain, prefix, hits, lineno)
        line = lines[lineno - 1]
        for index, want_mean, want_sd, label, source, metric in cells:
            for stat, want in (("mean", want_mean), ("sd", want_sd)):
                pattern = capture(index, stat)
                m = re.search(pattern, line)
                assert m, "%s %s: capture %r did not match line %d" % (chain, label,
                                                                      pattern, lineno)
                assert m.group(1) == want, "%s %s %s: captured %r, expected %r" % (
                    chain, label, stat, m.group(1), want)
                claim = {"label": "%s %s" % (label, stat), "anchor": anchor,
                         "capture": pattern, "stat": stat, "sources": [source]}
                if metric:
                    claim["metric"] = metric
                claims.append(claim)
    spec = {"chain": chain, "doc": meta["doc"], "note": meta["note"],
            "root": meta["root"], "metric": meta["metric"], "claims": claims}
    path = OUT / ("%s.json" % chain)
    io.open(path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    print("wrote %-28s %2d claims" % (path.name, len(claims)))
    total += len(claims)
print("%d claims over %d specs" % (total, len(SPECS)))
