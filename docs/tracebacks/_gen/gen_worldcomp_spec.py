"""Generate docs/tracebacks/td3bc_worldcomp_teacher_gap.json.

Why this report needed a chain of its own. Chapter 5 quotes its deployable cell
(`0.858 ± 0.080`) and `tests/test_audit_published_numbers.py` had to declare that source
as read by no report chain -- the figure was pinned on the chapter side only, so an edit
to the report it came from would have gone unnoticed. This closes that.

Provenance rule, established on 2026-08-24 by recomputation rather than read off the
prose, because the prose does not state one:

  screening tables (4.1 / 4.2)  `selection/alpha_<a>/seed_*/selected_checkpoint.json`,
                                key `best.eval_success_rate` (and `best.eval_return`),
                                two seeds, dispersion **ddof=0**.
                                NOT `validation/`, which holds no per-seed files at all,
                                and not `test_selected/`, which exists only for the one
                                alpha that was selected -- the coincidence that it
                                reproduces the alpha=0 row is what makes checking the
                                other three rows the thing that settles it.
  formal table (5.1)            `test_selected/alpha_<a>/seed_*.json`, five seeds,
                                dispersion **ddof=0**.
  baseline row (5.1)            `baselines/worldcomp_test_eval.json`, a single file.
  trajectory table (5.4)        the same five `test_selected` files, other keys.

One ambiguity is deliberate rather than resolved: for the deployable protocol the
selected alpha IS 0.0, so `test_bc_selected/alpha_0p0` holds the identical five files and
reproduces the row just as well. The spec cites `test_selected` because that is what the
table's 协议 column means; the two agreeing is a property of this run, not a rule.

Usage:
    python docs/tracebacks/_gen/gen_worldcomp_spec.py [out_dir]

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
DOC = "docs/td3bc_worldcomp_teacher_gap_experiment_report.md"
STEM = "worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone"

SCREEN = "%s/" + STEM + "/selection/alpha_%s/seed_*/selected_checkpoint.json"
FINAL = "%s/" + STEM + "/test_selected/alpha_%s/seed_*.json"
BASELINE = "deployable_final/" + STEM + "/baselines/worldcomp_test_eval.json"

claims: list[dict] = []


def cell(first: str, column: int, stat: str, sources, metric: str, section: str,
         label: str, expected: str) -> None:
    """One published cell of a markdown table row anchored by its first column.

    `column` counts from 1 at the first cell after the anchor, so it reads off the table
    header rather than off an offset. The capture consumes whole cells to get there --
    counting characters instead is how a spec ends up checking the column next door.

    The capture is written from the START of the line, not from where the anchor left
    off: the tool locates the line with `anchor` and then applies `capture` to that line
    independently. Two extra cells therefore have to be skipped -- the empty one before
    the leading pipe, and the anchor's own cell.
    """
    anchor = r"^\| " + re.escape(first) + r" \|"
    capture = r"(?:[^|]*\|){%d}\s*(-?[0-9.]+)\s*[ |]" % (column + 1)
    claims.append({
        "label": label,
        "section": section,
        "anchor": anchor,
        "capture": capture,
        "stat": stat,
        "metric": metric,
        "sources": sources if isinstance(sources, list) else [sources],
    })
    _verify(section, anchor, capture, expected, label)


def _verify(section: str, anchor: str, capture: str, expected: str, label: str) -> None:
    """Resolve against the live document now, so a mis-pointed column fails here.

    The expectation lives in this generator and never reaches the spec -- a spec that
    carried the figure would keep passing after the report changed, which is the one
    thing these tables exist to notice.

    `capture` is applied to the whole line, alone, because that is what the tool does.
    Verifying `anchor + capture` instead passed on 28 claims whose column was two cells
    off, since concatenating them hides exactly the offset the tool then walks.
    """
    lines = (REPO / DOC).read_text(encoding="utf-8").split("\n")
    lo = next(i for i, ln in enumerate(lines) if ln.startswith("### " + section))
    hi = next((i for i in range(lo + 1, len(lines))
               if re.match(r"^#{1,3} ", lines[i])), len(lines))
    hits = [i for i in range(lo, hi) if re.search(anchor, lines[i])]
    assert len(hits) == 1, f"{label}: anchor matched {len(hits)} lines in {section}"
    m = re.search(capture, lines[hits[0]])
    assert m, f"{label}: capture did not fire on {lines[hits[0]]!r}"
    assert m.group(1) == expected, f"{label}: captured {m.group(1)!r}, expected {expected!r}"


# --------------------------------------------------- 4.1 / 4.2 screening (2 seeds, ddof=0)
SCREEN_ROWS = [
    ("4.1 Deployable screening", "deployable_screen", [
        ("0.0", "0p0", "0.7875", "0.0625", "-12.77"),
        ("0.1", "0p1", "0.7250", "0.0750", "-30.03"),
        ("0.25", "0p25", "0.7250", "0.1000", "-22.09"),
        ("0.5", "0p5", "0.5125", "0.0125", "-33.82"),
    ]),
    ("4.2 Privileged-critic screening", "privileged_screen", [
        ("0.1", "0p1", "0.8125", "0.0875", "-29.83"),
        ("0.25", "0p25", "0.6500", "0.1250", "-18.40"),
    ]),
]
for section, phase, rows in SCREEN_ROWS:
    src = SCREEN % (phase, "%s")
    for alpha, tag, mean, sd, ret in rows:
        base = f"{phase} alpha={alpha}"
        cell(alpha, 1, "mean", src % tag, "best.eval_success_rate", section,
             f"{base} val success", mean)
        cell(alpha, 2, "sd0", src % tag, "best.eval_success_rate", section,
             f"{base} val success sd", sd)
        cell(alpha, 3, "mean", src % tag, "best.eval_return", section,
             f"{base} val return", ret)

# ------------------------------------------------------- 5.1 formal table (5 seeds, ddof=0)
FORMAL = [
    ("deployable", "deployable_final", "0p0",
     ["0.858", "0.080", "-14.29", "7.91", "63.19"]),
    ("privileged-critic", "privileged_final", "0p1",
     ["0.922", "0.086", "16.49", "5.73", "54.33"]),
]
COLUMNS = [(2, "mean", "eval_success_rate", "success"),
           (3, "sd0", "eval_success_rate", "success sd"),
           (4, "mean", "eval_return", "return"),
           (5, "mean", "eval_safety_cost", "safety cost"),
           (6, "mean", "eval_time_s", "time")]
for protocol, phase, tag, values in FORMAL:
    src = FINAL % (phase, tag)
    for (column, stat, metric, what), expected in zip(COLUMNS, values):
        cell(protocol, column, stat, src, metric, "5.1 正式主结果",
             f"{protocol} {what}", expected)

cell("worldcomp baseline", 2, "mean", BASELINE, "eval_success_rate", "5.1 正式主结果",
     "worldcomp baseline success", "0.990")
cell("worldcomp baseline", 4, "mean", BASELINE, "eval_return", "5.1 正式主结果",
     "worldcomp baseline return", "32.19")

# ---------------------------------------------------------------- 5.4 trajectory quality
TRAJECTORY = [
    ("progress ratio", "eval_progress_ratio", "0.735", "0.848"),
    ("path efficiency", "eval_path_efficiency", "0.701", "0.775"),
    ("mean path length (m)", "eval_path_length_m", "56.83", "51.23"),
]
for row, metric, dep, priv in TRAJECTORY:
    cell(row, 1, "mean", FINAL % ("deployable_final", "0p0"), metric, "5.4 轨迹质量指标",
         f"{row} deployable", dep)
    cell(row, 2, "mean", FINAL % ("privileged_final", "0p1"), metric, "5.4 轨迹质量指标",
         f"{row} privileged-critic", priv)

spec = {
    "chain": "td3bc_worldcomp_teacher_gap",
    "doc": DOC,
    "root": "results/offline/td3bc/phase0c/worldcomp_teacher_gap",
    "metric": "eval_success_rate",
    "note": (
        "Provenance rule recomputed on 2026-08-24, not read off the prose, which states "
        "none. Screening tables come from selection/alpha_*/seed_*/selected_checkpoint.json "
        "under the nested key best.eval_success_rate (two seeds, ddof=0); validation/ holds "
        "no per-seed files and test_selected/ exists only for the selected alpha, so it "
        "reproduces the alpha=0 row by coincidence and cannot supply the other three. The "
        "formal table and the trajectory table come from test_selected/alpha_*/seed_*.json "
        "(five seeds, ddof=0), and the baseline row from a single baselines/ file. For the "
        "deployable protocol test_bc_selected/alpha_0p0 holds the identical files, because "
        "the selected alpha is 0.0; the spec cites test_selected because that is what the "
        "table's 协议 column means."),
    "claims": claims,
}

path = OUT_DIR / "td3bc_worldcomp_teacher_gap.json"
path.parent.mkdir(parents=True, exist_ok=True)
io.open(path, "w", encoding="utf-8", newline="\n").write(
    json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {path}: {len(claims)} claims")
