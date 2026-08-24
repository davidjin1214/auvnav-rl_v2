"""Generate docs/tracebacks/online_a0.json -- the A0 sensor screen of the online line.

Edit this, never the JSON: the two result tables are a regular 2 x 2 x 3 grid, and
hand-editing 24 claims is how a column index silently slips by one.

Provenance rule, established by recomputation rather than read off the prose (see the
`note` written into the spec): the per-seed value is the terminal evaluation stored in
`final_eval.json`, and the aggregate is mean +/- population sd.  Three other candidate
reductions were tried against the same 24 figures and all three miss, `max` included --
which matters, because the doc's own argument list says "best per-cell success >= 70%"
and would tempt a reader into `max`.

Usage:
    python docs/tracebacks/_gen/gen_online_a0_spec.py [out_dir]

Edit this file, never the JSON it writes: `test_every_committed_spec_can_be_regenerated`
reruns it and compares byte for byte, so a hand edit to the spec shows up as a failure.
"""
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Optional output directory, so the regression test in tests/test_audit_published_numbers.py
# can regenerate into a scratch tree instead of overwriting the committed specs.
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "tracebacks"
OUT = OUT_DIR / "online_a0.json"

# Table column order in both result tables -> run directory under the objective.
COLUMNS = ["s1_k4", "s0_k4", "s2_k4"]
OBJECTIVES = ["efficiency_v2", "arrival_v1"]

# Both tables carry the same header and the same two row labels, so a row anchor alone
# hits two lines.  The bold marker line above each table is what separates them.
TABLES = [
    {"metric": "eval_success_rate",
     "title": "success rate",
     "after": "^\\*\\*\u7ed3\u679c\uff08success rate",
     "before": "^\\*\\*\u7ed3\u679c\uff08path efficiency"},
    {"metric": "eval_path_efficiency",
     "title": "path efficiency",
     "after": "^\\*\\*\u7ed3\u679c\uff08path efficiency",
     "before": "^\\*\\*\u53ef\u7528\u4e8e\u8bba\u6587\u7684\u8bba\u70b9"},
]

NOTE = (
    "Provenance rule, recomputed rather than taken from the prose: each seed contributes "
    "the terminal evaluation in final_eval.json (600k steps, 30 episodes, identical to the "
    "last row of eval_log.csv), and the published figure is mean +/- population sd over "
    "seeds 46/47/50. Three other reductions over eval_log.csv were tried against all 24 "
    "figures and every one of them misses: mean over the 60 evaluations (0.573 where 0.967 "
    "is published), max (0.989), and -- the tempting one -- max is what the doc's own "
    "argument list invites by saying 'best per-cell success >= 70%'. The sd claims re-derive "
    "the convention instead of trusting the tables' unqualified 'mean +/- std' header; it "
    "comes out ddof=0 in 11 of the 12 cells and `either` in the zero-variance one. This is "
    "the first chain for the online line, which the four 2026-05 hand passes never covered."
)


def claims() -> list[dict]:
    out = []
    for table in TABLES:
        for objective in OBJECTIVES:
            row = "^\\| `%s` \\|" % objective
            anchor = row + "(?:[^|]*\\|){3}$"
            for i, column in enumerate(COLUMNS):
                # Walk i cells past the row label, then read the cell's own two numbers.
                # `\*{0,2}` because only the first data column is bold, and which column
                # carries the emphasis is typography, not structure.
                skip = "(?:[^|]*\\|){%d}" % i if i else ""
                head = row + skip + " \\*{0,2}"
                for stat, capture in (("mean", head + "([0-9.]+) \u00b1"),
                                      ("sd", head + "[0-9.]+ \u00b1 ([0-9.]+)")):
                    out.append({
                        "label": "A0 %s | %s | %s %s" % (table["title"], objective, column, stat),
                        "anchor": anchor,
                        "after": table["after"],
                        "before": table["before"],
                        "capture": capture,
                        "stat": stat,
                        "sources": ["%s/%s/seed_*/final_eval.json" % (objective, column)],
                        "metric": table["metric"],
                    })
    return out


spec = {
    "chain": "online_a0",
    "doc": "docs/online_rl_line_summary.md",
    "note": NOTE,
    "root": "experiments/protocol_screen_v2/A0_single_u10_cross_tgt15",
    "metric": "eval_success_rate",
    "claims": claims(),
}

io.open(OUT, "w", encoding="utf-8", newline="\n").write(
    json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
print("wrote %s with %d claims" % (OUT, len(spec["claims"])))
