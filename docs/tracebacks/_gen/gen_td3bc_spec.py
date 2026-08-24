"""One-off: emit docs/tracebacks/td3bc_phase0c.json.

Hand-typing 60 positional capture regexes is how a spec ends up quietly checking the
wrong column. The JSON it writes is the artifact; this generator is disposable.

Usage:
    python docs/tracebacks/_gen/gen_td3bc_spec.py [out_dir]

Edit this file, never the JSON it writes: `test_every_committed_spec_can_be_regenerated`
reruns it and compares byte for byte, so a hand edit to the spec shows up as a failure.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Optional output directory, so the regression test in tests/test_audit_published_numbers.py
# can regenerate into a scratch tree instead of overwriting the committed specs.
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "tracebacks"

ROOT = "results/offline/td3bc/phase0c"
FIXDONE = "crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone"

METRICS = [
    ("success rate", "eval_success_rate"),
    ("return", "eval_return"),
    ("safety cost", "eval_safety_cost"),
    ("time", "eval_time_s"),
    ("path efficiency", "eval_path_efficiency"),
]

# The report holds a Stage B table whose first two columns are identical to Stage C's
# (`| 500 | 0.5 |`), so the size+alpha prefix alone anchors to both -- the sweep refused
# to guess, which is how this was found. The tail counts cells instead: Stage C rows
# carry six more, Stage B four. Structural, so it does not go stale when a value does.
TAIL6 = r"(?:[^|]*\|){6}$"
MAIN = [                       # (label, anchor prefix, dataset dir, alpha dir)
    ("500", r"^\| 500 \| 0\.5 \|", FIXDONE, "alpha_0p5"),
    ("1000", r"^\| 1000 \| 0\.25 \|", FIXDONE + "_ep1000", "alpha_0p25"),
    ("2000", r"^\| 2000 \| 0\.15 \|", FIXDONE + "_ep2000", "alpha_0p15"),
]

# Anchoring the BC rows on their own published value would mean a changed value shows up
# as "anchor missing" rather than as the mismatch it is. Anchor on the column's *shape*:
# in this table the cell after the size is a mean +/- sd, in Stage B it is a bare alpha.
BC = [                         # (label, capture prefix, dataset dir)
    ("500", r"^\| 500 \|", FIXDONE),
    ("1000", r"^\| 1000 \|", FIXDONE + "_ep1000"),
    ("2000", r"^\| 2000 \|", FIXDONE + "_ep2000"),
]
BC_SHAPE = r" [0-9.]+ ± [0-9.]+ \|"

claims = []


def skip(n: int) -> str:
    """Consume n whole table cells after the anchor."""
    return r"[^|]+\|" * n


def add(label, anchor, capture, stat, sources, metric=None):
    claim = {"label": label, "anchor": anchor, "capture": capture,
             "stat": stat, "sources": sources}
    if metric:
        claim["metric"] = metric
    re.compile(capture)                                  # fail here, not at audit time
    claims.append(claim)


# --- stage_c_final: five metrics x (mean, dispersion) x three dataset sizes ----------
for size, prefix, dataset, alpha in MAIN:
    anchor = prefix + TAIL6
    src = [f"stage_c_final/{dataset}/test_selected/{alpha}/seed_*.json"]
    for k, (col, metric) in enumerate(METRICS):
        add(f"stage_c_final | {size} | {col} mean", anchor,
            prefix + skip(k) + r" (-?[0-9.]+) ±", "mean", src, metric)
        add(f"stage_c_final | {size} | {col} ±", anchor,
            prefix + skip(k) + r" -?[0-9.]+ ± ([0-9.]+) ", "sd", src, metric)
    add(f"stage_c_final | {size} | baseline success", anchor,
        prefix + skip(len(METRICS)) + r" ([0-9.]+) \|", "mean",
        [f"stage_c_final/{FIXDONE}/baselines/crosscomp_test_eval.json"],
        "eval_success_rate")

# --- stage_c_bc_final: BC success / return beside the TD3BC ones ---------------------
for size, prefix, dataset in BC:
    anchor = prefix + BC_SHAPE
    src = [f"stage_c_bc_final/{dataset}/test_bc_selected/alpha_0p0/seed_*.json"]
    add(f"stage_c_bc_final | {size} | BC success mean", anchor,
        prefix + r" ([0-9.]+) ±", "mean", src, "eval_success_rate")
    add(f"stage_c_bc_final | {size} | BC success ±", anchor,
        prefix + r" [0-9.]+ ± ([0-9.]+) ", "sd", src, "eval_success_rate")
    add(f"stage_c_bc_final | {size} | BC return mean", anchor,
        prefix + skip(3) + r" (-[0-9.]+) ±", "mean", src, "eval_return")
    add(f"stage_c_bc_final | {size} | BC return ±", anchor,
        prefix + skip(3) + r" -[0-9.]+ ± ([0-9.]+) ", "sd", src, "eval_return")

# --- stage_c_shadow_ep2000_a0p2: the alpha=0.2 shadow row ---------------------------
shadow_anchor = r"^\| `2000, alpha=0\.2` \|"
shadow_src = [f"stage_c_shadow_ep2000_a0p2/{FIXDONE}_ep2000/test_selected/alpha_0p2/seed_*.json"]
add("shadow a0.2 | success mean", shadow_anchor,
    shadow_anchor + r" ([0-9.]+) \|", "mean", shadow_src, "eval_success_rate")
add("shadow a0.2 | success ±", shadow_anchor,
    shadow_anchor + r" [0-9.]+ \| ([0-9.]+) \|", "sd", shadow_src, "eval_success_rate")
for k, (col, metric) in enumerate(METRICS[1:]):
    add(f"shadow a0.2 | {col}", shadow_anchor,
        shadow_anchor + skip(2 + k) + r" (-?[0-9.]+) \|", "mean", shadow_src, metric)

# the finalist row printed in the same table, from the stage_c_final tree
final_anchor = r"^\| `2000, alpha=0\.15` \|"
final_src = [f"stage_c_final/{FIXDONE}_ep2000/test_selected/alpha_0p15/seed_*.json"]
add("shadow table | a0.15 | success mean", final_anchor,
    final_anchor + r" ([0-9.]+) \|", "mean", final_src, "eval_success_rate")
add("shadow table | a0.15 | success ±", final_anchor,
    final_anchor + r" [0-9.]+ \| ([0-9.]+) \|", "sd", final_src, "eval_success_rate")
for k, (col, metric) in enumerate(METRICS[1:]):
    add(f"shadow table | a0.15 | {col}", final_anchor,
        final_anchor + skip(2 + k) + r" (-?[0-9.]+) \|", "mean", final_src, metric)

spec = {
    "chain": "td3bc_phase0c",
    "doc": "docs/td3bc_phase0c_experiment_report.md",
    "note": ("Provenance rule is the report's own dispersion footnote under the "
             "stage_c_final table: the per-seed terminal evaluations under "
             "test_selected/alpha_*/seed_4{2..6}.json, with the +/- taken at ddof=0. "
             "Traced by hand once in 06ec295 (2026-08-17); this makes that repeatable, "
             "and the `sd` claims re-derive the convention rather than trusting it."),
    "root": ROOT,
    "metric": "eval_success_rate",
    "claims": claims,
}

out = OUT_DIR / "td3bc_phase0c.json"
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(spec, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print(f"wrote {out}: {len(claims)} claims")
