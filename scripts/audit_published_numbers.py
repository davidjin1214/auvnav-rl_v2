"""Recompute a report's published numbers from the per-seed JSON they came from.

Four reports were traced by hand, digit by digit, in 2026-05 (`48b8d06` ReBRAC,
`06ec295` td3bc phase0c, `6380082` arrival_v2, `2a8c311` FQL P2). Each pass took a
session, each found something, and each left nothing behind that could be run again --
the next person who doubts a published figure starts the archaeology from zero. This
makes those passes commands.

It also answers the question those passes kept running into and could not close: which
dispersion convention a given `±` is written in. `ch5_dispersion_audit.py` settles that
for the `.tex` rows that print their own per-seed values beside the mean, and says
plainly that the rest "need the ground-truth report". This is that ground truth: the
seeds come out of `results/`, so both conventions can be computed and the published
figure told which one it matches.

    published  0.902 ± 0.021   ddof=0 -> 0.0214    ddof=1 -> 0.0239    verdict ddof=0

What a spec looks like. One JSON per chain under `docs/tracebacks/`, holding the
provenance rule the report states in prose, made executable:

    {
      "chain":  "fql_succession_p2",
      "doc":    "docs/fql_succession_p2_results.md",
      "note":   "provenance rule: the report's own section 9",
      "root":   "results/fql_succession/p2",
      "metric": "eval_success_rate",
      "claims": [
        {"label":  "2x2 - E-uni - ReBRAC",
         "anchor": "^\\\\| \\\\*\\\\*uni\\\\*\\\\*",
         "capture": "R ([0-9.]+)",
         "stat":   "mean",
         "sources": ["e_uni/test/rebrac_seed*.json"]}
      ]
    }

`anchor` must match exactly one line of the doc -- a locator that survives the document
moving, and fails loudly when the sentence is rewritten, which is the moment the claim
needs re-checking anyway. `capture` reads the published figure off that line, so no
figure is transcribed into the spec: transcribe it and the spec keeps passing after the
doc changes, which is the one failure this tool exists to prevent.

Optional `section` / `after` / `before` narrow the search, for the case where two tables
in the same report have rows that are identical line by line and only what sits above
them says which is which. `section` takes a heading and runs to the next heading of the
same or higher level; `after` and `before` are resolved inside it. Each must itself
resolve to exactly one line, so a scope that has gone ambiguous is an error rather than
a silent pick.

`metric` may be a dotted path, for a readout that nests its figures rather than
keeping them flat: `best.eval_success_rate` reaches into `selected_checkpoint.json`'s
record of the checkpoint it chose, which is the only place the worldcomp screening tables
can be recomputed from. The flat key is tried first, so a readout whose key legitimately
contains a dot still wins.

A source glob may name `.csv` files instead of JSON, for the online-line runs whose
published figures are aggregates over a training run's periodic evaluations rather than
one terminal number. `metric` then names the reduction and the column, and the file
contributes that one number:

    mean(eval_success_rate)              the report's `mean39`
    max(eval_success_rate)               `peak`
    argmax(eval_success_rate, env_step)  the step it was first attained at
    count_gt(eval_success_rate, 0)       `n_succ`
    nrows(eval_success_rate)             how many evaluations there were

`scale` divides the recomputed value before comparison, and is for unit conversion only
-- `peak @ 475k` is a report writing 475002 steps in thousands. Anything else it could
be used for is fudging a figure into agreement, which is the opposite of the point.

Statistics, over the metric read from every JSON a source glob matches:

    mean   sd0 (population)   sd1 (sample)   n
    sd     both, and the report says which one the published figure matches
    seeds  the per-seed values themselves, compared as a multiset -- reports print them
           in whatever order reads well, and `docs/fql_succession_p2_results.md` does
    delta  mean(sources[1]) - mean(sources[0])

A published figure passes when it is a correct rounding of the recomputed one: the gap
must be within half a unit of its own last printed place. Exact string equality would
fail honest rows -- the FQL report prints -0.027 for an exact -0.0275, and which way
that rounds is a property of the formatter that produced it, not a defect.

`"expect": "mismatch"` inverts that, for a figure a report knowingly keeps and annotates
as wrong. `arrival_v2_experiment_report.md` §7.9.4 leaves a superseded 2026-05-19 column
standing and says underneath which two cells are mis-numbered; a plain claim on those
cells would sit red forever, and a red that is supposed to be there gets ignored. As an
erratum claim the same cell asserts the disclosure instead: it must keep failing to
reproduce, and the day it starts reproducing, the note above it needs re-reading. A
`note` saying which disclosure is being pinned is required.

`results/` is gitignored, so a clone has nothing to recompute from. Missing data is
reported as `no-data` and does not fail: point `--root` at a Drive mount to check it.
`--require-data` turns absence into a failure, for a machine that is supposed to have it.

Usage:
    python -m scripts.audit_published_numbers                     # every spec
    python -m scripts.audit_published_numbers --chain fql_succession_p2
    python -m scripts.audit_published_numbers --strict            # exit 1 on any defect
    python -m scripts.audit_published_numbers --ddof              # only the +/- verdicts
    python -m scripts.audit_published_numbers --root /mnt/drive/results/...
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_DIR = os.path.join(ROOT, "docs", "tracebacks")

STATS = ("mean", "sd0", "sd1", "sd", "seeds", "n", "delta")
EXPECTS = ("match", "mismatch")
SPEC_KEYS = {"chain", "doc", "note", "root", "metric", "claims"}
CLAIM_KEYS = {"label", "anchor", "capture", "stat", "sources", "metric", "note",
              "section", "after", "before", "scale", "expect"}

HEADING = re.compile(r"^(#{1,6}) ")

# `mean(col)`, `argmax(col, other)`, `count_gt(col, 0)` -- a reduction over one CSV.
AGG = re.compile(r"^([a-z_]+)\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,\s*([^,()\s]+)\s*)?\)$")

# Defects fail --strict. `no-data` does not: a clone legitimately has no results/.
DEFECT_BUCKETS = ("spec-error", "anchor-missing", "anchor-ambiguous",
                  "capture-failed", "value-mismatch", "erratum-stale")

NUM = re.compile(r"[-+−–]?[0-9]*\.?[0-9]+")

# These reports are typeset prose: a minus sign in them is U+2212, and an editor has
# turned some of them into en dashes. `float()` accepts none of the three.
DASHES = str.maketrans({"−": "-", "–": "-", "—": "-"})


def _norm(text: str) -> str:
    """A published figure as typeset -> something `float()` accepts.

    The comma is a thousands separator here, not a decimal point: transition counts are
    the only six-figure numbers these documents print and they print them `152,683`.
    Nothing in this corpus writes a decimal comma, and a claim whose capture straddled two
    numbers would already be caught by the space-stripping that has always been here.
    """
    return text.translate(DASHES).replace(" ", "").replace(",", "")


class SpecError(Exception):
    pass


def load_specs(spec_dir: str, chain: str | None = None) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(spec_dir, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            spec = json.load(fh)
        try:
            spec["_path"] = os.path.relpath(path, ROOT).replace("\\", "/")
        except ValueError:
            # Windows: relpath raises across drive letters, and a spec dir passed on
            # the command line routinely sits on another one.
            spec["_path"] = path.replace("\\", "/")
        if chain and spec.get("chain") != chain:
            continue
        out.append(spec)
    return out


def validate(spec: dict) -> None:
    """Reject a spec this tool would otherwise half-run.

    An unrecognised key is an error rather than something to ignore: a misspelt `stat`
    or `sources` would leave the claim silently unchecked, which looks exactly like a
    claim that passed.
    """
    unknown = set(spec) - SPEC_KEYS - {"_path"}
    if unknown:
        raise SpecError(f"unknown spec keys: {sorted(unknown)}")
    for key in ("chain", "doc", "root", "metric", "claims"):
        if key not in spec:
            raise SpecError(f"missing spec key: {key}")
    for i, claim in enumerate(spec["claims"]):
        unknown = set(claim) - CLAIM_KEYS
        if unknown:
            raise SpecError(f"claim {i} ({claim.get('label')}): unknown keys {sorted(unknown)}")
        for key in ("label", "anchor", "capture", "stat", "sources"):
            if key not in claim:
                raise SpecError(f"claim {i} ({claim.get('label')}): missing {key}")
        if claim["stat"] not in STATS:
            raise SpecError(f"claim {i}: stat {claim['stat']!r} not one of {list(STATS)}")
        if claim["stat"] == "delta" and len(claim["sources"]) != 2:
            raise SpecError(f"claim {i}: stat delta needs exactly two sources")
        if claim["stat"] != "delta" and len(claim["sources"]) != 1:
            raise SpecError(f"claim {i}: stat {claim['stat']} takes one source")
        if re.compile(claim["capture"]).groups != 1:
            raise SpecError(f"claim {i}: capture must have exactly one group")
        if claim.get("expect", "match") not in EXPECTS:
            raise SpecError(f"claim {i}: expect {claim['expect']!r} not one of {list(EXPECTS)}")
        if claim.get("expect") == "mismatch" and not claim.get("note"):
            raise SpecError(f"claim {i}: an erratum claim needs a note naming the "
                            "disclosure it pins")
        if "scale" in claim and not (isinstance(claim["scale"], (int, float))
                                     and not isinstance(claim["scale"], bool)
                                     and claim["scale"] > 0):
            raise SpecError(f"claim {i}: scale must be a positive number")


def read_metric(paths: list[str], metric: str) -> list[float]:
    """One number per file: a JSON key, or a reduction over a CSV column."""
    agg = AGG.match(metric)
    values = []
    for path in sorted(paths):
        is_csv = path.lower().endswith(".csv")
        if is_csv != bool(agg):
            want = ("an aggregation such as mean(eval_success_rate)" if is_csv
                    else "a plain JSON key")
            raise SpecError(f"{path}: metric {metric!r} does not fit this source; "
                            f"needs {want}")
        values.append(_reduce_csv(path, agg) if is_csv else _read_json(path, metric))
    return values


def _read_json(path: str, metric: str) -> float:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if metric in payload:
        return float(payload[metric])
    if "." not in metric:
        raise SpecError(f"{path}: no key {metric!r}")
    # A dotted path, for readouts that nest their figures. `selected_checkpoint.json`
    # keeps the chosen checkpoint's own metrics under `best`, which is the only place the
    # worldcomp screening tables can be recomputed from. Tried only after the flat key
    # misses, so a readout whose key legitimately contains a dot still wins.
    node = payload
    for part in metric.split("."):
        if not isinstance(node, dict) or part not in node:
            raise SpecError(f"{path}: no key {metric!r}")
        node = node[part]
    return float(node)


def _reduce_csv(path: str, agg: re.Match) -> float:
    how, column, arg = agg.group(1), agg.group(2), agg.group(3)
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for needed in (column, arg if how == "argmax" else None):
        if needed and (not rows or needed not in rows[0]):
            raise SpecError(f"{path}: no column {needed!r}")
    if not rows:
        raise SpecError(f"{path}: no rows")
    series = [float(r[column]) for r in rows]

    if how == "mean":
        return st.fmean(series)
    if how == "max":
        return max(series)
    if how == "nrows":
        return float(len(series))
    if how == "argmax":
        if arg is None:
            raise SpecError(f"argmax needs the column to report: argmax({column}, env_step)")
        # First attainment: a report's "peak @ 475k" is when the run got there, and a
        # plateau makes every later row tie.
        return float(rows[series.index(max(series))][arg])
    if how == "count_gt":
        if arg is None:
            raise SpecError(f"count_gt needs a threshold: count_gt({column}, 0)")
        try:
            threshold = float(arg)
        except ValueError:
            raise SpecError(f"count_gt threshold {arg!r} is not a number") from None
        return float(sum(1 for v in series if v > threshold))
    raise SpecError(f"unknown aggregation {how!r} in metric")


def decimals(text: str) -> int:
    return len(text.split(".")[-1]) if "." in text else 0


def rounds_to(value: float, printed_raw: str) -> bool:
    """Is `printed` a correct rounding of `value` at its own precision?

    Half a unit in the last place, inclusive. The boundary is left open to both
    directions on purpose -- -0.0275 printed as -0.027 and as -0.028 are both honest,
    and which one a report shows says nothing except which formatter wrote it.
    """
    printed = _norm(printed_raw)
    try:
        target = float(printed)
    except ValueError:
        return False
    half = 0.5 * 10 ** (-decimals(printed))
    return abs(value - target) <= half + 1e-12


def evaluate(claim: dict, published: str, values: list[list[float]]) -> tuple[str, str, str]:
    """(verdict, recomputed-as-shown, detail). verdict is "ok", "mismatch" or a note."""
    stat = claim["stat"]
    if stat == "delta":
        got = st.fmean(values[1]) - st.fmean(values[0])
    elif stat == "n":
        got = float(len(values[0]))
    elif stat == "mean":
        got = st.fmean(values[0])
    elif stat == "sd0":
        got = st.pstdev(values[0])
    elif stat == "sd1":
        got = st.stdev(values[0])
    elif stat == "sd":
        return _sd_verdict(published, values[0])
    elif stat == "seeds":
        return _seeds_verdict(published, values[0])
    else:                                     # unreachable; validate() gates this
        raise SpecError(f"unhandled stat {stat!r}")
    shown = f"{got:.{max(decimals(published), 4)}f}"
    ok = rounds_to(got, published)
    return ("ok" if ok else "mismatch"), shown, ""


def _sd_verdict(published: str, values: list[float]) -> tuple[str, str, str]:
    if len(values) < 2:
        return "mismatch", "n/a", "need at least two seeds for a dispersion"
    sd0, sd1 = st.pstdev(values), st.stdev(values)
    hit0, hit1 = rounds_to(sd0, published), rounds_to(sd1, published)
    detail = f"ddof=0 {sd0:.4f} | ddof=1 {sd1:.4f}"
    if hit0 and hit1:
        return "ok", "either", detail
    if hit0:
        return "ok", "ddof=0", detail
    if hit1:
        return "ok", "ddof=1", detail
    return "mismatch", "NEITHER", detail


def _seeds_verdict(published: str, values: list[float]) -> tuple[str, str, str]:
    """Per-seed lists are compared as multisets; reports order them for reading."""
    printed = NUM.findall(published)
    if len(printed) != len(values):
        return "mismatch", ", ".join(f"{v:g}" for v in sorted(values)), \
            f"published {len(printed)} values, source has {len(values)}"
    unmatched = list(values)
    for text in printed:
        hit = next((v for v in unmatched if rounds_to(v, text)), None)
        if hit is None:
            return "mismatch", ", ".join(f"{v:g}" for v in sorted(values)), \
                f"no source value rounds to {text}"
        unmatched.remove(hit)
    return "ok", ", ".join(f"{v:g}" for v in sorted(values)), ""


def _region(doc_lines: list[str], claim: dict) -> tuple[int, int]:
    """Line bounds, exclusive, that `anchor` is searched within.

    A report can hold two tables whose rows are indistinguishable line by line -- the
    ReBRAC screening grids repeat `| **4.0** | **2.0** |` under one dataset heading and
    then the other. Nothing in the row settles which grid it is; the heading above it
    does. `after`/`before` name those headings, and each must resolve to exactly one
    line, so a scope that has itself gone ambiguous is an error and not a silent pick.
    """
    lo, hi = 0, len(doc_lines) + 1
    if "section" in claim:
        lo, hi = _section(doc_lines, claim["section"])
    for key in ("after", "before"):
        pattern = claim.get(key)
        if pattern is None:
            continue
        hits = [i for i, ln in enumerate(doc_lines, 1)
                if lo < i < hi and re.search(pattern, ln)]
        if len(hits) != 1:
            raise SpecError(
                f"{key} {pattern!r} matches {len(hits)} lines in scope, need exactly 1")
        if key == "after":
            lo = hits[0]
        else:
            hi = hits[0]
    if lo >= hi:
        raise SpecError(f"empty scope: lower bound {lo}, upper bound {hi}")
    return lo, hi


def _section(doc_lines: list[str], pattern: str) -> tuple[int, int]:
    """A heading's body: from the heading to the next one of the same or higher level.

    Markdown levels do the bounding, so a section stays itself when subsections are
    added under it. `after`/`before` are then resolved inside -- which is what makes
    them usable at all in a report whose bold dataset markers repeat under every
    section that tabulates the same two datasets.
    """
    hits = [(i, HEADING.match(ln)) for i, ln in enumerate(doc_lines, 1)
            if re.search(pattern, ln)]
    if len(hits) != 1:
        raise SpecError(f"section {pattern!r} matches {len(hits)} lines, need exactly 1")
    start, head = hits[0]
    if head is None:
        raise SpecError(f"section {pattern!r} matched line {start}, which is no heading")
    level = len(head.group(1))
    for i, ln in enumerate(doc_lines[start:], start + 1):
        nxt = HEADING.match(ln)
        if nxt and len(nxt.group(1)) <= level:
            return start, i
    return start, len(doc_lines) + 1


def run_spec(spec: dict, data_root: str | None, require_data: bool) -> dict[str, list]:
    buckets: dict[str, list] = {b: [] for b in DEFECT_BUCKETS}
    buckets["no-data"] = []
    buckets["ok"] = []

    try:
        validate(spec)
    except SpecError as exc:
        buckets["spec-error"].append((spec.get("chain", spec["_path"]), str(exc)))
        return buckets

    doc_path = os.path.join(ROOT, spec["doc"])
    if not os.path.isfile(doc_path):
        buckets["spec-error"].append((spec["chain"], f"doc not found: {spec['doc']}"))
        return buckets
    with open(doc_path, encoding="utf-8") as fh:
        doc_lines = fh.read().split("\n")

    base = data_root or os.path.join(ROOT, spec["root"])

    for claim in spec["claims"]:
        where = f"{spec['chain']} :: {claim['label']}"
        try:
            lo, hi = _region(doc_lines, claim)
        except SpecError as exc:
            buckets["spec-error"].append((where, str(exc)))
            continue
        hits = [(i, ln) for i, ln in enumerate(doc_lines, 1)
                if lo < i < hi and re.search(claim["anchor"], ln)]
        if not hits:
            buckets["anchor-missing"].append((where, claim["anchor"]))
            continue
        if len(hits) > 1:
            buckets["anchor-ambiguous"].append(
                (where, claim["anchor"], [i for i, _ in hits]))
            continue
        lineno, line = hits[0]
        m = re.search(claim["capture"], line)
        if not m:
            buckets["capture-failed"].append((where, claim["capture"], line.strip()[:110]))
            continue
        published = m.group(1)

        groups, missing = [], []
        for pattern in claim["sources"]:
            paths = sorted(glob.glob(os.path.join(base, pattern)))
            if not paths:
                missing.append(pattern)
            groups.append(paths)
        if missing:
            buckets["no-data"].append((where, published, missing))
            continue

        metric = claim.get("metric", spec["metric"])
        scale = claim.get("scale", 1)
        try:
            values = [[v / scale for v in read_metric(paths, metric)] for paths in groups]
            verdict, shown, detail = evaluate(claim, published, values)
        except (SpecError, ValueError, OSError) as exc:
            buckets["spec-error"].append((where, str(exc)))
            continue

        expect = claim.get("expect", "match")
        reproduced = verdict == "ok"
        row = (where, f"{spec['doc']}:{lineno}", claim["stat"], published, shown, detail,
               expect)
        if expect == "mismatch":
            # The claim is the disclosure, not the figure: it holds while the figure
            # stays unreproducible. Reproducing it means the doc changed under the note.
            buckets["ok" if not reproduced else "erratum-stale"].append(row)
        else:
            buckets["ok" if reproduced else "value-mismatch"].append(row)

    if require_data and buckets["no-data"]:
        buckets["spec-error"].extend(
            (where, f"--require-data: no files match {miss}")
            for where, _pub, miss in buckets["no-data"])
        buckets["no-data"].clear()
    return buckets


def merge(all_buckets: list[dict[str, list]]) -> dict[str, list]:
    out: dict[str, list] = {}
    for buckets in all_buckets:
        for name, rows in buckets.items():
            out.setdefault(name, []).extend(rows)
    return out


LABELS = {
    "spec-error": "★ 溯源表本身有问题（键名/统计量/数据键写错）",
    "anchor-missing": "★ 锚定不到（文档已改写，该刊值需重新定位并复核）",
    "anchor-ambiguous": "★ 锚定到多行（锚太松，说不清核的是哪一处）",
    "capture-failed": "★ 锚到了行、取不出数（capture 与该行对不上）",
    "value-mismatch": "★ 刊值与复算不符",
    "erratum-stale": "★ 已声明的勘误现在复算得出来了（文档改过？那条勘误注需重读）",
    "no-data": "缺数据（results/ 未入库；用 --root 指到 Drive 挂载）",
    "ok": "复算吻合",
}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description="recompute published report numbers from their per-seed JSON")
    ap.add_argument("--chain", help="run only this chain")
    ap.add_argument("--root", help="data root, overriding each spec's own")
    ap.add_argument("--spec-dir", default=SPEC_DIR)
    ap.add_argument("--strict", action="store_true", help="exit 1 on any defect")
    ap.add_argument("--require-data", action="store_true",
                    help="treat missing results/ as a failure rather than a skip")
    ap.add_argument("--ddof", action="store_true",
                    help="print only the dispersion verdicts")
    ap.add_argument("--all", action="store_true", help="list matching readings too")
    args = ap.parse_args()

    specs = load_specs(args.spec_dir, args.chain)
    if not specs:
        print(f"没有溯源表可跑（--spec-dir {args.spec_dir}"
              + (f"，--chain {args.chain}" if args.chain else "") + "）")
        return 1
    buckets = merge([run_spec(s, args.root, args.require_data) for s in specs])

    if args.ddof:
        rows = [r for b in ("ok", "value-mismatch", "erratum-stale")
                for r in buckets[b] if r[2] == "sd"]
        print(f"± 口径判定：{len(rows)} 处")
        print("=" * 96)
        for _where, at, _stat, published, shown, detail, expect in sorted(
                rows, key=lambda r: r[1]):
            flag = "  ⟨已声明的勘误，本就不该复现⟩" if expect == "mismatch" else ""
            print(f"  {at}  ± {published}  -> {shown}    {detail}{flag}")
        return 1 if (args.strict and (buckets["value-mismatch"]
                                      or buckets["erratum-stale"])) else 0

    defects = sum(len(buckets[b]) for b in DEFECT_BUCKETS)
    print(f"溯源表 {len(specs)} 份；核对 {sum(len(v) for v in buckets.values())} 处刊值："
          f"吻合 {len(buckets['ok'])}，缺陷 {defects}，缺数据 {len(buckets['no-data'])}")
    print("=" * 96)

    for name in DEFECT_BUCKETS + ("no-data",):
        rows = buckets[name]
        print(f"\n--- {LABELS[name]}：{len(rows)} ---")
        for row in rows:
            if name in ("value-mismatch", "erratum-stale"):
                where, at, stat, published, shown, detail, _expect = row
                tail = f"  ｜{detail}" if detail else ""
                print(f"  {where}\n        {at}  刊 {stat}={published}  复算 {shown}{tail}")
            elif name == "no-data":
                where, published, missing = row
                print(f"  {where}  刊值 {published}  ｜无匹配文件 {', '.join(missing)}")
            elif name == "anchor-ambiguous":
                where, anchor, lines = row
                print(f"  {where}  ｜{anchor}  命中行 {lines}")
            else:
                print("  " + "  ｜".join(str(x) for x in row))

    if args.all:
        print(f"\n--- {LABELS['ok']}：{len(buckets['ok'])} ---")
        for where, at, stat, published, shown, detail, expect in buckets["ok"]:
            tail = f"  ｜{detail}" if detail else ""
            mark = " [勘误已确认不可复现]" if expect == "mismatch" else ""
            print(f"  {at}  {stat}={published}  复算 {shown}{tail}{mark}  ({where})")

    print("\n⚠ 本脚本只验「刊值能不能由它自称的源重算出来」。那个源是不是**该用**的源，"
          "是人的活——溯源表里的 provenance 规则就是那句人话，改动它要有依据。")
    return 1 if (args.strict and defects) else 0


if __name__ == "__main__":
    raise SystemExit(main())
