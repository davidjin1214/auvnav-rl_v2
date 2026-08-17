"""Resolve the dispersion convention behind the twelve non-zero +/- readings of section 5.9.

Why this exists
---------------
``ch5_dispersion_audit.py`` classifies a published ``mean +/- sd`` only when the same table row
also prints the per-seed values, so the sd can be recomputed from what the reader can see.  The
SAC-checkpoint ladder table does not print per-seed values, and the FQL P2 results report does not
carry these readings at all -- so the ladder was left as "convention unverified", which blocks the
per-section attribution that the option-C caliber declaration has to state.

This tool closes that gap the same way the section-5.5 A0 check did: it goes back to the per-run
records under ``results/`` and recomputes both conventions.

    sd_population = statistics.pstdev(per-seed success rates)   # numpy default, ddof=0
    sd_sample     = statistics.stdev(per-seed success rates)    # pandas default, ddof=1

Published values are parsed from the ``.tex`` at runtime rather than hard-coded, so a drifted
manuscript shows up as a mean mismatch instead of passing silently.

What "resolved" means here
--------------------------
A reading is resolved only if exactly one convention reproduces the published sd at the published
precision.  Three decimals over three seeds is coarse enough that both conventions can round to the
same string; those readings are reported as ``either`` (the caliber is then not observable from the
number, and for a zero-variance row it is genuinely irrelevant).  Nothing is inferred from the
majority verdict of the other cells -- each reading stands on its own recomputation.

Exit status
-----------
0  every published mean was reproduced (the sd verdicts are printed, not enforced)
1  at least one published mean does not match the per-run records, or a source dir is missing

Usage
-----
    python paper/thesis_ch5/tools/ch5_sac_ladder_dispersion_check.py
"""

from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEX = REPO_ROOT / "paper" / "thesis_ch5" / "sections" / "algo_compare.tex"
H2H = REPO_ROOT / "results" / "offline" / "sac_collector_h2h"

SEEDS = [0, 7, 42]

# Table column order in tab:ch5_algo_sac -> run directory under H2H.
COLUMNS = ["rebrac/b1_1p0", "rebrac/b1_4p0", "fql"]

# Table row label in the .tex -> dataset tier directory name.
ROWS = {
    "随机": "random",
    "中等": "medium",
    "中等偏专家": "mexp",
    "专家": "expert",
}

# The cross-source supplement quoted in the running text, not in the table.
XSOURCE = "xsource_supplement/b1_1p0/m_multi_mix"

PM = re.compile(r"(\d\.\d+)\s*\\pm\s*(\d\.\d+)")


def per_seed_success(rel: str) -> list[float]:
    """Read the final-test success rate of each seed of one cell."""
    rates = []
    for seed in SEEDS:
        path = H2H / rel / f"seed_{seed}" / "test_result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        rates.append(json.loads(path.read_text(encoding="utf-8"))["eval_success_rate"])
    return rates


def verdict(published_sd: str, rates: list[float], digits: int) -> tuple[str, str, str]:
    """Return (population string, sample string, verdict) at the published precision."""
    pop = f"{st.pstdev(rates):.{digits}f}"
    smp = f"{st.stdev(rates):.{digits}f}"
    if pop == published_sd and smp == published_sd:
        return pop, smp, "either"
    if pop == published_sd:
        return pop, smp, "ddof=0"
    if smp == published_sd:
        return pop, smp, "ddof=1"
    return pop, smp, "NEITHER"


def table_rows(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Extract (row label, [(mean, sd), ...]) for each tier row of tab:ch5_algo_sac."""
    out = []
    for label in ROWS:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{label} &") and r"\pm" in stripped:
                out.append((label, PM.findall(stripped)))
                break
        else:
            raise LookupError(f"row not found in {TEX.name}: {label}")
    return out


def main() -> int:
    text = TEX.read_text(encoding="utf-8")
    failures: list[str] = []
    tally: dict[str, int] = {}
    print(f"source: {H2H.relative_to(REPO_ROOT)}  seeds={SEEDS}\n")
    print(f"{'cell':<34} {'published':>15}  {'ddof=0':>7} {'ddof=1':>7}  verdict")
    print("-" * 82)

    readings = []
    for label, pairs in table_rows(text):
        tier = ROWS[label]
        if len(pairs) != len(COLUMNS):
            raise LookupError(f"row {label}: expected {len(COLUMNS)} +/- cells, got {len(pairs)}")
        for column, (mean, sd) in zip(COLUMNS, pairs):
            readings.append((f"{tier}/{column}", f"{column}/{tier}", mean, sd))

    # The running text at L272 quotes a run that is not in the table.
    for match in re.finditer(r"成功率 \$(\d\.\d+)\s*\\pm\s*(\d\.\d+)\$", text):
        readings.append(("m_multi_mix/b1_1p0", XSOURCE, match.group(1), match.group(2)))

    for name, rel, mean, sd in readings:
        digits = len(sd.split(".")[1])
        try:
            rates = per_seed_success(rel)
        except FileNotFoundError as exc:
            failures.append(f"{name}: missing {exc}")
            print(f"{name:<34} {mean + ' +/- ' + sd:>15}  {'--':>7} {'--':>7}  SOURCE MISSING")
            continue
        got_mean = f"{st.fmean(rates):.{len(mean.split('.')[1])}f}"
        pop, smp, v = verdict(sd, rates, digits)
        flag = "" if got_mean == mean else f"  <-- MEAN MISMATCH (records give {got_mean})"
        if flag:
            failures.append(f"{name}: published mean {mean}, records give {got_mean}")
        tally[v] = tally.get(v, 0) + 1
        print(f"{name:<34} {mean + ' +/- ' + sd:>15}  {pop:>7} {smp:>7}  {v}{flag}")
        print(f"{'':<34} per-seed {['%.4f' % r for r in rates]}")

    print("\nverdict tally: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    # The two readings quoted at L270 must be character-identical to their table cells.
    quoted = re.search(r"FQL 达 \$(\d\.\d+ ?\\pm ?\d\.\d+)\$.*?的 \$(\d\.\d+ ?\\pm ?\d\.\d+)\$", text)
    if quoted is None:
        failures.append("the two in-text quotations of the mexp cells were not found")
    else:
        cells = {f"{m} \\pm {s}" for m, s in table_rows(text)[2][1]}
        for group in quoted.groups():
            normalised = re.sub(r"\s*\\pm\s*", r" \\pm ", group)
            status = "matches table" if normalised in cells else "NOT IN TABLE ROW"
            if normalised not in cells:
                failures.append(f"in-text quotation {group!r} is not a mexp table cell")
            print(f"in-text quotation  ${group}$  ->  {status}")

    if failures:
        print("\nFAILURES:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nall published means reproduced from the per-run records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
