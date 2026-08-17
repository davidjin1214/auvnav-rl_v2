"""Classify every ``\\pm`` reading in chapter 5 as population (ddof=0) or sample (ddof=1) sd.

Two tables in the chapter print their per-seed values next to the mean, so those rows settle
their own convention with no external source: recompute both standard deviations from the printed
seeds and see which one the published figure matches. Rows without per-seed values are listed as
unresolved -- they need the ground-truth report, and this script deliberately does not guess.

Nothing is hard-coded: every number comes out of the ``.tex`` sources at run time, so this stays
correct if the chapter is edited.

Usage::

    python paper/thesis_ch5/tools/ch5_dispersion_audit.py
    python paper/thesis_ch5/tools/ch5_dispersion_audit.py --convert ddof1

``--convert`` prints what each self-resolving reading would become under a single convention --
the input to a decision about unifying, not a rewrite of anything.
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from dataclasses import dataclass
from pathlib import Path

SECTIONS = Path(__file__).resolve().parents[1] / "sections"

NUM = r"[0-9]*\.[0-9]+"
PM = re.compile(rf"({NUM})\s*\\pm\s*({NUM})")
BARE = re.compile(NUM)


@dataclass
class Reading:
    file: str
    line: int
    mean: str
    sd: str
    seeds: list[float]

    @property
    def resolved(self) -> bool:
        return len(self.seeds) > 1

    def verdict(self) -> tuple[float, float, str]:
        sd0 = st.pstdev(self.seeds)
        sd1 = st.stdev(self.seeds)
        digits = len(self.sd.split(".")[-1])
        hit0 = f"{sd0:.{digits}f}" == self.sd
        hit1 = f"{sd1:.{digits}f}" == self.sd
        if hit0 and hit1:
            return sd0, sd1, "either"
        if hit0:
            return sd0, sd1, "ddof=0"
        if hit1:
            return sd0, sd1, "ddof=1"
        return sd0, sd1, "NEITHER"


def _cells(line: str) -> list[str]:
    return line.split("&")


def _seeds_before_pm(line: str) -> list[float]:
    """Per-seed cells that precede the mean+/-sd cell (tab:ch5_rebrac_perseed layout)."""
    seeds: list[float] = []
    for cell in _cells(line):
        if "\\pm" in cell:
            return seeds
        seeds.extend(float(x) for x in BARE.findall(cell))
    return []


def _seeds_after_pm(line: str) -> list[float]:
    """Per-seed cells that follow the mean+/-sd cell (tab:ch5_rebrac_screen layout)."""
    _, _, tail = line.partition("\\pm")
    if "&" not in tail:
        return []
    return [float(x) for x in BARE.findall(tail.split("&", 1)[1])]


def collect() -> list[Reading]:
    readings: list[Reading] = []
    for tex in sorted(SECTIONS.glob("*.tex")):
        for lineno, line in enumerate(tex.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue  # rev-header commentary, not published text
            for match in PM.finditer(line):
                mean, sd = match.group(1), match.group(2)
                digits = len(mean.split(".")[-1])
                # The two layouts put the per-seed cells on opposite sides of the mean, and other
                # columns (beta values, seed counts) are decimals too. Accept a candidate only if
                # it reproduces the published mean -- guessing is worse than reporting nothing.
                seeds: list[float] = []
                for candidate in (_seeds_before_pm(line), _seeds_after_pm(line)):
                    if len(candidate) > 1 and f"{st.fmean(candidate):.{digits}f}" == mean:
                        seeds = candidate
                        break
                readings.append(Reading(tex.name, lineno, mean, sd, seeds))
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--convert",
        choices=["ddof0", "ddof1"],
        default=None,
        help="Also print what each resolved reading would become under one convention.",
    )
    args = parser.parse_args()

    readings = collect()
    resolved = [r for r in readings if r.resolved]
    print(f"{len(readings)} pm readings in the chapter body; {len(resolved)} settle themselves "
          f"from per-seed values printed in the same row\n")

    tally: dict[str, int] = {}
    for r in resolved:
        sd0, sd1, tag = r.verdict()
        tally[tag] = tally.get(tag, 0) + 1
        line = (f"  {r.file:18s} L{r.line:<5d} {r.mean} +/- {r.sd}  n={len(r.seeds)}  "
                f"ddof0={sd0:.4f} ddof1={sd1:.4f}  {tag}")
        if args.convert:
            target = sd0 if args.convert == "ddof0" else sd1
            digits = len(r.sd.split(".")[-1])
            now = f"{target:.{digits}f}"
            line += "  ->  " + (f"{r.mean} +/- {now}" if now != r.sd else "unchanged")
        print(line)

    print("\nby convention: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    unresolved: dict[str, int] = {}
    for r in readings:
        if not r.resolved:
            unresolved[r.file] = unresolved.get(r.file, 0) + 1
    print("\nno per-seed values in the row -- convention must come from the ground-truth report:")
    for name, count in sorted(unresolved.items()):
        print(f"  {name:18s} {count}")
    print("\nSwitching convention multiplies every sd in a cell by sqrt(n/(n-1)), so within a table")
    print("of equal n all rankings, ratios and 'smallest/largest' claims are invariant. Only")
    print("comparisons across different n, or between an sd and a non-sd quantity, can move.")


if __name__ == "__main__":
    main()
