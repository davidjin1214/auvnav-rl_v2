"""Recompute the item-(1) contamination-magnitude readout from the clean-manifest probe.

Item (1) of ``docs/data_integrity_open_items.md``: the ``crosscomp-2000`` dataset's training
episodes are the final-evaluation episodes.  The magnitude of the resulting inflation is measured
by re-evaluating the *already selected* checkpoints on a fresh manifest (seed 3000) that is
disjoint from both the training range 0..1999 and the published test set 1250..1349.

What the readout separates
--------------------------
Re-evaluating on a fresh manifest moves three things at once.  Only the third is item (1):

1. **Instance-draw difficulty.**  A new 100-episode draw is not equally hard.  Measured directly
   by the collector policy, which is analytic -- it never trained on anything and never underwent
   checkpoint selection, so its displacement is draw difficulty alone.
2. **Selection generalisation.**  Checkpoints were picked on ``val_40``, drawn from the
   seed-1250 instance family.  A fresh family costs something.  This is a item-(3) effect and it
   hits *both* cells; note the ``hold60`` split in ``ch5_holdout_split_audit.py`` structurally
   cannot see it, since hold60 stays inside the seed-1250 family.
3. **Item (1) itself**, which by construction can only touch the 2000-episode cell.

Because 1 and 2 are common-mode across the two cells, the item-(1) component is the
difference-in-differences: how much the *flip* (2000 minus 1000) shrinks when moving to the clean
manifest.  That is the number the disposition turns on -- not the per-cell displacement.

Inputs are read from ``results/`` (gitignored); the published per-seed values are re-read rather
than hard-coded so that a drifted ground truth shows up as a mismatch instead of passing silently.

Usage
-----
    python paper/thesis_ch5/tools/ch5_clean_probe_readout.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SEEDS = [42, 43, 44, 45, 46]

FORMAL = "results/offline/rebrac/formal"
DS_1000 = "crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000"
DS_2000 = "crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000"
PAIR = "actorb_4p0__criticb_2p0"

PUBLISHED_DIRS = {
    "cross-1000": f"{FORMAL}/{DS_1000}/{PAIR}/test",
    "cross-2000": f"{FORMAL}/{DS_2000}/{PAIR}/test",
}
CLEAN_DIRS = {
    "cross-1000": "results/offline/rebrac/clean_probe/cross-1000",
    "cross-2000": "results/offline/rebrac/clean_probe/cross-2000",
}
BASELINES = {
    "published (1250..1349)": "results/offline/rebrac/clean_probe/baselines/crosscomp_published.json",
    "clean (3000..3099)": "results/offline/rebrac/clean_probe/baselines/crosscomp_clean_s3000.json",
}

# Frozen so the tool fails loudly if the readouts it is pointed at are not the ones this
# analysis was written against.
EXPECTED = {
    ("published", "cross-1000"): 0.902,
    ("published", "cross-2000"): 0.918,
    ("clean", "cross-1000"): 0.862,
    ("clean", "cross-2000"): 0.870,
}

T_CRIT_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}


def _success_rates(dir_path: str) -> list[float]:
    out = []
    for seed in SEEDS:
        path = REPO_ROOT / dir_path / f"seed_{seed}.json"
        if not path.is_file():
            raise SystemExit(f"missing readout: {path}")
        out.append(float(json.loads(path.read_text(encoding="utf-8"))["eval_success_rate"]))
    return out


def _paired(diffs: list[float]) -> tuple[float, float, float, float]:
    """Mean, t, and 95% CI bounds for a one-sample (paired) test."""
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t_crit = T_CRIT_95[n]
    if se == 0.0:
        return mean, float("nan"), mean, mean
    return mean, mean / se, mean - t_crit * se, mean + t_crit * se


def main() -> None:
    published = {label: _success_rates(d) for label, d in PUBLISHED_DIRS.items()}
    clean = {label: _success_rates(d) for label, d in CLEAN_DIRS.items()}

    for tag, table in (("published", published), ("clean", clean)):
        for label, values in table.items():
            got = sum(values) / len(values)
            want = EXPECTED[(tag, label)]
            if abs(got - want) > 5e-4:
                raise SystemExit(f"{tag} {label}: mean {got:.4f} != expected {want:.4f}")

    print("=" * 78)
    print("per-cell success rate (mean +/- population sd, matching the chapter's convention)")
    print("=" * 78)
    for label in PUBLISHED_DIRS:
        for tag, table in (("published 1250..1349", published), ("clean     3000..3099", clean)):
            values = table[label]
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
            print(f"  {label:11s} {tag}  {mean:.4f} +/- {sd:.4f}   per-seed {[round(v, 3) for v in values]}")
        diffs = [p - c for p, c in zip(published[label], clean[label])]
        mean, t, lo, hi = _paired(diffs)
        print(f"  {label:11s} displacement  {100 * mean:+.2f} pp  t={t:+.3f}  "
              f"95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  per-seed {[round(v, 3) for v in diffs]}")
        print()

    print("=" * 78)
    print("the flip (2000 minus 1000), on each manifest")
    print("=" * 78)
    flips = {}
    for tag, table in (("published", published), ("clean    ", clean)):
        diffs = [a - b for a, b in zip(table["cross-2000"], table["cross-1000"])]
        flips[tag.strip()] = diffs
        mean, t, lo, hi = _paired(diffs)
        print(f"  {tag}  {100 * mean:+.2f} pp  t={t:+.3f}  95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  "
              f"non-negative {sum(d >= 0 for d in diffs)}/{len(diffs)}")

    print()
    print("=" * 78)
    print("difference-in-differences = the item-(1)-attributable inflation of the flip")
    print("=" * 78)
    did = [p - c for p, c in zip(flips["published"], flips["clean"])]
    mean, t, lo, hi = _paired(did)
    print(f"  {100 * mean:+.2f} pp  t={t:+.3f}  95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  "
          f"per-seed {[round(v, 3) for v in did]}")

    print()
    print("=" * 78)
    print("collector baseline -- analytic policy, no training and no checkpoint selection,")
    print("so its displacement is instance-draw difficulty alone")
    print("=" * 78)
    rates = {}
    for tag, rel in BASELINES.items():
        path = REPO_ROOT / rel
        if not path.is_file():
            print(f"  {tag:24s} missing ({rel})")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rates[tag] = float(payload["eval_success_rate"])
        print(f"  {tag:24s} {rates[tag]:.3f}  (n={int(payload['num_eval_episodes'])})")
    if len(rates) == 2:
        (_, a), (_, b) = rates.items()
        # Two independent proportions at n=100 each.
        se = math.sqrt(a * (1 - a) / 100 + b * (1 - b) / 100)
        print(f"  draw-difficulty offset   {100 * (a - b):+.1f} pp   "
              f"(SE ~ {100 * se:.1f} pp at n=100 -- indicative, not significant)")
        print()
        print("  Caveat: the collector sits at a much higher success level than the agents, and")
        print("  difficulty offsets are known not to transfer cleanly across competence levels")
        print("  (see the sel40/hold60 offsets in ch5_holdout_split_audit.py, which differ in sign")
        print("  by policy). Treat this as bounding the draw component as small, not as a subtraction.")


if __name__ == "__main__":
    main()
