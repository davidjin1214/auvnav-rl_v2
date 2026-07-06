"""
paper/thesis_ch5/figures/scripts/extract_ch5_rebrac_termination_stats.py

Termination-type composition for the SS5.7 mainline units (analysis helper, no
figure output). Prints, per experiment unit, the pooled episode counts by
termination reason across the formal 5-seed x test=100 protocol, plus the
per-seed breakdown for the hard seed (seed 44), so the SS5.7.m failure-mode
sentence can be hand-filled from a reproducible extraction rather than from
memory.

Data (REAL, local synced results; same runs as docs/rebrac_experiment_report.md
rev.8 SS7.4/SS7.10/SS7.12 -- do not edit from memory):
  crosscomp-1000 formal : results/offline/rebrac/formal/
      crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/
      actorb_4p0__criticb_2p0/test/seed_{42..46}.json
  crosscomp-2000 formal : .../crosscomp_..._ep2000/actorb_4p0__criticb_2p0/test/
  worldcomp deployable  : results/offline/rebrac/worldcomp_teacher_gap/
      deployable/worldcomp_..._ep1000/actorb_4p0__criticb_2p0/test/

NB: the worldcomp privileged-critic unit is NOT included -- its rev.8 5-seed
test jsons (seeds 45/46) are not synced to this machine (only 42/43/44 local),
and the SS5.7.m failure-composition sentence only cites the three mainline
units above.

Usage:
    python paper/thesis_ch5/figures/scripts/extract_ch5_rebrac_termination_stats.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

SEEDS = (42, 43, 44, 45, 46)

UNITS = {
    "crosscomp-1000 (4.0, 2.0)": (
        "results/offline/rebrac/formal/"
        "crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/"
        "actorb_4p0__criticb_2p0/test"
    ),
    "crosscomp-2000 (4.0, 2.0)": (
        "results/offline/rebrac/formal/"
        "crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000/"
        "actorb_4p0__criticb_2p0/test"
    ),
    "worldcomp deployable (4.0, 2.0)": (
        "results/offline/rebrac/worldcomp_teacher_gap/deployable/"
        "worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/"
        "actorb_4p0__criticb_2p0/test"
    ),
}


def unit_stats(test_dir: Path) -> None:
    pooled: Counter[str] = Counter()
    pooled_fail: Counter[str] = Counter()
    per_seed: dict[int, Counter[str]] = {}
    for seed in SEEDS:
        path = test_dir / f"seed_{seed}.json"
        data = json.loads(path.read_text())
        episodes = data["eval_episode_results"]
        assert len(episodes) == 100, (path, len(episodes))
        reasons = Counter(ep["reason"] for ep in episodes)
        fail_reasons = Counter(
            ep["reason"] for ep in episodes if not ep["success"]
        )
        # sanity: reported success rate matches episode-level count
        n_succ = sum(1 for ep in episodes if ep["success"])
        assert abs(n_succ / 100 - data["eval_success_rate"]) < 1e-9, path
        pooled.update(reasons)
        pooled_fail.update(fail_reasons)
        per_seed[seed] = fail_reasons
    total = sum(pooled.values())
    print(f"  pooled over {len(SEEDS)} seeds x 100 eps = {total} eps")
    print(f"    all reasons     : {dict(pooled)}")
    print(f"    failure reasons : {dict(pooled_fail)} "
          f"(n_fail={sum(pooled_fail.values())})")
    for seed in SEEDS:
        fails = per_seed[seed]
        print(f"    seed {seed}: n_fail={sum(fails.values()):3d}  "
              f"{dict(fails)}")


def main() -> None:
    for name, rel in UNITS.items():
        print(f"\n=== {name} ===")
        unit_stats(REPO / rel)


if __name__ == "__main__":
    main()
