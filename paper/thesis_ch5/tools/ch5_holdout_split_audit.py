"""
paper/thesis_ch5/tools/ch5_holdout_split_audit.py

Selection-set vs held-out split audit for the Chapter 5 offline units (review helper,
no figure output).

Why this exists
---------------
docs/data_integrity_open_items.md item (3): checkpoint selection ran on a 40-episode
validation manifest whose task instances are the *first 40* of the 100-episode test
manifest (both generated from benchmark key ``single_u10_cross_tgt15``, manifest_seed
1250, no seed offset). So every reported 100-episode success rate mixes 40 episodes
that took part in checkpoint selection with 60 that did not.

The per-episode records survive locally (``eval_episode_results`` in every
``test/seed_*.json``, each entry carrying its manifest ``seed``), so the split can be
recomputed at zero compute cost:

    seeds 1250..1289 -> "sel40"   (used for checkpoint / alpha selection)
    seeds 1290..1349 -> "hold60"  (never seen by any selection rule)

hold60 is the closest thing to a genuinely held-out readout that exists without
re-running anything. Two caveats it cannot remove:

* it does NOT address item (1): for the crosscomp-2000 cells all 100 evaluation
  instances are inside the training episode range (base_seed 0, 2000 episodes), so
  hold60 is selection-free but still instance-contaminated;
* sel40 vs hold60 also differs in intrinsic difficulty, so the gap between them is
  an upper bound on selection optimism, not a measurement of it.

Usage:
    python paper/thesis_ch5/tools/ch5_holdout_split_audit.py
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

SEL_MAX = 1289  # validation manifest = manifest_seed 1250 .. 1250+39
HOLD_MIN = 1290

UNITS: dict[str, str] = {
    # SS5.6 data-scale axis (TD3+BC, posterior-selected alpha_BC)
    "TD3+BC cross-500 (a=0.5)": "results/offline/td3bc/phase0c/stage_c_final/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone/test_selected/alpha_0p5",
    "TD3+BC cross-1000 (a=0.25)": "results/offline/td3bc/phase0c/stage_c_final/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/test_selected/alpha_0p25",
    "TD3+BC cross-2000 (a=0.15)": "results/offline/td3bc/phase0c/stage_c_final/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000/test_selected/alpha_0p15",
    # SS5.6 same-budget pure behaviour cloning
    "pureBC cross-500": "results/offline/td3bc/phase0c/stage_c_bc_final/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone/test_bc_selected/alpha_0p0",
    "pureBC cross-1000": "results/offline/td3bc/phase0c/stage_c_bc_final/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/test_bc_selected/alpha_0p0",
    "pureBC cross-2000": "results/offline/td3bc/phase0c/stage_c_bc_final/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000/test_bc_selected/alpha_0p0",
    # SS5.6 worldcomp deployable / privileged (TD3+BC)
    "TD3+BC world dep (a->0)": "results/offline/td3bc/phase0c/worldcomp_teacher_gap/deployable_final/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone/test_selected/alpha_0p0",
    "TD3+BC world priv (a=0.1)": "results/offline/td3bc/phase0c/worldcomp_teacher_gap/privileged_final/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone/test_selected/alpha_0p1",
    # SS5.7 ReBRAC-Q mainline + ablations
    "ReBRAC cross-1000 (4,2)": "results/offline/rebrac/formal/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/test",
    "ReBRAC cross-2000 (4,2)": "results/offline/rebrac/formal/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000/actorb_4p0__criticb_2p0/test",
    "ReBRAC cross-2000 (4,1) backup": "results/offline/rebrac/formal/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep2000/actorb_4p0__criticb_1p0/test",
    "ReBRAC world dep (4,2)": "results/offline/rebrac/worldcomp_teacher_gap/deployable/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/test",
    "ReBRAC world priv (4,2) [3 seeds local]": "results/offline/rebrac/worldcomp_teacher_gap/privileged_critic/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/test",
    "ReBRAC cross-1000 b2=0": "results/offline/rebrac/stage_e_critic_penalty_off/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_0p0/test",
    "ReBRAC cross-1000 LN-off": "results/offline/rebrac/critic_ln_off/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_2p0/test",
    "ReBRAC world b2=0 probe": "results/offline/rebrac/worldcomp_critic_penalty_off_probe/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/actorb_4p0__criticb_0p0/test",
    # collector reference on the same fixed evaluation set
    "collector crosscomp (ref)": "results/offline/td3bc/phase0c/stage_c_final/crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/baselines",
    "collector worldcomp (ref)": "results/offline/td3bc/phase0c/worldcomp_teacher_gap/deployable_final/worldcomp_s0_h4_efficiency_v2_re150_u10cross_fixdone/baselines",
}


def _files(unit_dir: Path) -> list[Path]:
    out = sorted(unit_dir.glob("seed_*.json"))
    if not out:
        out = sorted(unit_dir.glob("*_eval.json"))
    return out


def unit_split(unit_dir: Path) -> dict[str, object] | None:
    files = _files(unit_dir)
    if not files:
        return None
    full, sel, hold = [], [], []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes = payload.get("eval_episode_results")
        if not episodes:
            continue
        s = [e for e in episodes if e["seed"] <= SEL_MAX]
        h = [e for e in episodes if e["seed"] >= HOLD_MIN]
        if not s or not h:
            continue
        full.append(payload["eval_success_rate"])
        sel.append(sum(bool(e["success"]) for e in s) / len(s))
        hold.append(sum(bool(e["success"]) for e in h) / len(h))
    if not full:
        return None
    return {"n_seeds": len(full), "full": full, "sel": sel, "hold": hold}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def std_pop(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def std_sample(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_t(a: list[float], b: list[float]) -> tuple[float, float, float, float]:
    """Return (mean diff, t, half-width of 95% CI, sd of diffs) for paired samples."""
    d = [x - y for x, y in zip(a, b)]
    m, n = mean(d), len(d)
    sd = std_sample(d)
    se = sd / math.sqrt(n)
    crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}[n]
    return m, (m / se if se else float("nan")), crit * se, sd


def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
    va, vb = std_sample(a) ** 2 / len(a), std_sample(b) ** 2 / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    return mean(a) - mean(b), (mean(a) - mean(b)) / se, df


def main() -> None:
    results: dict[str, dict[str, object]] = {}
    print(f"{'unit':42s} {'n':>2s} {'full100':>8s} {'sel40':>7s} {'hold60':>7s} {'sel-hold':>9s}")
    print("-" * 82)
    for name, rel in UNITS.items():
        got = unit_split(REPO / rel)
        if got is None:
            print(f"{name:42s} -- not synced locally")
            continue
        results[name] = got
        f, s, h = got["full"], got["sel"], got["hold"]  # type: ignore[assignment]
        print(
            f"{name:42s} {got['n_seeds']:2d} "
            f"{mean(f):8.4f} {mean(s):7.4f} {mean(h):7.4f} {mean(s) - mean(h):+9.4f}"
        )

    print("\n--- SS5.7.1 the flip: cross-2000 vs cross-1000 (ReBRAC-Q, paired by seed) ---")
    for label, key in (("full 100", "full"), ("hold-out 60", "hold")):
        a = results["ReBRAC cross-2000 (4,2)"][key]  # type: ignore[index]
        b = results["ReBRAC cross-1000 (4,2)"][key]  # type: ignore[index]
        m, t, hw, _ = paired_t(a, b)  # type: ignore[arg-type]
        print(
            f"  {label:12s} 2000={mean(a):.4f} 1000={mean(b):.4f} diff={m:+.4f} "
            f"paired t={t:+.3f} 95%CI=[{m - hw:+.4f},{m + hw:+.4f}] "
            f"per-seed diffs={[round(x - y, 4) for x, y in zip(a, b)]}"  # type: ignore[arg-type]
        )

    print("\n--- SS5.7.1 baseline gaps (ReBRAC-Q minus TD3+BC, same dataset) ---")
    for ds, rk, tk in (
        ("cross-1000", "ReBRAC cross-1000 (4,2)", "TD3+BC cross-1000 (a=0.25)"),
        ("cross-2000", "ReBRAC cross-2000 (4,2)", "TD3+BC cross-2000 (a=0.15)"),
    ):
        for label, key in (("full 100", "full"), ("hold-out 60", "hold")):
            r = mean(results[rk][key])  # type: ignore[arg-type]
            t = mean(results[tk][key])  # type: ignore[arg-type]
            print(f"  {ds} {label:12s} ReBRAC={r:.4f} TD3+BC={t:.4f} gap={100 * (r - t):+.1f} pp")

    print("\n--- SS5.6.3 non-monotonicity (peak at 1000, fall at 2000) ---")
    for fam in ("TD3+BC", "pureBC"):
        for label, key in (("full 100", "full"), ("hold-out 60", "hold")):
            vals = []
            for size in ("500", "1000", "2000"):
                k = next(k for k in results if k.startswith(fam) and f"-{size}" in k)
                vals.append(mean(results[k][key]))  # type: ignore[arg-type]
            print(
                f"  {fam:8s} {label:12s} 500={vals[0]:.4f} 1000={vals[1]:.4f} 2000={vals[2]:.4f} "
                f"| 2000-1000={100 * (vals[2] - vals[1]):+.1f} pp"
            )

    print("\n--- SS5.7.2 deployable ReBRAC-Q vs privileged TD3+BC (worldcomp) ---")
    for label, key in (("full 100", "full"), ("hold-out 60", "hold")):
        dep = results["ReBRAC world dep (4,2)"][key]  # type: ignore[index]
        priv = results["TD3+BC world priv (a=0.1)"][key]  # type: ignore[index]
        base = mean(results["TD3+BC world dep (a->0)"][key])  # type: ignore[arg-type]
        ref = mean(results["collector worldcomp (ref)"][key])  # type: ignore[arg-type]
        diff, t, df = welch(dep, priv)  # type: ignore[arg-type]
        denom = ref - base
        print(
            f"  {label:12s} dep={mean(dep):.4f} priv={mean(priv):.4f} pureBC={base:.4f} "
            f"collector={ref:.4f}\n"
            f"               Welch diff={diff:+.4f} t={t:+.4f} df={df:.3f} | "
            f"closure dep={100 * (mean(dep) - base) / denom:.1f}% priv={100 * (mean(priv) - base) / denom:.1f}%"
        )

    print("\n--- per-seed hold-out 60 values (for hand-checking) ---")
    for name, got in results.items():
        print(f"  {name:42s} {[round(x, 4) for x in got['hold']]}")  # type: ignore[index]


if __name__ == "__main__":
    main()
