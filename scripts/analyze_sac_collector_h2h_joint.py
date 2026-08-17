"""Sprint 1+2 joint paired bootstrap analysis.

Matrix: 4 tier × 3 algo {ReBRAC β1=1, ReBRAC β1=4, FQL} × 3 seed × 30 eval episodes
     = 36 run × 30 ep = 1080 binary success outcomes total.

Per cell unit of analysis: (seed, episode) tuple — episodes are paired across algos
(same manifest, same test_seed=456). Bootstrap: resample 3 seeds w/replacement,
then for each resampled seed resample 30 episodes w/replacement, recompute mean SR
per algo, take paired Δ. N_boot=10000. 95% CI = [2.5, 97.5] percentile.
"""
import json
import numpy as np
from pathlib import Path

REPO = Path("/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/new_off_rl/rl_v2")
TIERS = ["random", "medium", "mexp", "expert"]
SEEDS = [42, 0, 7]
SAC_SRC_SR = {"random": 0.002, "medium": 0.515, "mexp": 0.751, "expert": 0.899}

def load_success_vector(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as f:
        d = json.load(f)
    eps = d["eval_episode_results"]
    return np.array([int(e["success"]) for e in eps], dtype=np.float64)

def gather(algo_path_fn) -> dict[str, np.ndarray]:
    """Return {tier: (n_seeds, n_episodes) success matrix}."""
    out = {}
    for tier in TIERS:
        rows = []
        for seed in SEEDS:
            v = load_success_vector(algo_path_fn(tier, seed))
            rows.append(v)
        # all seeds same n_episodes (30)
        assert all(len(r) == len(rows[0]) for r in rows), tier
        out[tier] = np.stack(rows, axis=0)  # (3, 30)
    return out

# Three algos
rebrac1 = gather(lambda t, s: REPO / f"results/offline/sac_collector_h2h/rebrac/b1_1p0/{t}/seed_{s}/test_result.json")
rebrac4 = gather(lambda t, s: REPO / f"results/offline/sac_collector_h2h/rebrac/b1_4p0/{t}/seed_{s}/test_result.json")
fql     = gather(lambda t, s: REPO / f"results/offline/sac_collector_h2h/fql/{t}/seed_{s}/test_result.json")

ALGOS = {"ReBRAC β1=1": rebrac1, "ReBRAC β1=4": rebrac4, "FQL": fql}

# ---- 1. Per-cell mean ± std (seed-level) ----
print("=" * 92)
print(" Cell-level: mean SR over 30 eps, then μ ± σ across 3 seeds")
print("=" * 92)
hdr = f"{'tier':<8} {'SAC src':>8} | " + " | ".join(f"{a:>16}" for a in ALGOS) + f" | {'ΔFQL−R1':>9} {'ΔR1−R4':>9}"
print(hdr)
print("-" * len(hdr))
for tier in TIERS:
    cells = {a: ALGOS[a][tier] for a in ALGOS}
    seed_means = {a: cells[a].mean(axis=1) for a in ALGOS}  # (3,)
    summary = {a: (seed_means[a].mean(), seed_means[a].std(ddof=1)) for a in ALGOS}
    fql_minus_r1 = seed_means["FQL"].mean() - seed_means["ReBRAC β1=1"].mean()
    r1_minus_r4 = seed_means["ReBRAC β1=1"].mean() - seed_means["ReBRAC β1=4"].mean()
    row = f"{tier:<8} {SAC_SRC_SR[tier]:>8.3f} | "
    row += " | ".join(f"{m:>7.3f} ± {s:.3f}" for (m, s) in (summary[a] for a in ALGOS))
    row += f" | {fql_minus_r1:>+9.3f} {r1_minus_r4:>+9.3f}"
    print(row)

# ---- 2. Stratified paired bootstrap CIs ----
print()
print("=" * 92)
print(" Stratified paired bootstrap (resample seeds w/replacement, then episodes w/replacement)")
print(" 95% CI from N_boot=10000")
print("=" * 92)

def stratified_paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, rng: np.random.Generator = None) -> tuple[float, float, float]:
    """a, b: (n_seeds, n_episodes) success matrices, paired by (seed_idx, ep_idx).
    Resample seeds w/replacement (n_seeds), for each resampled seed resample
    episodes w/replacement (n_episodes), compute Δ = mean(a) - mean(b).
    Returns (mean_delta, ci_lo, ci_hi).
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    n_seeds, n_eps = a.shape
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        seed_ix = rng.integers(0, n_seeds, size=n_seeds)
        ep_ix = rng.integers(0, n_eps, size=(n_seeds, n_eps))
        # Pull resampled cells from a/b using same (seed, ep) — paired
        a_resamp = np.array([a[seed_ix[k]][ep_ix[k]] for k in range(n_seeds)])
        b_resamp = np.array([b[seed_ix[k]][ep_ix[k]] for k in range(n_seeds)])
        deltas[i] = a_resamp.mean() - b_resamp.mean()
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

contrasts = [
    ("FQL", "ReBRAC β1=1"),
    ("FQL", "ReBRAC β1=4"),
    ("ReBRAC β1=1", "ReBRAC β1=4"),
]

rng = np.random.default_rng(2026)

for (algo_a, algo_b) in contrasts:
    print()
    print(f"  Contrast: Δ = SR({algo_a}) − SR({algo_b})")
    print(f"  {'tier':<8} {'Δ mean':>10} {'CI 95%':>22} {'verdict':>40}")
    print(f"  {'-'*8} {'-'*10} {'-'*22} {'-'*40}")
    for tier in TIERS:
        a = ALGOS[algo_a][tier]
        b = ALGOS[algo_b][tier]
        delta_mean, ci_lo, ci_hi = stratified_paired_bootstrap(a, b, n_boot=10000, rng=rng)
        # Verdict
        if abs(a.mean() - b.mean()) < 1e-9 and a.std() < 1e-9 and b.std() < 1e-9:
            v = "DEGENERATE (both 0)"
        elif ci_lo > 0:
            v = f"{algo_a} > {algo_b} (CI strictly above 0)"
        elif ci_hi < 0:
            v = f"{algo_a} < {algo_b} (CI strictly below 0)"
        else:
            v = "inconclusive (CI crosses 0)"
        print(f"  {tier:<8} {delta_mean:>+10.4f} [{ci_lo:>+8.4f}, {ci_hi:>+8.4f}]  {v}")

# ---- 3. Cross-source paired check: SAC mexp vs FQL P2 m_uni_noise ----
# FQL P2 v1.4 m_uni_noise: β1=1 mean 0.x — refer to that doc; here we just report own.
print()
print("=" * 92)
print(" Aggregate paired bootstrap: collapse across all power-bearing cells (medium+mexp+expert)")
print(" (random excluded — degenerate)")
print("=" * 92)
for (algo_a, algo_b) in contrasts:
    pooled_a = np.concatenate([ALGOS[algo_a][t] for t in ("medium", "mexp", "expert")], axis=1)  # (3, 90)
    pooled_b = np.concatenate([ALGOS[algo_b][t] for t in ("medium", "mexp", "expert")], axis=1)
    delta_mean, ci_lo, ci_hi = stratified_paired_bootstrap(pooled_a, pooled_b, n_boot=10000, rng=rng)
    if ci_lo > 0:
        v = f"{algo_a} > {algo_b}"
    elif ci_hi < 0:
        v = f"{algo_a} < {algo_b}"
    else:
        v = "inconclusive"
    print(f"  Δ = SR({algo_a}) − SR({algo_b}): mean {delta_mean:+.4f} CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  {v}")
