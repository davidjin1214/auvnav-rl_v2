"""Cross-source paired bootstrap: m_multi_mix supplement vs FQL P2 v1.4.

Three contrasts (all on shared 100-ep manifest single_u10_cross_tgt15_ep100, eval_seed=456):
  1. β1=1_supp vs β1=4_FQLP2: cross-source replicate of sprint 1 mexp finding
  2. β1=1_supp vs FQL_FQLP2:  cross-source vs sprint 2 mexp finding
  3. β1=4_FQLP2 vs FQL_FQLP2: reproduce FQL P2 v1.4 original m_multi_mix Δ (sanity)

Paired bootstrap unit: (seed_idx, episode_idx) — same eval_seed=456 + same manifest
means episodes match across (seed, algo). Bootstrap: resample seeds w/replacement (n=2),
then for each resampled seed resample 100 episodes w/replacement. N_boot=10000.
"""
import json
import numpy as np
from pathlib import Path

REPO = Path("/Users/xiangjin/Library/CloudStorage/OneDrive-Personal/我的/Code/new_off_rl/rl_v2")

def load_success_vector(path: Path) -> np.ndarray:
    d = json.loads(path.read_text())
    eps = d["eval_episode_results"]
    return np.array([int(e["success"]) for e in eps], dtype=np.float64)

# ---- Load matrices ----
PAIRED_SEEDS = [42, 0]
ALL_SUPP_SEEDS = [42, 0, 7]

# Supplement: β1=1.0 m_multi_mix
supp_paired = np.stack([
    load_success_vector(REPO / f"results/offline/sac_collector_h2h/xsource_supplement/b1_1p0/m_multi_mix/seed_{s}/test_result.json")
    for s in PAIRED_SEEDS
], axis=0)  # (2, 100)

supp_all = np.stack([
    load_success_vector(REPO / f"results/offline/sac_collector_h2h/xsource_supplement/b1_1p0/m_multi_mix/seed_{s}/test_result.json")
    for s in ALL_SUPP_SEEDS
], axis=0)  # (3, 100)

# FQL P2 v1.4: ReBRAC β1=4 + FQL on m_multi_mix
fqlp2_rebrac_b1_4 = np.stack([
    load_success_vector(REPO / f"results/fql_succession/p2/m_multi_mix/test/rebrac_seed{s}.json")
    for s in PAIRED_SEEDS
], axis=0)  # (2, 100)

fqlp2_fql = np.stack([
    load_success_vector(REPO / f"results/fql_succession/p2/m_multi_mix/test/fql_seed{s}.json")
    for s in PAIRED_SEEDS
], axis=0)  # (2, 100)

print(f"Supplement β1=1 (paired seeds [42, 0]): per-seed SR = {supp_paired.mean(axis=1).tolist()}, all 3 = {supp_all.mean(axis=1).tolist()}")
print(f"FQL P2 ReBRAC β1=4 (seeds [42, 0]):     per-seed SR = {fqlp2_rebrac_b1_4.mean(axis=1).tolist()}")
print(f"FQL P2 FQL (seeds [42, 0]):             per-seed SR = {fqlp2_fql.mean(axis=1).tolist()}")
print()

def stratified_paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, rng: np.random.Generator = None):
    """a, b: (n_seeds, n_episodes) paired binary success matrices.
    Resample seeds w/replacement, then episodes w/replacement, compute Δ = mean(a) - mean(b).
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    n_seeds, n_eps = a.shape
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        seed_ix = rng.integers(0, n_seeds, size=n_seeds)
        ep_ix = rng.integers(0, n_eps, size=(n_seeds, n_eps))
        a_resamp = np.array([a[seed_ix[k]][ep_ix[k]] for k in range(n_seeds)])
        b_resamp = np.array([b[seed_ix[k]][ep_ix[k]] for k in range(n_seeds)])
        deltas[i] = a_resamp.mean() - b_resamp.mean()
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))

# ---- Contrasts ----
contrasts = [
    ("β1=1 supp", supp_paired, "β1=4 FQL P2", fqlp2_rebrac_b1_4,
     "cross-source replicate of sprint 1 mexp finding (β1=1 > β1=4)"),
    ("β1=1 supp", supp_paired, "FQL FQL P2", fqlp2_fql,
     "cross-source vs sprint 2 mexp finding (FQL > ReBRAC β1=1)"),
    ("β1=4 FQL P2", fqlp2_rebrac_b1_4, "FQL FQL P2", fqlp2_fql,
     "reproduce FQL P2 v1.4 m_multi_mix Δ=-0.035 GRAY (sanity)"),
]

rng = np.random.default_rng(2026)

print("=" * 92)
print(" Cross-source paired bootstrap (100 ep × 2 paired seeds, N_boot=10000)")
print("=" * 92)
for (na, a, nb, b, note) in contrasts:
    delta_mean, ci_lo, ci_hi = stratified_paired_bootstrap(a, b, n_boot=10000, rng=rng)
    if ci_lo > 0:
        v = f"{na} > {nb} (CI strictly above 0)"
    elif ci_hi < 0:
        v = f"{na} < {nb} (CI strictly below 0)"
    else:
        v = "inconclusive (CI crosses 0)"
    print(f"\n  Δ = SR({na}) − SR({nb})")
    print(f"    {note}")
    print(f"    Δ mean = {delta_mean:+.4f}   CI 95% [{ci_lo:+.4f}, {ci_hi:+.4f}]   →  {v}")

# ---- Power check: own-source supplement variance ----
print()
print("=" * 92)
print(" Own-source supplement n=3 own bootstrap (informational, not paired)")
print("=" * 92)
n_boot = 10000
own_means = np.empty(n_boot)
for i in range(n_boot):
    sx = rng.integers(0, 3, size=3)
    ex = rng.integers(0, 100, size=(3, 100))
    own_means[i] = np.mean([supp_all[sx[k]][ex[k]] for k in range(3)])
own_ci_lo, own_ci_hi = float(np.percentile(own_means, 2.5)), float(np.percentile(own_means, 97.5))
print(f"  supplement β1=1 m_multi_mix mean SR = {supp_all.mean():.4f}  CI 95% [{own_ci_lo:.4f}, {own_ci_hi:.4f}]")

# ---- Cross-cell consistency check: m_multi_mix β1=1 vs sprint 1 mexp β1=1 (own-source) ----
# Sprint 1 mexp β1=1 used 30-ep manifest, m_multi_mix β1=1 uses 100-ep manifest. NOT paired.
# Just informational: data quality regime gap
sprint1_mexp_b1_1 = np.stack([
    load_success_vector(REPO / f"results/offline/sac_collector_h2h/rebrac/b1_1p0/mexp/seed_{s}/test_result.json")
    for s in (42, 0, 7)
], axis=0)
print()
print(f"  sprint 1 mexp β1=1 SR (30 ep × 3 seed):  {sprint1_mexp_b1_1.mean():.4f}")
print(f"  supplement m_multi_mix β1=1 SR (100 ep × 3 seed): {supp_all.mean():.4f}")
print(f"  gap: {(supp_all.mean() - sprint1_mexp_b1_1.mean()):+.4f}  (NOT paired — different eval manifest)")
