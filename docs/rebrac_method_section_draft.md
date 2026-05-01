# Method Section Draft — Q-normalized dual-penalty TD3+BC variant

> 文档版本：rev.1（由 `notebooks/rebrac_paper_followup.ipynb` §5 自动生成）
> 配套：[docs/rebrac_mainline_review.md §2.2.1 / §3.1.D](./rebrac_mainline_review.md)
> 用途：直接拷入 paper method section，并放入 implementation note。

## 1. Algorithm name and positioning

We refer to our offline RL algorithm as a **Q-normalized dual-penalty TD3+BC variant** (alias: ReBRAC-Q).
It implements the minimal recipe of ReBRAC (Tarasov et al., 2023)—dual BC penalty applied symmetrically to actor and critic plus critic LayerNorm—on top of the TD3+BC actor loss form (Fujimoto and Gu, 2021), in which the deterministic policy gradient is normalized by `|Q|.detach()`.

This variant is **not** identical to the original ReBRAC. The numerical value of `β_1` and `β_2` reported in this paper is therefore not directly comparable to the values in Tarasov et al. (2023).

## 2. Actor loss

For batch `(s, a)` drawn from the offline buffer:

- Let `π_θ(s)` be the deterministic actor and `Q_φ(s, a)` be the twin-critic minimum.
- Define the Q-normalization scalar (detached, per batch):

    λ = 1 / mean( |Q_φ(s, π_θ(s))| ).detach()
    (clamped at a numerical floor of 1e-6)

- The actor loss is:

    L_actor(θ) = − λ · E[ Q_φ(s, π_θ(s)) ] + β_1 · E[ ‖π_θ(s) − a‖² ]

The critic loss is the standard ReBRAC critic loss:

    L_critic(φ) = E[ (Q_φ(s, a) − y(s, a, r, s'))² ] + β_2 · E[ ‖a' − π_θ̄(s')‖² ]

where `a' ~ π_θ̄(s') + clipped noise` is the next-action target used for both bootstrapping and the critic-side BC penalty (TD3-style target smoothing applied to the actor target).

## 3. Difference from original ReBRAC

| Component | Original ReBRAC (Tarasov et al., 2023) | This work |
|---|---|---|
| Actor `Q` term scaling | `−E[Q]` (no normalization) | `−(1/|Q|.detach()) · E[Q]` |
| Actor BC penalty | `β_1 · E[‖π − a‖²]` | identical |
| Critic loss | TD3 + `β_2 · E[‖a' − π̄(s')‖²]` | identical |
| Critic LayerNorm | on by default | on (winner) |
| Policy update freq | every 2 critic updates | identical |
| Target smoothing | clipped Gaussian on next action | identical |

## 4. Difference from TD3+BC (Fujimoto and Gu, 2021)

Original TD3+BC actor loss:

    L_actor^{TD3+BC}(θ) = − λ_TD3+BC · E[ Q_φ(s, π_θ(s)) ] + E[ ‖π_θ(s) − a‖² ],
    with  λ_TD3+BC = α_TD3+BC / mean(|Q_φ|).detach()

Multiplying our actor loss by `1 / β_1` gives the equivalent normalized form:

    L_actor / β_1 = − (1 / (β_1 · |Q|.detach())) · E[ Q ] + E[ ‖π − a‖² ]

By matching coefficients, the BC anchoring strength of `β_1` in this work corresponds to TD3+BC `α ≈ 1 / β_1`. With `β_1 = 4.0` (winner), this matches `α ≈ 0.25`, which is also the α used by the TD3+BC baselines on `crosscomp` (phase0c).

The critical addition over TD3+BC is the **critic-side BC penalty** `β_2 · E[‖a' − π̄(s')‖²]`, which is the defining ReBRAC modification.

## 5. Implementation note (recommended for the paper)

> The actor loss in our implementation includes the TD3+BC-style Q normalization (`λ = 1/|Q|.detach()`) on the deterministic policy gradient term. The reported value `β_1 = 4.0` is the absolute coefficient of the BC penalty term and corresponds to the strongest anchoring level in our hyperparameter grid `{1.0, 2.0, 4.0}`. Numerical comparison of `β_1` and `β_2` against Tarasov et al. (2023) is not direct because of this scaling difference; we therefore restrict numerical hyperparameter comparisons to the TD3+BC baselines we trained ourselves under matched protocols.

## 6. Why this variant (justification, optional discussion bullet)

- Q normalization (TD3+BC) was retained because it removes a free hyperparameter (Q magnitude) that varies across `worldcomp` (mean Q ≈ +15) and `crosscomp` (mean Q ≈ −8), letting a single `β_1` work across datasets.
- Dual penalty (ReBRAC) was added because the critic-penalty-off probes (`β_2 = 0`) showed `mean_target_q` drift +46% / +98% across worldcomp and crosscomp, and the seed-44 outlier collapsed by −17pp in `crosscomp` without `β_2`.
- The combination lets us report a **single configuration** that is dataset-invariant across `worldcomp-1000`, `crosscomp-1000`, and `crosscomp-2000`, which would not be achievable with the original (un-normalized) ReBRAC actor loss without per-dataset retuning.
