# Step 3 — Dynamics Consistency Audit (`auv_nav/vehicle.py` vs `remus100_core.py`)

**Date:** 2026-05-13
**Goal:** answer the user's question — *"are the AUV motion-model files in `phnode_full_oc_clean/` and `auv_nav/vehicle.py` consistent?"* — at the level of detail needed to decide whether to swap one for the other.
**Method:** read both files end-to-end, compare module-by-module, and run a small numerical sanity script to put real numbers on each divergence.
**Scope:** physics only (mass, hydrodynamics, propulsion, actuators, current model). Controllers (`Remus100Controller` vs `IntegralSMCHeading + DepthController + ALOS2D`) are not compared here — they live above the dynamics layer and the project already uses `auv_nav` controllers, not the reference's.

---

## TL;DR

**No.** The two files are not consistent. They are two independent Fossen-style REMUS-100 implementations that agree on the **headline constants** (ρ, L, D, fin areas, propeller KT/KQ, time-of-surge/sway/yaw, ζ_roll/pitch) but diverge on **at least 7 physics formulas / parameters**, four of which are large enough to produce visible trajectory drift over a few seconds. The dominant divergences are:

| # | Topic | reference (NODE training) | downstream (`vehicle.py`) | Magnitude |
|---|---|---|---|---|
| 1 | `geometry_scale` | 1.0 (implicit) | **1.0096** | mass +2.91%, Ix/Iy +4.89% |
| 2 | Actuator τ (fin) | T_delta = **0.10 s** | delta_tau = **0.25 s** | δ response **2.5× slower** in downstream |
| 3 | Cross-flow drag Cd_2D | **constant ≈ 1.198** (lookup at fixed B/(2T)=0.5) | **Re- and AR-dependent** 0.25–0.80 | ratio **1.5×** (sub-critical) → **4.7×** (super-critical peak) |
| 4 | Oswald efficiency *e* | **0.7** | **0.3** | downstream induced drag **2.33×** higher at given α |
| 5 | Propeller inflow speed | `Va = 0.944 · |nu_r[0]|` (axial relative) | `Va = 0.944 · ‖nu[:3]‖` (3-D body speed) | identical at β≈α≈0; diverges under sideslip |
| 6 | Current model frame | inertial `v_c^n`, full 3-D `R^T` to body | planar yaw-only projection of body components | identical at φ=θ=0 (planar); divergent otherwise |
| 7 | Actuator integrator | u_actual ∈ state, RK4 jointly with ν_r | separate explicit Euler + rate limit (35°/s, 400 RPM/s) | shape of fin step response different even if τ matched |

Sub-percent items (gravity 0.12%, propeller Ja-branch on reverse RPM — never triggered because `n_rpm_min=200 > 0`) are noted but not load-bearing.

The full-3D Coriolis matrices (CRB, CA) are **mathematically equivalent** despite different code paths; we verified the algebra in §6.

---

## 1. Headline constants — match ✓

ρ=1026, L=1.6, D=0.19, r_bg=[0,0,0.02], r_bb=[0,0,0], S_fin=0.00665, A_r=A_s=2·S_fin, CL_δr=0.5, CL_δs=0.7, D_prop=0.14, t_prop=0.10, KT_0=0.4566, KQ_0=0.0700, KT_max=0.1798, KQ_max=0.0312, Ja_max=0.6632, body Cd=0.42, MA_44/Ix=0.3, T_surge=T_sway=20 s, T_yaw=1 s, ζ_roll=0.3, ζ_pitch=0.8. All identical.

---

## 2. Geometry — `geometry_scale=1.0096` (downstream extra; **high**)

[`auv_nav/vehicle.py`](../../auv_nav/vehicle.py):178 sets `geometry_scale: float = 1.0096`, applied in `_refresh_derived()`:

```python
self.a = p.geometry_scale * p.length / 2.0      # 0.80768  (ref: 0.8)
self.b = p.geometry_scale * p.diameter / 2.0    # 0.09591  (ref: 0.095)
```

Reference [`remus100_core.py`](../../phnode_full_oc_clean/reference_simulator/remus100_core.py):115 has no such factor.

| Quantity | scale=1.0 | scale=1.0096 | Δ |
|---|---:|---:|---:|
| a | 0.80000 | 0.80768 | +0.96% |
| b | 0.09500 | 0.09591 | +0.96% |
| m | 31.0294 kg | 31.9316 kg | **+2.91%** |
| Ix | 0.11202 | 0.11750 | **+4.89%** |
| Iy / Iz | 4.02777 | 4.22485 | **+4.89%** |

Knock-on effects: W=B scale as m → W+B both +2.91%; D[0..2, 0..2] base values (= M_ii/T_i) scale +2.91%; D[3,3], D[4,4] (≈ M_ii · 2ζ · √(W·dz/M_ii)) scale roughly +1.5%; CD_0 unchanged (b²/S ratio invariant).

Origin of the 1.0096 factor: not commented in `vehicle.py`. Likely an old calibration to a wave-tank measurement. **For NODE consistency it must be removed (or set to 1.0).**

---

## 3. Gravity — 0.12% (negligible)

| | reference | downstream |
|---|---|---|
| `g` | 9.81 m/s² (hardcoded) | Somigliana @ lat 63.4468°N → **9.82178 m/s²** |
| Δ | — | +0.120% |

Effect: W = m·g, B = W, both +0.12%. Below numerical noise of the other divergences. Could be hard-set to 9.81 in `Remus100` ctor for parity if desired.

---

## 4. Actuator dynamics — different parameters **and** different integrators (**high**)

### 4.1 Time constants

| | T_delta (fin) | T_n (RPM) |
|---|---|---|
| reference (`remus100_core.py`:153) | 0.10 s | 1.00 s |
| downstream (`vehicle.py`:185-186 `ActuatorConfig`) | **0.25 s** | **0.80 s** |
| ratio down/ref | **2.5× slower** | **0.8× (20% faster)** |

The NODE config (`config.json`) declares `T_actuator_init = (0.1, 0.1, 1.0)` and trains τ as **learnable** — i.e. NODE may have moved off 0.1/1.0 during training. Until we unpack `best_model.pt` weights, we cannot tell whether the learned τ ended up closer to 0.10 or 0.25.

### 4.2 Integration

**Reference** [`remus100_core.py`](../../phnode_full_oc_clean/reference_simulator/remus100_core.py):231-272:
```python
# u_actual is part of the 16-D state, integrated jointly with nu_r:
u_dot = np.array([
    (delta_r_c - delta_r) / self.T_delta,
    (delta_s_c - delta_s) / self.T_delta,
    (n_c - n)              / self.T_n,
])
# ...passed to RK4 via _state_derivative
```
Continuous first-order lag, RK4 within the same `dt_sim=0.01` step as ν_r.

**Downstream** [`auv_nav/vehicle.py`](../../auv_nav/vehicle.py):560-589 `step_actuators()`:
```python
rate = (target - current) / max(tau, 1e-6)
rate = clip(rate, -rate_limit, +rate_limit)      # 35°/s for δ, 400 RPM/s for n
return current + dt * rate
```
Explicit forward Euler, called separately *outside* the dynamics RK4, with **additional rate limits** that reference does not have. Saturation to ±15° / ±1525 RPM at the end.

**Implication:** even if T_delta were forced to match (0.10 s), the *shape* of the fin step response differs because (a) downstream is explicit Euler not RK4, (b) downstream rate-limits the slew. For a 15° step command, reference reaches 63% in 0.10 s smoothly; downstream is rate-limited to 35°/s → linear ramp for the first 0.43 s, then catches up.

---

## 5. Cross-flow drag Cd_2D — fundamentally different formulas (**high**)

### 5.1 Reference

[`remus100_core.py`](../../phnode_full_oc_clean/reference_simulator/remus100_core.py):341-358:
```python
x = np.array([0.0109, 0.1766, ..., 4.0031])       # B/(2T) abscissa
y = np.array([1.9661, 1.9657, ..., 0.5593])       # Cd_2D ordinate
Cd_2D = np.interp(self.diam / (2 * self.diam), x, y)
#                  └────────────────────────────┘
#                          = 0.5 always
```

This evaluates to **`np.interp(0.5, x, y) ≈ 1.198`** — **a constant**, completely independent of vehicle state (Reynolds, AR, cross-flow speed). The comment "B/(2T) for cylinder" suggests it was *meant* to be a per-strip aspect ratio lookup, but with `diam / (2·diam) = 0.5` the strip aspect collapses to a fixed point on the curve. (This may even be a latent bug in the reference, but it is the form NODE was trained against.)

### 5.2 Downstream

[`auv_nav/vehicle.py`](../../auv_nav/vehicle.py):253-261, `_cylinder_drag_coeff()`:
```python
reynolds = u_cross * length * 1e6
cd       = np.interp(reynolds, _CD_DATA_DNV[:, 0], _CD_DATA_DNV[:, 1])  # 28-pt DNV cylinder Cd(Re)
aspect_ratio = length / diameter                                         # 8.42
kappa    = np.interp(AR, _KAPPA_SUB_or_SUPER[:, 0], [:, 1])              # 0.66 sub / 0.81 super
return cd * kappa
```
Full Re- and AR-dependent formulation with sub/super-critical branch at Re=2·10⁵.

### 5.3 Numerical comparison

Hull is L=1.6 m, D=0.19 m, AR=8.42. Cross-flow speed `u_cross = |v_r + x·r|` typically 0.05–1.5 m/s in our tasks.

| u_cross (m/s) | Re | downstream cd(Re) | κ(AR) | downstream Cd_2D | ratio ref/down |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 3.2e4 | 1.210 | 0.661 (sub) | 0.800 | **1.50×** |
| 0.05 | 8.0e4 | 1.210 | 0.661 (sub) | 0.799 | **1.50×** |
| 0.10 | 1.6e5 | 1.212 | 0.661 (sub) | 0.801 | **1.50×** |
| 0.20 | 3.2e5 | 1.028 | 0.814 (sup) | 0.837 | 1.43× |
| 0.30 | 4.8e5 | 0.520 | 0.814 (sup) | 0.423 | **2.83×** |
| 0.50 | 8.0e5 | 0.310 | 0.814 (sup) | **0.253** | **4.74×** |
| 0.80 | 1.28e6 | 0.442 | 0.814 (sup) | 0.360 | 3.33× |
| 1.00 | 1.60e6 | 0.503 | 0.814 (sup) | 0.409 | 2.93× |
| 1.50 | 2.40e6 | 0.594 | 0.814 (sup) | 0.483 | 2.48× |
| 2.00 | 3.20e6 | 0.636 | 0.814 (sup) | 0.518 | 2.31× |

**Reading:** reference's horizontal/lateral drag force is **1.5×–4.7× larger** than downstream's across the operating envelope. Worst at u_cross ≈ 0.5 m/s (the "drag crisis" trough of the DNV cylinder curve), best at u_cross < 0.2 m/s. Since `Yh ∝ ρ·D·Cd_2D·dx·Σ |v|·v` and `Nh ∝ ρ·D·Cd_2D·dx·Σ x·|v|·v`, both lateral force and yaw moment scale linearly with Cd_2D.

**Trajectory impact:** during a 15° rudder turn, the AUV's body sees v_r ≈ 0.3–0.5 m/s lateral cross-flow at the stern. Reference produces ~3-5× the yaw-damping moment of downstream → reference settles faster, with smaller overshoot. NODE trained on this fast-damping regime; downstream env will look "loose" / "skiddy" by comparison.

> This is the single largest physics-formula divergence between the two files.

---

## 6. Lift–drag (body) — Oswald efficiency divergence (medium)

| | reference | downstream |
|---|---|---|
| `e` (Oswald) | 0.7 (`remus100_core.py`:330) | **0.3** (`vehicle.py`:298) |
| CL formula | CL = CL_α · α | identical |
| CD formula | CD = CD_0 + CL²/(π·e·AR) | identical |

At fixed CL, downstream's induced-drag term is **2.33× larger** than reference's. The body CL_α with AR² / S geometry is the same in both. At small α (cruise ±2°), induced drag is a tiny correction to CD_0 and the divergence is invisible. At α > 10° (rapid pitching, large heave), induced drag becomes a significant fraction of total and the two implementations diverge.

---

## 7. Propeller — inflow speed source and Ja-branch (medium / negligible)

### 7.1 Inflow speed

| | reference | downstream |
|---|---|---|
| `Va` computation | `0.944 · |nu_r[0]|` (axial relative velocity) | `0.944 · ‖nu[:3]‖` (3-D body-velocity magnitude) |

The reference is correct propeller theory (axial inflow only). Downstream uses the magnitude of all three body velocity components, which makes the thrust artificially larger when there is any sideslip / pitch (v ≠ 0 or w ≠ 0). For typical near-cruise behavior (β ≈ α ≈ 0), these are numerically equivalent. For aggressive maneuvers or wake-induced cross-flow, downstream overestimates thrust.

Note: downstream uses *total* body velocity, not *relative* — but in a uniform-current scenario, ‖nu‖ ≈ ‖nu_r‖ once steady. The frame difference washes out for most analyses.

### 7.2 Ja-branch on reverse RPM

| | reference | downstream |
|---|---|---|
| `KT` formula | `if n_rps > 0: KT = KT_0 + (KT_max-KT_0)/Ja_max · Ja; else: KT_0` | `if |n_rps|>1e-8: KT = KT_0 + (KT_max-KT_0)/Ja_max · Ja` (no sign check) |

Difference only matters for negative RPM. PlanarRemusEnv has `n_rpm_min=200 > 0`, so downstream never enters the n_rps<0 branch. **Not load-bearing in our setup.**

---

## 8. Current model — inertial-3D vs planar-yaw-only (high in 6-DOF, equivalent in planar)

### 8.1 Reference

[`remus100_core.py`](../../phnode_full_oc_clean/reference_simulator/remus100_core.py):169-300:
```python
v_c_n   = [V_c·cos(β_c), V_c·sin(β_c), w_c]      # inertial 3-vector
v_c_body = R^T(φ, θ, ψ) @ v_c_n                  # full 3-D rotation
ν_c      = [v_c_body, 0, 0, 0]                   # 6-D twist
ν_r      = ν - ν_c
Dν_c     = [-ω × v_c_body, 0]                    # full 3-D cross product
ν̇       = Dν_c + M⁻¹ · τ_sum(ν_r, ...)
```

### 8.2 Downstream

[`auv_nav/vehicle.py`](../../auv_nav/vehicle.py):421-446 + 610-611:
```python
nu_c[0] = V_c · cos(β_c - ψ)        # 2-D yaw projection only
nu_c[1] = V_c · sin(β_c - ψ)
nu_c[2] = w_c                       # direct passthrough
# ...
Dnu_c[0] = nu[5] · v_c              # only r × v_c term
Dnu_c[1] = -nu[5] · u_c
Dnu_c[2..5] = 0                     # all other cross-product components dropped
```

### 8.3 Where they coincide and where they don't

- **At φ = θ = 0 (forced planar)**: `R^T = R_z(-ψ)` and `ω = [0, 0, r]`. Then `R^T · v_c_n = [V_c·cos(β_c-ψ), V_c·sin(β_c-ψ), w_c]` (matches downstream) and `ω × v_c_body = [r·v_c_body[1], -r·v_c_body[0], 0]` whose x,y entries are exactly downstream's. **Functionally equivalent.**
- **Off planar (φ ≠ 0 or θ ≠ 0, or p, q ≠ 0)**: downstream drops:
  - cross-coupling between roll/pitch and current frame projection (e.g., if AUV pitches 5° nose-down with vertical current w_c ≠ 0, downstream still uses nu_c[2]=w_c verbatim; reference rotates w_c into surge/heave)
  - the cross-product components Dν_c[2..5] (heave reaction from pitching through a current, etc.)

PlanarRemusEnv forces depth-tracking PID to z_ref=0 and clips pitch ≲ 5°, so we are *mostly* in the equivalent regime. NODE training also runs near-planar most of the time (init_angular_std=0.15 rad/s allows non-zero p, q, but mean is small). **In our task setup, this divergence is small but nonzero**; in a future 6-DOF task it would be high.

---

## 9. Mass / Coriolis matrices (CRB, CA) — algebraically equivalent ✓

Reference uses `_m2c(M_after_H_transform, ν_r)` with the symmetric Fossen formula `C = [0, -S(p1); -S(p1), -S(p2)]`. Downstream constructs `C_RB_CG = diag-block(m·S(ω), -S(Ig·ω))` then applies `H^T · C_RB_CG · H`. These are two textbook formulations of the same rigid-body Coriolis term; both yield the same numerical matrix for a rigid body with offset CG.

Both then zero out the same 8 entries of CA (the Lamb-coefficient off-diagonals): (0,4), (0,5), (1,5), (2,4) and their transposes. Match.

**Caveat:** numerical equivalence depends on geometry_scale being aligned (§2) — at scale=1.0096, downstream's MRB is +4.89% larger, so its C(ω,ν)·ν force differs proportionally.

---

## 10. Damping matrix — same formula, different numerics

Both define:
```
D = diag(M00/T_surge, M11/T_sway, M22/T_heave, M33·2ζ_roll·w_roll, M44·2ζ_pitch·w_pitch, M55/T_yaw)
D[0,0] *= exp(-3 · U_r)
D[1,1] *= exp(-3 · U_r)
```
Identical formula; downstream's D values are ~3% higher because M is ~3% larger (§2).

---

## 11. State representation — equivalent for the planar task

| | reference | downstream |
|---|---|---|
| State dimension | **16-D**: `[pos(3), quat(4), nu_r(6), u_actual(3)]` | **12-D**: `[u, v, w, p, q, r, x_n, y_e, z_d, φ, θ, ψ]` |
| Attitude rep | quaternion (RK4, renormalised each step) | Euler angles (T_zyx Jacobian, fails at θ→±90°) |
| Velocity stored | relative `nu_r` (current subtracted) | absolute `nu`, current re-subtracted each step |
| Actuator state | embedded in 16-D state | held externally, integrated by `step_actuators()` |

For PlanarRemusEnv (φ ≈ 0, θ ≈ ±5° at worst), Euler is numerically safe; the only mismatch is the actuator integration coupling (§4.2). If we ever wanted to use the same RK4 to integrate u_actual jointly with ν, the env step loop would need to be restructured.

---

## 12. RK4 integrator — equivalent ✓

`auv_nav.vehicle.rk4_step` and `phnode_full_oc.reference_simulator.remus100_core._rk4_step` are standard textbook RK4. No divergence.

---

## 13. Constants observed but not load-bearing in our setup

| Item | Notes |
|---|---|
| Reference `T_heave = 20 s` | not present in downstream; PlanarRemusEnv suppresses heave via DepthController. |
| `n_rpm_min = 200` (downstream) | reference allows n=0 implicitly; we never hit it. |
| Reference attitude singularity protection at `cth < 1e-8` | downstream raises `ValueError` instead. Different policy under extreme pitching. |
| Reference `Remus100Controller` (depth PID + heading SMC) | not loaded in our setup; we use `auv_nav/autopilot.py`. |

---

## 14. Summary table (severity-sorted)

| Rank | Issue | Severity | Trajectory impact (~1 s window) |
|---|---|---|---|
| 1 | Cross-flow drag (§5) | **High** | Yaw turn-rate damping diff ~3-5× → tightly-coupled with rudder controller behavior |
| 2 | Actuator τ (§4.1) | **High** | δ step response 2.5× slower in downstream; rudder authority effectively reduced for fast commands |
| 3 | Actuator integrator (§4.2) | **High** | Forward Euler + rate limit = piecewise-linear vs RK4 exponential; large for big commanded steps |
| 4 | `geometry_scale` (§2) | **High** | +3-5% mass/inertia bias on all dynamic forces, including controls and damping |
| 5 | Oswald *e* (§6) | Medium | 2.33× induced drag at given α; matters above α ≈ 10° |
| 6 | Propeller inflow Va (§7.1) | Medium | Small at near-cruise; up to ~10% thrust offset under heavy sideslip |
| 7 | Current frame model (§8) | Medium (planar: low) | 6-DOF: significant; planar: numerically equivalent at φ=θ=0 |
| 8 | Gravity (§3) | Low | 0.12% |
| 9 | Propeller Ja sign branch (§7.2) | None (not triggered) | n_rpm_min=200 keeps n_rps > 0 always |

CRB / CA / damping formulae (§9, §10) are *equivalent*; RK4 (§12) is *equivalent*.

---

## 15. Implications for AUVHamNODE Offline MBRL v3.0 plan

This audit was triggered by the question "can we feed PlanarRemusEnv transitions to AUVHamNODE for 1-step augmentation?" The answer this audit gives:

1. **Whatever NODE error we measure in spike-lite (Step 2) cannot be attributed cleanly to "current OOD"** without controlling for these 7 dynamics divergences. A 50 cm position error after one 0.2 s block could be (a) current shift, (b) §2 mass bias, (c) §4 actuator τ bias, (d) §5 cross-flow drag bias, or any superposition.

2. **The "Path 4 = drop NODE, use vehicle.py as oracle dynamics" path (audit doc §5.4)** would import all 7 divergences as systematic biases on the *training data*; the offline RL agent would then over-fit to vehicle.py's specific drag/τ formula. Either acceptable (we always deploy on vehicle.py) or problematic (we want sim2real to a different downstream simulator).

3. **The "swap vehicle.py → remus100_core.py" path** solves divergences 1-7 *in one move*, at the cost of reset­ting every existing experiment baseline (47 online thesis runs, ReBRAC mainline, A0 cross_u10). It does **not** solve the dominant Step 1 §2 issue (wake current magnitude 2-4× over NODE training cap), since that issue lives in `wake_data/*.npy`, not in `vehicle.py`. See [`04_swap_vehicle_decision_memo.md`](04_swap_vehicle_decision_memo.md) for the swap analysis.

4. **Spike-lite design** ([`02_spike_lite_design.md`](02_spike_lite_design.md)) should be aware that "NODE prediction MSE" measured by feeding PlanarRemusEnv transitions into NODE will conflate dynamics-mismatch noise with current-OOD noise. A *clean* spike-lite would compare:
   - (A) NODE rollout vs `remus100_core.Remus100Simulator` rollout (zero dynamics gap, only sees same physics)
   - (B) NODE rollout vs `vehicle.py.Remus100` rollout (full 7-axis gap)
   - The difference (B − A) isolates the §1-§7 cost; A alone isolates NODE's own intrinsic error.

---

## 16. References

- Reference simulator: [`phnode_full_oc_clean/reference_simulator/remus100_core.py`](../../phnode_full_oc_clean/reference_simulator/remus100_core.py) (543 lines, numpy only)
- Reference simulator README: [`phnode_full_oc_clean/reference_simulator/README.md`](../../phnode_full_oc_clean/reference_simulator/README.md)
- Reference verification helper: [`phnode_full_oc_clean/reference_simulator/verify_remus_consistency.py`](../../phnode_full_oc_clean/reference_simulator/verify_remus_consistency.py)
- Downstream simulator: [`auv_nav/vehicle.py`](../../auv_nav/vehicle.py) (865 lines)
- Numerical script used in this audit: see inline `python3` block referenced in §5.3 and §2 (not committed; reproducible from the snippets in this doc)

---

*Step 3 complete. Audit produces 9-row severity table (§14). Decision implications routed to [`04_swap_vehicle_decision_memo.md`](04_swap_vehicle_decision_memo.md).*
