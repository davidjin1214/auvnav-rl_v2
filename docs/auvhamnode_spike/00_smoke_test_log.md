# Step 0 — Environment Smoke Test (AUVHamNODE phnode_full_oc_clean)

**Date:** 2026-05-13
**Goal:** verify `phnode_full_oc_clean` checkpoint loads and runs on local `mytorch1` env; confirm `torchdiffeq` dependency is available; capture baseline rollout numbers as reference for later spike work.

---

## 1. Environment

| Item | Value |
| --- | --- |
| Conda env | `mytorch1` |
| Python | `/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin/python` |
| torch | 2.10.0 (CPU-only on this Mac) |
| torchdiffeq | OK (installed, version not exposed as attr) |
| Device | `cpu` |

**Note on `conda` shell binding:** the `conda` command itself does not run cleanly under the harness shell (`__conda_exe: permission denied`). Workaround: invoke env-specific Python by absolute path: `/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin/python`. Functionally equivalent to `conda run -n mytorch1 python`.

## 2. Example 01 — `single_step` (verbatim run)

Command:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin/python phnode_full_oc_clean/examples/01_single_step.py
```

Initial state: rest at origin, identity rotation, zero velocity, zero actuator, no current. Command `u_cmd = [0, 0, 600]` (rudder 0, stern 0, propeller 600 RPM). Integration: `dt=0.2s`, `method='rk4'`.

Output:

```
Loaded phnode_full (seed=45, epoch=247, train_loss=4.0249e-03, state_dim=27)
After dt=0.2s with surge command 600 RPM:
  Δposition (m)         = [0.00158, -7.88e-06, 1.38e-05]
  body-frame v_r (m/s) = [0.01871, -4.23e-05, 0.00028]
  body-frame omega     = [-0.00204, 0.00147, -0.00097]
  actuator state (post-lag) = [0.0, 0.0, 110.12]
```

**Physical sanity:** RPM actuator time constant `T_actuator[2] = 1.0s` (from config) ⇒ at `t=0.2s` the realized RPM is `600 · (1 - exp(-0.2)) ≈ 109` — matches `actuator state = 110.12`. The vehicle has barely started moving (Δx ≈ 1.6 mm, v_x ≈ 0.019 m/s), which is correct for "starting from rest, accelerating into commanded thrust with first-order actuator lag." Sway/heave/omega numerics are at noise floor (1e-4 to 1e-3 magnitude).

**Result:** ✅ checkpoint loads, ODE solver runs, physics is self-consistent.

## 3. Example 02 — `rollout` (30s control schedule)

Command:

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin/python phnode_full_oc_clean/examples/02_rollout.py
```

Initial state: 1.0 m/s surge, 800 RPM (steady cruise). Schedule: 150 blocks × 0.2s = 30s; rudder square wave ±0.1 rad on blocks [30:60] and [90:120]; stern fixed at 0; RPM fixed at 800. No current.

Output:

```
Trajectory: 601 samples over 30.00 s
  start position = [0.0, 0.0, 0.0]
  end   position = [39.683, 3.204, 0.141]
  final body-frame v_r = [1.342, 0.036, 0.000]
  max |position| (m)   = 39.813
```

**Physical sanity:**
- Surge accelerated from 1.0 → 1.34 m/s (RPM stepped up from 800 → 800 cruise but body started below thrust equilibrium, then settled). Average speed ≈ 40/30 = 1.33 m/s, consistent with terminal `v_r_x = 1.34`.
- Lateral excursion +3.2 m after 30s with two symmetric ±0.1 rad rudder bursts is sensible (net positive sway from temporal asymmetry of square wave windows).
- Vertical drift (z = 0.14 m) is small — straight-and-level cruise stays straight-and-level, no obvious heave instability.

**Result:** ✅ multi-block rollout integrates cleanly over 30s; no NaN/Inf; numerics stable.

## 4. Provenance & Config readback (full)

Headline facts extracted from `phnode_full_oc_clean/checkpoints/seed45/provenance.json` + `config.json` (full dump used in `01_static_distribution_audit.md`):

| Field | Value |
| --- | --- |
| Model | phnode_full, seed 45, epoch 247, train_loss 4.02e-3 |
| Dataset id | `d0be9434` (auv_oc_traj1000_blk150_s23) |
| dt_sim / dt_state / dt_ctrl | 0.01 / 0.05 / **0.2** s |
| ODE solver | rk4 |
| State dim | 27 = `[x(3), R(9), nu_r(6), u_actual(3), u_cmd(3), v_c_n(3)]` |
| u_dim | 3 = `[rudder_rad, stern_rad, propeller_RPM]` |
| t_actuator (rudder, stern, RPM) | (0.1, 0.1, 1.0) s |
| u_act_scale | (1.0, 1.0, 0.001) |
| init_surge range | [0.8, 2.5] m/s |
| init_sway_std / init_heave_std | 0.25 / 0.15 m/s |
| init_angular_std | 0.15 rad/s |
| delta_max (rudder/stern) | 0.2618 rad (15°) |
| rpm_range | [400, 1400] |
| velocity_max (clip) | [4.0, 0.8, 0.8, 0.5, 0.5, 0.5] (body-frame [v, ω]) |
| max_attitude | 1.2 rad (~69°) |
| depth_bounds | [0.5, 100.0] m |
| current_speed_range | **[0.0, 0.5] m/s** |
| current_direction_range | [-π, π] |
| current_vertical_std | 0.05 |
| Excitation mix | PRBS 40%, CHIRP 35%, OU 25% |
| Bench horizon | 60 s (clean eval, seed-45 median pos_err = 0.43 m) |
| `absolute_depth_context` | **false** (model is depth-translation-invariant) |

**Implication for cross-domain transfer (preview of §1 audit):**
- `current_speed_range = [0.0, 0.5]` vs wake_data with `U_ref ∈ {1.0, 1.5}` ⇒ **shift ratio 2× – 3× at least, before accounting for vortex-core peaks.** This is the dominant risk.
- `init_surge ∈ [0.8, 2.5]` covers PlanarRemusEnv target speed 1.5 m/s comfortably.
- `delta_max = 15°` rudder + `rpm_range = [400, 1400]` covers PID-driven actuator outputs in typical AUV cruise.
- `dt_ctrl = 0.2s` ≠ PlanarRemusEnv `control_dt = 0.5s` ⇒ augmentation pipeline must split each env step into 2-3 inner blocks, not 1.

## 5. Step 0 verdict

**✅ Green light to continue with Step 1** (static distribution audit).

Environment, model load, single_step, multi-block rollout all clean. No infrastructure blockers. The one operational reminder for downstream spike work:

- Use absolute path `/opt/homebrew/Caskroom/miniforge/base/envs/mytorch1/bin/python` to invoke the model under the harness shell.
- Add `sys.path.insert(0, str(<repo>/phnode_full_oc_clean))` in any new script that imports `phnode_full_oc` (export uses package-relative imports).

---

*Step 0 complete in ~30 minutes (env check + 2 examples + this log).*
