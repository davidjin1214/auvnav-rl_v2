# Spike-Lite Design (preflight kill-test, ~4-6 hours, only run if user picks Path 1B)

**Purpose.** Before committing 1-2 days to the full adapter spike (Step 3 in the pre-notes), spend half a day on the **minimum viable evidence** that AUVHamNODE is or is not useful at U=1.0 wakes. The output is a single number: the median 1-step planar-state MSE on a fixed set of N=100 dataset transitions, with a confidence interval. This is the hard go / no-go for Path 1.

**What spike-lite does NOT do** (to stay within 4-6 hours):
- No PID layer (action layer Q is left for full Step 3)
- No σ_a action perturbation curve (v2.1 §4.2 B is full Step 3 only)
- No Check A (in-training-distribution control) — that needs upstream training dataset, which is not in the export
- No full lift/project class — hand-rolled inline functions, throwaway code
- No checkpoint / no logging infrastructure

**What spike-lite DOES do:**
- Loads 1 wake (U=1.0 `wake_v8_U1p00_Re150`) + collects N=100 PlanarRemusEnv transitions using a deterministic baseline policy (`baselines.goalseek` or similar)
- For each transition, hand-builds a 27-D SE(3) state from the recorded planar state and wake-sampled `v_c_n`
- Calls `single_step(model, state_27d, dt=0.2)` — **note: not 0.5s**, see Critical Detail 1 below
- Projects predicted 27-D next_state back to planar and compares to dataset `s_{t+1}`
- Reports median / p95 MSE on (position, velocity)

---

## Critical Detail 1: dt slicing

PlanarRemusEnv `control_dt = 0.5s`. AUVHamNODE `dt_ctrl = 0.2s`. So one env step ≠ one NODE step.

**Spike-lite simplification:** call `single_step` **twice with dt=0.2 each** (total 0.4s), holding `u_cmd` and `v_c_n` constant — this gives a NODE-predicted state at t+0.4s. Compare to PlanarRemusEnv state at t+0.4s, which we can reach by stepping the env with `sim_dt=0.1s` for 4 substeps inside the same control step boundary.

This loses 0.1s of the full 0.5s env step but is in-training-distribution. The full Step 3 spike will handle the 0.5s mismatch properly (probably with `2.5 * 0.2 = 0.5` via 2 full + 1 partial block, or with a learned interpolation).

**Alternative:** call `single_step(dt=0.5)` once and accept the 2.5× horizon extrapolation. This is the path of least resistance but contaminates the MSE measurement with horizon-OOD error. **Not recommended for kill-test.**

## Critical Detail 2: lift / project (hand-rolled, throwaway)

```python
def lift_planar_to_se3(planar_state: dict, wake_v_c_n: np.ndarray) -> torch.Tensor:
    """planar -> 27-D SE(3) state.

    planar_state: dict with keys
        u, v, r            : body-frame surge, sway, yaw rate
        psi                : heading angle (rad)
        x_world, y_world   : inertial position
        delta_r, delta_s   : actuator state from env's autopilot diagnostics
        n_rpm              : actuator state from env's autopilot diagnostics
        delta_r_cmd, delta_s_cmd, n_rpm_cmd : commanded actuator (this step)
    wake_v_c_n: numpy [3] body-frame current at hull center; we will rotate to inertial below

    Returns: torch.Tensor shape [1, 27]
    """
    # x: assume z = 0 (env depth_ref) — DOES NOT condition due to absolute_depth_context=false
    pos = np.array([planar_state['x_world'], planar_state['y_world'], 0.0])

    # R: planar -> SE(3) rotation matrix; roll=pitch=0
    psi = planar_state['psi']
    cp, sp = np.cos(psi), np.sin(psi)
    R = np.array([[cp, -sp, 0.0],
                  [sp,  cp, 0.0],
                  [0.0, 0.0, 1.0]])

    # nu_r: body-frame velocity RELATIVE to water (training convention).
    # planar gives total body-frame [u, v, 0]; w (heave) ~ 0; angular: [0, 0, r]
    # Relative velocity = total - body-frame current.
    # wake_v_c_n is sampled in inertial frame upstream; convert to body frame via R.T
    v_c_n_inertial = wake_v_c_n  # caller's responsibility to pass inertial
    v_c_body = R.T @ v_c_n_inertial  # rotate inertial->body
    nu_total_body = np.array([planar_state['u'], planar_state['v'], 0.0,
                              0.0, 0.0, planar_state['r']])
    nu_r = nu_total_body.copy()
    nu_r[:3] -= v_c_body  # subtract current in body frame
    # nu_r[3:] (angular) is already body-frame relative; current is translational only

    u_actual = np.array([planar_state['delta_r'], planar_state['delta_s'], planar_state['n_rpm']])
    u_cmd    = np.array([planar_state['delta_r_cmd'], planar_state['delta_s_cmd'], planar_state['n_rpm_cmd']])

    # Concatenate to 27-D
    state_27 = np.concatenate([pos, R.flatten(), nu_r, u_actual, u_cmd, v_c_n_inertial])
    return torch.from_numpy(state_27).float().unsqueeze(0)


def project_se3_to_planar(state_27: torch.Tensor) -> dict:
    """Inverse of lift_planar_to_se3."""
    s = state_27.squeeze(0).cpu().numpy()
    pos = s[0:3]
    R = s[3:12].reshape(3, 3)
    nu_r = s[12:18]  # body-frame relative velocity
    u_actual = s[18:21]
    v_c_n_inertial = s[24:27]
    # Recover total body-frame velocity: v_total = v_r + R.T @ v_c_n_inertial
    v_c_body = R.T @ v_c_n_inertial
    v_total_body = nu_r[:3] + v_c_body
    # heading from rotation matrix
    psi = np.arctan2(R[1, 0], R[0, 0])
    return {
        'u': float(v_total_body[0]),
        'v': float(v_total_body[1]),
        'r': float(nu_r[5]),  # yaw rate body-frame (no current correction needed for angular)
        'psi': float(psi),
        'x_world': float(pos[0]),
        'y_world': float(pos[1]),
        'delta_r': float(u_actual[0]),
        'delta_s': float(u_actual[1]),
        'n_rpm':   float(u_actual[2]),
    }
```

**Known limitations of this lift/project**:
1. `v_c_n` is sampled at hull center — could use `EquivalentCurrentModel` hull-integral for better fidelity. Full Step 3 spike should switch to hull-integral.
2. R assumes φ=θ=0; if env's depth_hold_autopilot has any non-zero pitch from depth correction, this introduces a small `θ` error which manifests as wrong nu_r decomposition.
3. Hand-built rotation conversion has a singularity-at-pole, but PlanarRemusEnv never approaches it.

## Critical Detail 3: ground-truth collection

```python
# in collect script
env = make_env(flow="wake_v8_U1p00_Re150_*.npy", probe_layout="s0",
               task_geometry="cross_stream", target_speed=1.5,
               objective="arrival_v2")
policy = WorldCompensationPolicy()  # or GoalSeekPolicy — pick deterministic
transitions = []
for ep_seed in range(20):  # 20 eps × ~5 steps each → ~100 transitions
    obs, info = env.reset(seed=ep_seed)
    for step in range(5):  # only need first few steps; each is a 0.5s control step
        action, _ = policy.act(obs, ...)
        next_obs, reward, term, trunc, next_info = env.step(action)
        # extract everything spike needs
        transitions.append({
            "planar_state": extract_planar_from_info(info),
            "planar_action_high_level": action,  # [heading_cmd, speed_cmd]
            "autopilot_diag": info["autopilot_delta_r_cmd"], ...
            "wake_v_c_n_inertial": sample_wake_at_position(env.wake_field, info["x_world"], info["y_world"], info["t"]),
            "planar_state_next": extract_planar_from_info(next_info),
        })
        if term or trunc: break
        info = next_info
```

## Output

A single CSV row + a single number:

```
n_transitions, mse_pos_med, mse_pos_p95, mse_vel_med, mse_vel_p95
100, 0.087, 0.412, 0.054, 0.231
```

(numbers are illustrative)

**Go criterion (Path 1 viable):**
- `mse_pos_med` < 0.2 m **AND** `mse_vel_med` < 0.2 m/s **AND** `mse_pos_p95` < 1.0 m
- These are 5× the in-training clean-eval pos error (0.43 m / 60s = 7 mm/s drift) — generous to account for cross-domain.

**No-go criterion:**
- Either median exceeds 0.5 m (position) / 0.5 m/s (velocity) → AUVHamNODE is not useful at U=1.0 either → pivot to Path 4

**Marginal:**
- p95 exceeds 1.0 m but median is OK → tail risk needs investigation; Step 3 spike should add EquivalentCurrentModel hull-integral lift before deciding

## Time budget

| Sub-task | Estimated |
|---|---|
| Write `scripts/spike_lite_auvhamnode_planar.py` | 1.5 h |
| Verify lift/project on synthetic test (planar straight-line) | 0.5 h |
| Collect 100 transitions from env | 0.5 h |
| Run model on all 100 transitions, compute MSE | 0.5 h |
| Write `02_spike_lite_results.md` with numbers + verdict | 1 h |
| Slack / sanity time | 1 h |
| **Total** | **5 h** |

## What "go" unlocks

- Commit to Path 1 (scope v3.0 to U=1.0)
- Start full Step 3 adapter spike with confidence (hull-integral lift, σ_a curve, etc.)
- Begin writing v3.0 plan in parallel

## What "no-go" triggers

- Switch to Path 4 (drop NODE, use `vehicle.py` oracle) immediately, no full Step 3
- Re-pre-note v3.0 around vehicle.py as the augmentation source
- AUVHamNODE export sits on shelf; revisit only if upstream training pipeline becomes accessible for finetune (Path 2)

---

*This document is a *design* for spike-lite, not a result. It exists so that if the user picks Path 1B, implementation can start immediately without re-deriving the protocol.*
