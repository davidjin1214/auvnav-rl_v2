# Design: Multi-Cylinder Wake Fields + Improved SAC

**Date:** 2026-04-04
**Status:** Approved
**Scope:** Two independent sub-projects, zero breaking changes to existing code.

---

## Sub-project 1: Multi-Cylinder Wake Field Generation

### Goal

Extend `scripts/generate_wake.py` to simulate tandem (串联) and side-by-side (并排) dual-cylinder configurations using the existing TRT-LBM solver. The representative gap ratio is G/D=3.5 for both configurations.

### Design Decisions

- **ROI for tandem**: starts 1D upstream of the first cylinder, covering the gap region and 20D past the downstream cylinder. This allows the AUV to learn gap-traversal vs. circumnavigation strategies.
- **Gap ratio**: G/D=3.5 (critical spacing, bistable regime, strongest POMDP challenge).
- **Approach**: High-level preset functions generate `PhysicsConfig` instances; bottom-level `TRTSolver` receives a combined mask. Existing single-cylinder code is untouched.

### Architecture

```
High-level (new)              Low-level (minimal change)     Solver (3-line change)
────────────────              ──────────────────────────     ──────────────────────
make_tandem_physics()    →    PhysicsConfig                  TRTSolver
make_side_by_side_physics()   + extra_cylinders: list        + _cylinder: combined mask
                              + roi_y_center_override         (OR of multiple circles)
                              ↓
                              LatticeConfig
                              + cylinders_lat: tuple
                                [(cx, cy, r), ...]
```

### File Changes: `scripts/generate_wake.py`

#### 1. `PhysicsConfig` — two new fields (backward compatible)

```python
extra_cylinders: list = field(default_factory=list)
# Format: [(x_phys, y_phys, D_phys), ...]  — does not include primary cylinder
# Default empty list → single-cylinder behavior unchanged

roi_y_center_override: float | None = None
# None → uses cyl_y_center as ROI y-center (existing behavior)
# Set to domain midpoint for side-by-side to center ROI between both cylinders
```

#### 2. `LatticeConfig` — one new field

```python
cylinders_lat: tuple = ()
# All cylinders (including primary) as (cx_lat, cy_lat, r_lat) in lattice units
# Populated by from_physics()
```

`LatticeConfig.from_physics()` changes:
- Computes `cylinders_lat` by converting primary + `extra_cylinders` to lattice coords using the same formula as existing `cx_lat`: `cx = int(round(x_phys / dx))`, `r = int(round(D_phys / (2 * dx)))`.
- Uses `roi_y_center_override` (if set) instead of `cyl_y_center` for ROI y-bounds: `roi_y0 = roi_cy_lat - int(round(roi_y_half_D * D_lat))`.

#### 3. `TRTSolver.__init__` — 3-line change

```python
# Before (1 line):
self._cylinder = (x - lc.cx_lat)**2 + (y - lc.cy_lat)**2 <= (lc.D_lat / 2)**2

# After (3 lines):
self._cylinder = xp.zeros((lc.Nx, lc.Ny), dtype=bool)
for cx, cy, r in lc.cylinders_lat:
    self._cylinder |= (x - cx)**2 + (y - cy)**2 <= xp.float32(r)**2
```

`apply_bcs()` and `macroscopic()` are unchanged — both operate on `self._cylinder`.

#### 4. Preset functions (new)

**`make_tandem_physics(Re, U_phys, gap_ratio=3.5) -> PhysicsConfig`**

```
Domain:  Lx=1200m, Ly=400m
Cyl 1:   x=200m, y=200m, D=20m  (upstream)
Cyl 2:   x=290m, y=200m, D=20m  (center-to-center = (1 + G/D)*D = 4.5D = 90m)
ROI x:   [180m, 690m]  → 1D before cyl1 (roi_x_start_D=-1.0) to 20D past cyl2 (roi_x_end_D=24.5)
ROI y:   [130m, 270m]  → ±3.5D about domain center
```

**`make_side_by_side_physics(Re, U_phys, gap_ratio=3.5) -> PhysicsConfig`**

```
Domain:  Lx=1000m, Ly=600m  (taller to avoid wall interference)
Cyl 1:   x=200m, y=245m, D=20m  (center + 2.25D)
Cyl 2:   x=200m, y=155m, D=20m  (center - 2.25D)
ROI x:   [240m, 700m]  → 2D past cylinders (roi_x_start_D=2.0) to 25D (roi_x_end_D=25.0)
ROI y:   [80m,  320m]  → ±6D about domain center y=200m  (roi_y_half_D=6.0)
roi_y_center_override=200m
```

#### 5. New `NavigationProfile` presets

```python
NAVIGATION_PROFILES["tandem_G35_nav"]       # Re=[150,200], U=[0.8, 1.0, 1.2]
NAVIGATION_PROFILES["side_by_side_G35_nav"] # Re=[150,200], U=[0.8, 1.0, 1.2]
```

#### 6. CLI extension

```bash
python -m scripts.generate_wake --case tandem_G35_nav
python -m scripts.generate_wake --case side_by_side_G35_nav
# Existing: --case single_cylinder_nav  (unchanged)
```

---

## Sub-project 2: Improved SAC

### Goal

Add three orthogonal improvements to the existing SAC: (1) LayerNorm for training stability, (2) DroQ-style Dropout + high UTD for sample efficiency, (3) Asymmetric Critic using privileged observations to address the POMDP gap. All improvements are flag-controlled for ablation experiments.

### Design Decisions

- **Config-driven flags**: `SACConfig` boolean fields enable/disable each improvement; single `SACAgent` class handles all combinations.
- **Privileged obs content**: true body-frame equivalent flow velocity (u_eq, v_eq) — 2 dimensions, already computed internally by the environment, zero extra simulation cost.
- **Dropout rate**: 0.01 (DroQ recommendation — very small dropout provides regularization without sacrificing policy expressiveness).
- **UTD ratio**: 4 for DroQ (configurable via `updates_per_step` in `TrainConfig`).
- **Observation history**: K=16 recommended (covers 8s ≈ 40–80% of vortex period, vs current K=4 covering only 10–20%).

### Architecture Overview

```
Training time:
  Actor    ← obs (probe-based, K stacked frames)
  Critic   ← obs + action + privileged_obs (true u_eq, v_eq)

Deployment time:
  Actor    ← obs (probe-based)  [unchanged]
  Critic   not used
```

### File Changes

#### `auv_nav/networks.py` — extend `build_mlp`

Add two optional parameters (defaults off, fully backward compatible):

```python
def build_mlp(in_dim, hidden_dim, out_dim, n_layers,
              activation=nn.ReLU,
              use_layernorm: bool = False,
              dropout_rate: float = 0.0) -> nn.Sequential
```

Layer order per hidden unit: `Linear → [LayerNorm] → ReLU → [Dropout(p)]`

Pre-activation LayerNorm matches DroQ paper and is more stable at high UTD than post-activation.

#### `auv_nav/sac.py` — `SACConfig` and `AsymmetricQNetwork`

New fields in `SACConfig`:

```python
# Architecture improvements
use_layernorm: bool = False
dropout_rate: float = 0.0       # 0.0 = off; DroQ uses 0.01

# Asymmetric Critic
privileged_obs_dim: int = 0     # 0 = off; set to 2 to enable
```

`SquashedGaussianActor` and `QNetwork` constructors pass `use_layernorm` and `dropout_rate` through to `build_mlp`. Training step logic is unchanged.

New class `AsymmetricQNetwork`:

```python
class AsymmetricQNetwork(nn.Module):
    """Critic input: obs + action + privileged_obs during training.
    Pass privileged_obs=None at eval time (zero-padded automatically)."""

    def forward(self, obs, action, privileged_obs=None):
        if privileged_obs is None:
            privileged_obs = torch.zeros(obs.shape[0], self._priv_dim,
                                         device=obs.device)
        x = torch.cat([obs, action, privileged_obs], dim=-1)
        return self.mlp(x).squeeze(-1)
```

When `privileged_obs_dim > 0`, `SACAgent` instantiates `AsymmetricQNetwork` instead of `QNetwork` for both critics. Only the critic update path passes `privileged_obs`; actor update and inference are unchanged.

#### `auv_nav/env.py` — expose privileged obs

In `step()`, add to the returned `info` dict:

```python
info["privileged_obs"] = np.array([
    self._u_eq_body,   # already computed by hull flow sampling
    self._v_eq_body,
], dtype=np.float32)
```

Compatible with `gymnasium.vector.AsyncVectorEnv` — vector env auto-stacks `info` arrays.

#### `auv_nav/replay.py` — optional privileged obs buffer

```python
class TransitionReplay:
    def __init__(self, obs_dim, action_dim, capacity,
                 privileged_obs_dim: int = 0):   # new, default off
        ...
        if privileged_obs_dim > 0:
            self.privileged_obs = np.zeros((capacity, privileged_obs_dim),
                                           dtype=np.float32)

    def add(self, obs, action, reward, cost, next_obs, done,
            privileged_obs=None):                # new optional arg
        ...

    def sample_batch(self, batch_size, device):
        batch = { ...existing... }
        if self._priv_dim > 0:
            batch["privileged_obs"] = torch.as_tensor(
                self.privileged_obs[idx], device=device)
        return batch
```

#### `scripts/train_sac.py` — UTD support

New field in `TrainConfig`:

```python
updates_per_step: int = 1
# Set to 4-8 for DroQ. Default 1 = existing behavior.
```

Training loop: replace single `agent.update(batch)` call with a loop of `updates_per_step` iterations. No other changes.

#### `scripts/run_suite.py` — ablation presets

```python
SAC_ABLATION_SPECS = {
    "sac_baseline":   dict(sac=SACConfig(),
                           train=TrainConfig(updates_per_step=1, history_length=4)),

    "sac_ln_k16":     dict(sac=SACConfig(use_layernorm=True),
                           train=TrainConfig(history_length=16)),

    "sac_droq":       dict(sac=SACConfig(use_layernorm=True, dropout_rate=0.01),
                           train=TrainConfig(updates_per_step=4, history_length=16)),

    "sac_asym_k16":   dict(sac=SACConfig(use_layernorm=True, privileged_obs_dim=2),
                           train=TrainConfig(history_length=16)),

    "sac_full":       dict(sac=SACConfig(use_layernorm=True, dropout_rate=0.01,
                                          privileged_obs_dim=2),
                           train=TrainConfig(updates_per_step=4, history_length=16)),
}
```

Run ablation:
```bash
python -m scripts.run_suite --suite sac_ablation --num-seeds 3
```

---

## Full File Change Summary

| File | Change type |
|------|-------------|
| `scripts/generate_wake.py` | 2 new `PhysicsConfig` fields, 1 new `LatticeConfig` field, 3-line mask rewrite, 2 preset functions, 2 `NavigationProfile` entries, CLI update |
| `auv_nav/networks.py` | `build_mlp`: 2 new optional params |
| `auv_nav/sac.py` | `SACConfig`: 3 new fields; new `AsymmetricQNetwork` class; critic instantiation branch |
| `auv_nav/env.py` | `step()`: add `privileged_obs` to `info` |
| `auv_nav/replay.py` | Optional `privileged_obs` buffer (new constructor param + add/sample methods) |
| `scripts/train_sac.py` | `TrainConfig`: new `updates_per_step`; training loop UTD support |
| `scripts/run_suite.py` | New `SAC_ABLATION_SPECS` dict |

**Zero breaking changes**: all new fields have defaults that reproduce current behavior exactly.

---

## Open Questions (resolved)

- ROI for tandem: Option A (include gap region) ✓
- Gap ratio: G/D=3.5 (single representative value) ✓
- SAC improvements: both DroQ and Asymmetric Critic, designed as ablation ✓
- Privileged obs: 2D body-frame equivalent flow velocity ✓
