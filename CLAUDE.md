# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for training autonomous underwater vehicles (AUVs) — specifically REMUS-100 class — to navigate wake fields using reinforcement learning. The primary algorithm is Soft Actor-Critic (SAC), with RLPD (RL with Prior Data) support for leveraging offline baseline data to accelerate online training.

## Common Commands

```bash
# Basic SAC training (uses synthetic wake data if no flow file found)
python -m scripts.train_sac

# Training with key options
python -m scripts.train_sac \
  --total-steps 200000 \
  --batch-size 256 \
  --device cuda \
  --seed 42 \
  --difficulty medium \
  --task-geometry cross_stream \
  --num-envs 8 \
  --history-length 4

# Resume from checkpoint
python -m scripts.train_sac --resume checkpoints/sac/

# Evaluate a saved checkpoint
python -m scripts.evaluate --checkpoint checkpoints/sac/ --episodes 20

# Run baseline policies (goal-seek, current-compensation, etc.)
python -m scripts.demo --policy all --episodes 10

# Run experiment suite across seeds
python -m scripts.run_suite --suite medium_formal_v1 --method sac --num-seeds 5

# Generate synthetic wake field data
python -m scripts.generate_wake

# Collect offline data from baseline policies (for RLPD)
python -m scripts.collect_offline_data \
  --policy worldcomp \
  --flow wake_data/wake_v8_U1p50_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy \
  --probe-layout s0 --difficulty hard --target-speed 1.5 \
  --episodes 500 --seed 0 \
  --output-dir offline_data/worldcomp

# RLPD training (SAC + offline data)
python -m scripts.train_sac \
  --offline-data offline_data/worldcomp/transitions.npz \
  --offline-ratio 0.5 \
  --use-layernorm --difficulty hard --target-speed 1.5

# Visualize trajectories and plot training curves
python -m scripts.visualize --checkpoint checkpoints/sac/
python -m scripts.plot_training --log-dir checkpoints/sac/
```

All scripts are run as modules from the repo root (`python -m scripts.<name>`).

## Architecture

### Package: `auv_nav/`

The core library. Components are loosely coupled; non-ML parts work without PyTorch.

| Module | Role |
|--------|------|
| `vehicle.py` | REMUS-100 6-DOF nonlinear dynamics (RK4 integrator, ~850 lines) |
| `flow.py` | Memory-mapped wake field dataset, flow sampling at body-relative positions |
| `env.py` | `PlanarRemusEnv` — Gymnasium environment (obs: 8-base + n_probes×2, 2-D action) |
| `autopilot.py` | Inner-loop PID controllers (heading hold, depth hold) |
| `sac.py` | SAC agent: `SquashedGaussianActor`, dual `QNetwork`, auto-tuned temperature |
| `networks.py` | MLP building blocks used by SAC |
| `replay.py` | `TransitionReplay` off-policy buffer, `DualBufferSampler` for RLPD symmetric sampling |
| `reward.py` | `RewardModel` (progress, success, timeout) + `SafetyCostModel` |
| `baselines.py` | Non-learning policies for comparison (goal-seek, current compensation) |

### Scripts: `scripts/`

| Script | Role |
|--------|------|
| `train_sac.py` | Main training entry point; supports pure SAC and RLPD modes via `--offline-data` |
| `train_utils.py` | Shared helpers: env creation, checkpointing, evaluation loop, CSV/JSONL logging |
| `run_suite.py` | Coordinates multi-seed experiment sweeps; defines `METHOD_SPECS` and `SUITE_PRESETS` |
| `collect_offline_data.py` | Collects transition data from baseline policies for RLPD training |
| `evaluate.py` | Loads a checkpoint and runs deterministic evaluation |

### Key Architectural Patterns

**Dataclass configs everywhere.** `TrainConfig`, `SACConfig`, `PlanarRemusEnvConfig`, `TaskSamplerConfig` are all `@dataclass`. They serialize to JSON for reproducibility and map directly to CLI args.

**Policy API.** All agents (SAC, baselines) implement `act(obs, policy_state, deterministic) → (action, policy_state)` and `reset_policy_state()`. This supports both stateless and recurrent policies uniformly.

**Checkpointing.** `save_training_state()` in `train_utils.py` saves agent weights, replay buffer, Python RNG state, and a JSON metadata file. `maybe_resume()` restores all of it to continue training exactly.

**Parallel environments.** Training supports `gymnasium.vector.AsyncVectorEnv` for wall-clock speedup. The `--num-envs` flag controls parallelism; `--update-every` should be scaled accordingly.

**RLPD (offline-to-online).** When `--offline-data` is provided, `train_sac.py` loads a pre-collected `.npz` dataset into a read-only `TransitionReplay` via `from_npz()`, then uses `DualBufferSampler` to draw each mini-batch with 50/50 (configurable via `--offline-ratio`) split between offline and online buffers. Without `--offline-data`, training behaves as standard SAC with no performance regression.

**Conditional PyTorch.** `vehicle.py`, `flow.py`, and `env.py` import PyTorch via a `require_torch()` guard, so they can be used for analysis without a GPU install.

### Environment Details

- **Observation:** 8 base channels + n_probes×2 (velocity) channels
  - Base 8: surge `u`, sway `v`, yaw rate `r`, `cos(ψ)`, `sin(ψ)`, goal body-frame x/y, distance-to-goal
  - Probe channels: (u, v) per probe in body frame
  - `s0` (default, 1 probe): **10-D**; `s1` (2 probes): **12-D**; `s2` (4 probes): **16-D**
- **Action (2-D):** continuous heading command, speed command
- **Probe layouts (`--probe-layout`):** all physically grounded in real REMUS-100 sensors
  - `s0` — 1 probe at (0,0); DVL water-track, REMUS-100 standard baseline
  - `s1` — 2 probes at (0,0)+(4.5,0); DVL + 2 MHz short-range forward ADCP, ~3 steps advance warning
  - `s2` — 4 probes at (0,0)+(5,0)+(8,±4); DVL + 1 MHz long-range ADCP, ~7 steps warning + lateral gradient
- **Task geometry:** `downstream`, `cross_stream`, `upstream`
- **Difficulty:** `easy`, `medium`, `hard` (controls start/goal separation and flow strength)
- **Observation history:** `--history-length N` wraps env with `ObservationHistoryWrapper` for stacking N recent observations

### Data

Wake field data lives in `wake_data/` (gitignored). `scripts/generate_wake.py` creates synthetic fields. `flow.py`'s `WakeField` class memory-maps `.npy` files with shape `(T, Nx, Ny, C)`. Each `.npy` file requires a co-located `<stem>_meta.json` companion file (auto-generated by `generate_wake.py`).

Offline transition data lives in `offline_data/` (gitignored). Each subdirectory contains `transitions.npz` (obs, actions, rewards, costs, next_obs, dones) and `metadata.json` (policy name, statistics). Generated by `scripts/collect_offline_data.py`. Supported baseline policies: `goalseek`, `crosscomp`, `worldcomp`, `privileged`.

### Default Hyperparameters (SACConfig)

`hidden_dim=256`, `gamma=0.995`, `tau=0.005`, `actor/critic/alpha_lr=3e-4`, `init_alpha=0.2`, `grad_clip_norm=10.0`. Training defaults: `batch_size=256`, `random_steps=2000`, `update_after=2000`.
