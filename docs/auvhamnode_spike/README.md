# `experiments/auvhamnode_spike/` — AUVHamNODE Offline MBRL Pre-Spike Workspace

> **⚠ PAUSED 2026-05-13** — 这个工作区的 Step 0-4(smoke + 2 audit + decision memo)已经完成,但 Path 1B 的实际 spike(`02_spike_lite_design.md` 描述的 ~5h kill-test)**没有执行**;用户决定暂停整条 AUVHamNODE+MBRL 线。详情 + resume 起点见 [`docs/auvhamnode_mbrl_line_pause_memo.md`](../../docs/auvhamnode_mbrl_line_pause_memo.md)。
>
> 本工作区的 Markdown 文档(00–04)是 read-only decision evidence;脚本 `_wake_stats.py` 可复用。

Lightweight pre-spike workspace from 2026-05-13 (the day the project pivoted from ReBRAC mainline to AUVHamNODE-based offline MBRL). Output of the α path described in [`docs/auvhamnode_offline_mbrl_plan_v3_pre_notes.md`](../../docs/auvhamnode_offline_mbrl_plan_v3_pre_notes.md) §7.

## Files

| File | Role | Status |
|---|---|---|
| [`00_smoke_test_log.md`](00_smoke_test_log.md) | Step 0: env + checkpoint + ODE solver smoke test | ✅ green (mytorch1 env has torch + torchdiffeq; checkpoint loads; example_01 + example_02 run cleanly) |
| [`01_static_distribution_audit.md`](01_static_distribution_audit.md) | Step 1: 7-axis training-vs-deployment distribution audit + 4-path decision matrix | ✅ complete |
| [`_wake_stats.py`](_wake_stats.py) | Reusable script: speed magnitude stats over wake_data .npy files | usable, leave on Colab when OneDrive sync is unreliable |
| [`_wake_stats.out`](_wake_stats.out) | Raw stdout from one run (3/6 files; remaining 3 cloud-stubbed) | partial |
| [`02_spike_lite_design.md`](02_spike_lite_design.md) | Design for half-day kill-test if user picks Path 1B (~5 hours) | implementation pending user decision |
| [`03_dynamics_consistency_audit.md`](03_dynamics_consistency_audit.md) | Step 3: physics-by-physics comparison of `auv_nav/vehicle.py` vs `phnode_full_oc_clean/reference_simulator/remus100_core.py` (9-row severity table) | ✅ complete |
| [`04_swap_vehicle_decision_memo.md`](04_swap_vehicle_decision_memo.md) | Step 4: should we replace `vehicle.py` with `remus100_core.py`? (decision matrix + recommendation) | ✅ complete — recommendation: do not swap; spike-lite first |

## One-page TL;DR

**Step 1 verdict** (distribution audit): AUVHamNODE checkpoint is **viable for U=1.0 wakes** (~2× current shift, marginal but plausible) and **likely not viable for U=1.5 wakes** (~3-4× shift). 6 of 7 conditioning axes align well; the only OOD axis is the ocean current magnitude.

**Step 3 verdict** (dynamics consistency audit): `auv_nav/vehicle.py` and `phnode_full_oc_clean/reference_simulator/remus100_core.py` are **not consistent**. 7 formula-level divergences, top 4 in severity:
- cross-flow drag Cd_2D constant ~1.20 vs Re/AR-dependent 0.25–0.80 (ratio 1.5–4.7×)
- actuator τ_fin = 0.10 s vs 0.25 s (2.5× slower in downstream)
- different actuator integrator (joint RK4 vs explicit Euler + rate-limit)
- `geometry_scale=1.0096` in downstream → +3% mass, +5% inertia

**Step 4 verdict** (swap decision): **do not swap**. Swap fixes ~30% of NODE-deployment problem (dynamics formulae) but leaves ~70% untouched (wake current magnitude OOD, spatial heterogeneity, control-period mismatch, frame-representation gap — all live outside `vehicle.py`). Cost = reset every existing baseline.

**Recommended next step:** Path 1B (spike-lite, ~5 hours) → either commits to Path 1 (scope v3.0 to U=1.0) or pivots to Path 4 (drop NODE, use `auv_nav/vehicle.py` as dynamics oracle). **Defer any `vehicle.py` swap decision until spike-lite isolates dynamics gap vs current-OOD as the dominant error source.**

**Awaiting:** user decision among A/B/C/D/E (see audit §5 and chat report).

## When done

This workspace can be `git add`-ed as decision artifacts (small markdown + script, no large data). The actual spike(s) will be in `scripts/` and their outputs in subdirectories of `experiments/auvhamnode_spike/` (e.g. `experiments/auvhamnode_spike/spike_lite_results/`).
