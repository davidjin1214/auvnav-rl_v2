# ReBRAC Broad Validation Implementation Plan

> ---
> **⚠ SUPERSEDED 2026-05-18**：本 v1 implementation plan 已被 [`docs/rebrac_broad_validation_v2_plan.md`](../../rebrac_broad_validation_v2_plan.md) 取代。详细取代原因见 v1 spec 头部 banner（[`docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`](../specs/2026-05-04-rebrac-broad-validation-design.md)）。
>
> v1 plan 在 `efficiency_v2` reward 下定义的 8 spoke × Probe-then-Deepen 流程不再 active；v2 plan 在 `arrival_v2` reward 下收敛到 5 cell core + 1 conditional sweep。
>
> v1 实验产物（`experiments/offline/rebrac/broad_validation/` + 完成的 notebook archive）保留作历史 reference，不重跑。
> ---
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the three-axis (data quality / sensor / task geometry) Probe-then-Deepen broad validation defined in [docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md](../specs/2026-05-04-rebrac-broad-validation-design.md), producing 7 new offline datasets, 16 P1 runs, conditional P2 deepening, and a paper-ready broad-validation subsection.

**Architecture:**
- New per-spoke driver `scripts/run_offline_rebrac_broad.sh` modelled on `run_offline_rebrac_screen.sh` but with a built-in spoke registry, parameterised algorithm (`rebrac` / `td3bc`), and per-spoke `(dataset_name, policy/mixture, probe, geom, flow, benchmark)` overrides. Reuses `train_offline.py`, `evaluate_offline.py`, and the same `[skip]`-resume conventions.
- Three small Python helpers: `concat_offline_datasets.py` (A2 mix5050), `write_sanity_card.py` (per-dataset sanity), `summarize_broad_validation.py` (P1/P2 aggregation + trigger gate).
- Single Colab notebook `notebooks/rebrac_broad_validation.ipynb` driving collect → P1 → P2 → analyse end-to-end.

**Tech Stack:** Python 3.11, PyTorch, Gymnasium, NumPy. Bash 3.2-compatible drivers (macOS Colab base image). No new dependencies.

**Compute:** Google Colab Pro / L4. Total ~26–32 run / 19–22 h L4 across 3–4 sessions. Editing/local smoke is on the local Mac (`mytorch1` conda env).

---

## File Structure

| Path | Role | Status |
|---|---|---|
| `scripts/broad_validation_spoke_registry.py` | Single source of truth for the 8 spoke configs (A1, A2, A2-td3bc, A3, B1, B2, C1, C3). Pure data, no I/O. | NEW |
| `scripts/run_offline_rebrac_broad.sh` | Bash driver. Takes `SPOKE_ID` + `SEEDS` + `PHASE` env vars, looks up spoke config via the registry, runs collect/train/validate/select/test/summarize for that spoke. | NEW |
| `scripts/concat_offline_datasets.py` | Concat two `transitions.npz` along episode dim, preserve `dones` boundaries + `privileged_obs`, write merged `metadata.json` with `mix_components` / `mix_strategy`. | NEW |
| `scripts/write_sanity_card.py` | Read `transitions.npz` + `metadata.json`, derive 7 sanity-card fields, write `<dataset>/sanity_card.json`. | NEW |
| `scripts/summarize_broad_validation.py` | Aggregate per-spoke `selected_checkpoint.json` + `test/seed_*.json` into `summaries/p1_overview.csv` and `summaries/p2_overview.csv`. Apply spec §6.1 trigger gate, emit `trigger_decisions.json`. | NEW |
| `notebooks/rebrac_broad_validation.ipynb` | Single Colab driver: 1 collect cell per dataset, 1 P1 cell per spoke, 1 analysis cell, 1 P2 cell. | NEW |
| `docs/rebrac_broad_validation_report.md` | Final report with §1–§5 (motivation, matrix, P1, P2, discussion). | NEW |
| `docs/rebrac_mainline_review.md` | Add §3.5 cross-link to broad validation. | MODIFY |

Datasets land in `offline_data/<spoke_dataset>/` (gitignored). Results land in `results/offline/rebrac/broad_validation/<spoke_id>/...` (gitignored). Checkpoints in `checkpoints/offline/rebrac/broad_validation/...` (gitignored).

---

## Task 1: Spoke Registry (single source of truth)

**Files:**
- Create: `scripts/broad_validation_spoke_registry.py`

The registry is pure data. The bash driver and the summary script both query it via `python -m scripts.broad_validation_spoke_registry --get-field <spoke_id> <field>` so the spec mapping lives in exactly one place.

- [ ] **Step 1: Write the failing unit test**

Create `tests/test_broad_validation_spoke_registry.py`:

```python
"""Unit tests for the broad-validation spoke registry."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.broad_validation_spoke_registry import REGISTRY, SpokeConfig, get_spoke


def test_all_eight_spokes_present() -> None:
    expected = {"A1", "A2", "A2-td3bc", "A3", "B1", "B2", "C1", "C3"}
    assert set(REGISTRY.keys()) == expected


def test_anchor_invariants() -> None:
    """Spec §3.5: spokes change exactly one axis vs anchor."""
    anchor = SpokeConfig(
        spoke_id="anchor",
        dataset_name="crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000",
        collector_policy="crosscomp",
        policy_mixture=None,
        probe_layout="s0",
        task_geometry="cross_stream",
        flow_path="wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
        benchmark_key="single_u10_cross_tgt15",
        target_speed=1.5,
        algo="rebrac",
        actor_penalty_coef=4.0,
        critic_penalty_coef=2.0,
        td3bc_alpha=None,
    )
    a_axis = {"A1", "A2", "A3"}
    b_axis = {"B1", "B2"}
    c_axis = {"C1", "C3"}
    for sid in a_axis:
        cfg = get_spoke(sid)
        assert cfg.probe_layout == anchor.probe_layout
        assert cfg.task_geometry == anchor.task_geometry
        assert cfg.flow_path == anchor.flow_path
    for sid in b_axis:
        cfg = get_spoke(sid)
        assert cfg.collector_policy == anchor.collector_policy
        assert cfg.task_geometry == anchor.task_geometry
        assert cfg.flow_path == anchor.flow_path
    for sid in c_axis:
        cfg = get_spoke(sid)
        assert cfg.collector_policy == anchor.collector_policy
        assert cfg.probe_layout == anchor.probe_layout


def test_a2_uses_episode_level_mixture() -> None:
    cfg = get_spoke("A2")
    assert cfg.policy_mixture == "goalseek:1.0,crosscomp:1.0"


def test_a2_td3bc_shares_dataset_with_a2() -> None:
    assert get_spoke("A2").dataset_name == get_spoke("A2-td3bc").dataset_name


def test_a2_td3bc_alpha_matches_phase0c_winner() -> None:
    cfg = get_spoke("A2-td3bc")
    assert cfg.algo == "td3bc"
    assert cfg.td3bc_alpha == 0.25


def test_b_axis_obs_dim_consistent_with_probe() -> None:
    assert get_spoke("B1").probe_layout == "s1"
    assert get_spoke("B2").probe_layout == "s2"


def test_c_axis_flow_paths_exist_in_repo() -> None:
    """C3 must use the tandem wake; C1 must keep single Re150 wake."""
    assert "tandem" in get_spoke("C3").flow_path
    assert "tandem" not in get_spoke("C1").flow_path
    assert get_spoke("C1").task_geometry == "upstream"
    assert get_spoke("C1").benchmark_key == "single_u10_upstream_tgt15"


def test_cli_get_field() -> None:
    """Bash driver must be able to query fields via subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.broad_validation_spoke_registry",
         "--get-field", "A1", "dataset_name"],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == (
        "goalseek_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000"
    )


def test_cli_get_json() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.broad_validation_spoke_registry",
         "--get-json", "A1"],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["spoke_id"] == "A1"
    assert payload["collector_policy"] == "goalseek"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=. pytest tests/test_broad_validation_spoke_registry.py -v
```

Expected: ImportError / "No module named 'scripts.broad_validation_spoke_registry'".

- [ ] **Step 3: Implement the registry**

Create `scripts/broad_validation_spoke_registry.py`:

```python
"""Single source of truth for the broad-validation spoke configurations.

The registry is queried by:
  - scripts/run_offline_rebrac_broad.sh (via --get-field)
  - scripts/summarize_broad_validation.py (via Python import)
  - notebooks/rebrac_broad_validation.ipynb (via Python import)

Spec reference: docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md §3.5.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SpokeConfig:
    spoke_id: str
    dataset_name: str
    collector_policy: str  # one of: goalseek, crosscomp, worldcomp, privileged
    policy_mixture: str | None  # for A2: "goalseek:1.0,crosscomp:1.0"
    probe_layout: str  # s0 / s1 / s2
    task_geometry: str  # cross_stream / upstream / downstream
    flow_path: str
    benchmark_key: str
    target_speed: float
    algo: str  # rebrac / td3bc
    actor_penalty_coef: float | None  # ReBRAC β1
    critic_penalty_coef: float | None  # ReBRAC β2
    td3bc_alpha: float | None  # TD3+BC α


_FLOW_RE150_SINGLE = (
    "wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
)
_FLOW_RE150_TANDEM = (
    "wake_data/wake_tandem_G35_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
)


def _spoke(
    spoke_id: str,
    *,
    dataset_name: str,
    collector_policy: str,
    policy_mixture: str | None = None,
    probe_layout: str = "s0",
    task_geometry: str = "cross_stream",
    flow_path: str = _FLOW_RE150_SINGLE,
    benchmark_key: str = "single_u10_cross_tgt15",
    target_speed: float = 1.5,
    algo: str = "rebrac",
    actor_penalty_coef: float | None = 4.0,
    critic_penalty_coef: float | None = 2.0,
    td3bc_alpha: float | None = None,
) -> SpokeConfig:
    return SpokeConfig(
        spoke_id=spoke_id,
        dataset_name=dataset_name,
        collector_policy=collector_policy,
        policy_mixture=policy_mixture,
        probe_layout=probe_layout,
        task_geometry=task_geometry,
        flow_path=flow_path,
        benchmark_key=benchmark_key,
        target_speed=target_speed,
        algo=algo,
        actor_penalty_coef=actor_penalty_coef,
        critic_penalty_coef=critic_penalty_coef,
        td3bc_alpha=td3bc_alpha,
    )


REGISTRY: dict[str, SpokeConfig] = {
    "A1": _spoke(
        "A1",
        dataset_name="goalseek_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000",
        collector_policy="goalseek",
    ),
    "A2": _spoke(
        "A2",
        dataset_name="mix5050_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000",
        collector_policy="goalseek",  # nominal anchor; mixture overrides
        policy_mixture="goalseek:1.0,crosscomp:1.0",
    ),
    "A2-td3bc": _spoke(
        "A2-td3bc",
        dataset_name="mix5050_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000",
        collector_policy="goalseek",
        policy_mixture="goalseek:1.0,crosscomp:1.0",
        algo="td3bc",
        actor_penalty_coef=None,
        critic_penalty_coef=None,
        td3bc_alpha=0.25,  # phase0c Stage C 1000-ep winner
    ),
    "A3": _spoke(
        "A3",
        dataset_name="privileged_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000",
        collector_policy="privileged",
    ),
    "B1": _spoke(
        "B1",
        dataset_name="crosscomp_s1_h4_efficiency_v2_re150_u10cross_fixdone_ep1000",
        collector_policy="crosscomp",
        probe_layout="s1",
    ),
    "B2": _spoke(
        "B2",
        dataset_name="crosscomp_s2_h4_efficiency_v2_re150_u10cross_fixdone_ep1000",
        collector_policy="crosscomp",
        probe_layout="s2",
    ),
    "C1": _spoke(
        "C1",
        dataset_name="crosscomp_s0_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000",
        collector_policy="crosscomp",
        task_geometry="upstream",
        benchmark_key="single_u10_upstream_tgt15",
    ),
    "C3": _spoke(
        "C3",
        dataset_name="crosscomp_s0_h4_efficiency_v2_re150tandem_u10cross_fixdone_ep1000",
        collector_policy="crosscomp",
        flow_path=_FLOW_RE150_TANDEM,
    ),
}


def get_spoke(spoke_id: str) -> SpokeConfig:
    if spoke_id not in REGISTRY:
        raise KeyError(
            f"Unknown spoke id: {spoke_id!r}. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[spoke_id]


def _format_field(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the spoke registry.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list", action="store_true", help="Print all spoke ids, one per line."
    )
    group.add_argument(
        "--get-field",
        nargs=2,
        metavar=("SPOKE_ID", "FIELD"),
        help="Print one field of the spoke config (for bash callers).",
    )
    group.add_argument(
        "--get-json",
        metavar="SPOKE_ID",
        help="Print the full spoke config as JSON.",
    )
    args = parser.parse_args()

    if args.list:
        for sid in REGISTRY:
            print(sid)
        return

    if args.get_field is not None:
        spoke_id, field = args.get_field
        cfg = get_spoke(spoke_id)
        if not hasattr(cfg, field):
            raise SystemExit(f"Unknown field: {field!r}")
        print(_format_field(getattr(cfg, field)))
        return

    if args.get_json is not None:
        cfg = get_spoke(args.get_json)
        json.dump(asdict(cfg), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=. pytest tests/test_broad_validation_spoke_registry.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Smoke-check the CLI**

```bash
python -m scripts.broad_validation_spoke_registry --list
python -m scripts.broad_validation_spoke_registry --get-field A2 dataset_name
python -m scripts.broad_validation_spoke_registry --get-field A2-td3bc td3bc_alpha
python -m scripts.broad_validation_spoke_registry --get-json C3
```

Expected output of the second line: `mix5050_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000`. The third line: `0.25`. The fourth line: full JSON for C3.

- [ ] **Step 6: Commit**

```bash
git add scripts/broad_validation_spoke_registry.py tests/test_broad_validation_spoke_registry.py
git commit -m "feat(rebrac-broad): spoke registry — single source of truth for 8 spoke configs"
```

---

## Task 2: Sanity Card Writer

**Files:**
- Create: `scripts/write_sanity_card.py`
- Test: `tests/test_write_sanity_card.py`

The 7 sanity-card fields specified in spec §4.4:
1. `collector_success_rate` — env-reported `is_success` mean (= `metadata.json["success_rate"]`)
2. `collector_mean_return` — = `metadata.json["mean_return"]`
3. `episode_length_mean / std` — derived from `transitions.npz["dones"]` boundaries
4. `obs_dim` — = `metadata.json["obs_dim"]`
5. `n_transitions` — = `metadata.json["num_transitions"]`
6. `privileged_obs_present` — = `metadata.json["privileged_obs_dim"] > 0`
7. `flow_file` — basename of `metadata.json["flow_path"]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_write_sanity_card.py`:

```python
"""Unit tests for the sanity card writer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.write_sanity_card import (
    derive_sanity_card,
    write_sanity_card,
)


@pytest.fixture
def tmp_dataset_dir(tmp_path: Path) -> Path:
    """Synthetic 3-episode dataset (lengths 5, 7, 4 = 16 transitions)."""
    n = 16
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, 10), dtype=np.float32)
    actions = rng.standard_normal((n, 2), dtype=np.float32)
    rewards = rng.standard_normal(n, dtype=np.float32)
    next_obs = rng.standard_normal((n, 10), dtype=np.float32)
    dones = np.zeros(n, dtype=np.float32)
    dones[[4, 11, 15]] = 1.0
    privileged_obs = rng.standard_normal((n, 2), dtype=np.float32)
    np.savez_compressed(
        tmp_path / "transitions.npz",
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
        privileged_obs=privileged_obs,
    )
    metadata = {
        "policy": "crosscomp",
        "obs_dim": 10,
        "action_dim": 2,
        "privileged_obs_dim": 2,
        "num_episodes": 3,
        "num_transitions": 16,
        "success_rate": 0.667,
        "mean_return": 12.5,
        "std_return": 3.1,
        "mean_episode_length": 5.33,
        "flow_path": "wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def test_derive_card_pulls_seven_fields(tmp_dataset_dir: Path) -> None:
    card = derive_sanity_card(tmp_dataset_dir)
    assert card["collector_success_rate"] == pytest.approx(0.667)
    assert card["collector_mean_return"] == pytest.approx(12.5)
    assert card["episode_length_mean"] == pytest.approx((5 + 7 + 4) / 3)
    assert card["episode_length_std"] > 0
    assert card["obs_dim"] == 10
    assert card["n_transitions"] == 16
    assert card["privileged_obs_present"] is True
    assert card["flow_file"] == "wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy"


def test_obs_dim_consistency_check(tmp_dataset_dir: Path) -> None:
    """Card writer must verify obs_dim matches probe_layout when available."""
    card = derive_sanity_card(tmp_dataset_dir, expected_probe_layout="s0")
    assert card["obs_dim_matches_probe_layout"] is True
    card_s1 = derive_sanity_card(tmp_dataset_dir, expected_probe_layout="s1")
    assert card_s1["obs_dim_matches_probe_layout"] is False


def test_write_creates_file(tmp_dataset_dir: Path) -> None:
    out_path = write_sanity_card(tmp_dataset_dir, expected_probe_layout="s0")
    assert out_path.name == "sanity_card.json"
    payload = json.loads(out_path.read_text())
    assert payload["obs_dim"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. pytest tests/test_write_sanity_card.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the writer**

Create `scripts/write_sanity_card.py`:

```python
"""Derive a 7-field sanity card from an offline-data directory.

Spec §4.4: every new dataset under offline_data/ must have a sanity_card.json
recording the 7 fields below before being used for training.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


_PROBE_TO_OBS_DIM = {"s0": 10, "s1": 12, "s2": 16}


def _episode_lengths(dones: np.ndarray) -> list[int]:
    """Return list of episode lengths inferred from `dones` boundaries."""
    lengths: list[int] = []
    current = 0
    for flag in dones:
        current += 1
        if bool(flag):
            lengths.append(current)
            current = 0
    if current > 0:  # trailing partial episode (should not happen on closed datasets)
        lengths.append(current)
    return lengths


def derive_sanity_card(
    dataset_dir: Path,
    *,
    expected_probe_layout: str | None = None,
) -> dict[str, Any]:
    metadata_path = dataset_dir / "metadata.json"
    transitions_path = dataset_dir / "transitions.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing {metadata_path}")
    if not transitions_path.exists():
        raise FileNotFoundError(f"missing {transitions_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    with np.load(transitions_path, mmap_mode="r") as payload:
        dones = np.asarray(payload["dones"])
    episode_lengths = _episode_lengths(dones)
    if not episode_lengths:
        raise ValueError(f"transitions.npz at {transitions_path} has no episode boundaries")

    flow_path = metadata.get("flow_path", "")
    flow_file = Path(flow_path).name if flow_path else ""

    privileged_dim = int(metadata.get("privileged_obs_dim", 0) or 0)

    card: dict[str, Any] = {
        "collector_success_rate": float(metadata.get("success_rate", float("nan"))),
        "collector_mean_return": float(metadata.get("mean_return", float("nan"))),
        "episode_length_mean": statistics.fmean(episode_lengths),
        "episode_length_std": statistics.pstdev(episode_lengths) if len(episode_lengths) > 1 else 0.0,
        "obs_dim": int(metadata.get("obs_dim", -1)),
        "n_transitions": int(metadata.get("num_transitions", dones.shape[0])),
        "privileged_obs_present": privileged_dim > 0,
        "flow_file": flow_file,
        "n_episodes": len(episode_lengths),
    }

    if expected_probe_layout is not None:
        expected_dim = _PROBE_TO_OBS_DIM.get(expected_probe_layout)
        card["expected_probe_layout"] = expected_probe_layout
        card["obs_dim_matches_probe_layout"] = (
            expected_dim is not None and card["obs_dim"] == expected_dim
        )

    return card


def write_sanity_card(
    dataset_dir: Path,
    *,
    expected_probe_layout: str | None = None,
) -> Path:
    card = derive_sanity_card(
        dataset_dir, expected_probe_layout=expected_probe_layout
    )
    output_path = dataset_dir / "sanity_card.json"
    output_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a dataset sanity card.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-probe-layout",
        choices=["s0", "s1", "s2"],
        default=None,
    )
    args = parser.parse_args()
    out_path = write_sanity_card(
        args.dataset_dir,
        expected_probe_layout=args.expected_probe_layout,
    )
    print(f"[write] sanity card: {out_path}")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    for key, value in payload.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. pytest tests/test_write_sanity_card.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/write_sanity_card.py tests/test_write_sanity_card.py
git commit -m "feat(rebrac-broad): sanity card writer (spec §4.4)"
```

---

## Task 3: A2 Mix Concat Helper

**Files:**
- Create: `scripts/concat_offline_datasets.py`
- Test: `tests/test_concat_offline_datasets.py`

Spec §4.3 mandates the A2 mix dataset is built from two separate collections (500 ep goalseek seed=0 + 500 ep crosscomp seed=1) and concatenated, with extended metadata (`mix_components`, `mix_strategy`, `task_sampler`). This is the only place we deviate from "use the existing tools as-is" — `--policy-mixture` would also work but produces ~500/500 (binomial noise) and lacks the spec's metadata schema.

- [ ] **Step 1: Write the failing test**

Create `tests/test_concat_offline_datasets.py`:

```python
"""Unit tests for the dataset concat helper (A2 mix5050)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.concat_offline_datasets import concat_datasets


def _write_synthetic(
    out_dir: Path,
    *,
    policy: str,
    num_episodes: int,
    transitions_per_episode: int,
    seed: int,
    obs_dim: int = 10,
    privileged_dim: int = 2,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = num_episodes * transitions_per_episode
    rng = np.random.default_rng(seed)
    payload = {
        "obs": rng.standard_normal((n, obs_dim), dtype=np.float32),
        "actions": rng.standard_normal((n, 2), dtype=np.float32),
        "rewards": rng.standard_normal(n, dtype=np.float32),
        "next_obs": rng.standard_normal((n, obs_dim), dtype=np.float32),
        "dones": np.zeros(n, dtype=np.float32),
        "privileged_obs": rng.standard_normal((n, privileged_dim), dtype=np.float32),
        "next_privileged_obs": rng.standard_normal((n, privileged_dim), dtype=np.float32),
    }
    payload["dones"][transitions_per_episode - 1::transitions_per_episode] = 1.0
    np.savez_compressed(out_dir / "transitions.npz", **payload)

    metadata = {
        "policy": policy,
        "obs_dim": obs_dim,
        "action_dim": 2,
        "privileged_obs_dim": privileged_dim,
        "num_episodes": num_episodes,
        "num_transitions": n,
        "success_rate": 0.5,
        "mean_return": 10.0,
        "std_return": 2.0,
        "mean_episode_length": float(transitions_per_episode),
        "flow_path": "wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy",
        "probe_layout": "s0",
        "history_length": 4,
        "task_geometry": "cross_stream",
        "target_speed": 1.5,
        "objective": "efficiency_v2",
        "seed": seed,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata))
    return out_dir


def test_concat_preserves_episode_boundaries(tmp_path: Path) -> None:
    a = _write_synthetic(tmp_path / "a", policy="goalseek", num_episodes=3, transitions_per_episode=5, seed=0)
    b = _write_synthetic(tmp_path / "b", policy="crosscomp", num_episodes=2, transitions_per_episode=4, seed=1)
    out_dir = tmp_path / "merged"

    concat_datasets([a, b], output_dir=out_dir, mix_strategy="episode_level")

    with np.load(out_dir / "transitions.npz") as merged:
        dones = merged["dones"]
    # Total = 3 ep * 5 + 2 ep * 4 = 23 transitions, 5 dones boundaries.
    assert dones.shape == (23,)
    assert int(dones.sum()) == 5
    # First 15 transitions belong to dataset A (3 episodes of length 5).
    assert dones[4] == 1.0 and dones[9] == 1.0 and dones[14] == 1.0
    # Last 8 transitions belong to dataset B (2 episodes of length 4).
    assert dones[18] == 1.0 and dones[22] == 1.0


def test_concat_writes_extended_metadata(tmp_path: Path) -> None:
    a = _write_synthetic(tmp_path / "a", policy="goalseek", num_episodes=3, transitions_per_episode=5, seed=0)
    b = _write_synthetic(tmp_path / "b", policy="crosscomp", num_episodes=2, transitions_per_episode=4, seed=1)
    out_dir = tmp_path / "merged"

    concat_datasets([a, b], output_dir=out_dir, mix_strategy="episode_level")

    metadata = json.loads((out_dir / "metadata.json").read_text())
    components = metadata["mix_components"]
    assert len(components) == 2
    assert components[0]["policy"] == "goalseek"
    assert components[0]["num_episodes"] == 3
    assert components[1]["policy"] == "crosscomp"
    assert components[1]["num_episodes"] == 2
    assert metadata["mix_strategy"] == "episode_level"
    assert metadata["num_episodes"] == 5
    assert metadata["num_transitions"] == 23
    assert metadata["task_sampler"] == "anchor_distribution"


def test_concat_rejects_dim_mismatch(tmp_path: Path) -> None:
    a = _write_synthetic(tmp_path / "a", policy="goalseek", num_episodes=3, transitions_per_episode=5, seed=0, obs_dim=10)
    b = _write_synthetic(tmp_path / "b", policy="crosscomp", num_episodes=2, transitions_per_episode=4, seed=1, obs_dim=12)
    with pytest.raises(ValueError, match="obs_dim"):
        concat_datasets([a, b], output_dir=tmp_path / "merged", mix_strategy="episode_level")


def test_concat_preserves_privileged_obs(tmp_path: Path) -> None:
    a = _write_synthetic(tmp_path / "a", policy="goalseek", num_episodes=3, transitions_per_episode=5, seed=0)
    b = _write_synthetic(tmp_path / "b", policy="crosscomp", num_episodes=2, transitions_per_episode=4, seed=1)
    out_dir = tmp_path / "merged"
    concat_datasets([a, b], output_dir=out_dir, mix_strategy="episode_level")
    with np.load(out_dir / "transitions.npz") as merged:
        assert "privileged_obs" in merged.files
        assert "next_privileged_obs" in merged.files
        assert merged["privileged_obs"].shape[1] == 2
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=. pytest tests/test_concat_offline_datasets.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the concat helper**

Create `scripts/concat_offline_datasets.py`:

```python
"""Concat multiple offline datasets along the transition (= episode) dimension.

Used by the broad-validation A2 mix5050 dataset (spec §4.3): produces an
episode-level mixture by concatenating two separate same-shape datasets
collected with different behavior policies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_dataset(dataset_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    transitions_path = dataset_dir / "transitions.npz"
    metadata_path = dataset_dir / "metadata.json"
    if not transitions_path.exists():
        raise FileNotFoundError(f"missing {transitions_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing {metadata_path}")
    with np.load(transitions_path) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return arrays, metadata


def _check_compatibility(metadatas: list[dict]) -> None:
    base = metadatas[0]
    for m in metadatas[1:]:
        if int(m.get("obs_dim", -1)) != int(base.get("obs_dim", -2)):
            raise ValueError(
                f"obs_dim mismatch: {base.get('obs_dim')} vs {m.get('obs_dim')}"
            )
        if int(m.get("action_dim", -1)) != int(base.get("action_dim", -2)):
            raise ValueError("action_dim mismatch across datasets")
        if int(m.get("privileged_obs_dim", 0) or 0) != int(
            base.get("privileged_obs_dim", 0) or 0
        ):
            raise ValueError("privileged_obs_dim mismatch across datasets")


def concat_datasets(
    dataset_dirs: list[Path],
    *,
    output_dir: Path,
    mix_strategy: str,
    task_sampler: str = "anchor_distribution",
) -> Path:
    if len(dataset_dirs) < 2:
        raise ValueError("Need at least two datasets to concat.")

    payloads: list[dict[str, np.ndarray]] = []
    metadatas: list[dict] = []
    for d in dataset_dirs:
        arrs, meta = _load_dataset(d)
        payloads.append(arrs)
        metadatas.append(meta)

    _check_compatibility(metadatas)

    keys = sorted({k for p in payloads for k in p})
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        # All payloads must contain the same keys; if any are missing, drop the key.
        if not all(key in p for p in payloads):
            continue
        merged[key] = np.concatenate([p[key] for p in payloads], axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "transitions.npz", **merged)

    base_meta = metadatas[0]
    components = [
        {
            "policy": m.get("policy"),
            "num_episodes": int(m.get("num_episodes", 0)),
            "num_transitions": int(m.get("num_transitions", 0)),
            "success_rate": float(m.get("success_rate", float("nan"))),
            "mean_return": float(m.get("mean_return", float("nan"))),
            "seed": int(m.get("seed", -1)),
            "source_dir": str(d),
        }
        for m, d in zip(metadatas, dataset_dirs)
    ]

    merged_meta = dict(base_meta)
    merged_meta["mix_components"] = components
    merged_meta["mix_strategy"] = mix_strategy
    merged_meta["task_sampler"] = task_sampler
    merged_meta["num_episodes"] = sum(c["num_episodes"] for c in components)
    merged_meta["num_transitions"] = int(merged["dones"].shape[0])
    merged_meta["policy"] = "+".join(c["policy"] or "?" for c in components)

    # Recompute aggregate scalars.
    weighted_success = sum(
        c["success_rate"] * c["num_episodes"] for c in components
    ) / max(1, merged_meta["num_episodes"])
    weighted_return = sum(
        c["mean_return"] * c["num_episodes"] for c in components
    ) / max(1, merged_meta["num_episodes"])
    merged_meta["success_rate"] = float(weighted_success)
    merged_meta["mean_return"] = float(weighted_return)
    merged_meta.pop("seed", None)
    merged_meta.pop("std_return", None)
    merged_meta.pop("mean_episode_length", None)

    (output_dir / "metadata.json").write_text(
        json.dumps(merged_meta, indent=2), encoding="utf-8"
    )
    return output_dir / "transitions.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Concat offline datasets.")
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        type=Path,
        help="Source dataset directory. Pass --input-dir twice or more.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mix-strategy", default="episode_level")
    parser.add_argument("--task-sampler", default="anchor_distribution")
    args = parser.parse_args()

    out_path = concat_datasets(
        args.input_dir,
        output_dir=args.output_dir,
        mix_strategy=args.mix_strategy,
        task_sampler=args.task_sampler,
    )
    print(f"[write] merged dataset: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=. pytest tests/test_concat_offline_datasets.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/concat_offline_datasets.py tests/test_concat_offline_datasets.py
git commit -m "feat(rebrac-broad): A2 mix5050 dataset concat helper (spec §4.3)"
```

---

## Task 4: Broad Validation Bash Driver

**Files:**
- Create: `scripts/run_offline_rebrac_broad.sh`

This driver executes one spoke at a time. Per (`SPOKE_ID`, `SEEDS`, `PHASE`) it:
1. Looks up the spoke config via `python -m scripts.broad_validation_spoke_registry --get-field`.
2. Ensures the dataset exists (via `--policy-mixture` for A2; via single policy otherwise; B/C axes share the same logic with overridden probe / geometry / flow).
3. Ensures the manifest exists for the spoke's `BENCHMARK_KEY`.
4. For each `(actor_beta, critic_beta, seed)` cell — or `(alpha, seed)` for `algo=td3bc` — runs train, validate, select_best, test.
5. Skip-resume by checking `trainer_state.json` + `agent_final.pt` per run dir.

Phase semantics:
- `PHASE=p1` → `SEEDS="42 44"` and the spoke's anchor `(β1, β2)` (or `α`) only.
- `PHASE=p2_refit` → `SEEDS="42"`, `ACTOR_PENALTY_COEFS` / `CRITIC_PENALTY_COEFS` overridable to e.g. `"2.0"` / `"2.0"` for one cell at a time.
- `PHASE=p2_5seed` → `SEEDS="42 43 44 45 46"` on the determined winner.

- [ ] **Step 1: Author the driver**

Create `scripts/run_offline_rebrac_broad.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# ReBRAC broad-validation per-spoke driver.
#
# Spec: docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md
# Plan: docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md
#
# Usage:
#   SPOKE_ID=A1 PHASE=p1 bash scripts/run_offline_rebrac_broad.sh
#   SPOKE_ID=B2 PHASE=p1 bash scripts/run_offline_rebrac_broad.sh
#   SPOKE_ID=C3 PHASE=p2_refit ACTOR_PENALTY_COEFS=2.0 CRITIC_PENALTY_COEFS=2.0 bash scripts/run_offline_rebrac_broad.sh
#   SPOKE_ID=C3 PHASE=p2_5seed bash scripts/run_offline_rebrac_broad.sh
#
# All other phase-independent configuration (TRAIN_EPOCHS, BATCH_SIZE, ...) is
# inherited from the same env-var contract as run_offline_rebrac_screen.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPOKE_ID="${SPOKE_ID:?SPOKE_ID is required (e.g. A1, A2, A2-td3bc, A3, B1, B2, C1, C3)}"
PHASE="${PHASE:?PHASE is required (p1, p2_refit, or p2_5seed)}"
MODE="${MODE:-all}"

PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_CMD=()
if [[ -n "$PYTHON_PREFIX" ]]; then
  read -r -a PYTHON_PREFIX_ARR <<< "$PYTHON_PREFIX"
  PYTHON_CMD+=("${PYTHON_PREFIX_ARR[@]}")
fi
PYTHON_CMD+=("$PYTHON_BIN")

# ---- Spoke registry lookup ----
get_spoke_field() {
  "${PYTHON_CMD[@]}" -m scripts.broad_validation_spoke_registry --get-field "$SPOKE_ID" "$1"
}

DATASET_NAME="$(get_spoke_field dataset_name)"
COLLECTOR_POLICY="$(get_spoke_field collector_policy)"
POLICY_MIXTURE="$(get_spoke_field policy_mixture)"
PROBE_LAYOUT="$(get_spoke_field probe_layout)"
TASK_GEOMETRY="$(get_spoke_field task_geometry)"
FLOW_PATH="$(get_spoke_field flow_path)"
BENCHMARK_KEY="$(get_spoke_field benchmark_key)"
TARGET_SPEED="$(get_spoke_field target_speed)"
ALGO="$(get_spoke_field algo)"
ACTOR_PENALTY_DEFAULT="$(get_spoke_field actor_penalty_coef)"
CRITIC_PENALTY_DEFAULT="$(get_spoke_field critic_penalty_coef)"
TD3BC_ALPHA_DEFAULT="$(get_spoke_field td3bc_alpha)"

DEVICE="${DEVICE:-cuda}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"
OBJECTIVE="${OBJECTIVE:-efficiency_v2}"
DATASET_EPISODES="${DATASET_EPISODES:-1000}"
DATASET_SEED="${DATASET_SEED:-0}"
COLLECT_WORKERS="${COLLECT_WORKERS:-8}"

# ---- Phase-driven seed / hyperparam defaults ----
case "$PHASE" in
  p1)
    SEEDS="${SEEDS:-42 44}"
    ACTOR_PENALTY_COEFS="${ACTOR_PENALTY_COEFS:-$ACTOR_PENALTY_DEFAULT}"
    CRITIC_PENALTY_COEFS="${CRITIC_PENALTY_COEFS:-$CRITIC_PENALTY_DEFAULT}"
    TD3BC_ALPHAS="${TD3BC_ALPHAS:-$TD3BC_ALPHA_DEFAULT}"
    ;;
  p2_refit)
    SEEDS="${SEEDS:-42}"
    ACTOR_PENALTY_COEFS="${ACTOR_PENALTY_COEFS:?ACTOR_PENALTY_COEFS required for p2_refit}"
    CRITIC_PENALTY_COEFS="${CRITIC_PENALTY_COEFS:?CRITIC_PENALTY_COEFS required for p2_refit}"
    TD3BC_ALPHAS="${TD3BC_ALPHAS:-$TD3BC_ALPHA_DEFAULT}"
    ;;
  p2_5seed)
    SEEDS="${SEEDS:-42 43 44 45 46}"
    ACTOR_PENALTY_COEFS="${ACTOR_PENALTY_COEFS:-$ACTOR_PENALTY_DEFAULT}"
    CRITIC_PENALTY_COEFS="${CRITIC_PENALTY_COEFS:-$CRITIC_PENALTY_DEFAULT}"
    TD3BC_ALPHAS="${TD3BC_ALPHAS:-$TD3BC_ALPHA_DEFAULT}"
    ;;
  *)
    echo "Unsupported PHASE: ${PHASE}" >&2
    exit 1
    ;;
esac

SAMPLING_MODE="${SAMPLING_MODE:-shuffle_no_replacement}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-64}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-8}"
DROP_LAST_BATCH="${DROP_LAST_BATCH:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-3}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.005}"
POLICY_NOISE="${POLICY_NOISE:-0.2}"
NOISE_CLIP="${NOISE_CLIP:-0.5}"
POLICY_FREQ="${POLICY_FREQ:-2}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
NORMALIZER_EPS="${NORMALIZER_EPS:-1e-3}"
LOG_EVERY="${LOG_EVERY:-1000}"
TRAIN_METRICS_WINDOW_FRACTION="${TRAIN_METRICS_WINDOW_FRACTION:-0.25}"

VAL_MANIFEST_EPISODES="${VAL_MANIFEST_EPISODES:-40}"
TEST_MANIFEST_EPISODES="${TEST_MANIFEST_EPISODES:-100}"
MANIFEST_ROOT="${MANIFEST_ROOT:-benchmarks/offline_rebrac_broad}"
VAL_MANIFEST_DIR="${VAL_MANIFEST_DIR:-${MANIFEST_ROOT}/val_${VAL_MANIFEST_EPISODES}}"
TEST_MANIFEST_DIR="${TEST_MANIFEST_DIR:-${MANIFEST_ROOT}/test_${TEST_MANIFEST_EPISODES}}"
VAL_MANIFEST_PATH="${VAL_MANIFEST_PATH:-${VAL_MANIFEST_DIR}/${BENCHMARK_KEY}.json}"
TEST_MANIFEST_PATH="${TEST_MANIFEST_PATH:-${TEST_MANIFEST_DIR}/${BENCHMARK_KEY}.json}"

EVAL_WORKERS="${EVAL_WORKERS:-6}"
EVAL_WORKER_DEVICE="${EVAL_WORKER_DEVICE:-cpu}"
VALIDATION_SEED="${VALIDATION_SEED:-123}"
TEST_SEED="${TEST_SEED:-456}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/offline/rebrac/broad_validation/${SPOKE_ID}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/offline/rebrac/broad_validation/${SPOKE_ID}}"

DATASET_DIR="offline_data/${DATASET_NAME}"

run_cmd() {
  echo
  echo "[cmd] $*"
  "$@"
}

ensure_manifest() {
  local episodes="$1" output_dir="$2" output_path="$3"
  if [[ -f "$output_path" ]]; then
    echo "[skip] manifest exists: $output_path"
    return
  fi
  mkdir -p "$output_dir"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.generate_standard_benchmarks \
    --benchmarks "$BENCHMARK_KEY" \
    --episodes "$episodes" \
    --output-dir "$output_dir"
}

ensure_manifests() {
  ensure_manifest "$VAL_MANIFEST_EPISODES" "$VAL_MANIFEST_DIR" "$VAL_MANIFEST_PATH"
  ensure_manifest "$TEST_MANIFEST_EPISODES" "$TEST_MANIFEST_DIR" "$TEST_MANIFEST_PATH"
}

ensure_dataset() {
  if [[ -f "${DATASET_DIR}/transitions.npz" ]]; then
    echo "[skip] dataset exists: ${DATASET_DIR}/transitions.npz"
    return
  fi
  mkdir -p "$DATASET_DIR"
  local extra_flags=()
  if [[ -n "$POLICY_MIXTURE" ]]; then
    extra_flags+=(--policy-mixture "$POLICY_MIXTURE")
  fi
  run_cmd "${PYTHON_CMD[@]}" -m scripts.collect_offline_data \
    --policy "$COLLECTOR_POLICY" \
    --flow "$FLOW_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --episodes "$DATASET_EPISODES" \
    --seed "$DATASET_SEED" \
    --num-workers "$COLLECT_WORKERS" \
    --output-dir "$DATASET_DIR" \
    "${extra_flags[@]}"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.write_sanity_card \
    --dataset-dir "$DATASET_DIR" \
    --expected-probe-layout "$PROBE_LAYOUT"
}

pair_tag_rebrac() {
  local actor_tag="${1//./p}"
  local critic_tag="${2//./p}"
  echo "actorb_${actor_tag}__criticb_${critic_tag}"
}
pair_tag_td3bc() {
  local alpha_tag="${1//./p}"
  echo "alpha_${alpha_tag}"
}

run_dir_for() {
  local pair="$1" seed="$2"
  echo "${CHECKPOINT_ROOT}/${pair}/seed_${seed}"
}
result_dir_for() {
  local pair="$1"
  echo "${RESULTS_ROOT}/${pair}"
}

compute_schedule() {
  local schedule_lines
  schedule_lines="$(
    "${PYTHON_CMD[@]}" - "$DATASET_DIR" "$BATCH_SIZE" "$TRAIN_EPOCHS" "$CHECKPOINT_EVERY_EPOCHS" "$DROP_LAST_BATCH" <<'PY'
import json, math, sys
from pathlib import Path
dataset_dir = Path(sys.argv[1])
batch_size = max(1, int(sys.argv[2]))
train_epochs = max(1, int(sys.argv[3]))
checkpoint_every_epochs = max(1, int(sys.argv[4]))
drop_last = sys.argv[5] == "1"
metadata_path = dataset_dir / "metadata.json"
num_transitions = None
if metadata_path.exists():
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = data.get("num_transitions")
    if value is not None:
        num_transitions = int(value)
if num_transitions is None:
    import numpy as np
    with np.load(dataset_dir / "transitions.npz", mmap_mode="r") as payload:
        num_transitions = int(payload["obs"].shape[0])
if drop_last:
    steps_per_epoch = num_transitions // batch_size
    if steps_per_epoch <= 0:
        raise ValueError("drop_last_batch=True requires num_transitions >= batch_size.")
else:
    steps_per_epoch = max(1, math.ceil(num_transitions / batch_size))
total_steps = steps_per_epoch * train_epochs
checkpoint_every_steps = max(1, steps_per_epoch * checkpoint_every_epochs)
print(f"total_steps={total_steps}")
print(f"checkpoint_every_steps={checkpoint_every_steps}")
PY
  )"
  SCHEDULE_TOTAL_STEPS=""
  SCHEDULE_CHECKPOINT_EVERY_STEPS=""
  while IFS='=' read -r key value; do
    case "$key" in
      total_steps) SCHEDULE_TOTAL_STEPS="$value" ;;
      checkpoint_every_steps) SCHEDULE_CHECKPOINT_EVERY_STEPS="$value" ;;
    esac
  done <<< "$schedule_lines"
}

train_one_rebrac() {
  local actor_beta="$1" critic_beta="$2" seed="$3"
  local pair run_dir
  pair="$(pair_tag_rebrac "$actor_beta" "$critic_beta")"
  run_dir="$(run_dir_for "$pair" "$seed")"
  if [[ -f "${run_dir}/trainer_state.json" && -f "${run_dir}/agent_final.pt" ]]; then
    echo "[skip] trained run exists: ${run_dir}"
    return
  fi
  mkdir -p "$run_dir"
  local extra_flags=()
  if [[ "$DROP_LAST_BATCH" == "1" ]]; then
    extra_flags+=(--drop-last-batch)
  fi
  run_cmd "${PYTHON_CMD[@]}" -m scripts.train_offline \
    --algo rebrac \
    --offline-data "${DATASET_DIR}/transitions.npz" \
    --flow "$FLOW_PATH" \
    --manifest "$VAL_MANIFEST_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --sampling-mode "$SAMPLING_MODE" \
    --num-epochs "$TRAIN_EPOCHS" \
    --total-steps "$SCHEDULE_TOTAL_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
    --actor-lr "$ACTOR_LR" \
    --critic-lr "$CRITIC_LR" \
    --gamma "$GAMMA" \
    --tau "$TAU" \
    --actor-penalty-coef "$actor_beta" \
    --critic-penalty-coef "$critic_beta" \
    --policy-noise "$POLICY_NOISE" \
    --noise-clip "$NOISE_CLIP" \
    --policy-freq "$POLICY_FREQ" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --normalizer-eps "$NORMALIZER_EPS" \
    --eval-every 0 \
    --checkpoint-every "$SCHEDULE_CHECKPOINT_EVERY_STEPS" \
    --skip-final-eval \
    --log-every "$LOG_EVERY" \
    --critic-layernorm \
    --no-actor-layernorm \
    --save-dir "$run_dir" \
    --seed "$seed" \
    --device "$DEVICE" \
    "${extra_flags[@]}"
}

train_one_td3bc() {
  local alpha="$1" seed="$2"
  local pair run_dir
  pair="$(pair_tag_td3bc "$alpha")"
  run_dir="$(run_dir_for "$pair" "$seed")"
  if [[ -f "${run_dir}/trainer_state.json" && -f "${run_dir}/agent_final.pt" ]]; then
    echo "[skip] trained run exists: ${run_dir}"
    return
  fi
  mkdir -p "$run_dir"
  local extra_flags=()
  if [[ "$DROP_LAST_BATCH" == "1" ]]; then
    extra_flags+=(--drop-last-batch)
  fi
  run_cmd "${PYTHON_CMD[@]}" -m scripts.train_offline \
    --algo td3bc \
    --offline-data "${DATASET_DIR}/transitions.npz" \
    --flow "$FLOW_PATH" \
    --manifest "$VAL_MANIFEST_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --sampling-mode "$SAMPLING_MODE" \
    --num-epochs "$TRAIN_EPOCHS" \
    --total-steps "$SCHEDULE_TOTAL_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
    --actor-lr "$ACTOR_LR" \
    --critic-lr "$CRITIC_LR" \
    --gamma "$GAMMA" \
    --tau "$TAU" \
    --alpha "$alpha" \
    --policy-noise "$POLICY_NOISE" \
    --noise-clip "$NOISE_CLIP" \
    --policy-freq "$POLICY_FREQ" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --normalizer-eps "$NORMALIZER_EPS" \
    --eval-every 0 \
    --checkpoint-every "$SCHEDULE_CHECKPOINT_EVERY_STEPS" \
    --skip-final-eval \
    --log-every "$LOG_EVERY" \
    --critic-layernorm \
    --no-actor-layernorm \
    --save-dir "$run_dir" \
    --seed "$seed" \
    --device "$DEVICE" \
    "${extra_flags[@]}"
}

validate_one() {
  local pair="$1" seed="$2"
  local run_dir result_dir val_dir
  run_dir="$(run_dir_for "$pair" "$seed")"
  result_dir="$(result_dir_for "$pair")"
  val_dir="${result_dir}/validation/seed_${seed}"
  if [[ ! -d "$run_dir" ]]; then
    echo "[skip] missing run dir: ${run_dir}"
    return
  fi
  if [[ ! -f "${run_dir}/trainer_state.json" || ! -f "${run_dir}/agent_final.pt" ]]; then
    echo "[warn] skipping ${run_dir}: missing trainer_state.json or agent_final.pt" >&2
    return
  fi
  mkdir -p "$val_dir"
  local checkpoint_files=()
  shopt -s nullglob
  local _step_files=("$run_dir"/agent_step_*.pt)
  shopt -u nullglob
  if [[ "${#_step_files[@]}" -gt 0 ]]; then
    while IFS= read -r _path; do
      checkpoint_files+=("$(basename "$_path")")
    done < <(printf '%s\n' "${_step_files[@]}" | sort)
  fi
  checkpoint_files+=("agent_final.pt")
  for agent_file in "${checkpoint_files[@]}"; do
    local agent_tag="${agent_file%.pt}"
    local output_json="${val_dir}/${agent_tag}.json"
    if [[ -f "$output_json" && "$FORCE_REEVAL" != "1" ]]; then
      echo "[skip] validation exists: ${output_json}"
      continue
    fi
    run_cmd "${PYTHON_CMD[@]}" -m scripts.evaluate_offline \
      --checkpoint "$run_dir" \
      --agent-file "$agent_file" \
      --manifest "$VAL_MANIFEST_PATH" \
      --device "$DEVICE" \
      --num-workers "$EVAL_WORKERS" \
      --worker-device "$EVAL_WORKER_DEVICE" \
      --seed "$VALIDATION_SEED" \
      --output-json "$output_json"
  done
}

select_best_checkpoint() {
  local pair="$1" seed="$2"
  local run_dir result_dir val_dir selection_dir output_json trainer_state_path
  run_dir="$(run_dir_for "$pair" "$seed")"
  result_dir="$(result_dir_for "$pair")"
  val_dir="${result_dir}/validation/seed_${seed}"
  selection_dir="${result_dir}/selection/seed_${seed}"
  output_json="${selection_dir}/selected_checkpoint.json"
  trainer_state_path="${run_dir}/trainer_state.json"
  if [[ -f "$output_json" && "$FORCE_REEVAL" != "1" ]]; then
    echo "[skip] checkpoint selection exists: ${output_json}"
    return
  fi
  if [[ ! -d "$val_dir" ]]; then
    echo "[skip] missing validation dir: ${val_dir}"
    return
  fi
  mkdir -p "$selection_dir"
  local agent_file_sidecar="${selection_dir}/selected_agent_file.txt"
  run_cmd "${PYTHON_CMD[@]}" - "$val_dir" "$trainer_state_path" "$output_json" "$agent_file_sidecar" <<'PY'
import json, sys
from pathlib import Path
eval_dir = Path(sys.argv[1])
trainer_state_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
agent_file_sidecar = Path(sys.argv[4])
trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
train_step = int(trainer_state.get("train_step", 0))
records = []
for json_path in sorted(eval_dir.glob("*.json")):
    if json_path.name == output_path.name:
        continue
    metrics = json.loads(json_path.read_text(encoding="utf-8"))
    agent_file = f"{json_path.stem}.pt"
    if json_path.stem.startswith("agent_step_"):
        try:
            step = int(json_path.stem.split("_")[-1])
        except ValueError:
            step = None
    elif json_path.stem == "agent_final":
        step = train_step
    else:
        step = None
    records.append({
        "agent_file": agent_file,
        "agent_tag": json_path.stem,
        "train_step": step,
        "eval_success_rate": float(metrics["eval_success_rate"]),
        "eval_return": float(metrics["eval_return"]),
        "eval_safety_cost": float(metrics["eval_safety_cost"]),
        "eval_time_s": float(metrics["eval_time_s"]),
        "metrics": metrics,
    })
if not records:
    raise ValueError(f"No validation metrics found under {eval_dir}")
best = max(records, key=lambda item: (
    item["eval_success_rate"], item["eval_return"],
    -item["eval_safety_cost"], -item["eval_time_s"],
))
payload = {
    "selection_metric": "eval_success_rate -> eval_return -> -eval_safety_cost -> -eval_time_s",
    "num_candidates": len(records),
    "best": best,
    "candidates": records,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
agent_file_sidecar.parent.mkdir(parents=True, exist_ok=True)
agent_file_sidecar.write_text(best["agent_file"] + "\n", encoding="utf-8")
print(f"[write] selection: {output_path}")
print(f"[best] agent={best['agent_file']} success={best['eval_success_rate']:.4f}")
PY
}

test_one() {
  local pair="$1" seed="$2"
  local run_dir result_dir selection_dir selection_path agent_file_sidecar test_dir output_json agent_file
  run_dir="$(run_dir_for "$pair" "$seed")"
  result_dir="$(result_dir_for "$pair")"
  selection_dir="${result_dir}/selection/seed_${seed}"
  selection_path="${selection_dir}/selected_checkpoint.json"
  agent_file_sidecar="${selection_dir}/selected_agent_file.txt"
  test_dir="${result_dir}/test"
  output_json="${test_dir}/seed_${seed}.json"
  if [[ ! -f "$selection_path" ]]; then
    echo "[skip] missing selection: ${selection_path}"
    return
  fi
  if [[ -f "$output_json" && "$FORCE_REEVAL" != "1" ]]; then
    echo "[skip] test exists: ${output_json}"
    return
  fi
  mkdir -p "$test_dir"
  if [[ -f "$agent_file_sidecar" ]]; then
    agent_file="$(< "$agent_file_sidecar")"
    agent_file="${agent_file//$'\n'/}"
  else
    agent_file="$("${PYTHON_CMD[@]}" - "$selection_path" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["best"]["agent_file"])
PY
)"
  fi
  run_cmd "${PYTHON_CMD[@]}" -m scripts.evaluate_offline \
    --checkpoint "$run_dir" \
    --agent-file "$agent_file" \
    --manifest "$TEST_MANIFEST_PATH" \
    --device "$DEVICE" \
    --num-workers "$EVAL_WORKERS" \
    --worker-device "$EVAL_WORKER_DEVICE" \
    --seed "$TEST_SEED" \
    --output-json "$output_json"
}

# ---- Top-level execution ----
ensure_dataset
ensure_manifests
compute_schedule

if [[ "$ALGO" == "rebrac" ]]; then
  for actor_beta in $ACTOR_PENALTY_COEFS; do
    for critic_beta in $CRITIC_PENALTY_COEFS; do
      pair="$(pair_tag_rebrac "$actor_beta" "$critic_beta")"
      for seed in $SEEDS; do
        train_one_rebrac "$actor_beta" "$critic_beta" "$seed"
        validate_one "$pair" "$seed"
        select_best_checkpoint "$pair" "$seed"
        test_one "$pair" "$seed"
      done
    done
  done
elif [[ "$ALGO" == "td3bc" ]]; then
  for alpha in $TD3BC_ALPHAS; do
    pair="$(pair_tag_td3bc "$alpha")"
    for seed in $SEEDS; do
      train_one_td3bc "$alpha" "$seed"
      validate_one "$pair" "$seed"
      select_best_checkpoint "$pair" "$seed"
      test_one "$pair" "$seed"
    done
  done
else
  echo "Unsupported ALGO: ${ALGO}" >&2
  exit 1
fi

echo
echo "[done] SPOKE_ID=${SPOKE_ID} PHASE=${PHASE}"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/run_offline_rebrac_broad.sh
```

- [ ] **Step 3: Smoke-check the dispatch logic without launching training**

The script's `ensure_dataset` step requires real wake_data + a torch install. We only verify the registry lookup path executes:

```bash
SPOKE_ID=A1 PHASE=p1 bash -n scripts/run_offline_rebrac_broad.sh
SPOKE_ID=A2-td3bc PHASE=p1 bash -n scripts/run_offline_rebrac_broad.sh
```

Expected: both return exit 0 (syntax check only). Real execution happens in Colab.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_offline_rebrac_broad.sh
git commit -m "feat(rebrac-broad): per-spoke driver with PHASE-driven seed/hyperparam defaults"
```

---

## Task 5: Summary + Trigger Gate

**Files:**
- Create: `scripts/summarize_broad_validation.py`
- Test: `tests/test_summarize_broad_validation.py`

This script reads each spoke's `test/seed_*.json` outputs and produces:
1. `results/offline/rebrac/broad_validation/summaries/p1_overview.csv` — per (spoke, pair) mean/std across seeds.
2. `results/offline/rebrac/broad_validation/summaries/trigger_decisions.json` — for each spoke, which trigger (mean shift / std blow-up / TD3BC gap collapse) fires; cites spec §6.1.
3. `results/offline/rebrac/broad_validation/summaries/p2_overview.csv` (after P2) — same for P2 deepening.

The anchor reference (success = 0.902 ± 0.021) is hardcoded per spec §3.1.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summarize_broad_validation.py`:

```python
"""Unit tests for the broad-validation summary + trigger gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_broad_validation import (
    ANCHOR_MEAN,
    ANCHOR_STD,
    apply_trigger_gate,
    collect_p1_results,
    summarize_p1,
)


def _write_test_json(path: Path, success: float, rtn: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "eval_success_rate": success,
        "eval_return": rtn,
        "eval_safety_cost": 0.0,
        "eval_time_s": 1.0,
    }))


def test_anchor_constants_match_spec() -> None:
    assert ANCHOR_MEAN == pytest.approx(0.902)
    assert ANCHOR_STD == pytest.approx(0.021)


def test_collect_p1_walks_directory(tmp_path: Path) -> None:
    base = tmp_path / "results" / "offline" / "rebrac" / "broad_validation"
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.5)
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.6)
    _write_test_json(base / "B2" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.95)
    rows = collect_p1_results(base)
    spokes = {r["spoke_id"] for r in rows}
    assert spokes == {"A1", "B2"}


def test_summarize_aggregates_seeds(tmp_path: Path) -> None:
    base = tmp_path / "broad"
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.5)
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.7)
    rows = summarize_p1(base)
    target = next(r for r in rows if r["spoke_id"] == "A1")
    assert target["num_seeds"] == 2
    assert target["mean_test_success_rate"] == pytest.approx(0.6)
    assert target["std_test_success_rate"] > 0


def test_trigger_mean_shift_negative(tmp_path: Path) -> None:
    base = tmp_path / "broad"
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.5)
    _write_test_json(base / "A1" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.5)
    decisions = apply_trigger_gate(summarize_p1(base))
    a1 = next(d for d in decisions if d["spoke_id"] == "A1")
    assert a1["triggered"] is True
    assert "mean_shift" in a1["reasons"]
    assert a1["delta_pp"] < 0


def test_trigger_mean_shift_positive(tmp_path: Path) -> None:
    """Spec §6.1: positive 5pp shift also triggers (bidirectional)."""
    base = tmp_path / "broad"
    _write_test_json(base / "B2" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.97)
    _write_test_json(base / "B2" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.98)
    decisions = apply_trigger_gate(summarize_p1(base))
    b2 = next(d for d in decisions if d["spoke_id"] == "B2")
    assert b2["triggered"] is True


def test_trigger_a2_rebrac_vs_td3bc_gap_collapse(tmp_path: Path) -> None:
    base = tmp_path / "broad"
    _write_test_json(base / "A2" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.85)
    _write_test_json(base / "A2" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.85)
    _write_test_json(base / "A2-td3bc" / "alpha_0p25" / "test" / "seed_42.json", 0.83)
    _write_test_json(base / "A2-td3bc" / "alpha_0p25" / "test" / "seed_44.json", 0.83)
    decisions = apply_trigger_gate(summarize_p1(base))
    a2 = next(d for d in decisions if d["spoke_id"] == "A2")
    assert a2["triggered"] is True
    assert "td3bc_gap_collapse" in a2["reasons"]


def test_no_trigger_within_tolerance(tmp_path: Path) -> None:
    base = tmp_path / "broad"
    _write_test_json(base / "B1" / "actorb_4p0__criticb_2p0" / "test" / "seed_42.json", 0.91)
    _write_test_json(base / "B1" / "actorb_4p0__criticb_2p0" / "test" / "seed_44.json", 0.92)
    decisions = apply_trigger_gate(summarize_p1(base))
    b1 = next(d for d in decisions if d["spoke_id"] == "B1")
    assert b1["triggered"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=. pytest tests/test_summarize_broad_validation.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the summary script**

Create `scripts/summarize_broad_validation.py`:

```python
"""Aggregate broad-validation P1/P2 results and apply the trigger gate.

Spec §3.1 anchor reference: success = 0.902 +/- 0.021.
Spec §6.1 trigger judgment:
  - |mean(spoke 2-seed) - 0.902| > 0.05  (bidirectional mean shift)
  - std(spoke 2-seed) > 2 * 0.021 = 0.042 (std blow-up)
  - For A2 only: |mean_ReBRAC - mean_TD3BC| < 0.05 (gap collapse)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ANCHOR_MEAN = 0.902
ANCHOR_STD = 0.021
MEAN_SHIFT_THRESHOLD = 0.05
STD_BLOW_UP_THRESHOLD = 2 * ANCHOR_STD
TD3BC_GAP_THRESHOLD = 0.05
WINNER_DRIFT_THRESHOLD = 0.03  # Spec §6.2


def _safe_mean(values: list[float]) -> float | None:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(cleaned) if cleaned else None


def _safe_std(values: list[float]) -> float:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0


def collect_p1_results(broad_root: Path) -> list[dict[str, Any]]:
    """Walk results/offline/rebrac/broad_validation/<spoke>/<pair>/test/seed_*.json."""
    rows: list[dict[str, Any]] = []
    if not broad_root.exists():
        return rows
    for spoke_dir in sorted(p for p in broad_root.iterdir() if p.is_dir()):
        if spoke_dir.name == "summaries":
            continue
        for pair_dir in sorted(p for p in spoke_dir.iterdir() if p.is_dir()):
            test_dir = pair_dir / "test"
            if not test_dir.is_dir():
                continue
            for json_path in sorted(test_dir.glob("seed_*.json")):
                metrics = json.loads(json_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "spoke_id": spoke_dir.name,
                        "pair": pair_dir.name,
                        "seed": int(json_path.stem.replace("seed_", "")),
                        "eval_success_rate": float(metrics["eval_success_rate"]),
                        "eval_return": float(metrics["eval_return"]),
                        "eval_safety_cost": float(metrics["eval_safety_cost"]),
                        "eval_time_s": float(metrics["eval_time_s"]),
                    }
                )
    return rows


def summarize_p1(broad_root: Path) -> list[dict[str, Any]]:
    rows = collect_p1_results(broad_root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["spoke_id"], row["pair"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (spoke_id, pair), seeds in sorted(grouped.items()):
        success = [r["eval_success_rate"] for r in seeds]
        returns = [r["eval_return"] for r in seeds]
        out.append(
            {
                "spoke_id": spoke_id,
                "pair": pair,
                "num_seeds": len(seeds),
                "seeds": sorted(r["seed"] for r in seeds),
                "mean_test_success_rate": _safe_mean(success),
                "std_test_success_rate": _safe_std(success),
                "mean_test_return": _safe_mean(returns),
                "std_test_return": _safe_std(returns),
            }
        )
    return out


def apply_trigger_gate(p1_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per spoke, decide whether P2 deepening is required.

    Returns a list of decisions, one per spoke (A2-td3bc folded into A2).
    """
    by_spoke: dict[str, list[dict[str, Any]]] = {}
    for row in p1_summary:
        by_spoke.setdefault(row["spoke_id"], []).append(row)

    decisions: list[dict[str, Any]] = []
    rebrac_spokes = sorted(s for s in by_spoke if s != "A2-td3bc")
    for spoke_id in rebrac_spokes:
        # The spoke's primary anchor pair is the one with the most seeds.
        rebrac_rows = sorted(
            by_spoke[spoke_id], key=lambda r: r["num_seeds"], reverse=True
        )
        primary = rebrac_rows[0]
        mean_val = primary["mean_test_success_rate"] or 0.0
        std_val = primary["std_test_success_rate"] or 0.0
        delta_pp = mean_val - ANCHOR_MEAN
        reasons: list[str] = []
        if abs(delta_pp) > MEAN_SHIFT_THRESHOLD:
            reasons.append("mean_shift")
        if std_val > STD_BLOW_UP_THRESHOLD:
            reasons.append("std_blow_up")
        if spoke_id == "A2" and "A2-td3bc" in by_spoke:
            td3bc_primary = sorted(
                by_spoke["A2-td3bc"],
                key=lambda r: r["num_seeds"],
                reverse=True,
            )[0]
            td3bc_mean = td3bc_primary["mean_test_success_rate"] or 0.0
            if abs(mean_val - td3bc_mean) < TD3BC_GAP_THRESHOLD:
                reasons.append("td3bc_gap_collapse")
        decisions.append(
            {
                "spoke_id": spoke_id,
                "pair": primary["pair"],
                "num_seeds": primary["num_seeds"],
                "mean_test_success_rate": mean_val,
                "std_test_success_rate": std_val,
                "delta_pp": delta_pp,
                "triggered": bool(reasons),
                "reasons": reasons,
            }
        )
    return decisions


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = {
                k: (",".join(str(v) for v in val) if isinstance(val, list) else val)
                for k, val in row.items()
            }
            writer.writerow(row_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize broad validation.")
    parser.add_argument(
        "--broad-root",
        type=Path,
        default=Path("results/offline/rebrac/broad_validation"),
    )
    parser.add_argument(
        "--summaries-dir",
        type=Path,
        default=None,
        help="Override summaries output dir (defaults to <broad-root>/summaries).",
    )
    args = parser.parse_args()

    summaries_dir = args.summaries_dir or (args.broad_root / "summaries")
    summaries_dir.mkdir(parents=True, exist_ok=True)

    p1_summary = summarize_p1(args.broad_root)
    write_csv(p1_summary, summaries_dir / "p1_overview.csv")
    (summaries_dir / "p1_overview.json").write_text(
        json.dumps(p1_summary, indent=2), encoding="utf-8"
    )

    decisions = apply_trigger_gate(p1_summary)
    (summaries_dir / "trigger_decisions.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8"
    )

    triggered = [d for d in decisions if d["triggered"]]
    print(f"[summary] P1 spokes summarized: {len(p1_summary)}")
    print(f"[summary] triggered for P2: {len(triggered)}")
    for d in triggered:
        print(
            f"  - {d['spoke_id']}: mean={d['mean_test_success_rate']:.3f} "
            f"std={d['std_test_success_rate']:.3f} "
            f"delta_pp={d['delta_pp']:+.3f} "
            f"reasons={','.join(d['reasons'])}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=. pytest tests/test_summarize_broad_validation.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/summarize_broad_validation.py tests/test_summarize_broad_validation.py
git commit -m "feat(rebrac-broad): summary + trigger gate (spec §6.1)"
```

---

## Task 6: Colab Notebook Scaffold

**Files:**
- Create: `notebooks/rebrac_broad_validation.ipynb`

The notebook is the user-facing Colab driver. Cell layout:

| Cell | Type | Purpose |
|---|---|---|
| 1 | markdown | Overview + spec/plan links |
| 2 | code | Mount Drive, `cd` into project, sanity print of git rev |
| 3 | code | Loop: collect 7 datasets via `bash scripts/run_offline_rebrac_broad.sh` with `MODE=collect_only` (each spoke triggers `ensure_dataset` + sanity card) — but this driver is unified, so we collect by running per-spoke `PHASE=p1` after spec adjustment OR run dedicated collect cells. **Implementation: a Python loop that calls `scripts.collect_offline_data` directly per spoke spec, plus the A2 mix concat.** |
| 4 | code | A2 mix5050 concat: collect `mix_tmp_goalseek` + `mix_tmp_crosscomp`, then call `scripts.concat_offline_datasets`. |
| 5 | code | Write all sanity cards via `scripts.write_sanity_card`. |
| 6 | code | Run P1 training: loop over the 8 spokes invoking `scripts/run_offline_rebrac_broad.sh PHASE=p1`. |
| 7 | code | Run summary + trigger gate: `python -m scripts.summarize_broad_validation`. |
| 8 | code | Print + commit `trigger_decisions.json` for review. |
| 9 | markdown | "If any spokes triggered: edit cell 10 to specify the SPOKE_IDs and the β refit grid; otherwise skip to cell 12." |
| 10 | code | P2 β refit (conditional, manual edit). |
| 11 | code | P2 5-seed expansion (conditional, manual edit). |
| 12 | code | Re-run summary, write final p2_overview.csv. |
| 13 | code | Sync results back to Drive checkpoint dir + commit summary CSVs to git. |

- [ ] **Step 1: Generate the notebook deterministically via a one-off helper**

Rather than hand-rolling the JSON, write a tiny Python helper that constructs the notebook from a list of cell sources, then run it once. Create `scripts/_generate_broad_validation_notebook.py` (this file is throw-away — delete after the notebook is generated):

```python
"""One-off helper: build notebooks/rebrac_broad_validation.ipynb.

Run once via `python -m scripts._generate_broad_validation_notebook`, then
delete this file. Provided as a script (not inlined in a bash heredoc) so the
notebook content stays readable.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


CELLS: list[tuple[str, str]] = [
    ("markdown", _MD_OVERVIEW),
    ("code", _CODE_MOUNT),
    ("code", _CODE_COLLECT_SINGLE),
    ("code", _CODE_COLLECT_MIX),
    ("code", _CODE_SANITY_CARDS),
    ("code", _CODE_P1),
    ("code", _CODE_SUMMARIZE),
    ("code", _CODE_REVIEW_TRIGGERS),
    ("markdown", _MD_P2_INSTRUCTIONS),
    ("code", _CODE_P2_REFIT),
    ("code", _CODE_P2_5SEED),
    ("code", _CODE_FINAL_SUMMARIZE),
    ("code", _CODE_COMMIT),
]
```

Each `_MD_*` / `_CODE_*` constant in that helper is the corresponding cell body shown verbatim below in steps 2–14. Set them as module-level string constants. The helper body finishes with:

```python
def main() -> None:
    nb = nbf.v4.new_notebook()
    for cell_type, source in CELLS:
        if cell_type == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbf.v4.new_code_cell(source))
    out = Path("notebooks/rebrac_broad_validation.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
```

The cell bodies follow verbatim:

Markdown cell 1:
```markdown
# ReBRAC Broad Validation — Colab Driver

Spec: `docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`
Plan: `docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md`

This notebook drives 4 Colab Pro / L4 sessions:

- **S1**: collect 7 new datasets (~6 h)
- **S2**: P1 training — 7 ReBRAC spokes × 2 seeds + A2 TD3+BC × 2 seeds (~8 h)
- **S3**: conditional P2 — β refit + 5-seed extension on triggered spokes (~5–8 h)
- **S4**: buffer / analysis (~2 h)

Run cells sequentially; each cell is `[skip]`-resume safe via `transitions.npz` /
`trainer_state.json` / `agent_final.pt` existence checks.
```

Code cell 2 — mount + cd + git rev:
```python
import os
import subprocess
from pathlib import Path

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    PROJECT_ROOT = Path("/content/drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5")
except ModuleNotFoundError:
    PROJECT_ROOT = Path.cwd()

os.chdir(PROJECT_ROOT)
print("cwd:", Path.cwd())
print("git rev:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip())
print("git status:")
print(subprocess.check_output(["git", "status", "--short"], text=True))
```

Code cell 3 — collect datasets (single-policy spokes):
```python
import subprocess
from scripts.broad_validation_spoke_registry import REGISTRY

SINGLE_POLICY_SPOKES = ["A1", "A3", "B1", "B2", "C1", "C3"]
for spoke_id in SINGLE_POLICY_SPOKES:
    cfg = REGISTRY[spoke_id]
    out_dir = Path("offline_data") / cfg.dataset_name
    if (out_dir / "transitions.npz").exists():
        print(f"[skip] {spoke_id}: dataset exists")
        continue
    cmd = [
        "python", "-m", "scripts.collect_offline_data",
        "--policy", cfg.collector_policy,
        "--flow", cfg.flow_path,
        "--probe-layout", cfg.probe_layout,
        "--task-geometry", cfg.task_geometry,
        "--target-speed", str(cfg.target_speed),
        "--history-length", "4",
        "--objective", "efficiency_v2",
        "--episodes", "1000",
        "--seed", "0",
        "--num-workers", "8",
        "--output-dir", str(out_dir),
    ]
    print(f"[collect] {spoke_id}:", " ".join(cmd))
    subprocess.run(cmd, check=True)
```

Code cell 4 — A2 mix5050 collect + concat:
```python
import subprocess
from pathlib import Path
from scripts.broad_validation_spoke_registry import REGISTRY

cfg = REGISTRY["A2"]
final_dir = Path("offline_data") / cfg.dataset_name

if (final_dir / "transitions.npz").exists():
    print(f"[skip] A2 mix already exists at {final_dir}")
else:
    sub_a = Path("offline_data") / "mix_tmp_goalseek_500ep_seed0"
    sub_b = Path("offline_data") / "mix_tmp_crosscomp_500ep_seed1"
    common = [
        "--flow", cfg.flow_path,
        "--probe-layout", cfg.probe_layout,
        "--task-geometry", cfg.task_geometry,
        "--target-speed", str(cfg.target_speed),
        "--history-length", "4",
        "--objective", "efficiency_v2",
        "--episodes", "500",
        "--num-workers", "8",
    ]
    if not (sub_a / "transitions.npz").exists():
        subprocess.run(
            ["python", "-m", "scripts.collect_offline_data",
             "--policy", "goalseek", "--seed", "0",
             "--output-dir", str(sub_a), *common],
            check=True,
        )
    if not (sub_b / "transitions.npz").exists():
        subprocess.run(
            ["python", "-m", "scripts.collect_offline_data",
             "--policy", "crosscomp", "--seed", "1",
             "--output-dir", str(sub_b), *common],
            check=True,
        )
    subprocess.run(
        ["python", "-m", "scripts.concat_offline_datasets",
         "--input-dir", str(sub_a),
         "--input-dir", str(sub_b),
         "--output-dir", str(final_dir),
         "--mix-strategy", "episode_level",
         "--task-sampler", "anchor_distribution"],
        check=True,
    )
    print(f"[done] A2 mix at {final_dir}")
```

Code cell 5 — write all sanity cards:
```python
import subprocess
from pathlib import Path
from scripts.broad_validation_spoke_registry import REGISTRY

UNIQUE_DATASETS = {
    cfg.dataset_name: cfg.probe_layout
    for cfg in REGISTRY.values()
}
for dataset_name, probe in UNIQUE_DATASETS.items():
    dataset_dir = Path("offline_data") / dataset_name
    if not (dataset_dir / "transitions.npz").exists():
        print(f"[warn] missing dataset: {dataset_dir}")
        continue
    subprocess.run(
        ["python", "-m", "scripts.write_sanity_card",
         "--dataset-dir", str(dataset_dir),
         "--expected-probe-layout", probe],
        check=True,
    )
```

Code cell 6 — P1 training loop:
```python
import os, subprocess
from scripts.broad_validation_spoke_registry import REGISTRY

P1_ORDER = ["A1", "A2", "A2-td3bc", "A3", "B1", "B2", "C1", "C3"]
for spoke_id in P1_ORDER:
    env = os.environ.copy()
    env["SPOKE_ID"] = spoke_id
    env["PHASE"] = "p1"
    env["DEVICE"] = "cuda"
    print(f"\n========== P1: {spoke_id} ==========")
    subprocess.run(
        ["bash", "scripts/run_offline_rebrac_broad.sh"],
        env=env, check=True,
    )
```

Code cell 7 — summary + trigger gate:
```python
import subprocess
subprocess.run(
    ["python", "-m", "scripts.summarize_broad_validation"],
    check=True,
)
```

Code cell 8 — review trigger decisions:
```python
import json
from pathlib import Path

decisions_path = Path("results/offline/rebrac/broad_validation/summaries/trigger_decisions.json")
decisions = json.loads(decisions_path.read_text())
triggered = [d for d in decisions if d["triggered"]]
print(f"Triggered spokes ({len(triggered)}):")
for d in triggered:
    print(f"  {d['spoke_id']}: delta_pp={d['delta_pp']:+.3f} reasons={d['reasons']}")
print()
print("Untriggered spokes:")
for d in decisions:
    if not d["triggered"]:
        print(f"  {d['spoke_id']}: delta_pp={d['delta_pp']:+.3f}")
```

Markdown cell 9:
```markdown
## P2 deepening (conditional)

Edit cell 10 below: set `TRIGGERED_SPOKES` to the spoke IDs from cell 8 that
need β refit. The β refit runs (β1=2, β2=2) and (β1=4, β2=1) on seed 42 only
(spec §6.2). After cell 10, inspect `summaries/p1_overview.csv` to determine
the winner. Edit cell 11 with the winner per spoke, then run cell 11 to extend
to 5 seeds.

If `len(triggered) == 0`, skip cells 10 and 11.
```

Code cell 10 — P2 β refit:
```python
import os, subprocess

# EDIT THIS LIST after running cell 8.
TRIGGERED_SPOKES: list[str] = []  # e.g. ["A1", "C3"]

for spoke_id in TRIGGERED_SPOKES:
    for actor_b, critic_b in [("2.0", "2.0"), ("4.0", "1.0")]:
        env = os.environ.copy()
        env.update({
            "SPOKE_ID": spoke_id,
            "PHASE": "p2_refit",
            "ACTOR_PENALTY_COEFS": actor_b,
            "CRITIC_PENALTY_COEFS": critic_b,
        })
        print(f"\n========== P2 refit: {spoke_id} (β1={actor_b}, β2={critic_b}) ==========")
        subprocess.run(
            ["bash", "scripts/run_offline_rebrac_broad.sh"],
            env=env, check=True,
        )
```

Code cell 11 — P2 5-seed expansion:
```python
import os, subprocess

# EDIT THIS DICT after inspecting refit results.
# Winner format: {"A1": ("4.0", "2.0"), "C3": ("2.0", "2.0"), ...}
WINNERS: dict[str, tuple[str, str]] = {}

for spoke_id, (actor_b, critic_b) in WINNERS.items():
    env = os.environ.copy()
    env.update({
        "SPOKE_ID": spoke_id,
        "PHASE": "p2_5seed",
        "ACTOR_PENALTY_COEFS": actor_b,
        "CRITIC_PENALTY_COEFS": critic_b,
    })
    print(f"\n========== P2 5-seed: {spoke_id} (β1={actor_b}, β2={critic_b}) ==========")
    subprocess.run(
        ["bash", "scripts/run_offline_rebrac_broad.sh"],
        env=env, check=True,
    )
```

Code cell 12 — final summary:
```python
import subprocess
subprocess.run(
    ["python", "-m", "scripts.summarize_broad_validation"],
    check=True,
)
```

Code cell 13 — commit summary CSVs to git:
```python
import subprocess
subprocess.run(["git", "add",
                "results/offline/rebrac/broad_validation/summaries/p1_overview.csv",
                "results/offline/rebrac/broad_validation/summaries/p1_overview.json",
                "results/offline/rebrac/broad_validation/summaries/trigger_decisions.json"],
               check=True)
print(subprocess.check_output(["git", "status", "--short"], text=True))
```

- [ ] **Step 2: Run the helper to write the notebook**

```bash
python -m scripts._generate_broad_validation_notebook
```

Expected: `[write] notebooks/rebrac_broad_validation.ipynb`.

- [ ] **Step 3: Delete the throw-away generator**

```bash
rm scripts/_generate_broad_validation_notebook.py
```

- [ ] **Step 4: Verify the notebook is valid JSON**

```bash
python -c "import json; json.loads(open('notebooks/rebrac_broad_validation.ipynb').read()); print('valid')"
```

Expected: `valid`.

- [ ] **Step 5: Verify it parses as a notebook**

```bash
python -c "import nbformat; nb = nbformat.read('notebooks/rebrac_broad_validation.ipynb', as_version=4); print(len(nb.cells), 'cells')"
```

Expected: `13 cells`.

- [ ] **Step 6: Commit**

```bash
git add notebooks/rebrac_broad_validation.ipynb
git commit -m "feat(rebrac-broad): Colab driver notebook (collect → P1 → P2 → analyse)"
```

---

## Task 7: Sync to Google Drive

This is a manual user step. After committing the infrastructure, sync the project tree to the Drive mount point so Colab sees the new code.

- [ ] **Step 1: Sync the working tree to Drive**

User-side action (no automation in plan): `rsync -av --exclude='.git' --exclude='offline_data' --exclude='checkpoints' --exclude='wake_data' /Users/xiangjin/.../rl_v2/ /Users/xiangjin/.../GoogleDrive/MyDrive/Colab\ Notebooks/new_offRL/rl_v2_5/`.

Or via the user's existing Drive sync workflow.

- [ ] **Step 2: Verify in a fresh Colab cell**

```python
import subprocess
print(subprocess.check_output(
    ["bash", "-n", "scripts/run_offline_rebrac_broad.sh"],
    cwd="/content/drive/MyDrive/Colab Notebooks/new_offRL/rl_v2_5",
    text=True,
))
```

Expected: empty output, exit 0.

---

## Task 8: Colab S1 — Data Collection (~6 h L4)

Run notebook cells 2–5 in a single Colab session. The notebook itself is `[skip]`-resume safe.

- [ ] **Step 1: Open `notebooks/rebrac_broad_validation.ipynb` in Colab**

- [ ] **Step 2: Run cell 2 (mount + cd)**

Expected: prints cwd, short git rev, status.

- [ ] **Step 3: Run cell 3 (collect 6 single-policy datasets)**

Expected: 6 datasets are collected sequentially. Each dataset prints `[done] Saved <N> transitions to <path>`. Datasets created (under `offline_data/`):

```
goalseek_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/
privileged_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/
crosscomp_s1_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/
crosscomp_s2_h4_efficiency_v2_re150_u10cross_fixdone_ep1000/
crosscomp_s0_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000/
crosscomp_s0_h4_efficiency_v2_re150tandem_u10cross_fixdone_ep1000/
```

Estimated time: ~5 h L4 total (1000 ep × 8 workers × ~30 min/ds with goalseek timing out more often).

- [ ] **Step 4: Run cell 4 (A2 mix5050)**

Expected: 2 sub-collections (`mix_tmp_goalseek_500ep_seed0`, `mix_tmp_crosscomp_500ep_seed1`), then a concatenated `mix5050_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000`. Estimated time: ~30 min L4.

- [ ] **Step 5: Run cell 5 (sanity cards)**

Expected: 7 `sanity_card.json` files written; each prints obs_dim matching its probe layout. Verify visually that:
- A1 / A2 / A3 / C1 / C3 have `obs_dim=10`, `obs_dim_matches_probe_layout=true`.
- B1 has `obs_dim=12`. B2 has `obs_dim=16`.
- All have `privileged_obs_present=true`.

- [ ] **Step 6: Sanity-check collector success rates**

Manually inspect each sanity card. Expected ranges (spec §3 / §8):
- A1 goalseek: `collector_success_rate < 0.3`.
- A2 mix5050: `collector_success_rate ≈ 0.5–0.7`.
- A3 privileged: `collector_success_rate ≈ 0.99`.
- B1 / B2 crosscomp: similar to anchor (~0.85–0.95).
- C1 upstream: ≥ 0.85 (upstream is geometrically easier).
- C3 tandem: 0.5–0.85 (multi-wake hurts collector).

Spec §9.1 (R1): if A1 `collector_success_rate < 0.1`, the spec authorises proceeding with the 1000-ep dataset as-is and flagging the abnormal sanity card in the report (no re-collection, no episode count expansion).

- [ ] **Step 7: Commit sanity cards to git**

Sanity cards are tiny (< 1 KB each) and have permanent reference value. From local terminal after Drive sync back:

```bash
git add offline_data/*/sanity_card.json
git commit -m "data(rebrac-broad): sanity cards for 7 broad-validation datasets"
```

---

## Task 9: Colab S2 — Phase 1 Training (~8 h L4)

Single Colab session, runs notebook cells 6–8.

- [ ] **Step 1: Run cell 6 (P1 training loop, 8 spokes)**

Order: A1 → A2 → A2-td3bc → A3 → B1 → B2 → C1 → C3. Each spoke trains 2 seeds (seed 42, seed 44) under its anchor `(β1, β2)` or `α`. Skip-resume kicks in for re-runs. Estimated time: ~7–8 h L4.

- [ ] **Step 2: Run cell 7 (summarize)**

Expected output: `[summary] P1 spokes summarized: 8` (7 ReBRAC + 1 TD3+BC). The trigger gate enumerates triggered spokes.

- [ ] **Step 3: Run cell 8 (review decisions)**

Inspect the printed list. Compare against spec §8:
- A1 (goalseek): expected mean ≈ 0.4–0.7, **triggers mean_shift** (almost certainly).
- A2 / A2-td3bc: A2 expected ≈ 0.80–0.92, A2 vs TD3BC gap is the trigger to watch.
- A3 (privileged): expected ≈ 0.93–0.97; may not trigger.
- B1 / B2: expected ≈ 0.92–0.96; may not trigger but watch positive shift.
- C1 (upstream): expected ≥ 0.85; may not trigger.
- C3 (tandem): expected 0.70–0.90; likely triggers mean_shift.

Save the printed list — it determines P2 work.

- [ ] **Step 4: Commit summary outputs to git**

Sync Drive → local, then:

```bash
git add results/offline/rebrac/broad_validation/summaries/
git commit -m "data(rebrac-broad): P1 summary + trigger decisions (16 run / 8 spokes)"
```

---

## Task 10: Colab S3 — Phase 2 Conditional Deepening (~5–8 h L4)

Only execute if cell 8 reported any triggered spokes. The decision tree per triggered spoke:

```
For each triggered spoke S:
  1. Run β refit cell with TRIGGERED_SPOKES = [S]:
     - Trains 2 single-seed (seed 42) cells: (β1=2, β2=2), (β1=4, β2=1).
  2. Inspect:
     - Read `results/offline/rebrac/broad_validation/<S>/<refit_pair>/test/seed_42.json`.
     - Compare new pair's success vs original (β1=4, β2=2) seed 42 success.
     - If max(new_pair_success) - original_success > 0.03 → winner drift.
  3. Run 5-seed expansion cell with WINNERS = {S: (winner_β1, winner_β2)}:
     - If winner unchanged: extend seeds 43, 45, 46 (seed 42 + seed 44 already present from P1) → 3 new runs.
     - If winner drifted: extend seeds 43, 44, 45, 46 (only seed 42 from refit) → 4 new runs.
```

- [ ] **Step 1: Edit cell 10 to set `TRIGGERED_SPOKES`**

Use the spoke IDs from cell 8 output. Skip this task if `TRIGGERED_SPOKES = []`.

- [ ] **Step 2: Run cell 10 (P2 β refit)**

Each triggered spoke runs 2 single-seed cells. Estimated time: 0.5 h × 2 × N triggered spokes. For N=2: ~2 h. For N=3: ~3 h.

- [ ] **Step 3: Determine winners per spoke**

For each spoke `S` in `TRIGGERED_SPOKES`, compute:

```python
# In a one-off Colab cell:
import json
from pathlib import Path

S = "A1"  # for example
broad_root = Path("results/offline/rebrac/broad_validation") / S
candidates = {
    "actorb_4p0__criticb_2p0": "anchor",  # already from P1
    "actorb_2p0__criticb_2p0": "refit_a",
    "actorb_4p0__criticb_1p0": "refit_b",
}
for pair, label in candidates.items():
    p = broad_root / pair / "test" / "seed_42.json"
    if p.exists():
        success = json.loads(p.read_text())["eval_success_rate"]
        print(f"  {label} ({pair}): success={success:.3f}")
```

Apply the +0.03 threshold (`WINNER_DRIFT_THRESHOLD` in `summarize_broad_validation.py`):
- If max(refit_a, refit_b) - anchor > 0.03 → winner = the higher refit pair.
- Else → winner stays at `(4.0, 2.0)`.

- [ ] **Step 4: Edit cell 11 to set `WINNERS`**

For each triggered spoke, set the winner pair tuple. Example:

```python
WINNERS = {
    "A1": ("4.0", "2.0"),  # winner unchanged
    "C3": ("2.0", "2.0"),  # winner drifted to (β1=2, β2=2)
}
```

- [ ] **Step 5: Run cell 11 (P2 5-seed expansion)**

For each spoke: trains seeds 43, 45, 46 if winner is `(4.0, 2.0)` (3 new runs); seeds 43, 44, 45, 46 otherwise (4 new runs). The script's `[skip]` logic handles seeds already present.

Estimated time: ~30 min × seeds × N. For N=2 spokes with 1 winner-drifted: ~3.5 h.

- [ ] **Step 6: Run cell 12 (final summary)**

Re-runs the summary script. The trigger gate's reasons remain in `trigger_decisions.json` from P1 — the new P2 results extend the per-pair tables.

- [ ] **Step 7: Run cell 13 (commit final summaries)**

```bash
git add results/offline/rebrac/broad_validation/summaries/
git commit -m "data(rebrac-broad): P2 β refit + 5-seed expansion on triggered spokes"
```

---

## Task 11: Write Broad-Validation Report

**Files:**
- Create: `docs/rebrac_broad_validation_report.md`

The report follows spec §10.2: §1 motivation, §2 matrix, §3 P1 results, §4 P2 results, §5 discussion.

- [ ] **Step 1: Author the report skeleton**

Create `docs/rebrac_broad_validation_report.md`:

```markdown
# ReBRAC Broad Validation — Report

> 文档版本：2026-XX-XX rev.1
> 配套 spec：[2026-05-04-rebrac-broad-validation-design.md](superpowers/specs/2026-05-04-rebrac-broad-validation-design.md)
> 配套 plan：[2026-05-04-rebrac-broad-validation-plan.md](superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md)

## 0. 摘要

[填一段总结：覆盖了哪三轴、跑了多少 run、几个 spoke 触发 P2、最终 paper 可写哪些 finding]

## 1. 动机与协议回顾

详见 spec §1。本报告是 spec 实施后的实证结论。

## 2. 数据集 + Sanity 卡

| spoke | dataset | collector_success_rate | episode_length_mean | obs_dim | n_transitions |
|---|---|---:|---:|---:|---:|
| Anchor | crosscomp_s0_…_ep1000 | [from card] | [from card] | 10 | [from card] |
| A1 | goalseek_s0_…_ep1000 | [fill] | [fill] | 10 | [fill] |
| A2 | mix5050_s0_…_ep1000 | [fill] | [fill] | 10 | [fill] |
| A3 | privileged_s0_…_ep1000 | [fill] | [fill] | 10 | [fill] |
| B1 | crosscomp_s1_…_ep1000 | [fill] | [fill] | 12 | [fill] |
| B2 | crosscomp_s2_…_ep1000 | [fill] | [fill] | 16 | [fill] |
| C1 | crosscomp_s0_…_upstream_ep1000 | [fill] | [fill] | 10 | [fill] |
| C3 | crosscomp_s0_…_tandem_ep1000 | [fill] | [fill] | 10 | [fill] |

## 3. Phase 1 结果（2-seed probe）

数据来源：`results/offline/rebrac/broad_validation/summaries/p1_overview.csv`。

### 3.1 三轴汇总

| spoke | 算法 | (β1, β2) / α | seeds | mean_success | std_success | Δ vs anchor (pp) |
|---|---|---|---|---:|---:|---:|
| Anchor | ReBRAC | (4.0, 2.0) | 5 | 0.902 | 0.021 | 0.0 |
| A1 | ReBRAC | (4.0, 2.0) | 2 | [fill] | [fill] | [fill] |
| A2 | ReBRAC | (4.0, 2.0) | 2 | [fill] | [fill] | [fill] |
| A2-td3bc | TD3+BC | α=0.25 | 2 | [fill] | [fill] | [fill] |
| A3 | ReBRAC | (4.0, 2.0) | 2 | [fill] | [fill] | [fill] |
| B1 | ReBRAC | (4.0, 2.0) | 2 | [fill] | [fill] | [fill] |
| B2 | ReBRAC | (4.0, 2.0) | 2 | [fill] | [fill] | [fill] |
| C1 | ReBRAC | (4.0, 2.0) | 2 | [fill] | [fill] | [fill] |
| C3 | ReBRAC | (4.0, 2.0) | 2 | [fill] | [fill] | [fill] |

### 3.2 触发判据落点

数据来源：`results/offline/rebrac/broad_validation/summaries/trigger_decisions.json`。

| spoke | triggered? | reasons |
|---|---|---|
| [fill all 7 ReBRAC spokes] | | |

## 4. Phase 2 结果（条件触发）

[如果触发数=0：本节简短声明 "广验未发现显著退化，paper 直接用 P1 表收口"。]

[如果触发数>0：]

### 4.1 β refit 结果

| spoke | original (4.0, 2.0) seed 42 | refit (2.0, 2.0) seed 42 | refit (4.0, 1.0) seed 42 | winner |
|---|---:|---:|---:|---|
| [fill] | | | | |

### 4.2 5-seed 扩展结果

| spoke | (β1, β2) | seeds | mean_success | std_success |
|---|---|---|---:|---:|

## 5. Discussion

### 5.1 A 轴：质量谱单调性

[根据数据填：单调还是非单调？哪一段最强 finding？]

### 5.2 B 轴：sensor 抬升 + sim2real narrative

[spec §8.2 的 paper claim 落地。如果 B1/B2 都接近 anchor，强调"deployment-realistic s0 上 ReBRAC 的价值仍在"。]

### 5.3 C 轴：geometry / wake topology generalization

[如果 C1/C3 都 ≥ 0.85，claim "ReBRAC 跨 task generalize"。否则在 §6 limitations 收口。]

### 5.4 P2 触发与 anchor robustness

[β refit 结果是否说明 anchor finalist (4.0, 2.0) 是 cross-condition robust？]

## 6. Limitations

[根据触发结果列：A1 下界、tandem 退化、TD3+BC head-to-head 仅 A2 一点等。]

## 7. Cross-references

- 主线 plan: [rebrac_experiment_plan.md](rebrac_experiment_plan.md) rev.8
- 主线 review: [rebrac_mainline_review.md](rebrac_mainline_review.md) rev.2 §3.5（新增链接）
- spec: [2026-05-04-rebrac-broad-validation-design.md](superpowers/specs/2026-05-04-rebrac-broad-validation-design.md)
- plan: [2026-05-04-rebrac-broad-validation-plan.md](superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md)
```

- [ ] **Step 2: Fill §2 (Sanity card table) from local sanity card files**

Read each `offline_data/<dataset>/sanity_card.json` and fill the table. Use a small Python script:

```python
import json
from pathlib import Path

DATASETS = [
    ("Anchor", "crosscomp_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000"),
    ("A1", "goalseek_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000"),
    ("A2", "mix5050_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000"),
    ("A3", "privileged_s0_h4_efficiency_v2_re150_u10cross_fixdone_ep1000"),
    ("B1", "crosscomp_s1_h4_efficiency_v2_re150_u10cross_fixdone_ep1000"),
    ("B2", "crosscomp_s2_h4_efficiency_v2_re150_u10cross_fixdone_ep1000"),
    ("C1", "crosscomp_s0_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000"),
    ("C3", "crosscomp_s0_h4_efficiency_v2_re150tandem_u10cross_fixdone_ep1000"),
]
for label, name in DATASETS:
    card = json.loads(Path(f"offline_data/{name}/sanity_card.json").read_text())
    print(f"| {label} | {name} | {card['collector_success_rate']:.3f} | "
          f"{card['episode_length_mean']:.1f} | {card['obs_dim']} | "
          f"{card['n_transitions']} |")
```

Paste output into the report.

- [ ] **Step 3: Fill §3 (P1 results) from `p1_overview.csv` + `trigger_decisions.json`**

Similar pattern: read both files, format as markdown tables.

- [ ] **Step 4: Fill §4 (P2 results) only if any spokes triggered**

Read per-spoke `test/seed_42.json` for each refit pair; compute the +0.03 winner-drift check; build the 5-seed table from `p1_overview.csv` (which now includes seeds 43/45/46 from P2).

- [ ] **Step 5: Write §5 discussion**

Anchor each subsection on the actual numbers. Pre-registered framings from spec §8:
- §5.2: B-axis gap narrowing supports sim2real (don't frame as ReBRAC weakening).
- §5.3: paper claim generality if both C spokes ≥ 0.85. **Note**: C1 has been retrofitted as a sensor-floor spoke after Task 11A's ablation chain — adapt the §5.3 narrative to the post-ablation framing (see Task 11A and spec §13).

- [ ] **Step 6: Write §0 abstract last**

Distil §3 (P1 results), §4 (P2 results if any), and §5 (discussion) into one short paragraph. Cite the headline numbers (e.g. "A1 collapses to 0.42 ± 0.03; B/C axes hold within 5pp of anchor").

- [ ] **Step 7: Commit the report**

```bash
git add docs/rebrac_broad_validation_report.md
git commit -m "docs(rebrac-broad): broad validation report — three-axis findings"
```

---

## Task 11A: C1 sensor-floor ablation closure (retrofit, 2026-05-06)

> **Status**: ablation chain completed; this task captures the closure work for the broad-validation report and points to spec §13 for full evidence.

**Files:**
- Reference: spec §13 in `docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`
- Notebooks (already committed):
  - `notebooks/rebrac_c1_reward_ablation_completed.ipynb`
  - `notebooks/rebrac_c1_asym_critic_ablation_completed.ipynb`
  - `notebooks/rebrac_c1_train_convergence_check_completed.ipynb`
  - `notebooks/rebrac_c1_epoch_sensitivity_ablation_completed.ipynb`
  - `notebooks/rebrac_c1_s1_sensor_upgrade_completed.ipynb` (Step 8)
- Reward preset: `auv_nav/reward.py::REWARD_OBJECTIVE_PRESETS["arrival_v2_simple"]`
- Tests: `tests/test_reward_objective.py::test_arrival_v2_simple_*`

### Background

C1 P1 5-seed result: success=0.225 ± 0.005, far below the spec §8.3 expectation of 0.85–0.95. Termination distribution was 77.5% timeout / 22.5% goal / 0% OOB — actor "deterministic-collapse". To rule out non-sensor root causes, three ablations were run; all failed to break the ~0.20 success ceiling. A fourth ablation (sensor upgrade s0 → s1) was added to test the sensor axis directly. **All four ablations failed**, confirming C1 (s0 / upstream u10 / crosscomp / Re150) as a **task-fundamental floor spoke** (deployment-impossible boundary, not just sensor floor).

### Ablation summary table

| Intervention | dataset / sensor / critic / budget | seeds | success | mean_R | Δ vs P1 (pp) | Verdict |
|---|---|---:|---:|---:|---:|---|
| P1 anchor | s0 / eff_v2 / sym / 64 ep | 5 | 0.225 ± 0.005 | −371.0 | 0.0 | baseline |
| Ablation A: reward swap | s0 / arr_v2_s / sym / 64 ep | 2 | 0.215 ± 0.015 | −98.2 | −1.0 | reward landscape ruled out |
| Ablation B: asym critic | s0 / arr_v2_s / asym / 64 ep | 2 | 0.195 ± 0.015 | −114.9 | −3.0 | privileged critic supervision ruled out |
| Ablation C: epoch 4× | s0 / arr_v2_s / sym / 256 ep | 1 | 0.220 (ep 256) | −96.2 | −0.5 | training budget ruled out |
| **Ablation D: sensor upgrade (C1-s1)** | **s1 / eff_v2 / sym / 64 ep** | **2** | **0.205 ± 0.005** | **−381.7** | **−2.0** | **deployable sensor upgrade ruled out** |

All five configurations land inside 0.195–0.225 — a 3-pp full range, within Ablation A/B's 3-pp noise radius. **Four independent root-cause hypotheses** (reward / critic supervision / budget / deployable sensor upgrade) all eliminated → **task-fundamental floor verdict** (stronger than the original sensor-floor verdict). Secondary finding: reward landscape decides actor's failure mode (P1 timeout-dominated 77.5/0 vs Ablation A/C/D's timeout/oob mixed ~53/26) but **does not** decide ceiling — success ceiling is decoupled from both reward and sensor dimensions.

### Steps (already done; recorded for traceability)

- [x] **Step 1: Add `arrival_v2_simple` reward preset + regression tests**

Commit: `feat(auv-nav): arrival_v2_simple reward preset + regression tests`. Field-lockdown test asserts dict equality; terminal-dominance test asserts `fast_success(170) > slow_success(114) > timeout_near(−138) > timeout_far(−144) > slow_OOB(−294)`.

- [x] **Step 2: Build + run `rebrac_c1_reward_ablation.ipynb` in Colab**

Output: `results/offline/rebrac/c1_reward_ablation/<dataset>/actorb_4p0__criticb_2p0/test/seed_{42,44}.json`. Verdict: β-bound (Δ=−1pp).

- [x] **Step 3: Build + run `rebrac_c1_asym_critic_ablation.ipynb`**

Output: `results/offline/rebrac/c1_asym_critic_ablation/<dataset>/actorb_4p0__criticb_2p0/test/seed_{42,44}.json`. Flags: `--use-asymmetric-critic --privileged-actor-update-mode zeros`. Verdict: Δ=−2pp; CLAUDE.md §3 doesn't carry over to offline ReBRAC on s0/upstream.

- [x] **Step 4: Run convergence diagnostic (no-GPU, ~30s)**

Notebook reads ablation A's `train_log.jsonl` and compares mid-window (epoch 16–32) vs late-window (56–64) for 6 metrics. 6/6 metrics still moving; signaled "epochs may be insufficient" → triggered Step 5.

- [x] **Step 5: Build + run `rebrac_c1_epoch_sensitivity_ablation.ipynb` (256 epoch × 1 seed)**

Periodic val eval (every 16 ep on `val_40` manifest) + periodic ckpt + post-train batch test eval at epoch {64, 128, 192, 256} on `test_100` manifest. Verdict: success {0.20, 0.20, 0.22, 0.22} → epoch 192/256 numerically identical → actor deterministic-locked → epochs ruled out. Coordination with Step 4: critic still fitting (mean_q +44.8%), but BC anchor pins actor → β floor textbook signature.

- [x] **Step 6: Write spec §13 (C1 sensor-floor ablation retrofit)**

Done in `docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md`. Includes evidence tables, three-ablation summary, sensor-floor conclusion, and follow-up plan for C1-s1.

- [ ] **Step 7: Reflect in broad-validation report § (deferred to Task 11)**

When Task 11 is executed, the report's §3 (P1 results), §5.3 (C-axis discussion), and §6 (limitations) must reflect the post-ablation framing (now task-fundamental floor, not just sensor floor):
- §3.1 row C1: fill `mean_success=0.225, std=0.005, Δ=−68pp vs anchor` (5-seed).
- §3.2 row C1: triggered (mean shift) → P2 deepening → 4-ablation chain (not β refit).
- §5.3: paper claim is no longer "C1 generality" or "C1-s1 sensor-axis controllable lever". Replace with **"C1 demonstrates a deployment-impossible boundary: four independent interventions (reward / asym critic / 4× epochs / sensor upgrade s0→s1) all fail to break the 0.19–0.23 ceiling; reward determines failure mode (timeout vs oob) but not ceiling; upstream u10 + crosscomp dataset is a task-fundamental floor."**
- §6 Limitations: explicit statement "upstream u10 + crosscomp dataset is deploy-impossible on s0/s1; sensor upgrade alone is not a sufficient lever." Mention C1-s2 (16-D) and target_speed=2.0 as backlog items.

- [x] **Step 8: Plan + run C1-s1 sensor-upgrade follow-up (per spec §13.6) — closed 2026-05-07**

Notebook: [`notebooks/rebrac_c1_s1_sensor_upgrade_completed.ipynb`](../../../notebooks/rebrac_c1_s1_sensor_upgrade_completed.ipynb).

**Empirical result (n=2 seeds × 64 ep × s1, sym critic):**

| seed | success | mean_R | termination (goal / timeout / oob) |
|---:|---:|---:|---|
| 42 | 0.210 | −379.98 | 21 / 53 / 26 |
| 44 | 0.200 | −383.42 | 20 / 54 / 26 |
| **mean** | **0.205 ± 0.005** | **−381.70 ± 1.72** | **20.5 / 53.5 / 26.0** |

Outputs:
- `offline_data/crosscomp_s1_h4_efficiency_v2_re150_u10upstream_fixdone_ep1000/transitions.npz` (collector_success_rate=1.0, n_transitions=268,329)
- `checkpoints/offline/rebrac/c1_s1_sensor_upgrade/<dataset>/actorb_4p0__criticb_2p0/seed_{42,44}/`
- `results/offline/rebrac/c1_s1_sensor_upgrade/<dataset>/actorb_4p0__criticb_2p0/test/seed_{42,44}.json`

**Verdict: < 0.30 → task-fundamental floor.** Δ vs P1 anchor = **−2.0 pp** (within ±1.5pp noise radius). Sensor upgrade ruled out as deployability lever.

Downstream actions triggered:
1. spec §13.4 conclusion upgraded from "sensor floor" to "task-fundamental floor" (4 ablations all fail).
2. spec §13.6 paper-claim branch fixed at "deployment-impossible boundary demonstration".
3. report §10A.3/§10A.4 updated with 5-row ablation table + § §10A.5 limitation "C1-s1 未跑" removed and replaced with "C1-s2 / target_speed=2.0 backlog".
4. C1-s2 (s2, 4 probes, 16-D) **NOT triggered**: C1-s1 already shows sensor upgrade is not the lever; s2 expected to land in same band; backlogged.
5. Optional follow-up `target_speed=2.0` (downstream geometry) backlogged as Step 9 below.

- [ ] **Step 9 (backlog, not blocking broad-validation report): target_speed=2.0 follow-up**

If broad-validation main pipeline closes and time permits, test whether raising `target_speed` from 1.5 to 2.0 m/s (downstream-favored geometry) flips C1 from deploy-impossible to deploy-grade. Single-seed P1 probe sufficient; ~1h L4. If success ≥ 0.5 → fine-grain ablation; if < 0.3 → confirm "u10 upstream is task-fundamental even at higher target speeds" and close.

---

## Task 12: Cross-link in Mainline Review

**Files:**
- Modify: `docs/rebrac_mainline_review.md`

Spec §10.2 / §11 require adding a §3.5 cross-link.

- [ ] **Step 1: Append a new §3.5 to mainline review**

Locate the §3 section in `docs/rebrac_mainline_review.md` (likely the "Open issues / future work" section). Add at the end of the §3 block:

```markdown
### 3.5 Broad validation across three axes (closed 2026-XX-XX)

主线 §3.2.G 留下的 cross-task / sensor / data-quality generality 三个边界，由 broad validation 阶段以 Probe-then-Deepen 协议在有限算力（19–22h L4）下回应。

- 设计 spec：[2026-05-04-rebrac-broad-validation-design.md](superpowers/specs/2026-05-04-rebrac-broad-validation-design.md)
- 实施 plan：[2026-05-04-rebrac-broad-validation-plan.md](superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md)
- 实证报告：[rebrac_broad_validation_report.md](rebrac_broad_validation_report.md)

总览：8 个 spoke（A1/A2/A2-td3bc/A3/B1/B2/C1/C3），16 P1 run + 条件性 P2 run。Paper §experiments 新增 broad-validation subsection 引用本报告 §3 / §4 主表与 §5 discussion。
```

Adjust the date once Task 11 is done.

- [ ] **Step 2: Verify the link renders**

```bash
grep -n "broad_validation" docs/rebrac_mainline_review.md
```

Expected: at least 3 lines (link to spec, plan, and report).

- [ ] **Step 3: Commit**

```bash
git add docs/rebrac_mainline_review.md
git commit -m "docs(rebrac): mainline review §3.5 cross-link to broad validation"
```

---

## Acceptance Criteria

Per spec §12, broad-validation closure requires:

1. ✅ 7 new datasets collected, each with `sanity_card.json` (Task 8 step 5–7).
2. ✅ 16 P1 runs complete, `summaries/p1_overview.csv` generated (Task 9).
3. ✅ All triggered spokes complete β refit + 5-seed expansion, OR `len(triggered) == 0` documented (Task 10).
4. ✅ `docs/rebrac_broad_validation_report.md` complete §1–§5 (Task 11).
5. ✅ C1 task-fundamental floor ablation chain (4 ablations: reward / asym critic / 4× epochs / sensor upgrade s0→s1) closed and reflected in spec §13 + report §10A (Task 11A Step 8 closed 2026-05-07; verdict = task-fundamental floor, < 0.30); C1-s2 and target_speed=2.0 backlogged as non-blocking.
6. ✅ `docs/rebrac_mainline_review.md` §3.5 cross-link added (Task 12).
7. ✅ `paper/sections/experiments.tex` broad-validation subsection drafted (out of scope for this plan; follows from Task 11 + 11A + 12).

After all 7 conditions are met → broad validation phase closed; paper drafting can ingest new findings.

## Operational Risks (cross-reference spec §9)

The plan does not duplicate spec §9's risk table — it is the authoritative reference at run-time. Quick pointers:

- **R1** (A1 collector success < 0.10): proceed with the 1000-ep dataset, mark sanity card abnormal, report it as a finding (Task 8 step 6).
- **R2** (P2 triggers > 4 spokes): handle in priority order A > C > B; defer overflow to a later round (Task 10 — execute in spec priority order if budget pressure arises).
- **R3** (P2 triggers = 0): close §4 of the report with one short paragraph noting "no significant degradation found" and emphasising the §3.1 generality claim (Task 11 step 4 already conditional).
- **R4** (Drive capacity > 50 GB): after each spoke's training completes, optionally remove `agent_step_*.pt` checkpoints other than the selected one (preserve `agent_final.pt` + `selected_checkpoint.json`). Out of scope as a plan task; user-discretion in S3.
- **R5** (β refit drifts in ≥ 2 spokes): document in report §5.4 / §6 limitations; do NOT modify the anchor finalist claim — the broad validation is exploring robustness, not redefining the main result.

## Implementation Notes

### Spec deviations (none)

This plan implements the spec literally. The only marginal call: `--policy-mixture` would be a simpler (but spec-deviating) implementation for A2; we kept the spec's concat path because the spec explicitly mandates `mix_components` / `mix_strategy` / `task_sampler` metadata fields that `--policy-mixture` doesn't produce.

### Skip-resume invariants

Every dataset / training cell relies on file-existence checks:
- `offline_data/<name>/transitions.npz` → skip collect.
- `<run_dir>/trainer_state.json` AND `<run_dir>/agent_final.pt` → skip train.
- `<val_dir>/<agent_tag>.json` → skip per-checkpoint validate.
- `<selection_dir>/selected_checkpoint.json` → skip select.
- `<test_dir>/seed_<S>.json` → skip test.

This means re-running a cell after a Colab session restart is always safe and zero-cost on completed work.

### Local Python environment

All Python invocations use `python` from the user's `mytorch1` conda env locally (per `feedback_machine_config`). Colab uses its system Python with the project's `pip install -r requirements.txt` already done.
