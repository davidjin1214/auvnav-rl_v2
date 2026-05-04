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
