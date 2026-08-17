"""Audit whether offline datasets were collected on the same task instances used for evaluation.

Collection draws episode seeds as ``base_seed + ep`` (``collect_offline_data.py``), while a
benchmark manifest freezes ``manifest_seed + idx`` (``generate_standard_benchmarks.py``). Both
paths call the same ``env.reset(seed=...)``, so a seed shared between the two ranges means the
evaluation episode is literally part of the training data.

Three passes:

* reconciliation pass (always, before the range pass) -- check the enumeration itself against a
  ledger of dataset names, probing each name with a direct ``stat`` that never lists a directory.
  See ``reconcile()`` for why an audit that trusts its own ``glob`` cannot report its own blind
  spot.
* range pass (default) -- intersect every dataset's seed range with every manifest's seed set,
  but only where the flow field, task geometry and target speed match, since a different task
  configuration produces a different instance regardless of seed.
* identity pass (``--verify DATASET MANIFEST``) -- replay the reset RNG for the intersecting
  seeds and compare the resulting instance against the frozen manifest entry. This costs one
  env construction (the flow field is memory-mapped) and no simulation.

The transition arrays do not store ``flow_time`` / ``start_xy``, so replaying reset is the only
way to compare instances.

Usage::

    python -m scripts.audit_seed_overlap
    python -m scripts.audit_seed_overlap --record
    python -m scripts.audit_seed_overlap --verify crosscomp_..._ep2000 single_u10_cross_tgt15_ep100

Exit status is 2 when the reconciliation pass finds a dataset the enumeration should have
returned and did not; the seed findings themselves do not affect it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .train_utils import make_planar_env

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_DATA_DIR = REPO_ROOT / "offline_data"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
LEDGER_PATH = REPO_ROOT / "scripts" / "offline_dataset_ledger.txt"
TOL = 1e-6


def _load_datasets() -> dict[str, dict[str, Any]]:
    # Recursive: collections such as fql_succession/ nest their datasets one level deeper, and a
    # single-level glob skipped them silently -- an audit that under-reports is worse than none.
    datasets: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(OFFLINE_DATA_DIR.rglob("metadata.json")):
        name = str(meta_path.parent.relative_to(OFFLINE_DATA_DIR)).replace("\\", "/")
        datasets[name] = json.loads(meta_path.read_text(encoding="utf-8"))
    return datasets


def _load_manifests() -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(BENCHMARKS_DIR.rglob("*.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "episodes" not in payload:
            continue
        manifests[str(manifest_path.relative_to(BENCHMARKS_DIR)).replace("\\", "/")] = payload
    return manifests


def _metadata_is_file(name: str) -> bool:
    """Direct stat of one full path -- a lookup, never a directory listing.

    This distinction is the whole point of the ledger. On the Colab/Drive FUSE mount a readdir
    of ``offline_data/`` has come back short in two separate processes while a lookup of the very
    same path succeeded in a third, so a name that stats fine but never appears in the
    enumeration is evidence about the listing, not about the file.
    """
    return (OFFLINE_DATA_DIR / name / "metadata.json").is_file()


def _read_ledger(path: Path) -> list[str]:
    if not path.is_file():
        return []
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.add(name)
    return sorted(names)


def _scandir_top_level() -> set[str]:
    """Top-level dataset names as ``os.scandir`` sees them, without descending anywhere."""
    if not OFFLINE_DATA_DIR.is_dir():
        return set()
    with os.scandir(OFFLINE_DATA_DIR) as entries:
        return {e.name for e in entries if _metadata_is_file(e.name)}


@dataclass
class Reconciliation:
    """What the enumeration owes an explanation for."""

    missed: list[str] = field(default_factory=list)
    shadowed: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    unrecorded: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.missed or self.shadowed)


def reconcile(enumerated: Iterable[str], ledger_path: Path = LEDGER_PATH) -> Reconciliation:
    """Cross-check one enumeration against sources that do not depend on it.

    Two independent checks, because they fail for different reasons:

    * ``missed`` -- a ledger name whose ``metadata.json`` stats fine, yet the enumeration never
      returned it. The ledger is a cross-host union of every dataset name ever recorded, so it
      supplies names to probe that this host's listing cannot be trusted to produce.
    * ``shadowed`` -- a top-level dataset ``os.scandir`` reports and the recursive glob dropped.
      This catches an asymmetry inside pathlib rather than a bad listing: ``rglob`` refuses to
      descend into a symlinked directory, and this repo mounts its data through links.

    ``os.listdir`` is deliberately *not* used as the reference for ``missed``. It and ``glob``
    are two consumers of the same readdir, so a short listing hands both the same short answer
    and the comparison comes back clean -- which is exactly how a check like that certifies a
    blind spot instead of reporting it.
    """
    seen = set(enumerated)
    ledger = set(_read_ledger(ledger_path))
    rec = Reconciliation()
    for name in sorted(ledger - seen):
        (rec.missed if _metadata_is_file(name) else rec.absent).append(name)
    rec.shadowed = sorted(_scandir_top_level() - seen)
    rec.unrecorded = sorted(seen - ledger)
    return rec


def print_reconciliation(rec: Reconciliation, ledger_path: Path = LEDGER_PATH) -> None:
    n_ledger = len(_read_ledger(ledger_path))
    print(f"=== enumeration reconciliation (ledger: {n_ledger} names, {ledger_path.name}) ===")
    if n_ledger == 0:
        print("  ledger empty -- run with --record on a host known to list completely")
    for name in rec.missed:
        print(f"  [ENUM MISS] {name}  <- metadata.json stats fine; the enumeration dropped it")
    for name in rec.shadowed:
        print(f"  [SHADOWED ] {name}  <- os.scandir sees it, the recursive glob did not")
    for name in rec.absent:
        print(f"  [absent   ] {name}  (recorded elsewhere, not on this host)")
    for name in rec.unrecorded:
        print(f"  [new      ] {name}  (not in the ledger yet -- rerun with --record)")
    if rec.failed:
        print("  VERDICT: the enumeration is incomplete. Any 'only N datasets are affected'")
        print("           claim drawn from it is unsupported until this is resolved.")
    else:
        print("  VERDICT: no ledger name is missing from the enumeration.")
    print()


def record_ledger(enumerated: Iterable[str], ledger_path: Path = LEDGER_PATH) -> list[str]:
    """Union the ledger with names seen here. Never removes: a short listing must not shrink it."""
    known = set(_read_ledger(ledger_path))
    added = sorted({name for name in enumerated if _metadata_is_file(name)} - known)
    if not added:
        return []
    header = _ledger_header(ledger_path)
    body = "\n".join(sorted(known | set(added)))
    ledger_path.write_text(f"{header}{body}\n", encoding="utf-8")
    return added


def _ledger_header(ledger_path: Path) -> str:
    """Keep the existing comment block -- it carries where the recorded names came from."""
    if ledger_path.is_file():
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        kept = list(itertools.takewhile(lambda line: line.startswith("#"), lines))
        if kept:
            return "\n".join(kept) + "\n"
    return (
        "# Dataset names ever seen under offline_data/, across hosts (local, Drive).\n"
        "# Union only -- `--record` adds, never removes, so a short listing cannot shrink it.\n"
    )


def _task_key_dataset(meta: dict[str, Any]) -> tuple[str, str, float]:
    return (
        str(meta.get("flow_path")),
        str(meta.get("task_geometry")),
        float(meta.get("target_speed") or 0.0),
    )


def _task_key_manifest(manifest: dict[str, Any]) -> tuple[str, str, float]:
    options = manifest.get("base_reset_options") or {}
    return (
        str(manifest.get("flow_path")),
        str(options.get("task_geometry")),
        float(options.get("target_auv_max_speed_mps") or 0.0),
    )


def _instance_from_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "flow_time": float(info["flow_time_s"]),
        "start_xy": np.asarray(info["start_xy_m"], dtype=float),
        "goal_xy": np.asarray(info["goal_xy_m"], dtype=float),
        "initial_heading": float(info["psi_rad"]),
    }


def _instance_from_manifest(episode: dict[str, Any]) -> dict[str, Any]:
    options = episode["reset_options"]
    return {
        "flow_time": float(options["flow_time"]),
        "start_xy": np.asarray(options["start_xy"], dtype=float),
        "goal_xy": np.asarray(options["goal_xy"], dtype=float),
        "initial_heading": float(options["initial_heading"]),
    }


def _same_instance(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        abs(a["flow_time"] - b["flow_time"]) < TOL
        and np.allclose(a["start_xy"], b["start_xy"], atol=TOL)
        and np.allclose(a["goal_xy"], b["goal_xy"], atol=TOL)
        and abs(a["initial_heading"] - b["initial_heading"]) < TOL
    )


def run_range_pass(datasets: dict[str, dict[str, Any]] | None = None) -> int:
    datasets = _load_datasets() if datasets is None else datasets
    manifests = _load_manifests()
    if not datasets:
        print(f"no datasets under {OFFLINE_DATA_DIR} (gitignored; nothing to audit)")
        return 0

    print(f"datasets: {len(datasets)}   manifests: {len(manifests)}\n")
    hits = 0
    for name, meta in datasets.items():
        base_seed = int(meta.get("seed") or 0)
        n_episodes = int(meta.get("num_episodes") or 0)
        train_seeds = range(base_seed, base_seed + n_episodes)
        flagged: list[str] = []
        for manifest_name, manifest in manifests.items():
            if _task_key_dataset(meta) != _task_key_manifest(manifest):
                continue
            eval_seeds = {int(ep["seed"]) for ep in manifest["episodes"]}
            shared = sorted(eval_seeds & set(train_seeds))
            if shared:
                flagged.append(
                    f"    OVERLAP {len(shared)}/{len(eval_seeds)} with {manifest_name} "
                    f"(seeds {shared[0]}..{shared[-1]})"
                )
        status = "OVERLAP" if flagged else "clean"
        print(f"[{status:7s}] {name}  seeds {base_seed}..{base_seed + n_episodes - 1}")
        for line in flagged:
            print(line)
            hits += 1
    print("\nOverlap means the evaluation episode is a training episode: same seed, same reset.")
    return hits


def run_identity_pass(dataset_name: str, manifest_name: str) -> int:
    meta_path = OFFLINE_DATA_DIR / dataset_name / "metadata.json"
    manifest_path = BENCHMARKS_DIR / manifest_name
    if not manifest_path.exists():
        manifest_path = BENCHMARKS_DIR / f"{manifest_name}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    base_seed = int(meta.get("seed") or 0)
    n_episodes = int(meta.get("num_episodes") or 0)
    train_seeds = set(range(base_seed, base_seed + n_episodes))

    # Rebuild the env the way the collector did; reset options come from the same helper inputs.
    env = make_planar_env(
        meta["flow_path"],
        history_length=int(meta.get("history_length") or 1),
        probe_layout=str(meta.get("probe_layout") or "s0"),
    )
    reset_options = {
        "task_geometry": meta["task_geometry"],
        "target_auv_max_speed_mps": meta["target_speed"],
    }

    checked = 0
    identical = 0
    first_mismatch: tuple[int, dict[str, Any], dict[str, Any]] | None = None
    for episode in manifest["episodes"]:
        seed = int(episode["seed"])
        if seed not in train_seeds:
            continue
        checked += 1
        _, info = env.reset(seed=seed, options=dict(reset_options))
        replayed = _instance_from_info(info)
        frozen = _instance_from_manifest(episode)
        if _same_instance(replayed, frozen):
            identical += 1
        elif first_mismatch is None:
            first_mismatch = (seed, frozen, replayed)
    env.close()

    print(f"dataset : {dataset_name}  seeds {base_seed}..{base_seed + n_episodes - 1}")
    print(f"manifest: {manifest_path.name}  {len(manifest['episodes'])} episodes")
    print(f"seeds in both ranges     : {checked}/{len(manifest['episodes'])}")
    print(f"identical task instances : {identical}/{checked}")
    if first_mismatch is not None:
        seed, frozen, replayed = first_mismatch
        print(f"first mismatch @ seed {seed}")
        print(f"  manifest: {frozen}")
        print(f"  replayed: {replayed}")
    return identical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify",
        nargs=2,
        metavar=("DATASET", "MANIFEST"),
        default=None,
        help="Replay reset for one dataset/manifest pair and compare frozen task instances.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help="Dataset-name ledger used to probe the enumeration (default: %(default)s).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Add the names seen here to the ledger. Union only; never removes.",
    )
    args = parser.parse_args()

    if args.verify is not None:
        run_identity_pass(*args.verify)
        return

    datasets = _load_datasets()
    rec = reconcile(datasets, args.ledger)
    print_reconciliation(rec, args.ledger)
    if args.record:
        added = record_ledger(datasets, args.ledger)
        print(f"ledger: added {len(added)} name(s){':' if added else ''}")
        for name in added:
            print(f"  + {name}")
        print()
    run_range_pass(datasets)
    if rec.failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
