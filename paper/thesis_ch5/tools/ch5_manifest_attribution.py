"""
paper/thesis_ch5/tools/ch5_manifest_attribution.py

Which evaluation-instance set did each published offline readout run on, and which
physical manifest file under ``benchmarks/`` was that?

Why this exists
---------------
The 2026-08-21 contamination enumeration closed over the 24 manifests in
``benchmarks/`` (``python -m scripts.audit_seed_overlap``). That closure only
transfers to the chapter's tables if every reported readout actually evaluated on
one of those 24 episode sets -- otherwise the audit swept files nobody used. No
readout records the manifest *path* it was given, so the link has to be rebuilt
from two independent directions:

evidence side
    Every ``eval_episode_results`` entry carries ``episode_id`` and ``seed``, and
    ``generate_standard_benchmarks.py`` builds those as
    ``f"{spec.key}_ep_{idx:04d}"`` / ``spec.manifest_seed + idx``. So the ordered
    (episode_id, seed) sequence is a fingerprint of the episode set, recoverable
    from the readout alone. This side is complete but cannot name one file: several
    manifests freeze identical sequences.

provenance side
    The launchers derive the path as
    ``${MANIFEST_ROOT}/{val,test}_${N}/${BENCHMARK_KEY}.json``, so the launcher's
    defaults plus the driving notebook's ``os.environ`` overrides pin the physical
    file exactly. This side names one file but only covers studies whose notebook
    is in the repo.

What the fingerprint cannot see, and why that is not a gap: a readout records only
``episode_id``, ``seed`` and per-episode geometry. Measured across the manifests
that share a seed range, the geometry is identical to the last bit, so it adds no
discrimination -- the equivalence classes below are a property of the manifests,
not a weakness of the method.

A ``prefix`` match means the readout's episodes are the opening run of a longer
manifest (a 40-episode evaluation against the head of a 100-episode file). The
contamination conclusion still transfers: a subset of an audited set is audited.

Exit code is 1 when some readout's episode set matches no manifest in
``benchmarks/`` -- that, not the size of an equivalence class, is the alarm.

Usage:
    python paper/thesis_ch5/tools/ch5_manifest_attribution.py
    python paper/thesis_ch5/tools/ch5_manifest_attribution.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BENCHMARKS = REPO / "benchmarks"
RESULTS = REPO / "results" / "offline"
NOTEBOOKS = REPO / "notebooks"
SCRIPTS = REPO / "scripts"

# The published-unit list is NOT duplicated here: it lives in ch5_holdout_split_audit
# and is imported, so the two tools cannot drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch5_holdout_split_audit import UNITS as PUBLISHED_UNITS  # noqa: E402

Fingerprint = tuple[tuple[str, int], ...]


# --------------------------------------------------------------- manifest side
def load_manifests() -> dict[str, Fingerprint]:
    out: dict[str, Fingerprint] = {}
    for path in sorted(BENCHMARKS.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes = payload.get("episodes")
        if not episodes:
            continue
        out[path.relative_to(BENCHMARKS).as_posix()] = tuple(
            (str(e["episode_id"]), int(e["seed"])) for e in episodes
        )
    return out


def equivalence_classes(manifests: dict[str, Fingerprint]) -> dict[Fingerprint, list[str]]:
    classes: dict[Fingerprint, list[str]] = defaultdict(list)
    for rel, fingerprint in manifests.items():
        classes[fingerprint].append(rel)
    return {k: sorted(v) for k, v in classes.items()}


# ---------------------------------------------------------------- readout side
def load_readouts() -> list[dict]:
    rows: list[dict] = []
    for path in RESULTS.rglob("*.json"):
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"eval_episode_results" not in raw:
            continue
        try:
            payload = json.loads(raw.decode("utf-8"))
        except ValueError:
            continue
        episodes = payload.get("eval_episode_results")
        if not episodes:
            # e.g. selected_checkpoint.json, which names the key inside a nested record
            continue
        rows.append(
            {
                "path": path.relative_to(REPO).as_posix(),
                "dir": path.parent.relative_to(REPO).as_posix(),
                "fp": tuple((str(e.get("episode_id")), int(e["seed"])) for e in episodes),
            }
        )
    return rows


def attribute(rows: list[dict], manifests: dict[str, Fingerprint]) -> None:
    exact = equivalence_classes(manifests)
    prefixes: dict[Fingerprint, set[str]] = defaultdict(set)
    for rel, fingerprint in manifests.items():
        for k in range(1, len(fingerprint)):
            prefixes[fingerprint[:k]].add(rel)
    for row in rows:
        fingerprint = row["fp"]
        if fingerprint in exact:
            row["match"], row["candidates"] = "exact", exact[fingerprint]
        elif fingerprint in prefixes:
            row["match"], row["candidates"] = "prefix", sorted(prefixes[fingerprint])
        else:
            row["match"], row["candidates"] = "none", []


# ------------------------------------------------------------- provenance side
ENV_ASSIGN = re.compile(
    r"""(?:os\.environ\[\s*['"]|%env\s+|^\s*export\s+)([A-Z0-9_]+)['"]?\s*\]?\s*=\s*['"]?([^'"\n]*)""",
    re.M,
)
LAUNCHER_CALL = re.compile(r"scripts/(run_offline_[a-z0-9_]+)\.sh")
SHELL_DEFAULT = re.compile(r'^([A-Z0-9_]+)="\$\{\1:-((?:[^"}]|\$\{[^}]*\})*)\}"', re.M)


def launcher_defaults(name: str) -> dict[str, str]:
    path = SCRIPTS / f"{name}.sh"
    if not path.is_file():
        return {}
    return dict(SHELL_DEFAULT.findall(path.read_text(encoding="utf-8", errors="replace")))


def launcher_manifest_roots() -> list[tuple[str, str, str, bool]]:
    """(launcher, variable, declared root, root exists) for every offline launcher.

    Mechanical and complete: it reports the roots a launcher *declares*, without
    trying to interpret which of them a given phase forwards to the worker. That
    interpretation is where a regex would start guessing.
    """
    out = []
    for path in sorted(SCRIPTS.glob("run_offline_*.sh")):
        for name, value in SHELL_DEFAULT.findall(
            path.read_text(encoding="utf-8", errors="replace")
        ):
            if not name.endswith("MANIFEST_ROOT"):
                continue
            out.append((path.name, name, value, (REPO / value).is_dir()))
    return out


def declared_paths() -> list[tuple[str, str, str, list[tuple[str, bool]]]]:
    """(notebook, results root, launcher, [(declared manifest path, exists)]).

    A notebook is replayed cell by cell so that each launcher call sees the
    environment as it stood at that point: several notebooks drive more than one
    study by reassigning RESULTS_ROOT between calls, and taking a whole-notebook
    union would attribute the last study's overrides to the first.
    """
    out = []
    for notebook in sorted(NOTEBOOKS.glob("*_completed.ipynb")):
        try:
            doc = json.loads(notebook.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            continue
        env: dict[str, str] = {}
        seen: set[tuple] = set()
        for cell in doc.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source")
            source = source if isinstance(source, str) else "".join(source)
            for name, value in ENV_ASSIGN.findall(source):
                env[name] = value.strip()
            for launcher in LAUNCHER_CALL.findall(source):
                defaults = launcher_defaults(launcher)
                results_root = env.get("RESULTS_ROOT") or defaults.get("RESULTS_ROOT", "")
                if not results_root.startswith("results/offline"):
                    continue
                if not defaults:
                    # e.g. run_offline_rebrac_critic_ln_off.sh, which the notebook
                    # generates at runtime by patching one flag into the screen
                    # launcher -- it is not, and never was, a repo file.
                    record = (notebook.name, results_root, launcher, [])
                    if record[:3] not in seen:
                        seen.add(record[:3])
                        out.append(record)
                    continue
                root = env.get("MANIFEST_ROOT") or defaults.get("MANIFEST_ROOT", "?")
                key = env.get("BENCHMARK_KEY") or defaults.get("BENCHMARK_KEY", "?")
                declared = []
                for split, var in (
                    ("val", "VAL_MANIFEST_EPISODES"),
                    ("test", "TEST_MANIFEST_EPISODES"),
                ):
                    count = env.get(var) or defaults.get(var, "?")
                    rel = f"{root}/{split}_{count}/{key}.json"
                    if "$" in rel or "?" in rel:
                        # The broad launcher takes BENCHMARK_KEY from its per-spoke
                        # table, so the filename is not fixed at this level. Report
                        # what the split directory actually holds rather than
                        # inventing one name for it.
                        split_dir = REPO / f"{root}/{split}_{count}"
                        found = sorted(p.name for p in split_dir.glob("*.json"))
                        rel = f"{root}/{split}_{count}/" + (
                            "{" + ", ".join(found) + "}" if found else "<empty>"
                        )
                        declared.append((rel, bool(found)))
                    else:
                        declared.append((rel, (REPO / rel).is_file()))
                key_tuple = (notebook.name, results_root, launcher)
                if key_tuple in seen:
                    continue
                seen.add(key_tuple)
                out.append((notebook.name, results_root, launcher, declared))
    return out


# --------------------------------------------------------------------- report
def brief(candidates: list[str]) -> str:
    if len(candidates) == 1:
        return candidates[0]
    heads = sorted({c.split("/")[0] for c in candidates})
    return "{" + ", ".join(heads) + "}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument(
        "--all",
        action="store_true",
        help="list every readout directory, not only the published units",
    )
    args = parser.parse_args()

    manifests = load_manifests()
    classes = equivalence_classes(manifests)
    rows = load_readouts()
    attribute(rows, manifests)

    print(f"manifests in benchmarks/: {len(manifests)}   readouts scanned: {len(rows)}")

    print("\n=== manifest equivalence classes (ordered episode_id + seed) ===")
    print("Members of one class are indistinguishable from any readout: same episodes,")
    print("same order, and -- measured -- identical geometry down to float noise.")
    for members in sorted(classes.values()):
        if len(members) == 1:
            continue
        print(f"  class of {len(members)}:")
        for member in members:
            print(f"      {member}")

    by_dir: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dir[row["dir"]].append(row)

    published = {rel: name for name, rel in PUBLISHED_UNITS.items()}
    shown = by_dir if args.all else {k: v for k, v in by_dir.items() if k in published}

    print("\n=== readout directory -> evaluation-instance set ===")
    header = f"{'unit':46s} {'n':>4s} {'episodes':>4s} {'seeds':>14s} {'match':>6s}  manifest(s)"
    print(header)
    print("-" * len(header))
    for unit_dir in sorted(shown):
        group = shown[unit_dir]
        label = published.get(unit_dir, unit_dir)
        # a directory can hold more than one episode set (e.g. baselines/ holds val + test)
        variants: dict[Fingerprint, list[dict]] = defaultdict(list)
        for row in group:
            variants[row["fp"]].append(row)
        for fingerprint, members in sorted(variants.items(), key=lambda kv: -len(kv[1])):
            seeds = [s for _, s in fingerprint]
            row = members[0]
            print(
                f"{label:46.46s} {len(members):4d} {len(fingerprint):8d} "
                f"{min(seeds):6d}..{max(seeds):<6d} {row['match']:>6s}  {brief(row['candidates'])}"
            )
            label = ""

    print("\n=== manifest roots the launchers declare ===")
    last = ""
    for launcher, var, root, exists in launcher_manifest_roots():
        if launcher != last:
            print(f"  {launcher}")
            last = launcher
        print(f"      [{'present' if exists else ' GONE  '}] {var:34s} {root}")

    print("\n=== physical manifest path each study declared (launcher + notebook) ===")
    print("Derived, not transcribed: ${MANIFEST_ROOT}/{val,test}_${N}/${BENCHMARK_KEY}.json")
    for notebook, results_root, launcher, declared in declared_paths():
        print(f"  {notebook}")
        print(f"      results   {results_root}   via scripts/{launcher}.sh")
        if not declared:
            print("      [ n/a   ] launcher is generated at runtime, not a repo file")
            continue
        for rel, exists in declared:
            print(f"      [{'present' if exists else ' GONE  '}] {rel}")

    unattributed = [r for r in rows if r["match"] == "none"]
    print(f"\nreadouts whose episode set matches no manifest in benchmarks/: {len(unattributed)}")
    for row in unattributed[:20]:
        print(f"  {row['path']}")
    return 1 if unattributed else 0


if __name__ == "__main__":
    raise SystemExit(main())
