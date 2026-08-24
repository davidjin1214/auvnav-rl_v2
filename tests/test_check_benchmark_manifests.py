"""Tests for the frozen-evaluation-manifest checker.

Every case runs against a tree that has the defect, not only against a healthy one: a
checker that passes on clean input has proved nothing about the case it was written for.
The tree is always a fixture under `tmp_path`. `benchmarks/` itself is a tracked record of
what was evaluated, and a test that mutated it to see the checker go red would be editing
the evidence; one test reads the real tree, and it only asserts that nothing is wrong.

The fixture writes its seeds as literals rather than reading `BENCHMARK_SPECS`. That looks
like duplication and is not: R1's seed-origin rule compares the file against the catalog, so
a fixture built *from* the catalog moves both sides together and the injection that changes
one becomes unobservable. The binding is the contract under test, so it is written twice.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_benchmark_manifests as cbm

REPO_ROOT = Path(__file__).resolve().parents[1]

# Written out rather than imported, for the reason in the module docstring.
CROSS_SEED = 1250          # BENCHMARK_SPECS["single_u10_cross_tgt15"].manifest_seed
UPSTREAM_SEED = 1400       # BENCHMARK_SPECS["single_u10_upstream_tgt15"].manifest_seed
CROSS_FLOW = "wake_data/cross.npy"
UPSTREAM_FLOW = "wake_data/upstream.npy"


def _episode(key: str, idx: int, seed: int) -> dict:
    """A frozen instance that varies with the seed.

    Deliberately not constant across episodes: two manifests holding the same instance in
    every slot would nest no matter how the comparison were broken, which is the numerical
    form of an input too symmetric to discriminate.
    """
    return {
        "episode_id": f"{key}_ep_{idx:04d}",
        "seed": seed,
        "reset_options": {
            "flow_time": 100.0 + seed * 0.25,
            "start_xy": [seed * 0.1, seed * 0.2],
            "goal_xy": [seed * 0.3, -seed * 0.05],
            "initial_heading": -1.5 + seed * 1e-4,
            "initial_speed": 0.3,
            "task_geometry": "cross_stream",
            "action_mode": "absolute_heading",
            "target_auv_max_speed_mps": 1.5,
        },
    }


def _manifest(key: str, n: int, first_seed: int, *, flow: str = CROSS_FLOW,
              geometry: str = "cross_stream", target: float = 1.5) -> dict:
    return {
        "schema_version": 2,
        "created_at": "2026-04-13T00:36:06",
        "flow_path": flow,
        "probe_layout": None,
        "history_length": None,
        "base_reset_options": {"task_geometry": geometry, "action_mode": "auto",
                               "target_auv_max_speed_mps": target, "initial_speed": 0.3},
        "episodes": [_episode(key, i, first_seed + i) for i in range(n)],
        "benchmark_id": key,
        "benchmark_group": None,
        "factor_values": {},
        "notes": None,
    }


def _put(root: Path, rel: str, payload: dict) -> Path:
    path = root / "benchmarks" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _edit(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


CROSS = "single_u10_cross_tgt15"
UPSTREAM = "single_u10_upstream_tgt15"


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """Two catalog defaults, one study's val/test split, and one deliberate reseed."""
    monkeypatch.setattr(cbm, "ROOT", tmp_path)
    _put(tmp_path, f"{CROSS}.json", _manifest(CROSS, 30, CROSS_SEED))
    _put(tmp_path, f"{CROSS}_ep100.json", _manifest(CROSS, 100, CROSS_SEED))
    _put(tmp_path, f"study/val_40/{CROSS}.json", _manifest(CROSS, 40, CROSS_SEED))
    _put(tmp_path, f"study/test_100/{CROSS}.json", _manifest(CROSS, 100, CROSS_SEED))
    _put(tmp_path, f"clean/{CROSS}_ep100_s3000.json", _manifest(CROSS, 100, 3000))
    _put(tmp_path, f"{UPSTREAM}.json",
         _manifest(UPSTREAM, 30, UPSTREAM_SEED, flow=UPSTREAM_FLOW, geometry="upstream"))
    return tmp_path


def scan(tree) -> dict[str, list[str]]:
    return cbm.scan(tree / "benchmarks")


def doc(tree, text: str) -> None:
    (tree / "docs").mkdir(exist_ok=True)
    (tree / "docs" / "notes.md").write_text(text, encoding="utf-8")


def defects(buckets: dict[str, list[str]]) -> dict[str, list[str]]:
    return {name: rows for name, rows in buckets.items() if name in cbm.DEFECT and rows}


# ------------------------------------------------------------------ the healthy baseline

def test_a_healthy_tree_has_no_defect_and_every_rule_actually_ran(tree):
    """The positive control, asserted from below.

    "No defect" is also what a checker that stopped looking reports, so each rule is
    required to have graded something: R1 four checks per manifest, R2 the three cross
    manifests that are not the longest of their family, R4 the two catalog defaults.
    """
    doc(tree, f"- `benchmarks/{CROSS}.json`：30 条，种子 {CROSS_SEED}..1279\n")
    buckets = scan(tree)

    assert defects(buckets) == {}
    assert buckets["ok"], "nothing was graded at all"
    assert sum(1 for row in buckets["ok"] if "⊂" in row) == 3, "R2 graded no nesting"
    assert sum(1 for row in buckets["ok"] if ".md:" in row) == 2, "R3 graded no document"
    assert f"{CROSS}.json" in buckets["ok"] and f"{UPSTREAM}.json" in buckets["ok"]


def test_the_shipped_tree_is_clean(tmp_path):
    """The real `benchmarks/`, which is tracked and therefore always present.

    No count is asserted -- adding a benchmark is not a defect, and a test that pinned
    "24 manifests" would go red for the wrong reason. Only that nothing is wrong.
    """
    buckets = cbm.scan(REPO_ROOT / "benchmarks")

    assert defects(buckets) == {}


# ------------------------------------------------------------------------- R1 结构自洽

def test_a_seed_that_skips_one_is_a_gap(tree):
    """Only the seed moves: the ids still run 0..n-1 and the count is unchanged.

    Injected into the reseeded manifest, which is the only one alone in its family. Doing
    it to a member of the cross family would break the nesting too, and a row that turns
    two buckets red does not say which mechanism caught it.
    """
    _edit(tree / "benchmarks" / f"clean/{CROSS}_ep100_s3000.json",
          lambda p: p["episodes"][7].__setitem__("seed", p["episodes"][7]["seed"] + 1))

    assert list(defects(scan(tree))) == ["seed-gap"]


def test_an_episode_id_whose_index_left_its_position(tree):
    """The `(episode_id, seed)` pairing is what attributes 3715 readouts to a manifest."""
    _edit(tree / "benchmarks" / f"clean/{CROSS}_ep100_s3000.json",
          lambda p: p["episodes"][7].__setitem__("episode_id", f"{CROSS}_ep_0070"))

    assert list(defects(scan(tree))) == ["id-drift"]


def test_a_directory_that_declares_a_count_the_file_does_not_hold(tree):
    """Drop the last episode of `val_40/`: still contiguous, still 0..n-1, now 39."""
    _edit(tree / "benchmarks" / f"study/val_40/{CROSS}.json",
          lambda p: p["episodes"].pop())

    assert list(defects(scan(tree))) == ["count-mismatch"]


def test_a_filename_can_declare_the_count_too(tree):
    """`_ep100` is the same declaration made in a name instead of a directory."""
    _edit(tree / "benchmarks" / f"{CROSS}_ep100.json", lambda p: p["episodes"].pop())

    assert list(defects(scan(tree))) == ["count-mismatch"]


def test_a_default_manifest_that_left_its_catalog_seed(tree):
    def shift(payload):
        for offset, episode in enumerate(payload["episodes"]):
            episode["seed"] = 5000 + offset

    _edit(tree / "benchmarks" / f"{CROSS}.json", shift)

    assert list(defects(scan(tree))) == ["seed-origin"]


def test_a_reseed_marker_moves_the_expectation_rather_than_lifting_it(tree):
    """`_s3000` is checked against 3000, not excused for disagreeing with the catalog.

    The healthy fixture already proves the permissive half: that file starts at 3000 while
    its key's catalog seed is 1250, and it passes. This is the other half.
    """
    assert "seed-origin" not in defects(scan(tree))

    def shift(payload):
        for offset, episode in enumerate(payload["episodes"]):
            episode["seed"] = 3001 + offset

    _edit(tree / "benchmarks" / f"clean/{CROSS}_ep100_s3000.json", shift)

    assert list(defects(scan(tree))) == ["seed-origin"]


# --------------------------------------------------------------------------- R2 族内嵌套

def test_a_shorter_manifest_whose_seeds_diverge_partway(tree):
    _edit(tree / "benchmarks" / f"study/val_40/{CROSS}.json",
          lambda p: [e.__setitem__("seed", e["seed"] + 500) for e in p["episodes"][20:]])

    assert "not-nested" in defects(scan(tree))


def test_a_last_bit_difference_is_not_a_defect(tree):
    """Two floats that made a JSON round trip months apart differ in the last bit."""
    _edit(tree / "benchmarks" / f"study/val_40/{CROSS}.json",
          lambda p: p["episodes"][3]["reset_options"].__setitem__(
              "initial_heading",
              p["episodes"][3]["reset_options"]["initial_heading"] + 3e-15))

    buckets = scan(tree)

    assert defects(buckets) == {}
    assert len(buckets["float-noise"]) == 1


def test_the_tolerance_does_not_swallow_a_real_difference(tree):
    """The strict side of the same threshold: 1e-6 is an instance, not a formatter."""
    _edit(tree / "benchmarks" / f"study/val_40/{CROSS}.json",
          lambda p: p["episodes"][3]["reset_options"].__setitem__(
              "initial_heading",
              p["episodes"][3]["reset_options"]["initial_heading"] + 1e-6))

    assert list(defects(scan(tree))) == ["not-nested"]


def test_manifests_of_different_task_configurations_are_never_compared(tree):
    """Same seeds, different flow field: a different instance regardless of the seed.

    Without the grouping key every manifest would be compared against every other, and a
    healthy tree would still pass -- so the rule has to be exercised on a tree where the
    comparison would fail if it happened.
    """
    def retask(payload):
        payload["flow_path"] = UPSTREAM_FLOW
        for episode in payload["episodes"]:
            episode["reset_options"]["flow_time"] += 999.0

    _edit(tree / "benchmarks" / f"study/val_40/{CROSS}.json", retask)

    assert defects(scan(tree)) == {}


# --------------------------------------------------------------------------- R3 文档所印

def test_a_document_printing_the_wrong_count(tree):
    doc(tree, f"- `benchmarks/{CROSS}.json`：40 条\n")

    assert list(defects(scan(tree))) == ["doc-mismatch"]


def test_a_document_printing_the_wrong_seed_range(tree):
    doc(tree, f"- `benchmarks/{CROSS}.json`，种子 {CROSS_SEED}..1289\n")

    assert list(defects(scan(tree))) == ["doc-mismatch"]


def test_a_brace_group_is_checked_against_every_file_it_expands_to(tree):
    """One figure quoted for several manifests is a claim about all of them.

    Both halves are asserted. Counting only the mismatch would pass just as well if the
    expansion stopped at the first option -- the val_40 half would still be wrong, and one
    is one either way.
    """
    doc(tree, f"| `study/{{val_40,test_100}}/{CROSS}.json` | 100 |\n")
    buckets = scan(tree)

    assert len(defects(buckets)["doc-mismatch"]) == 1, "only the val_40 half is wrong"
    assert sum(1 for row in buckets["ok"] if "notes.md:1" in row) == 1, "the test_100 half"


def test_a_row_pairs_its_figures_with_its_expansions_in_order(tree):
    """`{test_100,val_40}` | 100 / 40 -- the order is the claim, so reversing it is wrong.

    The first half asserts both cells were *graded*, not merely that nothing was reported:
    a pairing rule that gave up on this shape would also report nothing.
    """
    doc(tree, f"| `study/{{test_100,val_40}}/{CROSS}.json` | 100 / 40 |\n")
    buckets = scan(tree)
    assert defects(buckets) == {}
    assert sum(1 for row in buckets["ok"] if "notes.md:1" in row) == 2

    doc(tree, f"| `study/{{test_100,val_40}}/{CROSS}.json` | 40 / 100 |\n")
    assert len(defects(scan(tree))["doc-mismatch"]) == 2


def test_a_figure_ahead_of_the_path_is_not_a_claim_about_it(tree):
    """`- 100 条，与 X 逐条相同`: the 100 is the subject of the bullet, not of X.

    Kept deliberately short so the figure is *inside* the window and only the direction
    rule can reject it. Written long, the window rejects it first and removing the
    direction rule changes nothing -- which is what a first pass of this test did.
    """
    doc(tree, f"- 100 条，与 `{CROSS}.json` 逐条相同\n")

    assert defects(scan(tree)) == {}


def test_a_figure_far_along_the_line_is_not_attributed(tree):
    """The padding is a literal, not `FIGURE_WINDOW + k`.

    Deriving it from the constant makes the fixture follow whatever the constant becomes,
    so widening the window widens the padding too and the change is unobservable. 60 is
    comfortably outside the 40 this rule ships with; if that ever stops being true the
    assertion below fails rather than quietly passing.
    """
    assert cbm.FIGURE_WINDOW < 60, "the padding below no longer clears the window"
    doc(tree, f"- `benchmarks/{CROSS}.json` {'x' * 60} 40 条\n")

    assert defects(scan(tree)) == {}


def test_a_prefix_count_says_which_episodes_not_how_many(tree):
    doc(tree, f"- `benchmarks/{CROSS}_ep100.json`：前 30 条逐条等同\n")

    assert defects(scan(tree)) == {}


def test_a_directory_shorthand_resolves_to_the_one_manifest_under_it(tree):
    """`study/test_100/...` is how the quoted audit printout writes it."""
    doc(tree, "OVERLAP 100/100  study/test_100/...   种子 1250..1289\n")

    assert list(defects(scan(tree))) == ["doc-mismatch"], "the range is wrong, so it graded"


def test_a_directory_shorthand_over_several_manifests_is_refused(tree):
    _put(tree, f"study/test_100/{UPSTREAM}.json",
         _manifest(UPSTREAM, 100, UPSTREAM_SEED, flow=UPSTREAM_FLOW, geometry="upstream"))
    doc(tree, "OVERLAP 100/100  study/test_100/...   种子 1250..1289\n")

    assert defects(scan(tree)) == {}


def test_a_bare_basename_is_accepted_only_where_it_is_unique(tree):
    """Prose drops the directory. `..._s3000.json` is unique; `single_u10_cross_tgt15.json`
    names five files, and guessing which is how a checker starts inventing findings."""
    doc(tree, f"- 干净 manifest `{CROSS}_ep100_s3000.json`（40 条）\n")
    assert list(defects(scan(tree))) == ["doc-mismatch"]

    _put(tree, f"elsewhere/{CROSS}_ep100_s3000.json", _manifest(CROSS, 100, 3000))
    assert defects(scan(tree)) == {}


def test_a_dated_declaration_on_the_line_excuses_the_figure(tree):
    doc(tree, f"- `benchmarks/{CROSS}.json`（100 条）（2026-08-24 注：当时的计划值）\n")
    buckets = scan(tree)

    assert defects(buckets) == {}
    assert len(buckets["declared"]) == 1


def test_a_declaration_must_sit_on_the_line_it_excuses(tree):
    """One line up is not a declaration: a reader cannot tell which figure it covers."""
    doc(tree, f"2026-08-24 注：下一行是当时的计划值\n\n- `benchmarks/{CROSS}.json`（100 条）\n")

    assert list(defects(scan(tree))) == ["doc-mismatch"]


# ------------------------------------------------------------------------ R4 协议同规模

def test_a_default_manifest_shorter_than_its_protocol_siblings(tree):
    """`bd00950` truncated one of these from 30 episodes to 2 and nothing noticed.

    It is invisible at run time: `_resolved_eval_episodes` returns the manifest's episodes
    and ignores `--eval-episodes`, so a preset asking for 30 gets whatever the file holds.
    """
    _edit(tree / "benchmarks" / f"{UPSTREAM}.json",
          lambda p: p.__setitem__("episodes", p["episodes"][:2]))

    assert list(defects(scan(tree))) == ["cohort"]


def test_only_the_catalog_defaults_are_in_the_cohort(tree):
    """`..._ep100.json` is 100 episodes and is not a default path, so it is not ragged.

    Without the restriction the healthy fixture would already be reported, which is the
    whole reason the rule keys on `BENCHMARK_SPECS` rather than on the top-level listing.
    """
    buckets = scan(tree)

    assert not buckets["cohort"]
    assert (tree / "benchmarks" / f"{CROSS}_ep100.json").exists()


# ------------------------------------------------------------------------------ the CLI

def _cli(tree, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "scripts.check_benchmark_manifests",
         "--benchmarks-dir", str(tree / "benchmarks"), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def test_strict_exits_nonzero_only_on_a_defect(tree):
    healthy = _cli(tree)
    assert healthy.returncode == 0
    assert _cli(tree, "--strict").returncode == 0

    _edit(tree / "benchmarks" / f"{CROSS}_ep100.json",
          lambda p: p["episodes"][7].__setitem__("seed", 9999))

    assert _cli(tree).returncode == 0, "without --strict a defect is reported, not fatal"
    assert _cli(tree, "--strict").returncode == 1


def test_the_report_survives_a_non_utf8_stdout_pipe(tree):
    """The hook captures stdout through a pipe, cp936 on Windows, where `★` has no code.

    Running the module directly never shows this: the crash needs the pipe. A sibling
    sweep took every markdown edit in the repo down this way once already.
    """
    _edit(tree / "benchmarks" / f"{CROSS}_ep100.json",
          lambda p: p["episodes"][7].__setitem__("seed", 9999))
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.check_benchmark_manifests",
         "--benchmarks-dir", str(tree / "benchmarks")],
        cwd=REPO_ROOT, capture_output=True, encoding="cp936", errors="replace")

    assert proc.returncode == 0, proc.stderr
    assert "★" in proc.stdout, "the report was cut short before its defect section"
