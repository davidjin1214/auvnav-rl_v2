"""Tests for the published-readout to evaluation-manifest attribution (C-11).

The 2026-08-21 contamination enumeration closed over the manifests in `benchmarks/`. That
closure transfers to the chapter's tables only if every published readout actually ran on
one of those episode sets -- otherwise the audit swept files nobody used. No readout records
the manifest path it was given, so this tool rebuilds the link from the ordered
`(episode_id, seed)` sequence, and separately from what the launcher plus the driving
notebook declare.

Two properties carry the whole conclusion and are pinned here. The alarm is
`match == "none"` -- a readout whose episodes are in no manifest -- and *not* the size of an
equivalence class: several manifests freeze identical sequences, so a class of three is a
property of the manifests, not a failed attribution. And `prefix` is a pass: a 40-episode
evaluation against the head of a 100-episode file is a subset of an audited set.

`results/` and the notebook outputs are gitignored, so every tree here is a fixture.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "paper" / "thesis_ch5" / "tools"

if str(TOOLS) not in sys.path:
    sys.path.append(str(TOOLS))

attribution = importlib.import_module("ch5_manifest_attribution")

EPISODES_A = [("key_a_ep_0000", 1000), ("key_a_ep_0001", 1001),
              ("key_a_ep_0002", 1002), ("key_a_ep_0003", 1003)]
EPISODES_C = [("key_c_ep_0000", 2000), ("key_c_ep_0001", 2001)]
UNKNOWN = [("nowhere_ep_0000", 9000)]

LAUNCHER = """#!/usr/bin/env bash
MANIFEST_ROOT="${MANIFEST_ROOT:-benchmarks/suite}"
BENCHMARK_KEY="${BENCHMARK_KEY:-key_a}"
VAL_MANIFEST_EPISODES="${VAL_MANIFEST_EPISODES:-2}"
TEST_MANIFEST_EPISODES="${TEST_MANIFEST_EPISODES:-4}"
RESULTS_ROOT="${RESULTS_ROOT:-results/offline/unit_exact}"
"""


def manifest(root: Path, rel: str, episodes: list[tuple[str, int]] | None) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"note": "fixture"}
    if episodes is not None:
        payload["episodes"] = [{"episode_id": e, "seed": s} for e, s in episodes]
    path.write_text(json.dumps(payload), encoding="utf-8")


def readout(root: Path, rel: str, episodes: list[tuple[str, int]]) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"eval_episode_results": [{"episode_id": e, "seed": s} for e, s in episodes]}),
        encoding="utf-8")


def notebook(root: Path, name: str, *cells: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(
        {"cells": [{"cell_type": "code", "source": c} for c in cells]}), encoding="utf-8")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A miniature repo: two manifests freezing one sequence, plus a third."""
    bench = tmp_path / "benchmarks"
    results = tmp_path / "results" / "offline"
    scripts = tmp_path / "scripts"
    notebooks = tmp_path / "notebooks"
    scripts.mkdir(parents=True)
    (scripts / "run_offline_demo.sh").write_text(LAUNCHER, encoding="utf-8")
    notebooks.mkdir(parents=True)

    manifest(bench, "suite/test_4/key_a.json", EPISODES_A)
    manifest(bench, "suite/test_4/key_b.json", EPISODES_A)  # identical sequence
    manifest(bench, "suite/val_2/key_c.json", EPISODES_C)
    manifest(bench, "suite/not_a_manifest.json", None)      # no episodes key

    for name, value in (("REPO", tmp_path), ("BENCHMARKS", bench), ("RESULTS", results),
                        ("SCRIPTS", scripts), ("NOTEBOOKS", notebooks)):
        monkeypatch.setattr(attribution, name, value)
    monkeypatch.setattr(attribution, "PUBLISHED_UNITS",
                        {"exact unit": "results/offline/unit_exact"})
    monkeypatch.setattr(sys, "argv", ["tool", "--all"])
    return tmp_path


# --------------------------------------------------------------------------- #
# the manifest side
# --------------------------------------------------------------------------- #
def test_a_json_in_benchmarks_without_episodes_is_not_a_manifest(repo):
    assert "suite/not_a_manifest.json" not in attribution.load_manifests()


def test_manifests_freezing_the_same_sequence_form_one_equivalence_class(repo):
    classes = attribution.equivalence_classes(attribution.load_manifests())
    big = [members for members in classes.values() if len(members) > 1]
    assert big == [["suite/test_4/key_a.json", "suite/test_4/key_b.json"]]


def test_the_fingerprint_is_ordered_not_a_set(repo):
    # `generate_standard_benchmarks.py` numbers episodes in build order, so the sequence
    # is the identity of the set. Comparing as sets would merge manifests that differ
    # only in draw order -- and their episodes are then not the same episodes.
    readout(repo / "results" / "offline", "shuffled/test_result.json",
            list(reversed(EPISODES_A)))
    rows = attribution.load_readouts()
    attribution.attribute(rows, attribution.load_manifests())
    assert [r["match"] for r in rows] == ["none"]


# --------------------------------------------------------------------------- #
# the readout side
# --------------------------------------------------------------------------- #
def test_a_json_that_never_mentions_the_key_is_not_a_readout(repo):
    (repo / "results" / "offline" / "misc").mkdir(parents=True)
    (repo / "results" / "offline" / "misc" / "summary.json").write_text(
        json.dumps({"eval_success_rate": 0.9}), encoding="utf-8")
    assert attribution.load_readouts() == []


def test_a_file_naming_the_key_but_holding_no_episodes_is_skipped(repo):
    # selected_checkpoint.json names `eval_episode_results` inside a nested record; the
    # byte scan finds it, and only the parsed lookup can tell it holds nothing.
    (repo / "results" / "offline" / "misc").mkdir(parents=True)
    (repo / "results" / "offline" / "misc" / "selected_checkpoint.json").write_text(
        json.dumps({"record": {"eval_episode_results": []}}), encoding="utf-8")
    assert attribution.load_readouts() == []


def test_unparseable_json_is_skipped_rather_than_crashing_the_sweep(repo):
    (repo / "results" / "offline" / "misc").mkdir(parents=True)
    (repo / "results" / "offline" / "misc" / "broken.json").write_text(
        '{"eval_episode_results": [', encoding="utf-8")
    assert attribution.load_readouts() == []


# --------------------------------------------------------------------------- #
# attribution
# --------------------------------------------------------------------------- #
def test_an_exact_match_names_every_member_of_the_class(repo):
    readout(repo / "results" / "offline", "unit_exact/test_result.json", EPISODES_A)
    rows = attribution.load_readouts()
    attribution.attribute(rows, attribution.load_manifests())
    assert rows[0]["match"] == "exact"
    assert rows[0]["candidates"] == ["suite/test_4/key_a.json", "suite/test_4/key_b.json"]


def test_the_opening_run_of_a_longer_manifest_is_a_prefix_match(repo):
    # A 2-episode evaluation against the head of a 4-episode file. The contamination
    # conclusion still transfers: a subset of an audited set is audited.
    readout(repo / "results" / "offline", "unit_prefix/test_result.json", EPISODES_A[:2])
    rows = attribution.load_readouts()
    attribution.attribute(rows, attribution.load_manifests())
    assert rows[0]["match"] == "prefix"
    assert rows[0]["candidates"] == ["suite/test_4/key_a.json", "suite/test_4/key_b.json"]


def test_a_whole_manifest_is_not_a_prefix_of_itself(repo):
    # The prefix index stops one short of the full length, so an exact match is never
    # reported as the weaker verdict.
    readout(repo / "results" / "offline", "unit_exact/test_result.json", EPISODES_A)
    rows = attribution.load_readouts()
    attribution.attribute(rows, attribution.load_manifests())
    assert rows[0]["match"] != "prefix"


# --------------------------------------------------------------------------- #
# the alarm
# --------------------------------------------------------------------------- #
def test_a_fully_attributed_tree_exits_zero(repo, capsys):
    readout(repo / "results" / "offline", "unit_exact/test_result.json", EPISODES_A)
    readout(repo / "results" / "offline", "unit_prefix/test_result.json", EPISODES_A[:2])
    assert attribution.main() == 0
    assert "matches no manifest in benchmarks/: 0" in capsys.readouterr().out


def test_a_readout_matching_no_manifest_is_the_alarm(repo, capsys):
    readout(repo / "results" / "offline", "unit_exact/test_result.json", EPISODES_A)
    readout(repo / "results" / "offline", "stray/test_result.json", UNKNOWN)
    assert attribution.main() == 1
    out = capsys.readouterr().out
    assert "matches no manifest in benchmarks/: 1" in out
    assert "stray/test_result.json" in out


def test_a_large_equivalence_class_is_not_an_alarm(repo, capsys):
    # The exit code answers "is every readout attributable", not "is the attribution
    # unique". Two manifests freezing one sequence cannot be told apart by any readout.
    readout(repo / "results" / "offline", "unit_exact/test_result.json", EPISODES_A)
    assert attribution.main() == 0
    assert "class of 2" in capsys.readouterr().out


def test_brief_collapses_a_class_to_its_roots():
    assert attribution.brief(["suite/test_4/key_a.json"]) == "suite/test_4/key_a.json"
    assert attribution.brief(["a/x.json", "b/y.json"]) == "{a, b}"


# --------------------------------------------------------------------------- #
# the provenance side
# --------------------------------------------------------------------------- #
def test_launcher_shell_defaults_are_read_from_the_script(repo):
    defaults = attribution.launcher_defaults("run_offline_demo")
    assert defaults["MANIFEST_ROOT"] == "benchmarks/suite"
    assert defaults["BENCHMARK_KEY"] == "key_a"


def test_the_declared_manifest_path_is_derived_from_root_split_and_key(repo):
    notebook(repo / "notebooks", "demo_completed.ipynb",
             'os.environ["RESULTS_ROOT"] = "results/offline/unit_exact"\n'
             "!bash scripts/run_offline_demo.sh\n")
    (nb, results_root, launcher, declared), = attribution.declared_paths()
    assert (nb, results_root, launcher) == ("demo_completed.ipynb",
                                            "results/offline/unit_exact", "run_offline_demo")
    # Both derived from the launcher defaults; the val file is reported as absent
    # rather than quietly dropped, because "declared but missing" is a finding.
    assert declared == [("benchmarks/suite/val_2/key_a.json", False),
                        ("benchmarks/suite/test_4/key_a.json", True)]


def test_a_notebook_override_beats_the_launcher_default(repo):
    notebook(repo / "notebooks", "demo_completed.ipynb",
             'os.environ["RESULTS_ROOT"] = "results/offline/unit_exact"\n'
             'os.environ["BENCHMARK_KEY"] = "key_c"\n'
             "!bash scripts/run_offline_demo.sh\n")
    (_, _, _, declared), = attribution.declared_paths()
    assert declared == [("benchmarks/suite/val_2/key_c.json", True),
                        ("benchmarks/suite/test_4/key_c.json", False)]


def test_a_notebook_is_replayed_cell_by_cell_not_as_one_environment(repo):
    """Several notebooks drive more than one study by reassigning RESULTS_ROOT.

    Taking the whole-notebook union of `os.environ` assignments attributes the last
    study's overrides to the first, which silently re-points an earlier table's
    provenance at a manifest it never used.
    """
    notebook(repo / "notebooks", "demo_completed.ipynb",
             'os.environ["RESULTS_ROOT"] = "results/offline/unit_exact"\n'
             "!bash scripts/run_offline_demo.sh\n",
             'os.environ["RESULTS_ROOT"] = "results/offline/unit_prefix"\n'
             'os.environ["BENCHMARK_KEY"] = "key_c"\n'
             "!bash scripts/run_offline_demo.sh\n")
    records = attribution.declared_paths()
    assert [r[1] for r in records] == ["results/offline/unit_exact",
                                       "results/offline/unit_prefix"]
    # The first study must keep key_a: the reassignment happened after it ran.
    assert records[0][3][1][0].endswith("key_a.json")
    assert records[1][3][1][0].endswith("key_c.json")


def test_a_launcher_that_is_not_a_repo_file_is_reported_rather_than_guessed(repo, capsys):
    # One notebook patches a flag into the screen launcher at runtime; that file has
    # never existed in the repo, so there are no defaults to derive a path from.
    notebook(repo / "notebooks", "demo_completed.ipynb",
             'os.environ["RESULTS_ROOT"] = "results/offline/unit_exact"\n'
             "!bash scripts/run_offline_generated.sh\n")
    (_, _, launcher, declared), = attribution.declared_paths()
    assert launcher == "run_offline_generated"
    assert declared == []
    attribution.main()
    assert "launcher is generated at runtime" in capsys.readouterr().out


def test_a_launcher_manifest_root_that_no_longer_exists_is_flagged(repo, capsys):
    (repo / "scripts" / "run_offline_gone.sh").write_text(
        'MANIFEST_ROOT="${MANIFEST_ROOT:-benchmarks/deleted}"\n', encoding="utf-8")
    attribution.main()
    out = capsys.readouterr().out
    assert "GONE" in out and "benchmarks/deleted" in out
