"""Tests for the enumeration reconciliation in the seed-overlap audit.

The audit's own directory listing is the thing under suspicion here. On the Colab/Drive FUSE
mount a readdir of ``offline_data/`` came back short in two separate processes while a lookup of
one of the missing paths succeeded in a third, so an audit that passes on a healthy filesystem
proves nothing about the case that matters. Every test below hands the reconciliation a listing
that lies and checks that it says so.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_seed_overlap as audit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_dataset(root: Path, name: str, *, seed: int = 0, episodes: int = 100) -> Path:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": seed,
        "num_episodes": episodes,
        "flow_path": "wake_data/does_not_need_to_exist.npy",
        "task_geometry": "cross_stream",
        "target_speed": 1.5,
    }
    (out / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return out


@pytest.fixture()
def offline_data(tmp_path, monkeypatch):
    """A three-dataset tree: two at the top level, one nested a level down."""
    root = tmp_path / "offline_data"
    _write_dataset(root, "alpha_ep1000")
    _write_dataset(root, "beta_ep2000", episodes=2000)
    _write_dataset(root, "collection/gamma_ep500", episodes=500)
    monkeypatch.setattr(audit, "OFFLINE_DATA_DIR", root)
    return root


@pytest.fixture()
def ledger(tmp_path):
    return tmp_path / "ledger.txt"


def test_healthy_enumeration_reconciles(offline_data, ledger):
    names = audit._load_datasets()
    assert set(names) == {"alpha_ep1000", "beta_ep2000", "collection/gamma_ep500"}

    audit.record_ledger(names, ledger)
    rec = audit.reconcile(names, ledger)

    assert not rec.failed
    assert rec.missed == []
    assert rec.shadowed == []
    assert rec.unrecorded == []


def test_dropped_top_level_dataset_is_reported(offline_data, ledger):
    """The Drive failure, reproduced: the file is there, the listing does not mention it."""
    names = audit._load_datasets()
    audit.record_ledger(names, ledger)

    truncated = {k: v for k, v in names.items() if k != "beta_ep2000"}
    rec = audit.reconcile(truncated, ledger)

    assert rec.missed == ["beta_ep2000"]
    assert rec.failed
    # Same name from the other direction: os.scandir still sees what the listing dropped.
    assert rec.shadowed == ["beta_ep2000"]
    assert rec.absent == []


def test_dropped_nested_dataset_is_reported(offline_data, ledger):
    """A nested drop has no top-level shadow, so the ledger is the only thing that catches it."""
    names = audit._load_datasets()
    audit.record_ledger(names, ledger)

    truncated = {k: v for k, v in names.items() if k != "collection/gamma_ep500"}
    rec = audit.reconcile(truncated, ledger)

    assert rec.missed == ["collection/gamma_ep500"]
    assert rec.shadowed == []
    assert rec.failed


def test_deleted_dataset_is_absent_not_missed(offline_data, ledger):
    """The ledger is a cross-host union, so a name this host never had must not raise an alarm."""
    names = audit._load_datasets()
    audit.record_ledger(names, ledger)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("only_on_another_host_ep1000\n")

    rec = audit.reconcile(names, ledger)

    assert rec.absent == ["only_on_another_host_ep1000"]
    assert rec.missed == []
    assert not rec.failed


def test_record_never_shrinks_the_ledger(offline_data, ledger):
    """A short listing must not be able to quietly retire the names it failed to return."""
    names = audit._load_datasets()
    audit.record_ledger(names, ledger)

    truncated = {k: v for k, v in names.items() if k != "beta_ep2000"}
    added = audit.record_ledger(truncated, ledger)

    assert added == []
    assert "beta_ep2000" in audit._read_ledger(ledger)


def test_record_keeps_the_provenance_header(offline_data, ledger):
    ledger.write_text("# where these came from\n", encoding="utf-8")

    audit.record_ledger(audit._load_datasets(), ledger)

    assert ledger.read_text(encoding="utf-8").startswith("# where these came from\n")


def test_new_dataset_is_flagged_as_unrecorded(offline_data, ledger):
    audit.record_ledger(audit._load_datasets(), ledger)
    _write_dataset(offline_data, "delta_ep1000")

    rec = audit.reconcile(audit._load_datasets(), ledger)

    assert rec.unrecorded == ["delta_ep1000"]
    assert not rec.failed  # a new dataset is news, not a broken listing


def test_symlinked_dataset_is_shadowed(offline_data, ledger, tmp_path):
    """Why `shadowed` exists: pathlib's rglob refuses to descend into a symlinked directory.

    This repo mounts its data through links, so a dataset linked in rather than copied in would
    otherwise drop out of the audit without a word.
    """
    elsewhere = _write_dataset(tmp_path / "external", "epsilon_ep1000")
    try:
        (offline_data / "epsilon_ep1000").symlink_to(elsewhere, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # Windows without developer mode
        pytest.skip(f"cannot create a symlink here: {exc}")

    names = audit._load_datasets()
    rec = audit.reconcile(names, ledger)

    assert "epsilon_ep1000" not in names
    assert rec.shadowed == ["epsilon_ep1000"]
    assert rec.failed


def test_cli_exits_nonzero_when_the_listing_is_short(offline_data, ledger, monkeypatch, capsys):
    """The notebook runs this with `!python -m`, where the exit code is the only hard signal."""
    audit.record_ledger(audit._load_datasets(), ledger)
    monkeypatch.setattr(audit, "BENCHMARKS_DIR", offline_data.parent / "benchmarks_empty")
    (offline_data.parent / "benchmarks_empty").mkdir()
    full = audit._load_datasets()
    monkeypatch.setattr(
        audit, "_load_datasets", lambda: {k: v for k, v in full.items() if k != "beta_ep2000"}
    )
    monkeypatch.setattr(sys, "argv", ["audit_seed_overlap", "--ledger", str(ledger)])

    with pytest.raises(SystemExit) as excinfo:
        audit.main()

    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    assert "[ENUM MISS] beta_ep2000" in out
    assert "unsupported" in out


def test_shipped_ledger_parses_and_covers_this_host():
    """The checked-in ledger must stay usable: no stray syntax, and no miss on this machine."""
    names = audit._read_ledger(audit.LEDGER_PATH)

    assert len(names) > 25
    assert all(name == name.strip() and not name.startswith("#") for name in names)

    rec = audit.reconcile(audit._load_datasets(), audit.LEDGER_PATH)
    assert rec.missed == [], f"the enumeration dropped {rec.missed}"
    assert rec.shadowed == [], f"the recursive glob dropped {rec.shadowed}"


def test_module_runs_as_a_script(tmp_path):
    """`python -m scripts.audit_seed_overlap` is how every notebook and doc invokes it.

    Pointed at a fixture tree, not this host's `offline_data/`. Scanning the real one made
    the assertion below mean the opposite of what it reads: a non-zero exit is the tool
    reporting an incomplete enumeration -- a finding, not a defect -- so a true positive
    would fail the suite, and on a fresh clone (empty tree) the check passes vacuously.
    The exit-2 contract itself is pinned in-process by
    `test_cli_exits_nonzero_when_the_listing_is_short`.
    """
    data_dir = tmp_path / "offline_data"
    _write_dataset(data_dir, "alpha_ep1000")
    ledger_path = tmp_path / "ledger.txt"
    ledger_path.write_text("alpha_ep1000\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable, "-X", "utf8", "-m", "scripts.audit_seed_overlap",
            "--data-dir", str(data_dir),
            "--ledger", str(ledger_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "enumeration reconciliation" in proc.stdout
