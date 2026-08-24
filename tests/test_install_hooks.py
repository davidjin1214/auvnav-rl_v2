"""Tests for the hook installer and for the tracked hook configuration itself.

The installer is three branches and would not be worth a test file on its own. The tracked
config is: `.claude/settings.hooks.json` is the only copy of the wiring that runs
`.claude/hooks/doc_pointers.py`, and nothing else checks that it stays valid, that it still
names a script that exists, or that it has not acquired an absolute path from whichever
machine last edited it -- which would install cleanly and then fail silently on the other one.

The installer cases run against fixtures under `tmp_path` by pointing the module's SOURCE and
TARGET at it, rather than by touching the real `settings.json`: that file is this machine's
live configuration, and a test that rewrote it would be editing the thing it is checking.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAUDE_DIR = REPO_ROOT / ".claude"
SOURCE = CLAUDE_DIR / "settings.hooks.json"


def _load_installer():
    spec = importlib.util.spec_from_file_location("install_hooks",
                                                  CLAUDE_DIR / "install_hooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ih(tmp_path, monkeypatch):
    mod = _load_installer()
    monkeypatch.setattr(mod, "SOURCE", tmp_path / "settings.hooks.json")
    monkeypatch.setattr(mod, "TARGET", tmp_path / "settings.json")
    mod.write(mod.SOURCE, {"hooks": {"PostToolUse": [{"matcher": "Edit"}]}})
    return mod


def _target(ih) -> dict:
    return json.loads(io.open(ih.TARGET, encoding="utf-8").read())


# --------------------------------------------------------------------------- installer

def test_a_settings_file_that_does_not_exist_yet_is_created(ih):
    assert ih.main([]) == 0
    assert _target(ih)["hooks"]["PostToolUse"] == [{"matcher": "Edit"}]


def test_the_other_keys_are_preserved(ih):
    """`permissions` is per-machine and must survive the merge untouched."""
    ih.write(ih.TARGET, {"permissions": {"allow": ["Bash(git status)"]}})
    assert ih.main([]) == 0
    after = _target(ih)
    assert after["permissions"] == {"allow": ["Bash(git status)"]}
    assert "hooks" in after


def test_an_identical_hooks_block_is_a_no_op(ih):
    ih.main([])
    before = io.open(ih.TARGET, encoding="utf-8").read()
    assert ih.main([]) == 0
    assert io.open(ih.TARGET, encoding="utf-8").read() == before


def test_a_diverged_hooks_block_is_left_alone(ih):
    """The local one may be the newer of the two; refusing beats guessing."""
    ih.write(ih.TARGET, {"hooks": {"PostToolUse": [{"matcher": "LOCAL"}]}})
    assert ih.main([]) == 1
    assert _target(ih)["hooks"]["PostToolUse"] == [{"matcher": "LOCAL"}]


def test_force_replaces_a_diverged_hooks_block(ih):
    ih.write(ih.TARGET, {"hooks": {"PostToolUse": [{"matcher": "LOCAL"}]}})
    assert ih.main(["--force"]) == 0
    assert _target(ih)["hooks"]["PostToolUse"] == [{"matcher": "Edit"}]


def test_check_reports_a_missing_block_without_writing(ih):
    assert ih.main(["--check"]) == 1
    assert not ih.TARGET.exists()


def test_check_is_quiet_when_already_current(ih):
    ih.main([])
    assert ih.main(["--check"]) == 0


# --------------------------------------------------------------------------- the config

def test_the_tracked_hook_config_is_valid_json_with_a_hooks_key():
    payload = json.loads(io.open(SOURCE, encoding="utf-8").read())
    assert set(payload) == {"hooks"}, "只共享 hooks 段；permissions 是本机的"
    assert payload["hooks"], "hooks 段是空的"


def test_every_hook_script_the_config_names_exists():
    """A wired-up hook pointing at a moved script fails at edit time, not here."""
    text = io.open(SOURCE, encoding="utf-8").read()
    for rel in {".claude/hooks/" + n for n in text.split(".claude/hooks/")[1:]}:
        name = rel.split(".py")[0] + ".py"
        assert (REPO_ROOT / name).is_file(), f"{name} 不存在"


def test_the_config_carries_no_machine_specific_absolute_path():
    """It installs cleanly on the other machine and then never fires."""
    text = io.open(SOURCE, encoding="utf-8").read()
    for bad in ("C:\\\\", "C:/", "D:\\\\", "D:/", "/Users/", "/home/", "miniconda"):
        assert bad not in text, f"hook 配置里出现了本机路径片段 {bad!r}"
