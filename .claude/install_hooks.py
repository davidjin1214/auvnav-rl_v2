"""Merge the tracked hook configuration into this machine's `.claude/settings.json`.

Why this exists: `.claude/settings.json` is gitignored (it carries per-machine `permissions`
and env), so the hooks configured in it have never been reproducible. The hook *scripts*
have been tracked since 2026-07-28 while the wiring that runs them was not, which is why
`docs/handoff/2026-08-23-tooling-followup.md` §3 item 4 records that the doc-pointer hook
"does not fire in a clone or on another machine". `settings.hooks.json` next to this file is
that wiring, tracked; this script installs it.

The split is deliberate: only the `hooks` key is shared. `permissions` stays per-machine,
because an allowlist that suits the Windows box is not the one the Mac wants, and the same
argument that keeps the two `CLAUDE.md` files apart applies here.

    python .claude/install_hooks.py            # install, or report it is already current
    python .claude/install_hooks.py --check    # exit 1 if it would change anything
    python .claude/install_hooks.py --force    # overwrite a locally diverged hooks block

Without `--force` a hooks block that differs from the tracked one is left alone and printed,
never silently replaced: a local edit may be the newer of the two, and this script has no way
to tell. Every other key in `settings.json` is preserved untouched.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "settings.hooks.json"
TARGET = HERE / "settings.json"


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(io.open(path, encoding="utf-8").read())


def write(path: Path, payload: dict) -> None:
    io.open(path, "w", encoding="utf-8", newline="\n").write(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="report whether settings.json is current; change nothing")
    ap.add_argument("--force", action="store_true",
                    help="replace a hooks block that has diverged locally")
    args = ap.parse_args(argv)

    if not SOURCE.exists():
        print(f"找不到 {SOURCE}", file=sys.stderr)
        return 2
    wanted = load(SOURCE).get("hooks")
    if wanted is None:
        print(f"{SOURCE.name} 里没有 hooks 键", file=sys.stderr)
        return 2

    settings = load(TARGET)
    current = settings.get("hooks")

    if current == wanted:
        print(f"已是最新：{TARGET.name} 的 hooks 段与 {SOURCE.name} 一致")
        return 0

    if current is not None and not args.force:
        print(f"⚠ {TARGET.name} 的 hooks 段与 {SOURCE.name} 不一致，未改动。")
        print("  本机的可能才是新的那份——先看清差异，确认要用库里这份再 --force。")
        print(f"--- 本机 {TARGET.name}:\n{json.dumps(current, ensure_ascii=False, indent=2)}")
        print(f"--- 库里 {SOURCE.name}:\n{json.dumps(wanted, ensure_ascii=False, indent=2)}")
        return 1

    if args.check:
        verb = "缺 hooks 段" if current is None else "hooks 段已偏离"
        print(f"{TARGET.name} {verb}——跑 `python .claude/install_hooks.py` 装上")
        return 1

    settings["hooks"] = wanted
    write(TARGET, settings)
    kept = [k for k in settings if k != "hooks"]
    print(f"已写入 {TARGET.name} 的 hooks 段"
          + (f"；其余键原样保留：{', '.join(kept)}" if kept else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
