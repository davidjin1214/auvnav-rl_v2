"""PostToolUse hook: catch doc-pointer rot at the edit that introduces it.

Fires only on markdown edits, then runs the repo-wide sweep in --strict mode,
which exits non-zero solely on `real` misses -- a link to something that is
supposed to exist and does not. The other buckets the sweep reports (self-
declared absences, gitignored artefacts, plan-table forecasts) are legitimate
and stay silent here.

Baseline at the time of writing: 0 real misses, 888 ms.

Exit 2 feeds stderr back to the agent. PostToolUse runs after the write, so
this flags the miss rather than preventing it -- the point is to catch it in
the same turn instead of in a doc sweep three sessions later.

Wiring it up. `.claude/settings.json` is gitignored, so a clone gets this file
but not the configuration that runs it, and the other machine needs the entry
added by hand. Append to the existing PostToolUse "Edit|Write|MultiEdit" hooks
array:

    {
      "type": "command",
      "command": "s=\"${CLAUDE_PROJECT_DIR:-.}/.claude/hooks/doc_pointers.py\"; [ -f \"$s\" ] || exit 0; p=$(command -v python3 || command -v python) || exit 0; \"$p\" -X utf8 \"$s\"",
      "timeout": 30
    }

The `[ -f ]` guard makes a missing script a no-op rather than a failed hook,
and CLAUDE_PROJECT_DIR falls back to the working directory when unset.
"""
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TIMEOUT_S = 20  # hook-level timeout is 30s; stay well inside it


def main():
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0

    path = payload.get("tool_input", {}).get("file_path", "")
    if not path.replace("\\", "/").lower().endswith(".md"):
        return 0

    try:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_doc_pointers", "--strict"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        # A hook must never be the reason an edit fails.
        return 0

    if result.returncode == 0:
        return 0

    # Keep only the real-miss section; the full sweep prints ~60 lines.
    lines = (result.stdout or "").splitlines()
    kept, in_real = [], False
    for line in lines:
        if line.startswith("--- "):
            in_real = "真失效" in line
        if in_real:
            kept.append(line)

    detail = "\n".join(kept) if kept else (result.stdout or "")[-1500:]
    sys.stderr.write(
        "[hook] check_doc_pointers --strict failed after editing %s\n%s\n"
        "Declare the absence in the citing sentence, or fix the path.\n"
        % (os.path.basename(path), detail)
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
