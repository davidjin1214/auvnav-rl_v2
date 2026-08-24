"""PostToolUse hook: catch doc-reference rot at the edit that introduces it.

Fires only on markdown edits, then runs five repo-wide sweeps in --strict mode:

    check_doc_pointers        does the cited path exist
    check_doc_code_refs       is the cited *line* still that line, and does the
                              cited function name exist at all
    audit_published_numbers   can a published figure still be recomputed from the
                              per-seed JSON its report says it came from
    check_status_claims       does a table's status cell still agree with the banner
                              of the document that row is about
    check_benchmark_manifests does a line quoting an evaluation manifest's episode
                              count or seed range still agree with the manifest

The first two are complementary by construction -- check_doc_pointers strips a
trailing `:N` before checking, and says so in its own `resolve()`, which is the
gap the second fills. The third checks a different thing again: not whether a
citation resolves, but whether a number is still true. Editing a report is
exactly when its traceback spec stops pointing at the figure it covers, and this
says so in the same turn. The fourth is the same idea for prose state rather than
numbers: editing a doc's banner is exactly when the tables citing it go stale,
and nothing else in this set reads a status cell. The fifth covers the one class
of figure the third cannot: an episode count or a seed range is a property of a
`benchmarks/` manifest, not a metric inside a readout, so no traceback spec has a
source for it. All five grade their findings, and only the defect buckets fail
--strict: self-declared absences, gitignored artefacts, plan-table forecasts,
citations a line or two off a quoted fragment, readings whose `results/` is not on
this machine, last-bit float differences between manifests, and tables a doc
freezes on purpose are all legitimate and stay silent.

Baseline at the time of writing: 0 findings from any, 1.5 s + 2.1 s + 1.3 s + 0.6 s +
0.8 s (the third grew when the arrival_v2 chain landed -- it reads 39-row CSVs, not only
terminal JSON).

Exit 2 feeds stderr back to the agent. PostToolUse runs after the write, so this
flags the finding rather than preventing it -- the point is to catch it in the
same turn instead of in a doc sweep three sessions later.

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

# module, the marker that identifies a defect section in its report, and what to
# do about it. Both sweeps print every bucket; without the marker the hook would
# hand back ~60 lines of buckets that did not fail.
SWEEPS = (
    ("scripts.check_doc_pointers", "真失效",
     "Declare the absence in the citing sentence, or fix the path."),
    ("scripts.check_doc_code_refs", "★",
     "Re-read the cited line and correct the number, or say in the same sentence "
     "that the name is historical and give the real one."),
    ("scripts.audit_published_numbers", "★",
     "A traceback spec under docs/tracebacks/ no longer finds the figure it checks, "
     "or the figure no longer matches its per-seed source. Re-point the spec only "
     "after reading why the report changed."),
    ("scripts.check_status_claims", "★",
     "A table's status cell disagrees with the banner of the document it is about, or "
     "holds a forecast (TBD / a batch name) where a state belongs. Read what the cited "
     "doc says now; if the table is frozen on purpose, say so in the prose above it."),
    ("scripts.check_benchmark_manifests", "★",
     "A line quoting an evaluation manifest's episode count or seed range no longer "
     "matches the manifest, or the manifests under benchmarks/ stopped constraining each "
     "other. Read the manifest, not the other document that quotes it; if the figure is "
     "a planned or historical value being kept on purpose, date the note on that line."),
)


def _defect_section(stdout, marker):
    """The report's failing sections only. Sections start with `--- `."""
    kept, keeping = [], False
    for line in (stdout or "").splitlines():
        if line.startswith("--- "):
            keeping = marker in line
        if keeping:
            kept.append(line)
    return "\n".join(kept) if kept else (stdout or "")[-1500:]


def main():
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0

    path = payload.get("tool_input", {}).get("file_path", "")
    if not path.replace("\\", "/").lower().endswith(".md"):
        return 0

    failed = []
    for module, marker, hint in SWEEPS:
        try:
            result = subprocess.run(
                [sys.executable, "-m", module, "--strict"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError):
            # A hook must never be the reason an edit fails.
            continue
        if result.returncode != 0:
            failed.append((module, _defect_section(result.stdout, marker), hint))

    if not failed:
        return 0

    for module, detail, hint in failed:
        sys.stderr.write(
            "[hook] %s --strict failed after editing %s\n%s\n%s\n"
            % (module.rsplit(".", 1)[-1], os.path.basename(path), detail, hint)
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
