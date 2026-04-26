"""Helpers shared by Colab/Jupyter notebooks.

Purpose: keep notebook cells short and stop reinventing the same boilerplate
(subprocess streaming, env munging) in every sprint notebook.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Mapping, Sequence


def run_streaming(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    prefix: str | None = None,
) -> int:
    """Run ``cmd`` and stream its output to the notebook in real time.

    Why: ``subprocess.run`` block-buffers stdout when stdout is not a tty
    (which is the case inside Jupyter), so long training runs appear silent
    until they finish. ``Popen`` + line-iterator + ``PYTHONUNBUFFERED=1``
    forces the child to flush per line, so progress is visible live.

    stderr is merged into stdout to preserve interleaved log ordering.

    Args:
        cmd: argv list, e.g. ``['bash', 'scripts/run_protocol_stage_common.sh']``.
        env: environment for the child. ``PYTHONUNBUFFERED=1`` is force-set.
            Pass ``None`` to inherit the current process env.
        check: raise ``CalledProcessError`` on non-zero exit (default True).
        prefix: optional string prepended to every streamed line, e.g. ``'[P1] '``.
            Useful when running back-to-back stages in one cell.

    Returns:
        The child's exit code (always 0 if ``check=True`` and no error).
    """
    child_env = dict(env if env is not None else os.environ)
    child_env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        list(cmd),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    assert process.stdout is not None  # for type checkers; PIPE was given

    try:
        for line in process.stdout:
            if prefix is not None:
                sys.stdout.write(prefix)
            sys.stdout.write(line)
            sys.stdout.flush()
    finally:
        process.stdout.close()

    returncode = process.wait()
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(cmd))
    return returncode
