"""Friendly wrapper around :func:`subprocess.run` for shell-like strings."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Optional, Union


PathLike = Union[str, Path]


def run_command(
    command: str,
    *,
    cwd: Optional[PathLike] = None,
    capture_output: bool = False,
    check: bool = True,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run a shell-like command string without invoking a shell.

    The string is split with :func:`shlex.split` and passed as a list of
    arguments to :func:`subprocess.run`. Shell features such as pipes,
    redirection, and glob expansion are *not* supported — for those, use
    :mod:`subprocess` directly.

    Parameters
    ----------
    command
        The command to run, e.g. ``"mtt train options.yaml"``.
    cwd
        Working directory in which to run the command.
    capture_output
        If true, capture stdout/stderr as text and expose them on the
        returned :class:`~subprocess.CompletedProcess`.
    check
        If true (the default), raise
        :class:`subprocess.CalledProcessError` on non-zero exit status.
    env
        Optional environment mapping. If omitted, the current process
        environment is inherited.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process object.
    """
    args = shlex.split(command)
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=capture_output,
        check=check,
        env=dict(env) if env is not None else None,
        text=True if capture_output else None,
    )
