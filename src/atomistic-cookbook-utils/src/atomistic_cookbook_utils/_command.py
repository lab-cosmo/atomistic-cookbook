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
    check: bool = True,
    env: Optional[Mapping[str, str]] = None,
    print_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a shell-like command string.

    The string is split with :func:`shlex.split` and passed as a list of
    arguments to :func:`subprocess.run`. Shell features such as pipes,
    redirection, and glob expansion are *not* supported — for those, use
    :mod:`subprocess` directly.

    By default the child process inherits the parent's stdout and
    stderr, so output appears wherever the parent's output goes. Pass
    ``print_output=True`` to capture the command's combined stdout and
    stderr and re-emit it via :func:`print`, which makes the output show
    up in sphinx-gallery rendered cells (sphinx-gallery captures
    ``sys.stdout`` but not raw file-descriptor writes from children).

    Parameters
    ----------
    command
        The command to run, e.g. ``"ls -la"``.
    cwd
        Working directory in which to run the command.
    check
        If true (the default), raise
        :class:`subprocess.CalledProcessError` on non-zero exit status.
        When ``print_output`` is true, any captured output is printed
        before the exception is raised.
    env
        Optional environment mapping. If omitted, the current process
        environment is inherited.
    print_output
        If true, capture the child's combined stdout and stderr and
        re-emit it via :func:`print` so it is visible to
        ``sys.stdout``-based capture systems like sphinx-gallery. The
        captured text is also accessible as ``.stdout`` on the returned
        :class:`~subprocess.CompletedProcess`.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process. When ``print_output`` is true,
        ``result.stdout`` holds the captured combined output as text;
        otherwise it is unset.
    """
    args = shlex.split(command)
    if print_output:
        result = subprocess.run(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            env=dict(env) if env is not None else None,
            text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, args, output=result.stdout
            )
        return result
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        env=dict(env) if env is not None else None,
    )
