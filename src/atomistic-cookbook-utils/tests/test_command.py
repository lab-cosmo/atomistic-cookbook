"""Tests for ``run_command``."""

from __future__ import annotations

import subprocess
import sys

import pytest

from atomistic_cookbook_utils import run_command


def _python(args: str) -> str:
    """Return ``"<sys.executable> <args>"`` quoted for shlex."""
    import shlex

    return f"{shlex.quote(sys.executable)} {args}"


def test_runs_simple_command():
    result = run_command(_python("-c 'print(2 + 2)'"), capture_output=True)
    assert result.returncode == 0
    assert result.stdout.strip() == "4"


def test_check_raises_on_nonzero():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(_python("-c 'import sys; sys.exit(3)'"))


def test_check_false_returns_completed_process():
    result = run_command(_python("-c 'import sys; sys.exit(3)'"), check=False)
    assert result.returncode == 3


def test_cwd_kwarg(tmp_path):
    (tmp_path / "marker.txt").write_text("here")
    result = run_command(
        _python("-c 'import os; print(sorted(os.listdir(\".\")))'"),
        cwd=tmp_path,
        capture_output=True,
    )
    assert "marker.txt" in result.stdout


def test_capture_output_returns_text():
    result = run_command(_python("-c 'print(\"hi\")'"), capture_output=True)
    assert isinstance(result.stdout, str)
    assert result.stdout.strip() == "hi"


def test_env_is_passed_through():
    cmd = _python("-c 'import os; print(os.environ[\"COOKBOOK_TEST\"])'")
    result = run_command(cmd, env={"COOKBOOK_TEST": "ok"}, capture_output=True)
    assert result.stdout.strip() == "ok"


def test_quoting_is_respected():
    # The single argument "a b c" should arrive as one argv element.
    cmd = _python("-c 'import sys; print(len(sys.argv) - 1, sys.argv[1])' \"a b c\"")
    result = run_command(cmd, capture_output=True)
    assert result.stdout.strip() == "1 a b c"
