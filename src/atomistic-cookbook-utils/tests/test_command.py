"""Tests for ``run_command``."""

from __future__ import annotations

import shlex
import subprocess
import sys

import pytest

from atomistic_cookbook_utils import run_command


def _python(args: str) -> str:
    """Return ``"<sys.executable> <args>"`` quoted for shlex."""
    return f"{shlex.quote(sys.executable)} {args}"


def test_default_runs_without_python_capture(capsys):
    # With print_output=False (the default) nothing is re-emitted via
    # print(), so capsys (which only sees sys.stdout writes) gets nothing.
    result = run_command(_python("-c 'print(\"hello\")'"))
    assert result.returncode == 0
    assert result.stdout is None
    assert capsys.readouterr().out == ""


def test_print_output_returns_captured_text(capsys):
    result = run_command(_python("-c 'print(\"payload\")'"), print_output=True)
    assert result.returncode == 0
    assert "payload" in result.stdout
    assert "payload" in capsys.readouterr().out


def test_default_check_raises_on_nonzero():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(_python("-c 'import sys; sys.exit(3)'"))


def test_check_false_does_not_raise():
    run_command(_python("-c 'import sys; sys.exit(3)'"), check=False)


def test_print_output_emits_via_print(capsys):
    run_command(_python("-c 'print(2 + 2)'"), print_output=True)
    assert capsys.readouterr().out.strip() == "4"


def test_print_output_check_raises_and_still_prints(capsys):
    cmd = _python("-c 'print(\"before fail\"); import sys; sys.exit(3)'")
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_command(cmd, print_output=True)
    assert excinfo.value.returncode == 3
    assert "before fail" in capsys.readouterr().out


def test_print_output_merges_stderr_with_stdout(capsys):
    cmd = _python(
        '-c \'import sys; print("to-out"); print("to-err", file=sys.stderr)\''
    )
    run_command(cmd, print_output=True)
    out = capsys.readouterr().out
    assert "to-out" in out
    assert "to-err" in out


def test_cwd_kwarg(tmp_path, capsys):
    (tmp_path / "marker.txt").write_text("here")
    run_command(
        _python("-c 'import os; print(sorted(os.listdir(\".\")))'"),
        cwd=tmp_path,
        print_output=True,
    )
    assert "marker.txt" in capsys.readouterr().out


def test_env_is_passed_through(capsys):
    cmd = _python("-c 'import os; print(os.environ[\"COOKBOOK_TEST\"])'")
    run_command(cmd, env={"COOKBOOK_TEST": "ok"}, print_output=True)
    assert capsys.readouterr().out.strip() == "ok"


def test_quoting_is_respected(capsys):
    # The single argument "a b c" should arrive as one argv element.
    cmd = _python("-c 'import sys; print(len(sys.argv) - 1, sys.argv[1])' \"a b c\"")
    run_command(cmd, print_output=True)
    assert capsys.readouterr().out.strip() == "1 a b c"
