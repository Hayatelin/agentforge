"""Approval checkpoints when stdin has no input (CI, </dev/null).

Reading EOF at the prompt must behave exactly like answering "N":
the run is denied, exits with the same code, and writes the same trace.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agentforge.orchestrator import _default_approval
from agentforge.workflow import Agent, Task

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "workflow.yaml"


def _task_and_agent():
    agent = Agent(name="editor", role="Managing Editor")
    task = Task(
        name="edit",
        description="Edit the draft.",
        agent="editor",
        require_approval=True,
    )
    return task, agent


def _raise_eof(prompt=""):
    raise EOFError


# --------------------------------------------------------------------------- #
# Unit: _default_approval
# --------------------------------------------------------------------------- #
def test_default_approval_eof_denies_without_raising(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", _raise_eof)
    task, agent = _task_and_agent()

    assert _default_approval(task, agent) is False

    out = capsys.readouterr().out
    assert 'no input available (non-interactive); treating as "no"' in out


@pytest.mark.parametrize(
    "answer, expected",
    [("y", True), ("yes", True), (" Y ", True), ("n", False), ("", False)],
)
def test_default_approval_interactive_answers_unchanged(monkeypatch, answer, expected):
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    task, agent = _task_and_agent()

    assert _default_approval(task, agent) is expected


# --------------------------------------------------------------------------- #
# End to end: `python -m agentforge run` with no stdin vs answering "N"
# --------------------------------------------------------------------------- #
def _run_cli(workdir: Path, workflow: Path, **stdin_kwargs):
    env = dict(os.environ)
    # Make the package importable even when it is not pip-installed (CI).
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    return subprocess.run(
        [sys.executable, "-m", "agentforge", "run", str(workflow),
         "--provider", "mock"],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        **stdin_kwargs,
    )


def test_cli_no_stdin_matches_answering_no(tmp_path):
    workflow = tmp_path / "workflow.yaml"
    shutil.copyfile(EXAMPLE, workflow)
    workflow = workflow.resolve()

    no_dir = tmp_path / "answer_no"
    eof_dir = tmp_path / "no_stdin"
    no_dir.mkdir()
    eof_dir.mkdir()

    answered_no = _run_cli(no_dir, workflow, input="n\n")
    no_stdin = _run_cli(eof_dir, workflow, stdin=subprocess.DEVNULL)

    # Baseline: answering "N" is a clean denial.
    assert answered_no.returncode == 2
    assert "Traceback" not in answered_no.stderr

    # EOF behaves the same way.
    assert no_stdin.returncode == answered_no.returncode
    assert "Traceback" not in no_stdin.stderr
    assert "Run aborted: approval denied for task 'edit'." in no_stdin.stderr
    assert 'no input available (non-interactive); treating as "no"' in no_stdin.stdout

    # Same trace behaviour: written to the cwd in both cases, same content shape.
    no_trace = no_dir / "workflow.trace.json"
    eof_trace = eof_dir / "workflow.trace.json"
    assert eof_trace.exists() == no_trace.exists()
    assert no_trace.exists()

    no_data = json.loads(no_trace.read_text(encoding="utf-8"))
    eof_data = json.loads(eof_trace.read_text(encoding="utf-8"))
    assert eof_data["status"] == no_data["status"] == "aborted"
    assert [(s["task"], s["status"], s["approved"]) for s in eof_data["steps"]] == [
        (s["task"], s["status"], s["approved"]) for s in no_data["steps"]
    ]
