"""End-to-end CLI tests for init, validate and run."""

from __future__ import annotations

import json
import os

from agentforge.cli import main


def test_init_validate_run_cycle(tmp_path, capsys):
    wf_path = tmp_path / "workflow.yaml"

    # init
    assert main(["init", str(wf_path)]) == 0
    assert wf_path.exists()

    # validate
    assert main(["validate", str(wf_path)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "research -> write -> edit" in out

    # run (auto-approve, mock provider) with explicit trace path
    trace_path = tmp_path / "run.trace.json"
    rc = main(
        ["run", str(wf_path), "--yes", "--provider", "mock", "--trace", str(trace_path)]
    )
    assert rc == 0
    assert trace_path.exists()
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert {s["task"] for s in data["steps"]} == {"research", "write", "edit"}


def test_init_refuses_overwrite(tmp_path):
    wf_path = tmp_path / "workflow.yaml"
    assert main(["init", str(wf_path)]) == 0
    assert main(["init", str(wf_path)]) == 1  # refuse
    assert main(["init", str(wf_path), "--force"]) == 0  # allowed


def test_validate_bad_file(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nagents: []\ntasks: []\n", encoding="utf-8")
    assert main(["validate", str(bad)]) == 1


def test_example_workflow_is_valid():
    # The shipped example must validate.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example = os.path.join(here, "examples", "workflow.yaml")
    assert main(["validate", example]) == 0
