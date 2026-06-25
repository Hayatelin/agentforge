"""Tests for the orchestrator: mock run, context wiring, approvals, trace."""

from __future__ import annotations

import json

import pytest

from agentforge.orchestrator import ApprovalDenied, Orchestrator
from agentforge.providers import MockProvider
from agentforge.workflow import parse_workflow


PIPELINE = {
    "name": "pipeline",
    "mode": "dag",
    "agents": [
        {"name": "researcher", "role": "Researcher", "goal": "find facts"},
        {"name": "writer", "role": "Writer", "goal": "write"},
    ],
    "tasks": [
        {
            "name": "research",
            "description": "Research topic X",
            "agent": "researcher",
            "output_var": "notes",
        },
        {
            "name": "write",
            "description": "Write using notes: {{ notes }}",
            "agent": "writer",
            "depends_on": ["research"],
            "output_var": "draft",
        },
    ],
}


def _wf():
    return parse_workflow(PIPELINE)


def test_mock_run_completes_and_traces():
    orch = Orchestrator(_wf(), MockProvider(), auto_approve=True)
    trace = orch.run()
    assert trace.status == "completed"
    assert [s.task for s in trace.steps] == ["research", "write"]
    assert all(s.status == "completed" for s in trace.steps)
    assert "notes" in trace.context and "draft" in trace.context


def test_context_is_substituted_into_prompt():
    orch = Orchestrator(_wf(), MockProvider(), auto_approve=True)
    trace = orch.run()
    write_step = next(s for s in trace.steps if s.task == "write")
    # The {{ notes }} placeholder must have been replaced with real output.
    assert "{{ notes }}" not in write_step.rendered_prompt
    assert "Researcher" in write_step.rendered_prompt  # contains upstream output


def test_missing_variable_marker():
    raw = {
        "name": "wf",
        "agents": [{"name": "a", "role": "R"}],
        "tasks": [
            {"name": "t", "description": "use {{ ghost }}", "agent": "a"},
        ],
    }
    orch = Orchestrator(parse_workflow(raw), MockProvider(), auto_approve=True)
    trace = orch.run()
    assert "[missing: ghost]" in trace.steps[0].rendered_prompt


def test_approval_auto_approved():
    raw = dict(PIPELINE)
    raw = json_roundtrip(raw)
    raw["tasks"][1]["require_approval"] = True
    orch = Orchestrator(parse_workflow(raw), MockProvider(), auto_approve=True)
    trace = orch.run()
    assert trace.status == "completed"
    write_step = next(s for s in trace.steps if s.task == "write")
    assert write_step.approved is True


def test_approval_denied_aborts():
    raw = json_roundtrip(PIPELINE)
    raw["tasks"][1]["require_approval"] = True
    orch = Orchestrator(
        parse_workflow(raw),
        MockProvider(),
        auto_approve=False,
        approval_fn=lambda task, agent: False,  # always deny
    )
    with pytest.raises(ApprovalDenied):
        orch.run()
    assert orch.last_trace.status == "aborted"
    # research ran, write was denied (not completed).
    statuses = {s.task: s.status for s in orch.last_trace.steps}
    assert statuses["research"] == "completed"
    assert statuses["write"] == "denied"


def test_approval_granted_via_callback():
    raw = json_roundtrip(PIPELINE)
    raw["tasks"][1]["require_approval"] = True
    orch = Orchestrator(
        parse_workflow(raw),
        MockProvider(),
        auto_approve=False,
        approval_fn=lambda task, agent: True,
    )
    trace = orch.run()
    assert trace.status == "completed"


def test_trace_saves_to_json(tmp_path):
    orch = Orchestrator(_wf(), MockProvider(), auto_approve=True)
    trace = orch.run()
    out = tmp_path / "trace.json"
    trace.save(str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert len(data["steps"]) == 2
    assert data["steps"][0]["duration_ms"] >= 0


def json_roundtrip(obj):
    """Deep-copy a plain dict via JSON so mutations don't leak across tests."""
    return json.loads(json.dumps(obj))
