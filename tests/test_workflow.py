"""Tests for workflow parsing, validation and DAG ordering."""

from __future__ import annotations

import pytest

from agentforge.workflow import (
    WorkflowError,
    parse_workflow,
    _topological_sort,
    Task,
)


def _minimal(tasks):
    return {
        "name": "wf",
        "mode": "dag",
        "agents": [{"name": "a", "role": "Worker"}],
        "tasks": tasks,
    }


def test_parse_minimal_workflow():
    wf = parse_workflow(
        _minimal([{"name": "t1", "description": "do it", "agent": "a"}])
    )
    assert wf.name == "wf"
    assert "a" in wf.agents
    assert wf.tasks[0].name == "t1"


def test_dag_ordering_respects_dependencies():
    raw = _minimal(
        [
            {"name": "c", "description": "c", "agent": "a", "depends_on": ["b"]},
            {"name": "b", "description": "b", "agent": "a", "depends_on": ["a1"]},
            {"name": "a1", "description": "a1", "agent": "a"},
        ]
    )
    wf = parse_workflow(raw)
    order = [t.name for t in wf.execution_order()]
    assert order == ["a1", "b", "c"]


def test_dag_tie_break_is_declaration_order():
    raw = _minimal(
        [
            {"name": "first", "description": "x", "agent": "a"},
            {"name": "second", "description": "x", "agent": "a"},
        ]
    )
    wf = parse_workflow(raw)
    assert [t.name for t in wf.execution_order()] == ["first", "second"]


def test_sequential_mode_uses_file_order():
    raw = _minimal(
        [
            {"name": "z", "description": "z", "agent": "a", "depends_on": []},
            {"name": "y", "description": "y", "agent": "a"},
        ]
    )
    raw["mode"] = "sequential"
    wf = parse_workflow(raw)
    assert [t.name for t in wf.execution_order()] == ["z", "y"]


def test_cycle_detection():
    tasks = [
        Task(name="x", description="x", agent="a", depends_on=["y"]),
        Task(name="y", description="y", agent="a", depends_on=["x"]),
    ]
    with pytest.raises(WorkflowError, match="Cycle detected"):
        _topological_sort(tasks)


def test_cycle_detected_during_parse():
    raw = _minimal(
        [
            {"name": "x", "description": "x", "agent": "a", "depends_on": ["y"]},
            {"name": "y", "description": "y", "agent": "a", "depends_on": ["x"]},
        ]
    )
    with pytest.raises(WorkflowError, match="Cycle detected"):
        parse_workflow(raw)


# --- validation errors ----------------------------------------------------- #
def test_missing_agents_list():
    with pytest.raises(WorkflowError, match="agents"):
        parse_workflow({"name": "wf", "tasks": []})


def test_unknown_agent_reference():
    raw = _minimal([{"name": "t", "description": "d", "agent": "ghost"}])
    with pytest.raises(WorkflowError, match="unknown agent"):
        parse_workflow(raw)


def test_unknown_dependency():
    raw = _minimal(
        [{"name": "t", "description": "d", "agent": "a", "depends_on": ["nope"]}]
    )
    with pytest.raises(WorkflowError, match="unknown task"):
        parse_workflow(raw)


def test_self_dependency():
    raw = _minimal(
        [{"name": "t", "description": "d", "agent": "a", "depends_on": ["t"]}]
    )
    with pytest.raises(WorkflowError, match="depend on itself"):
        parse_workflow(raw)


def test_duplicate_task_name():
    raw = _minimal(
        [
            {"name": "dup", "description": "d", "agent": "a"},
            {"name": "dup", "description": "d", "agent": "a"},
        ]
    )
    with pytest.raises(WorkflowError, match="Duplicate task"):
        parse_workflow(raw)


def test_duplicate_agent_name():
    raw = {
        "name": "wf",
        "agents": [
            {"name": "a", "role": "R"},
            {"name": "a", "role": "R"},
        ],
        "tasks": [{"name": "t", "description": "d", "agent": "a"}],
    }
    with pytest.raises(WorkflowError, match="Duplicate agent"):
        parse_workflow(raw)


def test_invalid_mode():
    raw = _minimal([{"name": "t", "description": "d", "agent": "a"}])
    raw["mode"] = "parallelish"
    with pytest.raises(WorkflowError, match="Invalid mode"):
        parse_workflow(raw)


def test_task_missing_description():
    raw = _minimal([{"name": "t", "agent": "a"}])
    with pytest.raises(WorkflowError, match="description"):
        parse_workflow(raw)
