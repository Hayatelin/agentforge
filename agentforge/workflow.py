"""Workflow definition, loading and validation.

This module owns the data model (``Agent``, ``Task``, ``Workflow``) and all
of the parsing/validation logic that turns a YAML file into validated,
typed objects. Validation is deliberately strict and produces clear,
actionable error messages -- a bad workflow file should fail loudly long
before any provider is invoked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


class WorkflowError(Exception):
    """Raised when a workflow file is malformed or semantically invalid.

    The message is intended to be shown directly to the user, so it should
    explain *what* is wrong and ideally *where*.
    """


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Agent:
    """A persona that performs tasks.

    Attributes:
        name: Unique identifier referenced by tasks.
        role: Short job title (e.g. "Senior Researcher").
        goal: What the agent is trying to achieve.
        backstory: Flavour/context that shapes the agent's behaviour.
        model: Provider-specific model name (e.g. "gpt-4o-mini" or "mock").
    """

    name: str
    role: str
    goal: str = ""
    backstory: str = ""
    model: str = "mock"


@dataclass
class Task:
    """A single unit of work assigned to an agent.

    Attributes:
        name: Unique identifier.
        description: The instruction handed to the agent (may reference
            ``{{ output_var }}`` placeholders from upstream tasks).
        agent: Name of the agent that runs this task.
        depends_on: Names of tasks that must complete first.
        output_var: Name under which this task's output is stored in the
            shared context, so downstream tasks can reference it.
        require_approval: If True, the orchestrator pauses for human
            confirmation before running this task (human-in-the-loop).
    """

    name: str
    description: str
    agent: str
    depends_on: List[str] = field(default_factory=list)
    output_var: Optional[str] = None
    require_approval: bool = False


@dataclass
class Workflow:
    """A complete, validated workflow.

    Attributes:
        name: Human-friendly workflow name.
        description: What the workflow does.
        mode: "dag" (dependency order) or "sequential" (file order).
        agents: Mapping of agent name -> Agent.
        tasks: Tasks in their declared order.
    """

    name: str
    description: str
    mode: str
    agents: Dict[str, Agent]
    tasks: List[Task]

    def task_by_name(self, name: str) -> Task:
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(name)

    def execution_order(self) -> List[Task]:
        """Return tasks in the order they should be executed.

        For "sequential" mode this is simply the declared order. For "dag"
        mode this is a deterministic topological sort (Kahn's algorithm)
        that breaks ties using the original declaration order, so runs are
        reproducible.
        """

        if self.mode == "sequential":
            return list(self.tasks)
        return _topological_sort(self.tasks)


# --------------------------------------------------------------------------- #
# DAG ordering
# --------------------------------------------------------------------------- #
def _topological_sort(tasks: List[Task]) -> List[Task]:
    """Deterministic topological sort of tasks by their ``depends_on`` edges.

    Ties (tasks that are simultaneously ready) are resolved by original
    declaration order to keep output stable. Raises ``WorkflowError`` if a
    cycle is detected. Assumes dependency names have already been validated.
    """

    index = {task.name: pos for pos, task in enumerate(tasks)}
    indegree: Dict[str, int] = {task.name: 0 for task in tasks}
    dependents: Dict[str, List[str]] = {task.name: [] for task in tasks}

    for task in tasks:
        for dep in task.depends_on:
            indegree[task.name] += 1
            dependents[dep].append(task.name)

    # Ready set: tasks with no unmet dependencies, kept sorted by index.
    ready = sorted(
        (name for name, deg in indegree.items() if deg == 0),
        key=lambda n: index[n],
    )

    ordered: List[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        newly_ready: List[str] = []
        for child in dependents[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                newly_ready.append(child)
        if newly_ready:
            ready.extend(newly_ready)
            ready.sort(key=lambda n: index[n])

    if len(ordered) != len(tasks):
        remaining = sorted(set(indegree) - set(ordered))
        raise WorkflowError(
            "Cycle detected in task dependencies involving: "
            + ", ".join(remaining)
        )

    by_name = {task.name: task for task in tasks}
    return [by_name[name] for name in ordered]


# --------------------------------------------------------------------------- #
# Loading & validation
# --------------------------------------------------------------------------- #
def load_workflow(path: str) -> Workflow:
    """Load and validate a workflow from a YAML file.

    Raises:
        WorkflowError: if the file is missing, unparseable, or invalid.
    """

    if not os.path.isfile(path):
        raise WorkflowError(f"Workflow file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via tests
        raise WorkflowError(f"Could not parse YAML: {exc}") from exc

    return parse_workflow(raw)


def parse_workflow(raw: Any) -> Workflow:
    """Validate an already-parsed mapping and build a ``Workflow``.

    Separated from file loading so it can be unit-tested directly with
    in-memory dictionaries.
    """

    if not isinstance(raw, dict):
        raise WorkflowError("Top-level workflow must be a mapping/object.")

    name = str(raw.get("name", "Untitled workflow"))
    description = str(raw.get("description", ""))
    mode = str(raw.get("mode", "dag")).lower()
    if mode not in ("dag", "sequential"):
        raise WorkflowError(
            f"Invalid mode {mode!r}; expected 'dag' or 'sequential'."
        )

    agents = _parse_agents(raw.get("agents"))
    tasks = _parse_tasks(raw.get("tasks"))

    _validate_references(agents, tasks)
    # Validate DAG integrity eagerly so `validate` catches cycles too.
    if mode == "dag":
        _topological_sort(tasks)

    return Workflow(
        name=name,
        description=description,
        mode=mode,
        agents=agents,
        tasks=tasks,
    )


def _parse_agents(raw_agents: Any) -> Dict[str, Agent]:
    if not isinstance(raw_agents, list) or not raw_agents:
        raise WorkflowError("Workflow must define a non-empty 'agents' list.")

    agents: Dict[str, Agent] = {}
    for i, item in enumerate(raw_agents):
        if not isinstance(item, dict):
            raise WorkflowError(f"agents[{i}] must be a mapping.")
        agent_name = item.get("name")
        if not agent_name or not isinstance(agent_name, str):
            raise WorkflowError(f"agents[{i}] is missing a string 'name'.")
        if agent_name in agents:
            raise WorkflowError(f"Duplicate agent name: {agent_name!r}.")
        role = item.get("role")
        if not role or not isinstance(role, str):
            raise WorkflowError(
                f"agent {agent_name!r} is missing a string 'role'."
            )
        agents[agent_name] = Agent(
            name=agent_name,
            role=role,
            goal=str(item.get("goal", "")),
            backstory=str(item.get("backstory", "")),
            model=str(item.get("model", "mock")),
        )
    return agents


def _parse_tasks(raw_tasks: Any) -> List[Task]:
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise WorkflowError("Workflow must define a non-empty 'tasks' list.")

    tasks: List[Task] = []
    seen: set[str] = set()
    for i, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            raise WorkflowError(f"tasks[{i}] must be a mapping.")
        task_name = item.get("name")
        if not task_name or not isinstance(task_name, str):
            raise WorkflowError(f"tasks[{i}] is missing a string 'name'.")
        if task_name in seen:
            raise WorkflowError(f"Duplicate task name: {task_name!r}.")
        seen.add(task_name)

        description = item.get("description")
        if not description or not isinstance(description, str):
            raise WorkflowError(
                f"task {task_name!r} is missing a string 'description'."
            )
        agent = item.get("agent")
        if not agent or not isinstance(agent, str):
            raise WorkflowError(
                f"task {task_name!r} is missing a string 'agent'."
            )

        depends_on = item.get("depends_on", []) or []
        if not isinstance(depends_on, list) or any(
            not isinstance(d, str) for d in depends_on
        ):
            raise WorkflowError(
                f"task {task_name!r}: 'depends_on' must be a list of strings."
            )

        output_var = item.get("output_var")
        if output_var is not None and not isinstance(output_var, str):
            raise WorkflowError(
                f"task {task_name!r}: 'output_var' must be a string."
            )

        require_approval = bool(item.get("require_approval", False))

        tasks.append(
            Task(
                name=task_name,
                description=description,
                agent=agent,
                depends_on=list(depends_on),
                output_var=output_var,
                require_approval=require_approval,
            )
        )
    return tasks


def _validate_references(agents: Dict[str, Agent], tasks: List[Task]) -> None:
    """Ensure every task points at a real agent and real dependencies."""

    task_names = {task.name for task in tasks}
    for task in tasks:
        if task.agent not in agents:
            raise WorkflowError(
                f"task {task.name!r} references unknown agent {task.agent!r}."
            )
        for dep in task.depends_on:
            if dep == task.name:
                raise WorkflowError(
                    f"task {task.name!r} cannot depend on itself."
                )
            if dep not in task_names:
                raise WorkflowError(
                    f"task {task.name!r} depends on unknown task {dep!r}."
                )
