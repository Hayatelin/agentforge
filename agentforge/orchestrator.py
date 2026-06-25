"""The orchestrator -- runs a workflow step by step.

Responsibilities:

* Resolve execution order (DAG or sequential) via the :class:`Workflow`.
* Build a per-task system prompt from the agent persona and render the
  task description against the shared context (``{{ var }}`` substitution).
* Honour human-in-the-loop checkpoints (``require_approval``) by prompting
  on the console, unless auto-approval is enabled.
* Invoke the chosen provider and record everything into a :class:`RunTrace`.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Dict, Optional

from .providers import CompletionRequest, Provider
from .trace import RunTrace, StepRecord
from .workflow import Agent, Task, Workflow

# A prompt callback returns True to approve, False to deny. Injected so the
# orchestrator is testable without real stdin.
ApprovalFn = Callable[[Task, Agent], bool]
# A logger callback for user-facing progress output.
LogFn = Callable[[str], None]

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


class ApprovalDenied(Exception):
    """Raised when a human rejects a required-approval checkpoint."""

    def __init__(self, task: Task) -> None:
        super().__init__(f"Approval denied for task {task.name!r}.")
        self.task = task


def _default_approval(task: Task, agent: Agent) -> bool:
    """Console approval prompt used in interactive runs."""

    print(f"\n  [checkpoint] Task '{task.name}' (agent: {agent.name}) "
          "requires approval before running.")
    print(f"    {task.description.strip().splitlines()[0]}")
    answer = input("    Proceed? [y/N] ").strip().lower()
    return answer in ("y", "yes")


class Orchestrator:
    """Executes a :class:`Workflow` against a :class:`Provider`."""

    def __init__(
        self,
        workflow: Workflow,
        provider: Provider,
        *,
        auto_approve: bool = False,
        approval_fn: Optional[ApprovalFn] = None,
        log_fn: Optional[LogFn] = None,
    ) -> None:
        self.workflow = workflow
        self.provider = provider
        self.auto_approve = auto_approve
        self._approval_fn = approval_fn or _default_approval
        self._log = log_fn or (lambda msg: None)
        # Shared context: output_var -> output text, for templating.
        self.context: Dict[str, str] = {}
        # The most recent trace, exposed so callers can persist it even when
        # run() raises (e.g. on an approval denial).
        self.last_trace: Optional[RunTrace] = None

    # ------------------------------------------------------------------ #
    def run(self) -> RunTrace:
        """Run the whole workflow and return its correction-log trace."""

        trace = RunTrace(
            workflow=self.workflow.name,
            mode=self.workflow.mode,
            provider=self.provider.name,
        )
        self.last_trace = trace

        order = self.workflow.execution_order()
        self._log(
            f"Running '{self.workflow.name}' "
            f"({self.workflow.mode} mode, {len(order)} tasks) "
            f"via provider '{self.provider.name}'."
        )

        try:
            for position, task in enumerate(order, start=1):
                agent = self.workflow.agents[task.agent]
                self._log(
                    f"\n[{position}/{len(order)}] Task '{task.name}' "
                    f"-> agent '{agent.name}' ({agent.role})"
                )

                # --- human-in-the-loop checkpoint -------------------- #
                approved: Optional[bool] = None
                if task.require_approval:
                    if self.auto_approve:
                        approved = True
                        self._log("    [checkpoint] auto-approved (--yes).")
                    else:
                        approved = self._approval_fn(task, agent)
                    if not approved:
                        self._record_denied(trace, task, agent)
                        self._log("    [checkpoint] DENIED -- aborting run.")
                        trace.finish("aborted")
                        raise ApprovalDenied(task)

                # --- execute ----------------------------------------- #
                self._execute_task(trace, task, agent, approved)

            trace.finish("completed")
            trace.context = dict(self.context)
            self._log("\nWorkflow completed successfully.")
            return trace
        except ApprovalDenied:
            trace.context = dict(self.context)
            raise
        except Exception as exc:  # pragma: no cover - defensive
            trace.finish("error")
            trace.context = dict(self.context)
            self._log(f"\nWorkflow failed: {exc}")
            raise

    # ------------------------------------------------------------------ #
    def _execute_task(
        self,
        trace: RunTrace,
        task: Task,
        agent: Agent,
        approved: Optional[bool],
    ) -> None:
        system = self._build_system_prompt(agent)
        prompt = self._render(task.description)

        started = time.time()
        started_iso = _iso(started)
        status = "completed"
        error: Optional[str] = None
        output = ""
        try:
            output = self.provider.complete(
                CompletionRequest(system=system, prompt=prompt, model=agent.model)
            )
        except Exception as exc:
            status = "error"
            error = str(exc)
            raise
        finally:
            finished = time.time()
            record = StepRecord(
                task=task.name,
                agent=agent.name,
                provider=self.provider.name,
                model=agent.model,
                status=status,
                description=task.description,
                rendered_prompt=prompt,
                output=output,
                output_var=task.output_var,
                started_at=started_iso,
                finished_at=_iso(finished),
                duration_ms=round((finished - started) * 1000, 3),
                require_approval=task.require_approval,
                approved=approved,
                error=error,
            )
            trace.add_step(record)

        if task.output_var:
            self.context[task.output_var] = output
        self._log(f"    done in {record.duration_ms:.1f} ms")

    # ------------------------------------------------------------------ #
    def _record_denied(self, trace: RunTrace, task: Task, agent: Agent) -> None:
        now = time.time()
        trace.add_step(
            StepRecord(
                task=task.name,
                agent=agent.name,
                provider=self.provider.name,
                model=agent.model,
                status="denied",
                description=task.description,
                rendered_prompt=self._render(task.description),
                output="",
                output_var=task.output_var,
                started_at=_iso(now),
                finished_at=_iso(now),
                duration_ms=0.0,
                require_approval=True,
                approved=False,
            )
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_system_prompt(agent: Agent) -> str:
        """Compose the agent persona into a system prompt."""

        parts = [
            f"You are an autonomous agent in a multi-agent workflow.",
            f"Name: {agent.name}",
            f"Role: {agent.role}",
        ]
        if agent.goal:
            parts.append(f"Goal: {agent.goal}")
        if agent.backstory:
            parts.append(f"Backstory: {agent.backstory}")
        parts.append(
            "Produce a focused, high-quality result for the task you are given."
        )
        return "\n".join(parts)

    def _render(self, template: str) -> str:
        """Substitute ``{{ var }}`` placeholders from the shared context.

        Unknown variables are left as a readable ``[missing: var]`` marker
        rather than raising, so a partially-wired workflow still runs and
        the gap is visible in the correction log.
        """

        def repl(match: "re.Match[str]") -> str:
            key = match.group(1)
            if key in self.context:
                return self.context[key]
            return f"[missing: {key}]"

        return _PLACEHOLDER.sub(repl, template)


def _iso(epoch: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
