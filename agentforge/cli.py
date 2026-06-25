"""Command-line interface for AgentForge.

Implements three subcommands via ``argparse``:

* ``init``      -- scaffold an example ``workflow.yaml``.
* ``validate``  -- load and validate a workflow file, reporting problems.
* ``run``       -- execute a workflow, with optional auto-approval and a
                   choice of provider, writing a JSON correction log.

Runnable as ``python -m agentforge ...`` or via the ``agentforge`` script.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__
from .orchestrator import ApprovalDenied, Orchestrator
from .providers import get_provider
from .workflow import Workflow, WorkflowError, load_workflow

# The example workflow scaffolded by `agentforge init`.
EXAMPLE_WORKFLOW = """\
# AgentForge example workflow
# A 3-agent content pipeline: researcher -> writer -> editor.
# Run it offline with:  python -m agentforge run workflow.yaml --yes --provider mock
name: Blog Post Pipeline
description: Research a topic, draft a blog post, then edit it for publication.
mode: dag  # "dag" honours depends_on; "sequential" runs in file order.

agents:
  - name: researcher
    role: Senior Research Analyst
    goal: Gather accurate, relevant facts and angles on the topic
    backstory: A meticulous analyst who loves primary sources and clear notes.
    model: mock

  - name: writer
    role: Content Writer
    goal: Turn research into an engaging, well-structured blog post
    backstory: A storyteller who makes complex ideas approachable.
    model: mock

  - name: editor
    role: Managing Editor
    goal: Polish the draft and ensure it is ready to publish
    backstory: A sharp editor with an eye for clarity, tone and accuracy.
    model: mock

tasks:
  - name: research
    description: >
      Research the topic "Practical uses of multi-agent AI workflows".
      Produce 5 key points and 2 surprising angles.
    agent: researcher
    output_var: research_notes

  - name: write
    description: >
      Using these research notes, write a 600-word blog post draft with a
      title, intro, three sections and a conclusion.

      Research notes:
      {{ research_notes }}
    agent: writer
    depends_on: [research]
    output_var: draft

  - name: edit
    description: >
      Edit the following draft for clarity, flow and tone. Fix any issues
      and return the final publication-ready version.

      Draft:
      {{ draft }}
    agent: editor
    depends_on: [write]
    output_var: final_post
    require_approval: true  # human-in-the-loop checkpoint before publishing
"""


def _cmd_init(args: argparse.Namespace) -> int:
    target = args.path
    if os.path.exists(target) and not args.force:
        print(
            f"Refusing to overwrite existing file: {target} "
            "(use --force to overwrite).",
            file=sys.stderr,
        )
        return 1
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(EXAMPLE_WORKFLOW)
    print(f"Scaffolded example workflow at {target}")
    print("Next steps:")
    print(f"  python -m agentforge validate {target}")
    print(f"  python -m agentforge run {target} --yes --provider mock")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        workflow: Workflow = load_workflow(args.path)
    except WorkflowError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    order = workflow.execution_order()
    print(f"OK: '{workflow.name}' is valid.")
    print(f"  mode    : {workflow.mode}")
    print(f"  agents  : {len(workflow.agents)} ({', '.join(workflow.agents)})")
    print(f"  tasks   : {len(workflow.tasks)}")
    print(f"  order   : {' -> '.join(t.name for t in order)}")
    approvals = [t.name for t in workflow.tasks if t.require_approval]
    if approvals:
        print(f"  approval: {', '.join(approvals)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        workflow = load_workflow(args.path)
    except WorkflowError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    provider = get_provider(args.provider)

    orchestrator = Orchestrator(
        workflow,
        provider,
        auto_approve=args.yes,
        log_fn=lambda msg: print(msg),
    )

    try:
        trace = orchestrator.run()
        exit_code = 0
    except ApprovalDenied as exc:
        # The orchestrator records the denial in last_trace before raising,
        # so we can still persist a useful correction log.
        trace = orchestrator.last_trace
        exit_code = 2
        print(f"\nRun aborted: approval denied for task '{exc.task.name}'.",
              file=sys.stderr)

    out_path = args.trace or _default_trace_path(args.path)
    if exit_code == 0 and trace is not None:
        _emit_results(workflow, trace)
    if trace is not None:
        trace.save(out_path)
        print(f"\nCorrection log written to {out_path}")
    return exit_code


def _emit_results(workflow: Workflow, trace) -> None:
    """Print a human-friendly summary of the final outputs."""

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for step in trace.steps:
        if step.status != "completed":
            continue
        print(f"\n--- {step.task} (agent: {step.agent}) ---")
        print(step.output)
    if trace.context:
        print("\n" + "-" * 60)
        print("Final context variables: " + ", ".join(trace.context))


def _default_trace_path(workflow_path: str) -> str:
    base = os.path.splitext(os.path.basename(workflow_path))[0]
    return f"{base}.trace.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentforge",
        description=(
            "AgentForge: a lightweight multi-agent workflow orchestration CLI."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"agentforge {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Scaffold an example workflow.yaml")
    p_init.add_argument(
        "path", nargs="?", default="workflow.yaml",
        help="Where to write the example (default: workflow.yaml)",
    )
    p_init.add_argument(
        "--force", action="store_true", help="Overwrite if the file exists"
    )
    p_init.set_defaults(func=_cmd_init)

    p_val = sub.add_parser("validate", help="Validate a workflow file")
    p_val.add_argument("path", help="Path to the workflow YAML file")
    p_val.set_defaults(func=_cmd_validate)

    p_run = sub.add_parser("run", help="Run a workflow")
    p_run.add_argument("path", help="Path to the workflow YAML file")
    p_run.add_argument(
        "--yes", action="store_true",
        help="Auto-approve all human-in-the-loop checkpoints (for CI/demos)",
    )
    p_run.add_argument(
        "--provider", choices=["mock", "openai"], default="mock",
        help="LLM provider to use (default: mock, fully offline)",
    )
    p_run.add_argument(
        "--trace", default=None,
        help="Path for the JSON correction log (default: <workflow>.trace.json)",
    )
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
