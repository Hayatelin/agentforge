"""AgentForge: a lightweight multi-agent workflow orchestration CLI.

AgentForge lets you describe a small team of LLM-powered agents and the
tasks they collaborate on inside a single YAML file, then runs that
workflow as a dependency-aware DAG. It ships with a deterministic "mock"
provider so the whole thing runs end-to-end with no API key and no
internet access, and an OpenAI-compatible provider for real models.

Public API is intentionally tiny -- most users interact through the CLI
(``python -m agentforge ...``).
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "VictorLin"
__license__ = "MIT"

from .workflow import Agent, Task, Workflow, WorkflowError, load_workflow
from .providers import Provider, MockProvider, OpenAIProvider, get_provider
from .orchestrator import Orchestrator, ApprovalDenied
from .trace import RunTrace, StepRecord

__all__ = [
    "__version__",
    "Agent",
    "Task",
    "Workflow",
    "WorkflowError",
    "load_workflow",
    "Provider",
    "MockProvider",
    "OpenAIProvider",
    "get_provider",
    "Orchestrator",
    "ApprovalDenied",
    "RunTrace",
    "StepRecord",
]
