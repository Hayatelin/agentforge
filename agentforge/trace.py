"""Run tracing -- the "correction log".

Every executed step appends a :class:`StepRecord` to a :class:`RunTrace`.
The trace captures inputs, the resolved provider/model, the output, timing
and approval status. We frame this as a *correction log*: a structured,
replayable record that makes it easy to inspect what each agent did, spot
where a result went off the rails, and iteratively correct prompts,
dependencies or approvals on the next run.

The trace serialises to a single JSON file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StepRecord:
    """One executed (or skipped) task step in the run."""

    task: str
    agent: str
    provider: str
    model: str
    status: str  # "completed" | "skipped" | "denied" | "error"
    description: str
    rendered_prompt: str
    output: str
    output_var: Optional[str]
    started_at: str
    finished_at: str
    duration_ms: float
    require_approval: bool
    approved: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class RunTrace:
    """The full correction log for a single workflow run."""

    workflow: str
    mode: str
    provider: str
    started_at: str = field(default_factory=_now_iso)
    finished_at: Optional[str] = None
    status: str = "running"  # "running" | "completed" | "aborted" | "error"
    steps: List[StepRecord] = field(default_factory=list)
    context: Dict[str, str] = field(default_factory=dict)

    def add_step(self, record: StepRecord) -> None:
        self.steps.append(record)

    def finish(self, status: str) -> None:
        self.status = status
        self.finished_at = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    def save(self, path: str) -> str:
        """Write the trace to ``path`` as pretty-printed JSON. Returns path."""

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return path
