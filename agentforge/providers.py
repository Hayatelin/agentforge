"""LLM providers.

AgentForge talks to language models through a tiny ``Provider`` protocol.
Two implementations ship in the box:

* :class:`MockProvider` -- fully offline, deterministic, no API key. It
  produces structured, role-aware text so demos and tests look coherent.
* :class:`OpenAIProvider` -- talks to any OpenAI-compatible chat
  completions endpoint (OpenAI, Azure, Ollama, vLLM, LM Studio, ...) using
  only the standard library (``urllib``). Configurable via ``base_url``,
  ``model`` and an API key read from an environment variable.
"""

from __future__ import annotations

import json
import os
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Protocol


@dataclass
class CompletionRequest:
    """Everything a provider needs to generate one completion."""

    system: str
    prompt: str
    model: str


class Provider(Protocol):
    """The minimal interface every provider must implement."""

    name: str

    def complete(self, request: CompletionRequest) -> str:
        """Return the model's text response for ``request``."""
        ...


# --------------------------------------------------------------------------- #
# Mock provider
# --------------------------------------------------------------------------- #
class MockProvider:
    """A deterministic, offline provider.

    The mock does not call any network service. Instead it synthesises a
    plausible, role-aware response from the system prompt and task so that
    multi-agent demos read coherently and tests are reproducible. It is the
    default provider, which means AgentForge works out of the box.
    """

    name = "mock"

    def complete(self, request: CompletionRequest) -> str:
        role = _extract_field(request.system, "Role") or "Agent"
        goal = _extract_field(request.system, "Goal")
        task = request.prompt.strip()

        # A compact, structured "answer" that reflects the agent's role and
        # the task it was given. Deterministic for a given input.
        bullet = textwrap.shorten(task.replace("\n", " "), width=160) or "the task"
        lines = [
            f"[{role} | model={request.model}]",
            f"Objective: {goal or 'complete the assigned task'}.",
            "",
            f"Working on: {bullet}",
            "",
            "Result:",
            f"  - Analysed the request and the upstream context provided.",
            f"  - Produced a {role.lower()}-quality draft addressing the goal.",
            f"  - Key takeaway: {_summarise(task)}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# OpenAI-compatible provider
# --------------------------------------------------------------------------- #
class OpenAIProvider:
    """Provider for any OpenAI-compatible chat completions endpoint.

    Args:
        base_url: API root, default ``https://api.openai.com/v1``.
        default_model: Model used when an agent does not specify one.
        api_key_env: Environment variable to read the API key from.
        timeout: Network timeout in seconds.

    The API key is read lazily at call time so that merely *constructing*
    the provider (e.g. during ``validate``) never requires a key.
    """

    name = "openai"

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.default_model = os.environ.get("OPENAI_MODEL", default_model)
        self.api_key_env = api_key_env
        self.timeout = timeout

    def complete(self, request: CompletionRequest) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set the {self.api_key_env} environment "
                "variable, or use --provider mock to run offline."
            )

        model = request.model or self.default_model
        # "mock" is not a real OpenAI model; fall back to the default.
        if model == "mock":
            model = self.default_model

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": 0.7,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI API HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network
            raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover
            raise RuntimeError(
                f"Unexpected OpenAI response shape: {body!r}"
            ) from exc


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_provider(name: str, **kwargs) -> Provider:
    """Return a provider instance by name (``"mock"`` or ``"openai"``)."""

    key = (name or "mock").lower()
    if key == "mock":
        return MockProvider()
    if key == "openai":
        return OpenAIProvider(**kwargs)
    raise ValueError(
        f"Unknown provider {name!r}; choose 'mock' or 'openai'."
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_field(system: str, field: str) -> str:
    """Pull a ``Field: value`` line out of a system prompt, if present."""

    prefix = f"{field}:"
    for line in system.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip()
    return ""


def _summarise(text: str) -> str:
    """Produce a one-line, deterministic 'summary' of ``text``."""

    words = [w for w in text.replace("\n", " ").split(" ") if w]
    if not words:
        return "task completed as specified."
    head = " ".join(words[:12])
    return head + ("..." if len(words) > 12 else ".")
