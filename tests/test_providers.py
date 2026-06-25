"""Tests for the providers (mock determinism + factory)."""

from __future__ import annotations

import pytest

from agentforge.providers import (
    CompletionRequest,
    MockProvider,
    OpenAIProvider,
    get_provider,
)


def test_mock_provider_is_deterministic():
    p = MockProvider()
    req = CompletionRequest(
        system="Role: Writer\nGoal: write things", prompt="write a poem", model="mock"
    )
    assert p.complete(req) == p.complete(req)


def test_mock_provider_reflects_role_and_model():
    p = MockProvider()
    out = p.complete(
        CompletionRequest(
            system="Role: Editor\nGoal: polish", prompt="edit this", model="mock"
        )
    )
    assert "Editor" in out
    assert "model=mock" in out


def test_get_provider_factory():
    assert isinstance(get_provider("mock"), MockProvider)
    assert isinstance(get_provider("openai"), OpenAIProvider)
    with pytest.raises(ValueError):
        get_provider("nope")


def test_openai_provider_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = OpenAIProvider()
    with pytest.raises(RuntimeError, match="Missing API key"):
        p.complete(CompletionRequest(system="s", prompt="p", model="gpt-4o-mini"))


def test_openai_provider_construction_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = OpenAIProvider(base_url="http://localhost:1234/v1")
    assert p.base_url == "http://localhost:1234/v1"
