"""Stable interfaces that keep agents and providers replaceable."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .models import MemoryPacket, TaskRecord


class MemoryLLM(Protocol):
    """Provider-neutral model used only by the Long Memory arm."""

    model_id: str

    def complete_json(self, *, operation: str, prompt: str) -> Mapping[str, Any]:
        """Return a JSON object for reflection, reranking, or adaptation."""


class Embedder(Protocol):
    """Code-aware embedder used only by the Long Memory arm."""

    model_id: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one finite vector per input text."""


class AgentAdapter(Protocol):
    """Adapter boundary for mini-SWE-agent and later agent implementations."""

    agent_id: str

    def build_invocation(
        self, *, task: TaskRecord, memory: MemoryPacket
    ) -> Mapping[str, Any]:
        """Build an agent-specific invocation without executing it."""
