"""Domain records shared by the SWE-bench long-memory protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class MemoryMode(str, Enum):
    """The only supported experiment arms.

    ``DISABLED`` is deliberately stronger than an empty database: it must not
    instantiate a store, embedder, or memory LLM.
    """

    DISABLED = "disabled"
    ENABLED = "enabled"


class LessonKind(str, Enum):
    """Where a lesson's cause lies, decided by a separate routing call.

    ``PROJECT`` lessons (environment, tooling, how to run tests, repository-wide
    conventions) go to the common pool and reach every later task of the same
    repository; ``BUG`` lessons are retrieved by issue similarity.
    """

    PROJECT = "project"
    BUG = "bug"


@dataclass(frozen=True)
class TaskRecord:
    """Evidence produced during one agent episode.

    ``facts`` are extracted by code from the agent-visible trajectory (errors,
    commands, the agent's own test runs, changed files); they are neutral evidence,
    not conclusions.  ``harness_result`` is stored for reporting only.  Nothing in
    memory may receive it, because an official SWE-bench evaluation can reveal
    hidden-test information.
    """

    instance_id: str
    issue_text: str
    sequence_index: int
    order_id: str
    repo: str = ""
    trajectory_text: str = ""
    patch_text: str = ""
    facts: Mapping[str, Any] = field(default_factory=dict)
    run_config: Mapping[str, Any] = field(default_factory=dict)
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    harness_result: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReflectionCase:
    """One evidence-grounded, cross-task lesson admitted to memory."""

    case_id: str
    trajectory_id: str
    instance_id: str
    repo: str
    kind: LessonKind
    trigger: str
    evidence: str
    hypothesized_cause: str
    preventative_action: str
    limitations: str
    cause_confidence: str
    embedding_text: str
    severity: float = 0.0
    group_id: str = ""
    scope_reason: str = ""
    investigation: Mapping[str, Any] = field(default_factory=dict)
    prompt_version: str = ""
    source_refs: Sequence[str] = ()

    def render(self, max_chars: int = 900) -> str:
        text = (
            f"Trigger: {self.trigger}\n"
            f"Evidence: {self.evidence}\n"
            f"Likely cause: {self.hypothesized_cause}\n"
            f"Reusable action: {self.preventative_action}\n"
            f"Limitations: {self.limitations}"
        )
        return text[:max_chars]


@dataclass(frozen=True)
class MemoryPacket:
    """Bounded guidance injected into an agent's current-task prompt."""

    task_id: str
    guidance: str
    case_ids: Sequence[str]
    retrieval_event_id: str | None = None
    stats: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, task_id: str) -> "MemoryPacket":
        return cls(task_id=task_id, guidance="", case_ids=())

    def render(self) -> str:
        # The issue stays the task; lessons come after it and say so at both ends
        # (requirement first, reminder last: the first/last positions are used most).
        if not self.guidance.strip():
            return ""
        return (
            "## Lessons from earlier tasks in this repository\n"
            "The PR description above is the whole task. These notes are fallible lessons "
            "from earlier tasks, not a checklist; apply one only where it fits, and verify it "
            "against the current repository.\n\n"
            f"{self.guidance.strip()}\n\n"
            "Fix the issue as described above; the notes only help with how."
        )
