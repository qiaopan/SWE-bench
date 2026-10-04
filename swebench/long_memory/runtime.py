"""Continual Long Memory lifecycle independent of a specific SWE agent."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import MemoryMode, MemoryPacket, ReflectionCase, TaskRecord
from .prompts import RERANK_PROMPT_VERSION, reflection_prompt, rerank_rewrite_prompt
from .protocols import Embedder, MemoryLLM
from .store import MemoryStore


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


@dataclass(frozen=True)
class MemoryConfig:
    mode: MemoryMode
    database_path: str | Path | None = None
    max_candidates: int = 8
    max_guidance_chars: int = 1800
    minimum_confidence: float = 0.55

    def __post_init__(self) -> None:
        if self.mode is MemoryMode.ENABLED and not self.database_path:
            raise ValueError("enabled Long Memory requires a dedicated database path")
        if self.mode is MemoryMode.DISABLED and self.database_path:
            raise ValueError("disabled No Memory must not receive a database path")


class LongMemoryRuntime:
    """Retrieves before an episode and reflects only after it completes.

    The official harness result is never passed to ``after_task``.  Callers may
    report it separately after an experiment, but cannot feed it back into memory.
    """

    def __init__(
        self, config: MemoryConfig, *, embedder: Embedder | None = None,
        memory_llm: MemoryLLM | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.memory_llm = memory_llm
        if config.mode is MemoryMode.DISABLED:
            if embedder is not None or memory_llm is not None:
                raise ValueError("disabled No Memory cannot receive memory dependencies")
            self.store: MemoryStore | None = None
            return
        if embedder is None or memory_llm is None:
            raise ValueError("enabled Long Memory requires embedder and memory LLM")
        self.store = MemoryStore(config.database_path)  # type: ignore[arg-type]

    def close(self) -> None:
        if self.store is not None:
            self.store.close()

    def before_task(self, task: TaskRecord) -> MemoryPacket:
        if self.config.mode is MemoryMode.DISABLED:
            return MemoryPacket.empty(task.instance_id)
        assert self.store is not None and self.embedder is not None and self.memory_llm is not None
        query = self.embedder.embed([task.issue_text])[0]
        scored = [
            (_cosine(query, vector), case)
            for case, vector in self.store.candidate_cases(
                task=task, embedding_model=self.embedder.model_id, dimension=self.embedder.dimension
            )
        ]
        candidates = [case for _, case in sorted(scored, key=lambda item: item[0], reverse=True)[:self.config.max_candidates]]
        if not candidates:
            return MemoryPacket.empty(task.instance_id)
        response = self.memory_llm.complete_json(
            operation="rerank_rewrite",
            prompt=rerank_rewrite_prompt(
                task=task, candidates=candidates,
                max_guidance_chars=self.config.max_guidance_chars,
            ),
        )
        # Only advice that cites a retrieved candidate is injected.  Uncited text (e.g. the
        # memory model's own analysis of the current issue) is dropped, and no selected
        # lesson means no guidance at all: memory must not become a second solver.
        candidate_ids = {case.case_id for case in candidates}
        lines: list[str] = []
        selected: list[str] = []
        for item in response.get("items") or []:
            if not isinstance(item, Mapping):
                continue
            case_id, advice = str(item.get("case_id", "")), str(item.get("advice", "")).strip()
            line = f"- {advice}"
            if case_id not in candidate_ids or not advice:
                continue
            if len("\n".join([*lines, line])) > self.config.max_guidance_chars:
                break
            lines.append(line)
            if case_id not in selected:
                selected.append(case_id)
        guidance = "\n".join(lines)
        event_id = self.store.record_retrieval(
            task=task, candidate_case_ids=[case.case_id for case in candidates],
            selected_case_ids=selected, guidance=guidance, prompt_version=RERANK_PROMPT_VERSION,
        )
        return MemoryPacket(task_id=task.instance_id, guidance=guidance, case_ids=selected, retrieval_event_id=event_id)

    def after_task(self, task: TaskRecord) -> ReflectionCase | None:
        if self.config.mode is MemoryMode.DISABLED:
            return None
        assert self.store is not None and self.embedder is not None and self.memory_llm is not None
        trajectory_id = self.store.record_trajectory(task)
        response = self.memory_llm.complete_json(operation="reflection", prompt=reflection_prompt(task))
        if response.get("admit") is not True:
            return None
        confidence = float(response.get("confidence", 0.0))
        required = ("outcome", "trigger", "evidence", "hypothesized_cause", "preventative_action", "limitations", "embedding_text")
        if confidence < self.config.minimum_confidence or any(not str(response.get(key, "")).strip() for key in required):
            return None
        case = ReflectionCase(
            case_id=str(uuid.uuid4()), trajectory_id=trajectory_id, instance_id=task.instance_id,
            outcome=str(response["outcome"]), trigger=str(response["trigger"]),
            evidence=str(response["evidence"]), hypothesized_cause=str(response["hypothesized_cause"]),
            preventative_action=str(response["preventative_action"]), limitations=str(response["limitations"]),
            confidence=confidence, embedding_text=str(response["embedding_text"]),
            source_refs=tuple(task.artifact_paths.values()),
        )
        vector = self.embedder.embed([case.embedding_text])[0]
        self.store.admit_case(case, embedding_model=self.embedder.model_id, vector=vector)
        return case
