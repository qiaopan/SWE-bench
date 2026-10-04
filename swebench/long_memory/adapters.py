"""Optional adapters for the initial mini-SWE-agent / Jina experiment profile."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import MemoryPacket, TaskRecord


class AzureOpenAIMemoryLLM:
    """Thin Azure OpenAI adapter with an injected client.

    No sampling parameters are sent by default: gpt-5-mini rejects ``temperature=0``,
    and the formal profile uses provider defaults (reasoning effort ``medium``) to
    match the generator and the mini-SWE-agent baseline.  ``request_options`` is for
    explicitly recorded deviations such as ``reasoning_effort``.
    """

    def __init__(
        self, *, client: Any, deployment: str, request_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.model_id = deployment
        self.request_options = dict(request_options or {})
        # One entry per call, so a runner can report memory-side token use and cost.
        self.usage_log: list[dict[str, Any]] = []

    def complete_json(self, *, operation: str, prompt: str) -> Mapping[str, Any]:
        import json

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "Return one valid JSON object and no Markdown."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            **self.request_options,
        )
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        self.usage_log.append({
            "operation": operation,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
        })
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"Azure OpenAI returned empty content for {operation}")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise RuntimeError(f"Azure OpenAI returned non-object JSON for {operation}")
        return value


class MiniSWEAgentAdapter:
    """Builds explicit invocation metadata for a mini-SWE-agent episode.

    Execution remains owned by mini-SWE-agent.  Its exact configuration surface is
    versioned upstream, so this adapter records the prompt addition instead of
    silently patching a third-party prompt at import time.
    """

    agent_id = "mini-swe-agent"

    def build_invocation(self, *, task: TaskRecord, memory: MemoryPacket) -> Mapping[str, Any]:
        return {
            "instance_id": task.instance_id,
            "agent_id": self.agent_id,
            "issue": task.issue_text,
            "additional_prompt": memory.render(),
            "memory_case_ids": list(memory.case_ids),
            "memory_retrieval_event_id": memory.retrieval_event_id,
        }


class JinaCodeEmbedder:
    """Local-only adapter for jinaai/jina-embeddings-v2-base-code.

    The Jina model's tokenizer base configuration lives in a companion snapshot.
    Both paths are explicit, pinned inputs so formal runs never fetch or replace a
    model in the background.  Heavy ML imports occur lazily on first embedding.
    """

    model_id = "jinaai/jina-embeddings-v2-base-code@516f4baf13dec4ddddda8631e019b5737c8bc250"
    dimension = 768

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    @contextmanager
    def _tokenizer_redirect(self, transformers: Any):
        original = transformers.AutoTokenizer.from_pretrained

        def redirected(name: Any, *args: Any, **kwargs: Any) -> Any:
            if str(name) == str(self.model_path):
                return original(str(self.tokenizer_path), *args, **kwargs)
            return original(name, *args, **kwargs)

        transformers.AutoTokenizer.from_pretrained = redirected
        try:
            yield
        finally:
            transformers.AutoTokenizer.from_pretrained = original

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists() or not self.tokenizer_path.exists():
            raise FileNotFoundError("local Jina model/tokenizer snapshots are required; no download is attempted")
        try:
            import torch
            import transformers
            from transformers import AutoModel, BertTokenizerFast
        except ImportError as error:
            raise RuntimeError("install the documented local embedding environment before using JinaCodeEmbedder") from error
        self._tokenizer = BertTokenizerFast.from_pretrained(str(self.tokenizer_path), local_files_only=True)
        with self._tokenizer_redirect(transformers):
            self._model = AutoModel.from_pretrained(str(self.model_path), trust_remote_code=True, local_files_only=True)
        self._model.eval()
        self._torch = torch

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self._load()
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        with self._torch.no_grad():
            encoded = self._tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
            hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            vectors = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        result = vectors.cpu().tolist()
        if any(len(vector) != self.dimension or not all(math.isfinite(value) for value in vector) for vector in result):
            raise RuntimeError("Jina embedder returned invalid vectors")
        return result
