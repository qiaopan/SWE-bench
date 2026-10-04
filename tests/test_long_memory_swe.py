"""Fast protocol tests; they use fakes and never require Azure, Docker, or Jina."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from swebench.long_memory import LongMemoryRuntime, MemoryConfig, MemoryMode, TaskRecord
from swebench.long_memory.adapters import AzureOpenAIMemoryLLM, MiniSWEAgentAdapter
from swebench.long_memory.run_mini import MEMORY_BLOCK, compact_trajectory, memory_template


class FakeEmbedder:
    model_id = "fake-code-embedder-v1"
    dimension = 2

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] if "parser" in text.lower() else [0.0, 1.0] for text in texts]


class FakeMemoryLLM:
    model_id = "fake-memory-llm-v1"

    def __init__(self, rerank_response=None) -> None:
        self.calls = []
        self.rerank_response = rerank_response

    def complete_json(self, *, operation, prompt):
        self.calls.append(operation)
        if operation == "reflection":
            return {
                "admit": True, "outcome": "partial", "trigger": "Parser edge case",
                "evidence": "Agent test missed an empty input path.",
                "hypothesized_cause": "Only normal inputs were inspected.",
                "preventative_action": "Before editing parser code, search for empty and malformed input handling.",
                "limitations": "Verify the current repository API first.",
                "confidence": 0.9, "embedding_text": "parser empty malformed input checks",
            }
        if self.rerank_response is not None:
            return self.rerank_response
        # Skip the "..." placeholder of the prompt's JSON example.
        case_ids = [c for c in re.findall(r'"case_id": "([^"]+)"', prompt) if c != "..."]
        return {"items": [{"case_id": case_ids[0], "advice": "Check parser edge cases before patching."}]}


def task(instance_id: str, index: int, issue: str = "Fix parser error") -> TaskRecord:
    return TaskRecord(
        instance_id=instance_id, issue_text=issue, sequence_index=index, order_id="order-a",
        trajectory_text="looked at parser", patch_text="diff --git a/parser.py", self_test_log="one test passed",
        artifact_paths={"trajectory": f"artifacts/{instance_id}/trajectory.jsonl"},
    )


class LongMemoryProtocolTests(unittest.TestCase):
    def test_disabled_arm_touches_no_memory_dependency_or_database(self):
        runtime = LongMemoryRuntime(MemoryConfig(mode=MemoryMode.DISABLED))
        current = task("repo__one-1", 0)
        self.assertEqual(runtime.before_task(current).render(), "")
        self.assertIsNone(runtime.after_task(current))
        self.assertIsNone(runtime.store)

    def test_enabled_arm_reflects_then_retrieves_other_task(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "memory.sqlite"
            embedder, llm = FakeEmbedder(), FakeMemoryLLM()
            runtime = LongMemoryRuntime(
                MemoryConfig(mode=MemoryMode.ENABLED, database_path=db),
                embedder=embedder, memory_llm=llm,
            )
            case = runtime.after_task(task("repo__one-1", 0))
            self.assertIsNotNone(case)
            packet = runtime.before_task(task("repo__two-2", 1, "Parser rejects empty payload"))
            self.assertIn("Check parser edge cases", packet.render())
            self.assertEqual(list(packet.case_ids), [case.case_id])
            self.assertEqual(llm.calls, ["reflection", "rerank_rewrite"])
            self.assertTrue(db.exists())
            runtime.close()

    def test_uncited_or_unselected_guidance_is_never_injected(self):
        own_analysis = "Root cause is in Nested._deserialize; validate the type before load()."
        for response in (
            {"items": []},
            {"items": [{"case_id": "not-a-candidate", "advice": own_analysis}]},
            {"items": [{"advice": own_analysis}]},
            # v1-style payload: free text with no selected lesson.
            {"selected_case_ids": [], "guidance": own_analysis},
        ):
            with tempfile.TemporaryDirectory() as directory:
                runtime = LongMemoryRuntime(
                    MemoryConfig(mode=MemoryMode.ENABLED, database_path=Path(directory) / "memory.sqlite"),
                    embedder=FakeEmbedder(), memory_llm=FakeMemoryLLM(rerank_response=response),
                )
                runtime.after_task(task("repo__one-1", 0))
                packet = runtime.before_task(task("repo__two-2", 1, "Parser rejects empty payload"))
                self.assertEqual(packet.render(), "", response)
                self.assertEqual(list(packet.case_ids), [], response)
                runtime.close()

    def test_same_task_is_excluded_from_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = LongMemoryRuntime(
                MemoryConfig(mode=MemoryMode.ENABLED, database_path=Path(directory) / "memory.sqlite"),
                embedder=FakeEmbedder(), memory_llm=FakeMemoryLLM(),
            )
            first = task("repo__one-1", 0)
            runtime.after_task(first)
            self.assertEqual(runtime.before_task(first).render(), "")
            runtime.close()

    def test_agent_adapter_carries_memory_as_explicit_input(self):
        adapter = MiniSWEAgentAdapter()
        current = task("repo__one-1", 0)
        packet = LongMemoryRuntime(MemoryConfig(mode=MemoryMode.DISABLED)).before_task(current)
        invocation = adapter.build_invocation(task=current, memory=packet)
        self.assertEqual(invocation["agent_id"], "mini-swe-agent")
        self.assertEqual(invocation["additional_prompt"], "")

    def test_azure_adapter_requires_a_json_object(self):
        class Completion:
            class Choice:
                class Message:
                    content = '{"ok": true}'

                message = Message()

            choices = [Choice()]

        sent = {}

        class Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        sent.update(kwargs)
                        return Completion()

        result = AzureOpenAIMemoryLLM(client=Client(), deployment="swe-reflection-gpt5-mini").complete_json(
            operation="test", prompt="x"
        )
        self.assertEqual(result, {"ok": True})
        # gpt-5-mini rejects temperature=0; defaults must be left to the provider.
        self.assertNotIn("temperature", sent)
        self.assertNotIn("reasoning_effort", sent)

    def test_memory_template_adds_exactly_one_block_after_the_issue(self):
        official = "<pr_description>\n{{task}}\n</pr_description>\n\n<instructions>\nfix it\n</instructions>"
        changed = memory_template(official)
        # Removing the block restores the official prompt byte for byte.
        self.assertEqual(changed.replace(MEMORY_BLOCK, "", 1), official)
        self.assertLess(changed.index("</pr_description>"), changed.index("memory_guidance"))
        with self.assertRaises(ValueError):
            memory_template("<instructions>no issue marker</instructions>")

    def test_compact_trajectory_keeps_commands_and_clips_outputs(self):
        messages = [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "ISSUE PROMPT"},
            {"role": "assistant", "content": "Look around.",
             "tool_calls": [{"function": {"arguments": json.dumps({"command": "ls -la"})}}]},
            {"role": "tool", "content": "x" * 10_000},
            {"role": "exit", "content": "FULL PATCH", "extra": {"exit_status": "Submitted"}},
        ]
        text, stats = compact_trajectory(messages)
        self.assertNotIn("ISSUE PROMPT", text)
        self.assertNotIn("FULL PATCH", text)
        self.assertIn("COMMAND: ls -la", text)
        self.assertIn("EXIT: Submitted", text)
        self.assertIn("characters omitted", text)
        self.assertLess(len(text), 3_000)
        self.assertFalse(stats["truncated"])

    def test_compact_trajectory_reads_responses_api_turns(self):
        messages = [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "ISSUE PROMPT"},
            {"object": "response", "output": [
                {"type": "reasoning", "summary": [], "encrypted_content": "SECRET"},
                {"type": "message", "content": [{"type": "output_text", "text": "THOUGHT: inspect"}]},
                {"type": "function_call", "arguments": json.dumps({"command": "grep -rn foo"})},
            ]},
            {"type": "function_call_output", "output": "<returncode>0</returncode>"},
            {"role": "exit", "content": "FULL PATCH", "extra": {"exit_status": "Submitted"}},
        ]
        text, _ = compact_trajectory(messages)
        self.assertIn("THOUGHT: inspect", text)
        self.assertIn("COMMAND: grep -rn foo", text)
        self.assertIn("<returncode>0</returncode>", text)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("FULL PATCH", text)


if __name__ == "__main__":
    unittest.main()
