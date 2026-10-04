"""Versioned memory prompts with explicit benchmark-leakage boundaries.

Carried over from the WebGen phase: a step-by-step investigation before any lesson
(scientific debugging / differential diagnosis: list candidate causes and rule them
out with evidence), cause confidence that must match the evidence, a separate scope
routing call, root-cause grouping for the common pool, and guidance whose every item
cites a stored lesson.  The prompts teach how to investigate; they never state a
conclusion about a particular repository.  Official harness results are never input.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .models import ReflectionCase, TaskRecord

REFLECTION_INSTRUCTIONS = """You are recording reusable lessons after one SWE-bench agent run.
Use ONLY the JSON evidence below: the issue, the agent-visible trajectory, its patch, and
code_facts (extracted by code from the trajectory; neutral facts, not conclusions).
Do not infer hidden-test outcomes, a gold patch, or official harness results.

First fill "investigation", in this order:
1. issue: the expected and the observed behaviour described in the issue.
2. reproduced: did the agent reproduce the problem before editing? Cite the command/output.
3. errors: every error the agent hit. For each, say whether it is an ENVIRONMENT/TOOLING
   problem (missing module, wrong interpreter or environment, command not found, how tests
   or scripts must be run) or a CODE problem. Environment and tooling problems are not
   defects of the repository code; trace them to their actual cause.
4. code_path: which files/functions the agent changed, and whether that covers the code
   path the issue describes.
5. self_test: what the agent's own checks showed before and after its change.
6. candidates: 2-3 candidate causes for any failure or wasted effort; rule each in or out
   with evidence from the trajectory or code_facts.
7. conclusion: the cause(s) that survived step 6.
8. fix_check: would a later agent that applied only the lesson avoid the problem? If not,
   rewrite the lesson or mark it uncertain.

Then return "lessons": 0-3 lessons, one per distinct cause. Never mix an environment or
tooling lesson with a lesson about the bug's code. Each lesson has: trigger, evidence,
hypothesized_cause, preventative_action (specific, conditional, <= 400 characters, usable
on a later task), limitations, cause_confidence, embedding_text.
cause_confidence is "established" only when the cause is directly shown by evidence in the
trajectory or code_facts; a guess ("appears", "likely", "may") is "uncertain".
Return JSON: {"investigation": {...}, "lessons": [...]}; "lessons" may be empty."""

SCOPE_INSTRUCTIONS = """Decide the scope of one lesson learned in repository {repo}.
"project": the cause lies in the environment, tooling, how to run or import code or tests,
or a repository-wide structure or convention, so the same cause would affect other tasks in
this repository whatever their bug is.
"bug": the cause is specific to this issue's code path or feature. Missing or wrong code for
one feature is "bug" even when it lives in a widely shared file.
Fact from code (evidence, not an answer): evidence_cites_environment_error = {env_fact}.
Return JSON: {{"scope": "project" or "bug", "reason": "..."}}."""

GROUP_INSTRUCTIONS = """A new repository-level lesson, and existing groups of repository-level
lessons (one representative each). Does the new lesson have the SAME root cause as one group,
i.e. the same underlying environment/tooling/convention problem, not merely similar wording or
the same file? Return JSON: {"group_id": "<an existing group id>" or "new", "reason": "..."}."""

SELECT_INSTRUCTIONS = """You are selecting earlier bug lessons for a SWE-bench coding agent.
For EVERY candidate give a judgment: "hard_match" (its trigger clearly applies to the current
issue), "soft_transfer" (only its general principle applies), or "reject"; with a reason.
For hard_match/soft_transfer also give "advice": that ONE lesson restated so it is usable on
the current issue. Keep the element the lesson says was missing or wrong, and keep its
qualifiers; never turn a conditional statement into an absolute one. Carry over only what the
lesson says: do not diagnose the current issue or propose a fix for it.
Do not use any hidden test result or gold patch.
Return JSON: {"judgments": [{"case_id": "...", "judgment": "...", "reason": "...", "advice": "..."}]}."""

COMPOSE_INSTRUCTIONS = """You are writing the final notes for a SWE-bench coding agent from
lessons of earlier tasks in the same repository.
Inputs: bug advice already selected for this issue, and common-pool lessons (repository-level,
ordered by priority = occurrences x severity).
Rules:
- Rewrite each pool lesson so it holds for any task in this repository: keep the repository-
  wide part (environment, tooling, how to run tests, conventions), drop details of the bug it
  came from. Keep the first two pool lessons.
- Merge items that say the same thing. At most {max_items} items, each one imperative sentence
  of at most {item_chars} characters, total at most {budget} characters. Shorten wording
  rather than dropping an item.
- Every item cites the case_id(s) it comes from. Add nothing that is not in the inputs: no
  diagnosis of the current issue, no fix for it.
Return JSON: {{"items": [{{"case_ids": ["..."], "text": "..."}}]}}."""

SHORTEN_INSTRUCTIONS = """These notes exceed the length budget of {budget} characters. Shorten
the wording of each item (keep every item, its case_ids and its meaning); each item at most
{item_chars} characters. Return JSON: {{"items": [{{"case_ids": ["..."], "text": "..."}}]}}."""

PROMPT_VERSION = "swe-memory-" + hashlib.sha256("\n\x00\n".join([
    REFLECTION_INSTRUCTIONS, SCOPE_INSTRUCTIONS, GROUP_INSTRUCTIONS,
    SELECT_INSTRUCTIONS, COMPOSE_INSTRUCTIONS, SHORTEN_INSTRUCTIONS,
]).encode()).hexdigest()[:12]
# Kept for older imports and run metadata; every memory prompt shares one fingerprint.
REFLECTION_PROMPT_VERSION = RERANK_PROMPT_VERSION = PROMPT_VERSION


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def reflection_prompt(task: TaskRecord) -> str:
    evidence = {
        "issue": task.issue_text,
        "agent_trajectory": task.trajectory_text,
        "patch": task.patch_text,
        "code_facts": dict(task.facts),
    }
    return f"{REFLECTION_INSTRUCTIONS}\nPrompt version: {PROMPT_VERSION}\n\nEvidence:\n{_dump(evidence)}"


def scope_prompt(*, repo: str, lesson: Mapping[str, Any], env_fact: bool) -> str:
    return (SCOPE_INSTRUCTIONS.format(repo=repo or "unknown", env_fact=str(env_fact).lower())
            + f"\nPrompt version: {PROMPT_VERSION}\n\nLesson:\n{_dump(dict(lesson))}")


def group_prompt(*, lesson: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> str:
    return (f"{GROUP_INSTRUCTIONS}\nPrompt version: {PROMPT_VERSION}\n\nNew lesson:\n{_dump(dict(lesson))}"
            f"\n\nExisting groups:\n{_dump(list(groups))}")


def select_prompt(*, task: TaskRecord, candidates: Sequence[ReflectionCase]) -> str:
    data = [{"case_id": item.case_id, "lesson": item.render()} for item in candidates]
    return (f"{SELECT_INSTRUCTIONS}\nPrompt version: {PROMPT_VERSION}\n\nCurrent issue:\n{task.issue_text}"
            f"\n\nCandidate lessons:\n{_dump(data)}")


def compose_prompt(
    *, bug_items: Sequence[Mapping[str, Any]], pool: Sequence[Mapping[str, Any]],
    max_items: int, item_chars: int, budget: int,
) -> str:
    return (COMPOSE_INSTRUCTIONS.format(max_items=max_items, item_chars=item_chars, budget=budget)
            + f"\nPrompt version: {PROMPT_VERSION}\n\nBug advice:\n{_dump(list(bug_items))}"
            + f"\n\nCommon-pool lessons:\n{_dump(list(pool))}")


def shorten_prompt(*, items: Sequence[Mapping[str, Any]], item_chars: int, budget: int) -> str:
    return (SHORTEN_INSTRUCTIONS.format(budget=budget, item_chars=item_chars)
            + f"\nPrompt version: {PROMPT_VERSION}\n\nItems:\n{_dump(list(items))}")
