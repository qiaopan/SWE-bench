"""Sequential mini-SWE-agent v2 runner for the No Memory / Long Memory arms.

One command runs one arm over one batch of the frozen instance list.  Tasks run in
list order, one at a time, because a Long Memory task may read lessons written by
earlier tasks.  The same output directory (and, for Long Memory, the same database)
is reused across batches, so memory accumulates in the frozen draw order.

The agent uses mini-SWE-agent's official ``benchmarks/swebench.yaml`` templates, limits
and environment, with the model class and kwargs that the official gpt-5-mini baseline
actually ran (Responses API, reasoning effort and verbosity ``medium``).  A Long Memory
task with guidance gets one block inserted right after the issue; any task without
guidance (every No Memory task) gets the official prompt verbatim.
Official SWE-bench harness results are never read here.

Example:
    .venv/bin/python -m swebench.long_memory.run_mini --arm disabled --batch 1 \
        --ids-file config/swebench_lite_dev_groups.json \
        --parquet data/swebench_lite/dev-00000-of-00001.parquet \
        --output outputs/dev-lite-nomem
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import MemoryMode, TaskRecord
from .prompts import REFLECTION_PROMPT_VERSION, RERANK_PROMPT_VERSION
from .runtime import LongMemoryRuntime, MemoryConfig

MEMORY_MARKER = "</pr_description>\n"
MEMORY_BLOCK = "\n{{ memory_guidance }}\n"
TRAJECTORY_CHAR_LIMIT = 60_000
OBSERVATION_CHAR_LIMIT = 2_000
DOCKER_PLATFORM = "linux/amd64"
DOCKER_PULL_TIMEOUT = 1800
# Copied from the official gpt-5-mini v2.0.0 baseline trajectories
# (SWE-bench experiments, 20260217_mini-v2.0.0_gpt-5-mini): it ran the Responses API
# model class with these kwargs, not the chat-completions default of swebench.yaml.
BASELINE_MODEL_CLASS = "litellm_response"
BASELINE_MODEL_KWARGS = {
    "drop_params": True,
    "temperature": None,
    "parallel_tool_calls": True,
    "reasoning": {"effort": "medium"},
    "text": {"verbosity": "medium"},
}
BASELINE_PULL_TIMEOUT = 300


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_template(template: str) -> str:
    """Official instance template plus one memory block after the issue."""

    if template.count(MEMORY_MARKER) != 1:
        raise ValueError("official instance template no longer has exactly one </pr_description>")
    return template.replace(MEMORY_MARKER, MEMORY_MARKER + MEMORY_BLOCK, 1)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return f"{text[:head]}\n[... {len(text) - limit} characters omitted ...]\n{text[-tail:]}"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
    return "" if content is None else str(content)


def _command(arguments: Any) -> str:
    try:
        return str(json.loads(arguments).get("command", arguments))
    except (TypeError, ValueError, AttributeError):
        return str(arguments or "")


def compact_trajectory(messages: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Agent-visible trajectory for reflection: every thought and command, clipped outputs.

    The system prompt and the issue prompt are skipped (the issue is passed separately),
    and the final exit message keeps only its status because the patch is passed
    separately too.
    """

    parts = []
    for message in messages[2:]:
        role = message.get("role")
        if message.get("object") == "response":
            # Responses API turn: visible text and commands; encrypted reasoning is skipped.
            for item in message.get("output") or []:
                if item.get("type") == "message":
                    parts.append("ASSISTANT: " + _text(item.get("content")))
                elif item.get("type") == "function_call":
                    parts.append(f"COMMAND: {_command(item.get('arguments'))}")
        elif role == "assistant":
            commands = [f"\nCOMMAND: {_command((call.get('function') or {}).get('arguments'))}"
                        for call in message.get("tool_calls") or []]
            parts.append("ASSISTANT: " + _text(message.get("content")) + "".join(commands))
        elif role == "exit":
            parts.append(f"EXIT: {(message.get('extra') or {}).get('exit_status', '')}")
        else:
            # Command output: chat "tool" messages or Responses "function_call_output" items.
            output = message.get("content", message.get("output"))
            parts.append("OUTPUT: " + _clip(_text(output), OBSERVATION_CHAR_LIMIT))
    full = "\n\n".join(parts)
    text = _clip(full, TRAJECTORY_CHAR_LIMIT)
    return text, {"messages": len(messages), "chars_before_limit": len(full), "chars_kept": len(text),
                  "truncated": len(full) > TRAJECTORY_CHAR_LIMIT}


def load_batch(ids_file: Path, batch: int, parquet: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the batch's instances in frozen order, each with its global sequence index."""

    record = json.loads(ids_file.read_text())
    if sha256_file(parquet) != record["source_sha256"]:
        raise SystemExit(f"{parquet} does not match the pinned dataset file in {ids_file}")
    order = [iid for entry in record["batches"] for iid in entry["instance_ids"]]
    selected = next(entry["instance_ids"] for entry in record["batches"] if entry["batch"] == batch)

    import pyarrow.parquet as pq

    rows = {row["instance_id"]: row for row in pq.read_table(parquet).to_pylist()}
    instances = []
    for iid in selected:
        instance = dict(rows[iid])
        instance["sequence_index"] = order.index(iid)
        instances.append(instance)
    return instances, record


def build_base_config(registry: Path, generator_deployment: str) -> tuple[dict[str, Any], Path]:
    from minisweagent.config import builtin_config_dir, get_config_from_spec

    official = builtin_config_dir / "benchmarks" / "swebench.yaml"
    config = copy.deepcopy(get_config_from_spec(str(official)))
    config["model"]["model_name"] = f"azure/{generator_deployment}"
    config["model"]["model_class"] = BASELINE_MODEL_CLASS
    config["model"]["model_kwargs"] = copy.deepcopy(BASELINE_MODEL_KWARGS)
    config["model"]["litellm_model_registry"] = str(registry)
    config["environment"]["run_args"] = ["--rm", "--platform", DOCKER_PLATFORM]
    config["environment"]["pull_timeout"] = BASELINE_PULL_TIMEOUT
    return config, official


def run_instance(
    instance: Mapping[str, Any], config: dict[str, Any], output: Path, extra_vars: Mapping[str, str],
) -> dict[str, Any]:
    """Run one agent episode the way mini-SWE-agent's batch runner does."""

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import (
        get_sb_environment, get_swebench_docker_image_name, remove_from_preds_file, update_preds_file,
    )

    iid = instance["instance_id"]
    traj_path = output / iid / f"{iid}.traj.json"
    # Pull outside the episode: the container start has its own short timeout.
    subprocess.run(["docker", "pull", "--platform", DOCKER_PLATFORM, get_swebench_docker_image_name(dict(instance))],
                   check=True, capture_output=True, timeout=DOCKER_PULL_TIMEOUT)
    remove_from_preds_file(output / "preds.json", iid)
    traj_path.unlink(missing_ok=True)
    model = get_model(config=copy.deepcopy(config["model"]))
    agent = env = None
    exit_status, submission, extra = None, "", {}
    try:
        env = get_sb_environment(config, dict(instance))
        agent = DefaultAgent(model, env, **config["agent"])
        info = agent.run(instance["problem_statement"], **extra_vars)
        exit_status, submission = info.get("exit_status"), info.get("submission") or ""
    except Exception as error:
        exit_status, submission = type(error).__name__, ""
        extra = {"traceback": traceback.format_exc(), "exception_str": str(error)}
    finally:
        if agent is not None:
            agent.save(traj_path, {"info": {"exit_status": exit_status, "submission": submission, **extra},
                                   "instance_id": iid})
        update_preds_file(output / "preds.json", iid, model.config.model_name, submission)
        if env is not None:
            env.cleanup()
    return {"exit_status": exit_status, "submission": submission, "traj_path": traj_path, "error": extra}


def _memory_cost(usage: Sequence[Mapping[str, Any]], prices: Mapping[str, float]) -> float:
    return sum((u.get("prompt_tokens") or 0) * prices["input_cost_per_token"]
               + (u.get("completion_tokens") or 0) * prices["output_cost_per_token"] for u in usage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=[m.value for m in MemoryMode])
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default=Path("config/long_memory_swe.example.yaml"), type=Path)
    parser.add_argument("--registry", default=Path("config/litellm_model_registry.json"), type=Path)
    parser.add_argument("--env-file", default=Path(".env"), type=Path)
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N tasks of the batch; rerun later without it to continue")
    args = parser.parse_args()

    import yaml
    from dotenv import load_dotenv

    load_dotenv(args.env_file, override=False)
    for key in ("AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"):
        if not os.getenv(key):
            raise SystemExit(f"{key} is not set (expected in {args.env_file})")

    settings = yaml.safe_load(args.config.read_text())
    experiment, agent_settings, memory_settings = settings["experiment"], settings["agent"], settings["memory"]
    mode = MemoryMode(args.arm)
    instances, ids_record = load_batch(args.ids_file, args.batch, args.parquet)
    if args.limit is not None:
        instances = instances[:args.limit]
    order_id = f"{ids_record['dataset']}:{ids_record['split']}:seed{ids_record['seed']}"
    base_config, official_config = build_base_config(args.registry.resolve(), agent_settings["generator_deployment"])
    # Memory-side cost uses the memory deployment's own prices (it may differ from the generator).
    registry = json.loads(args.registry.read_text())
    memory_prices = registry[f"azure/{memory_settings['llm_deployment']}"]

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "arm": mode.value,
        "order_id": order_id,
        "ids_file": str(args.ids_file), "ids_file_sha256": sha256_file(args.ids_file),
        "dataset": ids_record["dataset"], "split": ids_record["split"], "revision": ids_record["revision"],
        "parquet_sha256": ids_record["source_sha256"],
        "mini_swe_agent": importlib.metadata.version("mini-swe-agent"),
        "litellm": importlib.metadata.version("litellm"),
        "official_config_sha256": sha256_file(official_config),
        "generator_model": base_config["model"]["model_name"],
        "generator_model_class": base_config["model"]["model_class"],
        "generator_model_kwargs": base_config["model"]["model_kwargs"],
        "registry_sha256": sha256_file(args.registry),
        "step_limit": base_config["agent"]["step_limit"], "cost_limit": base_config["agent"]["cost_limit"],
        "command_timeout": base_config["environment"]["timeout"],
        "docker_platform": DOCKER_PLATFORM, "host": f"{platform.system()} {platform.machine()}",
        "memory": None if mode is MemoryMode.DISABLED else {
            "deployment": memory_settings["llm_deployment"], "embedder": memory_settings["embedder"],
            "embedder_revision": memory_settings["embedder_revision"],
            "max_candidates": experiment["max_candidates"], "max_guidance_chars": experiment["max_guidance_chars"],
            "minimum_confidence": experiment["minimum_confidence"],
            "reflection_prompt": REFLECTION_PROMPT_VERSION, "rerank_prompt": RERANK_PROMPT_VERSION,
            "trajectory_char_limit": TRAJECTORY_CHAR_LIMIT, "observation_char_limit": OBSERVATION_CHAR_LIMIT,
        },
    }
    metadata_path = args.output / "run-metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["settings"] != metadata:
            raise SystemExit(f"{metadata_path} was written with different settings; use a new output directory")
        previous["batches_started"].append({"batch": args.batch, "at": datetime.datetime.now().isoformat(timespec="seconds")})
        metadata_path.write_text(json.dumps(previous, indent=2))
    else:
        metadata_path.write_text(json.dumps({"settings": metadata, "batches_started": [
            {"batch": args.batch, "at": datetime.datetime.now().isoformat(timespec="seconds")}]}, indent=2))

    if mode is MemoryMode.DISABLED:
        runtime = LongMemoryRuntime(MemoryConfig(mode=mode))
        memory_llm = None
    else:
        from openai import AzureOpenAI

        from .adapters import AzureOpenAIMemoryLLM, JinaCodeEmbedder

        memory_llm = AzureOpenAIMemoryLLM(
            client=AzureOpenAI(api_key=os.environ["AZURE_API_KEY"], azure_endpoint=os.environ["AZURE_API_BASE"],
                               api_version=os.environ["AZURE_API_VERSION"]),
            deployment=memory_settings["llm_deployment"],
        )
        runtime = LongMemoryRuntime(
            MemoryConfig(mode=mode, database_path=args.output / "memory.sqlite",
                         max_candidates=experiment["max_candidates"],
                         max_guidance_chars=experiment["max_guidance_chars"],
                         minimum_confidence=experiment["minimum_confidence"]),
            embedder=JinaCodeEmbedder(memory_settings["model_path"], memory_settings["tokenizer_path"]),
            memory_llm=memory_llm,
        )

    events_path = args.output / "events.jsonl"
    done = set()
    if events_path.exists():
        done = {json.loads(line)["instance_id"] for line in events_path.read_text().splitlines() if line.strip()}
    try:
        for instance in instances:
            iid = instance["instance_id"]
            if iid in done:
                print(f"skip {iid} (already in {events_path})")
                continue
            started = time.time()
            usage_start = len(memory_llm.usage_log) if memory_llm else 0
            task = TaskRecord(instance_id=iid, issue_text=instance["problem_statement"],
                              sequence_index=instance["sequence_index"], order_id=order_id)
            packet = runtime.before_task(task)
            rendered = packet.render()
            config = copy.deepcopy(base_config)
            extra_vars = {}
            if rendered:
                config["agent"]["instance_template"] = memory_template(config["agent"]["instance_template"])
                extra_vars = {"memory_guidance": rendered}
            print(f"[{mode.value}] {iid}: memory cases {list(packet.case_ids)}, guidance {len(rendered)} chars")

            result = run_instance(instance, config, args.output, extra_vars)
            trajectory = json.loads(result["traj_path"].read_text()) if result["traj_path"].exists() else {}
            messages = trajectory.get("messages", [])
            model_stats = trajectory.get("info", {}).get("model_stats", {})
            instance_prompt = _text(messages[1].get("content")) if len(messages) > 1 else ""
            compact, compact_stats = compact_trajectory(messages)

            case = runtime.after_task(TaskRecord(
                instance_id=iid, issue_text=instance["problem_statement"],
                sequence_index=instance["sequence_index"], order_id=order_id,
                trajectory_text=compact, patch_text=result["submission"],
                run_config={"arm": mode.value, "exit_status": result["exit_status"],
                            "agent_cost": model_stats.get("instance_cost"), "api_calls": model_stats.get("api_calls"),
                            "memory_case_ids": list(packet.case_ids), "trajectory_compaction": compact_stats},
                artifact_paths={"trajectory": str(result["traj_path"]), "preds": str(args.output / "preds.json")},
            ))
            usage = memory_llm.usage_log[usage_start:] if memory_llm else []
            event = {
                "instance_id": iid, "arm": mode.value, "batch": args.batch,
                "sequence_index": instance["sequence_index"],
                "exit_status": result["exit_status"], "patch_chars": len(result["submission"]),
                "agent_cost": model_stats.get("instance_cost"), "api_calls": model_stats.get("api_calls"),
                "memory_case_ids": list(packet.case_ids), "retrieval_event_id": packet.retrieval_event_id,
                "guidance_chars": len(rendered),
                # Checked against the prompt the model actually received, not the intended one.
                "memory_delivered": bool(rendered) and rendered in instance_prompt,
                "reflection_admitted": case is not None, "reflection_case_id": case.case_id if case else None,
                "memory_usage": usage, "memory_cost": _memory_cost(usage, memory_prices),
                "trajectory_compaction": compact_stats if mode is MemoryMode.ENABLED else None,
                "error": result["error"].get("exception_str"),
                "seconds": round(time.time() - started, 1),
            }
            with events_path.open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            print(f"[{mode.value}] {iid}: {result['exit_status']}, agent ${event['agent_cost'] or 0:.3f}, "
                  f"memory ${event['memory_cost']:.4f}, {event['seconds']}s")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
