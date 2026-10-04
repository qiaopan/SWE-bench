# SWE-bench Long Memory experiment scaffold

This directory contains the reusable part of the earlier WebGen Long Memory design,
adapted to repository-level bug fixing. It is intentionally independent of the
upstream `swebench infer` command until the exact mini-SWE-agent prompt-injection
configuration is pinned and smoke-tested.

## What is carried over

- A strict `disabled` No Memory arm: no database, embedding, retrieval, reflection,
  or memory-model call is allowed.
- Append-only task provenance and reflection cases in a per-run SQLite database.
- Interfaces for `AgentAdapter`, `MemoryLLM`, and `Embedder`, so model/provider and
  agent changes do not alter the experiment protocol.
- Post-task evidence-grounded reflection, local semantic retrieval, LLM reranking,
  and short task-specific guidance.
- Bounded prompt packets and retrieval-event records for later audit.

## SWE-bench-specific evidence and leakage boundary

The reflection model may use only the issue, agent-visible trajectory, generated
patch, and agent-run self-test log. It must not receive a gold patch, hidden tests,
or an official SWE-bench harness result. Harness results are reporting data only and
must be kept out of same-task reflection and all later task memory.

Each formal ordering owns a separate memory database. Dev/smoke runs use a separate
database and never seed a test run. The store also excludes an `instance_id` from
retrieving its own earlier record, so retries do not read their own reflection.

## Initial profile

- Agent/generator: mini-SWE-agent v2.0.0 with its `benchmarks/swebench.yaml`
  (step limit 250, cost limit $3, 60 s command timeout) and Azure `gpt-5-nano`
  (`2025-08-07`, deployment `swe-generator-gpt5-nano`). Chosen for headroom: on the 37
  sampled django instances that are also in Verified, the official per-instance results
  give gpt-5-mini 27/37 and gpt-5-nano 15/37, and 12 of nano's failures were solved by
  mini. gpt-5-nano's only official result is v1.7.0 (34.8% on Verified's 500 tasks) and
  those trajectories record no model config, so nano uses the model settings of the
  official gpt-5-mini v2.0.0 run (below). The No Memory arm is the comparison baseline.
- Long Memory LLM: `gpt-5-mini` (deployment `swe-reflection-gpt5-mini`) for reflection
  and reranking/rewrite only. A stronger memory model than the generator is a design
  choice (separate generator and memory roles, as in phase 1) and a stated limitation.
- No temperature is sent to either role; reasoning effort is `medium`.
- Tasks: one repository, in time order. 100 of the 114 `django/django` Lite test
  instances (seed 20261004), sorted by issue creation time; batch 1 is the earliest 50,
  batch 2 continues the same memory stream if the budget allows. Both arms use the same
  list and order. Rationale: lessons rarely transfer across repositories (in the first
  dev trial every cross-repository candidate was rejected), and an agent that keeps
  working on one codebase is a realistic setting; cross-repository transfer is out of
  scope and stated as a limitation. Dev runs use the 5 sqlfluff dev instances in time
  order.
- Embedder: local `jinaai/jina-embeddings-v2-base-code`, pinned to the downloaded
  revision and loaded only when Long Memory is enabled.
- Evaluation: the SWE-bench Docker harness. It is not an LLM evaluator.

The template at `config/long_memory_swe.example.yaml` pins these roles without
including Azure credentials. `AzureOpenAIMemoryLLM` receives an already-authenticated
client from the eventual runner, so endpoint and secret handling remain outside the
research protocol.

## Running

Two virtual environments, because the harness needs `huggingface_hub>=1.20` while the
pinned Jina stack (`transformers==4.57.6`) needs `<1.0`:

- `.venv`: agent and memory (`mini-swe-agent==2.0.0`, torch, transformers, this repo
  linked with `--no-deps`; `swebench` imports lazily, so the harness deps are unused).
- `.venv-eval`: this repo with its full dependencies, for the Docker harness only.

Docker on Apple silicon runs through Colima with Rosetta (x86_64 task images).
Credentials live in a git-ignored `.env` (`AZURE_API_KEY`, `AZURE_API_BASE`,
`AZURE_API_VERSION`). The pinned Lite parquet files live in the git-ignored
`data/swebench_lite/`; the runner refuses a file whose sha256 differs from the id list.

`swebench.long_memory.run_mini` runs one arm over one batch, sequentially, appending to
`<output>/events.jsonl`, `preds.json` and per-instance trajectories. Reuse the same
output directory for later batches of the same arm so memory accumulates in the frozen
order. Notes on fidelity to the baseline:

- The official `benchmarks/swebench.yaml` templates, limits and environment are used
  as is (verified identical to those recorded in the baseline trajectories). The model
  part follows what the official gpt-5-mini v2.0.0 baseline actually ran, read from its
  trajectories: the Responses API class `litellm_response` with `reasoning.effort` and
  `text.verbosity` `medium` and `temperature` unset, not the yaml's chat-completions
  default. Containers start with `--platform linux/amd64`. A first dev attempt with the
  chat-completions class was stopped after one task and kept as
  `outputs/dev-lite-nomem.chat-api-aborted`; it is not a valid result.
- Without guidance the instance prompt is the official one byte for byte; with guidance
  one block is inserted after `</pr_description>`. `memory_delivered` in each event is
  checked against the prompt actually sent.
- Guidance must come from memory. The rerank/rewrite step (`swe-rerank-rewrite-v2`)
  returns items that each cite a retrieved lesson; uncited items are dropped and no
  selected lesson means no guidance. Under v1 the memory model, having rejected the
  only candidate, wrote its own diagnosis and fix for the current issue and that text
  was injected (`outputs/dev-lite-longmem.unsourced-guidance-aborted`, invalid).
- Reflection receives a compacted trajectory: every thought and command, each command
  output clipped to 2,000 characters, and the whole clipped to 60,000 characters.
- Agent cost uses `config/litellm_model_registry.json` (gpt-5-mini Azure prices), so the
  $3 per-task cost limit is enforced.

The image named in the dataset's `image` field and the one mini-SWE-agent derives are
the same Docker Hub image, so the agent and the harness see the same environment.
