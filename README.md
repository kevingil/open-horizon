# Distributed RL for Long-Horizon LLM Agents

Training, evaluations, and observability for long-horizon LLM agents.

The control plane is Rust: durable jobs with leases, a sequenced event
log, the HTTP and WebSocket API, and the rollout loop that keeps the
inference server busy. Python stays where the research ecosystem lives:
verifiers environments, the GRPO and prime-rl trainers, and tokenizers,
driven as a supervised worker over NDJSON.

## Quick Start

Backend (no Python needed for the default configuration):

```bash
cp .env.example .env
make dev            # horizon serve, SQLite store under ./artifacts
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Then open:

- API: `http://127.0.0.1:8000` (OpenAPI at `/openapi.json`)
- Dashboard: `http://127.0.0.1:5173`

The default profile routes to verifiers, which needs the Python bridge.
For a zero-dependency first run, set `RL_ENV_BACKEND=repo` and point
`RL_LLM_BASE_URL` at any OpenAI-compatible endpoint.

## Development Loop

```bash
make lint           # clippy -D warnings + rustfmt check
make test           # cargo test --workspace
make py-test        # python/ bridge tests (needs python/.venv, see below)
make check          # all of the above
make openapi        # regenerate docs/openapi.json from the Rust types
```

Python bridge setup:

```bash
cd python
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
# Extras as needed: [envs] for verifiers, [train] for GRPO + tokenizers,
# [prime-rl] for the prime-rl trainer.
```

CLI (one binary, `horizon`):

```bash
cargo run -p horizon-server -- serve
cargo run -p horizon-server -- train --samples run-aaa,run-bbb [--parent adapter-x] [--steps 8]
cargo run -p horizon-server -- train --list
cargo run -p horizon-server -- eval --adapter adapter-y
cargo run -p horizon-server -- replay --run run-aaa --rubric coding-v1 [--dry-run]
cargo run -p horizon-server -- replay --list
```

## Layout

```
crates/
├── horizon-core/     # Canonical DTOs (OpenAPI via utoipa), events, pricing, rubrics, tool schema
├── horizon-store/    # SQLite: runs, workers, turn training, training runs, evals, jobs, event log
├── horizon-runner/   # Sandbox, repo snapshot, tool dispatch, OpenAI-compat client, rollout loop
├── horizon-bridge/   # Supervised Python worker over NDJSON (multiplexed, restart on crash)
└── horizon-server/   # Coordinator, job runner, training, eval, axum API + WebSocket, CLI
python/
└── horizon_bridge/   # verifiers rollouts, GRPO-lite + prime-rl trainers, tokenizer op
frontend/             # React + TanStack Router, live via /ws/events
plans/                # Master plan, track plans, agent-agnostic packets
docs/                 # RESEARCH.md (training workflow), openapi.json
```

## How work flows

1. `POST /api/runs` validates the profile and inserts a `rollout` job.
2. The job runner leases it (SQLite, `lease_until`), heartbeats while it
   runs, and marks it terminal. A crashed worker leaves an expired lease
   that the next runner picks up; after `RL_JOB_MAX_ATTEMPTS` it fails.
3. Repo profiles run the Rust loop: chat completion, tool dispatch into a
   sandboxed snapshot, token and cost accounting, per-turn events.
   Cancellation races the HTTP call, so a cancelled rollout never waits
   for the provider to finish generating.
4. Verifiers profiles hand the task to the Python bridge, which runs
   `Environment.run_rollout` and returns a scored trajectory. Rust owns
   cost, persistence, and events.
5. Every event is appended to the `events` table with a sequence number
   and broadcast. `/ws/events?since=<seq>` resumes from any point.

Training runs are jobs too. The stub trainer runs in Rust; `grpo` and
`prime-rl` run in the bridge with metrics streamed back as
`training.metric` events. Published adapters can be hot-loaded into
SGLang (`RL_SGLANG_AUTOLOAD_LORA=true`).

## Backends (swap via env)

```
RL_STORE_BACKEND    memory | sqlite
RL_ENV_BACKEND      verifiers | repo
RL_ENV_SANDBOX      none | docker
RL_TRAINER_BACKEND  stub | grpo | prime-rl
```

## Policy: any OpenAI-compatible provider

```bash
# OpenAI proper
RL_LLM_API_KEY=sk-... RL_LLM_MODEL=gpt-5.4-mini

# Local vLLM (vllm:* prefix marks it $0 in cost tracking)
RL_LLM_BASE_URL=http://127.0.0.1:8000/v1
RL_LLM_MODEL=vllm:Qwen/Qwen3-8B
RL_LLM_API_KEY=not-needed

# SGLang (scripts/serve_sglang.sh; sglang:* prefix marks it $0)
RL_LLM_BASE_URL=http://127.0.0.1:30000/v1
RL_LLM_MODEL=sglang:Qwen/Qwen3-8B
RL_LLM_API_KEY=not-needed

# Ollama
RL_LLM_BASE_URL=http://127.0.0.1:11434/v1
RL_LLM_MODEL=ollama:qwen3:8b

# Anthropic via OpenAI-compat
RL_LLM_BASE_URL=https://api.anthropic.com/v1/
RL_LLM_API_KEY=sk-ant-...
RL_LLM_MODEL=claude-haiku-4-5
```

Named profiles (`RL_POLICY_PROFILES`, JSON) let one server route
different rollouts to different providers and paths; see `.env.example`.

## Live observability

- `/ws/events` streams typed `DomainEvent`s (rollout lifecycle, steps,
  rewards, logs, worker state, progress, cancellation, budget, training).
- Tracing log lines from the Rust crates flow into the same stream as
  `log.line` events.
- `/api/events?since=N` and `/api/jobs` expose the durable log and the
  job table for debugging and for other workers.

## Smoke test

`scripts/smoke/run.sh` builds the release binary, starts the scripted
mock policy (`horizon mock-policy`), the server on SQLite, and the Vite
dashboard, then drives the browser through queue, live progress, run
detail, rescoring, training from a selection, eval, and a mid-flight
cancel. Screenshots and a `report.json` land under `artifacts/smoke/`.

## Compatibility notes

- The store schema is normalised (steps, reward signals, and training
  metrics are rows, timestamps are integers) and is not compatible with
  databases written by the previous Python service.
- Events are envelopes: `{seq, at, subject, kind, payload}`. Frontend
  types are generated from `/openapi.json` (`npm run gen:api`).
- Heuristic rubrics (`heuristic-v1`, `coding-v1`, `strict-finish-v1`)
  are kept for replay. Production reward should come from verifiers
  rubrics or judge models, not from extending these.

See `docs/RESEARCH.md` for the training-engineer workflow.
