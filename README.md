# RL Stack

Docs-first starter repo for a one-person, many-agent agentic reinforcement learning stack.

Three priorities:

1. Mac-runnable local debug path.
2. Cheap single-GPU path that preserves the same interfaces.
3. Ray-compatible scale-out path without redesigning core abstractions.

## Layout

```
src/rl_stack/
├── domain/           # Pure models, contracts, events, pure reward signals
├── application/      # Async coordinator + event bus
├── infrastructure/   # Policy / env / rewards / store / tools adapters
├── interface/api/    # FastAPI + WebSocket
├── settings.py       # pydantic-settings, env-driven
├── logging.py        # structlog with contextvars + event-bus bridge
└── bootstrap.py      # Wiring
frontend/             # React + TanStack Router, live via /ws/events
plans/                # Master plan, track plans, agent-agnostic packets
tests/                # unit / contract / integration / smoke
```

## Backends (swap via env)

```
RL_POLICY_BACKEND   static | openai
RL_STORE_BACKEND    memory | sqlite
RL_ENV_BACKEND      simulated | repo | verifiers
RL_ENV_SANDBOX      none   | docker
```

Every adapter lives behind the same `domain/contracts.py` ABC, so the
coordinator doesn't know which backend it's driving.

## Policy: any OpenAI-compatible provider

The `openai` backend talks to anything that speaks OpenAI's Chat Completions
API: OpenAI proper, vLLM, Ollama, OpenRouter, llama.cpp server, Anthropic via
their compat surface. Pick the provider with three env vars:

```bash
# OpenAI proper:
RL_LLM_API_KEY=sk-... RL_LLM_MODEL=gpt-4o-mini

# Local vLLM (the vllm:* prefix marks it $0 in cost tracking):
RL_LLM_BASE_URL=http://127.0.0.1:8000/v1
RL_LLM_MODEL=vllm:Qwen/Qwen2.5-7B-Instruct
RL_LLM_API_KEY=not-needed

# Ollama:
RL_LLM_BASE_URL=http://127.0.0.1:11434/v1
RL_LLM_MODEL=ollama:qwen2.5:7b

# SGLang (drop-in OpenAI-compat; the sglang:* prefix marks it $0):
RL_LLM_BASE_URL=http://127.0.0.1:30000/v1
RL_LLM_MODEL=sglang:Qwen/Qwen2.5-7B-Instruct
RL_LLM_API_KEY=not-needed

# Anthropic via OpenAI-compat:
RL_LLM_BASE_URL=https://api.anthropic.com/v1/
RL_LLM_API_KEY=sk-ant-...
RL_LLM_MODEL=claude-haiku-4-5
```

## Live observability

- FastAPI `/ws/events` streams typed `DomainEvent`s (rollout lifecycle, steps,
  rewards, logs, worker state, progress, cancellation, budget).
- Structlog log lines flow into the same stream via `install_event_bus_handler`.
- Frontend subscribes over WebSocket (auto-reconnect); no polling.
- Run-detail view parses the rubric-driven reward provenance into a signal
  breakdown (value, weight, reason).

## Training loop

A closed loop: rollouts feed a trainer, the trainer publishes an adapter,
the adapter slots into the policy server, the eval harness scores it.
All visualised live via WebSocket events.

```bash
# Default: stub trainer (deterministic fake; no ML deps required)
make dev

# Real GRPO-lite LoRA training
pip install -e '.[train]'
RL_TRAINER_BACKEND=grpo \
RL_GRPO_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    make dev

# Trigger a training run from completed rollouts
rl-train --samples run-aaa,run-bbb [--parent adapter-x] [--steps 16]
rl-train --list

# Evaluate an adapter against the built-in task set
rl-eval --adapter adapter-y
```

The frontend ships **Adapters**, **Training Runs**, and per-run live charts
(loss / mean_reward / kl) updated from `training.metric` events. Multi-select
completed rollouts on the dashboard and click "Train from selection" to kick
off a run from the UI.

### vLLM / SGLang / Ollama for serving the policy

The OpenAI-compat policy server already speaks any OpenAI-shape endpoint, so
swapping in vLLM, SGLang, or Ollama is two env vars:

```bash
# vLLM hosting Qwen with LoRA mounting:
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
    --lora-modules adapter-aaa=./artifacts/adapters/adapter-aaa
RL_POLICY_BACKEND=openai \
RL_LLM_BASE_URL=http://127.0.0.1:8000/v1 \
RL_LLM_MODEL=vllm:adapter-aaa \
    make dev

# SGLang (RadixAttention prefix caching speeds up multi-turn rollouts):
SGLANG_MODEL=Qwen/Qwen2.5-7B-Instruct \
SGLANG_LORA_PATHS="adapter-aaa=./artifacts/adapters/adapter-aaa" \
    scripts/serve_sglang.sh
RL_POLICY_BACKEND=openai \
RL_LLM_BASE_URL=http://127.0.0.1:30000/v1 \
RL_LLM_API_KEY=not-needed \
RL_LLM_MODEL=sglang:adapter-aaa \
    make dev

# Ollama:
RL_LLM_BASE_URL=http://127.0.0.1:11434/v1 \
RL_LLM_MODEL=ollama:qwen2.5:7b \
    make dev
```

`vllm:*` / `sglang:*` / `ollama:*` / `local:*` model id prefixes are treated
as $0 cost in the dashboard so self-hosted rollouts don't fake spend.

SGLang isn't supported on macOS — keep Mac on a remote SGLang reachable via
`RL_LLM_BASE_URL`, or stay on the static / OpenAI policy locally.

### verifiers as the rollout loop

`RL_ENV_BACKEND=verifiers` delegates the per-step rollout to the
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework. Its
`env.rollout(client, model, prompt, ...)` owns the loop and produces both
the trajectory and a rubric-scored reward in one shot - so on this path
`CompositeRewardPipeline` is bypassed and verifiers' rubric is the source
of truth.

```bash
pip install -e '.[envs]'                     # adds verifiers
RL_ENV_BACKEND=verifiers \
RL_VERIFIERS_ENV_ID=vf-math \
RL_POLICY_BACKEND=openai \
RL_LLM_BASE_URL=http://127.0.0.1:30000/v1 \
RL_LLM_MODEL=sglang:Qwen/Qwen2.5-7B-Instruct \
    make dev
```

`RL_VERIFIERS_ENV_ARGS` is JSON forwarded to `vf.load_environment`. Hub
envs (`rlm`, `opencode/*`, ...) are installed via the Prime Intellect CLI
(`pip install prime`).

## Reward iteration

Reward is a pure `RubricSpec` of signal functions
(`domain/rewards.py`): step-count, error-penalty, success-criteria-match,
finish-detection, tests-pass. Swap `CompositeRewardPipeline.rubric` to
iterate; stored trajectories can be re-scored without re-running rollouts,
and every `RewardRecord` carries the rubric name + per-signal breakdown in
its provenance.

```bash
rl-replay --list                                   # registered rubrics
rl-replay --run run-abc123 --rubric strict-finish-v1
```

## Local Startup

Backend:

```bash
pip install -e '.[dev]'
cp .env.example .env     # fill RL_LLM_API_KEY when using openai backend
make dev
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Development

```bash
make test     # pytest
make lint     # ruff
make check    # lint + test
make dev      # uvicorn --reload
```

### Smoke test (optional, real API)

```bash
# OpenAI default:
RL_SMOKE_API_KEY=sk-... pytest tests/smoke -v

# Against a local vLLM:
RL_SMOKE_API_KEY=not-needed \
RL_SMOKE_BASE_URL=http://127.0.0.1:8000/v1 \
RL_SMOKE_MODEL=Qwen/Qwen2.5-7B-Instruct \
    pytest tests/smoke -v
```

Runs one real LLM rollout against a tiny tempdir repo; designed to cost well
under a cent per invocation against gpt-4o-mini.

## Configuration

All settings are env-driven with the `RL_` prefix (see `.env.example`).
Highlights:

- `RL_POLICY_BACKEND=openai` · `RL_LLM_API_KEY` · `RL_LLM_BASE_URL` · `RL_LLM_MODEL`
- `RL_STORE_BACKEND=sqlite` (persists at `$RL_ARTIFACTS_DIR/runs.db`)
- `RL_ENV_BACKEND=repo` (per-rollout tempdir snapshot via `git ls-files`)
- `RL_ENV_SANDBOX=docker` (auto-falls-back to `none` if Docker is missing)
- `RL_MAX_TOKENS_PER_RUN` per-run cost ceiling; the coordinator aborts with
  a trajectory error when exceeded.
- `RL_DAILY_BUDGET_USD` rolling-window USD cap; new rollouts are refused
  with a `BudgetExceeded` event when reached.
