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
RL_POLICY_BACKEND   static | claude
RL_STORE_BACKEND    memory | sqlite
RL_ENV_BACKEND      simulated | repo
```

Every adapter lives behind the same `domain/contracts.py` ABC, so the
coordinator doesn't know which backend it's driving.

## Live observability

- FastAPI `/ws/events` streams typed `DomainEvent`s (rollout lifecycle, steps,
  rewards, logs, worker state).
- Structlog log lines flow into the same stream via `install_event_bus_handler`.
- Frontend subscribes over WebSocket (auto-reconnect); no polling.
- Run-detail view parses the rubric-driven reward provenance into a signal
  breakdown (value, weight, reason).

## Reward iteration

Reward is a pure `RubricSpec` of signal functions
(`domain/rewards.py`): step-count, error-penalty, success-criteria-match,
finish-detection. Swap `CompositeRewardPipeline.rubric` to iterate; stored
trajectories can be re-scored without re-running rollouts, and every
`RewardRecord` carries the rubric name + per-signal breakdown in its
provenance.

## Local Startup

Backend:

```bash
pip install -e '.[dev]'
cp .env.example .env     # fill RL_ANTHROPIC_API_KEY when using Claude
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
RL_SMOKE_API_KEY=sk-ant-... pytest tests/smoke -v
```

Runs one real Haiku rollout against a tiny tempdir repo; designed to cost
well under a cent per invocation.

## Configuration

All settings are env-driven with the `RL_` prefix (see `.env.example`).
Highlights:

- `RL_POLICY_BACKEND=claude` · `RL_ANTHROPIC_API_KEY` · `RL_CLAUDE_MODEL`
- `RL_STORE_BACKEND=sqlite` (persists at `$RL_ARTIFACTS_DIR/runs.db`)
- `RL_ENV_BACKEND=repo` (per-rollout tempdir snapshot via `git ls-files`)
- `RL_MAX_TOKENS_PER_RUN` enforces a per-run cost ceiling; the coordinator
  aborts with a trajectory error when exceeded.
