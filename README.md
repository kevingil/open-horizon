# RL Stack

Docs-first starter repo for a one-person, many-agent agentic reinforcement learning stack.

Three priorities:

1. Mac-runnable local debug path.
2. Cheap single-GPU path that preserves the same interfaces.
3. Ray-compatible scale-out path without redesigning core abstractions.

## Layout

```
src/rl_stack/
├── domain/           # Pure models, contracts, events (no I/O)
├── application/      # Orchestration + event bus
├── infrastructure/   # Policy / env / rewards / store / tools adapters
├── interface/api/    # FastAPI + WebSocket
├── settings.py       # pydantic-settings, env-driven
├── logging.py        # structlog with contextvars + event-bus bridge
└── bootstrap.py      # Wiring
frontend/             # React + TanStack Router, live via /ws/events
plans/                # Master plan, track plans, agent-agnostic packets
tests/                # unit / contract / integration
```

## Core contracts

- `EnvironmentRunner`, `ToolHarness`, `PolicyServer`, `RewardPipeline`, `ArtifactStore`, `RolloutCoordinator`
- All live in `domain/contracts.py`; every implementation lives in `infrastructure/*`
- Swap implementations via `Settings.policy_backend` and `bootstrap.py` wiring

## Live observability

- FastAPI `/ws/events` streams typed `DomainEvent`s (rollout lifecycle, steps, rewards, logs, worker state)
- Structlog log lines flow into the same stream via `install_event_bus_handler`
- Frontend subscribes over WebSocket (auto-reconnect), no polling

## Local Startup

Backend:

```bash
pip install -e '.[dev]'
uvicorn rl_stack.interface.api.app:app --reload
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

## Configuration

All settings are env-driven with the `RL_` prefix (see `src/rl_stack/settings.py`):

- `RL_POLICY_BACKEND=static` (future: `claude`, `vllm`)
- `RL_MAX_PARALLEL_ROLLOUTS=4`
- `RL_LOG_LEVEL=INFO` · `RL_LOG_JSON=false`
- `RL_WORKSPACE_ROOT=.` · `RL_ARTIFACTS_DIR=./artifacts`
