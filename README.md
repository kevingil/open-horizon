# RL Stack

Docs-first starter repo for a one-person, many-agent agentic reinforcement learning stack.

This repository is intentionally organized around three priorities:

1. A Mac-runnable local debug path.
2. A cheap single-GPU path that preserves the same interfaces.
3. A Ray-compatible scale-out path that can be added without redesigning the core abstractions.

## Layout

- `plans/`: master plan, track plans, and agent-agnostic `plan-*.md` work packets.
- `src/rl_stack/`: Python contracts, services, and a thin API server.
- `frontend/`: static React dashboard with TanStack Router for live observability.

## Backend Architecture

The initial backend keeps the important surfaces explicit:

- `EnvironmentRunner`
- `ToolHarness`
- `PolicyServer`
- `RolloutCoordinator`
- `RewardPipeline`
- `ArtifactStore`

The first implementation is a safe local scaffold with sample data and a tiny observability API. The shape is ready for local simulation, later GPU-backed inference, and optional Ray orchestration.

## Frontend

The dashboard is intentionally thin:

- static React app
- TanStack Router
- simple fetch-based API client
- live run state, rewards, artifacts, and worker status views

The frontend is not the system of record. It only visualizes backend records.

## Notes

- This repo intentionally avoids adding or running automated tests because the workspace instructions prohibit writing or running tests.
- Validation is expected to rely on reproducible manifests, smoke flows, artifacts, and replayable records.

## Local Startup

Backend:

```bash
uvicorn rl_stack.api.app:app --app-dir src --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```
