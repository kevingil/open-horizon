# Master Plan

## Objective

Build a docs-first, agentic reinforcement learning stack that is:

- runnable on a Mac for local debugging
- portable to short-lived rented GPUs
- compatible with Ray when scale-out becomes necessary
- observable through a thin web UI

## Program Shape

1. Track A proves the loop locally with small models or simulated policy logic.
2. Track B keeps the same contracts while swapping in cheap GPU execution.
3. Track C preserves interfaces and moves orchestration to Ray actors and queues.

## Architectural Contracts

- `EnvironmentRunner`: task lifecycle and environment stepping
- `ToolHarness`: controlled interaction surface for files, search, and terminal activity
- `PolicyServer`: inference abstraction that can remain local or later point to vLLM
- `RewardPipeline`: reward calculation, replay, and audit
- `ArtifactStore`: durable manifests, trajectories, rewards, and reports
- `RolloutCoordinator`: orchestration entrypoint across all tracks

## Observability

- thin Python API
- static React app with TanStack Router
- backend remains the source of truth
- frontend only visualizes run status, rewards, workers, and artifacts

## Delivery Rules

- prefer docs-first changes before broad implementation work
- keep all work packets agent-agnostic
- avoid hidden coupling between Track A and later infrastructure
- favor replayable artifacts and manifests over ad hoc debugging
