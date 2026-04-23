# Plan Packet: Observability UI

## Objective

Provide a lightweight live visualization layer for runs, rewards, worker state, and artifacts.

## Inputs

- dashboard snapshot API
- run detail API
- worker and artifact records

## Outputs

- static React app
- TanStack Router navigation
- run overview and detail views

## Dependencies

- artifact store
- API server

## Acceptance Checks

- frontend renders backend state without owning workflow logic
- no SSR or heavy API composition is required

## Rollback / Failure Notes

- if the UI starts absorbing orchestration logic, move that logic back into the backend immediately
