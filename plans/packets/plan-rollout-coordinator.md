# Plan Packet: Rollout Coordinator

## Objective

Centralize rollout orchestration without hardcoding local or Ray-specific execution assumptions.

## Inputs

- environment runner
- policy server
- reward pipeline
- artifact store

## Outputs

- orchestrated run lifecycle
- consistent manifest creation
- artifact-backed run details

## Dependencies

- environment runner
- policy serving
- reward pipeline
- artifact store

## Acceptance Checks

- one request produces a full run detail package
- local mode and future Ray mode can share the same coordinator contract

## Rollback / Failure Notes

- if the coordinator starts accumulating infrastructure logic, move that logic into adapters
