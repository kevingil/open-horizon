# Plan Track A: Mac Local Debug Loop

## Objective

Establish a safe, local, reproducible rollout loop that proves the architecture works end to end without depending on rented GPUs or Ray.

## Inputs

- local filesystem snapshot
- small open model or simulated local policy
- task prompt and success criteria

## Outputs

- run manifest
- trajectory record
- reward record
- observable worker state
- dashboard-ready artifacts

## Acceptance Checks

- task can be created, stepped, and reset locally
- trajectory can be inspected without external infrastructure
- reward can be replayed from saved trajectory inputs
- artifact-backed dashboard renders current run state

## Failure Notes

- if local policy inference is unstable, fall back to simulated policy output while preserving interfaces
- if local environment fidelity is too expensive, keep tasks narrow and deterministic for Track A
