# Plan Packet: Artifact Store

## Objective

Keep runs, trajectories, rewards, manifests, and observability data in a backend-owned source of truth.

## Inputs

- run detail records
- worker status records
- artifact metadata

## Outputs

- listable runs
- detailed run lookup
- dashboard snapshots

## Dependencies

- infrastructure foundation

## Acceptance Checks

- dashboard can be rendered entirely from artifact-store output
- artifacts remain stable across local and future distributed modes

## Rollback / Failure Notes

- if a storage backend becomes too complex too early, keep an in-memory or file-backed implementation with the same interface
