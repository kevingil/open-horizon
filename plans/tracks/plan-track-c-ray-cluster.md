# Plan Track C: Ray Cluster Replica

## Objective

Promote Ray to the main orchestration backend only after the local and cheap GPU paths are stable and artifact-compatible.

## Inputs

- stable contracts from Tracks A and B
- queueing and worker split requirements
- resumability and failure recovery requirements

## Outputs

- actor-based rollout architecture
- separated rollout, reward, inference, and observability workers
- resumable multi-worker execution path

## Acceptance Checks

- the same task contract runs under local mode and Ray mode
- run manifests and trajectory artifacts do not change shape
- worker failures can be surfaced and resumed from backend state

## Failure Notes

- if Ray adds unnecessary operational drag too early, keep it optional until the workload justifies the move
- do not let Ray-specific assumptions leak into core domain models
