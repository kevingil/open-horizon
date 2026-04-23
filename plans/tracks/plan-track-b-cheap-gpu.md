# Plan Track B: Cheap GPU Loop

## Objective

Run the same contracts on a short-lived single GPU so rollout throughput and policy realism improve without redesigning the system.

## Inputs

- Track A contracts and data formats
- rented GPU target
- optional vLLM-backed policy server

## Outputs

- GPU-targeted run manifests
- cost-aware rollout records
- replayable reward and artifact outputs identical in shape to Track A

## Acceptance Checks

- a Track A task can execute through the same API surface on the GPU path
- cost metadata is recorded per run
- worker and artifact views remain usable without frontend changes

## Failure Notes

- if a chosen model does not fit affordable VRAM, swap to a smaller model without changing the interface
- if vLLM adds too much complexity early, keep a simpler single-process policy adapter first
