# Plan Packet: Cost Controls

## Objective

Keep cheap local and rented-GPU experimentation practical and visible.

## Inputs

- infra target
- model choice
- run duration assumptions

## Outputs

- cost metadata in run manifests
- guidance for local vs rented GPU execution
- run-level budget visibility in the dashboard

## Dependencies

- rollout coordinator
- observability UI

## Acceptance Checks

- each run records estimated cost
- dashboard can surface cost alongside run status

## Rollback / Failure Notes

- if exact cost accounting is unavailable, store estimates first rather than omitting the signal
