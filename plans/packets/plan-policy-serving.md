# Plan Packet: Policy Serving

## Objective

Define a local-first policy interface that can later target vLLM or other inference backends.

## Inputs

- task context
- rollout history
- infra target

## Outputs

- `PolicyServer` contract
- local debug implementation
- future backend adapter seams

## Dependencies

- infrastructure foundation

## Acceptance Checks

- rollout code depends only on the policy interface
- model identity and adapter identity are captured in run manifests

## Rollback / Failure Notes

- if a real model is unavailable, use static or heuristic policy outputs rather than blocking the architecture
