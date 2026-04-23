# Plan Packet: Environment Runner

## Objective

Build the environment lifecycle surface for coding-task creation, stepping, and reset.

## Inputs

- task specs
- local repo snapshot
- tool permissions

## Outputs

- environment adapter contract
- local debug implementation
- logs suitable for replay and inspection

## Dependencies

- infrastructure foundation

## Acceptance Checks

- environment tasks can be created and stepped deterministically
- observations are stored as trajectory-compatible records

## Rollback / Failure Notes

- if real environments are too unstable early, use deterministic simulated observations while preserving the contract
