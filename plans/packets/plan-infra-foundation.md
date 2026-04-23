# Plan Packet: Infrastructure Foundation

## Objective

Define local-first project structure, dependency boundaries, and deployment assumptions.

## Inputs

- master architecture plan
- local Mac-first constraint
- later Ray compatibility requirement

## Outputs

- Python package structure
- API entrypoint
- frontend workspace boundary
- deployment notes for local and GPU targets

## Dependencies

- none

## Acceptance Checks

- project layout clearly separates domain logic, API, and UI
- interfaces are independent from infrastructure-specific adapters

## Rollback / Failure Notes

- if infrastructure assumptions force contract churn, move those assumptions behind adapters instead
