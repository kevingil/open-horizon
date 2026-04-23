# Plan Packet: Reward Pipeline

## Objective

Produce replayable, inspectable reward records and audit metadata from trajectory artifacts.

## Inputs

- task spec
- trajectory record
- reward heuristics or learned reward adapters

## Outputs

- reward records
- penalty records
- audit flags and provenance labels

## Dependencies

- environment runner
- policy serving

## Acceptance Checks

- reward can be recomputed from saved inputs without rerunning the environment
- reward provenance is explicit in stored records

## Rollback / Failure Notes

- if reward logic becomes opaque, add audit flags before adding sophistication
