"""Trigger a training run from the command line.

Usage (installed as a console script):
    rl-train --samples run-aaa,run-bbb [--parent adapter-x] [--steps 8]
    rl-train --list                    # list registered adapters

Wraps application.training.TrainingService.start_training; awaits the
in-process trainer task and prints the resulting AdapterRecord on success.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from ...application.training import TrainingRequest, TrainingService
from ...bootstrap import build_application_services
from ...domain.models import TrainingStatus
from ...settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rl-train")
    parser.add_argument("--samples", help="comma-separated rollout run ids to use as training samples")
    parser.add_argument("--parent", help="parent adapter id (optional)")
    parser.add_argument("--steps", type=int, help="training steps override")
    parser.add_argument("--list", action="store_true", help="list adapters and exit")
    args = parser.parse_args(argv)

    services = build_application_services(Settings())

    if args.list:
        for adapter in services.adapter_registry.list_adapters():
            score = f"{adapter.eval_score:.3f}" if adapter.eval_score is not None else "  -  "
            print(f"{adapter.id}  parent={adapter.parent_id or '-'}  eval={score}  base={adapter.base_model}")
        return 0

    if not args.samples:
        parser.error("--samples is required unless --list is passed")

    sample_ids = [s.strip() for s in args.samples.split(",") if s.strip()]
    hyperparams: dict = {}
    if args.steps is not None:
        hyperparams["steps"] = args.steps

    return asyncio.run(
        _run(services.training_service, sample_ids, args.parent, hyperparams)
    )


async def _run(
    svc: TrainingService,
    samples: list[str],
    parent: str | None,
    hyperparams: dict,
) -> int:
    record = await svc.start_training(
        TrainingRequest(
            sample_run_ids=samples,
            parent_adapter_id=parent,
            hyperparams=hyperparams,
        ),
    )
    print(f"queued    {record.id}  samples={len(samples)}  parent={parent or '-'}")

    # Wait for completion or failure by polling the store. Keeps the CLI
    # operationally simple - no event-bus subscription gymnastics needed.
    while True:
        await asyncio.sleep(0.1)
        latest = svc.training_store.get_training_run(record.id)
        if latest is None:
            continue
        if latest.status in {TrainingStatus.completed, TrainingStatus.failed, TrainingStatus.cancelled}:
            break

    if latest.status == TrainingStatus.completed:
        print(f"completed {latest.id}  adapter={latest.adapter_out}  steps={len(latest.metrics)}")
        return 0

    print(f"FAILED    {latest.id}  error={latest.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
