"""Run the eval harness against an adapter from the command line.

Usage (installed as a console script):
    rl-eval --adapter adapter-abc123
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from ...bootstrap import build_application_services
from ...settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rl-eval")
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args(argv)

    services = build_application_services(Settings())
    return asyncio.run(_run(services.eval_harness, args.adapter))


async def _run(harness, adapter_id: str) -> int:
    try:
        report = await harness.run(adapter_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"adapter   {report.adapter_id}")
    print(f"task_set  {report.task_set}")
    print(f"mean      {report.mean_reward:+.4f}")
    for entry in report.per_task:
        print(f"  - {entry.task_id:24s} {entry.terminal_reward:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
