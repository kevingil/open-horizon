"""Rescore a stored run against a registered rubric.

Usage (installed as a console script):
    rl-replay --run run-abc123 --rubric heuristic-v1
    rl-replay --list                 # show available rubrics
    rl-replay --run run-abc123 --rubric coding-v1 --dry-run

The command is a thin wrapper around application.rescore.rescore_run so the
CLI and the HTTP endpoint share their semantics.
"""
from __future__ import annotations

import argparse
import sys

from ...application.rescore import rescore_run
from ...bootstrap import build_application_services
from ...domain.rewards import RUBRICS
from ...settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rl_stack.cli.replay")
    parser.add_argument("--run", help="run id to rescore")
    parser.add_argument("--rubric", help="rubric provenance (e.g. heuristic-v1)")
    parser.add_argument("--dry-run", action="store_true", help="print diff, do not persist")
    parser.add_argument("--list", action="store_true", help="list registered rubrics and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(RUBRICS):
            signals = ", ".join(fn.__name__ for fn in RUBRICS[name].signals)
            print(f"{name:20s}  {signals}")
        return 0

    if not args.run or not args.rubric:
        parser.error("--run and --rubric are required unless --list is passed")

    services = build_application_services(Settings())
    try:
        result = rescore_run(
            services.artifact_store, args.run, args.rubric, persist=not args.dry_run,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    action = "dry-run" if args.dry_run else "saved"
    print(f"run    {result.run_id}")
    print(f"rubric {result.rubric} ({action})")
    print(f"  was: {result.previous_reward.terminal_reward:+.4f}  provenance={result.previous_reward.provenance[:60]}")
    print(f"  now: {result.new_reward.terminal_reward:+.4f}  delta={result.delta:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
