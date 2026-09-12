"""NDJSON request loop.

Protocol (one JSON object per line):

    -> {"id": "r1", "op": "verifiers.rollout", "params": {...}}
    <- {"id": "r1", "event": "metric", "data": {...}}     zero or more
    <- {"id": "r1", "ok": true, "result": {...}}
    <- {"id": "r1", "ok": false, "error": "..."}

Requests run concurrently: async ops (verifiers) share the event loop;
blocking ops (training, tokenizing) run in worker threads. stdout writes
are serialised so interleaved events never tear a line.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import traceback
from collections.abc import Callable
from typing import Any

log = logging.getLogger("horizon_bridge")

_WRITE_LOCK = threading.Lock()


def emit(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, default=str)
    with _WRITE_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _event_sink(rid: str) -> Callable[[str, dict[str, Any]], None]:
    def sink(name: str, data: dict[str, Any]) -> None:
        emit({"id": rid, "event": name, "data": data})

    return sink


async def handle(req: dict[str, Any]) -> None:
    rid = str(req.get("id", ""))
    op = str(req.get("op", ""))
    params = req.get("params") or {}
    sink = _event_sink(rid)
    try:
        result = await dispatch(op, params, sink)
        emit({"id": rid, "ok": True, "result": result})
    except Exception as exc:  # every failure must reach Rust
        log.debug("op %s failed:\n%s", op, traceback.format_exc())
        emit({"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})


async def dispatch(op: str, params: dict[str, Any], sink: Callable[[str, dict[str, Any]], None]) -> Any:
    if op == "ping":
        from horizon_bridge import __version__

        return {"pong": True, "version": __version__, "python": sys.version.split()[0]}
    if op == "verifiers.rollout":
        from horizon_bridge.verifiers_env import run_rollout

        return await run_rollout(params)
    if op == "train":
        from horizon_bridge.trainers import train

        return await asyncio.to_thread(train, params, sink)
    if op == "tokenize":
        from horizon_bridge.tokenize import tokenize_pair

        return await asyncio.to_thread(tokenize_pair, params)
    raise ValueError(f"unknown op: {op}")


async def serve(stdin=None) -> None:
    stdin = stdin or sys.stdin
    loop = asyncio.get_running_loop()
    pending: set[asyncio.Task] = set()
    while True:
        line = await loop.run_in_executor(None, stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            log.warning("dropping non-JSON request line")
            continue
        task = asyncio.create_task(handle(req))
        pending.add(task)
        task.add_done_callback(pending.discard)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="bridge %(levelname)s %(message)s")
    asyncio.run(serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
