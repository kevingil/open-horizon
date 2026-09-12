"""Protocol tests: drive the worker through a pipe exactly as Rust does."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from horizon_bridge import worker

ROOT = Path(__file__).resolve().parents[1]


def _spawn() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "horizon_bridge.worker"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": ""},
    )


def test_ping_and_unknown_op_over_pipe() -> None:
    proc = _spawn()
    assert proc.stdin and proc.stdout
    proc.stdin.write(json.dumps({"id": "r1", "op": "ping", "params": {}}) + "\n")
    proc.stdin.write(json.dumps({"id": "r2", "op": "nope", "params": {}}) + "\n")
    proc.stdin.flush()
    replies = {}
    for _ in range(2):
        msg = json.loads(proc.stdout.readline())
        replies[msg["id"]] = msg
    proc.stdin.close()
    proc.wait(timeout=10)
    assert replies["r1"]["ok"] is True
    assert replies["r1"]["result"]["pong"] is True
    assert replies["r2"]["ok"] is False
    assert "unknown op" in replies["r2"]["error"]


@pytest.mark.asyncio
async def test_dispatch_streams_events_and_result(monkeypatch) -> None:
    emitted: list[dict] = []
    monkeypatch.setattr(worker, "emit", emitted.append)

    async def fake_dispatch(op, params, sink):
        sink("metric", {"step": 0})
        return {"echo": params}

    monkeypatch.setattr(worker, "dispatch", fake_dispatch)
    await worker.handle({"id": "x", "op": "train", "params": {"a": 1}})
    assert emitted[0] == {"id": "x", "event": "metric", "data": {"step": 0}}
    assert emitted[1] == {"id": "x", "ok": True, "result": {"echo": {"a": 1}}}


@pytest.mark.asyncio
async def test_handle_reports_exceptions(monkeypatch) -> None:
    emitted: list[dict] = []
    monkeypatch.setattr(worker, "emit", emitted.append)

    async def boom(op, params, sink):
        raise RuntimeError("bad")

    monkeypatch.setattr(worker, "dispatch", boom)
    await worker.handle({"id": "y", "op": "train", "params": {}})
    assert emitted == [{"id": "y", "ok": False, "error": "RuntimeError: bad"}]
