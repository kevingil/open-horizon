from __future__ import annotations

import asyncio

import pytest

from rl_stack.application.event_bus import EventBus
from rl_stack.domain.events import LogLine


@pytest.mark.asyncio
async def test_publish_fans_out_to_all_subscribers() -> None:
    bus = EventBus()
    seen_a: list = []
    seen_b: list = []

    async def collector(sink: list) -> None:
        async for event in bus.subscribe(replay=False):
            sink.append(event)
            if len(sink) == 2:
                return

    a = asyncio.create_task(collector(seen_a))
    b = asyncio.create_task(collector(seen_b))
    await asyncio.sleep(0)  # let subscribers register

    await bus.publish(LogLine(level="INFO", logger="test", message="one"))
    await bus.publish(LogLine(level="INFO", logger="test", message="two"))

    await asyncio.wait_for(asyncio.gather(a, b), timeout=1)
    assert [e.message for e in seen_a] == ["one", "two"]
    assert [e.message for e in seen_b] == ["one", "two"]


@pytest.mark.asyncio
async def test_replay_delivers_recent_events_on_subscribe() -> None:
    bus = EventBus()
    await bus.publish(LogLine(level="INFO", logger="test", message="past"))

    got: list = []

    async def collector() -> None:
        async for event in bus.subscribe(replay=True):
            got.append(event)
            return

    task = asyncio.create_task(collector())
    await asyncio.wait_for(task, timeout=1)
    assert got[0].message == "past"


@pytest.mark.asyncio
async def test_slow_subscriber_does_not_block_bus() -> None:
    bus = EventBus(subscriber_queue_size=2)

    async def slow() -> None:
        async for _ in bus.subscribe(replay=False):
            await asyncio.sleep(0.5)  # won't keep up

    slow_task = asyncio.create_task(slow())
    await asyncio.sleep(0)

    # Publishing more than queue size must not hang even if the slow consumer blocks.
    for i in range(10):
        await asyncio.wait_for(
            bus.publish(LogLine(level="INFO", logger="test", message=str(i))),
            timeout=0.2,
        )
    slow_task.cancel()
