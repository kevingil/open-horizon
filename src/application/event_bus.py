from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator

from domain.events import DomainEvent


class EventBus:
    """In-process async pub/sub with bounded per-subscriber queues and replay buffer.

    Subscribers that can't keep up drop oldest events on their own queue, so a slow
    WebSocket client never blocks the coordinator.
    """

    def __init__(self, *, replay_size: int = 200, subscriber_queue_size: int = 256) -> None:
        self._subscribers: list[asyncio.Queue[DomainEvent]] = []
        self._replay: deque[DomainEvent] = deque(maxlen=replay_size)
        self._subscriber_queue_size = subscriber_queue_size
        self._lock = asyncio.Lock()

    async def publish(self, event: DomainEvent) -> None:
        self._replay.append(event)
        dead: list[asyncio.Queue[DomainEvent]] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    dead.append(queue)
        if dead:
            async with self._lock:
                for queue in dead:
                    if queue in self._subscribers:
                        self._subscribers.remove(queue)

    def recent(self) -> list[DomainEvent]:
        return list(self._replay)

    async def subscribe(self, *, replay: bool = True) -> AsyncIterator[DomainEvent]:
        queue: asyncio.Queue[DomainEvent] = asyncio.Queue(maxsize=self._subscriber_queue_size)
        async with self._lock:
            self._subscribers.append(queue)
        try:
            if replay:
                for event in list(self._replay):
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        break
            while True:
                yield await queue.get()
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
