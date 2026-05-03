from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import structlog

from application.event_bus import EventBus
from domain.events import LogLine


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    """Configure structlog with contextvars bound automatically."""
    logging.basicConfig(level=level.upper(), format="%(message)s")

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class EventBusLogHandler:
    """Structlog processor that mirrors log events onto the event bus as LogLine.

    Installed after configure_logging() once a running loop + bus are available.
    Non-fatal on missing loop (e.g. when tests call logging from sync code).
    """

    def __init__(self, event_bus: EventBus, loop: asyncio.AbstractEventLoop) -> None:
        self._bus = event_bus
        self._loop = loop

    def __call__(self, logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        if self._loop.is_closed() or not self._loop.is_running():
            return event_dict
        with contextlib.suppress(Exception):
            message = str(event_dict.get("event", ""))
            level = str(event_dict.get("level", method_name)).upper()
            run_id = event_dict.get("run_id")
            context = {
                k: str(v)
                for k, v in event_dict.items()
                if k not in {"event", "level", "timestamp", "run_id", "task_id"}
            }
            line = LogLine(
                run_id=str(run_id) if run_id else None,
                level=level,
                logger=getattr(logger, "name", "distributed_rl"),
                message=message,
                context=context,
            )
            asyncio.run_coroutine_threadsafe(self._bus.publish(line), self._loop)
        return event_dict


def install_event_bus_handler(event_bus: EventBus, loop: asyncio.AbstractEventLoop) -> None:
    """Attach an EventBusLogHandler to the existing structlog chain."""
    current = structlog.get_config()
    processors = list(current["processors"])
    renderer = processors[-1]
    new_processors = [*processors[:-1], EventBusLogHandler(event_bus, loop), renderer]
    structlog.configure(
        processors=new_processors,
        wrapper_class=current["wrapper_class"],
        logger_factory=current["logger_factory"],
        cache_logger_on_first_use=False,
    )
