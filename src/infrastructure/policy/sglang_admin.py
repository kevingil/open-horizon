"""Hot-reload trained LoRA adapters into a running SGLang server.

SGLang exposes admin endpoints to load and unload LoRA adapters without
restarting the server. We tail the event bus for `AdapterPublished` and
POST `{lora_name, lora_path}` to `/load_lora_adapter`. If the endpoint
isn't reachable or rejects the request we log and move on - the trained
adapter is still on disk, the server can pick it up at the next launch
via `--lora-paths`.

Stdlib-only (urllib) so no new runtime dep; the call runs in a worker
thread so it doesn't block the event loop.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

import structlog

from application.event_bus import EventBus
from domain.events import AdapterPublished

log = structlog.get_logger(__name__)


@dataclass
class SglangLoraReloader:
    """POSTs to SGLang's LoRA admin endpoints. Failures are logged, never raised."""

    admin_url: str
    timeout_s: float = 10.0

    def load(self, lora_name: str, lora_path: str) -> bool:
        url = self.admin_url.rstrip("/") + "/load_lora_adapter"
        return self._post_json(url, {"lora_name": lora_name, "lora_path": lora_path})

    def unload(self, lora_name: str) -> bool:
        url = self.admin_url.rstrip("/") + "/unload_lora_adapter"
        return self._post_json(url, {"lora_name": lora_name})

    def _post_json(self, url: str, payload: dict) -> bool:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                ok = 200 <= resp.status < 300
                if not ok:
                    log.warning("sglang.lora.bad_status", url=url, status=resp.status)
                return ok
        except urllib.error.URLError as exc:
            log.warning("sglang.lora.network_error", url=url, error=str(exc))
            return False
        except Exception as exc:
            log.warning("sglang.lora.error", url=url, error=str(exc))
            return False


def register_sglang_lora_autoreload(
    *,
    bus: EventBus,
    reloader: SglangLoraReloader,
) -> asyncio.Task:
    """Start a background task that hot-loads each newly published adapter
    into SGLang. Returns the task so the caller can hold a strong ref."""

    async def _consume() -> None:
        log.info("sglang.lora.autoreload.started", admin_url=reloader.admin_url)
        async for event in bus.subscribe(replay=False):
            if not isinstance(event, AdapterPublished):
                continue
            adapter = event.adapter
            ok = await asyncio.to_thread(reloader.load, adapter.id, adapter.path)
            if ok:
                log.info("sglang.lora.loaded", adapter_id=adapter.id, path=adapter.path)
            else:
                log.warning(
                    "sglang.lora.load_failed",
                    adapter_id=adapter.id,
                    path=adapter.path,
                    hint="verify SGLang was launched with --enable-lora",
                )

    task = asyncio.create_task(_consume(), name="sglang-lora-autoreload")

    def _on_done(t: asyncio.Task) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            exc = t.exception()
            if exc is not None:
                log.exception("sglang.lora.autoreload.crashed", error=str(exc))

    task.add_done_callback(_on_done)
    return task
