"""SGLang LoRA hot-reload: reloader posts to admin endpoints, the
autoreload subscriber tails AdapterPublished events and forwards them."""
from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from rl_stack.application.event_bus import EventBus
from rl_stack.domain.events import AdapterPublished
from rl_stack.domain.models import AdapterRecord
from rl_stack.infrastructure.policy.sglang_admin import (
    SglangLoraReloader,
    register_sglang_lora_autoreload,
)


def test_reloader_posts_to_load_endpoint(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    reloader = SglangLoraReloader(admin_url="http://127.0.0.1:30000", timeout_s=2.0)
    assert reloader.load("adapter-aaa", "/path/to/adapter") is True
    assert captured["url"] == "http://127.0.0.1:30000/load_lora_adapter"
    assert captured["method"] == "POST"
    assert captured["body"] == {"lora_name": "adapter-aaa", "lora_path": "/path/to/adapter"}
    assert captured["headers"].get("Content-type") == "application/json"


def test_reloader_returns_false_on_network_error(monkeypatch):
    import urllib.error

    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    reloader = SglangLoraReloader(admin_url="http://nope:30000")
    assert reloader.load("adapter-aaa", "/path") is False
    assert reloader.unload("adapter-aaa") is False


@pytest.mark.asyncio
async def test_autoreload_subscriber_loads_on_publish() -> None:
    bus = EventBus()
    loaded: list[tuple[str, str]] = []

    class _CapturingReloader:
        admin_url = "http://fake"
        def load(self, name, path):
            loaded.append((name, path))
            return True

    task = register_sglang_lora_autoreload(bus=bus, reloader=_CapturingReloader())
    # Give the subscriber a tick to start tailing before we publish.
    await asyncio.sleep(0)
    adapter = AdapterRecord(
        id="adapter-xyz", base_model="Qwen/Qwen2.5-7B",
        path="/tmp/artifacts/adapters/adapter-xyz",
    )
    await bus.publish(AdapterPublished(adapter=adapter))
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert loaded == [("adapter-xyz", "/tmp/artifacts/adapters/adapter-xyz")]
