"""Contract tests every AdapterRegistry impl must satisfy."""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from domain.contracts import AdapterRegistry
from domain.models import AdapterRecord
from infrastructure.adapters.local import LocalAdapterRegistry


def _record(adapter_id: str, *, parent: str | None = None, created_at: datetime | None = None) -> AdapterRecord:
    return AdapterRecord(
        id=adapter_id,
        parent_id=parent,
        base_model="vllm:Qwen/Qwen2.5-0.5B",
        path="(filled-by-registry)",
        created_at=created_at or datetime.now(UTC),
    )


def _local_factory(tmp_path_factory: pytest.TempPathFactory) -> Callable[[], AdapterRegistry]:
    base = tmp_path_factory.mktemp("local-registry")
    counter = {"n": 0}

    def make() -> AdapterRegistry:
        counter["n"] += 1
        return LocalAdapterRegistry(root=base / f"reg-{counter['n']}")

    return make


@pytest.fixture(params=["local"])
def registry(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> AdapterRegistry:
    if request.param == "local":
        return _local_factory(tmp_path_factory)()
    raise ValueError(request.param)


def test_register_then_get_roundtrip(registry: AdapterRegistry) -> None:
    saved = registry.register(_record("adapter-a"))
    assert registry.get("adapter-a") == saved
    assert registry.get("missing") is None


def test_register_normalises_path_to_registry_directory(registry: AdapterRegistry) -> None:
    saved = registry.register(_record("adapter-a"))
    assert Path(saved.path) == registry.path_for("adapter-a")


def test_list_adapters_sorted_most_recent_first(registry: AdapterRegistry) -> None:
    now = datetime.now(UTC)
    registry.register(_record("adapter-old", created_at=now - timedelta(hours=2)))
    registry.register(_record("adapter-new", created_at=now))
    listed = registry.list_adapters()
    assert [r.id for r in listed] == ["adapter-new", "adapter-old"]


def test_children_of_traces_lineage(registry: AdapterRegistry) -> None:
    registry.register(_record("root"))
    registry.register(_record("child-a", parent="root"))
    registry.register(_record("child-b", parent="root"))
    registry.register(_record("grandchild", parent="child-a"))

    assert {c.id for c in registry.children_of("root")} == {"child-a", "child-b"}
    assert {c.id for c in registry.children_of("child-a")} == {"grandchild"}
    assert registry.children_of("grandchild") == []


def test_register_overwrites_existing_manifest(registry: AdapterRegistry) -> None:
    registry.register(_record("adapter-a"))
    updated = registry.register(_record("adapter-a").model_copy(update={"eval_score": 0.8}))
    assert registry.get("adapter-a").eval_score == 0.8
    assert updated.eval_score == 0.8
    # No duplicate listing.
    assert sum(1 for r in registry.list_adapters() if r.id == "adapter-a") == 1
