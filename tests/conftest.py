from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from application.coordinator import LocalRolloutCoordinator
from application.event_bus import EventBus
from domain.events import DomainEvent
from domain.models import TaskSpec
from infrastructure.rewards.heuristic import HeuristicRewardPipeline
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.tools.local import LocalToolHarness
from tests._fakes.openai_compat import FakeOpenAI, text, tool_use


@dataclass
class _StubRepoRunner:
    """Test double for the repo_runner slot.

    Phase A removed SimulatedEnvironmentRunner + the EnvironmentRunner ABC,
    so the coordinator now requires either a verifiers external runner or
    a repo runner. Most coordinator tests exercise the in-house loop's
    lifecycle/budget/cancellation logic and don't need a real sandboxed
    repo - this stub is the minimum surface (create_task + step) the
    coordinator pokes at.
    """

    tasks: dict[str, TaskSpec] = field(default_factory=dict)

    def create_task(self, task: TaskSpec) -> TaskSpec:
        self.tasks[task.id] = task
        return task

    def step(self, task_id: str, action: str) -> str:
        return f"env:{task_id}:{action}"


def _scripted_repo_loop(turns: int = 3) -> FakeOpenAI:
    """FakeOpenAI scripted with N tool-call turns followed by a finish.

    Phase B replaced StaticPolicyServer with run_repo_rollout, which talks
    to a real OpenAI client. Tests now drive that loop via a FakeOpenAI
    that scripts deterministic responses; this helper produces the
    boilerplate sequence that most coordinator tests need.
    """
    responses = [
        tool_use(f"call_{i}", "list_files", {"path": "."}) for i in range(turns)
    ]
    responses.append(text("done"))
    return FakeOpenAI(responses)


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def coordinator(tmp_path: Path, event_bus: EventBus) -> LocalRolloutCoordinator:
    return LocalRolloutCoordinator(
        tool_harness=LocalToolHarness(root=tmp_path),
        reward_pipeline=HeuristicRewardPipeline(),
        artifact_store=InMemoryArtifactStore(),
        event_bus=event_bus,
        workspace_root=tmp_path,
        max_parallel=2,
        repo_runner=_StubRepoRunner(),  # type: ignore[arg-type]
        policy_client=_scripted_repo_loop(turns=3),  # type: ignore[arg-type]
        policy_model="gpt-5.4-mini",
    )


def _is_localhost(url: str | None) -> bool:
    if not url:
        # Empty/None means OpenAI's SDK default = api.openai.com. Refuse.
        return False
    lowered = url.lower()
    return (
        "127.0.0.1" in lowered
        or "localhost" in lowered
        or lowered.startswith("stub://")
    )


@pytest.fixture(autouse=True)
def _block_paid_openai_outside_smoke(request, monkeypatch):
    """Phase C paranoid safety net.

    Any test outside `tests/smoke/` that constructs `openai.OpenAI`
    against a non-localhost base_url fails loudly. The repo's policy is
    *no test inference, ever* against paid providers; this is the
    automated check that nobody ever silently regresses it. Smoke tests
    opt in to real APIs via their own gating (RL_*_SMOKE env vars).
    """
    path = str(getattr(request.node, "path", "") or request.node.fspath)
    if "tests/smoke/" in path.replace("\\", "/"):
        return

    import openai

    original = openai.OpenAI.__init__

    def _guarded(self, *args, **kwargs):
        base_url = kwargs.get("base_url")
        if not _is_localhost(base_url):
            raise AssertionError(
                "Test "
                f"{request.node.nodeid} attempted to construct "
                f"openai.OpenAI(base_url={base_url!r}); paid-provider "
                "calls are blocked outside tests/smoke/. Use FakeOpenAI "
                "or pre-populate coordinator._client_cache instead.",
            )
        return original(self, *args, **kwargs)

    monkeypatch.setattr("openai.OpenAI.__init__", _guarded)


@pytest.fixture
async def drain(event_bus: EventBus) -> list[DomainEvent]:
    import asyncio
    import contextlib

    captured: list[DomainEvent] = []

    async def pump():
        async for event in event_bus.subscribe(replay=False):
            captured.append(event)

    task = asyncio.create_task(pump())
    yield captured
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
