from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rl_stack.application.coordinator import LocalRolloutCoordinator
from rl_stack.application.event_bus import EventBus
from rl_stack.domain.models import RolloutRequest, RunStatus
from rl_stack.infrastructure.environment.repo_runner import RepoEnvironmentRunner
from rl_stack.infrastructure.policy.openai_compat import OpenAICompatPolicyServer
from rl_stack.infrastructure.rewards.composite import CompositeRewardPipeline
from rl_stack.infrastructure.store.memory import InMemoryArtifactStore
from rl_stack.infrastructure.tools.local import LocalToolHarness
from tests._fakes.openai_compat import FakeOpenAI, tool_use


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    src = tmp_path / "repo"
    src.mkdir()
    (src / "README.md").write_text("project: demo\n")
    return src


@pytest.fixture
def coordinator_factory(source_repo: Path, tmp_path: Path):
    def make(client: FakeOpenAI) -> LocalRolloutCoordinator:
        bus = EventBus()
        return LocalRolloutCoordinator(
            environment_runner=RepoEnvironmentRunner(
                source_root=source_repo,
                scratch_root=tmp_path / "scratch",
            ),
            tool_harness=LocalToolHarness(root=source_repo),
            policy_server=OpenAICompatPolicyServer(client=client, model="gpt-4o-mini"),
            reward_pipeline=CompositeRewardPipeline(),
            artifact_store=InMemoryArtifactStore(),
            event_bus=bus,
            workspace_root=source_repo,
            max_parallel=1,
            max_tokens_per_run=10_000,
        )

    return make


@pytest.mark.asyncio
async def test_openai_drives_real_rollout_to_finish(coordinator_factory) -> None:
    client = FakeOpenAI(
        [
            tool_use("call_1", "read_file", {"path": "README.md"}),
            tool_use("call_2", "finish", {"summary": "README surfaced"}),
        ]
    )
    coord = coordinator_factory(client)
    detail = await coord.start_rollout(
        RolloutRequest(prompt="summarise", horizon=6, success_criteria=["README surfaced"]),
    )

    assert detail.manifest.status == RunStatus.completed
    # Stopped early via finish: 2 action steps + 2 obs steps = 4.
    assert len(detail.trajectory.steps) == 4
    assert detail.manifest.estimated_cost_usd > 0
    # Second assistant turn must show the first observation as a tool message.
    second = client.calls[1]
    tool_msgs = [m for m in second["messages"] if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[-1]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_rollout_aborts_when_token_budget_exceeded(coordinator_factory) -> None:
    responses = [tool_use(f"call_{i}", "list_files", {}) for i in range(10)]
    client = FakeOpenAI(responses)
    coord = coordinator_factory(client)
    coord.max_tokens_per_run = 20

    detail = await coord.start_rollout(
        RolloutRequest(prompt="burn", horizon=10, success_criteria=["n/a"]),
    )
    assert detail.trajectory.errors
    assert "token budget" in detail.trajectory.errors[0]
    await asyncio.sleep(0)
