"""Opt-in smoke test: one real Claude rollout against a tiny workspace.

Enabled only when RL_SMOKE_API_KEY is set in the environment so the main suite
stays free and deterministic. Run manually or on a nightly job:

    RL_SMOKE_API_KEY=sk-ant-... pytest tests/smoke -v

Uses claude-haiku-4-5 on a 2-step horizon to keep cost well under $0.01.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

SMOKE_KEY = os.environ.get("RL_SMOKE_API_KEY")
pytestmark = pytest.mark.skipif(not SMOKE_KEY, reason="RL_SMOKE_API_KEY not set")


@pytest.mark.asyncio
async def test_real_haiku_rollout_completes(tmp_path: Path) -> None:
    from anthropic import Anthropic

    from rl_stack.application.coordinator import LocalRolloutCoordinator
    from rl_stack.application.event_bus import EventBus
    from rl_stack.domain.models import RolloutRequest, RunStatus
    from rl_stack.infrastructure.environment.repo_runner import RepoEnvironmentRunner
    from rl_stack.infrastructure.policy.claude import ClaudePolicyServer
    from rl_stack.infrastructure.rewards.composite import CompositeRewardPipeline
    from rl_stack.infrastructure.store.sqlite import SqliteArtifactStore
    from rl_stack.infrastructure.tools.local import LocalToolHarness

    source = tmp_path / "repo"
    source.mkdir()
    (source / "README.md").write_text("project: rl-smoke\nstatus: alive\n")

    coord = LocalRolloutCoordinator(
        environment_runner=RepoEnvironmentRunner(
            source_root=source, scratch_root=tmp_path / "scratch",
        ),
        tool_harness=LocalToolHarness(root=source),
        policy_server=ClaudePolicyServer(
            client=Anthropic(api_key=SMOKE_KEY), model="claude-haiku-4-5",
            max_output_tokens=512,
        ),
        reward_pipeline=CompositeRewardPipeline(),
        artifact_store=SqliteArtifactStore(path=tmp_path / "runs.db"),
        event_bus=EventBus(),
        workspace_root=source,
        max_parallel=1,
        max_tokens_per_run=5_000,
    )
    detail = await coord.start_rollout(
        RolloutRequest(
            prompt="Read README.md and report the project status.",
            horizon=3,
            success_criteria=["alive"],
        ),
    )
    assert detail.manifest.status == RunStatus.completed
    assert detail.manifest.estimated_cost_usd > 0
    assert detail.manifest.estimated_cost_usd < 0.01
