"""Opt-in smoke test: one real LLM rollout against a tiny workspace.

Enabled only when RL_SMOKE_API_KEY is set so the main suite stays free and
deterministic. Works with any OpenAI-compat provider via env vars:

    # Default: OpenAI proper
    RL_SMOKE_API_KEY=sk-... pytest tests/smoke -v

    # Local vLLM:
    RL_SMOKE_API_KEY=not-needed \\
    RL_SMOKE_BASE_URL=http://127.0.0.1:8000/v1 \\
    RL_SMOKE_MODEL=Qwen/Qwen3-8B \\
        pytest tests/smoke -v

Uses gpt-5.4-mini by default on a 3-step horizon. Cost is dominated by
input tokens; expect well under a cent per invocation on the default.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

SMOKE_KEY = os.environ.get("RL_SMOKE_API_KEY")
SMOKE_BASE_URL = os.environ.get("RL_SMOKE_BASE_URL", "https://api.openai.com/v1")
SMOKE_MODEL = os.environ.get("RL_SMOKE_MODEL", "gpt-5.4-mini")
pytestmark = pytest.mark.skipif(not SMOKE_KEY, reason="RL_SMOKE_API_KEY not set")


@pytest.mark.asyncio
async def test_real_llm_rollout_completes(tmp_path: Path) -> None:
    from openai import OpenAI

    from application.coordinator import LocalRolloutCoordinator
    from application.event_bus import EventBus
    from domain.models import RolloutRequest, RunStatus
    from infrastructure.environment.repo_runner import RepoEnvironmentRunner
    from infrastructure.policy.openai_compat import OpenAICompatPolicyServer
    from infrastructure.rewards.composite import CompositeRewardPipeline
    from infrastructure.store.sqlite import SqliteArtifactStore
    from infrastructure.tools.local import LocalToolHarness

    source = tmp_path / "repo"
    source.mkdir()
    (source / "README.md").write_text("project: rl-smoke\nstatus: alive\n")

    coord = LocalRolloutCoordinator(
        environment_runner=RepoEnvironmentRunner(
            source_root=source, scratch_root=tmp_path / "scratch",
        ),
        tool_harness=LocalToolHarness(root=source),
        policy_server=OpenAICompatPolicyServer(
            client=OpenAI(api_key=SMOKE_KEY, base_url=SMOKE_BASE_URL),
            model=SMOKE_MODEL,
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
