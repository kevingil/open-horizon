"""Opt-in smoke: drive verifiers env through a real SGLang server.

Skipped unless RL_VERIFIERS_SMOKE=1. Requires:
  - SGLang running and reachable at RL_VERIFIERS_SMOKE_BASE_URL
  - the `verifiers` package installed
  - the verifiers env id set in RL_VERIFIERS_SMOKE_ENV_ID (default vf-math)

    RL_VERIFIERS_SMOKE=1 \\
    RL_VERIFIERS_SMOKE_BASE_URL=http://127.0.0.1:30000/v1 \\
    RL_VERIFIERS_SMOKE_MODEL=Qwen/Qwen2.5-7B-Instruct \\
    RL_VERIFIERS_SMOKE_ENV_ID=vf-math \\
        pytest tests/smoke/test_verifiers_sglang.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ENABLED = os.environ.get("RL_VERIFIERS_SMOKE") == "1"
BASE_URL = os.environ.get("RL_VERIFIERS_SMOKE_BASE_URL", "http://127.0.0.1:30000/v1")
MODEL = os.environ.get("RL_VERIFIERS_SMOKE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ENV_ID = os.environ.get("RL_VERIFIERS_SMOKE_ENV_ID", "vf-math")
pytestmark = pytest.mark.skipif(not ENABLED, reason="RL_VERIFIERS_SMOKE not set")


@pytest.mark.asyncio
async def test_one_verifiers_rollout_via_sglang(tmp_path: Path) -> None:
    pytest.importorskip("verifiers")
    from openai import OpenAI

    from rl_stack.application.coordinator import LocalRolloutCoordinator
    from rl_stack.application.event_bus import EventBus
    from rl_stack.domain.models import RolloutRequest, RunStatus
    from rl_stack.infrastructure.environment.simulated import SimulatedEnvironmentRunner
    from rl_stack.infrastructure.environment.verifiers_runner import (
        VerifiersRolloutRunner,
    )
    from rl_stack.infrastructure.policy.static import StaticPolicyServer
    from rl_stack.infrastructure.rewards.composite import CompositeRewardPipeline
    from rl_stack.infrastructure.store.memory import InMemoryArtifactStore
    from rl_stack.infrastructure.tools.local import LocalToolHarness

    client = OpenAI(api_key="not-needed", base_url=BASE_URL)
    runner = VerifiersRolloutRunner(
        client=client,
        model=f"sglang:{MODEL}",
        env_id=ENV_ID,
        max_concurrent=1,
        rollout_timeout_s=120.0,
    )
    coord = LocalRolloutCoordinator(
        environment_runner=SimulatedEnvironmentRunner(),
        tool_harness=LocalToolHarness(root=tmp_path),
        policy_server=StaticPolicyServer(),
        reward_pipeline=CompositeRewardPipeline(),
        artifact_store=InMemoryArtifactStore(),
        event_bus=EventBus(),
        workspace_root=tmp_path,
        max_parallel=1,
        max_tokens_per_run=200_000,
        external_rollout_runner=runner,
    )
    detail = await coord.start_rollout(
        RolloutRequest(prompt="Solve: 17 + 25", horizon=8),
    )
    assert detail.manifest.status == RunStatus.completed
    assert detail.manifest.model_id.startswith("verifiers:")
    assert detail.trajectory.steps, "verifiers should produce a non-empty trajectory"
