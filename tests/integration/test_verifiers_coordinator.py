"""End-to-end coordinator test for the env_backend=verifiers path.

Hands the coordinator a fake VerifiersRolloutRunner and asserts the same
lifecycle events fire as the in-house path, that the rubric-driven reward
is preserved end-to-end, and that the run is persisted with the verifiers
provenance shape the frontend already knows how to render.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from application.coordinator import LocalRolloutCoordinator
from application.event_bus import EventBus
from domain.events import DomainEvent
from domain.models import (
    RewardRecord,
    RolloutRequest,
    RunStatus,
    TaskSpec,
    TrajectoryRecord,
    TrajectoryStep,
)
from infrastructure.environment.verifiers_runner import (
    VerifiersRolloutOutcome,
)
from infrastructure.policy.static import StaticPolicyServer
from infrastructure.rewards.heuristic import HeuristicRewardPipeline
from infrastructure.store.memory import InMemoryArtifactStore
from infrastructure.tools.local import LocalToolHarness


class _FakeVerifiersRunner:
    """Stand-in for VerifiersRolloutRunner that the coordinator can drive
    without pulling in the real verifiers package."""

    def __init__(self, outcome: VerifiersRolloutOutcome, *, name: str = "verifiers:vf-math:fake") -> None:
        self._outcome = outcome
        self._name = name
        self.calls: list[str] = []

    def policy_name(self) -> str:
        return self._name

    async def run(self, task: TaskSpec, run_id: str) -> VerifiersRolloutOutcome:
        self.calls.append(run_id)
        return self._outcome


def _build_outcome(task_id: str = "task-x") -> VerifiersRolloutOutcome:
    steps = [
        TrajectoryStep(index=0, actor="user", kind="prompt", content="solve me"),
        TrajectoryStep(
            index=1, actor="policy", kind="action",
            content=json.dumps({"tool": "calculator", "input": {"expr": "2+2"}}),
        ),
        TrajectoryStep(index=2, actor="environment", kind="observation", content="4"),
        TrajectoryStep(index=3, actor="policy", kind="action", content="answer: 4"),
    ]
    traj = TrajectoryRecord(id="traj-x", task_id=task_id, steps=steps)
    reward = RewardRecord(
        trajectory_id=traj.id,
        terminal_reward=0.85,
        step_rewards=[],
        provenance=json.dumps({
            "source": "verifiers-rubric",
            "rubric": "verifiers-vf-math",
            "signals": [
                {"name": "format", "value": 1.0, "weight": 0.5, "reason": "tool used"},
                {"name": "correctness", "value": 0.7, "weight": 1.0, "reason": ""},
            ],
        }),
    )
    return VerifiersRolloutOutcome(
        steps=steps, trajectory=traj, reward=reward,
        total_tokens=75, cost_usd=0.0,
    )


def _make_coordinator(tmp_path: Path, runner: _FakeVerifiersRunner) -> LocalRolloutCoordinator:
    return LocalRolloutCoordinator(
        tool_harness=LocalToolHarness(root=tmp_path),
        policy_server=StaticPolicyServer(),
        reward_pipeline=HeuristicRewardPipeline(),
        artifact_store=InMemoryArtifactStore(),
        event_bus=EventBus(),
        workspace_root=tmp_path,
        max_parallel=2,
        external_rollout_runner=runner,  # type: ignore[arg-type]
    )


async def _collect_until_terminal(bus) -> tuple[list[DomainEvent], asyncio.Task]:
    got: list[DomainEvent] = []

    async def pump() -> None:
        async for event in bus.subscribe(replay=False):
            got.append(event)
            if event.kind in {"rollout.completed", "rollout.failed"}:
                return

    task = asyncio.create_task(pump())
    await asyncio.sleep(0)
    return got, task


@pytest.mark.asyncio
async def test_external_runner_drives_full_rollout(tmp_path: Path) -> None:
    runner = _FakeVerifiersRunner(_build_outcome())
    coord = _make_coordinator(tmp_path, runner)
    got, pump_task = await _collect_until_terminal(coord.event_bus)
    detail = await coord.start_rollout(RolloutRequest(prompt="p", horizon=4))
    await asyncio.wait_for(pump_task, timeout=2)

    assert runner.calls == [detail.manifest.id]
    assert detail.manifest.status == RunStatus.completed
    # The external runner's policy_name flows onto the manifest.
    assert detail.manifest.model_id == "verifiers:vf-math:fake"
    # The verifiers reward (0.85) survives end-to-end - the heuristic
    # pipeline is bypassed when external_rollout_runner is set.
    assert detail.reward.terminal_reward == pytest.approx(0.85)
    assert json.loads(detail.reward.provenance)["source"] == "verifiers-rubric"

    kinds = [e.kind for e in got]
    assert "rollout.started" in kinds
    assert "step.recorded" in kinds
    assert "reward.computed" in kinds
    assert "rollout.completed" in kinds


@pytest.mark.asyncio
async def test_external_runner_token_overflow_marks_failed(tmp_path: Path) -> None:
    outcome = _build_outcome()
    outcome.total_tokens = 999_999
    runner = _FakeVerifiersRunner(outcome)
    coord = _make_coordinator(tmp_path, runner)
    coord.max_tokens_per_run = 100  # force overflow
    detail = await coord.start_rollout(RolloutRequest(prompt="p", horizon=4))
    assert detail.manifest.status == RunStatus.failed
    assert any("token budget exceeded" in e for e in detail.trajectory.errors)
