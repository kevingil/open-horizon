"""Unit tests for RoundScheduler and its concretes."""
from __future__ import annotations

import pytest

from domain.scheduling import (
    FixedRoundsScheduler,
    ScalingRoundsScheduler,
)


def test_fixed_returns_constant_regardless_of_step() -> None:
    sched = FixedRoundsScheduler(horizon=8)
    assert sched.current_horizon(training_step=0) == 8
    assert sched.current_horizon(training_step=1000) == 8


def test_scaling_linear_interpolates_between_endpoints() -> None:
    sched = ScalingRoundsScheduler(start=4, end=16, ramp_steps=100, mode="linear")
    # At 0 we're at start; at ramp_steps we're at end.
    assert sched.current_horizon(training_step=0) == 4
    assert sched.current_horizon(training_step=100) == 16
    # Halfway through the ramp -> halfway between 4 and 16 = 10.
    assert sched.current_horizon(training_step=50) == 10
    # Past the ramp clamps to end.
    assert sched.current_horizon(training_step=200) == 16


def test_scaling_step_makes_one_discrete_jump_at_midpoint() -> None:
    sched = ScalingRoundsScheduler(start=4, end=16, ramp_steps=100, mode="step")
    # First half stays at start
    assert sched.current_horizon(training_step=0) == 4
    assert sched.current_horizon(training_step=49) == 4
    # Crosses midpoint -> end
    assert sched.current_horizon(training_step=50) == 16
    assert sched.current_horizon(training_step=999) == 16


def test_scaling_clamps_negative_step_to_start() -> None:
    sched = ScalingRoundsScheduler(start=4, end=16, ramp_steps=100)
    assert sched.current_horizon(training_step=-5) == 4


@pytest.mark.parametrize("mode", ["linear", "step"])
def test_scaling_round_trip_to_end(mode: str) -> None:
    sched = ScalingRoundsScheduler(
        start=2, end=12, ramp_steps=10, mode=mode,  # type: ignore[arg-type]
    )
    # By the end of the ramp, both modes resolve to `end`.
    assert sched.current_horizon(training_step=10) == 12


def test_coordinator_falls_back_to_scheduler_when_request_omits_horizon(tmp_path) -> None:
    """Coordinator-level integration: when RolloutRequest.horizon is None,
    the configured scheduler decides. Explicit request.horizon overrides."""
    from application.coordinator import LocalRolloutCoordinator
    from application.event_bus import EventBus
    from domain.models import RolloutRequest
    from infrastructure.rewards.heuristic import HeuristicRewardPipeline
    from infrastructure.store.memory import InMemoryArtifactStore
    from infrastructure.tools.local import LocalToolHarness
    from tests.conftest import _scripted_repo_loop, _StubRepoRunner

    coord = LocalRolloutCoordinator(
        tool_harness=LocalToolHarness(root=tmp_path),
        reward_pipeline=HeuristicRewardPipeline(),
        artifact_store=InMemoryArtifactStore(),
        event_bus=EventBus(),
        workspace_root=tmp_path,
        max_parallel=1,
        repo_runner=_StubRepoRunner(),  # type: ignore[arg-type]
        policy_client=_scripted_repo_loop(turns=20),  # type: ignore[arg-type]
        round_scheduler=FixedRoundsScheduler(horizon=3),
    )
    # Omit horizon -> scheduler.current_horizon() = 3
    omitted = RolloutRequest(prompt="x")
    assert coord._resolve_horizon(omitted) == 3
    # Explicit horizon overrides
    explicit = RolloutRequest(prompt="x", horizon=9)
    assert coord._resolve_horizon(explicit) == 9
