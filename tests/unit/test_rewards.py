from __future__ import annotations

from rl_stack.domain.models import TaskSpec, ToolPermission, TrajectoryRecord, TrajectoryStep
from rl_stack.infrastructure.rewards.heuristic import HeuristicRewardPipeline


def _task(horizon: int = 4) -> TaskSpec:
    return TaskSpec(
        id="task-1",
        prompt="hello",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read],
        horizon=horizon,
        success_criteria=["none"],
    )


def test_terminal_reward_scales_with_step_count() -> None:
    pipeline = HeuristicRewardPipeline()
    traj = TrajectoryRecord(
        id="t1",
        task_id="task-1",
        steps=[TrajectoryStep(index=i, actor="a", kind="k", content="c") for i in range(2)],
    )
    reward = pipeline.score_trajectory(_task(horizon=4), traj)
    assert reward.terminal_reward == 0.5
    assert reward.provenance == "heuristic-local-v0"
    assert reward.audit_flags == []


def test_errors_penalise_reward_and_flag_audit() -> None:
    pipeline = HeuristicRewardPipeline()
    traj = TrajectoryRecord(
        id="t2",
        task_id="task-1",
        steps=[TrajectoryStep(index=0, actor="a", kind="k", content="c")],
        errors=["boom"],
    )
    reward = pipeline.score_trajectory(_task(horizon=4), traj)
    assert reward.penalties[0].code == "trajectory-errors"
    assert reward.audit_flags == ["requires-manual-audit"]
    assert reward.terminal_reward == 0.0


def test_reward_is_deterministic_for_identical_inputs() -> None:
    pipeline = HeuristicRewardPipeline()
    task = _task()
    traj = TrajectoryRecord(
        id="t3",
        task_id="task-1",
        steps=[TrajectoryStep(index=i, actor="a", kind="k", content="c") for i in range(3)],
    )
    a = pipeline.score_trajectory(task, traj)
    b = pipeline.score_trajectory(task, traj)
    assert a == b
