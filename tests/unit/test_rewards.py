from __future__ import annotations

import json

from rl_stack.domain.models import TaskSpec, ToolPermission, TrajectoryRecord, TrajectoryStep
from rl_stack.domain.rewards import (
    CODING_V1,
    HEURISTIC_V1,
    error_penalty_signal,
    finish_signal,
    score,
    step_count_signal,
    success_criteria_signal,
    tests_pass_signal,
)
from rl_stack.infrastructure.rewards.composite import CompositeRewardPipeline
from rl_stack.infrastructure.rewards.heuristic import HeuristicRewardPipeline


def _task(horizon: int = 4, criteria: list[str] | None = None) -> TaskSpec:
    return TaskSpec(
        id="task-1",
        prompt="inspect README",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read],
        horizon=horizon,
        success_criteria=criteria if criteria is not None else ["README surfaced"],
    )


def _steps(contents: list[tuple[str, str]]) -> list[TrajectoryStep]:
    return [
        TrajectoryStep(index=i, actor=a, kind="k", content=c)
        for i, (a, c) in enumerate(contents)
    ]


# --- pure signal tests ---------------------------------------------------


def test_step_count_signal_scales_with_steps() -> None:
    traj = TrajectoryRecord(id="t", task_id="task-1", steps=_steps([("policy", "x")] * 4))
    sig = step_count_signal(_task(horizon=4), traj)
    assert 0 < sig.value <= 1
    assert sig.name == "step_count"


def test_error_penalty_fires_only_on_errors() -> None:
    traj = TrajectoryRecord(id="t", task_id="task-1", steps=[], errors=["boom"])
    sig = error_penalty_signal(_task(), traj)
    assert sig.value == -1.0

    clean = TrajectoryRecord(id="t", task_id="task-1", steps=[])
    assert error_penalty_signal(_task(), clean).value == 0.0


def test_success_criteria_matcher_reads_trajectory_text() -> None:
    traj = TrajectoryRecord(
        id="t", task_id="task-1",
        steps=_steps([
            ("policy", '{"tool":"read_file","input":{"path":"README.md"}}'),
            ("environment", '{"content":"README surfaced\\nhello"}'),
        ]),
    )
    sig = success_criteria_signal(_task(criteria=["README surfaced"]), traj)
    assert sig.value == 1.0


def test_finish_signal_detects_terminal_finish_call() -> None:
    finish_step = TrajectoryStep(
        index=0, actor="policy", kind="action",
        content=json.dumps({"tool": "finish", "input": {"summary": "done"}}),
    )
    traj = TrajectoryRecord(id="t", task_id="task-1", steps=[finish_step])
    assert finish_signal(_task(), traj).value == 1.0


def test_finish_signal_zero_when_last_policy_step_is_not_finish() -> None:
    step = TrajectoryStep(
        index=0, actor="policy", kind="action",
        content=json.dumps({"tool": "read_file", "input": {"path": "README.md"}}),
    )
    traj = TrajectoryRecord(id="t", task_id="task-1", steps=[step])
    assert finish_signal(_task(), traj).value == 0.0


# --- rubric composition tests -------------------------------------------


def test_rubric_is_deterministic_for_identical_inputs() -> None:
    traj = TrajectoryRecord(
        id="t", task_id="task-1",
        steps=_steps([("policy", "x"), ("environment", "y")]),
    )
    task = _task()
    a = score(task, traj, HEURISTIC_V1)
    b = score(task, traj, HEURISTIC_V1)
    assert a == b


def test_rubric_provenance_includes_signal_breakdown() -> None:
    traj = TrajectoryRecord(
        id="t", task_id="task-1",
        steps=_steps([("policy", json.dumps({"tool": "finish", "input": {"summary": "done"}}))]),
    )
    record = score(_task(), traj, HEURISTIC_V1)
    payload = json.loads(record.provenance)
    assert payload["rubric"] == "heuristic-v1"
    names = {s["name"] for s in payload["signals"]}
    assert names == {"step_count", "error_penalty", "success_criteria", "finish"}


def test_composite_pipeline_matches_score_function() -> None:
    traj = TrajectoryRecord(id="t", task_id="task-1", steps=_steps([("policy", "x")]))
    pipeline = CompositeRewardPipeline()
    assert pipeline.score_trajectory(_task(), traj) == score(_task(), traj, HEURISTIC_V1)


def test_legacy_heuristic_pipeline_still_works() -> None:
    """Keep the v0 pipeline so older manifests can be rescored unchanged."""
    traj = TrajectoryRecord(id="t", task_id="task-1", steps=_steps([("policy", "x")]))
    record = HeuristicRewardPipeline().score_trajectory(_task(), traj)
    assert record.provenance == "heuristic-local-v0"


# --- tests_pass signal ---------------------------------------------------


def _env_step(idx: int, payload: dict) -> TrajectoryStep:
    return TrajectoryStep(index=idx, actor="environment", kind="observation", content=json.dumps(payload))


def test_tests_pass_neutral_when_no_tests_run() -> None:
    traj = TrajectoryRecord(
        id="t", task_id="task-1",
        steps=_steps([("policy", json.dumps({"tool": "read_file", "input": {"path": "x"}}))]),
    )
    assert tests_pass_signal(_task(), traj).value == 0.0


def test_tests_pass_positive_on_green_pytest() -> None:
    traj = TrajectoryRecord(
        id="t", task_id="task-1",
        steps=[
            _env_step(0, {"tool": "run_command", "command": "pytest -x", "returncode": 0, "sandbox": "docker"}),
        ],
    )
    sig = tests_pass_signal(_task(), traj)
    assert sig.value == 1.0
    assert "docker" in sig.reason


def test_tests_pass_negative_on_red_pytest() -> None:
    traj = TrajectoryRecord(
        id="t", task_id="task-1",
        steps=[
            _env_step(0, {"tool": "run_command", "command": "pytest", "returncode": 1, "sandbox": "none"}),
        ],
    )
    assert tests_pass_signal(_task(), traj).value == -1.0


def test_tests_pass_uses_last_pytest_invocation() -> None:
    traj = TrajectoryRecord(
        id="t", task_id="task-1",
        steps=[
            _env_step(0, {"tool": "run_command", "command": "pytest", "returncode": 1, "sandbox": "none"}),
            _env_step(1, {"tool": "run_command", "command": "pytest", "returncode": 0, "sandbox": "none"}),
        ],
    )
    assert tests_pass_signal(_task(), traj).value == 1.0


def test_coding_v1_rubric_is_registered() -> None:
    assert CODING_V1.provenance == "coding-v1"
    names = [fn.__name__ for fn in CODING_V1.signals]
    assert "tests_pass_signal" in names
