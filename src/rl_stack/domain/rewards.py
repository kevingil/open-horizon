"""Pure reward signal primitives. No I/O, deterministic, replayable.

Each reward function takes (TaskSpec, TrajectoryRecord) and returns a
RewardSignal: a named contribution with a value in [-1, 1] and a short
explanation. A CompositeRewardPipeline combines signals with weights and
produces the final RewardRecord + a list of per-signal breakdowns (stored
on the trajectory as summaries for now; a dedicated table lands when the
rubric registry grows up).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .models import RewardPenalty, RewardRecord, TaskSpec, TrajectoryRecord


@dataclass(frozen=True)
class RewardSignal:
    name: str
    value: float
    reason: str
    weight: float = 1.0


RewardFn = Callable[[TaskSpec, TrajectoryRecord], RewardSignal]


def step_count_signal(task: TaskSpec, trajectory: TrajectoryRecord, *, weight: float = 0.3) -> RewardSignal:
    step_count = len(trajectory.steps)
    value = min(1.0, step_count / max(task.horizon * 2, 1))
    return RewardSignal(
        name="step_count",
        value=value,
        reason=f"{step_count} trajectory steps against horizon={task.horizon}",
        weight=weight,
    )


def error_penalty_signal(task: TaskSpec, trajectory: TrajectoryRecord, *, weight: float = 1.0) -> RewardSignal:
    if not trajectory.errors:
        return RewardSignal(name="error_penalty", value=0.0, reason="no errors", weight=weight)
    return RewardSignal(
        name="error_penalty",
        value=-1.0,
        reason=f"{len(trajectory.errors)} errors: {trajectory.errors[0][:120]}",
        weight=weight,
    )


def success_criteria_signal(
    task: TaskSpec, trajectory: TrajectoryRecord, *, weight: float = 0.7
) -> RewardSignal:
    if not task.success_criteria:
        return RewardSignal(
            name="success_criteria", value=0.0, reason="no criteria specified", weight=weight,
        )
    haystack = _trajectory_text(trajectory).lower()
    matches = [c for c in task.success_criteria if c.strip().lower() in haystack]
    ratio = len(matches) / len(task.success_criteria)
    return RewardSignal(
        name="success_criteria",
        value=ratio,
        reason=f"matched {len(matches)}/{len(task.success_criteria)}: {matches}",
        weight=weight,
    )


def finish_signal(task: TaskSpec, trajectory: TrajectoryRecord, *, weight: float = 0.3) -> RewardSignal:
    for step in reversed(trajectory.steps):
        if step.actor != "policy":
            continue
        payload = _maybe_json(step.content)
        if payload and (payload.get("tool") or payload.get("name")) == "finish":
            return RewardSignal(
                name="finish", value=1.0, reason="policy called finish", weight=weight,
            )
        break
    return RewardSignal(name="finish", value=0.0, reason="no finish call", weight=weight)


def tests_pass_signal(
    task: TaskSpec, trajectory: TrajectoryRecord, *, weight: float = 0.8
) -> RewardSignal:
    """Scan tool observations for the last `pytest`/`python -m pytest` invocation
    and score by returncode. Neutral (0) if the policy never ran tests."""
    last_result: dict | None = None
    for step in trajectory.steps:
        if step.actor != "environment":
            continue
        payload = _maybe_json(step.content)
        if not payload or payload.get("tool") != "run_command":
            continue
        command = str(payload.get("command", ""))
        first = command.split(" ", 1)[0] if command else ""
        if first == "pytest" or command.startswith("python -m pytest") or command.startswith("python3 -m pytest"):
            last_result = payload
    if last_result is None:
        return RewardSignal(
            name="tests_pass", value=0.0, reason="no test run observed", weight=weight,
        )
    rc = last_result.get("returncode", 1)
    if rc == 0:
        return RewardSignal(
            name="tests_pass", value=1.0,
            reason=f"pytest exit 0 (sandbox={last_result.get('sandbox', 'unknown')})",
            weight=weight,
        )
    return RewardSignal(
        name="tests_pass", value=-1.0,
        reason=f"pytest exit {rc}",
        weight=weight,
    )


@dataclass(frozen=True)
class RubricSpec:
    name: str
    version: str
    signals: tuple[RewardFn, ...]

    @property
    def provenance(self) -> str:
        return f"{self.name}-{self.version}"


HEURISTIC_V1 = RubricSpec(
    name="heuristic",
    version="v1",
    signals=(
        step_count_signal,
        error_penalty_signal,
        success_criteria_signal,
        finish_signal,
    ),
)


CODING_V1 = RubricSpec(
    name="coding",
    version="v1",
    signals=(
        error_penalty_signal,
        success_criteria_signal,
        finish_signal,
        tests_pass_signal,
    ),
)


def score(task: TaskSpec, trajectory: TrajectoryRecord, rubric: RubricSpec) -> RewardRecord:
    """Run a rubric over a trajectory. Pure; safe to call in replay."""
    signals = [fn(task, trajectory) for fn in rubric.signals]
    weighted = [s.value * s.weight for s in signals]
    total_weight = sum(abs(s.weight) for s in signals) or 1.0
    terminal = max(-1.0, min(1.0, sum(weighted) / total_weight))

    penalties = [
        RewardPenalty(code=s.name, value=s.value * s.weight, reason=s.reason)
        for s in signals
        if s.value < 0
    ]
    audit_flags: list[str] = []
    if trajectory.errors:
        audit_flags.append("requires-manual-audit")
    if any(s.name == "success_criteria" and s.value == 0 for s in signals):
        audit_flags.append("no-success-criteria-match")

    step_count = max(len(trajectory.steps), 1)
    return RewardRecord(
        trajectory_id=trajectory.id,
        terminal_reward=round(terminal, 4),
        step_rewards=[round(terminal / step_count, 4)] * step_count,
        penalties=penalties,
        audit_flags=audit_flags,
        provenance=_encode_provenance(rubric, signals),
    )


def _encode_provenance(rubric: RubricSpec, signals: list[RewardSignal]) -> str:
    payload = {
        "rubric": rubric.provenance,
        "signals": [
            {"name": s.name, "value": round(s.value, 4), "weight": s.weight, "reason": s.reason}
            for s in signals
        ],
    }
    return json.dumps(payload)


def _trajectory_text(trajectory: TrajectoryRecord) -> str:
    return "\n".join(step.content for step in trajectory.steps)


def _maybe_json(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None
