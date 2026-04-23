from __future__ import annotations

from dataclasses import dataclass

from ..contracts import RewardPipeline
from ..models import RewardPenalty, RewardRecord, TaskSpec, TrajectoryRecord


@dataclass
class HeuristicRewardPipeline(RewardPipeline):
    provenance_label: str = "heuristic-local-v0"

    def score_trajectory(self, task: TaskSpec, trajectory: TrajectoryRecord) -> RewardRecord:
        step_count = len(trajectory.steps)
        penalties: list[RewardPenalty] = []
        terminal_reward = min(1.0, step_count / max(task.horizon, 1))
        if trajectory.errors:
            penalties.append(
                RewardPenalty(
                    code="trajectory-errors",
                    value=-0.25,
                    reason="Trajectory contains execution errors.",
                )
            )
            terminal_reward = max(0.0, terminal_reward - 0.25)
        return RewardRecord(
            trajectory_id=trajectory.id,
            terminal_reward=terminal_reward,
            step_rewards=[terminal_reward / max(step_count, 1)] * step_count,
            penalties=penalties,
            audit_flags=["requires-manual-audit"] if trajectory.errors else [],
            provenance=self.provenance_label,
        )
