from __future__ import annotations

from dataclasses import dataclass

from ...domain.contracts import RewardPipeline
from ...domain.models import RewardRecord, TaskSpec, TrajectoryRecord
from ...domain.rewards import HEURISTIC_V1, RubricSpec, score


@dataclass
class CompositeRewardPipeline(RewardPipeline):
    """RewardPipeline that runs a RubricSpec of pure signal functions.

    The rubric is value-like: swap it to iterate on reward design without
    rewriting the pipeline or touching the coordinator.
    """

    rubric: RubricSpec = HEURISTIC_V1

    def score_trajectory(self, task: TaskSpec, trajectory: TrajectoryRecord) -> RewardRecord:
        return score(task, trajectory, self.rubric)
