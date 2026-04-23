"""Rescore a stored run against a registered rubric.

Reward scoring is pure over (task, trajectory), so rescoring never re-executes
a rollout. We overwrite the stored RewardRecord in-place; the provenance JSON
already captures rubric+signal breakdown, so the previous state is fully
reconstructable by running rescore() again with any rubric.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..domain.contracts import ArtifactStore
from ..domain.models import RewardRecord, RunDetail
from ..domain.rewards import RubricSpec, get_rubric, score


@dataclass(frozen=True)
class RescoreResult:
    run_id: str
    rubric: str
    previous_reward: RewardRecord
    new_reward: RewardRecord

    @property
    def delta(self) -> float:
        return round(self.new_reward.terminal_reward - self.previous_reward.terminal_reward, 6)


def rescore_run(
    store: ArtifactStore,
    run_id: str,
    rubric: RubricSpec | str,
    *,
    persist: bool = True,
) -> RescoreResult:
    detail: RunDetail | None = store.get_run(run_id)
    if detail is None:
        raise ValueError(f"run not found: {run_id}")
    resolved = get_rubric(rubric) if isinstance(rubric, str) else rubric
    new_reward = score(detail.task, detail.trajectory, resolved)
    if persist:
        updated = detail.model_copy(update={"reward": new_reward})
        store.save_run(updated)
    return RescoreResult(
        run_id=run_id,
        rubric=resolved.provenance,
        previous_reward=detail.reward,
        new_reward=new_reward,
    )
