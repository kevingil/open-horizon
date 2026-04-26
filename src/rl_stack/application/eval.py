"""Eval harness: run a fixed task set against a given adapter, aggregate
terminal rewards, persist an EvalReport, and update the adapter's eval_score.

The harness reuses the existing RolloutCoordinator so eval rollouts go
through the same policy / env / reward stack as training rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from ..domain.contracts import AdapterRegistry, RolloutCoordinator, TrainingStore
from ..domain.events import EvalCompleted
from ..domain.models import (
    EvalReport,
    EvalTaskScore,
    RolloutRequest,
)
from .event_bus import EventBus

log = structlog.get_logger(__name__)


class EvalTask(BaseModel):
    id: str
    prompt: str
    success_criteria: list[str] = Field(default_factory=list)
    horizon: int = Field(default=4, ge=1)


# Built-in eval tasks; small, repo-agnostic. Override per workspace by passing
# `tasks=` when calling EvalHarness.run().
DEFAULT_EVAL_TASKS: list[EvalTask] = [
    EvalTask(
        id="eval-readme",
        prompt="Read README.md and summarise the project.",
        success_criteria=["readme"],
    ),
    EvalTask(
        id="eval-list-files",
        prompt="List the files in the repository root.",
        success_criteria=["files"],
        horizon=3,
    ),
    EvalTask(
        id="eval-search-imports",
        prompt="Find Python files that import pydantic.",
        success_criteria=["pydantic"],
    ),
]


@dataclass
class EvalHarness:
    coordinator: RolloutCoordinator
    training_store: TrainingStore
    adapter_registry: AdapterRegistry
    event_bus: EventBus
    default_tasks: list[EvalTask] = field(default_factory=lambda: list(DEFAULT_EVAL_TASKS))
    task_set_name: str = "builtin"

    async def run(
        self,
        adapter_id: str,
        *,
        tasks: list[EvalTask] | None = None,
        task_set: str | None = None,
    ) -> EvalReport:
        adapter = self.adapter_registry.get(adapter_id)
        if adapter is None:
            raise ValueError(f"adapter not found: {adapter_id}")

        eval_tasks = tasks or self.default_tasks
        if not eval_tasks:
            raise ValueError("eval task set is empty")

        per_task: list[EvalTaskScore] = []
        for et in eval_tasks:
            request = RolloutRequest(
                prompt=et.prompt,
                horizon=et.horizon,
                success_criteria=et.success_criteria,
                adapter_id=adapter_id,
            )
            detail = await self.coordinator.start_rollout(request)
            per_task.append(
                EvalTaskScore(
                    task_id=et.id, terminal_reward=detail.reward.terminal_reward,
                )
            )

        mean = round(sum(p.terminal_reward for p in per_task) / len(per_task), 4)
        report = EvalReport(
            id=f"eval-{uuid4().hex[:8]}",
            adapter_id=adapter_id,
            task_set=task_set or self.task_set_name,
            mean_reward=mean,
            per_task=per_task,
        )
        self.training_store.save_eval_report(report)
        # Stamp eval_score on the adapter so dashboards can rank without
        # joining tables. Re-register via the registry; LocalAdapterRegistry
        # overwrites the manifest in place.
        updated = adapter.model_copy(update={"eval_score": mean})
        self.adapter_registry.register(updated)
        await self.event_bus.publish(EvalCompleted(report=report))
        log.info("eval.completed", adapter=adapter_id, mean=mean, tasks=len(per_task))
        return report
