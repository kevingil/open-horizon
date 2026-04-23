from __future__ import annotations

from dataclasses import dataclass, field

from ...domain.contracts import EnvironmentRunner
from ...domain.models import TaskSpec


@dataclass
class SimulatedEnvironmentRunner(EnvironmentRunner):
    tasks: dict[str, TaskSpec] = field(default_factory=dict)
    logs: dict[str, list[str]] = field(default_factory=dict)

    def create_task(self, task: TaskSpec) -> TaskSpec:
        self.tasks[task.id] = task
        self.logs[task.id] = [f"created:{task.prompt}"]
        return task

    def step(self, task_id: str, action: str) -> str:
        message = f"env:{task_id}:{action}"
        self.logs.setdefault(task_id, []).append(message)
        return message

    def reset(self, task_id: str) -> None:
        self.logs[task_id] = ["reset"]
