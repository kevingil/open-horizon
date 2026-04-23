from __future__ import annotations

from dataclasses import dataclass

from ..contracts import PolicyServer
from ..models import TaskSpec


@dataclass
class StaticPolicyServer(PolicyServer):
    name: str = "debug-policy-v0"

    def policy_name(self) -> str:
        return self.name

    def generate_action(self, task: TaskSpec, context: list[str]) -> str:
        if not context:
            return f"inspect repo for task {task.id}"
        if len(context) < task.horizon // 2:
            return f"summarize and continue on {task.id}"
        return f"prepare artifact update for {task.id}"
