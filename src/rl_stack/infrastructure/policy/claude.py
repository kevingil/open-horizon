"""Claude-backed PolicyServer using native tool use.

Wire-level flow per coordinator step:
  1. Coordinator calls generate_action(task, context)
  2. This server appends the latest environment observation (from context[-1])
     to the per-task conversation as a tool_result, then calls Anthropic.
  3. The model either emits tool_use (we serialize as action JSON) or plain
     text (we treat as an implicit finish with the text as summary).
Token usage accumulates per task; cumulative_cost_usd() is read by the
coordinator after each step to update the RunManifest.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

from ...domain.contracts import PolicyServer
from ...domain.models import TaskSpec
from ...domain.pricing import estimate_cost_usd
from ..environment.tools import tool_definitions

log = structlog.get_logger(__name__)


class AnthropicLike(Protocol):
    """Minimal shape of anthropic.Anthropic used here; lets us inject fakes."""

    class _Messages(Protocol):
        def create(self, **kwargs: Any) -> Any: ...

    messages: _Messages


@dataclass
class _TaskContext:
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_tool_use_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class ClaudePolicyServer(PolicyServer):
    client: AnthropicLike
    model: str = "claude-haiku-4-5"
    max_output_tokens: int = 2048
    max_retries: int = 3
    system_prompt: str = (
        "You are a software-engineering agent working inside a sandboxed "
        "repository snapshot. Use the provided tools to inspect the repo, "
        "run read-only commands, and report findings. Call `finish` when the "
        "task's success criteria are met. Prefer small, observable steps."
    )
    contexts: dict[str, _TaskContext] = field(default_factory=dict)

    def policy_name(self) -> str:
        return f"claude:{self.model}"

    def begin_task(self, task: TaskSpec) -> None:
        self.contexts[task.id] = _TaskContext(
            messages=[{"role": "user", "content": _initial_user_prompt(task)}],
        )

    def generate_action(self, task: TaskSpec, context: list[str]) -> str:
        ctx = self.contexts.get(task.id)
        if ctx is None:
            self.begin_task(task)
            ctx = self.contexts[task.id]
        else:
            # The coordinator just handed us the latest observation string.
            if context and ctx.last_tool_use_id is not None:
                ctx.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": ctx.last_tool_use_id,
                                "content": context[-1],
                            }
                        ],
                    }
                )
                ctx.last_tool_use_id = None

        response = self._call_with_retry(ctx)
        self._absorb_usage(ctx, response)

        tool_call = _extract_tool_call(response)
        if tool_call is None:
            summary = _extract_text(response) or "done"
            ctx.last_tool_use_id = None
            return json.dumps({"tool": "finish", "input": {"summary": summary}})

        ctx.messages.append({"role": "assistant", "content": _assistant_content(response)})
        ctx.last_tool_use_id = tool_call["id"]
        return json.dumps({"tool": tool_call["name"], "input": tool_call["input"]})

    def cumulative_cost_usd(self, task_id: str) -> float:
        ctx = self.contexts.get(task_id)
        if ctx is None:
            return 0.0
        return estimate_cost_usd(
            self.model,
            input_tokens=ctx.input_tokens,
            output_tokens=ctx.output_tokens,
            cache_read_tokens=ctx.cache_read_tokens,
            cache_write_tokens=ctx.cache_write_tokens,
        )

    def total_tokens(self, task_id: str) -> int:
        ctx = self.contexts.get(task_id)
        if ctx is None:
            return 0
        return ctx.input_tokens + ctx.output_tokens + ctx.cache_read_tokens + ctx.cache_write_tokens

    def _call_with_retry(self, ctx: _TaskContext) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": self.system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=tool_definitions(),
                    messages=ctx.messages,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = min(8.0, 0.5 * (2**attempt))
                log.warning("claude.retry", attempt=attempt, delay=delay, error=str(exc))
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _absorb_usage(self, ctx: _TaskContext, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        ctx.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        ctx.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        ctx.cache_read_tokens += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        ctx.cache_write_tokens += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)


def _initial_user_prompt(task: TaskSpec) -> str:
    criteria = "\n".join(f"- {c}" for c in task.success_criteria) or "- (no explicit criteria)"
    return (
        f"Task: {task.prompt}\n\n"
        f"Success criteria:\n{criteria}\n\n"
        f"Horizon: up to {task.horizon} tool calls. Call `finish` when done."
    )


def _assistant_content(response: Any) -> list[dict[str, Any]]:
    """Serialize response.content blocks back into the message form we send."""
    content: list[dict[str, Any]] = []
    for block in getattr(response, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            content.append({"type": "text", "text": getattr(block, "text", "")})
        elif btype == "tool_use":
            content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": getattr(block, "input", {}) or {},
                }
            )
    return content


def _extract_tool_call(response: Any) -> dict[str, Any] | None:
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            return {
                "id": block.id,
                "name": block.name,
                "input": getattr(block, "input", {}) or {},
            }
    return None


def _extract_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(parts).strip()
