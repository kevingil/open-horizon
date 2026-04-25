"""PolicyServer backed by an OpenAI Chat Completions endpoint.

Works with anything that speaks the OpenAI API: OpenAI proper, vLLM,
Ollama, OpenRouter, llama.cpp server, Anthropic via their compat surface.
The only knobs are `base_url`, `api_key`, and `model`.

Wire-level flow per coordinator step:
  1. Coordinator calls generate_action(task, context)
  2. We thread the latest environment observation (context[-1]) back as a
     `tool` role message tied to the previous tool_call_id.
  3. We send messages + tools and parse the response: a `tool_calls` entry
     becomes our `{"tool": ..., "input": ...}` action JSON; bare text is
     treated as an implicit `finish` with the text as summary.

Token usage accumulates per task; the coordinator reads cumulative_cost_usd
and total_tokens via the optional PolicyServer methods we expose.
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
from ..environment.tools import openai_tool_definitions

log = structlog.get_logger(__name__)


class _Completions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    completions: _Completions


class OpenAILike(Protocol):
    """The slice of openai.OpenAI we depend on; lets us inject fakes."""

    chat: _Chat


@dataclass
class _TaskContext:
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_tool_call_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class OpenAICompatPolicyServer(PolicyServer):
    client: OpenAILike
    model: str = "gpt-4o-mini"
    max_output_tokens: int = 2048
    max_retries: int = 3
    system_prompt: str = (
        "You are a software-engineering agent working inside a sandboxed "
        "repository snapshot. Use the provided tools to inspect the repo, "
        "run read-only commands, and report findings. Call `finish` when the "
        "task's success criteria are met. Prefer small, observable steps."
    )
    extra_body: dict[str, Any] = field(default_factory=dict)
    contexts: dict[str, _TaskContext] = field(default_factory=dict)

    def policy_name(self) -> str:
        return f"openai:{self.model}"

    def begin_task(self, task: TaskSpec) -> None:
        self.contexts[task.id] = _TaskContext(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": _initial_user_prompt(task)},
            ],
        )

    def generate_action(self, task: TaskSpec, context: list[str]) -> str:
        ctx = self.contexts.get(task.id)
        if ctx is None:
            self.begin_task(task)
            ctx = self.contexts[task.id]
        else:
            if context and ctx.last_tool_call_id is not None:
                ctx.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": ctx.last_tool_call_id,
                        "content": context[-1],
                    }
                )
                ctx.last_tool_call_id = None

        response = self._call_with_retry(ctx)
        self._absorb_usage(ctx, response)

        message = _extract_message(response)
        tool_call = _extract_tool_call(message)
        if tool_call is None:
            summary = _extract_text(message) or "done"
            ctx.last_tool_call_id = None
            return json.dumps({"tool": "finish", "input": {"summary": summary}})

        ctx.messages.append(_assistant_turn(message, tool_call))
        ctx.last_tool_call_id = tool_call["id"]
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
        )

    def total_tokens(self, task_id: str) -> int:
        ctx = self.contexts.get(task_id)
        if ctx is None:
            return 0
        return ctx.input_tokens + ctx.output_tokens + ctx.cache_read_tokens

    def _call_with_retry(self, ctx: _TaskContext) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": ctx.messages,
                    "tools": openai_tool_definitions(),
                    "tool_choice": "auto",
                    "max_tokens": self.max_output_tokens,
                }
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = min(8.0, 0.5 * (2**attempt))
                log.warning("llm.retry", attempt=attempt, delay=delay, error=str(exc))
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _absorb_usage(self, ctx: _TaskContext, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        ctx.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        ctx.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        # OpenAI: usage.prompt_tokens_details.cached_tokens (cache hits are
        # counted *inside* prompt_tokens, not added on top, so subtract them
        # to avoid double-billing).
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        if cached:
            ctx.cache_read_tokens += cached
            ctx.input_tokens = max(0, ctx.input_tokens - cached)


def _initial_user_prompt(task: TaskSpec) -> str:
    criteria = "\n".join(f"- {c}" for c in task.success_criteria) or "- (no explicit criteria)"
    return (
        f"Task: {task.prompt}\n\n"
        f"Success criteria:\n{criteria}\n\n"
        f"Horizon: up to {task.horizon} tool calls. Call `finish` when done."
    )


def _extract_message(response: Any) -> Any:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    return getattr(choices[0], "message", None)


def _extract_tool_call(message: Any) -> dict[str, Any] | None:
    if message is None:
        return None
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return None
    call = tool_calls[0]
    fn = getattr(call, "function", None)
    if fn is None:
        return None
    name = getattr(fn, "name", None)
    raw_args = getattr(fn, "arguments", "") or ""
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        args = {"_raw": raw_args}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return {"id": getattr(call, "id", ""), "name": name, "input": args}


def _extract_text(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            getattr(p, "text", "") for p in content if getattr(p, "type", None) == "text"
        ]
        return "\n".join(parts).strip()
    return ""


def _assistant_turn(message: Any, tool_call: dict[str, Any]) -> dict[str, Any]:
    """Echo the assistant's tool_calls back into our message history so the
    next turn's tool_result properly threads on `tool_call_id`."""
    return {
        "role": "assistant",
        "content": _extract_text(message) or None,
        "tool_calls": [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call["input"]),
                },
            }
        ],
    }
