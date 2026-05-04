"""Per-turn OpenAI tool-call loop for the in-house repo path.

Replaces the OpenAICompatPolicyServer class with a stateless procedure.
The verifiers backend handles long-horizon agentic envs end-to-end via
Environment.run_rollout; this loop covers the one thing verifiers can't
help with - sandboxed shell access against a snapshotted git checkout.

Wire-level flow per turn:
  1. Send accumulated messages + the tool definitions to the model.
  2. Parse the response: a tool_calls entry becomes our
     {"tool": ..., "input": ...} action JSON; bare text is an implicit
     finish with the text as summary.
  3. Hand the action to the repo runner; record the observation as a
     tool-role message tied to the tool_call_id and loop.

Token accounting handles three real shapes:
  - Anthropic-via-compat top-level cache_creation_input_tokens /
    cache_read_input_tokens (folded into prompt_tokens by the proxy).
  - OpenAI prompt_tokens_details.cached_tokens (also folded in).
  - OpenAI completion_tokens_details.reasoning_tokens (folded into
    completion_tokens at the output rate; tracked for visibility).
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

import structlog

from domain.models import (
    RewardRecord,
    TaskSpec,
    TrajectoryRecord,
    TrajectoryStep,
)
from domain.pricing import estimate_cost_usd
from infrastructure.environment.tools import openai_tool_definitions

log = structlog.get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a software-engineering agent working inside a sandboxed "
    "repository snapshot. Use the provided tools to inspect the repo, "
    "run read-only commands, and report findings. Call `finish` when the "
    "task's success criteria are met. Prefer small, observable steps."
)


class _Completions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    completions: _Completions


class OpenAILike(Protocol):
    """The slice of openai.OpenAI we depend on; lets us inject fakes."""

    chat: _Chat


@dataclass
class RepoRolloutOutcome:
    """Result of a repo-runner-backed rollout.

    Mirrors VerifiersRolloutOutcome's shape so the coordinator's two
    branches can finish on the same persistence + event-emission code.
    Reward is left to the coordinator's RewardPipeline (verifiers-backed
    rollouts ship their own reward; this loop doesn't).
    """

    steps: list[TrajectoryStep]
    trajectory: TrajectoryRecord
    reward: RewardRecord | None  # filled by the coordinator's pipeline
    total_tokens: int
    cost_usd: float
    policy_name: str
    cancelled: bool = False
    token_overflow: bool = False


@dataclass
class _LoopState:
    """Per-rollout running state - replaces the per-task contexts dict
    that lived on OpenAICompatPolicyServer."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    last_tool_call_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


async def run_repo_rollout(
    *,
    client: OpenAILike,
    model: str,
    task: TaskSpec,
    repo_runner: Any,            # duck-typed: needs create_task + step
    record_command: Callable[[str], Any],  # tool_harness.record_command
    horizon: int,
    max_tokens_per_run: int,
    max_output_tokens: int = 2048,
    max_retries: int = 3,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    extra_body: dict[str, Any] | None = None,
    on_step: Callable[[TrajectoryStep], Awaitable[None]] | None = None,
    on_progress: Callable[[int, int, float, str | None], Awaitable[None]] | None = None,
    on_turn: (
        Callable[[int, str, str, dict[str, Any]], Awaitable[None]] | None
    ) = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> RepoRolloutOutcome:
    """Drive an OpenAI-compat tool-call loop turn-by-turn against a repo runner.

    Parameters mirror the old OpenAICompatPolicyServer's knobs but with
    callbacks for events and cancellation, since the coordinator no
    longer drives the per-turn loop itself.

    `on_step` fires once per recorded TrajectoryStep (action + observation).
    `on_progress` fires once per turn with (turn_index, tokens, cost_usd,
    tool_name). `on_turn` fires once per turn with (turn_index,
    prompt_text, completion_text, sampling_args) - that's the trainer-
    ready signal Phase E's TrainingRecorder consumes. `is_cancelled` is
    polled before and after each turn.

    Returns a RepoRolloutOutcome carrying the trajectory, token totals,
    and a string policy name for the manifest (`openai:<model>`). The
    coordinator scores reward via its RewardPipeline on top.
    """
    repo_runner.create_task(task)

    state = _LoopState(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _initial_user_prompt(task)},
        ],
    )
    steps: list[TrajectoryStep] = []
    errors: list[str] = []
    cancelled = False
    token_overflow = False

    sampling_args = {"max_tokens": max_output_tokens}
    if extra_body:
        sampling_args = {**sampling_args, **extra_body}

    for turn in range(horizon):
        if is_cancelled and is_cancelled():
            errors.append("cancelled")
            cancelled = True
            break

        # Snapshot the message context BEFORE _generate_action mutates it -
        # this is the prompt the recorder needs to pair with the completion.
        prompt_snapshot = json.dumps(state.messages, default=str)

        action = await asyncio.to_thread(
            _generate_action, client, model, state, max_output_tokens,
            max_retries, extra_body or {},
        )
        observation = repo_runner.step(task.id, action)
        record_command(action)

        action_step = TrajectoryStep(
            index=turn * 2, actor="policy", kind="action", content=action,
            has_training_metadata=on_turn is not None,
        )
        obs_step = TrajectoryStep(
            index=turn * 2 + 1, actor="environment", kind="observation",
            content=observation,
        )
        steps.extend([action_step, obs_step])
        if on_step is not None:
            await on_step(action_step)
            await on_step(obs_step)
        if on_turn is not None:
            await on_turn(turn, prompt_snapshot, action, sampling_args)

        tokens = _total_tokens(state)
        cost = estimate_cost_usd(
            model,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            cache_read_tokens=state.cache_read_tokens,
            cache_write_tokens=state.cache_write_tokens,
        )
        if on_progress is not None:
            await on_progress(turn, tokens, cost, _tool_from(action))

        # Thread the observation back as a tool message for the next turn.
        if state.last_tool_call_id is not None:
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": state.last_tool_call_id,
                    "content": observation,
                },
            )
            state.last_tool_call_id = None

        if is_cancelled and is_cancelled():
            errors.append("cancelled")
            cancelled = True
            break

        if _is_finish(action):
            break

        if tokens > max_tokens_per_run:
            errors.append(f"token budget exceeded: {tokens} > {max_tokens_per_run}")
            token_overflow = True
            break

    trajectory = TrajectoryRecord(
        id=f"traj-{uuid4().hex[:8]}",
        task_id=task.id,
        steps=steps,
        errors=errors,
    )
    return RepoRolloutOutcome(
        steps=steps,
        trajectory=trajectory,
        reward=None,
        total_tokens=_total_tokens(state),
        cost_usd=estimate_cost_usd(
            model,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            cache_read_tokens=state.cache_read_tokens,
            cache_write_tokens=state.cache_write_tokens,
        ),
        policy_name=f"openai:{model}",
        cancelled=cancelled,
        token_overflow=token_overflow,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _generate_action(
    client: OpenAILike,
    model: str,
    state: _LoopState,
    max_output_tokens: int,
    max_retries: int,
    extra_body: dict[str, Any],
) -> str:
    response = _call_with_retry(client, model, state, max_output_tokens, max_retries, extra_body)
    _absorb_usage(state, response)

    message = _extract_message(response)
    tool_call = _extract_tool_call(message)
    if tool_call is None:
        summary = _extract_text(message) or "done"
        state.last_tool_call_id = None
        return json.dumps({"tool": "finish", "input": {"summary": summary}})

    state.messages.append(_assistant_turn(message, tool_call))
    state.last_tool_call_id = tool_call["id"]
    return json.dumps({"tool": tool_call["name"], "input": tool_call["input"]})


def _call_with_retry(
    client: OpenAILike,
    model: str,
    state: _LoopState,
    max_output_tokens: int,
    max_retries: int,
    extra_body: dict[str, Any],
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": state.messages,
                "tools": openai_tool_definitions(),
                "tool_choice": "auto",
                "max_tokens": max_output_tokens,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = min(8.0, 0.5 * (2**attempt))
            log.warning("llm.retry", attempt=attempt, delay=delay, error=str(exc))
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _absorb_usage(state: _LoopState, response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    cache_write = int(_pick(usage, "cache_creation_input_tokens") or 0)
    cache_read_anthropic = int(_pick(usage, "cache_read_input_tokens") or 0)

    details_in = getattr(usage, "prompt_tokens_details", None)
    cache_read_openai = int(_pick(details_in, "cached_tokens") or 0) if details_in else 0

    details_out = getattr(usage, "completion_tokens_details", None)
    reasoning = int(_pick(details_out, "reasoning_tokens") or 0) if details_out else 0

    cache_read = cache_read_anthropic + cache_read_openai
    state.input_tokens += max(0, prompt_tokens - cache_read - cache_write)
    state.output_tokens += completion_tokens
    state.cache_read_tokens += cache_read
    state.cache_write_tokens += cache_write
    state.reasoning_tokens += reasoning


def _pick(obj: Any, attr: str) -> Any:
    """Tolerant getter that handles both attribute access (Pydantic model)
    and dict-style access (raw response or proxy)."""
    if obj is None:
        return None
    val = getattr(obj, attr, None)
    if val is None and isinstance(obj, dict):
        val = obj.get(attr)
    return val


def _total_tokens(state: _LoopState) -> int:
    return (
        state.input_tokens
        + state.output_tokens
        + state.cache_read_tokens
        + state.cache_write_tokens
    )


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


def _is_finish(action: str) -> bool:
    payload = _maybe_json(action)
    if payload is None:
        return False
    name = payload.get("tool") or payload.get("name")
    return name == "finish"


def _tool_from(action: str) -> str | None:
    payload = _maybe_json(action)
    if payload is None:
        return None
    value = payload.get("tool") or payload.get("name")
    return value if isinstance(value, str) else None


def _maybe_json(action: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(action)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
