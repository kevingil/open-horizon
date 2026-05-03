"""Rollout delegator for the `verifiers` framework.

`verifiers` owns its own rollout loop. Its public async entry point is
`Environment.run_rollout(input, client, model, sampling_args, ...)`,
which calls the abstract `rollout()` and then scores the rubric (when
`score_rollouts=True`). Returns a `RolloutOutput` carrying the populated
`State` (a dict subclass with `trajectory`, `completion`, `reward`,
`metrics`, `usage`, `error`, ...). We delegate to it directly rather
than driving step-by-step ourselves, since that would forfeit the
rubric grader, RLMEnv, and OpenEnv glue.

The verifiers package is an optional dependency; the import is lazy so
users on the default backend never need it installed.

References (verifiers >= 0.1.11):
  verifiers/envs/environment.py:Environment.run_rollout
  verifiers/types.py:State, TokenUsage, RolloutInput
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import structlog

from domain.models import (
    RewardPenalty,
    RewardRecord,
    TaskSpec,
    TrajectoryRecord,
    TrajectoryStep,
)
from domain.pricing import estimate_cost_usd

log = structlog.get_logger(__name__)


@dataclass
class VerifiersRolloutOutcome:
    steps: list[TrajectoryStep]
    trajectory: TrajectoryRecord
    reward: RewardRecord
    total_tokens: int
    cost_usd: float


@dataclass
class VerifiersRolloutRunner:
    """Drives a single verifiers env rollout end-to-end.

    Parameters mirror the env-driven side of verifiers' API:
      - `client`: an OpenAI-compatible client (vanilla `openai.OpenAI` works
        against SGLang / vLLM / Ollama / OpenAI proper).
      - `model`: the model id verifiers should pass through. We keep the
        provider prefix (`sglang:foo`, `vllm:foo`, ...) on the runner side so
        cost accounting stays $0 for self-hosted, and strip it when handing
        the id to verifiers.
      - `env_id`: the verifiers env to load (`vf-math`, `rlm`, ...).
      - `env_args`: kwargs forwarded to `vf.load_environment`.
      - `max_concurrent`: caps how many verifiers rollouts run concurrently
        across coordinator workers (separate from `max_parallel_rollouts`).
      - `rollout_timeout_s`: optional wallclock cap per rollout.
    """

    client: Any
    model: str
    env_id: str
    env_args: dict[str, Any] = field(default_factory=dict)
    max_concurrent: int = 4
    rollout_timeout_s: float | None = None

    _env: Any = field(default=None, init=False, repr=False)
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    def policy_name(self) -> str:
        return f"verifiers:{self.env_id}:{self.model}"

    def _ensure_env(self) -> Any:
        if self._env is not None:
            return self._env
        try:
            import verifiers as vf  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via integration env
            raise RuntimeError(
                "RL_ENV_BACKEND=verifiers requires the `verifiers` package. "
                "Install it with `pip install -e '.[envs]'` (after adding the "
                "envs extra) or `pip install verifiers`."
            ) from exc
        self._env = vf.load_environment(self.env_id, **self.env_args)
        return self._env

    async def run(self, task: TaskSpec, run_id: str) -> VerifiersRolloutOutcome:
        env = self._ensure_env()
        # Strip the provider prefix when calling verifiers; pricing keeps the
        # full id on our side so self-hosted stays $0.
        served_model = self.model.split(":", 1)[1] if ":" in self.model else self.model

        rollout_input = {
            "prompt": [{"role": "user", "content": task.prompt}],
            "task": task.id,
        }
        sampling_args = _sampling_args(task)

        async def _invoke() -> Any:
            # run_rollout is the scored entry point; it calls the abstract
            # rollout() and then `await self.rubric.score_rollout(state)`.
            return await env.run_rollout(
                input=rollout_input,
                client=self.client,
                model=served_model,
                sampling_args=sampling_args,
            )

        async with self._semaphore:
            try:
                if self.rollout_timeout_s is not None:
                    result = await asyncio.wait_for(_invoke(), timeout=self.rollout_timeout_s)
                else:
                    result = await _invoke()
            except TimeoutError as exc:
                raise RuntimeError(
                    f"verifiers rollout exceeded {self.rollout_timeout_s:.1f}s"
                ) from exc

        state = _state_of(result)
        errors = _collect_errors(state)
        steps = _messages_to_steps(_extract_messages(state))
        traj = TrajectoryRecord(
            id=f"traj-{uuid4().hex[:8]}",
            task_id=task.id,
            steps=steps,
            errors=errors,
        )
        terminal_reward, signals = _extract_reward(state)
        reward = _build_reward_record(traj.id, terminal_reward, signals, self.env_id)

        input_tokens, output_tokens = _extract_usage(state)
        total_tokens = input_tokens + output_tokens
        cost_usd = estimate_cost_usd(
            self.model, input_tokens=input_tokens, output_tokens=output_tokens,
        )
        log.info(
            "verifiers.rollout.completed",
            run_id=run_id,
            env=self.env_id,
            steps=len(steps),
            terminal_reward=terminal_reward,
            tokens=total_tokens,
        )
        return VerifiersRolloutOutcome(
            steps=steps,
            trajectory=traj,
            reward=reward,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
        )


def _sampling_args(task: TaskSpec) -> dict[str, Any]:
    return {"max_tokens": 2048}


def _state_of(result: Any) -> Any:
    """run_rollout returns a RolloutOutput; pull the State dict off it.

    Older 0.1.x return paths handed back the State directly. Try both, plus
    the historical `(messages, state)` tuple shape as a last resort.
    """
    if hasattr(result, "state"):
        return result.state
    if isinstance(result, tuple) and len(result) >= 2:
        return result[1]
    return result


def _extract_messages(state: Any) -> list[dict[str, Any]]:
    """Pull the conversation off a verifiers State.

    Prefers `state["completion"]` (the assembled final conversation), falls
    back to `state["trajectory"]` (per-turn structured records).
    """
    completion = _safe_get(state, "completion")
    if isinstance(completion, list) and completion:
        return [m for m in completion if isinstance(m, dict)]
    trajectory = _safe_get(state, "trajectory")
    if isinstance(trajectory, list):
        flat: list[dict[str, Any]] = []
        for step in trajectory:
            messages = _safe_get(step, "messages")
            if isinstance(messages, list):
                flat.extend(m for m in messages if isinstance(m, dict))
            elif isinstance(step, dict) and "role" in step:
                flat.append(step)
        return flat
    return []


def _collect_errors(state: Any) -> list[str]:
    err = _safe_get(state, "error")
    if err is None:
        return []
    if isinstance(err, str):
        return [err]
    if isinstance(err, dict):
        msg = err.get("message") or err.get("error") or json.dumps(err)
        return [str(msg)]
    return [str(err)]


def _messages_to_steps(messages: list[dict[str, Any]]) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    for i, msg in enumerate(messages):
        role = str(msg.get("role", "policy"))
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        if role in {"assistant", "policy"}:
            actor, kind = "policy", "action"
            if tool_calls:
                first = tool_calls[0]
                fn = first.get("function") or {}
                name = fn.get("name") or first.get("name") or "tool"
                raw_args = fn.get("arguments") or first.get("input") or {}
                args = _maybe_parse_args(raw_args)
                content = json.dumps({"tool": name, "input": args})
        elif role == "tool":
            actor, kind = "environment", "observation"
        elif role == "user":
            actor, kind = "user", "prompt"
        elif role == "system":
            # Skip system primer; trajectory should reflect interaction only.
            continue
        else:
            actor, kind = role, "message"
        steps.append(
            TrajectoryStep(index=i, actor=actor, kind=kind, content=str(content)),
        )
    return steps


def _maybe_parse_args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
    return raw


def _extract_reward(state: Any) -> tuple[float, list[dict[str, Any]]]:
    """Pull (terminal_reward, per-signal-breakdown) off a verifiers State.

    Per `verifiers/types.py`:
      State["reward"]:  float | None
      State["metrics"]: dict[str, float] | None

    `metrics` is a flat name -> float map written by Rubric.score_rollout.
    We flatten that into our `{name, value, weight, reason}` provenance
    shape; weight defaults to 1.0 since verifiers folds weights into the
    weighted-sum reward already.
    """
    raw_reward = _safe_get(state, "reward")
    try:
        reward_val = float(raw_reward) if raw_reward is not None else 0.0
    except (TypeError, ValueError):
        reward_val = 0.0

    metrics = _safe_get(state, "metrics")
    signals: list[dict[str, Any]] = []
    if isinstance(metrics, dict):
        for name, payload in metrics.items():
            try:
                value = float(payload)
            except (TypeError, ValueError):
                continue
            signals.append(
                {"name": str(name), "value": value, "weight": 1.0, "reason": ""},
            )
    return reward_val, signals


def _build_reward_record(
    trajectory_id: str,
    terminal: float,
    signals: list[dict[str, Any]],
    env_id: str,
) -> RewardRecord:
    payload = {
        "source": "verifiers-rubric",
        "rubric": f"verifiers-{env_id}",
        "signals": signals,
    }
    penalties = [
        RewardPenalty(
            code=str(s["name"]),
            value=float(s["value"]) * float(s.get("weight", 1.0)),
            reason=str(s.get("reason", "")),
        )
        for s in signals
        if float(s.get("value", 0.0)) < 0
    ]
    return RewardRecord(
        trajectory_id=trajectory_id,
        terminal_reward=round(max(-1.0, min(1.0, terminal)), 4),
        step_rewards=[],
        penalties=penalties,
        audit_flags=[],
        provenance=json.dumps(payload),
    )


def _extract_usage(state: Any) -> tuple[int, int]:
    """Pull (input_tokens, output_tokens) off a verifiers State.

    Per `verifiers/types.py`, State["usage"] is a TokenUsage TypedDict
    with {input_tokens, output_tokens, final_input_tokens?,
    final_output_tokens?}, all stored as floats.
    """
    usage = _safe_get(state, "usage")
    if usage is None:
        return (0, 0)
    if isinstance(usage, dict):
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _safe_get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
