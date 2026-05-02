"""Rollout delegator for the `verifiers` framework.

`verifiers` owns its own rollout loop (`env.rollout(client, model, prompt, ...)`),
which conflicts with our per-step `EnvironmentRunner` ABC. Wrapping it
behind a step-by-step shim would lose the rubric grader, RLMEnv, and
OpenEnv glue we actually want.

Instead this module exposes a `VerifiersRolloutRunner` that the
coordinator delegates to when `env_backend=verifiers`. It returns a fully
populated `TrajectoryRecord` and `RewardRecord` in one shot, plus token
usage so the existing budget/cost machinery stays accurate.

The verifiers package is an optional dependency; the import is lazy so
users on the default backend never need it installed.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import structlog

from ...domain.models import (
    RewardPenalty,
    RewardRecord,
    TaskSpec,
    TrajectoryRecord,
    TrajectoryStep,
)
from ...domain.pricing import estimate_cost_usd

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

        async def _invoke() -> Any:
            return await asyncio.to_thread(
                env.rollout,
                client=self.client,
                model=served_model,
                prompt=task.prompt,
                sampling_args=_sampling_args(task),
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

        steps = _messages_to_steps(_extract_messages(result))
        traj = TrajectoryRecord(
            id=f"traj-{uuid4().hex[:8]}",
            task_id=task.id,
            steps=steps,
            errors=[],
        )
        terminal_reward, signals = _extract_reward(result)
        reward = _build_reward_record(traj.id, terminal_reward, signals, self.env_id)

        input_tokens, output_tokens = _extract_usage(result)
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


def _extract_messages(result: Any) -> list[dict[str, Any]]:
    """Tolerant access to verifiers' rollout messages.

    verifiers' return shape has shifted across 0.1.x: it has variously been
    a list of message dicts, a `(messages, state)` tuple, or a typed object
    with a `.messages` / `.completion` attribute. Probe the common shapes
    before giving up.
    """
    candidate = result
    if isinstance(candidate, tuple) and candidate:
        candidate = candidate[0]
    for attr in ("messages", "completion", "trajectory"):
        attr_val = getattr(candidate, attr, None)
        if attr_val is not None:
            candidate = attr_val
            break
    if isinstance(candidate, dict):
        for key in ("messages", "completion", "trajectory"):
            if key in candidate and isinstance(candidate[key], list):
                candidate = candidate[key]
                break
    if not isinstance(candidate, list):
        return []
    return [m for m in candidate if isinstance(m, dict)]


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


def _extract_reward(result: Any) -> tuple[float, list[dict[str, Any]]]:
    """Pull (terminal_reward, per-signal-breakdown) out of a verifiers result.

    Verifiers' `Rubric` produces a dict of per-signal scores (often
    `{"name": float}` or `{"name": {"value": ..., "weight": ...}}`). We
    flatten that into our existing provenance shape.
    """
    candidate = result
    if isinstance(candidate, tuple) and len(candidate) >= 2:
        candidate = candidate[1]
    if hasattr(candidate, "state"):
        candidate = candidate.state

    reward_val: float = 0.0
    breakdown: dict[str, Any] | None = None

    for attr in ("reward", "rewards", "score", "rubric_score"):
        val = _safe_get(candidate, attr)
        if val is None:
            continue
        if isinstance(val, dict):
            breakdown = val
            reward_val = float(val.get("total", val.get("reward", val.get("score", 0.0))))
            break
        try:
            reward_val = float(val)
            break
        except (TypeError, ValueError):
            continue

    for attr in ("rubric_breakdown", "signals", "components"):
        val = _safe_get(candidate, attr)
        if isinstance(val, dict) and breakdown is None:
            breakdown = val
            break

    signals: list[dict[str, Any]] = []
    if isinstance(breakdown, dict):
        for name, payload in breakdown.items():
            if isinstance(payload, dict):
                signals.append(
                    {
                        "name": name,
                        "value": float(payload.get("value", payload.get("score", 0.0))),
                        "weight": float(payload.get("weight", 1.0)),
                        "reason": str(payload.get("reason", "")),
                    }
                )
            else:
                try:
                    signals.append(
                        {"name": name, "value": float(payload), "weight": 1.0, "reason": ""}
                    )
                except (TypeError, ValueError):
                    continue
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


def _extract_usage(result: Any) -> tuple[int, int]:
    """Tolerant probe for token counts in a verifiers rollout result."""
    candidates: list[Any] = [result]
    if isinstance(result, tuple):
        candidates.extend(result)
    for c in candidates:
        usage = _safe_get(c, "usage")
        if usage is None:
            usage = _safe_get(c, "token_usage")
        if usage is None and hasattr(c, "state"):
            usage = _safe_get(c.state, "usage")
        if usage is None:
            continue
        if isinstance(usage, dict):
            return (
                int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            )
        return (
            int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0),
        )
    return (0, 0)


def _safe_get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
