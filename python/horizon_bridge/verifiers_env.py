"""Rollout delegator for the `verifiers` framework.

`verifiers` owns its own rollout loop: `Environment.run_rollout(input,
client, model, sampling_args)` calls the abstract `rollout()` and scores
the rubric, returning a `RolloutOutput` whose `.state` carries
`completion`, `trajectory`, `reward`, `metrics`, `usage`, and `error`.

We translate that State into the flat shape the Rust coordinator
expects; cost, persistence, and events are Rust's job.

References (verifiers >= 0.1.11):
  verifiers/envs/environment.py:Environment.run_rollout
  verifiers/types.py:State, TokenUsage, RolloutInput
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

_ENV_CACHE: dict[str, Any] = {}


def _load_env(env_id: str, env_args: dict[str, Any]) -> Any:
    key = json.dumps([env_id, env_args], sort_keys=True, default=str)
    if key in _ENV_CACHE:
        return _ENV_CACHE[key]
    try:
        import verifiers as vf  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via integration env
        raise RuntimeError(
            "profiles routing to verifiers require the `verifiers` package: "
            "pip install -e 'python[envs]'"
        ) from exc
    env = vf.load_environment(env_id, **env_args)
    _ENV_CACHE[key] = env
    return env


def _make_client(base_url: str, api_key: str) -> Any:
    from openai import AsyncOpenAI  # type: ignore[import-not-found]

    return AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")


async def run_rollout(params: dict[str, Any]) -> dict[str, Any]:
    task = params["task"]
    env = _load_env(str(params["env_id"]), dict(params.get("env_args") or {}))
    client = params.get("client") or _make_client(str(params.get("base_url", "")), str(params.get("api_key", "")))
    sampling_args = dict(params.get("sampling_args") or {"max_tokens": 2048})
    timeout_s = params.get("timeout_s")

    rollout_input = {"prompt": [{"role": "user", "content": task["prompt"]}], "task": task["id"]}

    async def _invoke() -> Any:
        return await env.run_rollout(
            input=rollout_input, client=client, model=str(params["model"]), sampling_args=sampling_args,
        )

    try:
        result = await asyncio.wait_for(_invoke(), timeout=timeout_s) if timeout_s else await _invoke()
    except TimeoutError as exc:
        raise RuntimeError(f"verifiers rollout exceeded {float(timeout_s):.1f}s") from exc

    state = state_of(result)
    input_tokens, output_tokens = extract_usage(state)
    terminal, signals = extract_reward(state)
    return {
        "steps": messages_to_steps(extract_messages(state)),
        "errors": collect_errors(state),
        "terminal_reward": terminal,
        "signals": signals,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


# --- State translation -------------------------------------------------------


def state_of(result: Any) -> Any:
    """run_rollout returns a RolloutOutput; older paths returned State or (messages, state)."""
    if hasattr(result, "state"):
        return result.state
    if isinstance(result, tuple) and len(result) >= 2:
        return result[1]
    return result


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def extract_messages(state: Any) -> list[dict[str, Any]]:
    completion = _get(state, "completion")
    if isinstance(completion, list) and completion:
        return [m for m in completion if isinstance(m, dict)]
    trajectory = _get(state, "trajectory")
    if isinstance(trajectory, list):
        flat: list[dict[str, Any]] = []
        for step in trajectory:
            messages = _get(step, "messages")
            if isinstance(messages, list):
                flat.extend(m for m in messages if isinstance(m, dict))
            elif isinstance(step, dict) and "role" in step:
                flat.append(step)
        return flat
    return []


def collect_errors(state: Any) -> list[str]:
    err = _get(state, "error")
    if err is None:
        return []
    if isinstance(err, str):
        return [err]
    if isinstance(err, dict):
        return [str(err.get("message") or err.get("error") or json.dumps(err))]
    return [str(err)]


def _maybe_parse_args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
    return raw


def messages_to_steps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        role = str(msg.get("role", "policy"))
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        if role in {"assistant", "policy"}:
            actor, kind = "policy", "action"
            if tool_calls:
                first = tool_calls[0]
                fn = first.get("function") or {}
                name = fn.get("name") or first.get("name") or "tool"
                args = _maybe_parse_args(fn.get("arguments") or first.get("input") or {})
                content = json.dumps({"tool": name, "input": args})
        elif role == "tool":
            actor, kind = "environment", "observation"
        elif role == "user":
            actor, kind = "user", "prompt"
        elif role == "system":
            continue  # trajectory reflects interaction only
        else:
            actor, kind = role, "message"
        steps.append({"index": i, "actor": actor, "kind": kind, "content": str(content)})
    return steps


def extract_reward(state: Any) -> tuple[float, list[dict[str, Any]]]:
    raw = _get(state, "reward")
    try:
        reward = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        reward = 0.0
    signals: list[dict[str, Any]] = []
    metrics = _get(state, "metrics")
    if isinstance(metrics, dict):
        for name, payload in metrics.items():
            try:
                value = float(payload)
            except (TypeError, ValueError):
                continue
            signals.append({"name": str(name), "value": value, "weight": 1.0, "reason": ""})
    return reward, signals


def extract_usage(state: Any) -> tuple[int, int]:
    usage = _get(state, "usage")
    if usage is None:
        return (0, 0)
    return (int(_get(usage, "input_tokens") or 0), int(_get(usage, "output_tokens") or 0))
