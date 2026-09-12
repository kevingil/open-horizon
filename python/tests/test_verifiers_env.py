"""Verifiers translation tests with a fake env mimicking verifiers >= 0.1.11:
`await Environment.run_rollout(input, client, model, sampling_args) -> RolloutOutput(.state)`."""
from __future__ import annotations

import json
import sys
import types

import pytest

from horizon_bridge import verifiers_env


class _Output:
    def __init__(self, state):
        self.state = state


class _FakeEnv:
    def __init__(self, state):
        self._state = state
        self.calls = []

    async def run_rollout(self, *, input, client, model, sampling_args):
        self.calls.append({"input": input, "client": client, "model": model, "sampling_args": sampling_args})
        return _Output(self._state)


def _install(monkeypatch, env):
    module = types.ModuleType("verifiers")
    module.load_environment = lambda env_id, **kwargs: env
    monkeypatch.setitem(sys.modules, "verifiers", module)
    verifiers_env._ENV_CACHE.clear()


def _params(**over):
    base = {
        "run_id": "run-1",
        "task": {"id": "task-x", "prompt": "solve me", "horizon": 4, "success_criteria": ["answer"]},
        "model": "Qwen/Qwen3-8B",
        "env_id": "vf-math",
        "env_args": {},
        "client": object(),
        "sampling_args": {"max_tokens": 64},
    }
    base.update(over)
    return base


async def test_translates_state_into_flat_outcome(monkeypatch) -> None:
    state = {
        "completion": [
            {"role": "system", "content": "primer"},
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "calculator", "arguments": json.dumps({"expr": "2+2"})}}]},
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "the answer is 4"},
        ],
        "reward": 0.85,
        "metrics": {"format": 1.0, "correctness": 0.7},
        "usage": {"input_tokens": 50.0, "output_tokens": 25.0},
        "error": None,
    }
    env = _FakeEnv(state)
    _install(monkeypatch, env)
    out = await verifiers_env.run_rollout(_params())
    assert (out["input_tokens"], out["output_tokens"]) == (50, 25)
    assert [s["actor"] for s in out["steps"]] == ["user", "policy", "environment", "policy"]
    assert json.loads(out["steps"][1]["content"]) == {"tool": "calculator", "input": {"expr": "2+2"}}
    assert out["terminal_reward"] == pytest.approx(0.85)
    assert sorted(s["name"] for s in out["signals"]) == ["correctness", "format"]
    assert out["errors"] == []
    call = env.calls[0]
    assert call["model"] == "Qwen/Qwen3-8B"
    assert call["input"] == {"prompt": [{"role": "user", "content": "solve me"}], "task": "task-x"}


async def test_falls_back_to_trajectory_and_collects_errors(monkeypatch) -> None:
    state = {
        "completion": None,
        "trajectory": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "reward": None,
        "metrics": {"good": "not-a-number"},
        "usage": None,
        "error": {"message": "env exploded"},
    }
    _install(monkeypatch, _FakeEnv(state))
    out = await verifiers_env.run_rollout(_params())
    assert [s["kind"] for s in out["steps"]] == ["prompt", "action"]
    assert out["terminal_reward"] == 0.0
    assert out["signals"] == []
    assert out["errors"] == ["env exploded"]
    assert (out["input_tokens"], out["output_tokens"]) == (0, 0)


async def test_timeout_surfaces_as_runtime_error(monkeypatch) -> None:
    import asyncio

    class _Slow:
        async def run_rollout(self, **kwargs):
            await asyncio.sleep(5)

    _install(monkeypatch, _Slow())
    with pytest.raises(RuntimeError, match="exceeded"):
        await verifiers_env.run_rollout(_params(timeout_s=0.01))
