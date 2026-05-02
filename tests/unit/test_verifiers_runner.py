"""Unit tests for VerifiersRolloutRunner with a fake verifiers env.

The real verifiers package isn't a hard dep, so we monkeypatch
`verifiers.load_environment` with a stand-in object that mirrors the
shape we expect from `env.rollout(...)` (messages + reward + usage).
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from rl_stack.domain.models import TaskSpec, ToolPermission
from rl_stack.infrastructure.environment.verifiers_runner import (
    VerifiersRolloutRunner,
)


class _FakeRollout:
    def __init__(self, *, messages, reward, signals, usage):
        self.messages = messages
        self.reward = reward
        self.rubric_breakdown = signals
        self.usage = usage


class _FakeEnv:
    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []

    def rollout(self, *, client, model, prompt, sampling_args):
        self.calls.append(
            {"client": client, "model": model, "prompt": prompt, "sampling_args": sampling_args},
        )
        return self._result


def _install_fake_verifiers(monkeypatch, env):
    fake_module = types.ModuleType("verifiers")
    fake_module.load_environment = lambda env_id, **kwargs: env
    monkeypatch.setitem(sys.modules, "verifiers", fake_module)


def _make_task() -> TaskSpec:
    return TaskSpec(
        id="task-x",
        prompt="solve me",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read],
        horizon=4,
        success_criteria=["arrive at answer"],
    )


@pytest.mark.asyncio
async def test_run_translates_messages_into_trajectory(monkeypatch):
    result = _FakeRollout(
        messages=[
            {"role": "system", "content": "primer"},
            {"role": "user", "content": "what is 2+2?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"expr": "2+2"}),
                        },
                    },
                ],
            },
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "the answer is 4"},
        ],
        reward=0.85,
        signals={
            "format": {"value": 1.0, "weight": 0.5, "reason": "tool used"},
            "correctness": 0.7,
        },
        usage={"prompt_tokens": 50, "completion_tokens": 25},
    )
    env = _FakeEnv(result)
    _install_fake_verifiers(monkeypatch, env)

    runner = VerifiersRolloutRunner(
        client=object(),
        model="sglang:Qwen/Qwen2.5-7B-Instruct",
        env_id="vf-math",
    )
    outcome = await runner.run(_make_task(), run_id="run-1")

    assert outcome.total_tokens == 75
    # sglang:* is zero-cost
    assert outcome.cost_usd == 0.0
    # System primer is dropped; we keep user/assistant/tool turns
    assert [s.actor for s in outcome.steps] == [
        "user", "policy", "environment", "policy",
    ]
    tool_step = outcome.steps[1]
    payload = json.loads(tool_step.content)
    assert payload == {"tool": "calculator", "input": {"expr": "2+2"}}
    assert outcome.reward.terminal_reward == pytest.approx(0.85)
    prov = json.loads(outcome.reward.provenance)
    assert prov["source"] == "verifiers-rubric"
    names = sorted(s["name"] for s in prov["signals"])
    assert names == ["correctness", "format"]
    # Verifiers is invoked with the bare model id (no provider prefix).
    assert env.calls[0]["model"] == "Qwen/Qwen2.5-7B-Instruct"


@pytest.mark.asyncio
async def test_policy_name_includes_env_and_model(monkeypatch):
    env = _FakeEnv(
        _FakeRollout(messages=[], reward=0.0, signals={}, usage={"prompt_tokens": 0, "completion_tokens": 0}),
    )
    _install_fake_verifiers(monkeypatch, env)
    runner = VerifiersRolloutRunner(client=object(), model="vllm:foo", env_id="rlm")
    assert runner.policy_name() == "verifiers:rlm:vllm:foo"


@pytest.mark.asyncio
async def test_rollout_timeout_is_enforced(monkeypatch):
    import asyncio

    class _SlowEnv:
        def rollout(self, *, client, model, prompt, sampling_args):
            import time
            time.sleep(2)
            return _FakeRollout(messages=[], reward=0.0, signals={}, usage={})

    _install_fake_verifiers(monkeypatch, _SlowEnv())
    runner = VerifiersRolloutRunner(
        client=object(), model="sglang:foo", env_id="vf-math",
        rollout_timeout_s=0.1,
    )
    with pytest.raises(RuntimeError, match="exceeded"):
        await runner.run(_make_task(), run_id="run-1")
    # Force back to the main loop so background thread finishes cleanly.
    await asyncio.sleep(0)
