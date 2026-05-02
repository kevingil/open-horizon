"""Unit tests for VerifiersRolloutRunner with a fake verifiers env.

The real verifiers package isn't a hard dep, so we monkeypatch
`verifiers.load_environment` with a stand-in env that mimics the public
contract documented in verifiers >= 0.1.11:

    async Environment.run_rollout(input, client, model, sampling_args, ...)
        -> RolloutOutput  # has .state: State

State is a dict subclass with keys:
    completion: list[Message] | None
    trajectory: list[TrajectoryStep]
    reward:     float | None        (set by Rubric.score_rollout)
    metrics:    dict[str, float]    (per-signal scores)
    usage:      TokenUsage TypedDict {input_tokens, output_tokens, ...}
    error:      Error | None

Source of truth: verifiers/types.py and verifiers/envs/environment.py.
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


class _FakeRolloutOutput:
    def __init__(self, state: dict):
        self.state = state


class _FakeEnv:
    def __init__(self, state: dict):
        self._state = state
        self.calls: list[dict] = []

    async def run_rollout(self, *, input, client, model, sampling_args):
        self.calls.append(
            {"input": input, "client": client, "model": model, "sampling_args": sampling_args},
        )
        return _FakeRolloutOutput(self._state)


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
async def test_run_translates_state_into_trajectory(monkeypatch):
    state = {
        "completion": [
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
        "reward": 0.85,
        "metrics": {"format": 1.0, "correctness": 0.7},
        "usage": {"input_tokens": 50.0, "output_tokens": 25.0},
        "error": None,
    }
    env = _FakeEnv(state)
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
    # run_rollout is called with the bare model id (no provider prefix)
    # and a RolloutInput dict (not a free-floating prompt kwarg).
    call = env.calls[0]
    assert call["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert call["input"]["prompt"] == [{"role": "user", "content": "solve me"}]


@pytest.mark.asyncio
async def test_run_falls_back_to_trajectory_when_completion_empty(monkeypatch):
    state = {
        "completion": None,
        "trajectory": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        "reward": 0.5,
        "metrics": {"good": 0.5},
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    _install_fake_verifiers(monkeypatch, _FakeEnv(state))
    runner = VerifiersRolloutRunner(client=object(), model="sglang:foo", env_id="vf-math")
    outcome = await runner.run(_make_task(), run_id="run-2")
    assert [s.actor for s in outcome.steps] == ["user", "policy"]


@pytest.mark.asyncio
async def test_run_records_state_error(monkeypatch):
    state = {
        "completion": [{"role": "user", "content": "x"}],
        "reward": 0.0,
        "metrics": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "error": {"message": "tool call failed"},
    }
    _install_fake_verifiers(monkeypatch, _FakeEnv(state))
    runner = VerifiersRolloutRunner(client=object(), model="sglang:foo", env_id="vf-math")
    outcome = await runner.run(_make_task(), run_id="run-3")
    assert outcome.trajectory.errors == ["tool call failed"]


@pytest.mark.asyncio
async def test_policy_name_includes_env_and_model(monkeypatch):
    env = _FakeEnv(
        {"completion": [], "reward": 0.0, "metrics": {}, "usage": {"input_tokens": 0, "output_tokens": 0}},
    )
    _install_fake_verifiers(monkeypatch, env)
    runner = VerifiersRolloutRunner(client=object(), model="vllm:foo", env_id="rlm")
    assert runner.policy_name() == "verifiers:rlm:vllm:foo"


@pytest.mark.asyncio
async def test_rollout_timeout_is_enforced(monkeypatch):
    import asyncio

    class _SlowEnv:
        async def run_rollout(self, *, input, client, model, sampling_args):
            await asyncio.sleep(2)
            return _FakeRolloutOutput({})

    _install_fake_verifiers(monkeypatch, _SlowEnv())
    runner = VerifiersRolloutRunner(
        client=object(), model="sglang:foo", env_id="vf-math",
        rollout_timeout_s=0.1,
    )
    with pytest.raises(RuntimeError, match="exceeded"):
        await runner.run(_make_task(), run_id="run-1")
