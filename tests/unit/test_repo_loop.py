"""Unit tests for the repo-path tool-call loop (run_repo_rollout).

Phase B replaced the OpenAICompatPolicyServer class with a stateless
async procedure. These tests exercise the same behaviors:
  - tool call → action JSON in the trajectory
  - observations threaded back as tool-role messages on the next turn
  - plain text response → implicit finish
  - retries on transient errors
  - usage routing for OpenAI cached tokens, Anthropic cache buckets,
    and reasoning tokens
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from domain.models import TaskSpec, ToolPermission
from infrastructure.policy.repo_loop import RepoRolloutOutcome, run_repo_rollout
from tests._fakes.openai_compat import FakeOpenAI, text, tool_use


@dataclass
class _StubRepoRunner:
    """Mirror of the conftest stub; kept local so this test file is
    standalone and doesn't reach across the test tree."""

    observations: dict[str, str] = field(default_factory=dict)
    tasks: dict[str, TaskSpec] = field(default_factory=dict)

    def create_task(self, task: TaskSpec) -> TaskSpec:
        self.tasks[task.id] = task
        return task

    def step(self, task_id: str, action: str) -> str:
        # Default: echo the action so tool-result threading is visible.
        return self.observations.get(task_id, f"obs:{task_id}:{action}")


def _task(horizon: int = 4) -> TaskSpec:
    return TaskSpec(
        id="t1",
        prompt="summarise the repo",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read, ToolPermission.search],
        horizon=horizon,
        success_criteria=["README surfaced"],
    )


async def _drive(
    client: FakeOpenAI,
    *,
    model: str = "gpt-5.4-mini",
    horizon: int = 4,
    max_tokens: int = 100_000,
    repo_runner: _StubRepoRunner | None = None,
    **kwargs,
) -> RepoRolloutOutcome:
    repo_runner = repo_runner or _StubRepoRunner()
    return await run_repo_rollout(
        client=client,                   # type: ignore[arg-type]
        model=model,
        task=_task(horizon=horizon),
        repo_runner=repo_runner,
        record_command=lambda _action: None,
        horizon=horizon,
        max_tokens_per_run=max_tokens,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_tool_call_becomes_action_json() -> None:
    client = FakeOpenAI(
        [tool_use("call_1", "read_file", {"path": "README.md"}), text("done")],
    )
    outcome = await _drive(client, horizon=4)
    action_step = outcome.steps[0]
    payload = json.loads(action_step.content)
    assert payload == {"tool": "read_file", "input": {"path": "README.md"}}


@pytest.mark.asyncio
async def test_observations_feed_tool_role_message_on_next_call() -> None:
    repo = _StubRepoRunner(observations={"t1": '{"entries": ["README.md"]}'})
    client = FakeOpenAI(
        [
            tool_use("call_1", "list_files", {"path": "."}),
            tool_use("call_2", "read_file", {"path": "README.md"}),
            text("ok"),
        ],
    )
    await _drive(client, horizon=4, repo_runner=repo)
    second_call = client.calls[1]
    tool_msgs = [m for m in second_call["messages"] if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[-1]["tool_call_id"] == "call_1"
    assistant_msgs = [m for m in second_call["messages"] if m.get("role") == "assistant"]
    assert assistant_msgs and assistant_msgs[-1]["tool_calls"][0]["id"] == "call_1"


@pytest.mark.asyncio
async def test_plain_text_response_becomes_finish() -> None:
    client = FakeOpenAI([text("Task complete: README mentions welcome.")])
    outcome = await _drive(client, horizon=4)
    payload = json.loads(outcome.steps[0].content)
    assert payload["tool"] == "finish"
    assert "complete" in payload["input"]["summary"].lower()
    # Finish ends the loop early - one action + one observation only.
    assert len(outcome.steps) == 2


@pytest.mark.asyncio
async def test_token_accounting_drives_cost_for_known_model() -> None:
    client = FakeOpenAI(
        [tool_use("call_1", "read_file", {"path": "r"}), text("done")],
    )
    outcome = await _drive(client, horizon=4)
    assert outcome.total_tokens > 0
    assert 0 < outcome.cost_usd < 0.01


@pytest.mark.asyncio
async def test_self_hosted_model_costs_zero() -> None:
    client = FakeOpenAI([tool_use("call_1", "list_files", {}), text("done")])
    outcome = await _drive(client, model="vllm:Qwen/Qwen3-8B", horizon=4)
    assert outcome.total_tokens > 0
    assert outcome.cost_usd == 0.0


@pytest.mark.asyncio
async def test_anthropic_cache_buckets_route_to_their_own_tiers() -> None:
    # Anthropic-via-compat reports cache_creation_input_tokens and
    # cache_read_input_tokens at the top level of `usage`, folded INTO
    # prompt_tokens. The loop must split them out and bill at the
    # cache-write / cache-read rates instead of the input rate.
    client = FakeOpenAI([text("ok", cache_creation=20, cache_read_anthropic=5)])
    outcome = await _drive(client, model="claude-haiku-4-5", horizon=4)
    # Cost = (8 * 1.0 + 6 * 5.0 + 20 * 1.25 + 5 * 0.1) / 1e6 = 6.35e-5.
    expected = (8 * 1.0 + 6 * 5.0 + 20 * 1.25 + 5 * 0.1) / 1_000_000
    assert outcome.cost_usd == pytest.approx(expected, abs=1e-6)


@pytest.mark.asyncio
async def test_reasoning_tokens_tracked_but_not_double_billed() -> None:
    # Reasoning tokens live INSIDE completion_tokens at the output rate.
    # output cost for 18 completion_tokens vs 6 should differ by 12 tokens
    # worth of output rate, no double-counting.
    quiet_client = FakeOpenAI([text("done")])
    quiet = await _drive(quiet_client, horizon=4)
    loud_client = FakeOpenAI([text("done", reasoning=12)])
    loud = await _drive(loud_client, horizon=4)
    # 12 extra completion_tokens at gpt-5.4-mini's $4.50/Mtok output rate:
    delta = (loud.cost_usd - quiet.cost_usd) * 1_000_000
    assert delta == pytest.approx(12 * 4.5, abs=0.01)


@pytest.mark.asyncio
async def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    boom = RuntimeError("transient")
    client = FakeOpenAI(
        [boom, boom, tool_use("call_1", "list_files", {}), text("done")],
    )
    outcome = await _drive(client, horizon=4)
    payload = json.loads(outcome.steps[0].content)
    assert payload["tool"] == "list_files"


@pytest.mark.asyncio
async def test_retries_exhaust_then_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = FakeOpenAI([RuntimeError("x"), RuntimeError("x")])
    with pytest.raises(RuntimeError):
        await _drive(client, horizon=4, max_retries=1)


@pytest.mark.asyncio
async def test_extra_body_passes_through_to_provider() -> None:
    client = FakeOpenAI([tool_use("call_1", "list_files", {}), text("done")])
    await _drive(
        client,
        horizon=4,
        extra_body={"top_p": 0.95, "vendor_quirk": True},
    )
    assert client.calls[0]["extra_body"] == {"top_p": 0.95, "vendor_quirk": True}


@pytest.mark.asyncio
async def test_policy_name_in_outcome_includes_model() -> None:
    client = FakeOpenAI([text("done")])
    outcome = await _drive(client, model="gpt-5.4-mini", horizon=4)
    assert outcome.policy_name == "openai:gpt-5.4-mini"


@pytest.mark.asyncio
async def test_token_overflow_terminates_loop() -> None:
    # Many tool calls; 1-token budget guarantees overflow on the first turn.
    client = FakeOpenAI(
        [tool_use(f"call_{i}", "list_files", {}) for i in range(10)] + [text("done")],
    )
    outcome = await _drive(client, horizon=10, max_tokens=1)
    assert outcome.token_overflow
    assert outcome.trajectory.errors
    assert "token budget" in outcome.trajectory.errors[0]
