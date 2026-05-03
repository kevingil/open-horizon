from __future__ import annotations

import json

import pytest

from domain.models import TaskSpec, ToolPermission
from infrastructure.policy.openai_compat import OpenAICompatPolicyServer
from tests._fakes.openai_compat import FakeOpenAI, text, tool_use


def _task() -> TaskSpec:
    return TaskSpec(
        id="t1",
        prompt="summarise the repo",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read, ToolPermission.search],
        horizon=4,
        success_criteria=["README surfaced"],
    )


def test_tool_call_becomes_action_json() -> None:
    client = FakeOpenAI([tool_use("call_1", "read_file", {"path": "README.md"})])
    policy = OpenAICompatPolicyServer(client=client)
    action = policy.generate_action(_task(), context=[])
    payload = json.loads(action)
    assert payload == {"tool": "read_file", "input": {"path": "README.md"}}


def test_observations_feed_tool_role_message_on_next_call() -> None:
    client = FakeOpenAI(
        [
            tool_use("call_1", "list_files", {"path": "."}),
            tool_use("call_2", "read_file", {"path": "README.md"}),
        ]
    )
    policy = OpenAICompatPolicyServer(client=client)
    task = _task()
    first = policy.generate_action(task, context=[])
    assert json.loads(first)["tool"] == "list_files"

    policy.generate_action(task, context=['{"entries": ["README.md"]}'])
    second_call = client.calls[1]
    tool_msgs = [m for m in second_call["messages"] if m.get("role") == "tool"]
    assert tool_msgs
    assert tool_msgs[-1]["tool_call_id"] == "call_1"
    # The model also has to see the previous assistant turn echoing the tool_call.
    assistant_msgs = [m for m in second_call["messages"] if m.get("role") == "assistant"]
    assert assistant_msgs and assistant_msgs[-1]["tool_calls"][0]["id"] == "call_1"


def test_plain_text_response_becomes_finish() -> None:
    client = FakeOpenAI([text("Task complete: README mentions welcome.")])
    policy = OpenAICompatPolicyServer(client=client)
    action = policy.generate_action(_task(), context=[])
    payload = json.loads(action)
    assert payload["tool"] == "finish"
    assert "complete" in payload["input"]["summary"].lower()


def test_token_accounting_drives_cost_for_known_model() -> None:
    client = FakeOpenAI(
        [tool_use("call_1", "read_file", {"path": "r"}), text("done")]
    )
    policy = OpenAICompatPolicyServer(client=client, model="gpt-5.4-mini")
    task = _task()
    policy.generate_action(task, context=[])
    policy.generate_action(task, context=["obs"])
    assert policy.total_tokens(task.id) > 0
    cost = policy.cumulative_cost_usd(task.id)
    assert cost > 0
    assert cost < 0.01  # a couple dozen tokens on gpt-5.4-mini is tiny


def test_self_hosted_model_costs_zero() -> None:
    client = FakeOpenAI([tool_use("call_1", "list_files", {})])
    policy = OpenAICompatPolicyServer(
        client=client, model="vllm:Qwen/Qwen3-8B",
    )
    policy.generate_action(_task(), context=[])
    assert policy.total_tokens("t1") > 0
    assert policy.cumulative_cost_usd("t1") == 0.0


def test_cached_tokens_subtract_from_input_to_avoid_double_billing() -> None:
    client = FakeOpenAI([text("hi", cached=4)])
    policy = OpenAICompatPolicyServer(client=client, model="gpt-5.4-mini")
    task = _task()
    policy.generate_action(task, context=[])
    # text() seeds 8 + cached prompt tokens; the policy should subtract cached
    # from input so the two buckets sum back to the SDK's prompt_tokens.
    # 8 + 4 = 12; expect 8 input, 4 cache_read.
    sums = sum(
        getattr(policy.contexts[task.id], attr)
        for attr in ("input_tokens", "cache_read_tokens")
    )
    assert sums == 12
    assert policy.contexts[task.id].cache_read_tokens == 4


def test_anthropic_cache_buckets_route_to_their_own_tiers() -> None:
    # Anthropic-via-compat reports cache_creation_input_tokens and
    # cache_read_input_tokens at the top level of `usage`, folded INTO
    # prompt_tokens. The policy must split them out and bill at the
    # cache-write / cache-read rates instead of the input rate.
    client = FakeOpenAI([text("ok", cache_creation=20, cache_read_anthropic=5)])
    policy = OpenAICompatPolicyServer(client=client, model="claude-haiku-4-5")
    task = _task()
    policy.generate_action(task, context=[])
    ctx = policy.contexts[task.id]
    # prompt_tokens was 8 + 20 + 5 = 33; expect 8 input, 20 write, 5 read.
    assert ctx.input_tokens == 8
    assert ctx.cache_write_tokens == 20
    assert ctx.cache_read_tokens == 5
    # Cost = (8 * 1.0 + 6 * 5.0 + 20 * 1.25 + 5 * 0.1) / 1e6 = 6.35e-5
    # pricing.estimate_cost_usd rounds to 6 decimals so allow that tolerance.
    expected = (8 * 1.0 + 6 * 5.0 + 20 * 1.25 + 5 * 0.1) / 1_000_000
    assert policy.cumulative_cost_usd(task.id) == pytest.approx(expected, abs=1e-6)


def test_reasoning_tokens_tracked_but_not_double_billed() -> None:
    # Reasoning tokens live INSIDE completion_tokens at the output rate.
    # The policy must record them for visibility (ctx.reasoning_tokens)
    # without adding them to output_tokens a second time.
    client = FakeOpenAI([text("done", reasoning=12)])
    policy = OpenAICompatPolicyServer(client=client, model="gpt-5.4-mini")
    task = _task()
    policy.generate_action(task, context=[])
    ctx = policy.contexts[task.id]
    # text(reasoning=12) returns completion_tokens=6+12=18; output_tokens
    # should match that exactly (no double-counting), and reasoning_tokens
    # records the inner 12 for observability.
    assert ctx.output_tokens == 18
    assert ctx.reasoning_tokens == 12


def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    boom = RuntimeError("transient")
    client = FakeOpenAI([boom, boom, tool_use("call_1", "list_files", {})])
    policy = OpenAICompatPolicyServer(client=client, max_retries=3)
    action = policy.generate_action(_task(), context=[])
    assert json.loads(action)["tool"] == "list_files"


def test_retries_exhaust_then_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = FakeOpenAI([RuntimeError("x"), RuntimeError("x")])
    policy = OpenAICompatPolicyServer(client=client, max_retries=1)
    with pytest.raises(RuntimeError):
        policy.generate_action(_task(), context=[])


def test_extra_body_passes_through_to_provider() -> None:
    client = FakeOpenAI([tool_use("call_1", "list_files", {})])
    policy = OpenAICompatPolicyServer(
        client=client, extra_body={"top_p": 0.95, "vendor_quirk": True},
    )
    policy.generate_action(_task(), context=[])
    assert client.calls[0]["extra_body"] == {"top_p": 0.95, "vendor_quirk": True}


def test_policy_name_includes_model() -> None:
    policy = OpenAICompatPolicyServer(client=FakeOpenAI(), model="gpt-5.4-mini")
    assert policy.policy_name() == "openai:gpt-5.4-mini"
