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
    policy = OpenAICompatPolicyServer(client=client, model="gpt-4o-mini")
    task = _task()
    policy.generate_action(task, context=[])
    policy.generate_action(task, context=["obs"])
    assert policy.total_tokens(task.id) > 0
    cost = policy.cumulative_cost_usd(task.id)
    assert cost > 0
    assert cost < 0.01  # a couple dozen tokens on gpt-4o-mini is tiny


def test_self_hosted_model_costs_zero() -> None:
    client = FakeOpenAI([tool_use("call_1", "list_files", {})])
    policy = OpenAICompatPolicyServer(
        client=client, model="vllm:Qwen/Qwen2.5-7B-Instruct",
    )
    policy.generate_action(_task(), context=[])
    assert policy.total_tokens("t1") > 0
    assert policy.cumulative_cost_usd("t1") == 0.0


def test_cached_tokens_subtract_from_input_to_avoid_double_billing() -> None:
    client = FakeOpenAI([text("hi", cached=4)])
    policy = OpenAICompatPolicyServer(client=client, model="gpt-4o-mini")
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
    policy = OpenAICompatPolicyServer(client=FakeOpenAI(), model="gpt-4o-mini")
    assert policy.policy_name() == "openai:gpt-4o-mini"
