from __future__ import annotations

import json

import pytest

from rl_stack.domain.models import TaskSpec, ToolPermission
from rl_stack.infrastructure.policy.claude import ClaudePolicyServer
from tests._fakes.anthropic import FakeAnthropic, text, tool_use


def _task() -> TaskSpec:
    return TaskSpec(
        id="t1",
        prompt="summarise the repo",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read, ToolPermission.search],
        horizon=4,
        success_criteria=["README surfaced"],
    )


def test_tool_use_becomes_action_json() -> None:
    client = FakeAnthropic([tool_use("t_1", "read_file", {"path": "README.md"})])
    policy = ClaudePolicyServer(client=client)
    task = _task()
    action = policy.generate_action(task, context=[])
    payload = json.loads(action)
    assert payload == {"tool": "read_file", "input": {"path": "README.md"}}


def test_observations_feed_tool_result_on_next_call() -> None:
    client = FakeAnthropic(
        [
            tool_use("t_1", "list_files", {"path": "."}),
            tool_use("t_2", "read_file", {"path": "README.md"}),
        ]
    )
    policy = ClaudePolicyServer(client=client)
    task = _task()
    first = policy.generate_action(task, context=[])
    assert json.loads(first)["tool"] == "list_files"

    # The coordinator now hands us the observation from the env.
    policy.generate_action(task, context=["{\"entries\": [\"README.md\"]}"])
    # The SDK received a user turn with tool_result referencing the first id.
    second_call = client.calls[1]
    tool_results = [
        block for msg in second_call["messages"] for block in _blocks(msg)
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert tool_results
    assert tool_results[-1]["tool_use_id"] == "t_1"


def test_plain_text_response_becomes_finish() -> None:
    client = FakeAnthropic([text("Task complete: README mentions welcome.")])
    policy = ClaudePolicyServer(client=client)
    action = policy.generate_action(_task(), context=[])
    payload = json.loads(action)
    assert payload["tool"] == "finish"
    assert "complete" in payload["input"]["summary"].lower()


def test_token_accounting_drives_cost() -> None:
    client = FakeAnthropic(
        [tool_use("t_1", "read_file", {"path": "r"}), text("done")]
    )
    policy = ClaudePolicyServer(client=client, model="claude-haiku-4-5")
    task = _task()
    policy.generate_action(task, context=[])
    policy.generate_action(task, context=["obs"])
    tokens = policy.total_tokens(task.id)
    assert tokens > 0
    cost = policy.cumulative_cost_usd(task.id)
    assert cost > 0
    assert cost < 0.01  # sanity: a couple dozen tokens on Haiku is tiny


def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    boom = RuntimeError("transient")
    client = FakeAnthropic([boom, boom, tool_use("t_1", "list_files", {})])
    policy = ClaudePolicyServer(client=client, max_retries=3)
    action = policy.generate_action(_task(), context=[])
    assert json.loads(action)["tool"] == "list_files"


def test_retries_exhaust_then_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda *_: None)
    client = FakeAnthropic([RuntimeError("x"), RuntimeError("x")])
    policy = ClaudePolicyServer(client=client, max_retries=1)
    with pytest.raises(RuntimeError):
        policy.generate_action(_task(), context=[])


def _blocks(msg: dict) -> list:
    content = msg.get("content")
    if isinstance(content, list):
        return content
    return []
