"""A tiny stand-in for openai.OpenAI used in tests.

Scripts a list of responses to return in order; each response mimics the
SDK's chat.completions.create return shape (choices[].message + usage).
No network."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _FunctionCall:
    name: str
    arguments: str  # JSON string, exactly as the SDK delivers


@dataclass
class _ToolCall:
    id: str
    function: _FunctionCall
    type: str = "function"


@dataclass
class _Message:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[_ToolCall] = field(default_factory=list)


@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _PromptDetails:
    cached_tokens: int = 0


@dataclass
class _CompletionDetails:
    reasoning_tokens: int = 0


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: _PromptDetails | None = None
    completion_tokens_details: _CompletionDetails | None = None
    # Anthropic-via-OpenAI-compat top-level cache buckets.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class _Response:
    choices: list[_Choice] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)


def tool_use(call_id: str, name: str, arguments: dict[str, Any]) -> _Response:
    return _Response(
        choices=[
            _Choice(
                message=_Message(
                    tool_calls=[
                        _ToolCall(
                            id=call_id,
                            function=_FunctionCall(name=name, arguments=json.dumps(arguments)),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_Usage(prompt_tokens=10, completion_tokens=5),
    )


def text(
    body: str,
    *,
    cached: int = 0,
    cache_creation: int = 0,
    cache_read_anthropic: int = 0,
    reasoning: int = 0,
) -> _Response:
    """Script a text response with optional usage embellishments.

    `cached`              - OpenAI shape (prompt_tokens_details.cached_tokens)
    `cache_creation`      - Anthropic-via-compat (cache_creation_input_tokens)
    `cache_read_anthropic`- Anthropic-via-compat (cache_read_input_tokens)
    `reasoning`           - o-series / gpt-5 reasoning tokens
                            (completion_tokens_details.reasoning_tokens)

    Both providers fold cache buckets into prompt_tokens, so the fake
    reflects that: prompt_tokens grows with the cache fields. Reasoning
    lives inside completion_tokens for the same reason.
    """
    return _Response(
        choices=[_Choice(message=_Message(content=body))],
        usage=_Usage(
            prompt_tokens=8 + cached + cache_creation + cache_read_anthropic,
            completion_tokens=6 + reasoning,
            prompt_tokens_details=_PromptDetails(cached_tokens=cached) if cached else None,
            completion_tokens_details=(
                _CompletionDetails(reasoning_tokens=reasoning) if reasoning else None
            ),
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read_anthropic,
        ),
    )


class FakeOpenAI:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.chat = _Chat(self)


class _Chat:
    def __init__(self, parent: FakeOpenAI) -> None:
        self.completions = _Completions(parent)


class _Completions:
    def __init__(self, parent: FakeOpenAI) -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> _Response:
        # Deep-copy so tests can inspect the exact wire shape sent at this call,
        # even if the policy later mutates its own message buffer.
        self._parent.calls.append(copy.deepcopy(kwargs))
        if not self._parent._responses:
            return text("(no scripted response)")
        head = self._parent._responses.pop(0)
        if isinstance(head, Exception):
            raise head
        return head
