"""A tiny stand-in for anthropic.Anthropic used in tests.

Scripts a list of responses to return in order; each response is a dict-like
payload that mimics the SDK's object shape (blocks, usage). No network."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Block:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _Response:
    content: list[_Block] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)


def tool_use(tool_id: str, name: str, input: dict[str, Any]) -> _Response:
    return _Response(
        content=[_Block(type="tool_use", id=tool_id, name=name, input=input)],
        usage=_Usage(input_tokens=10, output_tokens=5),
    )


def text(body: str) -> _Response:
    return _Response(
        content=[_Block(type="text", text=body)],
        usage=_Usage(input_tokens=8, output_tokens=6),
    )


class FakeAnthropic:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.messages = _Messages(self)


class _Messages:
    def __init__(self, parent: FakeAnthropic) -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> _Response:
        self._parent.calls.append(kwargs)
        if not self._parent._responses:
            return text("(no scripted response)")
        head = self._parent._responses.pop(0)
        if isinstance(head, Exception):
            raise head
        return head
