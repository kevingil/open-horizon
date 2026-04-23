"""Tool schema shared between the policy (tool_use definitions) and the
environment runner (executor). Keeping them in one place keeps intent and
implementation aligned."""
from __future__ import annotations

from typing import Any


def tool_definitions() -> list[dict[str, Any]]:
    """Anthropic-SDK-compatible tool definitions the policy can call."""
    return [
        {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "list_files",
            "description": "List files under a workspace directory.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "required": [],
            },
        },
        {
            "name": "search",
            "description": "Search file contents for a regex (ripgrep-style).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "run_command",
            "description": (
                "Run a shell command from the workspace allowlist. "
                "Allowed: ls, cat, rg, grep, head, tail, wc, find, python, pytest."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
        {
            "name": "finish",
            "description": "Signal that the task is complete. Provide a final summary.",
            "input_schema": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    ]


TOOL_NAMES = {t["name"] for t in tool_definitions()}


COMMAND_ALLOWLIST = {
    "ls", "cat", "rg", "grep", "head", "tail", "wc", "find", "python", "python3", "pytest",
}
