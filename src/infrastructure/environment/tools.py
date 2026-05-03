"""Tool schema shared between the policy and the environment runner.

Two formats live here:
- The internal one (a list of `{name, description, parameters}`) is the
  source of truth and what the env runner uses.
- `openai_tool_definitions()` reshapes them into the OpenAI Chat Completions
  function-tool envelope so any OpenAI-compat provider (OpenAI, vLLM, Ollama,
  OpenRouter, llama.cpp) can call them.
"""
from __future__ import annotations

from typing import Any


def _tools() -> list[dict[str, Any]]:
    """Internal source of truth: name, description, JSON-Schema parameters."""
    return [
        {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "list_files",
            "description": "List files under a workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "required": [],
            },
        },
        {
            "name": "search",
            "description": "Search file contents for a regex (ripgrep-style).",
            "parameters": {
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
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
        {
            "name": "write_file",
            "description": (
                "Create or overwrite a UTF-8 text file inside the workspace. "
                "Path must stay within the workspace; no symlinks, no deletes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "finish",
            "description": "Signal that the task is complete. Provide a final summary.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    ]


def openai_tool_definitions() -> list[dict[str, Any]]:
    """Tools wrapped in OpenAI's function-tool envelope."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in _tools()
    ]


TOOL_NAMES = {t["name"] for t in _tools()}


COMMAND_ALLOWLIST = {
    "ls", "cat", "rg", "grep", "head", "tail", "wc", "find", "python", "python3", "pytest",
}
