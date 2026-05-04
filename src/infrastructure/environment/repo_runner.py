"""Environment runner backed by a real workspace snapshot.

The policy calls a tool; the coordinator passes the tool call as the action
string (JSON); this runner executes it against a sandboxed tempdir and returns
the observation as JSON. Kept deliberately simple: path allowlist, command
allowlist, wall-clock timeout, byte cap.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from domain.models import TaskSpec

from .sandbox import NullSandbox, Sandbox
from .snapshot import snapshot_repo
from .tools import COMMAND_ALLOWLIST, TOOL_NAMES

# Cap written files so a runaway policy can't fill disk.
_MAX_WRITE_BYTES = 256_000


@dataclass
class _TaskState:
    task: TaskSpec
    workspace: Path
    finished: bool = False
    final_summary: str | None = None


@dataclass
class RepoEnvironmentRunner:
    """Executes tool calls against per-task repo snapshots.

    No longer inherits from an ABC: verifiers' Environment.run_rollout is
    the production path for env-owned multi-turn rollouts; this class is
    the in-house path for "agent fixes a real bug in this checkout"
    scenarios that need shell access against a sandboxed tempdir copy
    of the repo. The two paths share nothing structural, so an ABC
    bridging them was abstraction-for-abstraction's-sake.
    """

    source_root: Path
    scratch_root: Path
    command_timeout_s: float = 10.0
    max_output_bytes: int = 16_384
    sandbox: Sandbox = field(default_factory=NullSandbox)
    states: dict[str, _TaskState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_root = Path(self.source_root).resolve()
        self.scratch_root = Path(self.scratch_root).resolve()
        self.scratch_root.mkdir(parents=True, exist_ok=True)

    def create_task(self, task: TaskSpec) -> TaskSpec:
        workspace = Path(tempfile.mkdtemp(prefix=f"rl-{task.id}-", dir=self.scratch_root))
        snapshot_repo(self.source_root, workspace)
        self.states[task.id] = _TaskState(task=task, workspace=workspace)
        return task

    def reset(self, task_id: str) -> None:
        state = self.states.get(task_id)
        if state is None:
            return
        if state.workspace.exists():
            shutil.rmtree(state.workspace, ignore_errors=True)
        state.workspace.mkdir(parents=True, exist_ok=True)
        snapshot_repo(self.source_root, state.workspace)
        state.finished = False
        state.final_summary = None

    def step(self, task_id: str, action: str) -> str:
        state = self.states.get(task_id)
        if state is None:
            return _err("unknown task")
        if state.finished:
            return _err("task already finished")

        call = _parse_action(action)
        if call is None:
            return _err(f"could not parse action: {action[:160]}")

        tool, args = call
        if tool not in TOOL_NAMES:
            return _err(f"unknown tool: {tool}")

        try:
            result = self._dispatch(state, tool, args)
        except Exception as exc:
            return _err(f"{tool} failed: {exc}")
        return _truncate_json(result, self.max_output_bytes)

    def _dispatch(self, state: _TaskState, tool: str, args: dict) -> dict:
        match tool:
            case "read_file":
                return self._read_file(state.workspace, args)
            case "list_files":
                return self._list_files(state.workspace, args)
            case "search":
                return self._search(state.workspace, args)
            case "run_command":
                return self._run_command(state.workspace, args)
            case "write_file":
                return self._write_file(state.workspace, args)
            case "finish":
                state.finished = True
                state.final_summary = str(args.get("summary", ""))
                return {"tool": "finish", "summary": state.final_summary}
            case other:
                return {"error": f"unhandled tool: {other}"}

    def _resolve(self, workspace: Path, rel: str) -> Path:
        target = (workspace / rel).resolve()
        if workspace not in target.parents and target != workspace:
            raise ValueError(f"path escapes workspace: {rel}")
        return target

    def _read_file(self, workspace: Path, args: dict) -> dict:
        path = str(args.get("path", ""))
        target = self._resolve(workspace, path)
        if not target.is_file():
            return {"tool": "read_file", "path": path, "error": "not a file"}
        data = target.read_text(errors="replace")
        return {"tool": "read_file", "path": path, "content": data}

    def _list_files(self, workspace: Path, args: dict) -> dict:
        path = str(args.get("path", "."))
        target = self._resolve(workspace, path)
        if not target.exists():
            return {"tool": "list_files", "path": path, "error": "missing"}
        entries = sorted(
            str(p.relative_to(workspace))
            for p in target.rglob("*")
            if p.is_file()
        )
        return {"tool": "list_files", "path": path, "entries": entries[:500]}

    def _search(self, workspace: Path, args: dict) -> dict:
        pattern = str(args.get("pattern", ""))
        path = str(args.get("path", "."))
        target = self._resolve(workspace, path)
        rg = shutil.which("rg")
        cmd = (
            [rg, "--no-heading", "--line-number", "--color", "never", pattern, str(target)]
            if rg
            else ["grep", "-rn", pattern, str(target)]
        )
        out = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=self.command_timeout_s, check=False,
        )
        return {
            "tool": "search",
            "pattern": pattern,
            "path": path,
            "matches": out.stdout.splitlines()[:500],
            "returncode": out.returncode,
        }

    def _run_command(self, workspace: Path, args: dict) -> dict:
        command = str(args.get("command", "")).strip()
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return {"tool": "run_command", "error": f"parse error: {exc}"}
        if not parts:
            return {"tool": "run_command", "error": "empty command"}
        if parts[0] not in COMMAND_ALLOWLIST:
            return {
                "tool": "run_command",
                "error": f"command not allowed: {parts[0]}",
                "allowlist": sorted(COMMAND_ALLOWLIST),
            }
        result = self.sandbox.run(workspace, parts, timeout=self.command_timeout_s)
        if result.timed_out:
            return {"tool": "run_command", "error": "timeout", "command": command}
        return {
            "tool": "run_command",
            "command": command,
            "stdout": result.stdout[-self.max_output_bytes :],
            "stderr": result.stderr[-self.max_output_bytes :],
            "returncode": result.returncode,
            "sandbox": self.sandbox.name,
        }

    def _write_file(self, workspace: Path, args: dict) -> dict:
        path = str(args.get("path", ""))
        content = args.get("content", "")
        if not isinstance(content, str):
            return {"tool": "write_file", "error": "content must be a string"}
        encoded = content.encode("utf-8")
        if len(encoded) > _MAX_WRITE_BYTES:
            return {
                "tool": "write_file",
                "error": f"content exceeds {_MAX_WRITE_BYTES} bytes",
            }
        target = self._resolve(workspace, path)
        if target.is_symlink() or (target.exists() and target.is_dir()):
            return {"tool": "write_file", "error": "refusing to overwrite symlink or directory"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {
            "tool": "write_file",
            "path": path,
            "bytes": len(encoded),
        }


def _parse_action(action: str) -> tuple[str, dict] | None:
    """Accepts {"tool": "...", "input": {...}} or {"name": "...", "input": {...}}."""
    try:
        payload = json.loads(action)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    tool = payload.get("tool") or payload.get("name")
    args = payload.get("input") or payload.get("args") or {}
    if not isinstance(tool, str) or not isinstance(args, dict):
        return None
    return tool, args


def _err(reason: str) -> str:
    return json.dumps({"error": reason})


def _truncate_json(data: dict, max_bytes: int) -> str:
    text = json.dumps(data)
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    truncated = text[: max_bytes - 64]
    return json.dumps({"_truncated": True, "preview": truncated})
