from __future__ import annotations

import json
from pathlib import Path

import pytest

from domain.models import TaskSpec, ToolPermission
from infrastructure.environment.repo_runner import RepoEnvironmentRunner


@pytest.fixture
def source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "README.md").write_text("hello world")
    sub = src / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    return src


@pytest.fixture
def runner(source: Path, tmp_path: Path) -> RepoEnvironmentRunner:
    return RepoEnvironmentRunner(
        source_root=source,
        scratch_root=tmp_path / "scratch",
        command_timeout_s=5.0,
        max_output_bytes=8_000,
    )


def _task() -> TaskSpec:
    return TaskSpec(
        id="t1",
        prompt="inspect repo",
        repo_snapshot=".",
        tool_permissions=[ToolPermission.read, ToolPermission.terminal],
        horizon=3,
        success_criteria=["done"],
    )


def _call(tool: str, **inp: object) -> str:
    return json.dumps({"tool": tool, "input": inp})


def test_read_file_returns_snapshot_content(runner: RepoEnvironmentRunner) -> None:
    task = _task()
    runner.create_task(task)
    obs = json.loads(runner.step(task.id, _call("read_file", path="README.md")))
    assert obs["tool"] == "read_file"
    assert obs["content"] == "hello world"


def test_list_files_lists_recursive_entries(runner: RepoEnvironmentRunner) -> None:
    task = _task()
    runner.create_task(task)
    obs = json.loads(runner.step(task.id, _call("list_files", path=".")))
    assert "README.md" in obs["entries"]
    assert "pkg/mod.py" in obs["entries"]


def test_path_escape_is_rejected(runner: RepoEnvironmentRunner) -> None:
    task = _task()
    runner.create_task(task)
    obs = json.loads(runner.step(task.id, _call("read_file", path="../../etc/passwd")))
    assert "error" in obs or obs.get("error")


def test_unknown_command_rejected(runner: RepoEnvironmentRunner) -> None:
    task = _task()
    runner.create_task(task)
    obs = json.loads(runner.step(task.id, _call("run_command", command="rm -rf /")))
    assert obs["error"].startswith("command not allowed")


def test_allowed_command_executes(runner: RepoEnvironmentRunner) -> None:
    task = _task()
    runner.create_task(task)
    obs = json.loads(runner.step(task.id, _call("run_command", command="ls")))
    assert obs["returncode"] == 0
    assert "README.md" in obs["stdout"]
    assert obs["sandbox"] == "none"


def test_write_file_creates_inside_workspace(runner: RepoEnvironmentRunner) -> None:
    task = _task()
    runner.create_task(task)
    obs = json.loads(runner.step(task.id, _call("write_file", path="NOTES.md", content="hi there")))
    assert obs["tool"] == "write_file"
    assert obs["bytes"] == len("hi there")
    # Read back through the same sandboxed runner.
    readback = json.loads(runner.step(task.id, _call("read_file", path="NOTES.md")))
    assert readback["content"] == "hi there"


def test_write_file_rejects_path_escape(runner: RepoEnvironmentRunner) -> None:
    task = _task()
    runner.create_task(task)
    obs = json.loads(runner.step(task.id, _call("write_file", path="../evil.txt", content="no")))
    assert "error" in obs


def test_write_file_caps_size(runner: RepoEnvironmentRunner) -> None:
    task = _task()
    runner.create_task(task)
    huge = "a" * 300_000
    obs = json.loads(runner.step(task.id, _call("write_file", path="big.txt", content=huge)))
    assert obs["error"].startswith("content exceeds")


def test_finish_marks_task_done(runner: RepoEnvironmentRunner) -> None:
    task = _task()
    runner.create_task(task)
    first = json.loads(runner.step(task.id, _call("finish", summary="all set")))
    assert first["tool"] == "finish"
    second = json.loads(runner.step(task.id, _call("list_files")))
    assert "error" in second


def test_unparseable_action_returns_error(runner: RepoEnvironmentRunner) -> None:
    task = _task()
    runner.create_task(task)
    obs = json.loads(runner.step(task.id, "not-json"))
    assert "error" in obs


def test_output_truncation_for_large_responses(runner: RepoEnvironmentRunner, source: Path) -> None:
    (source / "big.txt").write_text("x" * 50_000)
    task = _task()
    runner.create_task(task)
    raw = runner.step(task.id, _call("read_file", path="big.txt"))
    assert len(raw.encode("utf-8")) <= 8_000 + 200
    assert json.loads(raw).get("_truncated") is True
