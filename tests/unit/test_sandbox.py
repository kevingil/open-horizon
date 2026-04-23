from __future__ import annotations

from pathlib import Path

from rl_stack.infrastructure.environment.sandbox import (
    DockerSandbox,
    NullSandbox,
    build_sandbox,
)


def test_null_sandbox_runs_on_host(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi")
    result = NullSandbox().run(tmp_path, ["ls"], timeout=5)
    assert result.returncode == 0
    assert "hello.txt" in result.stdout
    assert not result.timed_out


def test_null_sandbox_reports_timeouts(tmp_path: Path) -> None:
    result = NullSandbox().run(tmp_path, ["python", "-c", "import time; time.sleep(3)"], timeout=0.2)
    assert result.timed_out
    assert result.returncode == 124


def test_docker_sandbox_unavailable_without_daemon(monkeypatch) -> None:
    import rl_stack.infrastructure.environment.sandbox as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    sb = DockerSandbox()
    assert sb.available() is False


def test_build_sandbox_falls_back_when_docker_missing(monkeypatch) -> None:
    import rl_stack.infrastructure.environment.sandbox as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    sb = build_sandbox("docker")
    assert sb.name == "none"


def test_build_sandbox_none_is_null() -> None:
    sb = build_sandbox("none")
    assert isinstance(sb, NullSandbox)
