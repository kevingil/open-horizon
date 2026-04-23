"""Sandbox abstractions for running tool commands in isolated environments.

- NullSandbox: runs commands directly on the host. Current Phase 1 behaviour.
  Kept for dev laptops without Docker and for tests.
- DockerSandbox: runs commands inside a disposable container with a mounted
  workspace, no network, cpu/mem caps, non-root user.

Availability is detected once per process. Commands that the policy calls
through `run_command` flow through whichever sandbox the RepoEnvironmentRunner
is configured with; file reads/lists/searches stay on the host (they're
read-only against a path-checked workspace).
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


class Sandbox(Protocol):
    name: str

    def available(self) -> bool: ...

    def run(
        self,
        workspace: Path,
        command: list[str],
        *,
        timeout: float,
    ) -> SandboxResult: ...


@dataclass(frozen=True)
class NullSandbox:
    """Host-side subprocess. No isolation beyond the path/command allowlist."""

    name: str = "none"

    def available(self) -> bool:
        return True

    def run(self, workspace: Path, command: list[str], *, timeout: float) -> SandboxResult:
        try:
            out = subprocess.run(
                command, cwd=workspace, capture_output=True, text=True,
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                returncode=124,
                stdout=(exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr="timeout",
                timed_out=True,
            )
        return SandboxResult(returncode=out.returncode, stdout=out.stdout, stderr=out.stderr)


@dataclass
class DockerSandbox:
    """Runs commands inside a disposable container with workspace mounted at /ws."""

    image: str = "python:3.11-slim"
    memory: str = "512m"
    cpus: str = "1"
    network: str = "none"
    user: str = "nobody"
    name: str = "docker"
    _available: bool | None = None

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        if shutil.which("docker") is None:
            self._available = False
            return False
        try:
            subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True, check=True, timeout=3,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            self._available = False
            return False
        self._available = True
        return True

    def run(self, workspace: Path, command: list[str], *, timeout: float) -> SandboxResult:
        docker_cmd = [
            "docker", "run", "--rm",
            f"--network={self.network}",
            f"--memory={self.memory}",
            f"--cpus={self.cpus}",
            "--user", self.user,
            "-v", f"{workspace}:/ws",
            "-w", "/ws",
            self.image,
            *command,
        ]
        try:
            out = subprocess.run(
                docker_cmd, capture_output=True, text=True,
                timeout=timeout + 5, check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(returncode=124, stdout="", stderr="timeout", timed_out=True)
        return SandboxResult(returncode=out.returncode, stdout=out.stdout, stderr=out.stderr)


def build_sandbox(kind: str, image: str | None = None) -> Sandbox:
    """Factory used by bootstrap. Falls back to NullSandbox if docker unavailable."""
    if kind == "none":
        return NullSandbox()
    if kind == "docker":
        sb = DockerSandbox(image=image) if image else DockerSandbox()
        if sb.available():
            return sb
        return NullSandbox()
    raise ValueError(f"Unknown sandbox kind: {kind}")
