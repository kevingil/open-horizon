"""Cheap per-rollout repo snapshots.

We copy only the files git already tracks so we skip node_modules, .venv,
artifacts, etc., without each caller needing to know the project layout.
Falls back to a filtered copytree when the source isn't a git repo.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_ALWAYS_IGNORE = {
    "__pycache__", ".git", ".venv", "node_modules", "dist", "build",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "artifacts",
}


def snapshot_repo(source: Path, destination: Path) -> Path:
    """Create a working-copy snapshot of `source` at `destination`.

    Prefers `git ls-files` when available so .gitignore is honoured. Returns
    the destination path.
    """
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    files = _git_tracked_files(source)
    if files is None:
        _copy_tree_filtered(source, destination)
        return destination

    for rel in files:
        src = source / rel
        if not src.is_file():
            continue
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return destination


def _git_tracked_files(source: Path) -> list[str] | None:
    if not (source / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "ls-files"],
            capture_output=True, text=True, check=True, timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


def _copy_tree_filtered(source: Path, destination: Path) -> None:
    def ignore(_: str, names: list[str]) -> list[str]:
        return [n for n in names if n in _ALWAYS_IGNORE]

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=ignore)
