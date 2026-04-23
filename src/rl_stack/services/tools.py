from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..contracts import ToolHarness


@dataclass
class LocalToolHarness(ToolHarness):
    root: Path
    command_log: list[str] = field(default_factory=list)

    def read_file(self, path: str) -> str:
        return (self.root / path).read_text()

    def list_files(self, path: str) -> list[str]:
        target = self.root / path
        if not target.exists():
            return []
        return sorted(str(item.relative_to(self.root)) for item in target.rglob("*") if item.is_file())

    def record_command(self, command: str) -> str:
        self.command_log.append(command)
        return command
