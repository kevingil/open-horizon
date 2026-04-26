"""File-system AdapterRegistry.

Layout under `root/`:
    {adapter_id}/
        manifest.json       -- AdapterRecord serialized (always present)
        adapter.safetensors -- weights blob (optional; absent for stubs)
        ... any provider-specific files

Loading is eager but cheap: every list_adapters() rereads manifests off disk
so external tools that drop adapters into the directory show up immediately.
The registry never moves or deletes weight files; deletion is a future op.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...domain.contracts import AdapterRegistry
from ...domain.models import AdapterRecord


@dataclass
class LocalAdapterRegistry(AdapterRegistry):
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def list_adapters(self) -> list[AdapterRecord]:
        records: list[AdapterRecord] = []
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            manifest = child / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                records.append(AdapterRecord.model_validate_json(manifest.read_text()))
            except Exception:
                # Skip malformed manifests; never fail the whole list call.
                continue
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    def get(self, adapter_id: str) -> AdapterRecord | None:
        manifest = self._manifest_path(adapter_id)
        if not manifest.is_file():
            return None
        try:
            return AdapterRecord.model_validate_json(manifest.read_text())
        except Exception:
            return None

    def register(self, record: AdapterRecord) -> AdapterRecord:
        directory = self.path_for(record.id)
        directory.mkdir(parents=True, exist_ok=True)
        # If the caller didn't fix the path, normalise it to point at the
        # registry directory so downstream consumers don't have to recompute.
        canonical = record.model_copy(update={"path": str(directory)})
        (directory / "manifest.json").write_text(canonical.model_dump_json(indent=2))
        return canonical

    def path_for(self, adapter_id: str) -> Path:
        return self.root / adapter_id

    def children_of(self, adapter_id: str | None) -> list[AdapterRecord]:
        return [r for r in self.list_adapters() if r.parent_id == adapter_id]

    def _manifest_path(self, adapter_id: str) -> Path:
        return self.root / adapter_id / "manifest.json"
