"""Trainer dispatch. Rust sends the full training context (run record,
sample runs, turn-training rows, parent adapter, target directory) and
receives an AdapterRecord back. Metrics stream through `sink`."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def train(params: dict[str, Any], sink: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
    trainer = str(params.get("trainer", ""))
    if trainer == "grpo":
        from horizon_bridge.trainers.grpo import GrpoTrainer

        return {"adapter": GrpoTrainer.from_options(params.get("options") or {}).train(params, sink)}
    if trainer == "prime-rl":
        from horizon_bridge.trainers.prime_rl import PrimeRLTrainer

        return {"adapter": PrimeRLTrainer.from_options(params.get("options") or {}).train(params, sink)}
    raise ValueError(f"unknown trainer backend: {trainer!r} (bridge supports grpo, prime-rl)")


def adapter_record(
    *,
    adapter_id: str,
    path: str,
    base_model: str,
    parent: dict[str, Any] | None,
    training_run_id: str,
    tags: list[str],
    metadata: dict[str, str],
) -> dict[str, Any]:
    """Shape of `horizon_core::models::AdapterRecord`."""
    from datetime import UTC, datetime

    return {
        "id": adapter_id,
        "parent_id": parent["id"] if parent else None,
        "base_model": base_model,
        "training_run_id": training_run_id,
        "eval_score": None,
        "path": path,
        "tags": tags,
        "metadata": metadata,
        "created_at": datetime.now(UTC).isoformat(),
    }
