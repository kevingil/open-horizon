"""Deterministic stand-in Trainer.

Produces a synthetic loss curve that decreases as a function of mean sample
reward, copies (or stubs) the parent adapter's bytes into a new directory,
emits per-step TrainingMetricPoint via the on_metric callback, and registers
the new AdapterRecord. Lets Phase 3 orchestration be tested end-to-end
without ML deps. The real GRPOTrainer in 3.6 implements the same contract.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from domain.contracts import AdapterRegistry, MetricCallback, Trainer
from domain.models import (
    AdapterRecord,
    RunDetail,
    TrainingMetricPoint,
    TrainingRunRecord,
)


@dataclass
class StubTrainer(Trainer):
    """Synthetic trainer used as the default backend.

    Hyperparam knobs honoured (all optional):
    - steps: int = 8        -- number of metric points emitted
    - step_delay_s: float = 0  -- sleep between steps; 0 in tests, >0 in dev to make the UI feel real
    """

    default_steps: int = 8
    default_step_delay_s: float = 0.0

    def name(self) -> str:
        return "stub-trainer-v1"

    def train(
        self,
        run: TrainingRunRecord,
        samples: list[RunDetail],
        parent: AdapterRecord | None,
        on_metric: MetricCallback,
        adapters: AdapterRegistry,
    ) -> AdapterRecord:
        steps = int(run.hyperparams.get("steps", self.default_steps))
        delay = float(run.hyperparams.get("step_delay_s", self.default_step_delay_s))
        baseline_reward = _mean_reward(samples)

        # Loss curve: starts high, decays toward (1 - baseline_reward) so a
        # batch with already-good rewards sees a smaller loss floor.
        floor = max(0.05, 1.0 - baseline_reward)
        for step in range(steps):
            decay = (steps - step) / steps
            loss = floor + (1.0 - floor) * decay
            mean_reward = baseline_reward + (1 - decay) * 0.05
            kl = 0.02 * decay
            on_metric(
                TrainingMetricPoint(
                    step=step,
                    loss=round(loss, 4),
                    mean_reward=round(mean_reward, 4),
                    kl=round(kl, 4),
                )
            )
            if delay > 0:
                time.sleep(delay)

        new_id = f"adapter-{uuid4().hex[:8]}"
        target_dir = adapters.path_for(new_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        _copy_parent_weights(parent, target_dir)

        adapter = AdapterRecord(
            id=new_id,
            parent_id=parent.id if parent else None,
            base_model=parent.base_model if parent else "stub:base",
            training_run_id=run.id,
            path=str(target_dir),
            tags=["stub"],
            metadata={
                "trainer": self.name(),
                "samples": str(len(samples)),
                "baseline_reward": f"{baseline_reward:.4f}",
            },
        )
        return adapters.register(adapter)


def _mean_reward(samples: list[RunDetail]) -> float:
    if not samples:
        return 0.0
    return sum(s.reward.terminal_reward for s in samples) / len(samples)


def _copy_parent_weights(parent: AdapterRecord | None, target_dir: Path) -> None:
    """Best-effort copy of parent weight files. Stubs from-scratch when the
    parent has none yet; real trainers will overwrite this."""
    if parent is None:
        (target_dir / "adapter.stub").write_text("stub-adapter\n")
        return
    parent_dir = Path(parent.path)
    if not parent_dir.is_dir():
        (target_dir / "adapter.stub").write_text("stub-adapter\n")
        return
    for src in parent_dir.iterdir():
        if src.name == "manifest.json":
            continue
        dst = target_dir / src.name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
