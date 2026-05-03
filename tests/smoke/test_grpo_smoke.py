"""Opt-in smoke test: one real GRPO-lite step against a tiny HF checkpoint.

Enabled only when RL_TRAINER_SMOKE=1 is set. Requires `[train]` extras
(`pip install -e '.[train]'`) and network access to download the model.

    RL_TRAINER_SMOKE=1 pytest tests/smoke/test_grpo_smoke.py -v

Override the model with RL_TRAINER_SMOKE_MODEL when local-only checkpoints
are needed (e.g. for an air-gapped CI cache).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

SMOKE_ENABLED = os.environ.get("RL_TRAINER_SMOKE") == "1"
SMOKE_MODEL = os.environ.get("RL_TRAINER_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
pytestmark = pytest.mark.skipif(not SMOKE_ENABLED, reason="RL_TRAINER_SMOKE not set")


def test_real_grpo_step_produces_adapter(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    from domain.models import (
        ArtifactRecord,
        RewardRecord,
        RunDetail,
        RunManifest,
        RunStatus,
        TaskSpec,
        ToolPermission,
        TrainingRunRecord,
        TrajectoryRecord,
        TrajectoryStep,
    )
    from infrastructure.adapters.local import LocalAdapterRegistry
    from infrastructure.training.grpo import GrpoTrainer

    def _detail(task_id: str, response: str, reward: float) -> RunDetail:
        task = TaskSpec(
            id=task_id, prompt=f"Prompt {task_id}", repo_snapshot=".",
            tool_permissions=[ToolPermission.read], horizon=2,
            success_criteria=["ok"],
        )
        steps = [TrajectoryStep(index=0, actor="policy", kind="action", content=response)]
        traj = TrajectoryRecord(id=f"{task_id}-traj", task_id=task_id, steps=steps)
        rew = RewardRecord(trajectory_id=traj.id, terminal_reward=reward, step_rewards=[], provenance="t")
        manifest = RunManifest(
            id=f"run-{task_id}-{int(reward*100)}",
            model_id="m", dataset_slice="s", infra_target="cpu-smoke",
            seed=1, status=RunStatus.completed, estimated_cost_usd=0.0,
        )
        return RunDetail(
            manifest=manifest, task=task, trajectory=traj, reward=rew,
            artifacts=[ArtifactRecord(name="a", kind="manifest", path="p")],
        )

    registry = LocalAdapterRegistry(root=tmp_path / "adapters")
    trainer = GrpoTrainer(base_model=SMOKE_MODEL)
    samples = [
        _detail("task-a", "Yes the README is alive.", reward=0.9),
        _detail("task-a", "I don't know.", reward=0.1),
    ]
    record = TrainingRunRecord(
        id="trun-smoke",
        hyperparams={"steps": 2, "batch_size": 2, "lr": 1e-5, "max_seq_len": 128},
    )

    metrics = []
    adapter = trainer.train(record, samples, parent=None, on_metric=metrics.append, adapters=registry)

    assert len(metrics) == 2
    assert "grpo-lite" in adapter.tags
    assert (Path(adapter.path) / "adapter_config.json").exists()
