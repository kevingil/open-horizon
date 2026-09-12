from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from horizon_bridge.trainers.prime_rl import DEFAULT_METRIC_PATTERN, PrimeRLTrainer, write_dataset


def _sample(run_id: str, reward: float) -> dict:
    return {
        "manifest": {"id": run_id},
        "task": {"id": f"task-{run_id}", "prompt": "solve", "success_criteria": ["done"]},
        "trajectory": {"id": f"traj-{run_id}", "task_id": f"task-{run_id}", "steps": []},
        "reward": {"trajectory_id": f"traj-{run_id}", "terminal_reward": reward},
    }


def _rows(run_id: str, turns: int) -> list[dict]:
    return [
        {
            "id": f"tt-{run_id}-{i}", "run_id": run_id, "step_index": i * 2,
            "prompt_ids": [1, 2, 3], "attention_mask": [1, 1, 1, 1, 1], "loss_mask": [0, 0, 0, 1, 1],
            "sampling_args": {"max_tokens": 64}, "model_name": "gpt-5.4-mini", "token_count": 5,
        }
        for i in range(turns)
    ]


def test_dataset_writer_emits_one_line_per_turn(tmp_path: Path) -> None:
    out = tmp_path / "ds.jsonl"
    n = write_dataset([_sample("run-A", 0.7)], {"run-A": _rows("run-A", 3)}, out)
    assert n == 3
    lines = out.read_text().strip().split("\n")
    first = json.loads(lines[0])
    assert first["run_id"] == "run-A"
    assert first["reward"] == pytest.approx(0.7)
    assert first["loss_mask"] == [0, 0, 0, 1, 1]


def test_dataset_writer_skips_runs_without_rows(tmp_path: Path) -> None:
    out = tmp_path / "ds.jsonl"
    assert write_dataset([_sample("bare", 0.0)], {}, out) == 0
    assert out.read_text() == ""


def test_render_config_default_and_template(tmp_path: Path) -> None:
    run = {"id": "trun-1", "hyperparams": {"learning_rate": 1e-5, "batch_size": 4, "use_lora": True}}
    trainer = PrimeRLTrainer(base_model="Qwen/Qwen3-8B")
    cfg = trainer.render_config(workdir=tmp_path, dataset_path=tmp_path / "ds.jsonl", output_dir=tmp_path / "out", run=run, parent=None)
    body = cfg.read_text()
    assert 'base_model = "Qwen/Qwen3-8B"' in body
    assert "learning_rate = 1e-05" in body and "batch_size = 4" in body and "use_lora = true" in body

    template = tmp_path / "tpl.toml"
    template.write_text("model = '{{base_model}}'\nparent = '{{parent_adapter_id}}'\n")
    trainer = PrimeRLTrainer(config_template_path=template)
    parent = {"id": "parent-x", "base_model": "Qwen/Qwen3-8B", "path": str(tmp_path / "parent")}
    body = trainer.render_config(workdir=tmp_path, dataset_path=tmp_path / "d", output_dir=tmp_path / "o", run={"id": "trun-2"}, parent=parent).read_text()
    assert "Qwen/Qwen3-8B" in body and "parent-x" in body


def test_metric_pattern_and_streaming() -> None:
    m = DEFAULT_METRIC_PATTERN.search("step=42  loss=0.3142  reward=0.71  kl=0.0024")
    assert m and m.group("step") == "42" and m.group("loss") == "0.3142"
    assert DEFAULT_METRIC_PATTERN.search("Loading dataset...") is None

    trainer = PrimeRLTrainer(metric_pattern=re.compile(r"step=(?P<step>\d+) loss=(?P<loss>[\d.]+)"))
    events: list[tuple[str, dict]] = []
    trainer.stream_metrics(iter(["noise", "step=1 loss=0.5 reward=0.2", "", "step=2 loss=0.25"]), lambda n, d: events.append((n, d)))
    assert [d["step"] for _, d in events] == [1, 2]
    assert events[0][1]["mean_reward"] == pytest.approx(0.2)
    assert events[1][1]["mean_reward"] is None
