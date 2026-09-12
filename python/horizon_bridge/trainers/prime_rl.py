"""prime-rl trainer backend.

Shells out to the `prime-rl` CLI, streams metrics back by regex over
stdout, and collects whatever it wrote into the adapter directory Rust
allocated. Turn-training rows arrive in the request (Rust reads them
from the store), so this module touches no database.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from horizon_bridge.trainers import adapter_record

DEFAULT_METRIC_PATTERN = re.compile(
    r"step\s*=\s*(?P<step>\d+).*?loss\s*=\s*(?P<loss>-?\d+(?:\.\d+)?)", re.IGNORECASE,
)
_REWARD_PATTERN = re.compile(r"reward\s*=\s*(?P<reward>-?\d+(?:\.\d+)?)", re.IGNORECASE)
_KL_PATTERN = re.compile(r"kl\s*=\s*(?P<kl>-?\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class PrimeRLTrainer:
    base_model: str = "Qwen/Qwen3-8B"
    cli_path: str = "prime-rl"
    config_template_path: Path | None = None
    metric_pattern: re.Pattern[str] = field(default=DEFAULT_METRIC_PATTERN)

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> PrimeRLTrainer:
        template = options.get("prime_rl_config_template")
        return cls(
            base_model=str(options.get("prime_rl_base_model") or "Qwen/Qwen3-8B"),
            cli_path=str(options.get("prime_rl_cli") or "prime-rl"),
            config_template_path=Path(template) if template else None,
        )

    def name(self) -> str:
        return "prime-rl"

    def train(self, params: dict[str, Any], sink: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        run = params["run"]
        samples: list[dict[str, Any]] = list(params.get("samples") or [])
        turn_training: dict[str, list[dict[str, Any]]] = dict(params.get("turn_training") or {})
        parent = params.get("parent")
        if not samples:
            raise RuntimeError("prime-rl requires at least one sample run; got an empty batch.")

        with tempfile.TemporaryDirectory(prefix=f"primerl-{run['id']}-") as workdir_str:
            workdir = Path(workdir_str)
            dataset_path = workdir / "dataset.jsonl"
            output_dir = workdir / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            n_turns = write_dataset(samples, turn_training, dataset_path)
            config_path = self.render_config(
                workdir=workdir, dataset_path=dataset_path, output_dir=output_dir, run=run, parent=parent,
            )
            argv = [self.cli_path, "train", "--config", str(config_path)]
            with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=workdir) as proc:
                self.stream_metrics(proc.stdout, sink)
                exit_code = proc.wait()
            if exit_code != 0:
                raise RuntimeError(f"prime-rl exited with code {exit_code}; check the metric stream for the failure mode.")
            target_dir = Path(params["adapter_dir"])
            target_dir.mkdir(parents=True, exist_ok=True)
            collect_adapter(output_dir, target_dir)

        return adapter_record(
            adapter_id=str(params["new_adapter_id"]),
            path=str(target_dir),
            base_model=str(parent["base_model"]) if parent else self.base_model,
            parent=parent,
            training_run_id=str(run["id"]),
            tags=["prime-rl"],
            metadata={"trainer": self.name(), "samples": str(len(samples)), "turns": str(n_turns)},
        )

    def render_config(self, *, workdir: Path, dataset_path: Path, output_dir: Path, run: dict[str, Any], parent: dict[str, Any] | None) -> Path:
        out_path = workdir / "config.toml"
        substitutions = {
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "base_model": str(parent["base_model"]) if parent else self.base_model,
            "training_run_id": str(run["id"]),
            "parent_adapter_id": str(parent["id"]) if parent else "",
            "parent_adapter_path": str(parent["path"]) if parent else "",
        }
        if self.config_template_path is not None:
            rendered = self.config_template_path.read_text()
            for key, value in substitutions.items():
                rendered = rendered.replace("{{" + key + "}}", value)
            out_path.write_text(rendered)
            return out_path
        lines = [
            f'dataset_path = "{dataset_path}"',
            f'output_dir = "{output_dir}"',
            f'base_model = "{substitutions["base_model"]}"',
            f'training_run_id = "{run["id"]}"',
        ]
        if parent is not None:
            lines.append(f'resume_adapter = "{parent["path"]}"')
        for key, value in (run.get("hyperparams") or {}).items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            else:
                lines.append(f"{key} = {value}")
        out_path.write_text("\n".join(lines) + "\n")
        return out_path

    def stream_metrics(self, stream: IO[str] | None, sink: Callable[[str, dict[str, Any]], None]) -> None:
        if stream is None:
            return
        for raw in stream:
            line = raw.rstrip()
            if not line:
                continue
            match = self.metric_pattern.search(line)
            if not match:
                continue
            try:
                step = int(match.group("step"))
                loss = float(match.group("loss"))
            except (KeyError, IndexError, ValueError, TypeError):
                continue
            reward_match = _REWARD_PATTERN.search(line)
            kl_match = _KL_PATTERN.search(line)
            sink(
                "metric",
                {
                    "step": step,
                    "loss": loss,
                    "mean_reward": float(reward_match.group("reward")) if reward_match else None,
                    "kl": float(kl_match.group("kl")) if kl_match else None,
                    "extra": {},
                },
            )


def write_dataset(samples: list[dict[str, Any]], turn_training: dict[str, list[dict[str, Any]]], dataset_path: Path) -> int:
    """One JSONL line per recorded turn: ids, masks, and the trajectory's
    terminal reward shared by every completion token in the turn."""
    n = 0
    with dataset_path.open("w") as fh:
        for sample in samples:
            run_id = sample["manifest"]["id"]
            terminal_reward = sample["reward"]["terminal_reward"]
            for row in turn_training.get(run_id, []):
                fh.write(json.dumps(record_to_line(row, terminal_reward)))
                fh.write("\n")
                n += 1
    return n


def record_to_line(row: dict[str, Any], terminal_reward: float) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "step_index": row["step_index"],
        "model_name": row.get("model_name", ""),
        "prompt_ids": row.get("prompt_ids", []),
        "attention_mask": row.get("attention_mask", []),
        "loss_mask": row.get("loss_mask", []),
        "reward": terminal_reward,
        "sampling_args": row.get("sampling_args", {}),
    }


def collect_adapter(output_dir: Path, target_dir: Path) -> None:
    copied = 0
    for src in output_dir.rglob("*"):
        if src.is_dir():
            continue
        dst = target_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        copied += 1
    if copied == 0:
        (target_dir / "PRIME_RL_NO_OUTPUT").write_text(
            "prime-rl finished but produced no files in output_dir;\n"
            "double-check the config's `output_dir` and adapter saving.\n",
        )
