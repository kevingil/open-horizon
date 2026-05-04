"""prime-rl trainer backend.

prime-rl is the native pair with the verifiers framework: it consumes
verifiers Environment rollouts (or pre-recorded TurnTrainingRecord
rows from our store) and trains a policy via GRPO. This module
implements the existing `Trainer` ABC by shelling out to the
`prime-rl` CLI from a subprocess, streaming metrics back as
TrainingMetricPoint events, and registering the produced adapter.

The integration is intentionally thin: the real prime-rl flags, config
shape, and metric line format are all calibrated at integration time
against the version that's actually installed. Helpers are isolated so
operators can override them per deployment without forking the trainer.

Heavy deps live behind the `[prime-rl]` extra; the import is lazy so
the default install doesn't drag the prime-rl package in.

Required at runtime (when `RL_TRAINER_BACKEND=prime-rl`):
  - `prime-rl` package importable / `prime-rl` CLI on $PATH
  - A populated turn_training table on the artifact store (Phase E)
  - GPU + Torch + flash-attn (per prime-rl's own README)
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

import structlog

from domain.contracts import (
    AdapterRegistry,
    ArtifactStore,
    MetricCallback,
    Trainer,
)
from domain.models import (
    AdapterRecord,
    RunDetail,
    TrainingMetricPoint,
    TrainingRunRecord,
    TurnTrainingRecord,
)

log = structlog.get_logger(__name__)

# Default heuristic. The exact format prime-rl prints depends on
# version + config; override `metric_pattern` per deployment if it
# drifts. The primary pattern only requires `step` + `loss`; reward
# and kl come from secondary searches over the same line so any one
# of them missing doesn't cause the whole line to drop.
DEFAULT_METRIC_PATTERN = re.compile(
    r"step\s*=\s*(?P<step>\d+).*?loss\s*=\s*(?P<loss>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_REWARD_PATTERN = re.compile(
    r"reward\s*=\s*(?P<reward>-?\d+(?:\.\d+)?)", re.IGNORECASE,
)
_KL_PATTERN = re.compile(r"kl\s*=\s*(?P<kl>-?\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class PrimeRLTrainer(Trainer):
    """Shell out to `prime-rl` for the actual training step.

    Constructor knobs:
      - artifact_store: needed to pull TurnTrainingRecord rows for the
        rollouts we're training on
      - base_model: HF id passed to prime-rl as the policy backbone
      - cli_path: the executable; defaults to "prime-rl" on $PATH
      - config_template_path: optional Path to a TOML template; the
        trainer fills in `dataset_path` / `output_dir` / `base_model`
        and writes the result to a tempfile per run
      - metric_pattern: regex over stdout lines; emits TrainingMetricPoint
        on every match
    """

    artifact_store: ArtifactStore
    base_model: str = "Qwen/Qwen3-8B"
    cli_path: str = "prime-rl"
    config_template_path: Path | None = None
    metric_pattern: re.Pattern[str] = field(default=DEFAULT_METRIC_PATTERN)

    def name(self) -> str:
        return "prime-rl"

    def train(
        self,
        run: TrainingRunRecord,
        samples: list[RunDetail],
        parent: AdapterRecord | None,
        on_metric: MetricCallback,
        adapters: AdapterRegistry,
    ) -> AdapterRecord:
        if not samples:
            raise RuntimeError(
                "prime-rl requires at least one sample run; got an empty batch.",
            )

        with tempfile.TemporaryDirectory(prefix=f"primerl-{run.id}-") as workdir_str:
            workdir = Path(workdir_str)
            dataset_path = workdir / "dataset.jsonl"
            output_dir = workdir / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            n_turns = self._write_dataset(samples, dataset_path)
            log.info(
                "prime_rl.dataset.written",
                training_run_id=run.id,
                turns=n_turns,
                samples=len(samples),
                dataset=str(dataset_path),
            )

            config_path = self._render_config(
                workdir=workdir,
                dataset_path=dataset_path,
                output_dir=output_dir,
                run=run,
                parent=parent,
            )

            argv = [self.cli_path, "train", "--config", str(config_path)]
            log.info("prime_rl.subprocess.launch", argv=argv, training_run_id=run.id)
            with subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=workdir,
            ) as proc:
                self._stream_metrics(proc.stdout, on_metric)
                exit_code = proc.wait()
            if exit_code != 0:
                raise RuntimeError(
                    f"prime-rl exited with code {exit_code}; check the metric "
                    "stream above for the failure mode.",
                )

            new_id = f"adapter-{uuid4().hex[:8]}"
            target_dir = adapters.path_for(new_id)
            target_dir.mkdir(parents=True, exist_ok=True)
            self._collect_adapter(output_dir, target_dir)

            adapter = AdapterRecord(
                id=new_id,
                parent_id=parent.id if parent else None,
                base_model=parent.base_model if parent else self.base_model,
                training_run_id=run.id,
                path=str(target_dir),
                tags=["prime-rl"],
                metadata={
                    "trainer": self.name(),
                    "samples": str(len(samples)),
                    "turns": str(n_turns),
                },
            )
            return adapters.register(adapter)

    # ------------------------------------------------------------------ helpers

    def _write_dataset(
        self, samples: list[RunDetail], dataset_path: Path,
    ) -> int:
        """Drain TurnTrainingRecord rows for each sample run into a
        JSONL dataset prime-rl can consume.

        Each line: {prompt_ids, completion_ids, attention_mask, loss_mask,
        reward, run_id, step_index, model_name}. The exact keys prime-rl
        wants vary by version; tweak _record_to_line per deployment.
        """
        n_turns = 0
        with dataset_path.open("w") as fh:
            for sample in samples:
                rows = self.artifact_store.list_turn_training(sample.manifest.id)
                terminal_reward = sample.reward.terminal_reward
                for row in rows:
                    fh.write(json.dumps(self._record_to_line(row, terminal_reward)))
                    fh.write("\n")
                    n_turns += 1
        return n_turns

    @staticmethod
    def _record_to_line(
        row: TurnTrainingRecord, terminal_reward: float,
    ) -> dict[str, Any]:
        return {
            "run_id": row.run_id,
            "step_index": row.step_index,
            "model_name": row.model_name,
            "prompt_ids": row.prompt_ids,
            "attention_mask": row.attention_mask,
            "loss_mask": row.loss_mask,
            # Terminal-only reward signal: every completion-token in this
            # turn shares the trajectory's terminal reward. Trainers that
            # want a finer assignment should compute it offline.
            "reward": terminal_reward,
            "sampling_args": row.sampling_args,
        }

    def _render_config(
        self,
        *,
        workdir: Path,
        dataset_path: Path,
        output_dir: Path,
        run: TrainingRunRecord,
        parent: AdapterRecord | None,
    ) -> Path:
        """Write a TOML config in `workdir`. If a template was provided,
        substitute placeholders; otherwise emit a minimal default.
        """
        out_path = workdir / "config.toml"
        substitutions = {
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "base_model": (parent.base_model if parent else self.base_model),
            "training_run_id": run.id,
            "parent_adapter_id": (parent.id if parent else ""),
            "parent_adapter_path": (parent.path if parent else ""),
        }
        if self.config_template_path is not None:
            template = self.config_template_path.read_text()
            rendered = template
            for key, value in substitutions.items():
                rendered = rendered.replace("{{" + key + "}}", value)
            out_path.write_text(rendered)
            return out_path
        # Fallback config: keep flat to dodge prime-rl version drift on
        # nested table names. Operators with structured configs supply a
        # template instead.
        lines = [
            f'dataset_path = "{dataset_path}"',
            f'output_dir = "{output_dir}"',
            f'base_model = "{substitutions["base_model"]}"',
            f'training_run_id = "{run.id}"',
        ]
        if parent is not None:
            lines.append(f'resume_adapter = "{parent.path}"')
        for key, value in run.hyperparams.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            else:
                lines.append(f"{key} = {value}")
        out_path.write_text("\n".join(lines) + "\n")
        return out_path

    def _stream_metrics(
        self, stream: IO[str] | None, on_metric: MetricCallback,
    ) -> None:
        """Pump prime-rl stdout, emitting TrainingMetricPoint on every
        line that matches `metric_pattern`. Non-matching lines are logged
        at debug level so operators can verify the regex still fits."""
        if stream is None:
            return
        for raw in stream:
            line = raw.rstrip()
            if not line:
                continue
            match = self.metric_pattern.search(line)
            if not match:
                log.debug("prime_rl.line", line=line[:200])
                continue
            try:
                step = int(match.group("step"))
                loss = float(match.group("loss"))
            except (KeyError, IndexError, ValueError, TypeError):
                continue
            reward_match = _REWARD_PATTERN.search(line)
            kl_match = _KL_PATTERN.search(line)
            on_metric(
                TrainingMetricPoint(
                    step=step,
                    loss=loss,
                    mean_reward=(
                        float(reward_match.group("reward"))
                        if reward_match else None
                    ),
                    kl=float(kl_match.group("kl")) if kl_match else None,
                ),
            )

    @staticmethod
    def _collect_adapter(output_dir: Path, target_dir: Path) -> None:
        """Copy files prime-rl wrote into the registry's adapter dir.

        prime-rl's exact output layout depends on its version - we copy
        whatever it left behind, ignoring directories so we don't pull in
        nested tensorboard / wandb caches by accident."""
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
            # Tag the dir so the registry doesn't think it's empty;
            # surfacing a stub instead of failing the run silently.
            (target_dir / "PRIME_RL_NO_OUTPUT").write_text(
                "prime-rl finished but produced no files in output_dir;\n"
                "double-check the config's `output_dir` and adapter saving.\n",
            )
