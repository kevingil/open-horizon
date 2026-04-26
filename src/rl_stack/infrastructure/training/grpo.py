"""GRPO-lite trainer.

Real LoRA training over Hugging Face causal-LM checkpoints. The math is a
simplified, single-pass variant of GRPO:

  1. Group `samples` by task id.
  2. Compute group-relative advantages: a_i = (r_i - mean(r)) / (std(r) + eps).
  3. For each (task.prompt, action_text, advantage) triple, run a forward
     pass, take the action-token negative-log-likelihood, multiply by
     -advantage, and step. (Higher-than-mean reward => positive advantage =>
     descend the NLL; lower-than-mean reward => climb away from that action.)

That's not full GRPO with clipping/KL/reference, but it's a sound first step
that exercises the whole loop end-to-end and produces a real LoRA. Drop in
TRL's GRPOTrainer when you want full PPO machinery; the contract here stays
the same.

Heavy deps (`torch`, `transformers`, `peft`) are imported lazily so the
module is safe to import without `[train]` installed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from ...domain.contracts import AdapterRegistry, MetricCallback, Trainer
from ...domain.models import (
    AdapterRecord,
    RunDetail,
    TaskSpec,
    TrainingMetricPoint,
    TrainingRunRecord,
)

log = structlog.get_logger(__name__)


# Public for tests. A "minimal tokenizer" needs only what _build_examples and
# the training loop call: tokenize a prompt + response into token ids and tell
# us how long the prompt half is.
class TokenizerLike:
    def encode_pair(self, prompt: str, response: str) -> tuple[list[int], int]:
        raise NotImplementedError


@dataclass
class _Example:
    """A single training tuple after grouping + advantage normalisation."""

    prompt: str
    response: str
    advantage: float


@dataclass
class GrpoTrainer(Trainer):
    """LoRA fine-tuner driven by group-relative trajectory rewards.

    Hyperparam knobs (read from `run.hyperparams`, all optional):
      steps           -- number of gradient steps (default 16)
      batch_size      -- examples per step (default 4)
      lr              -- AdamW learning rate (default 1e-5)
      lora_rank       -- LoRA r (default 8)
      lora_alpha      -- LoRA alpha (default 16)
      max_seq_len     -- truncation length (default 512)
    """

    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    default_steps: int = 16
    default_batch_size: int = 4
    default_lr: float = 1e-5
    default_lora_rank: int = 8
    default_lora_alpha: int = 16
    default_max_seq_len: int = 512
    # Overridable for tests (synthetic torch models without HF / network).
    model_loader: Callable[[str, dict], tuple[Any, Any]] | None = None
    adapter_saver: Callable[[Any, Path], None] | None = None
    extra_metadata: dict[str, str] = field(default_factory=dict)

    def name(self) -> str:
        return "grpo-lite-v1"

    def train(
        self,
        run: TrainingRunRecord,
        samples: list[RunDetail],
        parent: AdapterRecord | None,
        on_metric: MetricCallback,
        adapters: AdapterRegistry,
    ) -> AdapterRecord:
        if not samples:
            raise ValueError("GrpoTrainer requires at least one sample")

        steps = int(run.hyperparams.get("steps", self.default_steps))
        batch_size = int(run.hyperparams.get("batch_size", self.default_batch_size))
        lr = float(run.hyperparams.get("lr", self.default_lr))
        lora_rank = int(run.hyperparams.get("lora_rank", self.default_lora_rank))
        lora_alpha = int(run.hyperparams.get("lora_alpha", self.default_lora_alpha))
        max_seq_len = int(run.hyperparams.get("max_seq_len", self.default_max_seq_len))

        examples = _build_examples(samples)
        baseline_reward = sum(s.reward.terminal_reward for s in samples) / len(samples)

        loader = self.model_loader or _load_hf_model
        model, tokenizer = loader(
            self.base_model,
            {"lora_rank": lora_rank, "lora_alpha": lora_alpha, "parent_path": parent.path if parent else None},
        )

        import torch  # lazy

        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=lr,
        )

        for step in range(steps):
            batch = _sample_batch(examples, batch_size, step)
            loss_tensor, mean_adv = _weighted_nll_step(
                model, tokenizer, batch, max_seq_len=max_seq_len,
            )
            opt.zero_grad()
            loss_tensor.backward()
            opt.step()
            on_metric(
                TrainingMetricPoint(
                    step=step,
                    loss=float(loss_tensor.detach().item()),
                    mean_reward=baseline_reward,
                    kl=None,
                    extra={"mean_advantage": float(mean_adv)},
                )
            )

        new_id = f"adapter-{uuid4().hex[:8]}"
        target_dir = adapters.path_for(new_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        saver = self.adapter_saver or _save_peft_adapter
        saver(model, target_dir)

        adapter = AdapterRecord(
            id=new_id,
            parent_id=parent.id if parent else None,
            base_model=self.base_model,
            training_run_id=run.id,
            path=str(target_dir),
            tags=["grpo-lite"],
            metadata={
                "trainer": self.name(),
                "samples": str(len(samples)),
                "examples": str(len(examples)),
                "baseline_reward": f"{baseline_reward:.4f}",
                **self.extra_metadata,
            },
        )
        return adapters.register(adapter)


def _build_examples(samples: list[RunDetail]) -> list[_Example]:
    """Group samples by task, compute group-relative advantages, expand each
    sample into (prompt, action_text, advantage) examples."""
    grouped: dict[str, list[RunDetail]] = {}
    for s in samples:
        grouped.setdefault(s.task.id, []).append(s)

    examples: list[_Example] = []
    for group in grouped.values():
        rewards = [s.reward.terminal_reward for s in group]
        if len(rewards) > 1:
            mean = sum(rewards) / len(rewards)
            variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            std = variance**0.5 or 1.0
            advantages = [(r - mean) / std for r in rewards]
        else:
            # Single-sample group: use the raw reward (re-centred at 0) as a
            # weak signal. Real training should batch enough rollouts per task
            # for groups of size >= 2.
            advantages = [rewards[0] - 0.5]

        for sample, adv in zip(group, advantages, strict=True):
            prompt = _prompt_text(sample.task)
            for action_text in _action_texts(sample):
                examples.append(_Example(prompt=prompt, response=action_text, advantage=adv))
    return examples


def _prompt_text(task: TaskSpec) -> str:
    criteria = ", ".join(task.success_criteria) or "(none)"
    return f"Task: {task.prompt}\nSuccess criteria: {criteria}\n"


def _action_texts(detail: RunDetail) -> list[str]:
    """Pull the policy's action strings out of the trajectory."""
    return [s.content for s in detail.trajectory.steps if s.actor == "policy"]


def _sample_batch(examples: list[_Example], n: int, step: int) -> list[_Example]:
    if n >= len(examples):
        return list(examples)
    # Deterministic round-robin so tests are reproducible.
    start = (step * n) % len(examples)
    return [examples[(start + i) % len(examples)] for i in range(n)]


def _weighted_nll_step(
    model: Any,
    tokenizer: TokenizerLike,
    batch: list[_Example],
    *,
    max_seq_len: int,
) -> tuple[Any, float]:
    """One forward + advantage-weighted negative log-likelihood across `batch`.

    Returns (scalar loss tensor, mean advantage). Tests assert that loss
    decreases when advantages favour a particular response.
    """
    import torch  # lazy

    losses: list[Any] = []
    advs: list[float] = []
    for example in batch:
        ids, prompt_len = tokenizer.encode_pair(example.prompt, example.response)
        ids = ids[:max_seq_len]
        if prompt_len >= len(ids):
            continue  # nothing to score
        input_ids = torch.tensor([ids], dtype=torch.long)
        # Standard causal-LM loss but only on response tokens. We mask prompt
        # positions with -100 and rely on torch.nn.CrossEntropyLoss(ignore_index).
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        outputs = model(input_ids=input_ids, labels=labels)
        nll = outputs.loss
        # Policy-gradient identity (descent form):
        #   loss = A * nll    so optimiser.step() implements gradient ascent on
        #   E[A * log pi] = -E[A * nll].
        # Positive advantage -> minimise NLL (do this response more); negative
        # advantage -> climb NLL (do it less).
        weighted = float(example.advantage) * nll
        losses.append(weighted)
        advs.append(float(example.advantage))

    if not losses:
        return torch.tensor(0.0, requires_grad=True), 0.0
    loss = torch.stack(losses).mean()
    mean_adv = sum(advs) / len(advs)
    return loss, mean_adv


def _load_hf_model(base_model: str, opts: dict) -> tuple[Any, TokenizerLike]:
    """Default loader: transformers + peft LoRA. Lazy-imported."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(base_model)
    parent_path = opts.get("parent_path")
    if parent_path and (Path(parent_path) / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, parent_path, is_trainable=True)
    else:
        config = LoraConfig(
            r=opts.get("lora_rank", 8),
            lora_alpha=opts.get("lora_alpha", 16),
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, config)

    class _HFTokenizerAdapter:
        def encode_pair(self, prompt: str, response: str) -> tuple[list[int], int]:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            return prompt_ids + response_ids, len(prompt_ids)

    return model, _HFTokenizerAdapter()


def _save_peft_adapter(model: Any, target_dir: Path) -> None:
    """Default saver: peft writes adapter_config.json + adapter weights."""
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(target_dir))
    else:
        # Fallback: dump state_dict so synthetic test models still produce a
        # tangible artifact in the registry directory.
        import torch

        torch.save(model.state_dict(), target_dir / "weights.pt")
