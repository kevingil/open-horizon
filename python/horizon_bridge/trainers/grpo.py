"""GRPO-lite trainer.

Real LoRA training over Hugging Face causal-LM checkpoints. Simplified,
single-pass variant of GRPO:

  1. Group `samples` by task id.
  2. Group-relative advantages: a_i = (r_i - mean(r)) / (std(r) + eps).
  3. For each (prompt, action_text, advantage) run a forward pass, take
     the action-token NLL, multiply by the advantage, and step.

No clipping, KL, or reference model: a sound first step that exercises
the loop end-to-end and produces a real LoRA. Heavy deps are imported
lazily so the worker starts without `[train]` installed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from horizon_bridge.trainers import adapter_record


class TokenizerLike:
    def encode_pair(self, prompt: str, response: str) -> tuple[list[int], int]:
        raise NotImplementedError


@dataclass
class Example:
    prompt: str
    response: str
    advantage: float


@dataclass
class GrpoTrainer:
    """Hyperparams (from `run.hyperparams`, all optional): steps=16,
    batch_size=4, lr=1e-5, lora_rank=8, lora_alpha=16, max_seq_len=512."""

    base_model: str = "Qwen/Qwen3-0.6B"
    default_steps: int = 16
    default_batch_size: int = 4
    default_lr: float = 1e-5
    default_lora_rank: int = 8
    default_lora_alpha: int = 16
    default_max_seq_len: int = 512
    model_loader: Callable[[str, dict], tuple[Any, Any]] | None = None
    adapter_saver: Callable[[Any, Path], None] | None = None
    extra_metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> GrpoTrainer:
        return cls(base_model=str(options.get("grpo_base_model") or "Qwen/Qwen3-0.6B"))

    def name(self) -> str:
        return "grpo-lite-v1"

    def train(self, params: dict[str, Any], sink: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
        run = params["run"]
        samples: list[dict[str, Any]] = list(params.get("samples") or [])
        parent = params.get("parent")
        if not samples:
            raise ValueError("GrpoTrainer requires at least one sample")
        hp = run.get("hyperparams") or {}
        steps = int(hp.get("steps", self.default_steps))
        batch_size = int(hp.get("batch_size", self.default_batch_size))
        lr = float(hp.get("lr", self.default_lr))
        lora_rank = int(hp.get("lora_rank", self.default_lora_rank))
        lora_alpha = int(hp.get("lora_alpha", self.default_lora_alpha))
        max_seq_len = int(hp.get("max_seq_len", self.default_max_seq_len))

        examples = build_examples(samples)
        baseline_reward = sum(s["reward"]["terminal_reward"] for s in samples) / len(samples)

        loader = self.model_loader or _load_hf_model
        model, tokenizer = loader(
            self.base_model,
            {"lora_rank": lora_rank, "lora_alpha": lora_alpha, "parent_path": parent["path"] if parent else None},
        )

        import torch  # lazy

        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
        for step in range(steps):
            batch = sample_batch(examples, batch_size, step)
            loss_tensor, mean_adv = weighted_nll_step(model, tokenizer, batch, max_seq_len=max_seq_len)
            opt.zero_grad()
            loss_tensor.backward()
            opt.step()
            sink(
                "metric",
                {
                    "step": step,
                    "loss": float(loss_tensor.detach().item()),
                    "mean_reward": baseline_reward,
                    "kl": None,
                    "extra": {"mean_advantage": float(mean_adv)},
                },
            )

        target_dir = Path(params["adapter_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)
        (self.adapter_saver or _save_peft_adapter)(model, target_dir)
        return adapter_record(
            adapter_id=str(params["new_adapter_id"]),
            path=str(target_dir),
            base_model=self.base_model,
            parent=parent,
            training_run_id=str(run["id"]),
            tags=["grpo-lite"],
            metadata={
                "trainer": self.name(),
                "samples": str(len(samples)),
                "examples": str(len(examples)),
                "baseline_reward": f"{baseline_reward:.4f}",
                **self.extra_metadata,
            },
        )


def build_examples(samples: list[dict[str, Any]]) -> list[Example]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        grouped.setdefault(s["task"]["id"], []).append(s)
    examples: list[Example] = []
    for group in grouped.values():
        rewards = [s["reward"]["terminal_reward"] for s in group]
        if len(rewards) > 1:
            mean = sum(rewards) / len(rewards)
            variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            std = variance**0.5 or 1.0
            advantages = [(r - mean) / std for r in rewards]
        else:
            # Single-sample group: weak signal re-centred at 0.
            advantages = [rewards[0] - 0.5]
        for sample, adv in zip(group, advantages, strict=True):
            prompt = prompt_text(sample["task"])
            for action_text in action_texts(sample):
                examples.append(Example(prompt=prompt, response=action_text, advantage=adv))
    return examples


def prompt_text(task: dict[str, Any]) -> str:
    criteria = ", ".join(task.get("success_criteria") or []) or "(none)"
    return f"Task: {task['prompt']}\nSuccess criteria: {criteria}\n"


def action_texts(detail: dict[str, Any]) -> list[str]:
    return [s["content"] for s in detail["trajectory"]["steps"] if s.get("actor") == "policy"]


def sample_batch(examples: list[Example], n: int, step: int) -> list[Example]:
    if n >= len(examples):
        return list(examples)
    start = (step * n) % len(examples)
    return [examples[(start + i) % len(examples)] for i in range(n)]


def weighted_nll_step(model: Any, tokenizer: TokenizerLike, batch: list[Example], *, max_seq_len: int) -> tuple[Any, float]:
    """One forward + advantage-weighted NLL across `batch`.
    Returns (scalar loss tensor, mean advantage)."""
    import torch  # lazy

    losses: list[Any] = []
    advs: list[float] = []
    for example in batch:
        ids, prompt_len = tokenizer.encode_pair(example.prompt, example.response)
        ids = ids[:max_seq_len]
        if prompt_len >= len(ids):
            continue
        input_ids = torch.tensor([ids], dtype=torch.long)
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        outputs = model(input_ids=input_ids, labels=labels)
        # loss = A * nll: positive advantage descends NLL, negative climbs it.
        losses.append(float(example.advantage) * outputs.loss)
        advs.append(float(example.advantage))
    if not losses:
        return torch.tensor(0.0, requires_grad=True), 0.0
    return torch.stack(losses).mean(), sum(advs) / len(advs)


def _load_hf_model(base_model: str, opts: dict) -> tuple[Any, TokenizerLike]:
    from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

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

    class _HFTokenizerAdapter(TokenizerLike):
        def encode_pair(self, prompt: str, response: str) -> tuple[list[int], int]:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            return prompt_ids + response_ids, len(prompt_ids)

    return model, _HFTokenizerAdapter()


def _save_peft_adapter(model: Any, target_dir: Path) -> None:
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(target_dir))
    else:
        import torch

        torch.save(model.state_dict(), target_dir / "weights.pt")
