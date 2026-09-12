from __future__ import annotations

import pytest

from horizon_bridge.trainers.grpo import Example, build_examples


def _detail(task_id: str, action: str, reward: float) -> dict:
    return {
        "manifest": {"id": f"run-{task_id}-{int(reward * 100)}"},
        "task": {"id": task_id, "prompt": f"Prompt {task_id}", "success_criteria": ["ok"]},
        "trajectory": {"id": f"{task_id}-traj", "task_id": task_id, "steps": [{"index": 0, "actor": "policy", "kind": "action", "content": action}]},
        "reward": {"trajectory_id": f"{task_id}-traj", "terminal_reward": reward},
    }


def test_build_examples_normalises_within_group() -> None:
    examples = build_examples([_detail("a", "high", 0.9), _detail("a", "low", 0.1), _detail("a", "mid", 0.5)])
    advs = sorted(e.advantage for e in examples)
    assert advs[0] < -0.5 and advs[-1] > 0.5
    assert abs(sum(advs)) < 1e-6


def test_build_examples_groups_by_task() -> None:
    examples = build_examples([_detail("a", "x", 0.9), _detail("a", "y", 0.1), _detail("b", "z", 0.5)])
    by = {e.response: e.advantage for e in examples}
    assert by["x"] > 0 > by["y"]
    assert by["z"] == pytest.approx(0.0)


def test_weighted_nll_moves_in_advantage_direction() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from horizon_bridge.trainers.grpo import TokenizerLike, weighted_nll_step

    class CharTokenizer(TokenizerLike):
        def encode_pair(self, prompt, response):
            enc = lambda t: [(ord(c) % 63) + 1 for c in t]  # noqa: E731
            p = [*enc(prompt), 0]
            return [*p, *enc(response)], len(p)

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 16)
            self.lin = nn.Linear(16, 64)

        def forward(self, *, input_ids, labels):
            x = self.lin(self.embed(input_ids))
            loss = nn.functional.cross_entropy(x[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1), ignore_index=-100)
            return type("Out", (), {"loss": loss})()

    def nll(model, tok, ex):
        ids, plen = tok.encode_pair(ex.prompt, ex.response)
        t = torch.tensor([ids])
        labels = t.clone()
        labels[0, :plen] = -100
        return float(model(input_ids=t, labels=labels).loss.detach())

    for adv, expect_lower in ((1.0, True), (-1.0, False)):
        torch.manual_seed(0)
        model, tok = TinyLM(), CharTokenizer()
        ex = Example(prompt="ABC", response="XYZ", advantage=adv)
        before = nll(model, tok, ex)
        loss, _ = weighted_nll_step(model, tok, [ex], max_seq_len=64)
        opt = torch.optim.SGD(model.parameters(), lr=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
        after = nll(model, tok, ex)
        assert (after < before) is expect_lower
