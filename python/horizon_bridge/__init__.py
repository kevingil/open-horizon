"""Python worker for the Open Horizon control plane.

Rust owns orchestration, persistence, the API, and the repo rollout loop.
This package owns what stays Python: verifiers environments, trainers
(GRPO-lite, prime-rl), and tokenizer access. It is driven over NDJSON on
stdin/stdout by `horizon-bridge` and defines no domain types of its own;
every payload is the JSON the Rust DTOs produce.
"""

__all__ = ["__version__"]
__version__ = "0.3.0"
