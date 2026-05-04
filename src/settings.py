from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class PolicyProfile(BaseModel):
    """A named (base_url, api_key, model) triple a rollout can request.

    Per-run inference selection without per-run free-form configs:
    profiles are defined once at startup and rollouts opt into one by
    name (`RolloutRequest.policy_profile`). Adapter availability and
    paid-provider gating both happen at request time, never mid-rollout.

    `routes_to` decides whether the rollout uses the verifiers loop
    (long-horizon agentic envs) or the in-house repo loop (sandboxed
    shell against a snapshotted git checkout).
    """

    backend: Literal["openai-compat"] = "openai-compat"
    base_url: str
    model: str
    # api_key resolution order: literal `api_key` (discouraged) ->
    # os.environ[api_key_env] -> "not-needed" (for local providers).
    api_key_env: str | None = None
    api_key: SecretStr | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    routes_to: Literal["verifiers", "repo"] = "verifiers"
    # Repo-path knobs (consulted when routes_to="repo").
    max_output_tokens: int = Field(default=2048, ge=1)
    max_retries: int = Field(default=3, ge=0)
    # Verifiers-path knobs (consulted when routes_to="verifiers").
    env_id: str = "vf-math"
    env_args: dict[str, Any] = Field(default_factory=dict)
    max_concurrent: int = Field(default=4, ge=1)
    rollout_timeout_s: float | None = None


class Settings(BaseSettings):
    """Runtime configuration, env-driven. See `.env.example` for keys."""

    model_config = SettingsConfigDict(
        env_prefix="RL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    workspace_root: Path = Field(default=Path("."))
    artifacts_dir: Path = Field(default=Path("./artifacts"))

    # OpenAI-compat policy settings. base_url + api_key + model is the
    # "what server, what model" triple that selects a provider.
    # Phase B removed RL_POLICY_BACKEND; the openai-compat client is now
    # the only production option (used by both the verifiers path and
    # the repo path), and tests inject FakeOpenAI directly.
    llm_api_key: SecretStr | None = Field(default=None)
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_model: str = Field(default="gpt-5.4-mini")
    llm_max_output_tokens: int = Field(default=2048, ge=1)
    llm_max_retries: int = Field(default=3, ge=0)

    # Named policy profiles. Empty = legacy single-default mode (the
    # llm_* fields above synthesize a "default" profile at bootstrap).
    # When set, RolloutRequest.policy_profile picks one by name; unknown
    # profiles get rejected at request time, never mid-rollout.
    # JSON-encoded in the env, e.g.
    #   RL_POLICY_PROFILES='{"oai": {"base_url": "https://api.openai.com/v1", "model": "gpt-5.4-mini", "api_key_env": "OPENAI_API_KEY"}}'
    policy_profiles: dict[str, PolicyProfile] = Field(default_factory=dict)
    default_policy_profile: str = Field(default="default")

    # Artifact store backend: "memory" or "sqlite".
    store_backend: str = Field(default="memory")

    # Environment runner backend: "verifiers" or "repo".
    # - verifiers (default): delegate the multi-turn rollout loop to the
    #   verifiers framework. Production path for long-horizon agentic envs.
    #   Requires `pip install -e .[envs]`.
    # - repo: in-house tool-call loop against a sandboxed tempdir snapshot
    #   of a real git checkout. Used when verifiers can't help (e.g.
    #   "agent fixes a real bug in this checkout").
    # The "simulated" stub backend was deleted in Phase A; tests now use
    # repo or fake the verifiers runner directly.
    env_backend: str = Field(default="verifiers")

    # Repo environment guard rails
    env_command_timeout_s: float = Field(default=10.0, gt=0)
    env_max_output_bytes: int = Field(default=16_384, gt=0)

    # verifiers backend (only consulted when RL_ENV_BACKEND=verifiers).
    # `env_args` is JSON-encoded in the env var, e.g.
    #   RL_VERIFIERS_ENV_ARGS='{"max_recursion": 4}'
    verifiers_env_id: str = Field(default="vf-math")
    verifiers_env_args: dict[str, Any] = Field(default_factory=dict)
    verifiers_max_concurrent: int = Field(default=4, ge=1)
    verifiers_rollout_timeout_s: float | None = Field(default=None)

    # Sandbox for tool commands: "none" (host subprocess) or "docker".
    # Docker falls back to none if the daemon is unavailable.
    env_sandbox: str = Field(default="none")
    sandbox_image: str = Field(default="python:3.11-slim")

    # Concurrency
    max_parallel_rollouts: int = Field(default=4, ge=1)

    # Cost controls
    max_tokens_per_run: int = Field(default=100_000, ge=1)
    daily_budget_usd: float = Field(default=5.0, ge=0)
    budget_window_hours: float = Field(default=24.0, gt=0)

    # Training
    trainer_backend: str = Field(default="stub")  # "stub" (default) or "grpo"
    training_store_backend: str = Field(default="memory")  # "memory" or "sqlite"
    adapters_dir: Path = Field(default=Path("./artifacts/adapters"))
    train_step_delay_s: float = Field(default=0.0, ge=0)
    # GRPO trainer (only consulted when RL_TRAINER_BACKEND=grpo).
    grpo_base_model: str = Field(default="Qwen/Qwen3-0.6B")

    # SGLang admin endpoints. When sglang_admin_url is set and
    # sglang_autoload_lora is true, every published adapter is hot-loaded
    # into the running SGLang server via /load_lora_adapter so eval can
    # immediately address it as `sglang:<adapter_id>`.
    sglang_admin_url: str | None = Field(default=None)
    sglang_autoload_lora: bool = Field(default=False)

    # Observability
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )


def get_settings() -> Settings:
    return Settings()
