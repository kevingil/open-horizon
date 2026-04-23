from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Policy backend: "static" (dev stub) or "claude".
    policy_backend: str = Field(default="static")

    # Claude policy settings
    anthropic_api_key: SecretStr | None = Field(default=None)
    claude_model: str = Field(default="claude-haiku-4-5")
    claude_max_output_tokens: int = Field(default=2048, ge=1)
    claude_max_retries: int = Field(default=3, ge=0)

    # Artifact store backend: "memory" or "sqlite".
    store_backend: str = Field(default="memory")

    # Environment runner backend: "simulated" or "repo".
    env_backend: str = Field(default="simulated")

    # Repo environment guard rails
    env_command_timeout_s: float = Field(default=10.0, gt=0)
    env_max_output_bytes: int = Field(default=16_384, gt=0)

    # Concurrency
    max_parallel_rollouts: int = Field(default=4, ge=1)

    # Cost controls
    max_tokens_per_run: int = Field(default=100_000, ge=1)

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
