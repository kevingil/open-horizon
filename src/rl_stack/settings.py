from __future__ import annotations

from pathlib import Path

from pydantic import Field
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

    # Policy backend selection: "static" for now; "claude" and "vllm" land in Phase 1+.
    policy_backend: str = Field(default="static")

    # Concurrency
    max_parallel_rollouts: int = Field(default=4, ge=1)

    # Observability
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    # CORS origins for the FastAPI app.
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )


def get_settings() -> Settings:
    return Settings()
