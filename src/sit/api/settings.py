"""Runtime settings for the FastAPI app.

Reads from environment variables (and a local ``.env`` if present). Per
Locked Decision #4 the API is fully open in this build — the auth scaffold
exists but does not enforce. Per Locked Decision #1 there is no hard-coded
domain; CORS origins and downstream URLs come from env.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SIT_",
        extra="ignore",
    )

    # --- Environment ---
    env: str = Field("dev", description="Deployment env: dev/staging/prod.")
    app_version: str = Field("1.0.0")
    git_sha: str = Field("unknown")

    # --- Storage ---
    data_dir: Path = Field(REPO_ROOT / "data")
    benchmarks_dir: Path = Field(REPO_ROOT / "benchmarks" / "_results")

    # --- CORS ---
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # --- Redis ---
    redis_url: str | None = Field(None, description="redis://host:port/db; None disables.")
    redis_ttl_s: int = Field(300, description="TTL for L2 price cache.")

    # --- Rate limits (slowapi syntax) ---
    rate_limit_default: str = Field("120/minute")
    rate_limit_invest: str = Field("120/minute")
    rate_limit_invest_live: str = Field("30/minute")
    rate_limits_enabled: bool = Field(True)

    # --- Pricing cache ---
    pricing_lru_size: int = Field(256)

    # --- Live retraining guardrails ---
    live_universe_max_tickers: int = Field(
        120,
        description=(
            "Maximum constituents downloaded for very large live retrains. "
            "Used only when the universe exceeds live_universe_cap_threshold."
        ),
    )
    live_universe_cap_threshold: int = Field(
        750,
        description="Do not cap live retrains below this source-universe size.",
    )

    # --- Observability ---
    applicationinsights_connection_string: str | None = Field(None)

    # --- Auth scaffold (no-op until Phase 6) ---
    api_keys: list[str] = Field(default_factory=list)
    require_api_key: bool = Field(False)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings


def reset_settings() -> None:
    """Test helper: drop the cached settings so the next call rebuilds from env."""
    global _settings
    _settings = None
