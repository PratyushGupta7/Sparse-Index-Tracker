"""Dependency-injection helpers for FastAPI routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Header, HTTPException, Request, status

from sit.api.settings import Settings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Redis (None-safe)
# ---------------------------------------------------------------------------


_redis_client: Any | None = None
_redis_attempted = False


def init_redis(settings: Settings) -> Any | None:
    """Best-effort Redis client. Returns ``None`` if Redis is unavailable."""
    global _redis_client, _redis_attempted
    _redis_attempted = True
    if not settings.redis_url:
        _redis_client = None
        return None
    try:
        import redis

        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=True,
        )
        client.ping()
        _redis_client = client
        return client
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Redis unavailable (%s); falling back to in-process cache.", exc)
        _redis_client = None
        return None


def get_redis() -> Any | None:
    """Return the currently-initialised Redis client (or ``None``)."""
    return _redis_client


def reset_redis() -> None:
    """Test helper."""
    global _redis_client, _redis_attempted
    _redis_client = None
    _redis_attempted = False


# ---------------------------------------------------------------------------
# Auth scaffold (no-op until Phase 6 unless ``require_api_key`` is True)
# ---------------------------------------------------------------------------


def get_optional_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str | None:
    """Auth scaffold — returns the supplied key (or ``None``).

    Per Locked Decision #4 the public API is fully open in this build, so
    this dependency never raises. Flipping ``settings.require_api_key`` to
    ``True`` re-enables enforcement without touching route handlers.
    """
    settings = get_settings()
    if not settings.require_api_key:
        return x_api_key
    if x_api_key is None or x_api_key not in settings.api_keys:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key.",
        )
    return x_api_key


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def _make_limiter() -> Any:
    """Construct a slowapi limiter bound to the current settings."""
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    settings = get_settings()
    return Limiter(
        key_func=get_remote_address,
        default_limits=[settings.rate_limit_default],
        enabled=settings.rate_limits_enabled,
    )


_limiter: Any | None = None


def get_rate_limiter() -> Any:
    """Return the process-wide slowapi ``Limiter`` instance."""
    global _limiter
    if _limiter is None:
        _limiter = _make_limiter()
    return _limiter


def reset_rate_limiter() -> None:
    """Test helper — drop the cached limiter so the next call rebuilds it."""
    global _limiter
    _limiter = None


# ---------------------------------------------------------------------------
# Index validation (kept identical to legacy app.py behaviour)
# ---------------------------------------------------------------------------


DEFAULT_INDEX = "sp500"


def validate_index(name: str | None) -> str:
    from sit.data.universes import supported_universes

    if not name:
        return DEFAULT_INDEX
    key = name.strip().lower().replace(" ", "").replace("-", "")
    if key not in set(supported_universes()):
        raise HTTPException(
            status_code=400,
            detail=(f"Unknown index '{name}'. " f"Supported: {sorted(supported_universes())}."),
        )
    return key


def get_request(request: Request) -> Request:
    """Trivial pass-through — exposed so tests can monkeypatch easily."""
    return request
