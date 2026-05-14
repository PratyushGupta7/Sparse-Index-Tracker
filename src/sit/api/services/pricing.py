"""Pricing service: yfinance wrapper with L1 (in-process) + L2 (Redis) cache.

The cache key is ``(tuple(sorted(tickers)), today_yyyy_mm_dd, period)`` so
identical bursts within the same trading day result in a single yfinance
hit. Redis is optional — if ``get_redis()`` returns ``None`` the L1 cache
still works.
"""

from __future__ import annotations

import json
import logging
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from functools import lru_cache
from io import StringIO
from typing import Any

import pandas as pd

from sit.api.deps import get_redis
from sit.api.settings import get_settings

logger = logging.getLogger(__name__)

_REDIS_PREFIX = "sit:price:"


def _quiet_yf_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
    import yfinance as yf

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return yf.download(*args, **kwargs)


# ---------------------------------------------------------------------------
# In-process L1 cache (5-min granularity via the date key).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _cached_today_prices(tickers_key: tuple[str, ...], date_key: str) -> dict[str, float]:
    """LRU-cached today-prices fetcher. Always goes to yfinance."""
    if len(tickers_key) == 1:
        data = _quiet_yf_download(
            list(tickers_key),
            period="1d",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            row = data["Close"].iloc[-1]
        else:
            row = data["Close"].iloc[-1:]
        return {tickers_key[0]: float(row.iloc[-1] if hasattr(row, "iloc") else row)}

    data = _quiet_yf_download(
        list(tickers_key),
        period="1d",
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        row = data["Close"].iloc[-1]
    else:
        row = data["Close"].iloc[-1:]
    out: dict[str, float] = {}
    for t in tickers_key:
        try:
            raw = row[t]
            if pd.isna(raw):
                continue
            out[t] = float(raw)
        except (KeyError, ValueError, TypeError):
            continue
    return out


def get_latest_prices(tickers: list[str], *, today: str | None = None) -> dict[str, float]:
    """Fetch most-recent close prices for ``tickers``.

    Looks up the L2 (Redis) cache first, then the L1 (LRU) cache, then
    finally yfinance. Successful fetches are written through to L2.
    """
    settings = get_settings()
    if not tickers:
        return {}
    key_tuple = tuple(sorted(set(tickers)))
    date_key = today or datetime.utcnow().strftime("%Y-%m-%d")

    redis_client = get_redis()
    redis_key = f"{_REDIS_PREFIX}{date_key}:{','.join(key_tuple)}"

    if redis_client is not None:
        try:
            blob = redis_client.get(redis_key)
            if blob:
                return {k: float(v) for k, v in json.loads(blob).items()}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Redis GET failed (%s); ignoring.", exc)

    out = _cached_today_prices(key_tuple, date_key)

    if redis_client is not None and out:
        try:
            redis_client.setex(redis_key, settings.redis_ttl_s, json.dumps(out))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Redis SETEX failed (%s); ignoring.", exc)

    return dict(out)


def clear_pricing_caches() -> None:
    """Test helper — drop both the L1 LRU and any Redis price keys."""
    _cached_today_prices.cache_clear()
    redis_client = get_redis()
    if redis_client is not None:
        try:  # pragma: no cover - depends on real Redis
            for k in redis_client.scan_iter(f"{_REDIS_PREFIX}*"):
                redis_client.delete(k)
        except Exception:
            pass


def cache_info() -> dict[str, Any]:
    info = _cached_today_prices.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "currsize": info.currsize,
        "maxsize": info.maxsize,
    }
