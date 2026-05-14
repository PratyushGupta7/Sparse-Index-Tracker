"""sit/data/market_caps.py — fetch + cache market capitalisation data.

The naive cap-weighted baselines need a market-cap snapshot for every ticker
in the universe. ``yfinance.Ticker(t).info`` exposes ``marketCap`` but it is

* slow (one HTTP round-trip per ticker — ~5 minutes for the full S&P 500),
* flaky (occasional 404s, ratelimits, occasional ``None`` for ADRs),
* wasteful when re-running the comparison many times during development.

So we wrap fetching with an aggressive on-disk JSON cache. A second fallback
("price proxy") lets the comparison driver keep running if the network is
unreachable — using ``last_close × shares_outstanding`` (or just ``last_close``
when shares aren't available) as an approximate cap.

Cache contract
--------------
The cache file lives at ``data/market_caps.json`` (configurable). Schema::

    {
        "fetched_at_utc": "2026-05-11T12:34:56Z",
        "source": "yfinance.Ticker.info",
        "caps": {
            "AAPL": 3415000000000.0,
            "MSFT": 3120000000000.0,
            ...
        }
    }

Tickers missing from the cache are fetched fresh (incremental update).
Tickers whose cached value is ``None`` (last fetch failed) are re-tried.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

from sit.paths import DATA_DIR

DEFAULT_CACHE_PATH = DATA_DIR / "market_caps.json"


def load_cache(cache_path: Path | None = None) -> dict[str, float | None]:
    """Read the on-disk cache; return empty dict if absent or malformed."""
    cache_path = cache_path or DEFAULT_CACHE_PATH
    if not cache_path.is_file():
        return {}
    try:
        with cache_path.open("r") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "caps" in payload:
            return dict(payload["caps"])
        # Legacy flat dict
        if isinstance(payload, dict):
            return dict(payload)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_cache(
    caps: dict[str, float | None],
    cache_path: Path | None = None,
    source: str = "yfinance.Ticker.info",
) -> Path:
    """Persist a market-cap dict atomically to disk."""
    cache_path = cache_path or DEFAULT_CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "caps": caps,
    }
    tmp = cache_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, cache_path)
    return cache_path


def _fetch_one_yfinance(ticker: str) -> float | None:
    """Best-effort fetch via yfinance. Returns ``None`` on any failure."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info
    except Exception:  # pragma: no cover (network-dependent)
        return None
    if not isinstance(info, dict):
        return None
    cap = info.get("marketCap")
    if cap is None:
        # Some ADRs / smaller listings only expose enterpriseValue or
        # sharesOutstanding × regularMarketPrice — try those next.
        shares = info.get("sharesOutstanding")
        price = info.get("regularMarketPrice") or info.get("previousClose")
        if shares is not None and price is not None:
            try:
                return float(shares) * float(price)
            except (TypeError, ValueError):
                return None
    try:
        return float(cap) if cap is not None else None
    except (TypeError, ValueError):
        return None


def fetch_market_caps(
    tickers: Iterable[str],
    *,
    cache_path: Path | None = None,
    use_cache: bool = True,
    update_cache: bool = True,
    refetch_missing: bool = True,
    progress: bool = False,
) -> dict[str, float | None]:
    """Return a ``{ticker: market_cap_or_None}`` dict, hitting the cache first.

    Parameters
    ----------
    tickers
        Iterable of ticker symbols to fetch.
    cache_path
        Cache file location. ``None`` defaults to ``data/market_caps.json``.
    use_cache
        Read existing cached values first.
    update_cache
        Persist newly-fetched values back to the cache file.
    refetch_missing
        If ``True``, re-attempt tickers whose cached value is ``None``.
    progress
        Show a tqdm bar over the fetch loop (only fires for cache-misses).
    """
    tickers_list = list(tickers)
    cached = load_cache(cache_path) if use_cache else {}

    to_fetch: list[str] = []
    for t in tickers_list:
        if t in cached and cached[t] is not None:
            continue
        if t in cached and cached[t] is None and not refetch_missing:
            continue
        to_fetch.append(t)

    if to_fetch:
        iterable: Iterable[str] = to_fetch
        if progress:
            try:  # pragma: no cover
                from tqdm import tqdm  # type: ignore[import-untyped]

                iterable = tqdm(to_fetch, desc="market_caps", unit="ticker")
            except ImportError:  # pragma: no cover
                pass
        for t in iterable:
            cached[t] = _fetch_one_yfinance(t)

        if update_cache:
            save_cache(cached, cache_path)

    return {t: cached.get(t) for t in tickers_list}


def market_caps_array(
    tickers: list[str],
    *,
    cache_path: Path | None = None,
    fallback: float = float("nan"),
) -> np.ndarray:
    """Return market caps as a NumPy array aligned with ``tickers``.

    Missing values become ``fallback`` (default ``NaN`` so downstream code can
    decide whether to interpolate, error, or use a proxy).
    """
    caps_dict = fetch_market_caps(tickers, cache_path=cache_path, use_cache=True, update_cache=True)
    return np.asarray(
        [caps_dict.get(t) if caps_dict.get(t) is not None else fallback for t in tickers],
        dtype=np.float64,
    )


def price_proxy_caps(
    last_prices: np.ndarray,
    shares_outstanding: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a price-based proxy for market cap when network data is unavailable.

    If ``shares_outstanding`` is provided, returns ``last_prices * shares``;
    otherwise returns ``last_prices`` (a *price-weighted* proxy that is
    accurate up to a per-stock multiplier).
    """
    last_prices = np.asarray(last_prices, dtype=np.float64)
    if shares_outstanding is None:
        return last_prices
    shares = np.asarray(shares_outstanding, dtype=np.float64)
    if shares.shape != last_prices.shape:
        raise ValueError("shares_outstanding must align with last_prices.")
    return last_prices * shares


__all__ = [
    "DEFAULT_CACHE_PATH",
    "fetch_market_caps",
    "load_cache",
    "market_caps_array",
    "price_proxy_caps",
    "save_cache",
]
