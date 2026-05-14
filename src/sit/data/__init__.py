"""Data layer: ticker universes, price ingestion, returns, preprocessing."""

from __future__ import annotations

from sit.data.loader import IndexDataLoader, SP500DataLoader
from sit.data.market_caps import (
    fetch_market_caps,
    market_caps_array,
    price_proxy_caps,
)
from sit.data.universes import (
    INDEX_METADATA,
    UNIVERSE_REGISTRY,
    get_universe,
    nasdaq100,
    nifty50,
    russell2000,
    sp500,
    supported_universes,
)

__all__ = [
    "INDEX_METADATA",
    "IndexDataLoader",
    "SP500DataLoader",
    "UNIVERSE_REGISTRY",
    "fetch_market_caps",
    "get_universe",
    "market_caps_array",
    "nasdaq100",
    "nifty50",
    "price_proxy_caps",
    "russell2000",
    "sp500",
    "supported_universes",
]
