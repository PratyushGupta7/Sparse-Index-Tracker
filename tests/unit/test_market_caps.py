"""Tests for sit.data.market_caps (cache + fetch + array conversion)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sit.data.market_caps import (
    fetch_market_caps,
    load_cache,
    market_caps_array,
    price_proxy_caps,
    save_cache,
)


def _tmp_cache(tmp_path: Path) -> Path:
    return tmp_path / "market_caps_test.json"


def test_save_then_load_roundtrip(tmp_path):
    path = _tmp_cache(tmp_path)
    caps = {"AAPL": 3.0e12, "MSFT": 3.1e12, "TSLA": None}
    save_cache(caps, cache_path=path)

    assert path.is_file()
    loaded = load_cache(path)
    assert loaded["AAPL"] == 3.0e12
    assert loaded["MSFT"] == 3.1e12
    assert loaded["TSLA"] is None


def test_load_cache_missing_file_returns_empty(tmp_path):
    assert load_cache(_tmp_cache(tmp_path)) == {}


def test_load_cache_handles_legacy_flat_dict(tmp_path):
    """Older cache files were a bare dict, not wrapped in {caps: …}."""
    path = _tmp_cache(tmp_path)
    path.write_text(json.dumps({"AAPL": 1.0e12, "GOOG": 2.0e12}))
    loaded = load_cache(path)
    assert loaded == {"AAPL": 1.0e12, "GOOG": 2.0e12}


def test_load_cache_handles_corrupt_json(tmp_path):
    path = _tmp_cache(tmp_path)
    path.write_text("{not valid json")
    assert load_cache(path) == {}


def test_fetch_uses_cache_only(tmp_path, monkeypatch):
    """When all tickers are cached we never hit the network."""
    path = _tmp_cache(tmp_path)
    save_cache({"AAPL": 3.0e12, "MSFT": 3.1e12}, cache_path=path)

    # Sabotage yfinance: any fetch attempt would raise.
    def _boom(_ticker):
        raise AssertionError("fetch_market_caps should not have called network")

    monkeypatch.setattr("sit.data.market_caps._fetch_one_yfinance", _boom)

    out = fetch_market_caps(["AAPL", "MSFT"], cache_path=path)
    assert out == {"AAPL": 3.0e12, "MSFT": 3.1e12}


def test_fetch_only_misses_call_network(tmp_path, monkeypatch):
    """Cached tickers are skipped; only missing tickers trigger fetch."""
    path = _tmp_cache(tmp_path)
    save_cache({"AAPL": 3.0e12}, cache_path=path)

    fetched: list[str] = []

    def _stub(ticker):
        fetched.append(ticker)
        return 1.234e12 if ticker == "MSFT" else None

    monkeypatch.setattr("sit.data.market_caps._fetch_one_yfinance", _stub)

    out = fetch_market_caps(["AAPL", "MSFT", "BAD"], cache_path=path)
    assert fetched == ["MSFT", "BAD"]
    assert out["AAPL"] == 3.0e12
    assert out["MSFT"] == 1.234e12
    assert out["BAD"] is None
    # Cache file got updated
    assert "MSFT" in load_cache(path)
    assert "BAD" in load_cache(path)


def test_market_caps_array_alignment(tmp_path, monkeypatch):
    path = _tmp_cache(tmp_path)
    save_cache({"AAPL": 3.0e12, "MSFT": 3.1e12, "TSLA": None}, cache_path=path)
    # Ensure no real network call: any retry on a None entry must stay None.
    monkeypatch.setattr("sit.data.market_caps._fetch_one_yfinance", lambda _t: None)
    arr = market_caps_array(["TSLA", "AAPL", "MSFT"], cache_path=path, fallback=np.nan)
    assert arr.shape == (3,)
    assert np.isnan(arr[0])
    assert arr[1] == 3.0e12
    assert arr[2] == 3.1e12


def test_market_caps_array_fallback(tmp_path, monkeypatch):
    path = _tmp_cache(tmp_path)
    save_cache({"AAPL": 1.0}, cache_path=path)
    monkeypatch.setattr("sit.data.market_caps._fetch_one_yfinance", lambda _t: None)
    arr = market_caps_array(["AAPL", "MISSING"], cache_path=path, fallback=-1.0)
    np.testing.assert_array_equal(arr, [1.0, -1.0])


def test_price_proxy_caps_default_is_price():
    prices = np.array([100.0, 200.0, 50.0])
    np.testing.assert_array_equal(price_proxy_caps(prices), prices)


def test_price_proxy_caps_with_shares():
    prices = np.array([100.0, 200.0, 50.0])
    shares = np.array([1e6, 5e5, 2e6])
    expected = prices * shares
    np.testing.assert_array_equal(price_proxy_caps(prices, shares), expected)


def test_price_proxy_caps_shape_mismatch():
    with pytest.raises(ValueError):
        price_proxy_caps(np.zeros(3), np.zeros(4))
