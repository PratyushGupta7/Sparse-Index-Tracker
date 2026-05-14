"""Unit tests for the Phase 5 service layer."""

from __future__ import annotations

import json

import pytest

from sit.api.services import artefacts, lambda_path, pricing


def test_pricing_lru_hits_yfinance_once(monkeypatch):
    """5 identical calls → 1 yfinance hit."""
    pricing.clear_pricing_caches()
    pricing._cached_today_prices.cache_clear()

    calls = {"n": 0}

    def fake_yf_download(tickers, **kwargs):
        calls["n"] += 1
        import pandas as pd

        idx = pd.MultiIndex.from_product([["Close", "Open"], list(tickers)])
        return pd.DataFrame([[101.0] * len(idx)], columns=idx)

    import yfinance as yf

    monkeypatch.setattr(yf, "download", fake_yf_download)

    for _ in range(5):
        out = pricing.get_latest_prices(["AAPL", "MSFT"], today="2026-05-12")
    assert calls["n"] == 1
    assert set(out.keys()) == {"AAPL", "MSFT"}


def test_pricing_redis_l2_writethrough(monkeypatch):
    """If Redis is set, prices get cached there."""
    pricing._cached_today_prices.cache_clear()

    store: dict[str, str] = {}

    class FakeRedis:
        def get(self, k):
            return store.get(k)

        def setex(self, k, ttl, v):
            store[k] = v

        def scan_iter(self, pattern):
            return iter(list(store.keys()))

        def delete(self, k):
            store.pop(k, None)

    from sit.api import deps

    monkeypatch.setattr(deps, "_redis_client", FakeRedis())

    import yfinance as yf

    def fake_dl(tickers, **kw):
        import pandas as pd

        idx = pd.MultiIndex.from_product([["Close"], list(tickers)])
        return pd.DataFrame([[42.0] * len(idx)], columns=idx)

    monkeypatch.setattr(yf, "download", fake_dl)

    out1 = pricing.get_latest_prices(["AAPL"], today="2026-05-12")
    assert out1 == {"AAPL": 42.0}
    assert any("sit:price:" in k for k in store)

    pricing._cached_today_prices.cache_clear()
    out2 = pricing.get_latest_prices(["AAPL"], today="2026-05-12")
    assert out2 == {"AAPL": 42.0}


def test_lambda_path_cache_write_then_read(tmp_path, monkeypatch):
    from sit.api import settings as st

    st.reset_settings()
    monkeypatch.setenv("SIT_DATA_DIR", str(tmp_path))
    st.reset_settings()

    out = lambda_path.compute_path("sp500", prefer_real=False)
    assert len(out["points"]) == 12
    cache_path = tmp_path / "lambda_paths" / "sp500.json"
    assert cache_path.exists()

    loaded = json.loads(cache_path.read_text())
    assert loaded["index"] == "sp500"

    out2 = lambda_path.compute_path("sp500", prefer_real=False)
    assert out2["cached"] is True
    assert len(out2["points"]) == 12


def test_artefacts_downsample_curve():
    curve = {f"2020-{m:02d}-01": float(m) for m in range(1, 13)}
    out = artefacts.downsample_curve(curve, max_points=100)
    assert len(out) == 12
    assert out[0]["date"] == "2020-01-01"

    big = {f"2020-{i:04d}": float(i) for i in range(1000)}
    out2 = artefacts.downsample_curve(big, max_points=100)
    assert len(out2) <= 101


def test_artefacts_filter_by_window():
    curve = {"2020-01-01": 1.0, "2020-06-01": 2.0, "2021-01-01": 3.0}
    assert artefacts.filter_by_window(curve, "2020-05-01", "2020-12-31") == {"2020-06-01": 2.0}
    assert artefacts.filter_by_window(curve, None, None) == curve


def test_artefacts_parse_window_validates():
    assert artefacts.parse_window("2020-01-01", "2020-12-31") == {
        "start": "2020-01-01",
        "end": "2020-12-31",
    }
    with pytest.raises(ValueError):
        artefacts.parse_window("not-a-date", None)


def test_validate_index_helper():
    from fastapi import HTTPException

    from sit.api.deps import validate_index

    assert validate_index("sp500") == "sp500"
    assert validate_index("SP-500") == "sp500"
    assert validate_index(None) == "sp500"
    with pytest.raises(HTTPException):
        validate_index("ftse100")


def test_live_retrain_caps_large_universe(monkeypatch):
    from sit.api import settings as st
    from sit.api.services import retraining

    st.reset_settings()
    monkeypatch.setenv("SIT_LIVE_UNIVERSE_MAX_TICKERS", "3")
    monkeypatch.setenv("SIT_LIVE_UNIVERSE_CAP_THRESHOLD", "5")
    st.reset_settings()
    monkeypatch.setattr(
        retraining,
        "get_universe",
        lambda index: ([f"T{i}" for i in range(10)], "IWM"),
    )

    tickers, benchmark, source_size, capped = retraining._fetch_tickers_live("russell2000")
    assert benchmark == "IWM"
    assert source_size == 10
    assert capped is True
    assert tickers == ["T0", "T1", "T2"]
    st.reset_settings()
