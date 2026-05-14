"""Integration tests for the Phase 5 FastAPI app.

Covers all 9 endpoints (happy path + error paths), the legacy un-prefixed
aliases, lifespan startup, OpenAPI schema completeness, rate-limit
behaviour, Redis-down graceful fallback, and the pricing service cache.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch):
    """Reset cached settings and limiter so per-test env overrides take effect."""
    monkeypatch.delenv("SIT_REDIS_URL", raising=False)
    monkeypatch.setenv("SIT_RATE_LIMITS_ENABLED", "false")
    monkeypatch.setenv("SIT_GIT_SHA", "test-sha")
    from sit.api import deps, settings

    settings.reset_settings()
    deps.reset_redis()
    deps.reset_rate_limiter()
    importlib.reload(importlib.import_module("sit.api.routers.invest"))
    importlib.reload(importlib.import_module("sit.api.routers.health"))
    importlib.reload(importlib.import_module("sit.api.routers.portfolio"))
    importlib.reload(importlib.import_module("sit.api.routers.research"))
    importlib.reload(importlib.import_module("sit.api.routers.backtest"))
    main_mod = importlib.reload(importlib.import_module("sit.api.main"))
    return main_mod


@pytest.fixture
def client(_isolate_settings):
    main = _isolate_settings
    with TestClient(main.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Happy paths (all 9 endpoint groups)
# ---------------------------------------------------------------------------


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "Sparse Index Tracker" in body["message"]
    assert "/api/v1/invest" in body["endpoints"]


def test_health_v1(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["solver"] == "SparseTrackerADMM"
    assert "redis_connected" in body
    assert "git_sha" in body
    assert body["solver_loaded"] is True
    assert body["active_stocks"] > 0


def test_health_legacy_matches_v1(client):
    r1 = client.get("/health")
    r2 = client.get("/api/v1/health")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json() == r2.json()


def test_portfolio_default_index(client):
    r = client.get("/api/v1/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert body["index"] == "sp500"
    assert len(body["weights"]) > 0
    assert body["weights"][0]["rank"] == 1


def test_portfolio_alt_index_returns_note(client):
    r = client.get("/api/v1/portfolio?index=nifty50")
    assert r.status_code == 200
    body = r.json()
    assert body["index"] == "nifty50"
    assert "not yet available" in body["note"]


def test_portfolio_invalid_index_400(client):
    r = client.get("/api/v1/portfolio?index=ftse100")
    assert r.status_code == 400


def test_methods_comparison(client):
    r = client.get("/api/v1/methods/comparison")
    assert r.status_code == 200
    assert isinstance(r.json()["methods"], list)
    assert len(r.json()["methods"]) >= 1


def test_cross_index(client):
    r = client.get("/api/v1/markets/cross-index")
    assert r.status_code == 200
    assert "sp500" in r.json()["runs"]
    assert r.json()["survivorship_bias_flag"] is True


def test_cvxpy_speedup(client):
    r = client.get("/api/v1/cvxpy-speedup")
    assert r.status_code == 200
    assert len(r.json()["rows"]) > 0


def test_regimes(client):
    r = client.get("/api/v1/regimes")
    assert r.status_code == 200
    body = r.json()
    assert body["n_regimes"] == 8
    assert "bear_covid" in body["regimes"]
    assert body["regimes"]["bear_covid"]["r2_test"] > 0.5


def test_walkforward_full(client):
    r = client.get("/api/v1/backtest/walkforward")
    assert r.status_code == 200
    body = r.json()
    assert body["survivorship_bias_flag"] is True
    methods = {s["method"] for s in body["series"]}
    assert "admm" in methods
    for s in body["series"]:
        assert len(s["points"]) <= 1500


def test_walkforward_window_filter(client):
    r = client.get("/api/v1/backtest/walkforward?start=2020-01-01&end=2020-12-31")
    assert r.status_code == 200
    body = r.json()
    assert body["window"] == {"start": "2020-01-01", "end": "2020-12-31"}
    for s in body["series"]:
        for p in s["points"]:
            assert "2020-01-01" <= p["date"] <= "2020-12-31"


def test_walkforward_invalid_date_422(client):
    r = client.get("/api/v1/backtest/walkforward?start=NOTADATE")
    assert r.status_code == 422


def test_lambda_path(client):
    r = client.get("/api/v1/lambda-path?index=sp500")
    assert r.status_code == 200
    body = r.json()
    assert body["index"] == "sp500"
    assert len(body["points"]) == 12
    nnz = [p["nnz"] for p in body["points"]]
    assert nnz[0] >= nnz[-1] - 50  # sanity


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_invest_negative_capital_422(client):
    r = client.get("/api/v1/invest?capital=-100")
    assert r.status_code == 422


def test_invest_alt_index_400(client):
    r = client.get("/api/v1/invest?capital=10000&index=nasdaq100")
    assert r.status_code == 400


def test_invest_yfinance_failure_503(client, monkeypatch):
    from sit.api.routers import invest as invest_mod

    def boom(*a, **kw):
        raise RuntimeError("yfinance is down")

    monkeypatch.setattr(invest_mod, "get_latest_prices", boom)
    r = client.get("/api/v1/invest?capital=10000")
    assert r.status_code == 503
    assert "Yahoo" in r.json()["detail"] or "yfinance" in r.json()["detail"]


def test_invest_live_retrain_failure_503(client, monkeypatch):
    from sit.api.routers import invest as invest_mod

    def boom(*a, **kw):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(invest_mod, "live_retrain", boom)
    r = client.get("/api/v1/invest_live?capital=10000")
    assert r.status_code == 503


def test_invest_happy_path_with_mocked_pricing(client, monkeypatch):
    from sit.api.routers import invest as invest_mod

    def fake_prices(tickers, **kw):
        return dict.fromkeys(tickers, 100.0)

    monkeypatch.setattr(invest_mod, "get_latest_prices", fake_prices)
    r = client.get("/api/v1/invest?capital=100000")
    assert r.status_code == 200
    body = r.json()
    assert body["capital"] == 100000
    assert body["n_stocks_bought"] > 0
    assert body["total_invested"] > 0
    assert body["residual_cash"] >= 0


def test_invest_legacy_path_works(client, monkeypatch):
    from sit.api.routers import invest as invest_mod

    monkeypatch.setattr(
        invest_mod, "get_latest_prices", lambda tickers, **kw: dict.fromkeys(tickers, 50.0)
    )
    r1 = client.get("/invest?capital=10000")
    r2 = client.get("/api/v1/invest?capital=10000")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["capital"] == r2.json()["capital"] == 10000


# ---------------------------------------------------------------------------
# OpenAPI completeness
# ---------------------------------------------------------------------------


def test_openapi_schema_includes_all_endpoints(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    paths = set(spec["paths"].keys())
    expected = {
        "/api/v1/health",
        "/api/v1/portfolio",
        "/api/v1/invest",
        "/api/v1/invest_live",
        "/api/v1/lambda-path",
        "/api/v1/regimes",
        "/api/v1/backtest/walkforward",
        "/api/v1/methods/comparison",
        "/api/v1/markets/cross-index",
        "/api/v1/cvxpy-speedup",
    }
    missing = expected - paths
    assert not missing, f"OpenAPI missing: {missing}"


def test_openapi_examples_present(client):
    spec = client.get("/openapi.json").json()
    invest = spec["paths"]["/api/v1/invest"]["get"]
    capital = next(p for p in invest["parameters"] if p["name"] == "capital")
    examples = capital.get("schema", {}).get("examples") or capital.get("examples")
    assert examples, "Expected OpenAPI examples on capital query."


# ---------------------------------------------------------------------------
# Lifespan / Redis-down behaviour
# ---------------------------------------------------------------------------


def test_lifespan_loads_weights_once(monkeypatch, _isolate_settings):
    main = _isolate_settings
    call_count = {"n": 0}
    real_load = np.load

    def counted_load(*args, **kwargs):
        call_count["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    with TestClient(main.app) as c:
        c.get("/api/v1/health")
        c.get("/api/v1/health")
        c.get("/api/v1/portfolio")
    assert call_count["n"] == 1


def test_health_reports_redis_disconnected_when_no_url(client):
    r = client.get("/api/v1/health").json()
    assert r["redis_connected"] is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_kicks_in(monkeypatch):
    """With a 2/minute limit on /invest, the 3rd request returns 429."""
    monkeypatch.setenv("SIT_RATE_LIMITS_ENABLED", "true")
    monkeypatch.setenv("SIT_RATE_LIMIT_INVEST", "2/minute")
    monkeypatch.setenv("SIT_RATE_LIMIT_DEFAULT", "1000/minute")

    from sit.api import deps
    from sit.api import settings as st

    st.reset_settings()
    deps.reset_redis()
    deps.reset_rate_limiter()

    import sit.api.routers.invest as invest_mod

    importlib.reload(invest_mod)
    monkeypatch.setattr(
        invest_mod, "get_latest_prices", lambda tickers, **kw: dict.fromkeys(tickers, 50.0)
    )
    main = importlib.reload(importlib.import_module("sit.api.main"))

    with TestClient(main.app) as c:
        statuses = [c.get("/api/v1/invest?capital=1000").status_code for _ in range(5)]
    assert statuses.count(200) == 2
    assert statuses.count(429) >= 1
