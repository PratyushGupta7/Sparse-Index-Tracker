"""FastAPI application entry-point.

Mounts every functional route under ``/api/v1/...`` and re-mounts the
legacy un-prefixed paths (``/health``, ``/portfolio``, ``/invest``,
``/invest_live``) so the existing Dockerfile and smoke tests keep working
until the Phase-6 deploy lands.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from sit.api.deps import get_rate_limiter, init_redis
from sit.api.routers import backtest, health, invest, portfolio, research
from sit.api.settings import get_settings

logger = logging.getLogger(__name__)


def _load_weights_bundle(data_dir: Path) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "weights": None,
        "active_stocks": {},
        "n_active": 0,
        "n_total": 0,
        "metrics": {},
    }
    weights_path = data_dir / "sparse_weights.npy"
    active_path = data_dir / "active_stocks.csv"
    metrics_path = data_dir / "phase3_metrics.json"
    if weights_path.exists():
        bundle["weights"] = np.load(weights_path)
        bundle["n_total"] = len(bundle["weights"])
    if active_path.exists():
        df = pd.read_csv(active_path)
        bundle["active_stocks"] = dict(zip(df["ticker"], df["weight"]))
        bundle["n_active"] = len(bundle["active_stocks"])
    if metrics_path.exists():
        bundle["metrics"] = json.loads(metrics_path.read_text())
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.weights_bundle = _load_weights_bundle(Path(settings.data_dir))
    app.state.redis = init_redis(settings)

    if settings.applicationinsights_connection_string:
        try:  # pragma: no cover - optional
            azure_monitor = importlib.import_module("azure.monitor.opentelemetry")
            configure_azure_monitor = azure_monitor.configure_azure_monitor

            configure_azure_monitor(
                connection_string=settings.applicationinsights_connection_string,
            )
            logger.info("Azure Monitor / App Insights instrumentation enabled.")
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not init Azure Monitor (%s); continuing.", exc)

    yield


def _build_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Sparse Index Tracker API",
        description=(
            "Production-grade API for the Sparse Index Replication Engine. "
            "Custom ADMM solver tracks the S&P 500 (and 3 other indices) with "
            "~50 stocks. Mathematically derived, walk-forward backtested, "
            "8-regime stress tested."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    limiter = get_rate_limiter()
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded: {exc.detail}"},
            headers={"Retry-After": "60"},
        )

    api_routers = [
        health.router,
        portfolio.router,
        invest.router,
        research.router,
        backtest.router,
    ]
    for r in api_routers:
        app.include_router(r, prefix="/api/v1")

    app.include_router(health.router)
    app.include_router(portfolio.router)
    app.include_router(invest.router)

    return app


app = _build_app()


def reload_app() -> FastAPI:
    """Test helper — rebuild the app against the current settings."""
    global app
    app = _build_app()
    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "sit.api.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
    )
