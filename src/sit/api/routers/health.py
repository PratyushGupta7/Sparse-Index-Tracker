"""Health & root endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from sit.api.deps import get_redis
from sit.api.schemas import HealthResponse, RootResponse
from sit.api.settings import get_settings

router = APIRouter(tags=["health"])


def _state(request: Request) -> dict[str, Any]:
    return getattr(request.app.state, "weights_bundle", {})


@router.get("/", response_model=RootResponse, summary="API welcome")
def root(request: Request) -> RootResponse:
    bundle = _state(request)
    n_active = int(bundle.get("n_active", 0))
    n_total = int(bundle.get("n_total", 0))
    return RootResponse(
        message="Sparse Index Tracker API",
        description=(
            f"Track the S&P 500 with only {n_active} stocks instead of {n_total}. "
            "Use /api/v1/invest?capital=100000 to get share allocations."
        ),
        endpoints={
            "/docs": "Interactive API docs (Swagger UI)",
            "/api/v1/health": "Solver metadata & health check",
            "/api/v1/invest": "Get share allocations (pre-baked weights)",
            "/api/v1/invest_live": "Get share allocations (live retrained)",
            "/api/v1/portfolio": "View the raw portfolio weights",
            "/api/v1/lambda-path": "λ-path for the interactive slider",
            "/api/v1/regimes": "8-regime stress test results",
            "/api/v1/backtest/walkforward": "Walk-forward equity curves + risk metrics",
            "/api/v1/methods/comparison": "Head-to-head baseline panel",
            "/api/v1/markets/cross-index": "Cross-index summary",
            "/api/v1/cvxpy-speedup": "CVXPY-vs-ADMM speedup proof",
        },
        active_stocks=n_active,
        total_universe=n_total,
    )


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health(request: Request) -> HealthResponse:
    bundle = _state(request)
    settings = get_settings()
    active = bundle.get("active_stocks", {}) or {}
    metrics = bundle.get("metrics", {}) or {}
    weight_sum = float(sum(active.values())) if isinstance(active, dict) else 0.0
    top_5 = (
        dict(sorted(active.items(), key=lambda x: -x[1])[:5]) if isinstance(active, dict) else {}
    )
    return HealthResponse(
        status="healthy",
        solver="SparseTrackerADMM",
        active_stocks=int(bundle.get("n_active", 0)),
        total_universe=int(bundle.get("n_total", 0)),
        weight_sum=weight_sum,
        top_5=top_5,
        metrics={
            "r2_train": metrics.get("r2_train"),
            "r2_test": metrics.get("r2_test"),
            "te_train": metrics.get("te_train"),
            "te_test": metrics.get("te_test"),
            "train_period": (
                f"{metrics.get('train_start', '?')} - {metrics.get('train_end', '?')}"
            ),
            "test_period": (f"{metrics.get('test_start', '?')} - {metrics.get('test_end', '?')}"),
        },
        redis_connected=get_redis() is not None,
        solver_loaded=bool(bundle.get("weights") is not None),
        git_sha=settings.git_sha,
        app_version=settings.app_version,
    )
