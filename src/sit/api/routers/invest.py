"""Invest endpoints — pre-baked and live-retrain."""

from __future__ import annotations

import math
import time

from fastapi import APIRouter, HTTPException, Query, Request

from sit.api.deps import DEFAULT_INDEX, get_rate_limiter, validate_index
from sit.api.schemas import (
    Allocation,
    InvestLiveModelInfo,
    InvestLiveResponse,
    InvestResponse,
)
from sit.api.services.pricing import get_latest_prices
from sit.api.services.retraining import live_retrain
from sit.api.settings import get_settings
from sit.data.universes import INDEX_METADATA

router = APIRouter(tags=["invest"])
_limiter = get_rate_limiter()
_settings = get_settings()


@router.get(
    "/invest",
    response_model=InvestResponse,
    summary="Allocate capital to the pre-baked sparse portfolio",
    responses={
        400: {"description": "Pre-baked weights unavailable for the selected index."},
        503: {"description": "Yahoo Finance unavailable — try again shortly."},
        429: {"description": "Rate limit exceeded."},
    },
)
@_limiter.limit(lambda: get_settings().rate_limit_invest)
def invest(
    request: Request,
    capital: float = Query(
        ...,
        gt=0,
        description="Total capital to invest in USD (must be > 0).",
        examples=[10_000, 100_000],
    ),
    index: str = Query(
        DEFAULT_INDEX,
        description="Index to track. One of sp500, nasdaq100, russell2000, nifty50.",
        examples=["sp500"],
    ),
) -> InvestResponse:
    name = validate_index(index)
    if name != DEFAULT_INDEX:
        meta = INDEX_METADATA[name]
        raise HTTPException(
            status_code=400,
            detail=(
                f"Pre-baked weights for {meta['label']} are not yet available. "
                f"Use /api/v1/invest_live?capital={capital}&index={name} for a live solve."
            ),
        )

    bundle = getattr(request.app.state, "weights_bundle", {}) or {}
    active = bundle.get("active_stocks", {}) or {}
    if not active:
        raise HTTPException(503, "Solver weights not loaded.")

    t_start = time.time()
    tickers = list(active.keys())

    try:
        prices = get_latest_prices(tickers)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch live prices from Yahoo Finance: {exc}",
        ) from exc

    allocations: list[Allocation] = []
    warnings_list: list[str] = []
    total_invested = 0.0

    for ticker, weight in sorted(active.items(), key=lambda x: -x[1]):
        price = prices.get(ticker)
        if price is None or price <= 0:
            warnings_list.append(f"{ticker}: price unavailable")
            continue
        allocated = capital * weight
        shares = math.floor(allocated / price)
        actual_cost = shares * price
        if shares > 0:
            allocations.append(
                Allocation(
                    ticker=ticker,
                    shares=shares,
                    price=round(price, 2),
                    weight=round(weight, 6),
                    allocated=round(allocated, 2),
                    actual_cost=round(actual_cost, 2),
                )
            )
            total_invested += actual_cost

    residual = capital - total_invested
    fetch_time = round(time.time() - t_start, 2)
    from datetime import datetime

    return InvestResponse(
        capital=capital,
        total_invested=round(total_invested, 2),
        residual_cash=round(residual, 2),
        utilization_pct=f"{(total_invested / capital * 100):.1f}%" if capital else "0.0%",
        n_stocks_bought=len(allocations),
        price_date=datetime.utcnow().strftime("%Y-%m-%d"),
        fetch_time_seconds=fetch_time,
        allocations=allocations,
        warnings=warnings_list or None,
    )


@router.get(
    "/invest_live",
    response_model=InvestLiveResponse,
    summary="Allocate capital to a freshly-retrained sparse portfolio",
    responses={
        503: {"description": "Retraining failed (yfinance / network)."},
        429: {"description": "Rate limit exceeded."},
    },
)
@_limiter.limit(lambda: get_settings().rate_limit_invest_live)
def invest_live(
    request: Request,
    capital: float = Query(
        ...,
        gt=0,
        description="Total capital to invest in USD (must be > 0).",
        examples=[10_000, 100_000],
    ),
    index: str = Query(
        DEFAULT_INDEX,
        description="Index to track. One of sp500, nasdaq100, russell2000, nifty50.",
        examples=["sp500"],
    ),
) -> InvestLiveResponse:
    name = validate_index(index)
    t_start = time.time()

    try:
        model_info = live_retrain(n_days=120, lam_frac=0.05, index=name)
    except Exception as exc:
        raise HTTPException(503, f"Retraining failed: {exc}") from exc

    weights_dict: dict[str, float] = model_info["weights"]
    tickers = list(weights_dict.keys())
    if not tickers:
        raise HTTPException(503, "Retrained model produced zero active stocks.")

    try:
        prices = get_latest_prices(tickers)
    except Exception as exc:
        raise HTTPException(503, f"Price fetch failed: {exc}") from exc

    allocations: list[Allocation] = []
    warn_list: list[str] = []
    if model_info.get("live_universe_capped"):
        warn_list.append(
            "Large live universe capped to "
            f"{model_info['n_universe']} of {model_info['source_universe_size']} "
            "constituents to avoid public yfinance rate limits."
        )
    if model_info.get("requested_benchmark") != model_info.get("benchmark"):
        warn_list.append(
            f"Benchmark quote fell back from {model_info['requested_benchmark']} "
            f"to {model_info['benchmark']} for live data availability."
        )
    total_invested = 0.0
    for ticker, weight in sorted(weights_dict.items(), key=lambda x: -x[1]):
        price = prices.get(ticker)
        if price is None or price <= 0:
            warn_list.append(f"{ticker}: price unavailable")
            continue
        allocated = capital * weight
        shares = math.floor(allocated / price)
        cost = shares * price
        if shares > 0:
            allocations.append(
                Allocation(
                    ticker=ticker,
                    shares=shares,
                    price=round(price, 2),
                    weight=round(weight, 6),
                    allocated=round(allocated, 2),
                    actual_cost=round(cost, 2),
                )
            )
            total_invested += cost

    residual = capital - total_invested
    elapsed = round(time.time() - t_start, 2)

    from datetime import datetime

    return InvestLiveResponse(
        mode="live_retrained",
        index=model_info["index"],
        benchmark=model_info["benchmark"],
        capital=capital,
        total_invested=round(total_invested, 2),
        residual_cash=round(residual, 2),
        utilization_pct=f"{(total_invested / capital * 100):.1f}%" if capital else "0.0%",
        n_stocks_bought=len(allocations),
        price_date=datetime.utcnow().strftime("%Y-%m-%d"),
        model=InvestLiveModelInfo(
            train_period=f"{model_info['train_start']} - {model_info['train_end']}",
            r2_train=model_info["r2_train"],
            te_train_pct=model_info["te_train"],
            active_stocks=model_info["n_active"],
            universe_size=model_info["n_universe"],
            source_universe_size=model_info.get("source_universe_size"),
            live_universe_capped=model_info.get("live_universe_capped"),
            converged=model_info["converged"],
            iterations=model_info["iterations"],
            solve_time_ms=model_info["solve_time_ms"],
            solver_iterations=model_info["solver_iterations"],
        ),
        total_time_seconds=elapsed,
        allocations=allocations,
        warnings=warn_list or None,
    )
