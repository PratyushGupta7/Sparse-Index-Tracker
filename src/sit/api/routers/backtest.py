"""Backtest endpoints — walk-forward, methods comparison, cross-index, CVXPY speedup."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from sit.api.schemas import (
    CrossIndexResponse,
    CVXPYSpeedupResponse,
    EquityPoint,
    MethodComparisonResponse,
    WalkForwardResponse,
    WalkForwardSeries,
)
from sit.api.services import artefacts

router = APIRouter(tags=["backtest"])


@router.get(
    "/backtest/walkforward",
    response_model=WalkForwardResponse,
    summary="Walk-forward equity curves + risk metrics",
)
def walkforward(
    start: str | None = Query(None, examples=["2020-01-01"]),
    end: str | None = Query(None, examples=["2022-12-31"]),
    max_points: int = Query(1500, ge=50, le=5000),
) -> WalkForwardResponse:
    try:
        window = artefacts.parse_window(start, end)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        raw = artefacts.load_walkforward()
    except FileNotFoundError as exc:
        raise HTTPException(503, "Walk-forward artefact not generated yet.") from exc

    series: list[WalkForwardSeries] = []
    for method, curve in raw.get("equity_curves", {}).items():
        filtered = artefacts.filter_by_window(curve, start, end)
        downsampled = artefacts.downsample_curve(filtered, max_points=max_points)
        series.append(
            WalkForwardSeries(
                method=method,
                points=[EquityPoint(**p) for p in downsampled],
            )
        )
    bench_curve = artefacts.filter_by_window(raw.get("benchmark_curve", {}), start, end)
    bench_points = [
        EquityPoint(**p) for p in artefacts.downsample_curve(bench_curve, max_points=max_points)
    ]
    return WalkForwardResponse(
        config=raw.get("config", {}),
        metadata=raw.get("metadata", {}),
        rebalance_dates=raw.get("rebalance_dates", []),
        survivorship_bias_flag=bool(raw.get("survivorship_bias_flag", False)),
        series=series,
        benchmark=bench_points,
        risk_metrics=raw.get("risk_metrics", {}),
        window=window,
        downsampled_to=max_points,
    )


@router.get(
    "/methods/comparison",
    response_model=MethodComparisonResponse,
    summary="Head-to-head baseline panel",
)
def methods_comparison() -> MethodComparisonResponse:
    try:
        raw = artefacts.load_method_comparison()
    except FileNotFoundError as exc:
        raise HTTPException(503, "Method comparison artefact missing.") from exc
    return MethodComparisonResponse(
        config=raw.get("config", {}),
        methods=raw.get("methods", []),
    )


@router.get(
    "/markets/cross-index",
    response_model=CrossIndexResponse,
    summary="Cross-index summary (S&P 500, Nasdaq-100, Russell 2000, Nifty 50)",
)
def cross_index() -> CrossIndexResponse:
    try:
        raw = artefacts.load_multi_index()
    except FileNotFoundError as exc:
        raise HTTPException(503, "Multi-index artefact missing.") from exc
    return CrossIndexResponse(
        config=raw.get("config", {}),
        survivorship_bias_flag=bool(raw.get("survivorship_bias_flag", False)),
        elapsed_s=float(raw.get("elapsed_s", 0.0)),
        runs=raw.get("runs", {}),
    )


@router.get(
    "/cvxpy-speedup",
    response_model=CVXPYSpeedupResponse,
    summary="CVXPY-vs-ADMM speedup proof",
)
def cvxpy_speedup() -> CVXPYSpeedupResponse:
    try:
        raw = artefacts.load_cvxpy_speedup()
    except FileNotFoundError as exc:
        raise HTTPException(503, "CVXPY speedup artefact missing.") from exc
    return CVXPYSpeedupResponse(
        config=raw.get("config", {}),
        rows=raw.get("rows", []),
        summary=raw.get("summary"),
    )
