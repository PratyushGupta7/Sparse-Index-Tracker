"""Pydantic v2 request/response models for the Phase 5 API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------


class HealthResponse(_Base):
    status: str = Field("healthy", examples=["healthy"])
    solver: str = Field("SparseTrackerADMM", examples=["SparseTrackerADMM"])
    active_stocks: int = Field(..., examples=[50])
    total_universe: int = Field(..., examples=[502])
    weight_sum: float = Field(..., examples=[1.0])
    top_5: dict[str, float] = Field(default_factory=dict, examples=[{"AAPL": 0.07}])
    metrics: dict[str, Any] = Field(default_factory=dict)
    redis_connected: bool = Field(..., examples=[False])
    solver_loaded: bool = Field(..., examples=[True])
    git_sha: str = Field(..., examples=["abc1234"])
    app_version: str = Field(..., examples=["1.0.0"])


class RootResponse(_Base):
    message: str
    description: str
    endpoints: dict[str, str]
    active_stocks: int
    total_universe: int


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


class PortfolioWeight(_Base):
    rank: int = Field(..., examples=[1])
    ticker: str = Field(..., examples=["AAPL"])
    weight: float = Field(..., examples=[0.0723])
    pct: str = Field(..., examples=["7.23%"])


class PortfolioResponse(_Base):
    index: str
    label: str
    benchmark: str
    portfolio_name: str | None = None
    active_stocks: int | None = None
    total_universe: int | None = None
    reduction_pct: float | None = None
    weights: list[PortfolioWeight] = Field(default_factory=list)
    note: str | None = None
    supported_indices: list[str] | None = None


# ---------------------------------------------------------------------------
# Invest / Invest-live
# ---------------------------------------------------------------------------


class Allocation(_Base):
    ticker: str = Field(..., examples=["AAPL"])
    shares: int = Field(..., examples=[12])
    price: float = Field(..., examples=[178.42])
    weight: float = Field(..., examples=[0.0723])
    allocated: float = Field(..., examples=[7234.0])
    actual_cost: float = Field(..., examples=[2141.04])


class InvestResponse(_Base):
    capital: float
    total_invested: float
    residual_cash: float
    utilization_pct: str
    n_stocks_bought: int
    price_date: str
    fetch_time_seconds: float
    allocations: list[Allocation]
    warnings: list[str] | None = None


class InvestLiveModelInfo(_Base):
    train_period: str
    r2_train: float
    te_train_pct: float
    active_stocks: int
    universe_size: int
    source_universe_size: int | None = None
    live_universe_capped: bool | None = None
    converged: bool
    iterations: int
    solve_time_ms: float = Field(..., description="ADMM solve wall-clock time.")
    solver_iterations: int = Field(..., description="Number of ADMM iterations executed.")


class InvestLiveResponse(_Base):
    mode: str = Field("live_retrained")
    index: str
    benchmark: str
    capital: float
    total_invested: float
    residual_cash: float
    utilization_pct: str
    n_stocks_bought: int
    price_date: str
    model: InvestLiveModelInfo
    total_time_seconds: float
    allocations: list[Allocation]
    warnings: list[str] | None = None


# ---------------------------------------------------------------------------
# Research / λ-path
# ---------------------------------------------------------------------------


class LambdaPathPoint(_Base):
    lam: float = Field(..., examples=[0.012])
    lam_frac: float = Field(..., examples=[0.05])
    nnz: int = Field(..., examples=[42])
    in_sample_r2: float = Field(..., examples=[0.96])
    oos_te: float = Field(..., examples=[0.041])


class LambdaPathResponse(_Base):
    index: str
    n_train: int
    n_test: int
    universe_size: int
    points: list[LambdaPathPoint]
    cached: bool = False


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------


class RegimeResult(_Base):
    regime: str
    short: str
    type: str
    color: str | None = None
    description: str | None = None
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_stocks_universe: int
    n_active: int
    r2_train: float
    r2_test: float
    te_train: float
    te_test: float
    correlation: float
    converged: bool
    iterations: int


class RegimesResponse(_Base):
    regimes: dict[str, RegimeResult]
    n_regimes: int
    cached: bool = True


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


class EquityPoint(_Base):
    date: str = Field(..., examples=["2018-01-02"])
    value: float = Field(..., examples=[1_007_157.07])


class WalkForwardSeries(_Base):
    method: str
    points: list[EquityPoint]


class WalkForwardResponse(_Base):
    config: dict[str, Any]
    metadata: dict[str, Any]
    rebalance_dates: list[str]
    survivorship_bias_flag: bool
    series: list[WalkForwardSeries]
    benchmark: list[EquityPoint]
    risk_metrics: dict[str, dict[str, float]]
    window: dict[str, str]
    downsampled_to: int


# ---------------------------------------------------------------------------
# Methods comparison + cross-index + cvxpy speedup
# ---------------------------------------------------------------------------


class MethodComparisonResponse(_Base):
    config: dict[str, Any]
    methods: list[dict[str, Any]]


class CrossIndexResponse(_Base):
    config: dict[str, Any]
    survivorship_bias_flag: bool
    elapsed_s: float
    runs: dict[str, dict[str, Any]]


class CVXPYSpeedupResponse(_Base):
    config: dict[str, Any]
    rows: list[dict[str, Any]]
    summary: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorResponse(_Base):
    detail: str
    retry_after: float | None = None
