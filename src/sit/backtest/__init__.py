"""Backtesting modules.

- phase3_validator : single train/test split + plots (legacy Phase 3).
- walkforward      : rolling weekly rebalance + transaction costs (Phase 3).
- risk_metrics     : Sharpe/Sortino/Calmar/DD/IR/factor regression (Phase 3).
- transaction_costs: linear / sqrt-impact / no-cost models (Phase 3).
"""

from __future__ import annotations

from sit.backtest.risk_metrics import (
    CapmFit,
    FamaFrench3Fit,
    annualised_return,
    annualised_vol,
    calmar_ratio,
    capm_regression,
    compute_risk_metrics,
    drawdown,
    famafrench3_regression,
    information_ratio,
    max_drawdown,
    rolling_factor_betas,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
    ulcer_index,
)
from sit.backtest.transaction_costs import (
    LinearCost,
    NoCost,
    SqrtImpactCost,
    TransactionCostModel,
)
from sit.backtest.walkforward import (
    WalkForwardConfig,
    WalkForwardResult,
    load_price_panel_yfinance,
    run,
    save_walkforward_result,
)

__all__ = [
    "CapmFit",
    "FamaFrench3Fit",
    "LinearCost",
    "NoCost",
    "SqrtImpactCost",
    "TransactionCostModel",
    "WalkForwardConfig",
    "WalkForwardResult",
    "annualised_return",
    "annualised_vol",
    "calmar_ratio",
    "capm_regression",
    "compute_risk_metrics",
    "drawdown",
    "famafrench3_regression",
    "information_ratio",
    "load_price_panel_yfinance",
    "max_drawdown",
    "rolling_factor_betas",
    "run",
    "save_walkforward_result",
    "sharpe_ratio",
    "sortino_ratio",
    "tracking_error",
    "ulcer_index",
]
