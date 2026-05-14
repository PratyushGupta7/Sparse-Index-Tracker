"""Unit tests for sit.backtest.transaction_costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sit.backtest.transaction_costs import (
    LinearCost,
    NoCost,
    SqrtImpactCost,
    TransactionCostModel,
)

# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_all_models_satisfy_protocol():
    for model in (LinearCost(), NoCost(), SqrtImpactCost()):
        assert isinstance(model, TransactionCostModel)


# ---------------------------------------------------------------------------
# NoCost
# ---------------------------------------------------------------------------


def test_no_cost_is_always_zero():
    cost = NoCost()
    assert cost.apply(pd.Series([1e6, -2e5, 3e3])) == 0.0
    assert cost.apply(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# LinearCost
# ---------------------------------------------------------------------------


def test_linear_cost_zero_trades_zero_cost():
    cost = LinearCost(bps_per_side=5.0)
    assert cost.apply(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_linear_cost_round_trip_is_two_sides():
    """Buying $1m of A and selling $1m of B at 5 bps/side ⇒ $1000 cost."""
    cost = LinearCost(bps_per_side=5.0)
    trades = pd.Series({"A": 1_000_000.0, "B": -1_000_000.0})
    # 2_000_000 * 5e-4 = 1000
    assert cost.apply(trades) == pytest.approx(1_000.0)


def test_linear_cost_signs_dont_matter():
    """Cost depends on |notional|, not direction."""
    cost = LinearCost(bps_per_side=10.0)
    a = pd.Series([100.0, 200.0, 300.0])
    b = pd.Series([-100.0, -200.0, -300.0])
    assert cost.apply(a) == cost.apply(b)


def test_linear_cost_monotone_in_turnover():
    """Doubling every trade exactly doubles the cost."""
    cost = LinearCost(bps_per_side=7.5)
    trades = pd.Series([1_000.0, -2_000.0, 3_000.0])
    assert cost.apply(trades * 2.0) == pytest.approx(2.0 * cost.apply(trades))
    assert cost.apply(trades * 5.0) == pytest.approx(5.0 * cost.apply(trades))


def test_linear_cost_additive_across_partitions():
    """Cost over a concatenated trade vector equals the sum of partial costs."""
    cost = LinearCost(bps_per_side=2.5)
    a = pd.Series({"AAPL": 1_000.0, "MSFT": -500.0})
    b = pd.Series({"GOOG": 250.0, "META": -2_000.0})
    full = pd.concat([a, b])
    assert cost.apply(full) == pytest.approx(cost.apply(a) + cost.apply(b))


def test_linear_cost_zero_bps_returns_zero():
    cost = LinearCost(bps_per_side=0.0)
    assert cost.apply(pd.Series([1e6, -2e6])) == 0.0


def test_linear_cost_rejects_negative_bps():
    with pytest.raises(ValueError):
        LinearCost(bps_per_side=-1.0)


def test_linear_cost_empty_series_zero():
    cost = LinearCost(bps_per_side=5.0)
    assert cost.apply(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# SqrtImpactCost
# ---------------------------------------------------------------------------


def test_sqrt_cost_is_strictly_positive_for_nonzero_trades():
    cost = SqrtImpactCost(kappa=1.0, daily_vol=0.01, participation_rate=0.05)
    assert cost.apply(pd.Series([1e6])) > 0
    assert cost.apply(pd.Series([-1e6])) > 0


def test_sqrt_cost_zero_kappa_zero_cost():
    cost = SqrtImpactCost(kappa=0.0, daily_vol=0.01, participation_rate=0.05)
    assert cost.apply(pd.Series([1e6])) == 0.0


def test_sqrt_cost_validation():
    with pytest.raises(ValueError):
        SqrtImpactCost(kappa=-1.0)
    with pytest.raises(ValueError):
        SqrtImpactCost(daily_vol=-1.0)
    with pytest.raises(ValueError):
        SqrtImpactCost(participation_rate=1.5)
    with pytest.raises(ValueError):
        SqrtImpactCost(participation_rate=-0.1)


def test_sqrt_cost_scales_linearly_with_notional_when_participation_fixed():
    """For fixed kappa+vol+participation the model collapses to linear in |notional|."""
    cost = SqrtImpactCost(kappa=1.0, daily_vol=0.02, participation_rate=0.04)
    trades = pd.Series([1_000.0, -2_000.0])
    assert cost.apply(trades * 2.0) == pytest.approx(2.0 * cost.apply(trades))


def test_sqrt_cost_grows_with_kappa():
    base = SqrtImpactCost(kappa=1.0, daily_vol=0.01, participation_rate=0.05)
    bigger = SqrtImpactCost(kappa=2.0, daily_vol=0.01, participation_rate=0.05)
    trades = pd.Series([1_000.0, -2_000.0])
    assert bigger.apply(trades) == pytest.approx(2.0 * base.apply(trades))


# ---------------------------------------------------------------------------
# Composability — cost models swap into the same call site
# ---------------------------------------------------------------------------


def test_models_can_be_swapped_at_call_site():
    """Demonstrate that any TransactionCostModel works through a uniform API."""
    trades = pd.Series([5_000.0, -3_000.0, 2_500.0])

    def fee_total(model: TransactionCostModel, t: pd.Series) -> float:
        return model.apply(t)

    fees = [fee_total(m, trades) for m in (NoCost(), LinearCost(5), SqrtImpactCost())]
    assert fees[0] == 0.0
    assert fees[1] > 0
    assert fees[2] > 0
    assert np.isfinite(fees).all()
