"""End-to-end integration tests for the Phase-3 walk-forward orchestrator.

Mirrors the architecture of ``tests/integration/test_comparison.py``: build a
small synthetic universe (no network) → run the orchestrator → assert
invariants on the equity curves, weights, turnover and metric outputs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sit.backtest import (
    LinearCost,
    NoCost,
    SqrtImpactCost,
    WalkForwardConfig,
    WalkForwardResult,
    compute_risk_metrics,
    run,
)
from sit.data.famafrench import make_synthetic_ff3

# ---------------------------------------------------------------------------
# Synthetic price-panel factory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_prices():
    """A 2-year synthetic price panel of 30 stocks + a market-like benchmark."""
    rng = np.random.default_rng(2026_05_11)
    dates = pd.date_range("2020-01-02", periods=520, freq="B")  # ≈2 years
    p = 30
    mu = rng.normal(2e-4, 1e-4, size=p)
    sigma = rng.uniform(0.005, 0.02, size=p)
    returns = rng.normal(mu, sigma, size=(len(dates), p))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + returns, axis=0),
        index=dates,
        columns=[f"S{i:02d}" for i in range(p)],
    )
    # Benchmark = simple equal-weighted index over the universe (the LASSO
    # regression should be able to recover ≈uniform weights).
    bench_returns = returns.mean(axis=1)
    benchmark = pd.Series(
        100.0 * np.cumprod(1.0 + bench_returns),
        index=dates,
        name="BENCH",
    )
    return prices, benchmark


@pytest.fixture(scope="module")
def baseline_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        start_date="2020-04-01",
        end_date="2021-12-31",
        lookback_days=60,
        rebalance="weekly",
        tx_bps=5.0,
        initial_capital=1_000_000.0,
        methods=["admm", "lasso", "omp", "equal_weight_topn", "benchmark"],
        K=10,
        benchmark_ticker="BENCH",
    )


@pytest.fixture(scope="module")
def baseline_result(synthetic_prices, baseline_config) -> WalkForwardResult:
    prices, benchmark = synthetic_prices
    return run(baseline_config, prices, benchmark)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def test_all_methods_produced_curves(baseline_result, baseline_config):
    assert set(baseline_result.equity_curves) == set(baseline_config.methods)


def test_no_nans_in_equity_curves(baseline_result):
    for name, curve in baseline_result.equity_curves.items():
        assert curve.notna().all(), f"{name} curve has NaNs"
        assert (curve > 0).all(), f"{name} curve hit zero"


def test_curves_start_at_initial_capital(baseline_result, baseline_config):
    for curve in baseline_result.equity_curves.values():
        assert curve.iloc[0] == pytest.approx(baseline_config.initial_capital, rel=1e-2)


def test_rebalance_dates_are_subset_of_simulation_index(baseline_result):
    rebal = baseline_result.rebalance_dates
    sim_index = next(iter(baseline_result.equity_curves.values())).index
    assert set(rebal).issubset(set(sim_index))


def test_weights_on_simplex_at_each_rebalance(baseline_result):
    for name, hist in baseline_result.weights_history.items():
        if name == "benchmark":
            continue
        if hist.empty:
            continue
        assert (hist >= -1e-9).all().all(), f"{name} has negative weights"
        sums = hist.sum(axis=1)
        assert (np.abs(sums - 1.0) < 1e-6).all(), f"{name} sums: {sums.unique()}"


def test_turnover_nonnegative(baseline_result):
    for name, t in baseline_result.turnover.items():
        if name == "benchmark":
            assert t.empty
            continue
        assert (t >= -1e-12).all(), f"{name} has negative turnover"


# ---------------------------------------------------------------------------
# Story-level invariants — what we promise the recruiter
# ---------------------------------------------------------------------------


def test_with_tx_costs_underperforms_no_costs(synthetic_prices, baseline_config):
    """ADMM-with-tx-costs ≤ ADMM-without-tx-costs in terminal NAV."""
    prices, benchmark = synthetic_prices
    no_cost_cfg = baseline_config.model_copy(update={"methods": ["admm"]})
    no_cost = run(no_cost_cfg, prices, benchmark, cost_model=NoCost())
    with_cost = run(no_cost_cfg, prices, benchmark, cost_model=LinearCost(bps_per_side=20.0))
    assert with_cost.equity_curves["admm"].iloc[-1] <= no_cost.equity_curves["admm"].iloc[-1] + 1.0


def test_zero_bps_matches_no_cost(synthetic_prices, baseline_config):
    """Linear(0 bps) ≡ NoCost equity curve to floating-point precision."""
    prices, benchmark = synthetic_prices
    cfg = baseline_config.model_copy(update={"methods": ["admm"], "tx_bps": 0.0})
    a = run(cfg, prices, benchmark, cost_model=LinearCost(bps_per_side=0.0))
    b = run(cfg, prices, benchmark, cost_model=NoCost())
    np.testing.assert_allclose(
        a.equity_curves["admm"].to_numpy(),
        b.equity_curves["admm"].to_numpy(),
        rtol=1e-9,
    )


def test_costs_monotone_in_bps(synthetic_prices, baseline_config):
    """5 bps ≤ 20 bps total cost dollars (ascending in bps)."""
    prices, benchmark = synthetic_prices
    cfg = baseline_config.model_copy(update={"methods": ["admm"]})
    a = run(cfg, prices, benchmark, cost_model=LinearCost(bps_per_side=5.0))
    b = run(cfg, prices, benchmark, cost_model=LinearCost(bps_per_side=20.0))
    assert a.transaction_costs["admm"].sum() < b.transaction_costs["admm"].sum()


def test_admm_lasso_track_each_other(baseline_result):
    """ADMM and LASSO solve the same problem ⇒ near-identical equity curves."""
    admm_curve = baseline_result.equity_curves["admm"]
    lasso_curve = baseline_result.equity_curves["lasso"]
    rel_err = np.abs(admm_curve.to_numpy() - lasso_curve.to_numpy()) / np.abs(
        lasso_curve.to_numpy()
    )
    assert rel_err.max() < 0.05, f"ADMM/LASSO diverge by {rel_err.max():.3%}"


def test_benchmark_curve_matches_benchmark_method(baseline_result):
    """The 'benchmark' method's equity should match the explicit benchmark curve."""
    bench_method = baseline_result.equity_curves["benchmark"]
    bench_explicit = baseline_result.benchmark_curve
    rel = np.abs(bench_method.to_numpy() - bench_explicit.to_numpy()) / np.abs(
        bench_explicit.to_numpy()
    )
    assert rel.max() < 1e-9


# ---------------------------------------------------------------------------
# compute_risk_metrics on a real walk-forward result
# ---------------------------------------------------------------------------


def test_risk_metrics_pipe_through_walkforward(baseline_result):
    metrics = compute_risk_metrics(
        baseline_result.equity_curves["admm"],
        baseline_result.benchmark_curve,
        rebalance_dates=baseline_result.rebalance_dates,
        weights_history=baseline_result.weights_history["admm"],
        turnover=baseline_result.turnover["admm"],
    )
    for key in (
        "ann_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "tracking_error_annual",
        "information_ratio",
        "beta",
        "turnover_per_rebal_avg",
        "hhi_avg",
    ):
        assert key in metrics, f"missing {key}"


def test_factor_metrics_via_synthetic_ff3(baseline_result):
    factors = make_synthetic_ff3(baseline_result.equity_curves["admm"].index, seed=2024)
    metrics = compute_risk_metrics(
        baseline_result.equity_curves["admm"],
        baseline_result.benchmark_curve,
        factors=factors,
    )
    for key in ("ff3_alpha_daily", "ff3_beta_mkt", "ff3_beta_smb", "ff3_beta_hml", "ff3_r2"):
        assert key in metrics


# ---------------------------------------------------------------------------
# Cost-model swap test
# ---------------------------------------------------------------------------


def test_sqrt_impact_cost_swap(synthetic_prices, baseline_config):
    prices, benchmark = synthetic_prices
    cfg = baseline_config.model_copy(update={"methods": ["admm"]})
    result = run(
        cfg,
        prices,
        benchmark,
        cost_model=SqrtImpactCost(kappa=1.0, daily_vol=0.012, participation_rate=0.05),
    )
    assert (result.transaction_costs["admm"] >= 0).all()
    assert result.transaction_costs["admm"].sum() > 0


# ---------------------------------------------------------------------------
# Membership overlay (survivorship-bias-aware)
# ---------------------------------------------------------------------------


def test_membership_overlay_zero_masks_ineligible_tickers(synthetic_prices, baseline_config):
    prices, benchmark = synthetic_prices
    # Allow only the first 10 tickers
    eligible = set(prices.columns[:10])
    overlay = {d.date(): eligible for d in prices.index}
    cfg = baseline_config.model_copy(update={"methods": ["admm"], "K": 6})
    result = run(cfg, prices, benchmark, membership_overlay=overlay)
    assert not result.survivorship_bias_flag
    held = result.weights_history["admm"]
    if not held.empty:
        # Non-eligible columns must always be 0
        for col in prices.columns[10:]:
            assert (held[col].abs() < 1e-9).all(), f"{col} should be masked"


# ---------------------------------------------------------------------------
# Monthly schedule
# ---------------------------------------------------------------------------


def test_monthly_rebalance_yields_fewer_dates(synthetic_prices, baseline_config):
    prices, benchmark = synthetic_prices
    weekly_cfg = baseline_config.model_copy(update={"methods": ["admm"]})
    monthly_cfg = baseline_config.model_copy(update={"methods": ["admm"], "rebalance": "monthly"})
    weekly = run(weekly_cfg, prices, benchmark)
    monthly = run(monthly_cfg, prices, benchmark)
    assert len(monthly.rebalance_dates) < len(weekly.rebalance_dates)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_unknown_method():
    with pytest.raises(ValueError):
        WalkForwardConfig(methods=["nope"])


def test_config_rejects_bad_dates():
    with pytest.raises(ValueError):
        WalkForwardConfig(start_date="2025-01-01", end_date="2024-01-01")


def test_config_rejects_bad_rebalance():
    with pytest.raises(ValueError):
        WalkForwardConfig(rebalance="hourly")


def test_config_rejects_negative_bps():
    with pytest.raises(ValueError):
        WalkForwardConfig(tx_bps=-1.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_result_round_trip_via_json(baseline_result, tmp_path):
    """Equity curves / metadata should round-trip through JSON cleanly."""
    payload = {
        "metadata": baseline_result.metadata,
        "rebalance_dates": [str(d) for d in baseline_result.rebalance_dates],
        "equity_curves": {
            k: {str(d): float(v) for d, v in s.items()}
            for k, s in baseline_result.equity_curves.items()
        },
    }
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(payload))
    decoded = json.loads(path.read_text())
    assert set(decoded["equity_curves"]) == set(baseline_result.equity_curves)
    assert decoded["metadata"]["rebalance"] == "weekly"


def test_save_walkforward_result_writes_full_payload(baseline_result, tmp_path):
    """``save_walkforward_result`` ⇒ self-describing JSON containing every output."""
    from sit.backtest import save_walkforward_result

    out = tmp_path / "wf_full.json"
    save_walkforward_result(
        baseline_result,
        out,
        risk_metrics={"admm": {"sharpe": 1.234, "max_drawdown": -0.05}},
        extra={"smoke": True},
    )
    decoded = json.loads(out.read_text())
    assert "config" in decoded
    assert "equity_curves" in decoded
    assert "turnover" in decoded
    assert "transaction_costs" in decoded
    assert "fit_times" in decoded
    assert "benchmark_curve" in decoded
    assert decoded["risk_metrics"]["admm"]["sharpe"] == pytest.approx(1.234)
    assert decoded["extra"]["smoke"] is True
    # Curves are dict-of-dicts keyed by ISO date string
    bench = decoded["benchmark_curve"]
    assert all(len(k) == 10 for k in bench)
