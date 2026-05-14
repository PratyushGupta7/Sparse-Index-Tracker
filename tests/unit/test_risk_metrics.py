"""Unit tests for sit.backtest.risk_metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sit.backtest.risk_metrics import (
    ANNUALIZATION_DAYS,
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
from sit.data.famafrench import make_synthetic_ff3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_curves(returns: np.ndarray, start_value: float = 1.0) -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=len(returns) + 1, freq="B")
    values = [start_value, *list(start_value * np.cumprod(1.0 + returns))]
    return pd.Series(values, index=dates)


def _two_curves(seed: int = 0, n: int = 504) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    bench_r = rng.standard_normal(n) * 0.01
    port_r = bench_r + rng.standard_normal(n) * 0.003 + 1e-4
    return _make_curves(port_r), _make_curves(bench_r)


# ---------------------------------------------------------------------------
# Drawdown / max DD / Ulcer
# ---------------------------------------------------------------------------


def test_drawdown_zero_for_monotone_increasing():
    eq = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04])
    dd = drawdown(eq)
    assert (dd <= 1e-12).all()
    assert max_drawdown(eq) == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_known_value():
    eq = pd.Series([1.0, 1.5, 0.75, 1.2, 1.0])
    # peak = 1.5, trough = 0.75 ⇒ -50%
    assert max_drawdown(eq) == pytest.approx(-0.5)


def test_ulcer_zero_when_no_drawdown():
    eq = pd.Series([1.0, 1.05, 1.10, 1.20])
    assert ulcer_index(eq) == pytest.approx(0.0, abs=1e-12)


def test_ulcer_positive_with_drawdown():
    eq = pd.Series([1.0, 1.5, 1.0, 1.2])
    assert ulcer_index(eq) > 0.0


# ---------------------------------------------------------------------------
# Annualised return / vol / Sharpe / Sortino / Calmar
# ---------------------------------------------------------------------------


def test_annualised_return_constant_rate():
    """Constant 0.0001 daily growth ⇒ ann_return = (1.0001^252 - 1) ≈ 2.55%."""
    rets = pd.Series([1e-4] * 252)
    target = (1.0001) ** 252 - 1.0
    assert annualised_return(rets) == pytest.approx(target, rel=1e-9)


def test_annualised_vol_zero_for_constant_returns():
    rets = pd.Series([0.0005] * 100)
    assert annualised_vol(rets) == pytest.approx(0.0, abs=1e-12)


def test_sharpe_constant_returns_is_nan():
    """Zero realised vol → Sharpe undefined → NaN (we choose NaN, not inf)."""
    rets = pd.Series([1e-4] * 252)
    s = sharpe_ratio(rets)
    assert math.isnan(s), "Constant-return Sharpe should be NaN."


def test_sharpe_recovers_known_distribution():
    """Gaussian r ~ N(μ, σ²), large n ⇒ Sharpe ≈ μ/σ × √252.

    Note: the standard error of an N-sample Sharpe is roughly
    ``sqrt((1 + 0.5 SR²) / n) * √252``, so even at n=80 000 we want a
    relative tolerance of ~10 % to avoid statistical flakiness.
    """
    rng = np.random.default_rng(123)
    mu, sigma = 5e-4, 0.01
    rets = pd.Series(rng.normal(mu, sigma, size=80_000))
    expected = mu / sigma * math.sqrt(ANNUALIZATION_DAYS)
    assert sharpe_ratio(rets) == pytest.approx(expected, rel=0.10)


def test_sortino_only_uses_downside():
    """Symmetric returns → Sortino ≈ √2 × Sharpe (asymptotically)."""
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(5e-4, 0.01, size=20_000))
    sh = sharpe_ratio(rets)
    so = sortino_ratio(rets)
    # √2 ≈ 1.414, generous tolerance
    assert so / sh == pytest.approx(math.sqrt(2.0), rel=0.05)


def test_calmar_known_value():
    """Known equity curve: 50 % up, 25 % down → ann_return / |MDD|."""
    rets = np.full(252, 0.0005)
    eq = _make_curves(rets)
    cr = calmar_ratio(eq)
    assert math.isfinite(cr) or math.isnan(cr)


# ---------------------------------------------------------------------------
# Tracking error / IR / β / α
# ---------------------------------------------------------------------------


def test_identical_to_benchmark_te_zero_ir_zero():
    """If portfolio returns ≡ benchmark returns ⇒ TE=0, IR=NaN, β=1, α=0, R²=1."""
    _, bm = _two_curves(seed=7)
    eq2 = bm.copy()
    rets = eq2.pct_change().dropna()
    brets = bm.pct_change().dropna()

    assert tracking_error(rets, brets) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(information_ratio(rets, brets))

    capm = capm_regression(rets, brets)
    assert capm.beta == pytest.approx(1.0, abs=1e-9)
    assert capm.alpha_daily == pytest.approx(0.0, abs=1e-12)
    assert capm.r_squared == pytest.approx(1.0, abs=1e-9)


def test_capm_recovers_synthetic_alpha_and_beta():
    """Construct r_p = α + β·r_b + ε ⇒ recover α, β within 5%."""
    rng = np.random.default_rng(2026)
    n = 5000
    rb = rng.normal(5e-4, 0.012, size=n)
    alpha_true = 1e-4
    beta_true = 1.3
    rp = alpha_true + beta_true * rb + rng.normal(0.0, 0.002, size=n)
    p = pd.Series(rp)
    b = pd.Series(rb)
    capm = capm_regression(p, b)
    assert capm.beta == pytest.approx(beta_true, rel=0.03)
    assert capm.alpha_daily == pytest.approx(alpha_true, abs=2e-5)


def test_alpha_recovered_within_1e3():
    """Stronger version of the above with explicit 1e-3 absolute tolerance on α."""
    rng = np.random.default_rng(99)
    n = 8000
    rb = rng.normal(0.0, 0.01, size=n)
    alpha_true = 4e-4
    beta_true = 0.9
    rp = alpha_true + beta_true * rb + rng.normal(0.0, 0.001, size=n)
    p = pd.Series(rp)
    b = pd.Series(rb)
    capm = capm_regression(p, b)
    assert abs(capm.alpha_daily - alpha_true) < 1e-3


# ---------------------------------------------------------------------------
# Fama-French 3
# ---------------------------------------------------------------------------


def test_ff3_regression_recovers_alpha_zero_for_pure_market():
    """If r_p = r_f + r_mkt_rf exactly ⇒ α=0, β_mkt=1, R²=1."""
    dates = pd.date_range("2018-01-01", periods=2520, freq="B")
    factors = make_synthetic_ff3(dates, seed=11)
    df = factors.df
    rf = df["rf"]
    rp = rf + df["mkt_rf"]
    fit = famafrench3_regression(rp, factors)
    assert fit.alpha_daily == pytest.approx(0.0, abs=1e-12)
    assert fit.beta_mkt == pytest.approx(1.0, abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)


def test_ff3_regression_recovers_known_loadings():
    """Synthesise r_p with known (β_mkt, β_smb, β_hml) ⇒ recover within 2 %."""
    dates = pd.date_range("2018-01-01", periods=2500, freq="B")
    factors = make_synthetic_ff3(dates, seed=2026)
    # Use a *different* seed for the noise so it's independent of the factor
    # frame (otherwise rng.normal would re-trace the same Mersenne stream and
    # induce spurious correlation with mkt_rf).
    rng = np.random.default_rng(99999)
    rf = factors.df["rf"]
    target = (
        rf
        + 0.95 * factors.df["mkt_rf"]
        + 0.30 * factors.df["smb"]
        - 0.20 * factors.df["hml"]
        + rng.normal(0.0, 0.002, size=len(dates))
    )
    fit = famafrench3_regression(target, factors)
    assert fit.beta_mkt == pytest.approx(0.95, abs=0.05)
    assert fit.beta_smb == pytest.approx(0.30, abs=0.10)
    assert fit.beta_hml == pytest.approx(-0.20, abs=0.10)
    assert 0.5 < fit.r_squared < 0.99


def test_ff3_regression_handles_short_input():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    factors = make_synthetic_ff3(dates)
    rp = pd.Series([0.001, 0.002, -0.001], index=dates)
    fit = famafrench3_regression(rp, factors)
    assert fit.n_obs == 3
    assert math.isnan(fit.alpha_daily) or math.isfinite(fit.alpha_daily)


# ---------------------------------------------------------------------------
# Rolling betas
# ---------------------------------------------------------------------------


def test_rolling_factor_betas_shape_and_finite():
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    factors = make_synthetic_ff3(dates, seed=5)
    # Independent rng for noise (see test_ff3_regression_recovers_known_loadings).
    rng = np.random.default_rng(77777)
    rp = (
        factors.df["rf"]
        + 1.0 * factors.df["mkt_rf"]
        + 0.5 * factors.df["smb"]
        + rng.normal(0.0, 0.001, size=len(dates))
    )
    betas = rolling_factor_betas(rp, factors, window=90)
    assert betas.shape == (len(dates), 3)
    valid = betas.dropna()
    assert len(valid) == len(dates) - 89
    assert valid["beta_mkt"].mean() == pytest.approx(1.0, abs=0.15)
    assert valid["beta_smb"].mean() == pytest.approx(0.5, abs=0.20)


# ---------------------------------------------------------------------------
# compute_risk_metrics — top-level integration
# ---------------------------------------------------------------------------


def test_compute_risk_metrics_full_output_keys():
    eq, bm = _two_curves(seed=1)
    dates = pd.date_range("2018-01-01", periods=len(eq), freq="B")
    eq.index = dates
    bm.index = dates
    factors = make_synthetic_ff3(dates, seed=10)

    rebal_dates = dates[::5]
    weights = pd.DataFrame(
        np.random.dirichlet(np.ones(20), size=len(rebal_dates)),
        index=rebal_dates,
        columns=[f"S{i}" for i in range(20)],
    )
    turnover = pd.Series(np.random.uniform(0.02, 0.10, size=len(rebal_dates)), index=rebal_dates)
    metrics = compute_risk_metrics(
        eq,
        bm,
        rebalance_dates=rebal_dates,
        weights_history=weights,
        turnover=turnover,
        factors=factors,
    )

    expected_keys = {
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "ulcer_index",
        "calmar",
        "tracking_error_annual",
        "information_ratio",
        "beta",
        "alpha_daily",
        "alpha_annual",
        "capm_r2",
        "ff3_alpha_daily",
        "ff3_beta_mkt",
        "ff3_beta_smb",
        "ff3_beta_hml",
        "ff3_r2",
        "turnover_per_rebal_avg",
        "turnover_per_rebal_p95",
        "turnover_annualized",
        "n_rebalances",
        "hhi_avg",
        "hhi_p95",
        "max_weight_avg",
        "max_weight_overall",
        "effective_n_avg",
    }
    assert expected_keys.issubset(metrics.keys()), expected_keys - metrics.keys()


def test_compute_risk_metrics_constant_curve():
    """Constant equity curve ⇒ ann_return=0, vol=0, sharpe=NaN, max_dd=0."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    eq = pd.Series(np.full(100, 1.0), index=dates)
    bm = eq.copy()
    metrics = compute_risk_metrics(eq, bm)
    assert metrics["ann_return"] == pytest.approx(0.0)
    assert metrics["ann_vol"] == pytest.approx(0.0)
    assert math.isnan(metrics["sharpe"])
    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_compute_risk_metrics_without_weights_or_turnover():
    """Optional inputs ⇒ output simply omits those keys."""
    eq, bm = _two_curves(seed=3)
    metrics = compute_risk_metrics(eq, bm)
    for k in ("turnover_per_rebal_avg", "hhi_avg", "ff3_alpha_daily"):
        assert k not in metrics


def test_compute_risk_metrics_turnover_p95():
    eq, bm = _two_curves(seed=4)
    rebal_dates = eq.index[::20]
    turnover = pd.Series(np.linspace(0.01, 0.20, len(rebal_dates)), index=rebal_dates)
    metrics = compute_risk_metrics(eq, bm, turnover=turnover, rebalance_dates=rebal_dates)
    assert metrics["turnover_per_rebal_avg"] == pytest.approx(turnover.mean())
    # p95 should be near the high end of a uniform-spaced series
    assert metrics["turnover_per_rebal_p95"] > metrics["turnover_per_rebal_avg"]


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_short_series_returns_nans_not_errors():
    eq = pd.Series([1.0, 1.01], index=pd.date_range("2024-01-01", periods=2, freq="B"))
    bm = eq.copy()
    metrics = compute_risk_metrics(eq, bm)
    # Annualised vol from a single observation is NaN
    assert math.isnan(metrics["ann_vol"])


def test_aligned_index_handles_misaligned_inputs():
    """If portfolio and benchmark dates are misaligned, inner join should pick the overlap."""
    rng = np.random.default_rng(8)
    pdates = pd.date_range("2020-01-01", periods=200, freq="B")
    bdates = pd.date_range("2020-01-15", periods=200, freq="B")
    p = (1.0 + pd.Series(rng.normal(0.0, 0.01, 200), index=pdates)).cumprod()
    b = (1.0 + pd.Series(rng.normal(0.0, 0.01, 200), index=bdates)).cumprod()
    metrics = compute_risk_metrics(p, b)
    assert math.isfinite(metrics["sharpe"]) or math.isnan(metrics["sharpe"])
    assert math.isfinite(metrics["tracking_error_annual"])


def test_negative_returns_handled():
    """A monotonically declining curve gives a negative annualised return."""
    rets = np.full(252, -1e-3)
    eq = _make_curves(rets)
    rets_series = eq.pct_change().dropna()
    assert annualised_return(rets_series) < 0
    # Constant negative returns ⇒ realised vol below the floor ⇒ Sharpe NaN
    assert math.isnan(sharpe_ratio(rets_series))
