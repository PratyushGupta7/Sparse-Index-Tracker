"""sit/backtest/risk_metrics.py — annualised risk + factor analytics.

The headline function :func:`compute_risk_metrics` consumes an equity curve
and a benchmark equity curve (both ``pd.Series`` indexed by date, both
starting at the same value), plus optional rebalance dates / weights / turnover,
and returns a flat ``dict[str, float]`` containing every Phase-3 metric the
recruiter expects on the table.

Conventions
-----------
* All return-derived statistics are computed on **simple daily returns**
  ``r_t = P_t / P_{t-1} - 1``.
* All risk numbers are annualised assuming 252 trading days, ``ddof=1``.
* "Excess returns" mean ``r_p - r_b`` (vs benchmark) for IR / TE / β / α; for
  the FF3 regression we use ``r_p - r_f`` and ``r_b - r_f``.
* When a metric is undefined (e.g. Sharpe with zero realised vol) we return
  ``float('nan')`` rather than ``inf`` — easier to filter downstream.

Reference
---------
- Sharpe, W. (1966). *Mutual fund performance.* J. of Business.
- Sortino, F. & van der Meer, R. (1991). *Downside risk.* J. of Portfolio Mgmt.
- Calmar = annual return / max DD (Young 1991).
- Ulcer index = √mean(drawdown²) over the path.
- Fama, E. F. & French, K. R. (1993). *Common risk factors in the returns on
  stocks and bonds.* JFE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from sit.data.famafrench import FamaFrenchFactors


ANNUALIZATION_DAYS = 252


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float) -> float:
    """Return ``num/den`` or NaN when ``den`` is unsafe (zero / non-finite)."""
    if not math.isfinite(den) or abs(den) < 1e-30:
        return float("nan")
    return float(num) / float(den)


def _equity_to_returns(equity: pd.Series) -> pd.Series:
    """Convert a strictly positive equity curve to simple daily returns."""
    if equity.empty:
        return equity
    rets = equity.pct_change().dropna()
    return rets


def drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown series ``equity / running_max - 1`` (≤ 0 everywhere)."""
    cum_max = equity.cummax()
    return equity / cum_max - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (returns a negative number, e.g. ``-0.32`` for 32 %)."""
    if equity.empty:
        return float("nan")
    return float(drawdown(equity).min())


def ulcer_index(equity: pd.Series) -> float:
    """Ulcer index ``= sqrt(mean(drawdown^2))`` over the full equity path."""
    dd = drawdown(equity)
    if dd.empty:
        return float("nan")
    return float(np.sqrt(np.mean(dd.to_numpy() ** 2)))


# ---------------------------------------------------------------------------
# Annualised return / vol / Sharpe / Sortino
# ---------------------------------------------------------------------------


def annualised_return(returns: pd.Series, *, periods_per_year: int = ANNUALIZATION_DAYS) -> float:
    """CAGR-equivalent annualised return ``(prod(1+r))^{T/n} - 1``.

    Falls back to NaN on empty input.
    """
    if returns.empty:
        return float("nan")
    growth = float(np.prod(1.0 + returns.to_numpy()))
    if growth <= 0.0:
        return float("nan")
    n = int(returns.shape[0])
    return float(growth ** (periods_per_year / n) - 1.0)


def annualised_vol(returns: pd.Series, *, periods_per_year: int = ANNUALIZATION_DAYS) -> float:
    if returns.shape[0] < 2:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


# Threshold below which we declare "effectively zero realised vol" and return
# NaN rather than a meaningless huge ratio. Real daily-return std is ~1e-2;
# anything below 1e-12 is numerical noise from a constant-return path.
_VOL_FLOOR = 1e-12


def sharpe_ratio(
    returns: pd.Series,
    *,
    rf_daily: float = 0.0,
    periods_per_year: int = ANNUALIZATION_DAYS,
) -> float:
    """Annualised Sharpe ratio of *daily-decimal* returns.

    Convention: arithmetic mean ``mean(r - rf)`` divided by realised vol,
    times ``√252``. Returns ``NaN`` if the realised vol is effectively zero.
    """
    if returns.shape[0] < 2:
        return float("nan")
    excess = returns - rf_daily
    vol = float(excess.std(ddof=1))
    if vol < _VOL_FLOOR:
        return float("nan")
    return float(excess.mean() / vol * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    *,
    rf_daily: float = 0.0,
    periods_per_year: int = ANNUALIZATION_DAYS,
) -> float:
    """Sortino ratio ``= ann_excess_return / ann_downside_deviation``.

    Downside deviation is the RMS of the *negative* excess returns
    (``min(0, r)`` clipped). Returns ``NaN`` if there are no down-days or
    fewer than two observations.
    """
    if returns.shape[0] < 2:
        return float("nan")
    excess = (returns - rf_daily).to_numpy(dtype=np.float64)
    downside = np.clip(excess, a_min=None, a_max=0.0)
    downside_dev = math.sqrt(float(np.mean(downside**2)))
    if downside_dev < _VOL_FLOOR:
        return float("nan")
    return float(excess.mean() / downside_dev * math.sqrt(periods_per_year))


def calmar_ratio(
    equity: pd.Series,
    *,
    periods_per_year: int = ANNUALIZATION_DAYS,
) -> float:
    """Calmar = annualised return / |max drawdown|."""
    rets = _equity_to_returns(equity)
    ar = annualised_return(rets, periods_per_year=periods_per_year)
    mdd = max_drawdown(equity)
    if not math.isfinite(ar) or not math.isfinite(mdd) or abs(mdd) < 1e-30:
        return float("nan")
    return float(ar / abs(mdd))


# ---------------------------------------------------------------------------
# Tracking error / IR / β / α
# ---------------------------------------------------------------------------


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int = ANNUALIZATION_DAYS,
) -> float:
    """Annualised tracking error ``std(p - b) * √252``."""
    a, b = portfolio_returns.align(benchmark_returns, join="inner")
    diff = a - b
    if diff.shape[0] < 2:
        return float("nan")
    return float(diff.std(ddof=1) * math.sqrt(periods_per_year))


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int = ANNUALIZATION_DAYS,
) -> float:
    a, b = portfolio_returns.align(benchmark_returns, join="inner")
    diff = a - b
    if diff.shape[0] < 2:
        return float("nan")
    sd = float(diff.std(ddof=1))
    if sd < _VOL_FLOOR:
        return float("nan")
    return float(diff.mean() / sd * math.sqrt(periods_per_year))


@dataclass(frozen=True)
class CapmFit:
    """Fitted CAPM (single-factor) regression of ``r_p - r_f`` on ``r_b - r_f``."""

    alpha_daily: float
    alpha_annual: float
    beta: float
    r_squared: float
    n_obs: int


def capm_regression(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    rf_daily: float = 0.0,
) -> CapmFit:
    """Estimate β and Jensen's α via OLS on excess returns.

    Returns NaNs if the inner-join has < 3 observations.
    """
    p, b = portfolio_returns.align(benchmark_returns, join="inner")
    p = p - rf_daily
    b = b - rf_daily
    n = int(p.shape[0])
    if n < 3:
        return CapmFit(float("nan"), float("nan"), float("nan"), float("nan"), n)

    # Degenerate inputs (zero-variance benchmark or zero-variance portfolio)
    # break the OLS variance / R² calculation. Bail out cleanly in that case.
    if float(p.std(ddof=1)) < _VOL_FLOOR or float(b.std(ddof=1)) < _VOL_FLOOR:
        return CapmFit(float("nan"), float("nan"), float("nan"), float("nan"), n)

    X = sm.add_constant(b.to_numpy(dtype=np.float64), prepend=True)
    y = p.to_numpy(dtype=np.float64)
    res = sm.OLS(y, X).fit()
    alpha_daily, beta = float(res.params[0]), float(res.params[1])
    alpha_annual = (
        (1.0 + alpha_daily) ** ANNUALIZATION_DAYS - 1.0 if abs(alpha_daily) < 1 else float("nan")
    )
    r2 = float(res.rsquared)
    return CapmFit(alpha_daily, alpha_annual, beta, r2, n)


# ---------------------------------------------------------------------------
# Fama-French 3-factor regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamaFrench3Fit:
    """Fitted FF3 regression with HAC-Newey-West standard errors."""

    alpha_daily: float
    alpha_annual: float
    beta_mkt: float
    beta_smb: float
    beta_hml: float
    r_squared: float
    n_obs: int
    t_alpha: float
    t_mkt: float
    t_smb: float
    t_hml: float


def famafrench3_regression(
    portfolio_returns: pd.Series,
    factors: FamaFrenchFactors | pd.DataFrame,
    *,
    hac_lags: int = 5,
) -> FamaFrench3Fit:
    """Run an OLS regression of excess portfolio returns on FF3 factors.

    The FF3 frame is forward-filled to the portfolio's date index over short
    gaps (≤ 3 days) to handle weekend / holiday mismatches. The regression
    uses HAC (Newey-West) covariance with ``hac_lags`` lags so the t-stats
    survive autocorrelated residuals.
    """
    df: pd.DataFrame = factors.df if hasattr(factors, "df") else factors  # type: ignore[assignment]
    aligned = df.reindex(portfolio_returns.index).ffill(limit=3).dropna()
    p = portfolio_returns.reindex(aligned.index).dropna()
    aligned = aligned.reindex(p.index)
    n = int(p.shape[0])
    if n < 5:
        nan = float("nan")
        return FamaFrench3Fit(
            alpha_daily=nan,
            alpha_annual=nan,
            beta_mkt=nan,
            beta_smb=nan,
            beta_hml=nan,
            r_squared=nan,
            n_obs=n,
            t_alpha=nan,
            t_mkt=nan,
            t_smb=nan,
            t_hml=nan,
        )

    excess = (p - aligned["rf"]).to_numpy(dtype=np.float64)
    X = np.column_stack(
        [
            np.ones(n),
            aligned["mkt_rf"].to_numpy(dtype=np.float64),
            aligned["smb"].to_numpy(dtype=np.float64),
            aligned["hml"].to_numpy(dtype=np.float64),
        ]
    )
    res = sm.OLS(excess, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
    coef = res.params
    tvals = res.tvalues
    alpha_daily = float(coef[0])
    alpha_annual = (
        (1.0 + alpha_daily) ** ANNUALIZATION_DAYS - 1.0 if abs(alpha_daily) < 1 else float("nan")
    )
    return FamaFrench3Fit(
        alpha_daily=alpha_daily,
        alpha_annual=alpha_annual,
        beta_mkt=float(coef[1]),
        beta_smb=float(coef[2]),
        beta_hml=float(coef[3]),
        r_squared=float(res.rsquared),
        n_obs=n,
        t_alpha=float(tvals[0]),
        t_mkt=float(tvals[1]),
        t_smb=float(tvals[2]),
        t_hml=float(tvals[3]),
    )


def rolling_factor_betas(
    portfolio_returns: pd.Series,
    factors: FamaFrenchFactors | pd.DataFrame,
    *,
    window: int = 90,
    rf_daily: float | None = None,
) -> pd.DataFrame:
    """Rolling-OLS betas of the portfolio on (Mkt-RF, SMB, HML).

    Returns a DataFrame with columns ``["beta_mkt", "beta_smb", "beta_hml"]``,
    indexed by the right edge of each window. NaNs occupy the first
    ``window-1`` rows.
    """
    df: pd.DataFrame = factors.df if hasattr(factors, "df") else factors  # type: ignore[assignment]
    aligned = df.reindex(portfolio_returns.index).ffill(limit=3).dropna()
    p = portfolio_returns.reindex(aligned.index).dropna()
    aligned = aligned.reindex(p.index)
    rf = aligned["rf"] if rf_daily is None else pd.Series(rf_daily, index=p.index)
    excess = p - rf
    fac = aligned[["mkt_rf", "smb", "hml"]]

    out = pd.DataFrame(
        index=p.index,
        columns=["beta_mkt", "beta_smb", "beta_hml"],
        dtype=float,
    )
    n = int(p.shape[0])
    if n < window:
        return out

    y = excess.to_numpy(dtype=np.float64)
    F = fac.to_numpy(dtype=np.float64)
    ones = np.ones((n, 1))
    X_full = np.concatenate([ones, F], axis=1)

    betas = np.full((n, 3), np.nan)
    for end in range(window, n + 1):
        Xw = X_full[end - window : end]
        yw = y[end - window : end]
        # solve OLS via lstsq for numerical stability
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        betas[end - 1] = coef[1:]
    out.loc[:, "beta_mkt"] = betas[:, 0]
    out.loc[:, "beta_smb"] = betas[:, 1]
    out.loc[:, "beta_hml"] = betas[:, 2]
    return out


# ---------------------------------------------------------------------------
# Top-level metric bundle
# ---------------------------------------------------------------------------


def compute_risk_metrics(
    equity_curve: pd.Series,
    benchmark: pd.Series,
    *,
    rebalance_dates: pd.Index | None = None,
    weights_history: pd.DataFrame | None = None,
    turnover: pd.Series | None = None,
    factors: FamaFrenchFactors | pd.DataFrame | None = None,
    rf_daily: float = 0.0,
    periods_per_year: int = ANNUALIZATION_DAYS,
) -> dict[str, float]:
    """Compute every Phase-3 risk metric in one go.

    Parameters
    ----------
    equity_curve, benchmark
        Strictly positive ``pd.Series`` indexed by trading date. Both should
        be expressed in the same units (e.g. both starting at 1.0 or both in
        $-NAV space).
    rebalance_dates
        Optional index of dates at which the portfolio was rebalanced. Used to
        annualise turnover correctly.
    weights_history
        Optional ``DataFrame`` with one row per rebalance date and one column
        per ticker, holding *target weights*. Used to compute Herfindahl and
        max-weight metrics across the path.
    turnover
        Optional series indexed by rebalance date with per-rebalance turnover
        (``L1 / 2`` of weight changes). Used for the per-rebalance turnover
        statistics.
    factors
        Optional Fama-French factor frame for the FF3 regression block.
    rf_daily
        Daily risk-free rate (decimal). When ``factors`` is provided, the FF3
        regression uses the file's ``rf`` column instead.

    Returns
    -------
    Flat dict of finite floats (NaN when undefined).
    """
    out: dict[str, float] = {}

    eq = equity_curve.dropna()
    bm = benchmark.dropna()
    eq, bm = eq.align(bm, join="inner")
    rets = _equity_to_returns(eq)
    brets = _equity_to_returns(bm)

    # --------------------------- core return / vol ---------------------------
    out["ann_return"] = annualised_return(rets, periods_per_year=periods_per_year)
    out["ann_vol"] = annualised_vol(rets, periods_per_year=periods_per_year)
    out["sharpe"] = sharpe_ratio(rets, rf_daily=rf_daily, periods_per_year=periods_per_year)
    out["sortino"] = sortino_ratio(rets, rf_daily=rf_daily, periods_per_year=periods_per_year)
    out["max_drawdown"] = max_drawdown(eq)
    out["ulcer_index"] = ulcer_index(eq)
    out["calmar"] = calmar_ratio(eq, periods_per_year=periods_per_year)

    # --------------------------- vs benchmark --------------------------------
    out["benchmark_ann_return"] = annualised_return(brets, periods_per_year=periods_per_year)
    out["benchmark_ann_vol"] = annualised_vol(brets, periods_per_year=periods_per_year)
    out["tracking_error_annual"] = tracking_error(rets, brets, periods_per_year=periods_per_year)
    out["information_ratio"] = information_ratio(rets, brets, periods_per_year=periods_per_year)

    capm = capm_regression(rets, brets, rf_daily=rf_daily)
    out["beta"] = capm.beta
    out["alpha_daily"] = capm.alpha_daily
    out["alpha_annual"] = capm.alpha_annual
    out["capm_r2"] = capm.r_squared

    # --------------------------- Fama-French 3 -------------------------------
    if factors is not None:
        ff = famafrench3_regression(rets, factors)
        out["ff3_alpha_daily"] = ff.alpha_daily
        out["ff3_alpha_annual"] = ff.alpha_annual
        out["ff3_beta_mkt"] = ff.beta_mkt
        out["ff3_beta_smb"] = ff.beta_smb
        out["ff3_beta_hml"] = ff.beta_hml
        out["ff3_r2"] = ff.r_squared
        out["ff3_t_alpha"] = ff.t_alpha
        out["ff3_t_mkt"] = ff.t_mkt
        out["ff3_t_smb"] = ff.t_smb
        out["ff3_t_hml"] = ff.t_hml
        out["ff3_n_obs"] = float(ff.n_obs)

    # --------------------------- turnover ------------------------------------
    if turnover is not None and len(turnover) > 0:
        t = turnover.dropna()
        out["turnover_per_rebal_avg"] = float(t.mean()) if not t.empty else float("nan")
        out["turnover_per_rebal_p95"] = (
            float(np.quantile(t.to_numpy(), 0.95)) if t.shape[0] > 0 else float("nan")
        )
        # Annualise turnover assuming the rebalance dates are evenly spaced.
        # If we have rebalance_dates we infer the frequency from them.
        n_rebal = int(t.shape[0])
        years_span = _span_years(rebalance_dates if rebalance_dates is not None else t.index)
        out["turnover_annualized"] = (
            float(t.sum() / years_span) if years_span and years_span > 0 else float("nan")
        )
        out["n_rebalances"] = float(n_rebal)

    # --------------------------- weights -------------------------------------
    if weights_history is not None and not weights_history.empty:
        w = weights_history.fillna(0.0)
        hhi_path = (w**2).sum(axis=1)
        out["hhi_avg"] = float(hhi_path.mean())
        out["hhi_p95"] = float(np.quantile(hhi_path.to_numpy(), 0.95))
        out["max_weight_avg"] = float(w.max(axis=1).mean())
        out["max_weight_overall"] = float(w.to_numpy().max())
        out["effective_n_avg"] = float(_safe_div(1.0, hhi_path.mean()))

    return out


def _span_years(index: pd.Index | None) -> float | None:
    """Return the number of *calendar* years spanned by an index."""
    if index is None or len(index) < 2:
        return None
    try:
        delta = pd.to_datetime(index[-1]) - pd.to_datetime(index[0])
    except (TypeError, ValueError):
        return None
    days = delta.days
    if days <= 0:
        return None
    return float(days) / 365.25


__all__ = [
    "ANNUALIZATION_DAYS",
    "CapmFit",
    "FamaFrench3Fit",
    "annualised_return",
    "annualised_vol",
    "calmar_ratio",
    "capm_regression",
    "compute_risk_metrics",
    "drawdown",
    "famafrench3_regression",
    "information_ratio",
    "max_drawdown",
    "rolling_factor_betas",
    "sharpe_ratio",
    "sortino_ratio",
    "tracking_error",
    "ulcer_index",
]
