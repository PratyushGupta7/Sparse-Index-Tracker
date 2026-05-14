"""sit/backtest/walkforward.py — rolling weekly/monthly walk-forward backtester.

Architecture mirrors ``sit.benchmarks.comparison`` (the Phase-2 head-to-head
orchestrator): a Pydantic v2 ``WalkForwardConfig`` describes the experiment, a
single ``run(config, prices, benchmark)`` entry point loops over rebalance
dates, and a flat ``WalkForwardResult`` dataclass packages the equity curves,
turnover, costs, weights history and per-rebalance diagnostics.

Module layout
-------------
``WalkForwardConfig``     Pydantic v2 spec.
``WalkForwardResult``     Output bundle.
``run(...)``              The entry point.
``save_walkforward_result(...)``  Persist a result as JSON.
``load_price_panel_yfinance(...)`` Convenience downloader for the CLI.

Critical invariants
-------------------
* **Standardised vs raw features.** ADMM/LASSO/FISTA/OMP train on standardised
  returns (per-window ``σ_train``) and their weights are converted to raw,
  tradeable simplex weights via ``sit.solvers.base.raw_to_simplex_via_std``.
  MIQP and the naive baselines train on raw features because their
  ``sum(w) = 1`` constraint lives in raw-return units. (Same discipline as
  the Phase-2 orchestrator; this is the #1 historical bug source.)
* **Survivorship-bias flag.** When the price frame uses today's S&P 500
  membership we set ``result.survivorship_bias_flag = True`` and emit a
  warning in the JSON metadata. A ``data/membership_changes.csv`` overlay
  hook is exposed (best-effort; Phase 3 default is honest disclosure rather
  than partial data).
* **Cash residual.** Any unallocated cash earns the BIL T-bill proxy daily
  return. The orchestrator looks up BIL prices once at the start of the run
  and re-uses them across every method.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from sit.backtest.transaction_costs import LinearCost, TransactionCostModel
from sit.solvers.admm import SparseTrackerADMM
from sit.solvers.base import raw_to_simplex_via_std, to_simplex
from sit.solvers.fista import FISTA
from sit.solvers.lasso import SklearnLassoSolver
from sit.solvers.omp import OMPSolver

if TYPE_CHECKING:
    from datetime import date

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class WalkForwardConfig(BaseModel):
    """Pydantic v2 spec for one walk-forward experiment.

    Field semantics
    ---------------
    start_date / end_date
        Inclusive ISO-format strings (``"2018-01-01"``) bounding the
        backtest. The orchestrator demands at least ``lookback_days``
        observations before ``start_date`` to bootstrap the first refit.
    lookback_days
        Rolling training window in trading days. 120 ≈ 6 calendar months.
    rebalance
        ``"weekly"`` (Friday close) or ``"monthly"`` (last trading day of
        month).
    tx_bps
        Cost in basis points *per side* for the linear cost model.
    initial_capital
        Starting NAV in dollars.
    cash_proxy_ticker
        Yahoo ticker used for unallocated cash returns. ``"BIL"`` is the
        SPDR 1–3 month T-bill ETF.
    methods
        List of solver-name strings. Supported names: ``admm``, ``lasso``,
        ``fista``, ``omp``, ``miqp``, ``equal_weight_topn``,
        ``benchmark`` (special-cased to track the benchmark itself).
    lam_frac, K
        Convex penalty fraction and cardinality target shared across methods.
    solver_kwargs
        Per-method overrides keyed by solver name.
    """

    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    lookback_days: int = Field(120, ge=20)
    rebalance: str = Field("weekly", pattern=r"^(weekly|monthly)$")
    tx_bps: float = Field(5.0, ge=0.0)
    initial_capital: float = Field(1_000_000.0, gt=0)
    cash_proxy_ticker: str = "BIL"
    methods: list[str] = Field(
        default_factory=lambda: [
            "admm",
            "lasso",
            "omp",
            "equal_weight_topn",
            "benchmark",
        ]
    )
    lam_frac: float = Field(0.05, gt=0.0, le=1.0)
    K: int = Field(50, ge=1)
    solver_kwargs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    benchmark_ticker: str = "SPY"
    membership_changes_path: str | None = None
    """Optional CSV path with point-in-time membership changes for survivorship correction."""
    miqp_time_limit: float = Field(60.0, gt=0.0)
    seed: int = 20260511
    threshold: float = Field(1e-6, ge=0.0)

    @field_validator("methods")
    @classmethod
    def _validate_methods(cls, v: list[str]) -> list[str]:
        allowed = {
            "admm",
            "lasso",
            "fista",
            "omp",
            "miqp",
            "equal_weight_topn",
            "benchmark",
        }
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}; allowed: {sorted(allowed)}")
        if not v:
            raise ValueError("`methods` must contain at least one entry.")
        return v

    @model_validator(mode="after")
    def _dates_consistent(self) -> WalkForwardConfig:
        try:
            start = pd.Timestamp(self.start_date)
            end = pd.Timestamp(self.end_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"start_date/end_date must be ISO-parseable: {exc}") from exc
        if start >= end:
            raise ValueError(f"start_date ({start}) must precede end_date ({end}).")
        return self


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Bundle of outputs from one walk-forward run.

    All ``Series``/``DataFrame`` fields are indexed by the daily price index.
    """

    equity_curves: dict[str, pd.Series]
    """Daily NAV per method (units = dollars, starting at config.initial_capital)."""

    weights_history: dict[str, pd.DataFrame]
    """Per-rebalance target weights per method (rows=rebal dates, cols=tickers)."""

    turnover: dict[str, pd.Series]
    """Per-rebalance turnover (= L1 weight change / 2) per method."""

    transaction_costs: dict[str, pd.Series]
    """Per-rebalance dollar transaction cost per method."""

    rebalance_dates: pd.DatetimeIndex
    fit_times: dict[str, pd.Series]
    """Per-rebalance solver wall-clock time per method."""

    config: WalkForwardConfig
    benchmark_curve: pd.Series
    """The benchmark equity curve (initial_capital × cumulative product)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    survivorship_bias_flag: bool = True
    """True when the run used today's universe membership (default)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_rebalance_dates(
    index: pd.DatetimeIndex,
    rebalance: str,
) -> pd.DatetimeIndex:
    """Return the trading-day rebalance schedule embedded in ``index``."""
    s = pd.Series(0, index=index)
    if rebalance == "weekly":
        # Last trading day of each ISO week.
        weekly = s.groupby([index.isocalendar().year, index.isocalendar().week])
        rebal = pd.DatetimeIndex(weekly.apply(lambda g: g.index[-1]).tolist())
    elif rebalance == "monthly":
        monthly = s.groupby([index.year, index.month])
        rebal = pd.DatetimeIndex(monthly.apply(lambda g: g.index[-1]).tolist())
    else:  # pragma: no cover (validated upstream)
        raise ValueError(f"Unknown rebalance frequency: {rebalance}")
    return pd.DatetimeIndex(sorted(set(rebal)))


def _train_window_indices(
    all_index: pd.DatetimeIndex,
    rebal_date: pd.Timestamp,
    lookback_days: int,
) -> slice | None:
    """Return the *positional* slice for the rolling-window training data.

    The training window is the last ``lookback_days`` rows *strictly before*
    ``rebal_date``. Returns ``None`` if there aren't enough rows yet.
    """
    pos = int(all_index.get_indexer(pd.DatetimeIndex([rebal_date]))[0])
    if pos < 0:
        # rebal_date isn't in the price index — should not happen since we
        # generate it from the index itself.
        return None
    if pos < lookback_days:
        return None
    return slice(pos - lookback_days, pos)


def _herfindahl_columnwise(weights: pd.DataFrame) -> pd.Series:
    """Per-row Herfindahl: sum of squared weights along each row."""
    return (weights**2).sum(axis=1)


def _solver_for(name: str, *, lam: float, K: int, cfg: WalkForwardConfig) -> Any:
    """Instantiate a solver by name with the orchestrator's defaults."""
    user_kwargs = cfg.solver_kwargs.get(name, {})
    if name == "admm":
        return SparseTrackerADMM(
            lam=lam,
            max_iter=user_kwargs.get("max_iter", 3000),
            tol=user_kwargs.get("tol", 1e-6),
            verbose=False,
        )
    if name == "lasso":
        return SklearnLassoSolver(
            lam=lam,
            max_iter=user_kwargs.get("max_iter", 20_000),
            tol=user_kwargs.get("tol", 1e-7),
        )
    if name == "fista":
        return FISTA(
            lam=lam,
            max_iter=user_kwargs.get("max_iter", 3000),
            tol=user_kwargs.get("tol", 1e-6),
        )
    if name == "omp":
        return OMPSolver(K=K)
    if name == "miqp":
        # Lazy import — MOSEK may not be available everywhere.
        from sit.solvers.miqp import MIQPSolver

        return MIQPSolver(
            K=K,
            time_limit=user_kwargs.get("time_limit", cfg.miqp_time_limit),
            mip_gap=user_kwargs.get("mip_gap", 1e-2),
            enforce_simplex=True,
        )
    raise ValueError(f"Unknown solver: {name}")


def _equal_weight_topn(
    train_returns: NDArray[np.floating],
    K: int,
) -> NDArray[np.floating]:
    """Equal-weight the top-K stocks by *average return* in the training window.

    Used as the ``equal_weight_topn`` baseline inside the walk-forward
    orchestrator when no market-cap snapshot is available (the production
    cap-weighted variant is in ``sit.solvers.naive``).
    """
    avg_ret = train_returns.mean(axis=0)
    top_idx = np.argsort(avg_ret)[::-1][:K]
    w = np.zeros(train_returns.shape[1])
    w[top_idx] = 1.0 / K
    return w


def _fit_one(
    name: str,
    *,
    train_raw: NDArray[np.floating],
    train_std: NDArray[np.floating],
    sigma: NDArray[np.floating],
    y_train: NDArray[np.floating],
    lam: float,
    K: int,
    cfg: WalkForwardConfig,
    threshold: float,
) -> tuple[NDArray[np.floating], float, bool]:
    """Fit one method on the rolling window. Returns (raw simplex weights, fit time, converged)."""
    t0 = time.perf_counter()

    if name == "benchmark":
        # Special-case: "benchmark" portfolio = literally the benchmark itself.
        # We model it as a single position with weight 1 in a phantom ticker
        # whose returns equal y_train. The orchestrator handles this in
        # _advance_one_period via the BENCHMARK_PORTFOLIO sentinel.
        return np.array([]), time.perf_counter() - t0, True

    if name == "equal_weight_topn":
        w_raw = _equal_weight_topn(train_raw, K=K)
        elapsed = time.perf_counter() - t0
        return to_simplex(w_raw, threshold=threshold, raise_if_empty=False), elapsed, True

    solver = _solver_for(name, lam=lam, K=K, cfg=cfg)

    # Raw vs std fit modality
    if name == "miqp":
        solver.fit(train_raw, y_train)
        elapsed = time.perf_counter() - t0
        try:
            w_raw = solver.get_sparse_weights(threshold=threshold)
        except ValueError:
            w_raw = to_simplex(np.ones(train_raw.shape[1]), raise_if_empty=False)
        return w_raw, elapsed, bool(getattr(solver, "converged", True))

    # Convex / greedy convex methods → standardised features, post-hoc unscale
    solver.fit(train_std, y_train)
    elapsed = time.perf_counter() - t0
    converged = bool(getattr(solver, "converged", True))
    try:
        w_std = solver.get_sparse_weights(threshold=threshold)
        w_raw = raw_to_simplex_via_std(
            w_std,
            sigma,
            threshold=threshold,
            raise_if_empty=False,
        )
    except ValueError:
        # Fully degenerate (e.g. λ ≥ λ_max ⇒ all-zero weights) — fall back to uniform.
        w_raw = to_simplex(np.ones(train_raw.shape[1]), raise_if_empty=False)
    return w_raw, elapsed, converged


def _safe_pad_weights(w: NDArray[np.floating], universe: list[str]) -> pd.Series:
    """Convert a numpy weight vector to a Series indexed by ``universe``."""
    if w.size == 0:
        return pd.Series(dtype=float)
    if w.size != len(universe):
        raise ValueError(f"Weight length {w.size} != universe length {len(universe)}.")
    return pd.Series(w, index=universe)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run(
    config: WalkForwardConfig,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    *,
    cash_returns: pd.Series | None = None,
    membership_overlay: Mapping[date, set[str]] | None = None,
    cost_model: TransactionCostModel | None = None,
) -> WalkForwardResult:
    """Run a walk-forward backtest over the provided price panel.

    Parameters
    ----------
    config
        ``WalkForwardConfig`` describing the experiment.
    prices
        Wide DataFrame of daily prices (rows=trading dates, cols=tickers).
        Must include the full lookback ``[config.start_date - lookback,
        config.end_date]``.
    benchmark
        Daily price series for the benchmark, same index as ``prices``.
    cash_returns
        Optional daily simple-return series for the cash proxy. If omitted,
        cash earns 0 % per day (equivalent to NoCost sleeves).
    membership_overlay
        Optional point-in-time membership map (``{date → {tickers active on
        that date}}``). When provided, the orchestrator zero-masks weights
        of any tickers not in the membership set on each rebalance date and
        sets ``survivorship_bias_flag = False``.
    cost_model
        Transaction cost model. Defaults to ``LinearCost(config.tx_bps)``.

    Returns
    -------
    ``WalkForwardResult``.
    """
    if cost_model is None:
        cost_model = LinearCost(bps_per_side=float(config.tx_bps))

    prices = prices.copy().sort_index()
    benchmark = benchmark.copy().sort_index()
    benchmark.name = config.benchmark_ticker

    # Align benchmark with constituents
    common = prices.index.intersection(benchmark.index)
    prices = prices.reindex(common)
    benchmark = benchmark.reindex(common)
    universe = list(prices.columns)
    if config.benchmark_ticker in universe:
        prices = prices.drop(columns=[config.benchmark_ticker])
        universe = list(prices.columns)
    if cash_returns is None:
        cash_returns = pd.Series(0.0, index=common)
    cash_returns = cash_returns.reindex(common).fillna(0.0)

    # Compute simple daily returns
    returns = prices.pct_change().fillna(0.0)
    bench_returns = benchmark.pct_change().fillna(0.0)

    # Constrain to [start_date - lookback, end_date]
    start_ts = pd.Timestamp(config.start_date)
    end_ts = pd.Timestamp(config.end_date)
    if start_ts < common[0] or end_ts > common[-1]:
        warnings.warn(
            f"Requested {start_ts.date()}–{end_ts.date()} but prices cover "
            f"{common[0].date()}–{common[-1].date()}; clipping.",
            stacklevel=2,
        )
    sim_index = pd.DatetimeIndex(common[(common >= start_ts) & (common <= end_ts)])
    if sim_index.empty:
        raise ValueError("Simulation index is empty after clipping; check date bounds.")

    # Rebalance schedule
    rebal_dates = _generate_rebalance_dates(sim_index, config.rebalance)
    if rebal_dates.empty:
        raise ValueError("No rebalance dates generated; check rebalance frequency.")

    survivorship_flag = membership_overlay is None

    # Output containers
    equity_curves: dict[str, pd.Series] = {}
    weights_history: dict[str, pd.DataFrame] = {}
    turnover_log: dict[str, pd.Series] = {}
    cost_log: dict[str, pd.Series] = {}
    fit_time_log: dict[str, pd.Series] = {}

    # Run each method independently — keeps things simple, makes parallelism
    # trivial later. The cost is one universe-wide standardisation per method
    # per rebalance, but ``lookback_days × p`` is small (120 × 500) so it
    # stays cheap.
    for method in config.methods:
        logger.info("walkforward: starting %s", method)
        t_method0 = time.perf_counter()
        out = _run_one_method(
            method=method,
            sim_index=sim_index,
            common=pd.DatetimeIndex(common),
            returns=returns,
            bench_returns=bench_returns,
            cash_returns=cash_returns,
            universe=universe,
            rebal_dates=rebal_dates,
            cfg=config,
            cost_model=cost_model,
            membership_overlay=membership_overlay,
        )
        equity_curves[method] = out["equity"]
        weights_history[method] = out["weights"]
        turnover_log[method] = out["turnover"]
        cost_log[method] = out["costs"]
        fit_time_log[method] = out["fit_times"]
        logger.info("walkforward: finished %s in %.1fs", method, time.perf_counter() - t_method0)

    # Construct benchmark curve with the same start-of-day NAV convention as
    # the per-method equity series: NAV[t] reflects holdings at the start of
    # day t (so NAV[sim_index[0]] = initial_capital, then we accrue r_t).
    # ``equity[t]`` is the close-of-day-t NAV, so the cumulative product
    # uses (1 + r_t) directly without a leading shift. This keeps the
    # benchmark curve labelled by the trading-day index, matching how
    # ``compute_risk_metrics`` and FF3 regressions consume the series.
    bench_at_sim = bench_returns.reindex(sim_index).fillna(0.0)
    one_plus = 1.0 + bench_at_sim
    benchmark_curve = config.initial_capital * one_plus.cumprod()

    return WalkForwardResult(
        equity_curves=equity_curves,
        weights_history=weights_history,
        turnover=turnover_log,
        transaction_costs=cost_log,
        rebalance_dates=pd.DatetimeIndex(rebal_dates),
        fit_times=fit_time_log,
        config=config,
        benchmark_curve=benchmark_curve,
        survivorship_bias_flag=survivorship_flag,
        metadata={
            "n_assets": len(universe),
            "n_dates": len(sim_index),
            "n_rebalances": len(rebal_dates),
            "tx_bps": float(config.tx_bps),
            "rebalance": config.rebalance,
            "cost_model": getattr(cost_model, "name", type(cost_model).__name__),
        },
    )


def _run_one_method(
    *,
    method: str,
    sim_index: pd.DatetimeIndex,
    common: pd.DatetimeIndex,
    returns: pd.DataFrame,
    bench_returns: pd.Series,
    cash_returns: pd.Series,
    universe: list[str],
    rebal_dates: pd.DatetimeIndex,
    cfg: WalkForwardConfig,
    cost_model: TransactionCostModel,
    membership_overlay: Mapping[date, set[str]] | None,
) -> dict[str, Any]:
    """Internal: run a single method, return its outputs as a dict.

    The flow is:
        for each trading day t in sim_index:
            if t is a rebalance date and we have enough lookback:
                refit on returns[t - lookback : t]
                compute trades vs. previous holdings
                deduct linear tx-cost from NAV (= scale weights)
            mark-to-market: NAV_{t+1} = sum_i (NAV_t * w_i_t * (1 + r_{i,t+1})) + cash * (1 + r_cash_{t+1})
    """
    nav = float(cfg.initial_capital)
    weights = pd.Series(0.0, index=universe)
    cash_frac = 1.0  # Initially fully in cash until first rebalance.
    equity_values: list[float] = []
    weights_records: list[pd.Series] = []
    weights_index: list[pd.Timestamp] = []
    turnover_records: list[float] = []
    cost_records: list[float] = []
    fit_records: list[float] = []

    sim_pos = {ts: i for i, ts in enumerate(common)}
    for t in sim_index:
        if t in rebal_dates and method != "benchmark":
            train_slice = _train_window_indices(common, t, cfg.lookback_days)
            if train_slice is not None:
                train_returns_df = returns.iloc[train_slice]
                # Apply membership overlay (zero-out non-members)
                active_mask = np.ones(len(universe), dtype=bool)
                if membership_overlay is not None:
                    members = membership_overlay.get(t.date(), set(universe))
                    active_mask = np.array([u in members for u in universe], dtype=bool)
                train_raw = train_returns_df.to_numpy(dtype=np.float64)
                # Mask out non-members for fitting
                if not active_mask.all():
                    train_raw = train_raw.copy()
                    train_raw[:, ~active_mask] = 0.0
                sigma = train_raw.std(axis=0)
                sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)
                train_std = train_raw / sigma_safe
                y_train = bench_returns.iloc[train_slice].to_numpy(dtype=np.float64)
                lam_max = float(SparseTrackerADMM.compute_lambda_max(train_std, y_train))
                lam = cfg.lam_frac * lam_max
                w_new, fit_time, _ = _fit_one(
                    method,
                    train_raw=train_raw,
                    train_std=train_std,
                    sigma=sigma_safe,
                    y_train=y_train,
                    lam=lam,
                    K=cfg.K,
                    cfg=cfg,
                    threshold=cfg.threshold,
                )
                if w_new.size == 0:
                    # Benchmark or empty weights; do nothing.
                    pass
                else:
                    if not active_mask.all():
                        w_new = w_new * active_mask
                        s = float(w_new.sum())
                        if s < 1e-12:
                            w_new = active_mask.astype(np.float64) / max(int(active_mask.sum()), 1)
                        else:
                            w_new = w_new / s

                    new_weights = pd.Series(w_new, index=universe)
                    # Compute trades and tx cost
                    old_dollars = nav * weights * (1.0 - 0.0)  # weights already fraction
                    new_dollars = nav * new_weights
                    trades = new_dollars - old_dollars
                    tx_cost = cost_model.apply(trades)
                    nav = max(nav - float(tx_cost), 0.0)
                    weights = new_weights
                    cash_frac = max(0.0, 1.0 - float(weights.sum()))

                    weights_records.append(new_weights)
                    weights_index.append(t)
                    # Turnover = ½ × L1 weight change. Use the previous
                    # period's weights captured *before* updating.
                    if len(weights_records) >= 2:
                        prev = weights_records[-2]
                        diff = (new_weights - prev).abs().sum()
                    else:
                        diff = float(new_weights.abs().sum())
                    turnover_records.append(float(diff) * 0.5)
                    cost_records.append(float(tx_cost))
                    fit_records.append(float(fit_time))

        # Apply day-t's return *first*, then record the end-of-day NAV.
        # This way ``equity[t]`` is the close-of-day-t value and
        # ``pct_change()[t]`` is the return *earned over* day t — which is
        # what FF3 / CAPM regressions and tracking-error metrics expect.
        if method == "benchmark":
            r_t = float(bench_returns.iloc[sim_pos[t]])
            nav *= 1.0 + r_t
        else:
            r_t = float(returns.iloc[sim_pos[t]].dot(weights.values))
            nav = nav * (
                1.0 + r_t * (1.0 - cash_frac) + float(cash_returns.iloc[sim_pos[t]]) * cash_frac
            )
            # Drift weights with returns (NAV-aware re-normalisation between rebalances).
            if cash_frac < 1.0:
                growth = 1.0 + returns.iloc[sim_pos[t]].to_numpy(dtype=np.float64)
                weights_arr = weights.to_numpy(dtype=np.float64) * growth
                tot = float(weights_arr.sum())
                if tot > 1e-12:
                    weights = pd.Series(weights_arr / tot, index=universe) * (1.0 - cash_frac)
                # else: fully drifted to zero (extreme drawdown) — keep flat
        equity_values.append(float(nav))

    weights_df = (
        pd.DataFrame(weights_records, index=weights_index, columns=universe)
        if weights_records
        else pd.DataFrame(columns=universe)
    )
    turnover_series = pd.Series(turnover_records, index=weights_index, name="turnover")
    cost_series = pd.Series(cost_records, index=weights_index, name="cost_dollars")
    fit_series = pd.Series(fit_records, index=weights_index, name="fit_time_s")
    equity = pd.Series(equity_values, index=sim_index, name="equity").astype(float)
    return {
        "equity": equity,
        "weights": weights_df,
        "turnover": turnover_series,
        "costs": cost_series,
        "fit_times": fit_series,
    }


# ---------------------------------------------------------------------------
# Convenience: load price panel from yfinance
# ---------------------------------------------------------------------------


def load_price_panel_yfinance(
    tickers: list[str],
    benchmark: str,
    start_date: str,
    end_date: str,
    *,
    cash_proxy_ticker: str = "BIL",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Download the price panel for a walk-forward run.

    Returns ``(prices, benchmark_series, cash_returns, active_tickers)``.
    The cash returns are simple daily returns of the cash proxy ETF, dropped
    to the same index as ``prices``.

    This is the *runtime* convenience helper used by the CLI; tests bypass it
    by constructing their own price frames.
    """
    import yfinance as yf

    all_tickers = [*tickers, benchmark, cash_proxy_ticker]
    df = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        threads=True,
        group_by="ticker",
    )
    if isinstance(df.columns, pd.MultiIndex):
        try:
            prices_all = df.xs("Close", axis=1, level=1)
        except KeyError:
            prices_all = df.xs("Adj Close", axis=1, level=1)
    else:
        prices_all = df["Close"]

    prices_all = prices_all.dropna(axis=1, how="any")
    if benchmark not in prices_all.columns:
        raise RuntimeError(f"Benchmark {benchmark} did not download cleanly.")
    benchmark_prices = prices_all[benchmark]
    cash_prices = prices_all.get(cash_proxy_ticker)
    constituents = prices_all.drop(
        columns=[c for c in (benchmark, cash_proxy_ticker) if c in prices_all.columns]
    )
    active = list(constituents.columns)
    if cash_prices is not None:
        cash_returns = cash_prices.pct_change().fillna(0.0)
    else:
        cash_returns = pd.Series(0.0, index=constituents.index)
    return constituents, benchmark_prices, cash_returns, active


def save_walkforward_result(
    result: WalkForwardResult,
    out_path: Path,
    *,
    risk_metrics: Mapping[str, Mapping[str, float]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a walk-forward run as a single self-describing JSON file.

    The schema is::

        {
          "config": {...},
          "metadata": {...},
          "rebalance_dates": [...],
          "equity_curves": {method: {date: nav}},
          "turnover": {method: {date: turnover}},
          "transaction_costs": {method: {date: cost_dollars}},
          "fit_times": {method: {date: seconds}},
          "benchmark_curve": {date: nav},
          "survivorship_bias_flag": bool,
          "risk_metrics": {method: {metric: value}}    # optional
        }
    """
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _series_to_dict(s: pd.Series) -> dict[str, float]:
        return {pd.Timestamp(str(k)).strftime("%Y-%m-%d"): float(v) for k, v in s.items()}

    payload: dict[str, Any] = {
        "config": result.config.model_dump(),
        "metadata": dict(result.metadata),
        "rebalance_dates": [d.strftime("%Y-%m-%d") for d in result.rebalance_dates],
        "equity_curves": {k: _series_to_dict(v) for k, v in result.equity_curves.items()},
        "turnover": {k: _series_to_dict(v) for k, v in result.turnover.items()},
        "transaction_costs": {k: _series_to_dict(v) for k, v in result.transaction_costs.items()},
        "fit_times": {k: _series_to_dict(v) for k, v in result.fit_times.items()},
        "benchmark_curve": _series_to_dict(result.benchmark_curve),
        "survivorship_bias_flag": bool(result.survivorship_bias_flag),
    }
    if risk_metrics is not None:
        payload["risk_metrics"] = {k: dict(v) for k, v in risk_metrics.items()}
    if extra is not None:
        payload["extra"] = dict(extra)
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    return out_path


__all__ = [
    "WalkForwardConfig",
    "WalkForwardResult",
    "load_price_panel_yfinance",
    "run",
    "save_walkforward_result",
]
