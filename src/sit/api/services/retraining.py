"""Live retraining service — extracted verbatim from the legacy ``app.py``.

The math is unchanged. The only addition is a ``solve_time_ms`` field in
the return dict so the API can emit it as an OpenTelemetry attribute and
the frontend can drive its "ADMM iteration k / 5000" loading state.
"""

from __future__ import annotations

import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, cast

import numpy as np
import pandas as pd

from sit.api.settings import get_settings
from sit.data.universes import get_universe
from sit.solvers.admm import SparseTrackerADMM

DEFAULT_INDEX = "sp500"
BENCHMARK_FALLBACKS = {
    # Yahoo's index quote can intermittently fail for intraday/demo requests.
    # NIFTYBEES is the most liquid NSE ETF proxy and keeps the live demo usable.
    "^NSEI": ["NIFTYBEES.NS"],
}


def _quiet_yf_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
    import yfinance as yf

    # yfinance prints failed-ticker diagnostics directly to stdout/stderr.
    # We validate and drop missing columns ourselves, so keep API terminals clean.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return yf.download(*args, **kwargs)


def _fetch_tickers_live(index: str = DEFAULT_INDEX) -> tuple[list[str], str, int, bool]:
    tickers, benchmark = get_universe(index)
    if benchmark in tickers:
        tickers = [t for t in tickers if t != benchmark]
    source_universe_size = len(tickers)

    settings = get_settings()
    should_cap = source_universe_size > settings.live_universe_cap_threshold
    if should_cap:
        tickers = tickers[: settings.live_universe_max_tickers]
    return list(tickers), benchmark, source_universe_size, should_cap


def _close_series(data: pd.DataFrame, *, label: str) -> pd.Series:
    if data.empty:
        raise RuntimeError(f"No price data returned for {label}.")
    if isinstance(data.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"
        close = cast(pd.DataFrame, data[field])
        if close.shape[1] == 1:
            return close.iloc[:, 0].dropna()
        raise RuntimeError(f"Expected one close series for {label}, got {close.shape[1]}.")
    field = "Adj Close" if "Adj Close" in data.columns else "Close"
    return data[field].dropna()


def _download_benchmark(
    benchmark: str,
    *,
    start_date: str,
    end_date: str,
) -> tuple[pd.Series, str]:
    candidates = [benchmark, *BENCHMARK_FALLBACKS.get(benchmark, [])]
    errors: list[str] = []
    for candidate in candidates:
        try:
            data = _quiet_yf_download(
                candidate,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                multi_level_index=False,
                progress=False,
            )
            close = _close_series(data, label=candidate)
            if not close.empty:
                return close, candidate
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("Benchmark price download failed (" + "; ".join(errors) + ")")


def live_retrain(
    n_days: int = 120,
    lam_frac: float = 0.05,
    *,
    index: str = DEFAULT_INDEX,
) -> dict[str, Any]:
    """Download last ``n_days`` of returns, retrain ADMM, return fresh weights.

    Returns a dict with the same keys as the legacy ``_live_retrain`` plus
    ``solve_time_ms`` and ``solver_iterations``.
    """
    tickers, benchmark, source_universe_size, capped = _fetch_tickers_live(index=index)

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=n_days * 2)).strftime("%Y-%m-%d")

    bench_prices, benchmark_used = _download_benchmark(
        benchmark,
        start_date=start_date,
        end_date=end_date,
    )

    const_data = _quiet_yf_download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if const_data.empty:
        raise RuntimeError(f"No constituent data returned for {index}.")
    const_prices = (
        const_data["Close"] if isinstance(const_data.columns, pd.MultiIndex) else const_data
    )

    bench_ret = bench_prices.pct_change().dropna()
    const_ret = const_prices.pct_change().dropna(how="all")

    merged = pd.merge(
        bench_ret.rename(benchmark_used),
        const_ret,
        left_index=True,
        right_index=True,
        how="inner",
    )
    spy_col = merged[[benchmark_used]]
    const_clean = merged.drop(columns=[benchmark_used]).dropna(axis=1)
    merged = pd.concat([spy_col, const_clean], axis=1)

    if merged.shape[0] < n_days:
        raise RuntimeError(f"Only {merged.shape[0]} aligned trading days available, need {n_days}.")

    train_df = merged.tail(n_days)
    stock_names = [c for c in train_df.columns if c != benchmark_used]

    y_train = np.asarray(train_df[benchmark_used].values, dtype=np.float64)
    X_raw = np.asarray(train_df[stock_names].values, dtype=np.float64)

    var = X_raw.var(axis=0)
    good = var > 1e-12
    X_raw = X_raw[:, good]
    stock_names = [s for s, g in zip(stock_names, good) if g]

    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    X_std = (X_raw - mu) / sigma

    lam_max = SparseTrackerADMM.compute_lambda_max(X_std, y_train)
    lam = lam_frac * lam_max

    solver = SparseTrackerADMM(
        lam=lam,
        rho=1.0,
        max_iter=5000,
        tol=1e-6,
        adaptive_rho=True,
        verbose=False,
    )
    t0 = time.perf_counter()
    solver.fit(X_std, y_train)
    solve_time_ms = (time.perf_counter() - t0) * 1000.0
    weights = solver.get_raw_weights(sigma)

    active: dict[str, float] = {}
    for i, name in enumerate(stock_names):
        if weights[i] > 0:
            active[name] = float(weights[i])

    port_ret = X_raw @ weights
    ss_res = float(np.sum((y_train - port_ret) ** 2))
    ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    te = float(np.std(port_ret - y_train) * np.sqrt(252) * 100)

    return {
        "weights": active,
        "n_active": len(active),
        "n_universe": len(stock_names),
        "source_universe_size": source_universe_size,
        "live_universe_capped": capped,
        "benchmark": benchmark_used,
        "requested_benchmark": benchmark,
        "index": index,
        "r2_train": float(r2),
        "te_train": te,
        "iterations": int(solver.n_iter),
        "converged": bool(solver.converged),
        "train_start": train_df.index[0].strftime("%Y-%m-%d"),
        "train_end": train_df.index[-1].strftime("%Y-%m-%d"),
        "solve_time_ms": float(solve_time_ms),
        "solver_iterations": int(solver.n_iter),
    }
