"""sit/benchmarks/datasets.py — assemble ``ComparisonInputs`` from various sources.

Two factories:

* :func:`make_synthetic_dataset` — fully reproducible, network-free k-sparse
  problem. Used by tests + as a CI default.
* :func:`make_sp500_snapshot` — fresh fetch of the S&P 500 universe + train /
  held-out test split (the same pipeline the Phase-3 validator uses).
  Requires network.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sit.benchmarks.comparison import ComparisonInputs

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


@dataclass
class SyntheticTruth:
    """Side-information returned with a synthetic dataset for support-recovery checks."""

    support: NDArray[np.integer]
    weights: NDArray[np.floating]
    sigma: NDArray[np.floating]


def make_synthetic_dataset(
    n_train: int = 120,
    n_test: int = 60,
    p: int = 100,
    k: int = 8,
    *,
    sigma_x: float = 0.02,
    noise_sigma: float = 0.001,
    seed: int = 20260511,
) -> tuple[ComparisonInputs, SyntheticTruth]:
    """Build a k-sparse synthetic tracking problem with train + test data.

    Returns the ``ComparisonInputs`` *plus* the truth (support + weights) so
    tests can assert support recovery.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_test

    # Daily-return-scale noise (~2 % vol) so the resulting series feels real.
    X_all = rng.standard_normal((n, p)) * sigma_x
    X_train_raw = X_all[:n_train]
    X_test_raw = X_all[n_train:]
    sigma = X_train_raw.std(axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    X_train_std = X_train_raw / sigma

    support = rng.choice(p, k, replace=False)
    w_true = np.zeros(p)
    w_true[support] = rng.uniform(0.5, 1.5, k)
    w_true = w_true / w_true.sum()

    y_train = X_train_raw @ w_true + noise_sigma * rng.standard_normal(n_train)
    y_test = X_test_raw @ w_true + noise_sigma * rng.standard_normal(n_test)

    # Synthetic market caps: log-normal so they span several orders of magnitude
    market_caps = np.exp(rng.standard_normal(p) * 1.5) * 1e9
    tickers = [f"SYN{j:03d}" for j in range(p)]

    inputs = ComparisonInputs(
        X_train_std=X_train_std,
        y_train=y_train,
        X_test_raw=X_test_raw,
        y_test=y_test,
        sigma_train=sigma,
        X_train_raw=X_train_raw,
        market_caps=market_caps,
        tickers=tickers,
    )
    truth = SyntheticTruth(support=support, weights=w_true, sigma=sigma)
    return inputs, truth


# ---------------------------------------------------------------------------
# Real S&P 500 snapshot (network)
# ---------------------------------------------------------------------------


def _fetch_sp500_tickers(benchmark: str = "SPY") -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    )
    html = urllib.request.urlopen(req).read().decode("utf-8")
    table = pd.read_html(html)[0]
    tickers = table["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    if benchmark in tickers:
        tickers.remove(benchmark)
    return tickers


def _download_returns(
    tickers: list[str],
    benchmark: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Download adjusted-close prices and convert to log returns.

    Returns
    -------
    train_returns
        Wide DataFrame of constituent log returns indexed by date.
    bench_returns
        Series of benchmark log returns aligned to ``train_returns``.
    active_tickers
        Final list of tickers with full coverage over the window.
    """
    import yfinance as yf

    all_tickers = [*tickers, benchmark]
    print(f"   Downloading {len(all_tickers)} tickers from {start_date} to {end_date}…")
    df = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        threads=True,
        group_by="ticker",
    )

    # Try Adj Close, fall back to Close
    if isinstance(df.columns, pd.MultiIndex):
        try:
            prices = df.xs("Adj Close", axis=1, level=1)
        except KeyError:
            prices = df.xs("Close", axis=1, level=1)
    else:  # single-ticker quirk
        prices = df.get("Adj Close", df.get("Close"))

    prices = prices.dropna(axis=1, how="any")
    if benchmark not in prices.columns:
        raise RuntimeError(f"Benchmark {benchmark} did not download cleanly.")

    # Log returns (small-return approximation to simple returns)
    returns = np.log(prices / prices.shift(1)).dropna()
    bench = returns[benchmark]
    constituents = returns.drop(columns=[benchmark])
    active = list(constituents.columns)
    print(f"   ✅ {len(active)}/{len(tickers)} stocks have full coverage")
    return constituents, bench, active


def make_sp500_snapshot(
    *,
    start_date: str = "2025-04-01",
    end_date: str = "2026-03-09",
    n_train: int = 120,
    n_test: int = 60,
    benchmark: str = "SPY",
    market_cap_cache: bool = True,
) -> tuple[ComparisonInputs, dict]:
    """Fresh S&P 500 snapshot with a clean train / held-out test split.

    Mirrors ``Phase3Validator.download_and_split`` so Phase-2 numbers are
    directly comparable to Phase-3 ones.
    """
    tickers = _fetch_sp500_tickers(benchmark=benchmark)
    constituents, bench, active = _download_returns(tickers, benchmark, start_date, end_date)

    needed = n_train + n_test
    if len(constituents) < needed:
        raise RuntimeError(
            f"Got {len(constituents)} trading days but need {needed} (n_train + n_test)."
        )
    constituents = constituents.iloc[:needed]
    bench = bench.iloc[:needed]

    train_returns_raw = constituents.iloc[:n_train].to_numpy(dtype=np.float64)
    test_returns_raw = constituents.iloc[n_train:].to_numpy(dtype=np.float64)
    y_train = bench.iloc[:n_train].to_numpy(dtype=np.float64)
    y_test = bench.iloc[n_train:].to_numpy(dtype=np.float64)

    sigma = train_returns_raw.std(axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    X_train_std = train_returns_raw / sigma

    # Market caps for naive baselines (cached)
    market_caps = None
    if market_cap_cache:
        from sit.data.market_caps import market_caps_array

        try:
            market_caps = market_caps_array(active, fallback=np.nan)
        except Exception as exc:  # pragma: no cover
            print(f"   ⚠️  market-cap fetch failed: {exc!r}")
            market_caps = None

    inputs = ComparisonInputs(
        X_train_std=X_train_std,
        y_train=y_train,
        X_test_raw=test_returns_raw,
        y_test=y_test,
        sigma_train=sigma,
        X_train_raw=train_returns_raw,
        market_caps=market_caps,
        tickers=active,
    )
    metadata = {
        "source": "yfinance + Wikipedia",
        "benchmark": benchmark,
        "start_date": start_date,
        "end_date": end_date,
        "n_train": n_train,
        "n_test": n_test,
        "n_active_tickers": len(active),
    }
    return inputs, metadata


__all__ = [
    "SyntheticTruth",
    "make_sp500_snapshot",
    "make_synthetic_dataset",
]
