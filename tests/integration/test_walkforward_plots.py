"""Integration tests for sit.backtest.plots — verify each plot renders to a PNG."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from sit.backtest import WalkForwardConfig, run
from sit.backtest.plots import (
    plot_walkforward_drawdown,
    plot_walkforward_equity,
    plot_walkforward_factor_loadings,
    plot_walkforward_turnover,
)
from sit.data.famafrench import make_synthetic_ff3


@pytest.fixture(scope="module")
def small_walkforward_result():
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2020-01-02", periods=400, freq="B")
    p = 12
    returns = rng.normal(2e-4, 0.012, size=(len(dates), p))
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + returns, axis=0),
        index=dates,
        columns=[f"S{i:02d}" for i in range(p)],
    )
    benchmark = pd.Series(100.0 * np.cumprod(1.0 + returns.mean(axis=1)), index=dates, name="BENCH")
    cfg = WalkForwardConfig(
        start_date="2020-04-01",
        end_date="2021-06-30",
        lookback_days=60,
        rebalance="weekly",
        tx_bps=5.0,
        methods=["admm", "lasso", "benchmark"],
        K=4,
        benchmark_ticker="BENCH",
    )
    return run(cfg, prices, benchmark)


def test_plot_equity_writes_file(tmp_path, small_walkforward_result):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
    out = tmp_path / "wf_equity.png"
    plot_walkforward_equity(small_walkforward_result, out, subtitle="integration test")
    assert out.is_file() and out.stat().st_size > 1000


def test_plot_drawdown_writes_file(tmp_path, small_walkforward_result):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
    out = tmp_path / "wf_dd.png"
    plot_walkforward_drawdown(small_walkforward_result, out)
    assert out.is_file() and out.stat().st_size > 1000


def test_plot_factor_loadings_writes_file(tmp_path, small_walkforward_result):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
    factors = make_synthetic_ff3(small_walkforward_result.equity_curves["admm"].index, seed=2024)
    out = tmp_path / "wf_betas.png"
    plot_walkforward_factor_loadings(
        small_walkforward_result, factors, out, method="admm", window=60
    )
    assert out.is_file() and out.stat().st_size > 1000


def test_plot_turnover_writes_file(tmp_path, small_walkforward_result):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
    out = tmp_path / "wf_turnover.png"
    plot_walkforward_turnover(small_walkforward_result, out, method="admm")
    assert out.is_file() and out.stat().st_size > 1000


def test_plot_turnover_handles_empty_series(tmp_path, small_walkforward_result):
    """Benchmark method has no turnover; plot should still produce a file."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
    out = tmp_path / "wf_turnover_bench.png"
    plot_walkforward_turnover(small_walkforward_result, out, method="benchmark")
    assert out.is_file() and out.stat().st_size > 1000
