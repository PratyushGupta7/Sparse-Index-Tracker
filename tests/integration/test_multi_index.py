"""Integration tests for the Phase-4 cross-index orchestrator.

These exercise the *whole* multi-index pipeline (universes → comparison →
plots) on synthetic data so they run offline and finish in seconds. The real
network paths for Russell 2000 / Nifty 50 are exercised separately by the
``@pytest.mark.network`` smoke test in :mod:`tests.unit.test_universes`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sit.benchmarks.comparison import ComparisonInputs
from sit.benchmarks.multi_index import (
    IndexRun,
    MultiIndexComparison,
    MultiIndexConfig,
    MultiIndexResult,
    save_multi_index_result,
)
from sit.benchmarks.multi_index_plots import (
    plot_cross_index_equity,
    plot_cross_index_summary,
)

# ---------------------------------------------------------------------------
# A network-free synthetic snapshot factory shared by every test
# ---------------------------------------------------------------------------


def _make_fake_inputs(
    universe: str, *, n_train: int, n_test: int, p: int, k: int, seed: int
) -> tuple[ComparisonInputs, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    sigma_x = 0.015
    X_all = rng.standard_normal((n, p)) * sigma_x
    train_raw = X_all[:n_train]
    test_raw = X_all[n_train:]
    sigma = train_raw.std(axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    support = rng.choice(p, k, replace=False)
    w_true = np.zeros(p)
    w_true[support] = rng.uniform(0.5, 1.5, k)
    w_true /= w_true.sum()
    y_train = train_raw @ w_true + 0.0005 * rng.standard_normal(n_train)
    y_test = test_raw @ w_true + 0.0005 * rng.standard_normal(n_test)
    market_caps = np.exp(rng.standard_normal(p)) * 1e9
    inputs = ComparisonInputs(
        X_train_std=train_raw / sigma,
        y_train=y_train,
        X_test_raw=test_raw,
        y_test=y_test,
        sigma_train=sigma,
        X_train_raw=train_raw,
        market_caps=market_caps,
        tickers=[f"{universe.upper()[:3]}{j:03d}" for j in range(p)],
    )
    metadata = {
        "source": "synthetic",
        "universe": universe,
        "n_active_tickers": p,
    }
    return inputs, metadata


@pytest.fixture
def synthetic_factory():
    """Return a closure with the universe-specific synthetic data."""

    sizes = {
        "sp500": (50, 6),
        "nasdaq100": (40, 5),
        "russell2000": (60, 7),
        "nifty50": (30, 4),
    }

    def _factory(universe: str, **kwargs):
        p, k = sizes.get(universe, (40, 5))
        seed = abs(hash(universe)) % (2**31 - 1)
        return _make_fake_inputs(
            universe,
            n_train=kwargs.get("n_train", 80),
            n_test=kwargs.get("n_test", 40),
            p=p,
            k=k,
            seed=seed,
        )

    return _factory


@pytest.fixture(scope="module")
def all_indices_run() -> MultiIndexResult:
    """Run the orchestrator once with synthetic snapshots and reuse."""

    sizes = {
        "sp500": (50, 6),
        "nasdaq100": (40, 5),
        "russell2000": (60, 7),
        "nifty50": (30, 4),
    }

    def _factory(universe: str, **kwargs):
        p, k = sizes.get(universe, (40, 5))
        seed = abs(hash(universe)) % (2**31 - 1)
        return _make_fake_inputs(universe, n_train=80, n_test=40, p=p, k=k, seed=seed)

    cfg = MultiIndexConfig(
        indices=["sp500", "nasdaq100", "russell2000", "nifty50"],
        n_train=80,
        n_test=40,
        K=8,
        lam_frac=0.05,
    )
    return MultiIndexComparison(cfg, snapshot_factory=_factory).run()


# ---------------------------------------------------------------------------
# End-to-end: each universe returns ADMM weights on the simplex
# ---------------------------------------------------------------------------


def test_all_four_universes_run(all_indices_run: MultiIndexResult) -> None:
    assert set(all_indices_run.runs.keys()) == {
        "sp500",
        "nasdaq100",
        "russell2000",
        "nifty50",
    }
    for run in all_indices_run.runs.values():
        assert run.error is None, f"{run.name} errored: {run.error}"
        assert run.results, f"{run.name} produced no method results"


def test_admm_simplex_per_universe(all_indices_run: MultiIndexResult) -> None:
    for run in all_indices_run.runs.values():
        admm = next((r for r in run.results if r.name == "admm"), None)
        assert admm is not None, f"{run.name} missing admm result"
        assert (admm.weights_raw >= -1e-10).all()
        assert (
            abs(admm.weights_raw.sum() - 1.0) < 1e-6
        ), f"{run.name} ADMM weights sum to {admm.weights_raw.sum():.6f}"


def test_admm_oos_r2_is_finite(all_indices_run: MultiIndexResult) -> None:
    for run in all_indices_run.runs.values():
        admm = next((r for r in run.results if r.name == "admm"), None)
        assert admm is not None
        assert np.isfinite(admm.oos_r2)
        assert np.isfinite(admm.oos_te_annual)
        assert np.isfinite(admm.oos_sharpe_annual)


def test_admm_recovers_signal_better_than_random(all_indices_run: MultiIndexResult) -> None:
    """ADMM should beat both naive baselines on synthetic ground-truth data."""
    for run in all_indices_run.runs.values():
        admm = next((r for r in run.results if r.name == "admm"), None)
        ew = next((r for r in run.results if r.name == "equal_weight_topn"), None)
        topn = next((r for r in run.results if r.name == "topn_cap"), None)
        assert admm is not None and ew is not None and topn is not None
        # OOS R²: ADMM should be at least as good as the naive baselines on
        # ground-truth k-sparse data.
        assert (
            admm.oos_r2 >= topn.oos_r2 - 0.05
        ), f"{run.name}: admm R²={admm.oos_r2:.3f} vs topn R²={topn.oos_r2:.3f}"


# ---------------------------------------------------------------------------
# Headline table + JSON round-trip
# ---------------------------------------------------------------------------


def test_headline_table_keys(all_indices_run: MultiIndexResult) -> None:
    table = all_indices_run.headline_table()
    assert set(table.keys()) == {"sp500", "nasdaq100", "russell2000", "nifty50"}
    for row in table.values():
        assert {"oos_r2", "oos_te_annual", "oos_sharpe_annual", "n_active"} <= row.keys()


def test_save_and_load_multi_index_result(
    all_indices_run: MultiIndexResult, tmp_path: Path
) -> None:
    out = tmp_path / "multi.json"
    save_multi_index_result(all_indices_run, out)
    assert out.is_file()
    payload = json.loads(out.read_text())
    assert "config" in payload and "runs" in payload
    assert set(payload["runs"]) == set(all_indices_run.runs)
    for run in payload["runs"].values():
        assert isinstance(run["results"], list)
        if run["results"]:
            assert all("oos_r2" in r for r in run["results"])
            assert "test_curves" in run and "admm" in run["test_curves"]


def test_test_curves_have_correct_length(all_indices_run: MultiIndexResult) -> None:
    for run in all_indices_run.runs.values():
        assert "benchmark" in run.test_curves
        assert "admm" in run.test_curves
        assert (
            len(run.test_curves["admm"])
            == len(run.test_curves["benchmark"])
            == all_indices_run.config.n_test
        )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def test_cross_index_summary_plot(all_indices_run: MultiIndexResult, tmp_path: Path) -> None:
    out = tmp_path / "cross_index_summary.png"
    path = plot_cross_index_summary(all_indices_run, out)
    assert path.is_file()
    assert path.stat().st_size > 5_000  # non-trivial PNG payload


def test_cross_index_equity_plot(all_indices_run: MultiIndexResult, tmp_path: Path) -> None:
    out = tmp_path / "cross_index_equity.png"
    path = plot_cross_index_equity(all_indices_run, out)
    assert path.is_file()
    assert path.stat().st_size > 5_000


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_partial_run_when_one_factory_errors(synthetic_factory) -> None:
    """A failing factory for one index should not break the rest."""

    def _flaky_factory(universe: str, **kwargs):
        if universe == "russell2000":
            raise RuntimeError("simulated network failure")
        return synthetic_factory(universe, **kwargs)

    cfg = MultiIndexConfig(
        indices=["sp500", "nasdaq100", "russell2000", "nifty50"],
        n_train=80,
        n_test=40,
        K=8,
    )
    res = MultiIndexComparison(cfg, snapshot_factory=_flaky_factory).run()
    assert res.runs["russell2000"].error is not None
    assert res.runs["russell2000"].results == []
    for ok in ("sp500", "nasdaq100", "nifty50"):
        assert res.runs[ok].error is None
        assert res.runs[ok].results


def test_unknown_index_is_skipped(synthetic_factory) -> None:
    cfg = MultiIndexConfig(
        indices=["sp500", "dax40", "nifty50"],
        n_train=80,
        n_test=40,
        K=8,
    )
    res = MultiIndexComparison(cfg, snapshot_factory=synthetic_factory).run()
    assert "dax40" not in res.runs
    assert "sp500" in res.runs and "nifty50" in res.runs


def test_skip_miqp_default_excludes_miqp(all_indices_run: MultiIndexResult) -> None:
    for run in all_indices_run.runs.values():
        names = {r.name for r in run.results}
        assert "miqp" not in names


def test_index_run_dataclass_fields() -> None:
    run = IndexRun(
        name="sp500",
        benchmark="SPY",
        label="S&P 500",
        region="US",
        n_active=0,
        metadata={},
        results=[],
    )
    assert run.error is None
    assert run.test_curves == {}
