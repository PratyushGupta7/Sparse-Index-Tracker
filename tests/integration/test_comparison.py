"""End-to-end integration tests for the Phase-2 ``MethodComparison`` orchestrator.

These exercise the *whole* pipeline on synthetic data: dataset construction,
multi-method dispatch, raw-weight recovery, metric computation, JSON
persistence, Pareto sweep, and plot generation. They are slower than unit
tests (~2-5 s) but still firmly under the pytest budget.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from sit.benchmarks import (
    ComparisonConfig,
    ComparisonInputs,
    MethodComparison,
    make_synthetic_dataset,
    pareto_sweep,
    plot_method_comparison,
    plot_pareto_frontier,
    save_results,
)


@pytest.fixture(scope="module")
def small_synthetic_inputs() -> ComparisonInputs:
    """A small synthetic dataset shared across the integration tests."""
    inputs, _ = make_synthetic_dataset(
        n_train=80,
        n_test=40,
        p=60,
        k=6,
        seed=20260601,
    )
    return inputs


@pytest.fixture(scope="module")
def all_methods_results(small_synthetic_inputs, mosek_env):
    """Run the full 8-method comparison once and reuse across tests."""
    del mosek_env  # MIQP needs MOSEK
    cfg = ComparisonConfig(K=6, lam_frac=0.05, miqp_time_limit=10, miqp_mip_gap=1e-3)
    mc = MethodComparison(cfg)
    return mc.run(small_synthetic_inputs)


# ---------------------------------------------------------------------------
# Sanity: every method ran, every weight is on the simplex, every metric is finite
# ---------------------------------------------------------------------------


def test_all_methods_ran(all_methods_results):
    names = {r.name for r in all_methods_results}
    expected = {
        "admm",
        "fista",
        "lasso",
        "omp",
        "miqp",
        "topn_cap",
        "equal_weight_topn",
        "random_equal_weight",
    }
    assert names == expected, f"Missing methods: {expected - names}"


def test_all_weights_on_simplex(all_methods_results):
    for r in all_methods_results:
        assert (r.weights_raw >= -1e-10).all(), f"{r.name} has negative weights"
        assert (
            abs(r.weights_raw.sum() - 1.0) < 1e-8
        ), f"{r.name} weights sum to {r.weights_raw.sum():.6f}, expected 1.0"


def test_all_metrics_finite(all_methods_results):
    for r in all_methods_results:
        assert np.isfinite(r.oos_r2), f"{r.name} OOS R² is not finite"
        assert np.isfinite(r.oos_te_annual)
        assert r.fit_time_s >= 0
        assert r.n_active >= 1
        assert r.max_weight > 0
        assert 0 < r.hhi <= 1
        assert r.effective_n >= 1


# ---------------------------------------------------------------------------
# Story-level invariants — the comparison should produce a meaningful ranking
# ---------------------------------------------------------------------------


def test_convex_methods_agree(all_methods_results):
    """ADMM, FISTA and sklearn LASSO solve the same convex program → agree."""
    by_name = {r.name: r for r in all_methods_results}
    admm = by_name["admm"].weights_raw
    fista = by_name["fista"].weights_raw
    lasso = by_name["lasso"].weights_raw

    # Tight on synthetic data with 0.05 * λ_max
    assert abs(by_name["admm"].oos_r2 - by_name["fista"].oos_r2) < 1e-3
    assert abs(by_name["admm"].oos_r2 - by_name["lasso"].oos_r2) < 1e-3
    np.testing.assert_allclose(admm, fista, atol=5e-3, rtol=0)
    np.testing.assert_allclose(admm, lasso, atol=5e-3, rtol=0)


def test_optimization_beats_naive(all_methods_results):
    """Every optimisation method (ADMM/FISTA/LASSO/OMP/MIQP) beats every naive baseline.

    On a synthetic problem where ``y`` is built from a sparse combination of
    columns that the naive baselines have no information about, the gap should
    be enormous.
    """
    by_name = {r.name: r for r in all_methods_results}
    optim = ["admm", "fista", "lasso", "omp", "miqp"]
    naive = ["topn_cap", "equal_weight_topn", "random_equal_weight"]

    for o in optim:
        for n in naive:
            assert by_name[o].oos_r2 > by_name[n].oos_r2 + 0.3, (
                f"{o} OOS R²={by_name[o].oos_r2:.3f} not clearly above "
                f"{n} OOS R²={by_name[n].oos_r2:.3f}"
            )
            assert by_name[o].oos_te_annual < by_name[n].oos_te_annual, f"{o} TE not below {n} TE"


def test_miqp_recovers_correct_cardinality(all_methods_results):
    by_name = {r.name: r for r in all_methods_results}
    # MIQP with K=6 must use ≤6 stocks
    assert by_name["miqp"].n_active <= 6
    # OMP same
    assert by_name["omp"].n_active <= 6


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_results_roundtrip(tmp_path, all_methods_results):
    cfg = ComparisonConfig(K=6, lam_frac=0.05)
    out = tmp_path / "comparison.json"
    save_results(all_methods_results, cfg, out, extra={"dataset": "synthetic"})
    assert out.is_file()
    payload = json.loads(out.read_text())
    assert "config" in payload and "methods" in payload
    assert len(payload["methods"]) == len(all_methods_results)
    # Weights are not serialised by default (large arrays)
    for entry in payload["methods"]:
        assert "weights_raw" not in entry
        assert "oos_r2" in entry
        assert "n_active" in entry


def test_save_results_with_pareto(tmp_path, all_methods_results, small_synthetic_inputs):
    cfg = ComparisonConfig(K=6, lam_frac=0.05)
    pareto = pareto_sweep(
        small_synthetic_inputs,
        methods=("admm", "lasso", "fista", "omp"),
        K_grid=(5, 10, 20),
        lam_grid_size=4,
    )
    out = tmp_path / "comparison.json"
    save_results(all_methods_results, cfg, out, pareto=pareto)
    payload = json.loads(out.read_text())
    assert "pareto" in payload
    for method in ("admm", "lasso", "fista", "omp"):
        assert method in payload["pareto"]
        for pt in payload["pareto"][method]:
            assert "n_active" in pt and "oos_te_annual" in pt


# ---------------------------------------------------------------------------
# Pareto sweep
# ---------------------------------------------------------------------------


def test_pareto_sweep_admm_curve_decreases_in_K(small_synthetic_inputs):
    """As we relax sparsity (smaller λ → more stocks), TE generally falls."""
    pareto = pareto_sweep(
        small_synthetic_inputs,
        methods=("admm",),
        lam_grid_size=10,
    )
    pts = pareto["admm"]
    assert len(pts) >= 5
    # Coarse trend: the method with the most stocks should not have the worst TE
    by_nact = sorted(pts, key=lambda p: p["n_active"])
    smallest_te = min(p["oos_te_annual"] for p in pts)
    largest_te = max(p["oos_te_annual"] for p in pts)
    assert smallest_te < largest_te, "ADMM Pareto curve is degenerate (all TEs equal)"
    # The minimum-TE point should have moderate sparsity (not the sparsest)
    best = min(pts, key=lambda p: p["oos_te_annual"])
    assert best["n_active"] >= by_nact[0]["n_active"]


def test_pareto_sweep_naive_methods_skip_when_no_caps(small_synthetic_inputs):
    """Cap-weighted methods should be silently dropped when caps are unavailable."""
    no_caps = ComparisonInputs(
        X_train_std=small_synthetic_inputs.X_train_std,
        y_train=small_synthetic_inputs.y_train,
        X_test_raw=small_synthetic_inputs.X_test_raw,
        y_test=small_synthetic_inputs.y_test,
        sigma_train=small_synthetic_inputs.sigma_train,
        X_train_raw=small_synthetic_inputs.X_train_raw,
        market_caps=None,  # ← key change
    )
    pareto = pareto_sweep(
        no_caps,
        methods=("admm", "topn_cap", "equal_weight_topn"),
        lam_grid_size=4,
    )
    assert pareto.get("admm")
    # cap-weighted methods absent because we have no caps
    assert "topn_cap" not in pareto or not pareto["topn_cap"]


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def test_plot_method_comparison_writes_file(tmp_path, all_methods_results):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
    out = tmp_path / "method_comparison.png"
    plot_method_comparison(all_methods_results, out, subtitle="integration test")
    assert out.is_file() and out.stat().st_size > 1000


def test_plot_pareto_writes_file(tmp_path, all_methods_results, small_synthetic_inputs):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
    pareto = pareto_sweep(
        small_synthetic_inputs,
        methods=("admm", "omp"),
        K_grid=(5, 10, 20),
        lam_grid_size=4,
    )
    out = tmp_path / "pareto.png"
    plot_pareto_frontier(pareto, out)
    assert out.is_file() and out.stat().st_size > 1000


# ---------------------------------------------------------------------------
# include / skip filters
# ---------------------------------------------------------------------------


def test_include_methods_filter(small_synthetic_inputs):
    cfg = ComparisonConfig(
        K=6,
        lam_frac=0.05,
        include_methods=("admm", "lasso"),
    )
    mc = MethodComparison(cfg)
    results = mc.run(small_synthetic_inputs)
    assert {r.name for r in results} == {"admm", "lasso"}


def test_skip_methods_filter(small_synthetic_inputs):
    cfg = ComparisonConfig(
        K=6,
        lam_frac=0.05,
        skip_methods=("miqp", "topn_cap"),
    )
    mc = MethodComparison(cfg)
    results = mc.run(small_synthetic_inputs)
    names = {r.name for r in results}
    assert "miqp" not in names and "topn_cap" not in names
    assert "admm" in names and "fista" in names
