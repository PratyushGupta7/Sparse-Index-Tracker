"""Tests for sit.solvers.omp (non-negative OMP)."""

from __future__ import annotations

import numpy as np
import pytest

from sit.solvers.omp import OMPSolver


def test_omp_recovers_true_support(synthetic_sparse):
    """OMP with K = true sparsity should recover the true support exactly."""
    omp = OMPSolver(K=synthetic_sparse.k)
    omp.fit(synthetic_sparse.X, synthetic_sparse.y)
    recovered = set(np.where(omp.w > 1e-6)[0].tolist())
    missing = set(synthetic_sparse.support.tolist()) - recovered
    assert not missing, f"OMP missed {sorted(missing)}"
    assert len(recovered) <= synthetic_sparse.k


def test_omp_respects_cardinality_cap(synthetic_sparse):
    omp = OMPSolver(K=5)
    omp.fit(synthetic_sparse.X, synthetic_sparse.y)
    assert len(omp.selected) <= 5
    assert int((omp.w > 1e-6).sum()) <= 5


def test_omp_weights_nonneg(synthetic_sparse):
    omp = OMPSolver(K=10)
    omp.fit(synthetic_sparse.X, synthetic_sparse.y)
    assert (omp.w >= -1e-12).all(), "Non-negative OMP must yield w_j ≥ 0"


def test_omp_simplex_normalisation(synthetic_sparse):
    omp = OMPSolver(K=10)
    omp.fit(synthetic_sparse.X, synthetic_sparse.y)
    w = omp.get_sparse_weights()
    assert (w >= 0).all()
    assert abs(w.sum() - 1.0) < 1e-10


def test_omp_residual_monotone_decreasing(synthetic_sparse):
    """The residual norm should decrease with each successful selection step."""
    omp = OMPSolver(K=15)
    omp.fit(synthetic_sparse.X, synthetic_sparse.y)
    residuals = np.asarray(omp.residual_history)
    diffs = np.diff(residuals)
    # Allow tiny rounding noise (≤1e-10) but no real increases
    assert (diffs <= 1e-9).all(), f"OMP residual not monotone: max increase {diffs.max():.3e}"


def test_omp_early_stop_on_zero_residual(rng):
    """If the linear system is exactly solvable in K=k* steps, OMP stops early."""
    n, p, k = 50, 30, 4
    X = rng.standard_normal((n, p))
    X /= np.linalg.norm(X, axis=0) + 1e-12
    support = rng.choice(p, k, replace=False)
    w_star = np.zeros(p)
    w_star[support] = rng.uniform(0.5, 1.5, size=k)
    y = X @ w_star  # noiseless
    omp = OMPSolver(K=20, tol=1e-8)
    omp.fit(X, y)
    # Should have stopped well before 20 iterations once the residual fell below tol
    assert omp.n_iter <= k + 2
    if omp.converged:
        assert omp.residual_history[-1] < 1e-7


@pytest.mark.parametrize("K", [3, 5, 10, 15, 20])
def test_omp_cardinality_grid(synthetic_sparse, K):
    """Across a K grid, achieved cardinality is always ≤ K and weights are valid."""
    omp = OMPSolver(K=K)
    omp.fit(synthetic_sparse.X, synthetic_sparse.y)
    n_act = int((omp.w > 1e-6).sum())
    assert n_act <= K
    w = omp.get_sparse_weights()
    assert abs(w.sum() - 1.0) < 1e-10


def test_omp_K_too_small_rejected(synthetic_sparse):
    with pytest.raises(ValueError):
        OMPSolver(K=0).fit(synthetic_sparse.X, synthetic_sparse.y)


def test_omp_K_clipped_to_p(synthetic_sparse):
    """K > p should not crash; OMP should stop after picking all p columns."""
    omp = OMPSolver(K=10_000)
    omp.fit(synthetic_sparse.X, synthetic_sparse.y)
    assert len(omp.selected) <= synthetic_sparse.p
