"""Tests for sit.solvers.lasso (sklearn baseline)."""

from __future__ import annotations

import numpy as np
import pytest

from sit.solvers.admm import SparseTrackerADMM
from sit.solvers.lasso import SklearnLassoSolver


def test_lambda_to_alpha_conversion():
    """The static converter must match the documented α = λ / n formula."""
    assert SklearnLassoSolver.lambda_to_alpha(10.0, 100) == 0.1
    assert SklearnLassoSolver.lambda_to_alpha(0.0, 50) == 0.0
    assert SklearnLassoSolver.lambda_to_alpha(5.0, 1) == 5.0


def test_lasso_basic_fit(synthetic_sparse):
    s = SklearnLassoSolver(lam=0.5, max_iter=10_000, tol=1e-7)
    s.fit(synthetic_sparse.X, synthetic_sparse.y)
    assert s.w is not None
    assert s.w.shape == (synthetic_sparse.p,)
    assert (s.w >= -1e-12).all(), "sklearn Lasso(positive=True) should give non-negative weights"


def test_lasso_get_sparse_weights_simplex(synthetic_sparse):
    s = SklearnLassoSolver(lam=0.5, max_iter=10_000)
    s.fit(synthetic_sparse.X, synthetic_sparse.y)
    w = s.get_sparse_weights()
    assert (w >= 0).all()
    assert abs(w.sum() - 1.0) < 1e-10


def test_lasso_matches_admm(synthetic_sparse):
    """LASSO and ADMM solve the *same* convex program → minimisers must agree."""
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    lam = 0.05 * lam_max

    admm = SparseTrackerADMM(lam=lam, max_iter=8000, tol=1e-9, verbose=False)
    admm.fit(synthetic_sparse.X, synthetic_sparse.y)

    lasso = SklearnLassoSolver(lam=lam, max_iter=20_000, tol=1e-9)
    lasso.fit(synthetic_sparse.X, synthetic_sparse.y)

    diff = float(np.max(np.abs(admm.z - lasso.w)))
    # Both target the same minimiser; agreement on (n=80, p=200) is typically <1e-3
    assert diff < 5e-3, f"LASSO and ADMM disagree by ‖Δw‖∞ = {diff:.3e}"


def test_lasso_at_lambda_max_zero_solution(synthetic_sparse):
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    s = SklearnLassoSolver(lam=2.0 * lam_max, max_iter=10_000)
    s.fit(synthetic_sparse.X, synthetic_sparse.y)
    assert np.allclose(s.w, 0.0, atol=1e-8)


def test_lasso_recovers_support(synthetic_sparse):
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    lasso = SklearnLassoSolver(lam=0.01 * lam_max, max_iter=20_000, tol=1e-8)
    lasso.fit(synthetic_sparse.X, synthetic_sparse.y)
    recovered = set(np.where(lasso.w > 1e-5)[0].tolist())
    missing = set(synthetic_sparse.support.tolist()) - recovered
    assert not missing, f"LASSO missed {sorted(missing)} from true support"


def test_lasso_negative_lambda_rejected(synthetic_sparse):
    s = SklearnLassoSolver(lam=-0.1)
    with pytest.raises(ValueError):
        s.fit(synthetic_sparse.X, synthetic_sparse.y)
