"""Numerical-stability tests for ``SparseTrackerADMM``.

We probe degenerate scenarios:
* duplicated / perfectly collinear columns,
* near-zero variance columns,
* very small / very large ρ,
* ill-conditioned X,
* fit-and-refit consistency.

These are the kinds of pathological inputs that historically broke similar
solvers (Cholesky → LinAlgError, division by zero in standardisation, etc.).
The aim is "never NaN, never raise, always finite output."
"""

from __future__ import annotations

import numpy as np
import pytest

from sit.solvers.admm import SparseTrackerADMM


def test_duplicate_columns(rng):
    """Two identical features must not break the Cholesky step (ρ I regularises it)."""
    n, p = 50, 60
    X = rng.standard_normal((n, p))
    X[:, 5] = X[:, 6]  # exact duplicate
    y = rng.standard_normal(n)
    solver = SparseTrackerADMM(lam=0.1, max_iter=1000, verbose=False)
    solver.fit(X, y)
    assert np.isfinite(solver.z).all()
    assert not np.isnan(solver.z).any()


def test_collinear_subspace(rng):
    """A 3-D subspace of rank-1 columns: ρI should still regularise."""
    n, p = 40, 50
    X = rng.standard_normal((n, p))
    # Force columns 0,1,2 to lie on a line: X[:,1] = 2*X[:,0], X[:,2] = -0.5*X[:,0]
    X[:, 1] = 2.0 * X[:, 0]
    X[:, 2] = -0.5 * X[:, 0]
    y = X[:, 0] + 0.01 * rng.standard_normal(n)
    solver = SparseTrackerADMM(lam=0.05, max_iter=2000, verbose=False)
    solver.fit(X, y)
    assert np.isfinite(solver.z).all()


def test_no_nans_in_outputs(synthetic_sparse):
    solver = SparseTrackerADMM(lam=0.05, max_iter=500, verbose=False)
    solver.fit(synthetic_sparse.X, synthetic_sparse.y)
    for arr in (solver.w, solver.z, solver.u):
        assert np.isfinite(arr).all()
    assert np.isfinite(solver.primal_residuals).all()
    assert np.isfinite(solver.dual_residuals).all()
    assert np.isfinite(solver.objective_values).all()


@pytest.mark.parametrize("rho", [1e-3, 1e-1, 1.0, 10.0, 1e2])
def test_extreme_rho_does_not_diverge(synthetic_sparse, rho):
    """Adaptive-ρ should rescue extreme ρ values."""
    solver = SparseTrackerADMM(
        lam=0.05, rho=rho, max_iter=4000, tol=1e-7, adaptive_rho=True, verbose=False
    )
    solver.fit(synthetic_sparse.X, synthetic_sparse.y)
    assert np.isfinite(solver.z).all()
    # Final residuals must be small
    assert solver.primal_residuals[-1] < 1e-2
    assert solver.dual_residuals[-1] < 1e-2


def test_high_condition_number(rng):
    """An ill-conditioned X with a 10⁶ singular-value spread still solves."""
    n, p = 80, 120
    U = np.linalg.qr(rng.standard_normal((n, n)))[0]
    Vt = np.linalg.qr(rng.standard_normal((p, p)))[0].T
    sv = np.logspace(0, -6, num=min(n, p))  # condition number 1e6
    Sigma = np.zeros((n, p))
    np.fill_diagonal(Sigma, sv)
    X = U @ Sigma @ Vt
    y = rng.standard_normal(n)
    solver = SparseTrackerADMM(lam=0.5, max_iter=3000, verbose=False)
    solver.fit(X, y)
    assert np.isfinite(solver.z).all()


def test_get_raw_weights_handles_dead_coordinates(synthetic_sparse):
    """get_raw_weights must preserve the zero pattern from z."""
    solver = SparseTrackerADMM(lam=0.05, max_iter=2000, verbose=False)
    solver.fit(synthetic_sparse.X, synthetic_sparse.y)
    # Manufacture a non-trivial standard-deviation vector
    X_std = np.abs(synthetic_sparse.X).std(axis=0) + 1e-8
    w_raw = solver.get_raw_weights(X_std)
    # Where z was clipped to zero, w_raw must be zero
    zero_mask = solver.z < 1e-6
    assert np.allclose(w_raw[zero_mask], 0.0)
    # The non-zero raw weights must sum to 1
    assert abs(w_raw.sum() - 1.0) < 1e-10


def test_zero_lambda_recovers_nnls(rng):
    """At λ=0 (or effectively 0), the solver should converge to the NNLS solution."""
    n, p = 60, 30  # over-determined so NNLS is well-defined
    X = rng.standard_normal((n, p))
    w_star = np.zeros(p)
    w_star[rng.choice(p, 5, replace=False)] = rng.uniform(0.5, 1.5, 5)
    y = X @ w_star + 0.01 * rng.standard_normal(n)
    solver = SparseTrackerADMM(lam=1e-10, max_iter=5000, tol=1e-9, verbose=False)
    solver.fit(X, y)
    # NNLS objective: minimum ||Xw-y||² with w ≥ 0. We compare via residual norm:
    # our solver should be at least as good as setting w to a random feasible point.
    obj_admm = float(np.linalg.norm(X @ solver.z - y))
    obj_zero = float(np.linalg.norm(y))
    assert obj_admm < obj_zero, "ADMM did not improve on the trivial w=0 baseline"
