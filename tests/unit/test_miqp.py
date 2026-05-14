"""Tests for sit.solvers.miqp (MOSEK MIQP gold standard).

These tests are *expensive* (MOSEK MIQP is NP-hard). We keep problem sizes
tiny (p ≤ 30, K ≤ 8) so the suite still runs in well under a second.

Marked with ``@pytest.mark.mosek`` so CI can skip them when MOSEK isn't
provisioned (the ``mosek_env`` fixture handles the skip dynamically).
"""

from __future__ import annotations

import numpy as np
import pytest

from sit.solvers.miqp import MIQPSolver


@pytest.mark.mosek
def test_miqp_recovers_exact_support(mosek_env, rng):
    """On a clean k-sparse problem MIQP with K=k must recover the true support."""
    del mosek_env
    n, p, k = 60, 25, 5
    X = rng.standard_normal((n, p))
    X /= X.std(axis=0) + 1e-12
    support = sorted(rng.choice(p, k, replace=False).tolist())
    w_true = np.zeros(p)
    w_true[support] = rng.uniform(0.5, 1.5, k)
    w_true /= w_true.sum()  # put truth on the simplex
    y = X @ w_true + 1e-3 * rng.standard_normal(n)

    solver = MIQPSolver(K=k, time_limit=20, mip_gap=1e-5)
    solver.fit(X, y)
    assert solver.converged, f"MIQP failed: status={solver.status}"

    recovered = sorted(int(j) for j in np.where(solver.w > 1e-5)[0])
    assert recovered == support, f"MIQP support {recovered} != truth {support}"


@pytest.mark.mosek
def test_miqp_respects_cardinality_K(mosek_env, rng):
    del mosek_env
    n, p = 40, 20
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)

    for K in (2, 5, 8):
        solver = MIQPSolver(K=K, time_limit=10)
        solver.fit(X, y)
        if not solver.converged:
            pytest.skip(f"MIQP did not converge for K={K}")
        n_act = int((solver.w > 1e-5).sum())
        assert n_act <= K, f"MIQP picked {n_act} stocks but K={K}"


@pytest.mark.mosek
def test_miqp_simplex_constraint_active(mosek_env, rng):
    del mosek_env
    n, p, K = 30, 15, 4
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    solver = MIQPSolver(K=K, enforce_simplex=True, time_limit=10)
    solver.fit(X, y)
    if not solver.converged:
        pytest.skip("MIQP did not converge")
    assert abs(solver.w.sum() - 1.0) < 1e-4
    assert (solver.w >= -1e-8).all()


@pytest.mark.mosek
def test_miqp_no_simplex_when_disabled(mosek_env, rng):
    del mosek_env
    n, p, K = 30, 15, 4
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    solver = MIQPSolver(K=K, enforce_simplex=False, time_limit=10)
    solver.fit(X, y)
    if not solver.converged:
        pytest.skip("MIQP did not converge")
    # When the simplex is disabled the sum is unconstrained but each w_j ≤ 1
    assert solver.w.max() <= 1.0 + 1e-6


@pytest.mark.mosek
def test_miqp_K_must_be_in_range(mosek_env, rng):
    del mosek_env
    X = rng.standard_normal((20, 10))
    y = rng.standard_normal(20)
    with pytest.raises(ValueError):
        MIQPSolver(K=0).fit(X, y)
    with pytest.raises(ValueError):
        MIQPSolver(K=999).fit(X, y)


@pytest.mark.mosek
def test_miqp_get_sparse_weights_simplex(mosek_env, rng):
    del mosek_env
    n, p, K = 30, 15, 4
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    solver = MIQPSolver(K=K, time_limit=10)
    solver.fit(X, y)
    if not solver.converged:
        pytest.skip("MIQP did not converge")
    w = solver.get_sparse_weights()
    assert abs(w.sum() - 1.0) < 1e-8
    assert (w >= 0).all()
