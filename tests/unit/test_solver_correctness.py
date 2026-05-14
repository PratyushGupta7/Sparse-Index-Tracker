"""Correctness tests for ``SparseTrackerADMM``.

Each test starts from a ground-truth k-sparse problem and checks that the
solver recovers the correct *support* and reasonably accurate *values*. We
deliberately avoid round-trip-with-itself tests (which prove nothing) — every
assertion compares against external truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from sit.solvers.admm import SparseTrackerADMM


@pytest.mark.parametrize(
    "n,p,k",
    [
        (60, 150, 5),
        (80, 200, 8),
        (100, 300, 12),
        (120, 400, 15),
    ],
)
def test_support_recovery_clean(make_problem, n, p, k):
    """In the noiseless+positive regime the solver recovers the true support."""
    prob = make_problem(n=n, p=p, k=k, noise=0.0, simplex=False)
    lam_max = SparseTrackerADMM.compute_lambda_max(prob.X, prob.y)
    # Small λ → small bias → exact support recovery
    solver = SparseTrackerADMM(lam=1e-3 * lam_max, max_iter=4000, tol=1e-8, verbose=False)
    solver.fit(prob.X, prob.y)
    recovered = np.where(solver.z > 1e-5)[0]
    # We allow recovered to be a *superset* but it must contain the true support
    missing = set(prob.support.tolist()) - set(recovered.tolist())
    assert not missing, (
        f"Support recovery failed: missing indices {sorted(missing)} "
        f"out of true support {prob.support.tolist()}"
    )


def test_support_recovery_with_noise(make_problem):
    """With small Gaussian noise the support is still recovered."""
    prob = make_problem(n=120, p=300, k=10, noise=0.02, simplex=False)
    lam_max = SparseTrackerADMM.compute_lambda_max(prob.X, prob.y)
    solver = SparseTrackerADMM(lam=0.01 * lam_max, max_iter=4000, tol=1e-7, verbose=False)
    solver.fit(prob.X, prob.y)
    recovered = set(np.where(solver.z > 1e-4)[0].tolist())
    true_support = set(prob.support.tolist())
    # ≥ 90% of true support recovered, false-positive rate ≤ 5%
    tp = len(true_support & recovered)
    fp = len(recovered - true_support)
    assert tp / max(len(true_support), 1) >= 0.9, f"Only {tp}/{len(true_support)} recovered"
    assert fp / max(prob.p, 1) <= 0.05, f"Too many false positives: {fp}/{prob.p}"


def test_recovered_values_close_to_truth(make_problem):
    """For a noiseless simplex problem, recovered weights match truth on support."""
    prob = make_problem(n=120, p=300, k=8, noise=0.0, simplex=True)
    lam_max = SparseTrackerADMM.compute_lambda_max(prob.X, prob.y)
    solver = SparseTrackerADMM(lam=1e-4 * lam_max, max_iter=8000, tol=1e-10, verbose=False)
    solver.fit(prob.X, prob.y)
    w_hat = solver.get_sparse_weights(threshold=1e-6)
    # Compare on the true support — both vectors sum to 1, so a direct compare works
    diff = np.abs(w_hat - prob.w_true)
    # Allow loose tolerance because the bias is not zero with positive λ
    assert diff.max() < 0.05, f"Largest deviation {diff.max():.4f} > 5 % weight"


def test_lambda_max_zeros_solution(synthetic_sparse):
    """At λ ≥ λ_max the unique solution is z = 0."""
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    solver = SparseTrackerADMM(lam=2.0 * lam_max, max_iter=500, verbose=False)
    solver.fit(synthetic_sparse.X, synthetic_sparse.y)
    assert np.allclose(solver.z, 0.0, atol=1e-8), "z should be exactly zero at λ ≥ λ_max"


def test_lambda_just_below_max_yields_tiny_support(synthetic_sparse):
    """At λ comfortably below λ_max, the support is non-empty but tiny."""
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    solver = SparseTrackerADMM(lam=0.5 * lam_max, max_iter=2000, tol=1e-8, verbose=False)
    solver.fit(synthetic_sparse.X, synthetic_sparse.y)
    n_active = int((solver.z > 1e-6).sum())
    # 0 < n_active <= a few — the regularisation should kill almost all coords
    assert (
        0 < n_active <= 10
    ), f"Just below λ_max we expect a small support, got {n_active} non-zero entries"


def test_objective_decreases(synthetic_sparse):
    """The recorded objective trajectory should end below where it started."""
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    solver = SparseTrackerADMM(lam=0.05 * lam_max, max_iter=2000, verbose=False)
    solver.fit(synthetic_sparse.X, synthetic_sparse.y)
    obj = np.asarray(solver.objective_values)
    assert obj[-1] < obj[0], "Objective did not decrease over the iterations"


def test_normalized_weights_sum_to_one(synthetic_sparse):
    """``get_sparse_weights`` always returns a vector on the probability simplex."""
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    solver = SparseTrackerADMM(lam=0.05 * lam_max, max_iter=2000, verbose=False)
    solver.fit(synthetic_sparse.X, synthetic_sparse.y)
    w = solver.get_sparse_weights()
    assert np.isfinite(w).all()
    assert (w >= 0).all()
    assert abs(w.sum() - 1.0) < 1e-10
