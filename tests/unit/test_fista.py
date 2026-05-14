"""FISTA correctness + ADMM equivalence."""

from __future__ import annotations

import numpy as np
import pytest

from sit.solvers.admm import SparseTrackerADMM
from sit.solvers.fista import FISTA, fista_l1_nonneg


def test_fista_recovers_support(synthetic_sparse):
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    result = fista_l1_nonneg(
        synthetic_sparse.X,
        synthetic_sparse.y,
        lam=0.01 * lam_max,
        max_iter=3000,
        tol=1e-8,
    )
    w = result["w"]
    recovered = set(np.where(w > 1e-5)[0].tolist())
    missing = set(synthetic_sparse.support.tolist()) - recovered
    assert not missing, f"FISTA missed {sorted(missing)} from true support"


def test_fista_solution_matches_admm(synthetic_sparse):
    """FISTA and ADMM solve the same convex program — minimizers must agree."""
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    lam = 0.02 * lam_max

    admm = SparseTrackerADMM(lam=lam, max_iter=8000, tol=1e-9, verbose=False)
    admm.fit(synthetic_sparse.X, synthetic_sparse.y)
    fista = fista_l1_nonneg(
        synthetic_sparse.X, synthetic_sparse.y, lam=lam, max_iter=8000, tol=1e-9
    )

    w_admm = admm.z
    w_fista = np.asarray(fista["w"])
    diff = float(np.linalg.norm(w_admm - w_fista))
    # Both are non-negative ℓ₁-regularised solutions of the same strictly
    # convex (after positivity) program, so the agreement should be tight.
    assert diff < 1e-3, f"FISTA and ADMM disagree by {diff:.3e}"


def test_fista_nonneg_constraint_active(synthetic_sparse):
    """Output is element-wise non-negative."""
    result = fista_l1_nonneg(synthetic_sparse.X, synthetic_sparse.y, lam=0.1, max_iter=1000)
    assert (result["w"] >= 0).all()


def test_fista_at_lambda_max_returns_zero(synthetic_sparse):
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    result = fista_l1_nonneg(
        synthetic_sparse.X, synthetic_sparse.y, lam=2.0 * lam_max, max_iter=500
    )
    assert np.allclose(result["w"], 0.0, atol=1e-8)


def test_fista_class_api(synthetic_sparse):
    """Class wrapper exposes the same surface as ``SparseTrackerADMM``."""
    f = FISTA(lam=0.05, max_iter=3000, tol=1e-8, track_history=True)
    f.fit(synthetic_sparse.X, synthetic_sparse.y)
    assert f.w is not None
    assert f.w.shape == (synthetic_sparse.p,)
    w = f.get_sparse_weights()
    assert abs(w.sum() - 1.0) < 1e-10
    assert (w >= 0).all()
    assert len(f.objective_values) == f.n_iter


@pytest.mark.parametrize("lam_factor", [0.001, 0.01, 0.1, 0.5])
def test_fista_sparsity_monotone(synthetic_sparse, lam_factor):
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    lo = fista_l1_nonneg(
        synthetic_sparse.X, synthetic_sparse.y, lam=lam_factor * lam_max, max_iter=2000
    )
    hi = fista_l1_nonneg(
        synthetic_sparse.X, synthetic_sparse.y, lam=(lam_factor * 4) * lam_max, max_iter=2000
    )
    n_lo = int((np.asarray(lo["w"]) > 1e-6).sum())
    n_hi = int((np.asarray(hi["w"]) > 1e-6).sum())
    assert n_hi <= n_lo + 2
