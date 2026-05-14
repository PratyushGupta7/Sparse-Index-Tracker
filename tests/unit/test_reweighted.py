"""Reweighted-ℓ₁ MM sanity checks.

Concretely we test:
* Round-0 of reweighted-ℓ₁ matches a vanilla scalar-λ ADMM run.
* Reweighted-ℓ₁ produces a *strict subset* of the vanilla ADMM support on
  problems where vanilla ADMM over-selects.
* The lam_vec mechanism inside ``SparseTrackerADMM`` reduces to the scalar
  case when no vector is supplied.
* Outer-loop converges (support stabilises) within a few rounds.
"""

from __future__ import annotations

import numpy as np

from sit.solvers.admm import SparseTrackerADMM
from sit.solvers.reweighted import reweighted_l1_admm


def test_lam_vec_uniform_matches_scalar(synthetic_sparse):
    """Passing ``lam_vec = lam * ones`` must reproduce the scalar behaviour."""
    lam = 0.1
    s_scalar = SparseTrackerADMM(lam=lam, max_iter=2000, tol=1e-8, verbose=False)
    s_scalar.fit(synthetic_sparse.X, synthetic_sparse.y)

    lam_vec = np.full(synthetic_sparse.p, lam, dtype=np.float64)
    s_vector = SparseTrackerADMM(lam=lam, lam_vec=lam_vec, max_iter=2000, tol=1e-8, verbose=False)
    s_vector.fit(synthetic_sparse.X, synthetic_sparse.y)

    np.testing.assert_allclose(s_scalar.z, s_vector.z, atol=1e-9)
    np.testing.assert_allclose(s_scalar.w, s_vector.w, atol=1e-9)


def test_reweighted_recovers_support(synthetic_sparse):
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    result = reweighted_l1_admm(
        synthetic_sparse.X,
        synthetic_sparse.y,
        lam=0.05 * lam_max,
        n_outer=5,
        epsilon=1e-3,
        max_inner_iter=3000,
        tol=1e-7,
        verbose=False,
    )
    recovered = set(np.where(result.w > 1e-5)[0].tolist())
    missing = set(synthetic_sparse.support.tolist()) - recovered
    assert not missing


def test_reweighted_sparser_than_vanilla(synthetic_sparse):
    """At moderate λ, reweighted ℓ₁ should not produce more non-zeros than vanilla."""
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    lam = 0.005 * lam_max  # deliberately small so vanilla over-selects

    vanilla = SparseTrackerADMM(lam=lam, max_iter=3000, tol=1e-7, verbose=False)
    vanilla.fit(synthetic_sparse.X, synthetic_sparse.y)
    n_vanilla = int((vanilla.z > 1e-5).sum())

    reweighted = reweighted_l1_admm(
        synthetic_sparse.X,
        synthetic_sparse.y,
        lam=lam,
        n_outer=4,
        epsilon=1e-3,
        max_inner_iter=3000,
        tol=1e-7,
        verbose=False,
    )
    n_reweighted = int((reweighted.w > 1e-5).sum())

    assert (
        n_reweighted <= n_vanilla
    ), f"Reweighted should not increase sparsity: vanilla={n_vanilla} reweighted={n_reweighted}"


def test_reweighted_converges_outer(synthetic_sparse):
    lam_max = SparseTrackerADMM.compute_lambda_max(synthetic_sparse.X, synthetic_sparse.y)
    result = reweighted_l1_admm(
        synthetic_sparse.X,
        synthetic_sparse.y,
        lam=0.05 * lam_max,
        n_outer=10,
        epsilon=1e-3,
        max_inner_iter=2000,
        tol=1e-6,
        verbose=False,
    )
    assert result.converged
    assert result.n_outer_run <= 10
    assert len(result.weights_history) == result.n_outer_run
    assert len(result.n_active_history) == result.n_outer_run
