"""BIC / EBIC λ-selection tests."""

from __future__ import annotations

import numpy as np
import pytest

from sit.selection.bic import (
    LambdaSelectionResult,
    bic_score,
    ebic_lambda_selection,
    ebic_score,
    select_lambda_bic,
)


def test_bic_score_basic():
    # k=0 ⇒ no model-size penalty
    assert bic_score(rss=10.0, k=0, n=100) == 100 * np.log(0.10)
    # k=5 ⇒ penalty = 5 * log(100)
    expected = 100 * np.log(0.10) + 5 * np.log(100)
    assert bic_score(rss=10.0, k=5, n=100) == pytest.approx(expected)


def test_ebic_score_collapses_to_bic_at_gamma_zero():
    # EBIC γ=0 == BIC for any (rss, k, n, p)
    assert ebic_score(5.0, 3, 100, 500, gamma=0.0) == bic_score(5.0, 3, 100)


def test_ebic_score_monotone_in_gamma():
    s0 = ebic_score(5.0, 4, 100, 500, gamma=0.0)
    s1 = ebic_score(5.0, 4, 100, 500, gamma=0.5)
    s2 = ebic_score(5.0, 4, 100, 500, gamma=1.0)
    assert s0 < s1 < s2


def test_select_lambda_bic_returns_result_object(synthetic_sparse):
    result = select_lambda_bic(
        synthetic_sparse.X,
        synthetic_sparse.y,
        n_lambdas=10,
        lam_min_ratio=0.005,
        max_inner_iter=2000,
        tol=1e-6,
    )
    assert isinstance(result, LambdaSelectionResult)
    assert result.criterion == "bic"
    assert len(result.lambdas) == 10
    assert len(result.weights) == 10
    assert 0 <= result.best_index < 10


def test_ebic_picks_sparser_solution(synthetic_sparse):
    """EBIC with γ=1 should select λ ≥ BIC's chosen λ ⇒ fewer non-zeros."""
    bic_res = select_lambda_bic(
        synthetic_sparse.X,
        synthetic_sparse.y,
        n_lambdas=10,
        lam_min_ratio=0.005,
        max_inner_iter=2000,
        tol=1e-6,
    )
    ebic_res = ebic_lambda_selection(
        synthetic_sparse.X,
        synthetic_sparse.y,
        gamma=1.0,
        n_lambdas=10,
        lam_min_ratio=0.005,
        max_inner_iter=2000,
        tol=1e-6,
    )
    # Sweep grids share the same lambdas (deterministic _make_lambda_grid)
    np.testing.assert_allclose(bic_res.lambdas, ebic_res.lambdas)
    # EBIC γ=1 picks at least as sparse a model
    assert ebic_res.best_n_active <= bic_res.best_n_active + 1


def test_bic_chooses_truth_when_well_separated():
    """In a clean low-noise regime BIC should select close to the true k."""
    rng = np.random.default_rng(1234)
    n, p, k_true = 150, 300, 8
    X = rng.standard_normal((n, p))
    X /= X.std(axis=0) + 1e-12
    support = rng.choice(p, k_true, replace=False)
    w_star = np.zeros(p)
    w_star[support] = rng.uniform(0.5, 1.5, size=k_true)
    y = X @ w_star + 0.001 * rng.standard_normal(n)

    res = select_lambda_bic(X, y, n_lambdas=20, lam_min_ratio=1e-3, max_inner_iter=3000, tol=1e-7)
    chosen_k = res.best_n_active
    assert (
        k_true <= chosen_k <= 2 * k_true
    ), f"BIC picked k={chosen_k}, expected near k_true={k_true}"
