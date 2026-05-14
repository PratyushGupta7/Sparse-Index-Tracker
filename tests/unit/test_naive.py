"""Tests for sit.solvers.naive (top-N market-cap, equal-weight, random)."""

from __future__ import annotations

import numpy as np
import pytest

from sit.solvers.naive import (
    EqualWeightTopNSolver,
    RandomEqualWeightSolver,
    TopNMarketCapSolver,
    equal_weight_top_n_market_cap,
    random_equal_weight,
    random_equal_weight_ensemble,
    top_n_market_cap_weights,
)

# ---------------------------------------------------------------------------
# top_n_market_cap_weights
# ---------------------------------------------------------------------------


def test_top_n_market_cap_picks_largest():
    caps = np.array([10.0, 50.0, 30.0, 5.0, 80.0])
    w = top_n_market_cap_weights(caps, N=3)
    # Top 3: 80 (idx 4), 50 (idx 1), 30 (idx 2) → total 160
    expected = np.zeros(5)
    expected[1] = 50 / 160
    expected[2] = 30 / 160
    expected[4] = 80 / 160
    np.testing.assert_allclose(w, expected)
    assert abs(w.sum() - 1.0) < 1e-12


def test_top_n_market_cap_handles_nan():
    caps = np.array([10.0, np.nan, 30.0, np.nan, 80.0])
    w = top_n_market_cap_weights(caps, N=3)
    # Only 3 non-NaN: 10, 30, 80 → total 120
    expected = np.zeros(5)
    expected[0] = 10 / 120
    expected[2] = 30 / 120
    expected[4] = 80 / 120
    np.testing.assert_allclose(w, expected)


def test_top_n_market_cap_clips_to_eligible():
    """Asking for more stocks than have valid caps should silently clip."""
    caps = np.array([10.0, np.nan, 30.0, np.nan, 80.0])
    w = top_n_market_cap_weights(caps, N=10)  # only 3 eligible
    assert int((w > 0).sum()) == 3


def test_top_n_market_cap_rejects_all_nan():
    with pytest.raises(ValueError):
        top_n_market_cap_weights(np.full(5, np.nan), N=3)


def test_top_n_market_cap_empty_array_rejected():
    with pytest.raises(ValueError):
        top_n_market_cap_weights(np.array([]), N=3)


# ---------------------------------------------------------------------------
# equal_weight_top_n_market_cap
# ---------------------------------------------------------------------------


def test_equal_weight_top_n_uniform():
    caps = np.array([10.0, 50.0, 30.0, 5.0, 80.0])
    w = equal_weight_top_n_market_cap(caps, N=3)
    # Top 3 stocks, each getting 1/3
    expected = np.zeros(5)
    expected[1] = expected[2] = expected[4] = 1 / 3
    np.testing.assert_allclose(w, expected)


def test_equal_weight_top_n_simplex():
    caps = np.array([10.0, 50.0, 30.0, 5.0, 80.0])
    for N in (1, 2, 3, 4, 5):
        w = equal_weight_top_n_market_cap(caps, N=N)
        assert abs(w.sum() - 1.0) < 1e-12
        assert int((w > 0).sum()) == N


# ---------------------------------------------------------------------------
# random_equal_weight
# ---------------------------------------------------------------------------


def test_random_equal_weight_basic():
    p, N = 100, 10
    w = random_equal_weight(p, N, seed=0)
    assert w.shape == (p,)
    assert int((w > 0).sum()) == N
    assert abs(w.sum() - 1.0) < 1e-12
    # Each non-zero weight should be exactly 1/N
    np.testing.assert_allclose(w[w > 0], 1.0 / N)


def test_random_equal_weight_reproducible():
    w1 = random_equal_weight(50, 5, seed=42)
    w2 = random_equal_weight(50, 5, seed=42)
    np.testing.assert_array_equal(w1, w2)


def test_random_equal_weight_different_seeds_differ():
    w1 = random_equal_weight(50, 5, seed=1)
    w2 = random_equal_weight(50, 5, seed=2)
    assert not np.array_equal(w1, w2)


def test_random_equal_weight_rejects_bad_N():
    with pytest.raises(ValueError):
        random_equal_weight(10, 0, seed=0)
    with pytest.raises(ValueError):
        random_equal_weight(10, 11, seed=0)


def test_random_equal_weight_ensemble_sums_to_one():
    w = random_equal_weight_ensemble(50, 5, n_seeds=20, base_seed=0)
    assert abs(w.sum() - 1.0) < 1e-10
    assert (w >= 0).all()


def test_random_equal_weight_ensemble_smoothes_to_uniform():
    """As n_seeds → ∞, the ensemble mean converges to the uniform 1/p."""
    p, N = 50, 10
    w_few = random_equal_weight_ensemble(p, N, n_seeds=5)
    w_many = random_equal_weight_ensemble(p, N, n_seeds=2000)
    spread_few = w_few.std()
    spread_many = w_many.std()
    # Many seeds → smaller std (closer to uniform)
    assert spread_many < spread_few


# ---------------------------------------------------------------------------
# Class wrappers — Protocol compliance + fit/get_sparse_weights
# ---------------------------------------------------------------------------


def test_top_n_solver_class(rng):
    p, N = 20, 5
    caps = np.abs(rng.standard_normal(p)) + 1
    s = TopNMarketCapSolver(N=N, market_caps=caps)
    s.fit(np.zeros((10, p)), np.zeros(10))
    w = s.get_sparse_weights()
    assert int((w > 0).sum()) == N
    assert abs(w.sum() - 1.0) < 1e-12


def test_equal_weight_top_n_solver_class(rng):
    p, N = 20, 5
    caps = np.abs(rng.standard_normal(p)) + 1
    s = EqualWeightTopNSolver(N=N, market_caps=caps)
    s.fit(np.zeros((10, p)), np.zeros(10))
    w = s.get_sparse_weights()
    assert int((w > 0).sum()) == N
    np.testing.assert_allclose(w[w > 0], 1.0 / N)


def test_random_solver_single_draw():
    s = RandomEqualWeightSolver(N=5, p=20, seed=0, ensemble=False)
    s.fit(np.zeros((10, 20)), np.zeros(10))
    w = s.get_sparse_weights()
    assert int((w > 0).sum()) == 5
    assert abs(w.sum() - 1.0) < 1e-10


def test_random_solver_ensemble():
    s = RandomEqualWeightSolver(N=5, p=20, seed=0, ensemble=True, n_seeds=50)
    s.fit(np.zeros((10, 20)), np.zeros(10))
    w = s.get_sparse_weights()
    assert abs(w.sum() - 1.0) < 1e-10
    # Ensemble produces non-zero weight on most coordinates
    assert int((w > 0).sum()) > 5


def test_naive_solvers_get_sparse_weights_before_fit_raises():
    s = TopNMarketCapSolver(N=5, market_caps=np.ones(10))
    with pytest.raises(RuntimeError):
        s.get_sparse_weights()
