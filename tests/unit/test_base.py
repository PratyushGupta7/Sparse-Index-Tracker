"""Tests for sit.solvers.base utilities."""

from __future__ import annotations

import numpy as np
import pytest

from sit.solvers.base import (
    SimplexSolver,
    effective_n,
    herfindahl,
    n_active,
    raw_to_simplex_via_std,
    to_simplex,
)

# ---------------------------------------------------------------------------
# to_simplex
# ---------------------------------------------------------------------------


def test_to_simplex_basic():
    w = np.array([0.5, 0.3, 0.2])
    out = to_simplex(w)
    np.testing.assert_allclose(out, w)
    assert abs(out.sum() - 1.0) < 1e-12


def test_to_simplex_renormalises():
    w = np.array([2.0, 4.0, 4.0])
    out = to_simplex(w)
    np.testing.assert_allclose(out, [0.2, 0.4, 0.4])


def test_to_simplex_zeroes_negatives():
    w = np.array([0.5, -0.3, 0.5])
    out = to_simplex(w)
    np.testing.assert_allclose(out, [0.5, 0.0, 0.5])


def test_to_simplex_threshold_clip():
    w = np.array([0.001, 0.5, 0.5])
    out = to_simplex(w, threshold=1e-2)
    np.testing.assert_allclose(out, [0.0, 0.5, 0.5])


def test_to_simplex_raises_on_empty_default():
    w = np.zeros(5)
    with pytest.raises(ValueError):
        to_simplex(w)


def test_to_simplex_falls_back_to_uniform_when_allowed():
    w = np.zeros(4)
    out = to_simplex(w, raise_if_empty=False)
    np.testing.assert_allclose(out, [0.25, 0.25, 0.25, 0.25])


# ---------------------------------------------------------------------------
# n_active / herfindahl / effective_n
# ---------------------------------------------------------------------------


def test_n_active_counts_strictly_above_threshold():
    w = np.array([0.0, 1e-7, 0.5, 0.5])
    assert n_active(w) == 2  # only 0.5 and 0.5 are > 1e-6
    assert n_active(w, threshold=1e-8) == 3


def test_herfindahl_uniform():
    p = 10
    w = np.full(p, 1.0 / p)
    assert herfindahl(w) == pytest.approx(1.0 / p)


def test_herfindahl_concentrated():
    w = np.array([1.0, 0.0, 0.0])
    assert herfindahl(w) == pytest.approx(1.0)


def test_effective_n_matches_uniform():
    w = np.full(20, 0.05)
    assert effective_n(w) == pytest.approx(20.0)


def test_effective_n_concentrated():
    w = np.array([1.0, 0.0, 0.0])
    assert effective_n(w) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# raw_to_simplex_via_std
# ---------------------------------------------------------------------------


def test_raw_to_simplex_via_std_basic():
    w_std = np.array([0.5, 0.5])
    sigma = np.array([1.0, 2.0])
    out = raw_to_simplex_via_std(w_std, sigma)
    # raw = [0.5, 0.25] → renormalise → [2/3, 1/3]
    np.testing.assert_allclose(out, [2 / 3, 1 / 3])
    assert abs(out.sum() - 1.0) < 1e-12


def test_raw_to_simplex_via_std_zero_sigma_safety():
    """Zero σ entries are treated as 1.0 to avoid division-by-zero."""
    w_std = np.array([0.5, 0.5])
    sigma = np.array([1.0, 0.0])
    # With safe_sigma=[1, 1]: raw=[0.5, 0.5], simplex=[0.5, 0.5]
    out = raw_to_simplex_via_std(w_std, sigma)
    np.testing.assert_allclose(out, [0.5, 0.5])


def test_raw_to_simplex_via_std_shape_mismatch():
    with pytest.raises(ValueError):
        raw_to_simplex_via_std(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# Protocol structural typing
# ---------------------------------------------------------------------------


def test_simplex_solver_protocol_runtime_check():
    """All Phase-1+2 solver classes must satisfy the SimplexSolver Protocol."""
    from sit.solvers import (
        FISTA,
        EqualWeightTopNSolver,
        MIQPSolver,
        OMPSolver,
        RandomEqualWeightSolver,
        SklearnLassoSolver,
        SparseTrackerADMM,
        TopNMarketCapSolver,
    )

    # Construct cheaply (no fit needed for isinstance check)
    instances = [
        SparseTrackerADMM(lam=0.01, verbose=False),
        FISTA(lam=0.01),
        SklearnLassoSolver(lam=0.01),
        OMPSolver(K=5),
        MIQPSolver(K=5),
        TopNMarketCapSolver(N=5, market_caps=np.ones(10)),
        EqualWeightTopNSolver(N=5, market_caps=np.ones(10)),
        RandomEqualWeightSolver(N=5, p=10),
    ]
    for s in instances:
        assert isinstance(s, SimplexSolver), f"{type(s).__name__} does not satisfy SimplexSolver"
