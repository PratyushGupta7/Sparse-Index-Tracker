"""Shared pytest fixtures.

All fixtures live here so any test module can grab them by name.

Key fixtures
------------
``rng``                  — a seeded ``numpy.random.Generator``.
``synthetic_sparse``     — a (n, p, k) ground-truth recovery problem.
``simplex_problem``      — like ``synthetic_sparse`` but the truth lies on the
                           probability simplex (positive weights summing to 1)
                           — closer to the actual portfolio-optimisation setting.
``sp500_snapshot``       — loads the pickled ``data/X_standardized.npy`` etc.
                           Skips the test if those files are absent.
``mosek_env``            — exports ``MOSEKLM_LICENSE_FILE`` and skips if absent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Random number generation
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic Generator for property-style tests."""
    return np.random.default_rng(seed=20260511)


# ---------------------------------------------------------------------------
# Synthetic problems
# ---------------------------------------------------------------------------


@dataclass
class SparseProblem:
    """A ground-truth k-sparse least-squares problem."""

    X: NDArray[np.floating]
    y: NDArray[np.floating]
    w_true: NDArray[np.floating]
    support: NDArray[np.integer]
    noise_sigma: float

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def p(self) -> int:
        return int(self.X.shape[1])

    @property
    def k(self) -> int:
        return int(self.support.size)


def _make_sparse_problem(
    rng: np.random.Generator,
    n: int,
    p: int,
    k: int,
    *,
    noise_sigma: float = 0.01,
    positive: bool = True,
    simplex: bool = False,
) -> SparseProblem:
    """Generate a reproducible sparse least-squares problem."""
    X = rng.standard_normal((n, p))
    X /= X.std(axis=0, ddof=0) + 1e-12  # column-standardise for L1 fairness
    support = rng.choice(p, size=k, replace=False)
    w_true = np.zeros(p)
    raw = rng.uniform(0.5, 1.5, size=k) if positive else rng.standard_normal(k)
    if simplex:
        raw = np.abs(raw)
        raw /= raw.sum()
    w_true[support] = raw
    y = X @ w_true + noise_sigma * rng.standard_normal(n)
    return SparseProblem(X=X, y=y, w_true=w_true, support=np.sort(support), noise_sigma=noise_sigma)


@pytest.fixture
def synthetic_sparse(rng: np.random.Generator) -> SparseProblem:
    """Default small-scale ground-truth problem: n=80, p=200, k=8, positive weights."""
    return _make_sparse_problem(rng, n=80, p=200, k=8, positive=True, simplex=False)


@pytest.fixture
def simplex_problem(rng: np.random.Generator) -> SparseProblem:
    """A ground-truth problem with weights on the probability simplex (sum=1)."""
    return _make_sparse_problem(rng, n=100, p=250, k=10, positive=True, simplex=True)


@pytest.fixture
def make_problem(rng: np.random.Generator):
    """Factory fixture for ad-hoc shapes."""

    def _factory(
        n: int = 80, p: int = 200, k: int = 8, *, simplex: bool = False, noise: float = 0.01
    ) -> SparseProblem:
        # Re-seed deterministically per (n, p, k) to make matrix-parameterised
        # tests reproducible.
        sub_rng = np.random.default_rng(seed=(n * 100003 + p * 113 + k))
        return _make_sparse_problem(
            sub_rng, n=n, p=p, k=k, noise_sigma=noise, positive=True, simplex=simplex
        )

    return _factory


# ---------------------------------------------------------------------------
# Real-world cached data (S&P 500 snapshot from Phase 1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sp500_snapshot() -> dict[str, NDArray[np.floating]]:
    """Load the cached S&P 500 data if available; skip otherwise."""
    from sit.paths import DATA_DIR

    needed = ("X_standardized.npy", "y_spy.npy", "X_raw.npy", "X_std.npy")
    paths = {name: DATA_DIR / name for name in needed}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        pytest.skip(f"S&P 500 cached data missing: {missing}")
    return {name.replace(".npy", ""): np.load(str(p)) for name, p in paths.items()}


# ---------------------------------------------------------------------------
# MOSEK licence detection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mosek_env() -> str:
    """Resolve the MOSEK academic licence path, or skip."""
    candidates = [
        os.environ.get("MOSEKLM_LICENSE_FILE"),
        str(Path.home() / "mosek" / "mosek.lic"),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            os.environ.setdefault("MOSEKLM_LICENSE_FILE", path)
            return path
    pytest.skip(
        "MOSEK licence file not found (looked at MOSEKLM_LICENSE_FILE and ~/mosek/mosek.lic)"
    )
