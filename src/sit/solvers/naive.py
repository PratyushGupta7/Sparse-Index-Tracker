"""sit/solvers/naive.py — non-optimisation baselines.

Three "smart-money would never do this" baselines that we still need in the
comparison so a recruiter can see *exactly* how much value the solver adds:

1. **Top-N market-cap weighted** — buy the N largest stocks weighted by their
   market capitalisation (the standard "approximate the index naively"
   strategy).
2. **Equal-weight top-N market-cap** — same N stocks, equal weights ``1/N``.
3. **Random-N equal-weight** — pick N tickers uniformly at random and weight
   them equally; reported as a *distribution* over many seeds with mean ±
   confidence band.

These do **not** see ``y`` at all (they are not "fitted" in any optimisation
sense); they only need the candidate universe and (for the cap-weighted
variants) market-capitalisation data. To keep the unified ``fit(X, y)``
interface alive, ``fit`` accepts the data but ignores the labels.

For the comparison driver to call them, we expose the same
``SimplexSolver`` Protocol as the convex / greedy / MIQP baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Stateless weighting functions (pure numerics — easy to unit-test)
# ---------------------------------------------------------------------------


def top_n_market_cap_weights(
    market_caps: NDArray[np.floating],
    N: int,
) -> NDArray[np.floating]:
    """Return cap-weighted weights over the top-N stocks by market cap.

    Parameters
    ----------
    market_caps
        Shape ``(p,)``. ``NaN`` or non-positive entries are treated as
        ineligible (market-cap unknown / company delisted).
    N
        Target cardinality.

    Returns
    -------
    Length-p simplex vector with at most N non-zero entries.
    """
    caps = np.asarray(market_caps, dtype=np.float64).copy()
    p = caps.size
    if p == 0:
        raise ValueError("market_caps must be non-empty.")
    eligible = np.isfinite(caps) & (caps > 0)
    if not eligible.any():
        raise ValueError("No eligible market caps (all NaN or non-positive).")

    N = min(int(N), int(eligible.sum()))
    # argpartition for the top-N — only the partition matters, not the order
    eligible_idx = np.where(eligible)[0]
    eligible_caps = caps[eligible_idx]
    if N == eligible_caps.size:
        top_local = np.arange(N)
    else:
        top_local = np.argpartition(eligible_caps, -N)[-N:]
    top_idx = eligible_idx[top_local]

    w = np.zeros(p)
    w[top_idx] = caps[top_idx]
    total = float(w.sum())
    if total <= 0:  # pragma: no cover
        raise ValueError("Sum of selected market caps is zero — nothing to weight.")
    return w / total


def equal_weight_top_n_market_cap(
    market_caps: NDArray[np.floating],
    N: int,
) -> NDArray[np.floating]:
    """Equal weights over the top-N stocks by market cap (each gets ``1/N``)."""
    caps = np.asarray(market_caps, dtype=np.float64)
    p = caps.size
    eligible = np.isfinite(caps) & (caps > 0)
    if not eligible.any():
        raise ValueError("No eligible market caps.")
    N = min(int(N), int(eligible.sum()))
    eligible_idx = np.where(eligible)[0]
    eligible_caps = caps[eligible_idx]
    if N == eligible_caps.size:
        top_local = np.arange(N)
    else:
        top_local = np.argpartition(eligible_caps, -N)[-N:]
    top_idx = eligible_idx[top_local]

    w = np.zeros(p)
    w[top_idx] = 1.0 / N
    return w


def random_equal_weight(
    p: int,
    N: int,
    seed: int | np.random.Generator,
) -> NDArray[np.floating]:
    """Equal-weight portfolio over a uniformly random subset of N tickers."""
    if N < 1 or N > p:
        raise ValueError(f"N must be in [1, {p}]; got {N}.")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    chosen = rng.choice(p, size=N, replace=False)
    w = np.zeros(p)
    w[chosen] = 1.0 / N
    return w


def random_equal_weight_ensemble(
    p: int,
    N: int,
    n_seeds: int = 100,
    base_seed: int = 42,
) -> NDArray[np.floating]:
    """Returns the *mean* random-N portfolio across ``n_seeds`` draws.

    The mean of equally-weighted random subsets converges to the uniform
    portfolio as ``n_seeds → ∞``; with a small finite sample (n_seeds=100)
    you get a smoothed but still randomly-perturbed weight vector. This is
    the canonical "no-information baseline" that any honest portfolio
    manager can beat.
    """
    rng = np.random.default_rng(base_seed)
    acc = np.zeros(p)
    for _ in range(n_seeds):
        acc += random_equal_weight(p, N, rng)
    acc /= n_seeds
    # Renormalise (it should already sum to 1, but float drift)
    return acc / acc.sum()


# ---------------------------------------------------------------------------
# Class wrappers (so MethodComparison can treat them like any other solver)
# ---------------------------------------------------------------------------


@dataclass
class TopNMarketCapSolver:
    """Cap-weighted top-N portfolio. Requires market caps at construction."""

    N: int
    market_caps: NDArray[np.floating]

    w: NDArray[np.floating] | None = field(default=None, init=False)
    converged: bool = field(default=True, init=False)
    n_iter: int = field(default=0, init=False)

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> TopNMarketCapSolver:
        # X, y are accepted to satisfy the Protocol but ignored: this baseline
        # uses only the universe identity (encoded in the order of market_caps).
        del X, y
        if self.market_caps.shape[0] == 0:
            raise ValueError("market_caps must have shape matching X.shape[1]")
        self.w = top_n_market_cap_weights(self.market_caps, self.N)
        return self

    def get_sparse_weights(self, threshold: float = 1e-6) -> NDArray[np.floating]:
        if self.w is None:
            raise RuntimeError("TopNMarketCapSolver has not been fitted. Call .fit() first.")
        # Already on the simplex; only zero out stuff under threshold for consistency
        w = self.w.copy()
        w[w < threshold] = 0.0
        s = w.sum()
        return w / s if s > 0 else w


@dataclass
class EqualWeightTopNSolver:
    """Equal-weight top-N market-cap portfolio."""

    N: int
    market_caps: NDArray[np.floating]

    w: NDArray[np.floating] | None = field(default=None, init=False)
    converged: bool = field(default=True, init=False)
    n_iter: int = field(default=0, init=False)

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> EqualWeightTopNSolver:
        del X, y
        self.w = equal_weight_top_n_market_cap(self.market_caps, self.N)
        return self

    def get_sparse_weights(self, threshold: float = 1e-6) -> NDArray[np.floating]:
        if self.w is None:
            raise RuntimeError("EqualWeightTopNSolver has not been fitted.")
        w = self.w.copy()
        w[w < threshold] = 0.0
        s = w.sum()
        return w / s if s > 0 else w


@dataclass
class RandomEqualWeightSolver:
    """Random-N equal-weight; can return a single draw or an ensemble mean.

    Set ``ensemble=True`` for the smoothed mean over ``n_seeds`` random draws
    (the "no-information baseline"); set ``ensemble=False`` for a single
    deterministic draw seeded with ``seed`` (handy for reproducible tests).
    """

    N: int
    p: int
    seed: int = 42
    ensemble: bool = True
    n_seeds: int = 100

    w: NDArray[np.floating] | None = field(default=None, init=False)
    converged: bool = field(default=True, init=False)
    n_iter: int = field(default=0, init=False)

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> RandomEqualWeightSolver:
        del X, y
        if self.ensemble:
            self.w = random_equal_weight_ensemble(self.p, self.N, self.n_seeds, self.seed)
        else:
            self.w = random_equal_weight(self.p, self.N, self.seed)
        return self

    def get_sparse_weights(self, threshold: float = 1e-6) -> NDArray[np.floating]:
        if self.w is None:
            raise RuntimeError("RandomEqualWeightSolver has not been fitted.")
        w = self.w.copy()
        w[w < threshold] = 0.0
        s = w.sum()
        return w / s if s > 0 else w


__all__ = [
    "EqualWeightTopNSolver",
    "RandomEqualWeightSolver",
    "TopNMarketCapSolver",
    "equal_weight_top_n_market_cap",
    "random_equal_weight",
    "random_equal_weight_ensemble",
    "top_n_market_cap_weights",
]
