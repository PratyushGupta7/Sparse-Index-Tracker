"""sit/solvers/base.py — shared interface and utilities for every baseline.

All seven Phase-2 baselines (ADMM, FISTA, sklearn-LASSO, OMP, MIQP, top-N
market-cap, equal-weight) expose the *same* surface so the
``MethodComparison`` orchestrator and any downstream consumer (the API, the
walk-forward backtester, the frontend) can treat them as drop-in replacements.

The contract:

  >>> solver.fit(X_train, y_train)  # may take ms (FISTA) or seconds (MIQP)
  >>> w = solver.get_sparse_weights()  # returns weights ∈ Δ_p (probability simplex)

In addition we expose a few stateless helpers that *every* solver needs:

  - ``to_simplex(w, threshold)`` : projection + renormalisation utility.
  - ``n_active(w, threshold)``   : count non-zero weights.
  - ``herfindahl(w)``            : Herfindahl-Hirschman concentration index.
  - ``raw_to_simplex_via_std(...)``: convert standardised-feature weights back
    to a tradeable raw-return portfolio (the inverse-standardisation trick
    that matters for *any* solver fit on standardised X).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Common interface (Protocol = structural typing — no inheritance required)
# ---------------------------------------------------------------------------


@runtime_checkable
class SimplexSolver(Protocol):
    """Anything that can produce probability-simplex portfolio weights."""

    def fit(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> SimplexSolver: ...

    def get_sparse_weights(self, threshold: float = 1e-6) -> NDArray[np.floating]:
        """Return weights with ``w_j >= 0`` and ``sum(w) == 1`` (after thresholding)."""
        ...


# ---------------------------------------------------------------------------
# Stateless utilities
# ---------------------------------------------------------------------------


def to_simplex(
    w: NDArray[np.floating],
    threshold: float = 1e-6,
    *,
    raise_if_empty: bool = True,
) -> NDArray[np.floating]:
    """Hard-clip negatives + below-threshold weights, then renormalise to sum 1.

    This is the canonical "post-process raw solver output into a tradeable
    portfolio" step. It is **not** the same as L1-ball projection
    (``sum |w_j| <= 1``); we project onto the *non-negative* simplex by
    truncation + scaling, which is sound when the upstream optimiser already
    enforced ``w >= 0``.

    Parameters
    ----------
    w
        Raw weight vector. May contain small negative noise (e.g. from sklearn).
    threshold
        Coordinates with ``w_j < threshold`` are zeroed.
    raise_if_empty
        If ``True`` (default), raise ``ValueError`` when *every* weight is
        below ``threshold``. Set to ``False`` to silently return a uniform
        vector — useful in stress-tests.

    Returns
    -------
    NDArray with shape ``(p,)``, ``>= 0``, sum exactly 1.
    """
    w = np.maximum(np.asarray(w, dtype=np.float64), 0.0).copy()
    w[w < threshold] = 0.0
    s = float(w.sum())
    if s < 1e-12:
        if raise_if_empty:
            raise ValueError(
                f"All weights are below threshold {threshold:.2e}; cannot project to simplex."
            )
        return np.full_like(w, 1.0 / max(w.size, 1))
    return w / s


def n_active(w: NDArray[np.floating], threshold: float = 1e-6) -> int:
    """Number of strictly active coordinates (the achieved sparsity)."""
    return int((np.asarray(w) > threshold).sum())


def herfindahl(w: NDArray[np.floating]) -> float:
    """Herfindahl-Hirschman concentration index ``∑ w_j²``.

    For weights on the simplex, ``HHI`` ranges from ``1/p`` (perfectly equal
    weights, lowest concentration) to ``1`` (single asset, highest
    concentration). Lower is more diversified.
    """
    w = np.asarray(w, dtype=np.float64)
    return float(np.sum(w * w))


def effective_n(w: NDArray[np.floating]) -> float:
    """Effective number of holdings ``1 / HHI`` (a.k.a. inverse Herfindahl).

    For an equal-weighted portfolio of N stocks this returns N exactly.
    For a Pareto-distributed portfolio it returns the equivalent N.
    """
    h = herfindahl(w)
    return float(1.0 / h) if h > 0 else 0.0


def raw_to_simplex_via_std(
    w_std: NDArray[np.floating],
    sigma: NDArray[np.floating],
    *,
    threshold: float = 1e-6,
    raise_if_empty: bool = True,
) -> NDArray[np.floating]:
    """Convert standardised-feature weights to tradeable raw-return weights.

    If a solver was fit on ``X_std = X_raw / σ`` (the financially-correct
    way), its weights ``w_std`` interpret each *standardised* return. To make
    them tradeable on real prices we need ``w_raw = w_std / σ``, then
    re-simplex.

    This is the same inverse-transform that ``SparseTrackerADMM.get_raw_weights``
    implements; we centralise it here so every baseline can share it.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != w_std.shape:
        raise ValueError(f"sigma shape {sigma.shape} must match w_std shape {w_std.shape}.")
    safe_sigma = np.where(sigma > 1e-12, sigma, 1.0)
    w_raw = np.asarray(w_std, dtype=np.float64) / safe_sigma
    return to_simplex(w_raw, threshold=threshold, raise_if_empty=raise_if_empty)


__all__ = [
    "SimplexSolver",
    "effective_n",
    "herfindahl",
    "n_active",
    "raw_to_simplex_via_std",
    "to_simplex",
]
