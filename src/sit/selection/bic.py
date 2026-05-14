"""sit/selection/bic.py — BIC & EBIC for high-dimensional λ selection.

Why information criteria?
-------------------------
* **No held-out data needed.** Cross-validation is expensive in the
  walk-forward setting (Phase 3) and statistically wobbly when the look-back
  is short (n=120 days). Information criteria give a closed-form, *data-free*
  λ selection.
* **Asymptotically consistent.** Under standard conditions, EBIC consistently
  selects the true active set in the high-dimensional regime where :math:`p`
  grows with :math:`n` (Chen & Chen 2008, *Extended Bayesian Information
  Criterion for Model Selection with Large Model Spaces*, Biometrika).

Definitions
-----------
For an estimator :math:`\\hat w(\\lambda)` of size :math:`k(\\lambda) = \\#\\{j: \\hat w_j \\ne 0\\}`
fit on :math:`n` observations and :math:`p` candidate features with residual
sum of squares :math:`\\mathrm{RSS}(\\lambda) = \\|X\\hat w(\\lambda) - y\\|_2^2`:

.. math::

    \\mathrm{BIC}(\\lambda)  &= n \\log\\!\\bigl(\\mathrm{RSS}(\\lambda)/n\\bigr) + k(\\lambda)\\,\\log n \\\\
    \\mathrm{EBIC}_\\gamma(\\lambda) &= \\mathrm{BIC}(\\lambda) + 2\\gamma\\,k(\\lambda)\\,\\log p

with :math:`\\gamma \\in [0, 1]`. :math:`\\gamma = 0` reduces EBIC to BIC;
:math:`\\gamma = 1` is the most conservative (recommended for :math:`p \\gg n`).

For our sparse index-tracking problem with :math:`n \\approx 120,\\, p \\approx 500`,
EBIC with :math:`\\gamma = 0.5` (a common default) tends to pick 30–80 stocks,
matching the desired "sparse but informative" target band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sit.solvers.admm import SparseTrackerADMM


@dataclass
class LambdaSelectionResult:
    """Output of a BIC/EBIC λ sweep along the regularization path."""

    criterion: str
    """Which criterion was used (``'bic'`` or ``'ebic'`` with γ)."""

    lambdas: NDArray[np.floating]
    """The λ grid that was swept (length M, descending in 1/λ)."""

    rss: NDArray[np.floating]
    """Residual sum of squares :math:`\\|X\\hat w(\\lambda) - y\\|_2^2` at each λ."""

    n_active: NDArray[np.integer]
    """Number of non-zero coefficients at each λ."""

    scores: NDArray[np.floating]
    """The criterion value at each λ. Lower = better."""

    weights: list[NDArray[np.floating]]
    """The fitted weight vector at each λ (raw ADMM ``z``, pre-normalisation)."""

    best_index: int
    """Index into ``lambdas`` of the chosen λ (argmin of ``scores``)."""

    @property
    def best_lambda(self) -> float:
        return float(self.lambdas[self.best_index])

    @property
    def best_weights(self) -> NDArray[np.floating]:
        return self.weights[self.best_index]

    @property
    def best_n_active(self) -> int:
        return int(self.n_active[self.best_index])


# ---------------------------------------------------------------------------
# Core criterion computations
# ---------------------------------------------------------------------------


def bic_score(rss: float, k: int, n: int) -> float:
    """Compute classical BIC for a least-squares fit with Gaussian residuals.

    .. math::

        \\mathrm{BIC} = n \\log(\\mathrm{RSS}/n) + k \\log n.

    Notes
    -----
    The constant :math:`n + n\\log(2\\pi)` from the log-likelihood is dropped
    (same for every λ, so it doesn't affect argmin).
    """
    if rss <= 0:
        # Perfect fit: penalise by -∞ on log term would dominate; treat as
        # very-negative log term truncated.
        log_term = float("-inf")
    else:
        log_term = float(n) * float(np.log(rss / n))
    return log_term + k * float(np.log(n))


def ebic_score(rss: float, k: int, n: int, p: int, gamma: float = 0.5) -> float:
    """Compute Extended BIC (Chen & Chen 2008).

    .. math::

        \\mathrm{EBIC}_\\gamma = \\mathrm{BIC} + 2\\gamma\\,k\\,\\log p.

    Falls back to BIC when :math:`\\gamma = 0`. Use :math:`\\gamma \\in \\{0.5, 1\\}`
    for high-dimensional problems.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1] (got {gamma}).")
    return bic_score(rss, k, n) + 2.0 * gamma * float(k) * float(np.log(max(p, 2)))


# ---------------------------------------------------------------------------
# λ-path sweeps
# ---------------------------------------------------------------------------


def _make_lambda_grid(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    n_lambdas: int,
    lam_min_ratio: float,
) -> NDArray[np.floating]:
    lam_max = SparseTrackerADMM.compute_lambda_max(X, y)
    return np.logspace(
        np.log10(lam_min_ratio * lam_max),
        np.log10(lam_max),
        n_lambdas,
    )


def select_lambda_bic(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    n_lambdas: int = 30,
    lam_min_ratio: float = 0.01,
    rho: float = 1.0,
    max_inner_iter: int = 3000,
    tol: float = 1e-6,
    support_tol: float = 1e-6,
    verbose: bool = False,
) -> LambdaSelectionResult:
    """Sweep λ on a log-spaced grid and pick the BIC-minimising value.

    Parameters
    ----------
    X, y
        Standardised features and target.
    n_lambdas
        Number of grid points (default 30 is plenty for visual inspection).
    lam_min_ratio
        Smallest λ as a fraction of :math:`\\lambda_{\\max}`.
    Other arguments are forwarded to ``SparseTrackerADMM``.

    Returns
    -------
    LambdaSelectionResult
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = X.shape[0]
    lambdas = _make_lambda_grid(X, y, n_lambdas, lam_min_ratio)

    rss_list: list[float] = []
    k_list: list[int] = []
    scores: list[float] = []
    weights: list[NDArray[np.floating]] = []

    for i, lam in enumerate(lambdas):
        solver = SparseTrackerADMM(
            lam=float(lam),
            rho=rho,
            max_iter=max_inner_iter,
            tol=tol,
            adaptive_rho=True,
            verbose=False,
        )
        solver.fit(X, y)
        w = solver.z.copy()
        active = w > support_tol
        k = int(active.sum())
        residual = X @ w - y
        rss = float(residual @ residual)
        score = bic_score(rss, k, n)
        rss_list.append(rss)
        k_list.append(k)
        scores.append(score)
        weights.append(w)
        if verbose:
            print(
                f"[BIC sweep {i + 1}/{n_lambdas}] lam={lam:.6f} "
                f"nnz={k:4d} rss={rss:.6e} BIC={score:.4f}"
            )

    scores_arr = np.asarray(scores)
    best_idx = int(np.argmin(scores_arr))
    return LambdaSelectionResult(
        criterion="bic",
        lambdas=lambdas,
        rss=np.asarray(rss_list),
        n_active=np.asarray(k_list, dtype=int),
        scores=scores_arr,
        weights=weights,
        best_index=best_idx,
    )


def ebic_lambda_selection(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    gamma: float = 0.5,
    n_lambdas: int = 30,
    lam_min_ratio: float = 0.01,
    rho: float = 1.0,
    max_inner_iter: int = 3000,
    tol: float = 1e-6,
    support_tol: float = 1e-6,
    verbose: bool = False,
) -> LambdaSelectionResult:
    """Like ``select_lambda_bic`` but with the EBIC criterion (γ-tunable).

    For :math:`p \\gg n` use ``gamma >= 0.5``. With :math:`\\gamma = 0` this
    coincides with classical BIC.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n, p = X.shape
    lambdas = _make_lambda_grid(X, y, n_lambdas, lam_min_ratio)

    rss_list: list[float] = []
    k_list: list[int] = []
    scores: list[float] = []
    weights: list[NDArray[np.floating]] = []

    for i, lam in enumerate(lambdas):
        solver = SparseTrackerADMM(
            lam=float(lam),
            rho=rho,
            max_iter=max_inner_iter,
            tol=tol,
            adaptive_rho=True,
            verbose=False,
        )
        solver.fit(X, y)
        w = solver.z.copy()
        active = w > support_tol
        k = int(active.sum())
        residual = X @ w - y
        rss = float(residual @ residual)
        score = ebic_score(rss, k, n, p, gamma=gamma)
        rss_list.append(rss)
        k_list.append(k)
        scores.append(score)
        weights.append(w)
        if verbose:
            print(
                f"[EBIC γ={gamma} sweep {i + 1}/{n_lambdas}] lam={lam:.6f} "
                f"nnz={k:4d} rss={rss:.6e} EBIC={score:.4f}"
            )

    scores_arr = np.asarray(scores)
    best_idx = int(np.argmin(scores_arr))
    return LambdaSelectionResult(
        criterion=f"ebic_gamma_{gamma}",
        lambdas=lambdas,
        rss=np.asarray(rss_list),
        n_active=np.asarray(k_list, dtype=int),
        scores=scores_arr,
        weights=weights,
        best_index=best_idx,
    )
