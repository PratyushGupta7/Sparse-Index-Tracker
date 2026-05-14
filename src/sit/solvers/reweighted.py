"""sit/solvers/reweighted.py — iterative reweighted ℓ₁ via ADMM inner loop.

Implements the *Majorization–Minimization* (MM) scheme of

    Candès, Wakin & Boyd (2008),
    "Enhancing Sparsity by Reweighted ℓ₁ Minimization",
    J. Fourier Anal. Appl. 14, 877–905.

The idea: the ℓ₀ "norm" is not convex, but its tight convex envelope on the
unit cube — the ℓ₁ norm — over-penalises large coefficients (it shrinks them
the same as small ones). Replacing :math:`\\lambda \\|w\\|_1` with the
*weighted* surrogate :math:`\\sum_j w_j |w_j|` and iteratively setting
:math:`w_j^{(l+1)} = \\lambda \\big/(|w_j^{(l)}| + \\varepsilon)` yields a
sequence of convex subproblems whose minimizers tend to a stationary point
of the (non-convex) log-sum penalty

    .. math::

        P_\\varepsilon(w) = \\sum_j \\log\\!\\left(\\frac{|w_j|}{\\varepsilon} + 1\\right),

which approximates :math:`\\|w\\|_0` arbitrarily well as :math:`\\varepsilon \\to 0`.

In practice this *sharpens* sparsity recovery: noise-level coefficients that
escaped the first round get killed in subsequent rounds, while informative
coefficients with large magnitudes are barely penalised.

This module reuses ``SparseTrackerADMM``'s vector-λ mode (added in Phase 1)
so we don't duplicate the ADMM update logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sit.solvers.admm import SparseTrackerADMM


@dataclass
class ReweightedResult:
    """Container for reweighted-ℓ₁ outer-loop output."""

    w: NDArray[np.floating]
    """Final weights in standardised feature space (the ``z`` ADMM iterate)."""

    weights_history: list[NDArray[np.floating]]
    """Sequence of iterates across outer rounds (length = n_outer + 1)."""

    lam_vec_history: list[NDArray[np.floating]]
    """Per-coordinate λ vector used in each outer round."""

    n_active_history: list[int]
    """Sparsity at the end of each outer round."""

    objective_history: list[float]
    """Final convex-subproblem objective per outer round."""

    inner_iters: list[int]
    """ADMM iterations used in each outer round (for diagnostics)."""

    converged: bool
    """``True`` iff the outer support stopped changing before ``n_outer``."""

    n_outer_run: int
    """Number of outer rounds actually executed."""


def reweighted_l1_admm(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    lam: float,
    *,
    n_outer: int = 5,
    epsilon: float = 1e-3,
    rho: float = 1.0,
    max_inner_iter: int = 3000,
    tol: float = 1e-6,
    adaptive_rho: bool = True,
    support_tol: float = 1e-8,
    verbose: bool = False,
) -> ReweightedResult:
    """Solve the reweighted-ℓ₁ surrogate via repeated ADMM.

    At outer iteration :math:`\\ell`:

    .. math::

        w^{(\\ell+1)} = \\arg\\min_{w \\ge 0} \\tfrac{1}{2}\\|Xw - y\\|_2^2
            + \\sum_{j=1}^{p} \\lambda_j^{(\\ell)} w_j,
        \\quad
        \\lambda_j^{(\\ell+1)} = \\frac{\\lambda}{|w_j^{(\\ell+1)}| + \\varepsilon}.

    Parameters
    ----------
    X, y
        Same as the ADMM solver: ``X`` is ``(n, p)`` standardised features,
        ``y`` is the ``(n,)`` benchmark return vector.
    lam
        Base penalty controlling overall sparsity. Each coordinate's effective
        λ is ``lam / (|w_j| + epsilon)``, so larger weights see a *smaller*
        penalty (this is the entire point — non-uniform shrinkage).
    n_outer
        Maximum number of outer (MM) iterations. Candès–Wakin–Boyd report
        diminishing returns after 3–5 rounds.
    epsilon
        Smoothing parameter. Smaller ε → behaviour closer to ℓ₀, but with
        more numerical instability. Should be set on the same scale as the
        smallest "informative" weight you want to keep (≈ 10⁻³ for our data).
    support_tol
        A coordinate is considered "active" when ``w_j > support_tol``.
        Outer-loop convergence: support unchanged between rounds.
    other args
        Forwarded to the inner ``SparseTrackerADMM``.

    Returns
    -------
    ReweightedResult dataclass.
    """
    if lam <= 0:
        raise ValueError(f"lam must be > 0 (got {lam}).")
    if n_outer < 1:
        raise ValueError(f"n_outer must be >= 1 (got {n_outer}).")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0 (got {epsilon}).")

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    _, p = X.shape

    # Round 0: solve the *unweighted* problem (uniform λ on every coordinate).
    lam_vec = np.full(p, float(lam), dtype=np.float64)

    weights_history: list[NDArray[np.floating]] = []
    lam_vec_history: list[NDArray[np.floating]] = []
    n_active_history: list[int] = []
    objective_history: list[float] = []
    inner_iters: list[int] = []

    prev_support: NDArray[np.bool_] | None = None
    converged = False
    last_solver: SparseTrackerADMM | None = None

    for outer in range(n_outer):
        solver = SparseTrackerADMM(
            lam=float(lam),  # ignored when lam_vec is set, but kept for logs
            rho=rho,
            max_iter=max_inner_iter,
            tol=tol,
            adaptive_rho=adaptive_rho,
            verbose=False,
            lam_vec=lam_vec.copy(),
        )
        solver.fit(X, y)
        last_solver = solver

        w_new = solver.z.copy()
        support = w_new > support_tol
        n_active = int(support.sum())

        weights_history.append(w_new)
        lam_vec_history.append(lam_vec.copy())
        n_active_history.append(n_active)
        objective_history.append(
            float(solver.objective_values[-1] if solver.objective_values else float("nan"))
        )
        inner_iters.append(int(solver.n_iter))

        if verbose:
            print(
                f"[reweighted ℓ₁] outer {outer + 1}/{n_outer}: "
                f"nnz={n_active:4d} | inner_iter={solver.n_iter:4d} | "
                f"obj={objective_history[-1]:.6e}"
            )

        # Check outer convergence: support unchanged
        if prev_support is not None and np.array_equal(support, prev_support):
            converged = True
            if verbose:
                print(f"[reweighted ℓ₁] support stabilised after {outer + 1} outer round(s)")
            break
        prev_support = support

        # Build next-round weights
        lam_vec = lam / (np.abs(w_new) + epsilon)

    assert last_solver is not None  # for type-checker peace of mind

    return ReweightedResult(
        w=last_solver.z.copy(),
        weights_history=weights_history,
        lam_vec_history=lam_vec_history,
        n_active_history=n_active_history,
        objective_history=objective_history,
        inner_iters=inner_iters,
        converged=converged,
        n_outer_run=len(weights_history),
    )
