"""sit/solvers/omp.py — Non-negative Orthogonal Matching Pursuit.

OMP is the *greedy* baseline. At each iteration it adds the column most
correlated with the current residual, then re-fits all selected columns by
non-negative least squares. This gives a solution with cardinality :math:`\\le K`
*by construction* (no λ tuning needed) — making it a perfect comparator for
showing that a *convex relaxation* (LASSO/ADMM) is competitive with a *combinatorial*
greedy approach.

Why custom (and not ``sklearn.linear_model.OrthogonalMatchingPursuit``)?
-----------------------------------------------------------------------
sklearn's OMP does **not** support a non-negativity constraint and uses a
plain OLS re-fit on the selected support, which can return *negative*
coefficients — useless for a long-only portfolio. We implement non-negative
OMP from scratch using ``scipy.optimize.nnls`` for the inner least-squares
re-fit and a sign-aware correlation criterion for column selection.

Algorithm (one iteration)
-------------------------
1. Compute correlations :math:`c = X^\\top r` of every column with the
   current residual :math:`r`.
2. Restrict to columns *not yet selected* and pick :math:`j^\\star = \\arg\\max_j c_j`
   (no absolute value — only positive correlations help us reduce the
   non-negative residual).
3. Add :math:`j^\\star` to the active set :math:`S`.
4. Re-fit: :math:`w_S = \\arg\\min_{w \\ge 0} \\|X_S w - y\\|_2^2` via NNLS.
5. Update residual :math:`r = y - X_S w_S`.

Stop when ``|S| == K`` or the residual norm drops below ``tol``.
NNLS may zero some coordinates that earlier rounds added, so the *final*
sparsity satisfies ``n_active(w) <= K``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import nnls

from sit.solvers.base import to_simplex

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class OMPSolver:
    """Non-negative OMP with cardinality :math:`\\le K`.

    Parameters
    ----------
    K
        Maximum number of active stocks (target cardinality).
    tol
        Stop early if the residual norm drops below this value.
    nnls_max_iter
        Inner NNLS iteration cap (forwarded to ``scipy.optimize.nnls``).
    """

    K: int = 50
    tol: float = 1e-9
    nnls_max_iter: int | None = None

    w: NDArray[np.floating] | None = field(default=None, init=False)
    selected: list[int] = field(default_factory=list, init=False)
    n_iter: int = field(default=0, init=False)
    converged: bool = field(default=False, init=False)
    residual_history: list[float] = field(default_factory=list, init=False)

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> OMPSolver:
        if self.K < 1:
            raise ValueError(f"K must be >= 1 (got {self.K}).")

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n, p = X.shape
        if y.shape != (n,):
            raise ValueError(f"y must have shape ({n},); got {y.shape}.")

        K_eff = min(self.K, p)

        r = y.copy()
        active: list[int] = []
        active_mask = np.zeros(p, dtype=bool)
        w = np.zeros(p)
        residual_norms: list[float] = [float(np.linalg.norm(r))]

        for it in range(K_eff):
            # 1. Correlations of every column with the current residual
            corr = X.T @ r
            # 2. Mask out already-selected columns AND non-positive correlations
            #    (a negative correlation cannot improve a non-negative regression).
            corr_masked = corr.copy()
            corr_masked[active_mask] = -np.inf
            j_star = int(np.argmax(corr_masked))
            if not np.isfinite(corr_masked[j_star]) or corr_masked[j_star] <= 0:
                # No positive-correlation column left → can't improve.
                break

            # 3. Add to active set
            active.append(j_star)
            active_mask[j_star] = True

            # 4. NNLS re-fit on the active sub-matrix
            X_sub = X[:, active]
            try:
                w_sub, _ = nnls(X_sub, y, maxiter=self.nnls_max_iter)
            except RuntimeError:
                # NNLS may rarely fail (linear-dependence pathology); back off.
                break

            w = np.zeros(p)
            w[active] = w_sub

            # 5. Update residual
            r = y - X_sub @ w_sub
            residual_norms.append(float(np.linalg.norm(r)))

            if residual_norms[-1] < self.tol:
                self.converged = True
                self.n_iter = it + 1
                break
        else:
            self.n_iter = K_eff

        self.w = w
        self.selected = active
        self.residual_history = residual_norms
        return self

    def get_sparse_weights(self, threshold: float = 1e-6) -> NDArray[np.floating]:
        if self.w is None:
            raise RuntimeError("OMPSolver has not been fitted. Call .fit(X, y) first.")
        return to_simplex(self.w, threshold=threshold)
