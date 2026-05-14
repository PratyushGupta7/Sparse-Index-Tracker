"""sit/solvers/lasso.py — sklearn ``Lasso(positive=True)`` baseline.

This is the **textbook** answer to "*why don't you just call sklearn?*". We
include it so that we can quantitatively answer that question: same data,
same penalty, head-to-head on (R², TE, time, sparsity, max-weight,
Herfindahl).

Key conversion you must get right
---------------------------------
``sklearn.linear_model.Lasso`` minimises

.. math::

    \\frac{1}{2n}\\,\\|y - X w\\|_2^2 \\;+\\; \\alpha\\,\\|w\\|_1

while our ADMM minimises

.. math::

    \\tfrac{1}{2}\\,\\|X w - y\\|_2^2 \\;+\\; \\lambda\\,\\|w\\|_1.

The two are equivalent under the substitution :math:`\\alpha = \\lambda / n`.
We perform that conversion automatically so callers always work in our
:math:`\\lambda` units.

We also call ``Lasso(positive=True, fit_intercept=False)`` because (a) our
problem requires non-negative weights and (b) we never want sklearn to learn
an intercept — the intercept of an index-tracking regression is meaningless
on standardised returns and adding one would silently shift all weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import Lasso

from sit.solvers.base import to_simplex

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SklearnLassoSolver:
    """Drop-in baseline matching the ``SparseTrackerADMM`` API surface.

    Parameters
    ----------
    lam
        L1 penalty in *our* convention (matches ``SparseTrackerADMM.lam``).
        Internally converted to sklearn's ``alpha = lam / n_samples`` at
        ``fit`` time.
    max_iter, tol
        Forwarded to ``sklearn.linear_model.Lasso``.
    """

    lam: float = 0.01
    max_iter: int = 10_000
    tol: float = 1e-6

    w: NDArray[np.floating] | None = field(default=None, init=False)
    n_iter: int = field(default=0, init=False)
    converged: bool = field(default=False, init=False)
    _backend: Lasso | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> SklearnLassoSolver:
        if self.lam < 0:
            raise ValueError(f"lam must be >= 0 (got {self.lam}).")

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n = X.shape[0]
        # sklearn alpha is the per-sample L1 weight; convert from our convention
        alpha = float(self.lam) / max(n, 1)

        self._backend = Lasso(
            alpha=alpha,
            positive=True,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.tol,
            selection="cyclic",
            warm_start=False,
        )
        self._backend.fit(X, y)

        self.w = np.asarray(self._backend.coef_, dtype=np.float64).copy()
        self.n_iter = int(getattr(self._backend, "n_iter_", 0))
        # sklearn raises ConvergenceWarning if max_iter hit; we treat
        # n_iter < max_iter as "converged"
        self.converged = self.n_iter < self.max_iter
        return self

    def get_sparse_weights(self, threshold: float = 1e-6) -> NDArray[np.floating]:
        if self.w is None:
            raise RuntimeError("SklearnLassoSolver has not been fitted. Call .fit(X, y) first.")
        return to_simplex(self.w, threshold=threshold)

    @staticmethod
    def lambda_to_alpha(lam: float, n_samples: int) -> float:
        """Public converter — handy for tests and the comparison driver."""
        return float(lam) / max(int(n_samples), 1)
