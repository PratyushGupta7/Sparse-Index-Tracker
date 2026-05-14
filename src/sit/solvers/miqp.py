"""sit/solvers/miqp.py — exact cardinality-constrained baseline (MIQP via MOSEK).

This is the **gold-standard** solver: it solves the *combinatorially correct*
non-convex problem

.. math::

    \\min_{w \\in \\mathbb{R}^p} \\;\\tfrac{1}{2}\\|Xw - y\\|_2^2
    \\quad \\text{s.t.} \\quad
    w \\ge 0,\\;\\; \\mathbf{1}^\\top w = 1,\\;\\; \\|w\\|_0 \\le K.

The :math:`\\ell_0` constraint is encoded by the standard *big-M* mixed-integer
formulation:

.. math::

    w_j \\le M\\,z_j, \\qquad z_j \\in \\{0, 1\\}, \\qquad \\sum_j z_j \\le K,

with :math:`M = 1` because the simplex constraint :math:`\\mathbf{1}^\\top w = 1`
already forces :math:`w_j \\le 1`.

Why include MIQP?
-----------------
Every other baseline (LASSO, OMP, ADMM) is either a *convex relaxation* of
this program or a *greedy approximation* to it. Showing that our ADMM lands
within (say) 0.4 % R² of MIQP at 1/40th the runtime is the **strongest
empirical defence of the ℓ₁ relaxation we use**. Without MIQP in the
comparison, "we use ℓ₁ for tractability" is unsubstantiated.

Cost
----
MIQP is **NP-hard**. We cap :math:`K \\le 60` and total problem size to a
few hundred candidate stocks; even at that scale MOSEK takes 5–60 seconds.
For larger universes use the ``time_limit`` parameter and accept the
incumbent (often near-optimal) solution.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import cvxpy as cp

    _CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    _CVXPY_AVAILABLE = False

from sit.solvers.base import to_simplex

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _ensure_mosek_license() -> None:
    """Set ``MOSEKLM_LICENSE_FILE`` to ``~/mosek/mosek.lic`` if not already set."""
    if "MOSEKLM_LICENSE_FILE" not in os.environ:
        candidate = Path.home() / "mosek" / "mosek.lic"
        if candidate.is_file():
            os.environ["MOSEKLM_LICENSE_FILE"] = str(candidate)


@dataclass
class MIQPSolver:
    """Exact cardinality-constrained portfolio optimisation via MOSEK MIQP.

    Parameters
    ----------
    K
        Hard cap on the number of active stocks. Must be in ``[1, p]``.
        For tractability we recommend :math:`K \\le 60` on universes of
        :math:`p \\le 500`.
    big_M
        Big-M coefficient for the ``w_j <= M * z_j`` linkage constraint.
        Default 1.0 is tight because of the simplex constraint.
    enforce_simplex
        If ``True`` (default) add the equality ``sum(w) = 1``. If ``False``
        we drop the simplex and just penalise tracking error — useful for
        comparing on the same scale as the LASSO/ADMM problems whose
        formulation lacks the explicit simplex.
    time_limit
        Wall-clock seconds before the solver returns its incumbent
        (potentially sub-optimal) solution. ``None`` = no limit.
    mip_gap
        Relative MIP gap tolerance. Default 1 % is plenty for our purposes;
        tighter (1e-4) doubles run-time for marginal gain.
    solver
        CVXPY solver name. Defaults to MOSEK (best MIQP performance);
        falls back to HIGHS if MOSEK is unavailable.
    """

    K: int = 50
    big_M: float = 1.0  # noqa: N815 — capital M follows the optimisation literature
    enforce_simplex: bool = True
    time_limit: float | None = 120.0
    mip_gap: float = 1e-2
    solver: str = "MOSEK"

    w: NDArray[np.floating] | None = field(default=None, init=False)
    z: NDArray[np.floating] | None = field(default=None, init=False)
    objective_value: float = field(default=float("nan"), init=False)
    elapsed_s: float = field(default=0.0, init=False)
    converged: bool = field(default=False, init=False)
    status: str = field(default="not_fit", init=False)

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> MIQPSolver:
        if not _CVXPY_AVAILABLE:
            raise RuntimeError(
                "MIQPSolver requires cvxpy + a MIQP-capable backend (MOSEK preferred). "
                "Install with `pip install cvxpy mosek`."
            )

        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        n, p = X.shape
        if y.shape != (n,):
            raise ValueError(f"y must have shape ({n},); got {y.shape}.")
        if not 1 <= self.K <= p:
            raise ValueError(f"K must be in [1, {p}]; got {self.K}.")

        _ensure_mosek_license()

        w = cp.Variable(p, nonneg=True)
        z = cp.Variable(p, boolean=True)

        constraints = [
            w <= self.big_M * z,
            cp.sum(z) <= self.K,
        ]
        if self.enforce_simplex:
            constraints.append(cp.sum(w) == 1)

        objective = cp.Minimize(0.5 * cp.sum_squares(X @ w - y))
        prob = cp.Problem(objective, constraints)

        # Choose solver, with graceful fallback
        installed = set(cp.installed_solvers())
        chosen = self.solver if self.solver in installed else None
        if chosen is None:
            for fallback in ("MOSEK", "HIGHS"):
                if fallback in installed:
                    chosen = fallback
                    break
        if chosen is None:
            raise RuntimeError(
                "No MIQP-capable solver installed. Tried MOSEK, HIGHS. "
                f"Available: {sorted(installed)}"
            )

        solver_kwargs: dict = {}
        if chosen == "MOSEK":
            mosek_params: dict[str, float | int] = {
                "MSK_DPAR_MIO_TOL_REL_GAP": self.mip_gap,
            }
            if self.time_limit is not None:
                mosek_params["MSK_DPAR_MIO_MAX_TIME"] = self.time_limit
            solver_kwargs = {"mosek_params": mosek_params}
        elif chosen == "HIGHS":
            if self.time_limit is not None:
                solver_kwargs = {"time_limit": self.time_limit}

        t0 = time.perf_counter()
        prob.solve(solver=chosen, verbose=False, **solver_kwargs)
        self.elapsed_s = time.perf_counter() - t0
        self.status = str(prob.status)

        if w.value is None:
            self.w = np.zeros(p)
            self.z = np.zeros(p)
            self.objective_value = float("nan")
            self.converged = False
            return self

        self.w = np.asarray(w.value, dtype=np.float64).ravel()
        self.z = np.asarray(z.value, dtype=np.float64).ravel()
        self.objective_value = float(prob.value)
        self.converged = prob.status in {"optimal", "optimal_inaccurate"}
        return self

    def get_sparse_weights(self, threshold: float = 1e-6) -> NDArray[np.floating]:
        if self.w is None:
            raise RuntimeError("MIQPSolver has not been fitted. Call .fit(X, y) first.")
        return to_simplex(self.w, threshold=threshold)


__all__ = ["MIQPSolver"]
