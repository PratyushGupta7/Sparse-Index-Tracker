"""Solvers for the sparse index-tracking problem.

Phase 1 (math core):
  - SparseTrackerADMM : custom ADMM with Wohlberg adaptive ρ + auto-scaling.
  - FISTA             : proximal-gradient with O'Donoghue-Candès restart.
  - reweighted_l1_admm: iterative reweighted ℓ₁ MM (Candès–Wakin–Boyd 2008).

Phase 2 (baselines):
  - SklearnLassoSolver        : sklearn ``Lasso(positive=True)`` wrapper.
  - OMPSolver                 : non-negative OMP + NNLS, cardinality K.
  - MIQPSolver                : exact ‖w‖₀ ≤ K via MOSEK MIQP (gold standard).
  - TopNMarketCapSolver       : cap-weighted top-N (naive baseline).
  - EqualWeightTopNSolver     : 1/N over top-N market-cap (naive baseline).
  - RandomEqualWeightSolver   : 1/N over random N (no-information baseline).

All solvers expose the same ``fit(X, y) -> self`` and
``get_sparse_weights(threshold) -> simplex`` surface (see ``sit.solvers.base``).
"""

from __future__ import annotations

from sit.solvers.admm import SparseTrackerADMM
from sit.solvers.base import (
    SimplexSolver,
    effective_n,
    herfindahl,
    n_active,
    raw_to_simplex_via_std,
    to_simplex,
)
from sit.solvers.fista import FISTA, fista_l1_nonneg
from sit.solvers.lasso import SklearnLassoSolver
from sit.solvers.miqp import MIQPSolver
from sit.solvers.naive import (
    EqualWeightTopNSolver,
    RandomEqualWeightSolver,
    TopNMarketCapSolver,
    equal_weight_top_n_market_cap,
    random_equal_weight,
    random_equal_weight_ensemble,
    top_n_market_cap_weights,
)
from sit.solvers.omp import OMPSolver
from sit.solvers.reweighted import reweighted_l1_admm

__all__ = [
    # Phase 1 — math core
    "FISTA",
    "SparseTrackerADMM",
    "fista_l1_nonneg",
    "reweighted_l1_admm",
    # Phase 2 — baselines
    "EqualWeightTopNSolver",
    "MIQPSolver",
    "OMPSolver",
    "RandomEqualWeightSolver",
    "SklearnLassoSolver",
    "TopNMarketCapSolver",
    # Helpers
    "SimplexSolver",
    "effective_n",
    "equal_weight_top_n_market_cap",
    "herfindahl",
    "n_active",
    "random_equal_weight",
    "random_equal_weight_ensemble",
    "raw_to_simplex_via_std",
    "to_simplex",
    "top_n_market_cap_weights",
]
