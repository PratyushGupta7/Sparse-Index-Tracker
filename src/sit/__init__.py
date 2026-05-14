"""sit — Sparse Index Tracker.

A custom ADMM solver for L1-regularized, non-negative portfolio optimization
that replicates a benchmark index using a sparse subset of constituents.

Public entry points are re-exported here for convenience:

    >>> from sit import SparseTrackerADMM
    >>> from sit.data.loader import SP500DataLoader
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["SparseTrackerADMM", "__version__"]

# Re-export the most important symbol so callers can write `from sit import ...`
from sit.solvers.admm import SparseTrackerADMM
