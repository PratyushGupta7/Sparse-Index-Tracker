"""sit.benchmarks — head-to-head comparison drivers and Pareto sweeps."""

from __future__ import annotations

from sit.benchmarks.comparison import (
    ComparisonConfig,
    ComparisonInputs,
    MethodComparison,
    MethodResult,
    compute_metrics,
    pareto_sweep,
    save_results,
)
from sit.benchmarks.datasets import (
    SyntheticTruth,
    make_sp500_snapshot,
    make_synthetic_dataset,
)
from sit.benchmarks.plots import plot_method_comparison, plot_pareto_frontier

__all__ = [
    "ComparisonConfig",
    "ComparisonInputs",
    "MethodComparison",
    "MethodResult",
    "SyntheticTruth",
    "compute_metrics",
    "make_sp500_snapshot",
    "make_synthetic_dataset",
    "pareto_sweep",
    "plot_method_comparison",
    "plot_pareto_frontier",
    "save_results",
]
