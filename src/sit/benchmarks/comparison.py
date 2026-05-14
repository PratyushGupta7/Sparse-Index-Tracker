"""sit/benchmarks/comparison.py — head-to-head Phase-2 baseline orchestrator.

Runs all seven (or eight) Phase-2 methods on the *same* train/test split with
the *same* λ/K/N hyperparameters, computes a battery of metrics, and emits

    (a) ``data/method_comparison.json`` — full machine-readable result.
    (b) ``plots/method_comparison.png`` — grouped bar chart of headline metrics.
    (c) ``plots/sparsity_vs_te_pareto.png`` — cardinality vs OOS TE Pareto.

The orchestrator is *data-source agnostic*: callers pass a
``ComparisonInputs`` bundle (X_train, y_train, X_test, y_test, sigma_train,
optional market_caps + tickers). The CLI in ``benchmarks/method_comparison.py``
wires this up against the cached S&P 500 snapshot or fresh yfinance data;
unit tests wire it up against synthetic data.

Metrics
-------
Per method:
  * ``in_sample_r2``    : 1 - RSS / TSS on training residuals.
  * ``oos_r2``          : same on the held-out test residuals.
  * ``oos_te_annual``   : annualised tracking error  ``std(p - b) * √252``.
  * ``oos_ir_annual``   : annualised information ratio ``mean / std * √252``.
  * ``oos_sharpe_annual``: portfolio Sharpe (raw, not vs benchmark).
  * ``fit_time_s``      : wall-clock fit time (excludes data prep).
  * ``n_active``        : non-zero count after thresholding.
  * ``max_weight``      : single largest position.
  * ``hhi``             : Herfindahl-Hirschman concentration index.
  * ``effective_n``     : ``1 / hhi`` (equivalent equally-weighted N).

The Pareto sweep additionally exposes per-method (#stocks, OOS TE) curves
across a cardinality grid (λ for ADMM/LASSO/FISTA, K for OMP/MIQP, N for
naive baselines).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sit.solvers.admm import SparseTrackerADMM
from sit.solvers.base import effective_n, herfindahl, n_active, raw_to_simplex_via_std, to_simplex
from sit.solvers.fista import FISTA
from sit.solvers.lasso import SklearnLassoSolver
from sit.solvers.miqp import MIQPSolver
from sit.solvers.naive import (
    EqualWeightTopNSolver,
    RandomEqualWeightSolver,
    TopNMarketCapSolver,
)
from sit.solvers.omp import OMPSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Annualisation factor for daily-frequency US equity returns.
ANNUALIZATION_DAYS = 252


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass
class ComparisonInputs:
    """Bundle of arrays the comparison driver needs."""

    X_train_std: NDArray[np.floating]
    """Standardised training feature matrix, shape (n_train, p)."""

    y_train: NDArray[np.floating]
    """Training benchmark return vector, shape (n_train,)."""

    X_test_raw: NDArray[np.floating]
    """Held-out **raw** feature matrix, shape (n_test, p)."""

    y_test: NDArray[np.floating]
    """Held-out benchmark return vector, shape (n_test,)."""

    sigma_train: NDArray[np.floating]
    """Per-column std used to standardise X_train, shape (p,)."""

    X_train_raw: NDArray[np.floating] | None = None
    """Raw training feature matrix (optional; needed for in-sample raw-weight R²)."""

    market_caps: NDArray[np.floating] | None = None
    """Per-stock market caps, shape (p,). Required for cap-weighted baselines."""

    tickers: list[str] | None = None
    """Per-stock ticker symbols (length p). Optional, used in summary tables."""

    @property
    def p(self) -> int:
        return int(self.X_train_std.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.X_train_std.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test_raw.shape[0])


@dataclass
class ComparisonConfig:
    """Knobs for the comparison run. Reasonable defaults for SP500 (n=120, p≈500)."""

    K: int = 50
    """Target cardinality for OMP, MIQP, and naive top-N baselines."""

    lam_frac: float = 0.05
    """Penalty for ADMM/FISTA/LASSO as a fraction of λ_max."""

    miqp_time_limit: float = 60.0
    """Hard wall-clock cap on MIQP (seconds). NP-hard ⇒ budget required."""

    miqp_mip_gap: float = 1e-2
    """MIQP relative MIP gap tolerance."""

    n_random_seeds: int = 100
    """Number of seeds for the random-equal-weight ensemble."""

    admm_max_iter: int = 5000
    fista_max_iter: int = 5000
    lasso_max_iter: int = 20_000
    tol: float = 1e-6
    threshold: float = 1e-6

    include_methods: tuple[str, ...] | None = None
    """If set, only these methods will run. Useful for fast iteration."""

    skip_methods: tuple[str, ...] = ()
    """Methods to omit (e.g. ('miqp',) when MOSEK isn't available)."""

    seed: int = 20260511
    """Top-level RNG seed for any stochastic baseline."""


@dataclass
class MethodResult:
    """Per-method numerical result. JSON-serialisable when weights are dropped."""

    name: str
    weights_raw: NDArray[np.floating]
    """Tradeable raw-return weights (on the simplex)."""

    n_active: int
    max_weight: float
    hhi: float
    effective_n: float

    fit_time_s: float
    in_sample_r2: float
    oos_r2: float
    oos_te_annual: float
    oos_ir_annual: float
    oos_sharpe_annual: float

    converged: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_weights: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_weights:
            d.pop("weights_raw")
        else:
            d["weights_raw"] = list(map(float, self.weights_raw))
        return d


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _r2(predictions: NDArray, targets: NDArray) -> float:
    """Coefficient of determination ``1 - RSS / TSS`` (uncentred-target safe)."""
    residuals = targets - predictions
    rss = float(residuals @ residuals)
    centred = targets - float(targets.mean())
    tss = float(centred @ centred)
    if tss < 1e-30:
        return float("nan")
    return 1.0 - rss / tss


def compute_metrics(
    weights_raw: NDArray[np.floating],
    inputs: ComparisonInputs,
    *,
    fit_time_s: float = 0.0,
    converged: bool = True,
    threshold: float = 1e-6,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute every Phase-2 metric for a single (method, weights) pair.

    Returns a dict suitable for ``MethodResult(**...)``.
    """
    weights_raw = np.asarray(weights_raw, dtype=np.float64)

    # Predictions on raw test returns
    pred_test = inputs.X_test_raw @ weights_raw
    excess_test = pred_test - inputs.y_test

    oos_r2 = _r2(pred_test, inputs.y_test)
    oos_te = (
        float(excess_test.std(ddof=1) * np.sqrt(ANNUALIZATION_DAYS))
        if excess_test.size > 1
        else 0.0
    )
    oos_ir = (
        float(excess_test.mean() / excess_test.std(ddof=1) * np.sqrt(ANNUALIZATION_DAYS))
        if excess_test.std(ddof=1) > 1e-12 and excess_test.size > 1
        else 0.0
    )
    oos_sharpe = (
        float(pred_test.mean() / pred_test.std(ddof=1) * np.sqrt(ANNUALIZATION_DAYS))
        if pred_test.std(ddof=1) > 1e-12 and pred_test.size > 1
        else 0.0
    )

    # In-sample R² (uses raw train data when available; otherwise std → raw conversion)
    if inputs.X_train_raw is not None:
        pred_train = inputs.X_train_raw @ weights_raw
        in_r2 = _r2(pred_train, inputs.y_train)
    else:
        in_r2 = float("nan")

    return {
        "weights_raw": weights_raw,
        "n_active": n_active(weights_raw, threshold=threshold),
        "max_weight": float(weights_raw.max()),
        "hhi": herfindahl(weights_raw),
        "effective_n": effective_n(weights_raw),
        "fit_time_s": float(fit_time_s),
        "in_sample_r2": float(in_r2),
        "oos_r2": float(oos_r2),
        "oos_te_annual": float(oos_te),
        "oos_ir_annual": float(oos_ir),
        "oos_sharpe_annual": float(oos_sharpe),
        "converged": bool(converged),
        "extra": dict(extra or {}),
    }


# ---------------------------------------------------------------------------
# Method registration
# ---------------------------------------------------------------------------


@dataclass
class _MethodSpec:
    """Internal record of how to instantiate one method for the comparison."""

    name: str
    pretty_name: str
    color: str
    family: str  # 'convex', 'greedy', 'exact', 'naive'
    needs_market_caps: bool = False
    skip_if_solver_missing: bool = False


def _default_method_specs() -> list[_MethodSpec]:
    return [
        _MethodSpec("admm", "ADMM (ours)", "#22C55E", family="convex"),
        _MethodSpec("fista", "FISTA", "#06B6D4", family="convex"),
        _MethodSpec("lasso", "sklearn LASSO", "#3B82F6", family="convex"),
        _MethodSpec("omp", "OMP (greedy)", "#A855F7", family="greedy"),
        _MethodSpec("miqp", "MIQP (MOSEK)", "#F43F5E", family="exact", skip_if_solver_missing=True),
        _MethodSpec(
            "topn_cap", "Top-N market-cap", "#94A3B8", family="naive", needs_market_caps=True
        ),
        _MethodSpec(
            "equal_weight_topn",
            "Equal-weight top-N",
            "#64748B",
            family="naive",
            needs_market_caps=True,
        ),
        _MethodSpec("random_equal_weight", "Random N (ensemble)", "#CBD5E1", family="naive"),
    ]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class MethodComparison:
    """Runs every Phase-2 method on a single (X_train, y_train, X_test, y_test) split."""

    def __init__(self, config: ComparisonConfig | None = None):
        self.config = config or ComparisonConfig()
        self._specs = _default_method_specs()

    # ------------------------------------------------------------------ run
    def run(
        self,
        inputs: ComparisonInputs,
        *,
        verbose: bool = False,
    ) -> list[MethodResult]:
        cfg = self.config
        results: list[MethodResult] = []

        # Compute λ_max once on standardised training data
        lam_max = float(SparseTrackerADMM.compute_lambda_max(inputs.X_train_std, inputs.y_train))
        lam = cfg.lam_frac * lam_max
        if verbose:
            print(f"[comparison] λ_max={lam_max:.4f}, λ={lam:.4f}, K={cfg.K}")

        for spec in self._specs:
            if not self._should_run(spec, inputs):
                if verbose:
                    print(f"[comparison] skipping {spec.name}")
                continue
            if verbose:
                print(f"[comparison] running {spec.name}…")
            try:
                res = self._run_one(spec, inputs, lam=lam, K=cfg.K)
            except Exception as exc:  # pragma: no cover (defence in depth)
                if verbose:
                    print(f"[comparison] {spec.name} failed: {exc!r}")
                continue
            if verbose:
                print(
                    f"   {spec.pretty_name:24s}: "
                    f"nnz={res.n_active:3d} "
                    f"OOS R²={res.oos_r2:+.3f} "
                    f"TE={res.oos_te_annual:.4f} "
                    f"time={res.fit_time_s:.2f}s"
                )
            results.append(res)

        return results

    # --------------------------------------------------------- run dispatch
    def _should_run(self, spec: _MethodSpec, inputs: ComparisonInputs) -> bool:
        cfg = self.config
        if cfg.include_methods and spec.name not in cfg.include_methods:
            return False
        if spec.name in cfg.skip_methods:
            return False
        return not (spec.needs_market_caps and inputs.market_caps is None)

    def _run_one(
        self,
        spec: _MethodSpec,
        inputs: ComparisonInputs,
        *,
        lam: float,
        K: int,
    ) -> MethodResult:
        """Dispatch table: spec.name → solver factory + raw-weight extraction.

        Three weight-recovery paths exist depending on the method:

        * **Standardised path** (ADMM, LASSO, FISTA, OMP): we trained on
          ``X_train_std`` because the L1 / greedy machinery prefers
          equally-scaled columns. We then divide by ``sigma_train`` and
          re-simplex to recover tradeable raw-return weights.
        * **Native-raw path** (MIQP): MIQP enforces ``sum(w) = 1`` *during*
          optimisation, so we must train on raw features — otherwise the
          simplex constraint refers to standardised-feature units and post-hoc
          renormalisation distorts the optimum. (Verified empirically: MIQP
          on raw recovers the exact 8-stock support in 0.3 s; on standardised
          features the same problem is intractable.)
        * **No-fit path** (naive baselines): the weights are computed from
          market caps + universe identity and never see ``X`` or ``y``.
        """
        cfg = self.config

        # Build the solver
        solver: Any
        if spec.name == "admm":
            solver = SparseTrackerADMM(
                lam=lam, max_iter=cfg.admm_max_iter, tol=cfg.tol, verbose=False
            )
            mode = "std"
        elif spec.name == "fista":
            solver = FISTA(lam=lam, max_iter=cfg.fista_max_iter, tol=cfg.tol)
            mode = "std"
        elif spec.name == "lasso":
            solver = SklearnLassoSolver(lam=lam, max_iter=cfg.lasso_max_iter, tol=cfg.tol)
            mode = "std"
        elif spec.name == "omp":
            solver = OMPSolver(K=K)
            mode = "std"
        elif spec.name == "miqp":
            if inputs.X_train_raw is None:
                raise ValueError("MIQP requires X_train_raw in ComparisonInputs.")
            solver = MIQPSolver(
                K=K,
                time_limit=cfg.miqp_time_limit,
                mip_gap=cfg.miqp_mip_gap,
                enforce_simplex=True,
            )
            mode = "raw"
        elif spec.name == "topn_cap":
            assert inputs.market_caps is not None
            solver = TopNMarketCapSolver(N=K, market_caps=inputs.market_caps)
            mode = "naive"
        elif spec.name == "equal_weight_topn":
            assert inputs.market_caps is not None
            solver = EqualWeightTopNSolver(N=K, market_caps=inputs.market_caps)
            mode = "naive"
        elif spec.name == "random_equal_weight":
            solver = RandomEqualWeightSolver(
                N=K,
                p=inputs.p,
                seed=cfg.seed,
                ensemble=True,
                n_seeds=cfg.n_random_seeds,
            )
            mode = "naive"
        else:  # pragma: no cover
            raise ValueError(f"Unknown method: {spec.name}")

        # Fit
        t0 = time.perf_counter()
        if mode == "std":
            solver.fit(inputs.X_train_std, inputs.y_train)
        elif mode == "raw":
            solver.fit(inputs.X_train_raw, inputs.y_train)
        else:  # naive — args ignored
            solver.fit(inputs.X_train_std, inputs.y_train)
        elapsed = time.perf_counter() - t0
        converged = bool(getattr(solver, "converged", True))

        # Get raw simplex weights
        if mode == "std":
            try:
                w_std = solver.get_sparse_weights(threshold=cfg.threshold)
                w_raw = raw_to_simplex_via_std(
                    w_std,
                    inputs.sigma_train,
                    threshold=cfg.threshold,
                    raise_if_empty=False,
                )
            except ValueError:
                # Fully degenerate fit (e.g. λ ≥ λ_max → all zeros). Fall back to uniform.
                w_raw = to_simplex(np.ones(inputs.p), raise_if_empty=False)
        else:
            # Native-raw and naive baselines already produce raw simplex weights
            w_raw = solver.get_sparse_weights(threshold=cfg.threshold)

        metrics = compute_metrics(
            w_raw,
            inputs,
            fit_time_s=elapsed,
            converged=converged,
            threshold=cfg.threshold,
            extra={
                "family": spec.family,
                "lam": lam if mode == "std" else None,
                "K": K,
                "mode": mode,
            },
        )
        return MethodResult(name=spec.name, **metrics)


# ---------------------------------------------------------------------------
# Pareto sweep (cardinality vs OOS TE)
# ---------------------------------------------------------------------------


def pareto_sweep(
    inputs: ComparisonInputs,
    *,
    methods: Iterable[str] = (
        "admm",
        "lasso",
        "fista",
        "omp",
        "miqp",
        "topn_cap",
        "equal_weight_topn",
    ),
    K_grid: tuple[int, ...] = (5, 10, 20, 30, 50, 75, 100, 150),
    lam_grid_size: int = 20,
    lam_min_ratio: float = 1e-3,
    miqp_K_max: int = 50,
    miqp_time_limit: float = 30.0,
    threshold: float = 1e-6,
    verbose: bool = False,
) -> dict[str, list[dict[str, float]]]:
    """For each method, sweep its sparsity dial and return ``[(n_active, oos_te), …]``.

    For convex penalties (ADMM, LASSO, FISTA) we sweep λ on a log-spaced grid
    and let the solver decide the cardinality. For greedy / exact / naive
    baselines we sweep the explicit cardinality target K (or N).
    """
    lam_max = float(SparseTrackerADMM.compute_lambda_max(inputs.X_train_std, inputs.y_train))
    lam_grid = np.logspace(np.log10(lam_min_ratio * lam_max), np.log10(lam_max), lam_grid_size)

    out: dict[str, list[dict[str, float]]] = {}

    def _one_point(
        name: str, w_raw: NDArray[np.floating], lam: float | None, K: int | None, elapsed: float
    ) -> dict[str, float]:
        m = compute_metrics(w_raw, inputs, fit_time_s=elapsed, threshold=threshold)
        return {
            "n_active": float(m["n_active"]),
            "oos_te_annual": float(m["oos_te_annual"]),
            "oos_r2": float(m["oos_r2"]),
            "fit_time_s": float(m["fit_time_s"]),
            "lam": float(lam) if lam is not None else float("nan"),
            "K": int(K) if K is not None else -1,
        }

    convex_factories: dict[str, Callable[[float], Any]] = {
        "admm": lambda lam: SparseTrackerADMM(lam=lam, max_iter=3000, verbose=False),
        "lasso": lambda lam: SklearnLassoSolver(lam=lam, max_iter=20_000, tol=1e-7),
        "fista": lambda lam: FISTA(lam=lam, max_iter=3000, tol=1e-6),
    }

    for method in methods:
        points: list[dict[str, float]] = []
        if method in convex_factories:
            for lam in lam_grid:
                solver = convex_factories[method](float(lam))
                t0 = time.perf_counter()
                solver.fit(inputs.X_train_std, inputs.y_train)
                elapsed = time.perf_counter() - t0
                try:
                    w_std = solver.get_sparse_weights(threshold=threshold)
                    w_raw = raw_to_simplex_via_std(
                        w_std, inputs.sigma_train, threshold=threshold, raise_if_empty=False
                    )
                except ValueError:
                    continue
                points.append(_one_point(method, w_raw, float(lam), None, elapsed))
        elif method == "omp":
            for K in K_grid:
                solver = OMPSolver(K=K)
                t0 = time.perf_counter()
                solver.fit(inputs.X_train_std, inputs.y_train)
                elapsed = time.perf_counter() - t0
                try:
                    w_std = solver.get_sparse_weights(threshold=threshold)
                    w_raw = raw_to_simplex_via_std(
                        w_std, inputs.sigma_train, threshold=threshold, raise_if_empty=False
                    )
                except ValueError:
                    continue
                points.append(_one_point(method, w_raw, None, int(K), elapsed))
        elif method == "miqp":
            if inputs.X_train_raw is None:
                if verbose:
                    print("[pareto] miqp skipped — X_train_raw not provided")
                continue
            miqp_grid = [K for K in K_grid if K <= miqp_K_max]
            for K in miqp_grid:
                solver = MIQPSolver(K=K, time_limit=miqp_time_limit, enforce_simplex=True)
                t0 = time.perf_counter()
                try:
                    solver.fit(inputs.X_train_raw, inputs.y_train)
                except RuntimeError as exc:
                    if verbose:
                        print(f"[pareto] miqp K={K} skipped ({exc})")
                    continue
                elapsed = time.perf_counter() - t0
                if not solver.converged:
                    continue
                # Native-raw weights — no inverse standardisation needed.
                w_raw = solver.get_sparse_weights(threshold=threshold)
                points.append(_one_point(method, w_raw, None, int(K), elapsed))
        elif method in {"topn_cap", "equal_weight_topn"} and inputs.market_caps is not None:
            naive_solver: TopNMarketCapSolver | EqualWeightTopNSolver
            for N in K_grid:
                if method == "topn_cap":
                    naive_solver = TopNMarketCapSolver(N=int(N), market_caps=inputs.market_caps)
                else:
                    naive_solver = EqualWeightTopNSolver(N=int(N), market_caps=inputs.market_caps)
                t0 = time.perf_counter()
                naive_solver.fit(inputs.X_train_std, inputs.y_train)
                elapsed = time.perf_counter() - t0
                w_raw = naive_solver.get_sparse_weights(threshold=threshold)
                points.append(_one_point(method, w_raw, None, int(N), elapsed))
        else:
            continue

        # Sort each curve by sparsity for clean line plots
        points.sort(key=lambda p: p["n_active"])
        out[method] = points
        if verbose:
            print(f"[pareto] {method}: {len(points)} points")

    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_results(
    results: list[MethodResult],
    config: ComparisonConfig,
    out_path: Path,
    *,
    pareto: dict[str, list[dict[str, float]]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "config": asdict(config),
        "methods": [r.to_dict(include_weights=False) for r in results],
        "extra": extra or {},
    }
    if pareto is not None:
        payload["pareto"] = pareto
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    return out_path
