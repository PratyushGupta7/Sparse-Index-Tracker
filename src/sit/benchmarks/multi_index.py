"""sit/benchmarks/multi_index.py — Phase-4 cross-index Phase-2 orchestrator.

Runs the full Phase-2 head-to-head pipeline (ADMM + LASSO + OMP + MIQP + naive
baselines) on multiple universes (S&P 500, Nasdaq-100, Russell 2000, Nifty 50)
back-to-back and emits a single cross-index summary JSON + two plots:

    plots/cross_index_summary.png   — grouped bars (R²/TE/Sharpe per universe)
    plots/cross_index_equity.png    — normalised equity curves (one per index)

Design
------
* Architecture mirrors :mod:`sit.benchmarks.comparison` — one orchestrator
  class, a config dataclass, a result dataclass, dispatch by universe name.
* Data plumbing goes through :mod:`sit.data.universes` so the loader is
  uniform across all four indices.
* Network access happens once per universe; failures are caught and reported
  in the metadata so a partial run still produces a useful artefact.
* The orchestrator is **completely synthetic-friendly**: pass a custom
  ``snapshot_factory`` that returns a ``ComparisonInputs`` per universe
  (used heavily in tests).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sit.benchmarks.comparison import (
    ComparisonConfig,
    ComparisonInputs,
    MethodComparison,
    MethodResult,
)
from sit.data.universes import INDEX_METADATA, supported_universes
from sit.paths import BENCHMARK_DIR, PLOTS_DIR

logger = logging.getLogger(__name__)


SnapshotFactory = Callable[..., "tuple[ComparisonInputs, dict[str, Any]]"]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MultiIndexConfig:
    """Configuration for the cross-index orchestrator.

    Attributes
    ----------
    indices
        Ordered list of universe names to evaluate. Must be a subset of
        :func:`sit.data.universes.supported_universes`.
    start_date, end_date
        Date range for the snapshot. Same semantics as
        :func:`sit.benchmarks.datasets.make_sp500_snapshot`.
    n_train, n_test
        Length of the train and held-out test windows in trading days.
    K
        Sparsity target shared across MIQP / OMP / Top-N baselines.
    lam_frac
        ADMM / LASSO regularisation (``λ = lam_frac × λ_max``).
    miqp_time_limit, miqp_mip_gap
        MOSEK MIQP solve time budget and gap target.
    skip_miqp
        Phase-4 default. Cross-index sweeps include four full pipelines
        each, so the MIQP time budget (≈ 1 minute / index) is usually
        skipped to keep the entire run under 5 minutes. Phase-2 already
        proves ADMM tracks MIQP within 0.3 pp R².
    """

    indices: list[str] = field(default_factory=lambda: list(supported_universes()))
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    n_train: int = 120
    n_test: int = 60
    K: int = 50
    lam_frac: float = 0.05
    miqp_time_limit: int = 60
    miqp_mip_gap: float = 1e-3
    skip_miqp: bool = True


@dataclass
class IndexRun:
    """A single completed (or skipped) per-universe run."""

    name: str
    benchmark: str
    label: str
    region: str
    n_active: int
    metadata: dict[str, Any]
    results: list[MethodResult]
    test_curves: dict[str, list[float]] = field(default_factory=dict)
    """Per-method (and ``"benchmark"``) hold-out daily-return paths.

    Stored as plain Python floats so the structure is JSON-serialisable.
    Used by :mod:`sit.benchmarks.multi_index_plots` to build the rebased
    equity panels without re-running the comparison pipeline.
    """

    error: str | None = None


@dataclass
class MultiIndexResult:
    """Cross-index orchestrator result."""

    config: MultiIndexConfig
    runs: dict[str, IndexRun]
    elapsed_s: float
    survivorship_bias_flag: bool = True

    def headline_table(self) -> dict[str, dict[str, float]]:
        """Compact ``{universe -> {metric -> value}}`` table for plotting."""
        out: dict[str, dict[str, float]] = {}
        for name, run in self.runs.items():
            if not run.results:
                continue
            admm = next((r for r in run.results if r.name == "admm"), None)
            if admm is None:
                continue
            out[name] = {
                "oos_r2": float(admm.oos_r2),
                "oos_te_annual": float(admm.oos_te_annual),
                "oos_sharpe_annual": float(admm.oos_sharpe_annual),
                "n_active": float(admm.n_active),
            }
        return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _default_snapshot_factory(
    universe: str, *, start_date: str, end_date: str, n_train: int, n_test: int
) -> tuple[ComparisonInputs, dict[str, Any]]:
    """Default factory: build a ``ComparisonInputs`` from yfinance prices."""
    from sit.benchmarks.datasets import _download_returns
    from sit.data.universes import get_universe

    tickers, benchmark = get_universe(universe)
    constituents, bench, active = _download_returns(tickers, benchmark, start_date, end_date)
    needed = n_train + n_test
    if len(constituents) < needed:
        raise RuntimeError(f"{universe}: got {len(constituents)} trading days, need {needed}.")
    constituents = constituents.iloc[:needed]
    bench = bench.iloc[:needed]
    train_raw = constituents.iloc[:n_train].to_numpy(dtype=np.float64)
    test_raw = constituents.iloc[n_train:].to_numpy(dtype=np.float64)
    y_train = bench.iloc[:n_train].to_numpy(dtype=np.float64)
    y_test = bench.iloc[n_train:].to_numpy(dtype=np.float64)
    sigma = train_raw.std(axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    inputs = ComparisonInputs(
        X_train_std=train_raw / sigma,
        y_train=y_train,
        X_test_raw=test_raw,
        y_test=y_test,
        sigma_train=sigma,
        X_train_raw=train_raw,
        market_caps=None,
        tickers=active,
    )
    metadata = {
        "source": "yfinance + universes registry",
        "universe": universe,
        "benchmark": benchmark,
        "n_active_tickers": len(active),
        "start_date": start_date,
        "end_date": end_date,
    }
    return inputs, metadata


class MultiIndexComparison:
    """Run the Phase-2 baseline panel across multiple universes."""

    def __init__(
        self,
        config: MultiIndexConfig | None = None,
        *,
        snapshot_factory: SnapshotFactory | None = None,
    ) -> None:
        self.config = config or MultiIndexConfig()
        self.snapshot_factory = snapshot_factory or _default_snapshot_factory

    def _comparison_config(self) -> ComparisonConfig:
        skip = ("miqp",) if self.config.skip_miqp else ()
        return ComparisonConfig(
            K=self.config.K,
            lam_frac=self.config.lam_frac,
            miqp_time_limit=self.config.miqp_time_limit,
            miqp_mip_gap=self.config.miqp_mip_gap,
            skip_methods=skip,
        )

    def run(self) -> MultiIndexResult:
        """Loop over universes, skipping any that fail with an error message."""
        t0 = time.perf_counter()
        cmp_cfg = self._comparison_config()
        runs: dict[str, IndexRun] = {}

        for name in self.config.indices:
            if name not in INDEX_METADATA:
                logger.warning("Unknown index %s — skipping", name)
                continue
            meta = INDEX_METADATA[name]
            logger.info("=== Running %s (%s) ===", name, meta["label"])
            try:
                inputs, snap_meta = self.snapshot_factory(
                    name,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    n_train=self.config.n_train,
                    n_test=self.config.n_test,
                )
            except Exception as exc:  # pragma: no cover (network)
                logger.warning("snapshot for %s failed: %r", name, exc)
                runs[name] = IndexRun(
                    name=name,
                    benchmark=meta["benchmark"],
                    label=meta["label"],
                    region=meta["region"],
                    n_active=0,
                    metadata={},
                    results=[],
                    error=repr(exc),
                )
                continue
            try:
                mc = MethodComparison(cmp_cfg)
                results = mc.run(inputs)
            except Exception as exc:
                logger.warning("comparison for %s failed: %r", name, exc)
                runs[name] = IndexRun(
                    name=name,
                    benchmark=meta["benchmark"],
                    label=meta["label"],
                    region=meta["region"],
                    n_active=snap_meta.get("n_active_tickers", 0),
                    metadata=snap_meta,
                    results=[],
                    error=repr(exc),
                )
                continue

            curves: dict[str, list[float]] = {
                "benchmark": [float(v) for v in inputs.y_test],
            }
            for r in results:
                curves[r.name] = [float(v) for v in (inputs.X_test_raw @ r.weights_raw)]

            runs[name] = IndexRun(
                name=name,
                benchmark=meta["benchmark"],
                label=meta["label"],
                region=meta["region"],
                n_active=snap_meta.get("n_active_tickers", 0),
                metadata=snap_meta,
                results=results,
                test_curves=curves,
            )
        elapsed = time.perf_counter() - t0
        return MultiIndexResult(config=self.config, runs=runs, elapsed_s=elapsed)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _result_to_jsonable(r: MethodResult) -> dict[str, Any]:
    return {
        "name": r.name,
        "in_sample_r2": float(r.in_sample_r2),
        "oos_r2": float(r.oos_r2),
        "oos_te_annual": float(r.oos_te_annual),
        "oos_ir_annual": float(r.oos_ir_annual),
        "oos_sharpe_annual": float(r.oos_sharpe_annual),
        "fit_time_s": float(r.fit_time_s),
        "n_active": int(r.n_active),
        "max_weight": float(r.max_weight),
        "hhi": float(r.hhi),
        "effective_n": float(r.effective_n),
    }


def save_multi_index_result(result: MultiIndexResult, path: Path) -> Path:
    """Persist a :class:`MultiIndexResult` to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": asdict(result.config),
        "elapsed_s": result.elapsed_s,
        "survivorship_bias_flag": bool(result.survivorship_bias_flag),
        "runs": {},
    }
    for name, run in result.runs.items():
        payload["runs"][name] = {
            "name": run.name,
            "benchmark": run.benchmark,
            "label": run.label,
            "region": run.region,
            "n_active": int(run.n_active),
            "metadata": run.metadata,
            "results": [_result_to_jsonable(r) for r in run.results],
            "test_curves": run.test_curves,
            "error": run.error,
        }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-4 cross-index Phase-2 orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--indices",
        default=",".join(supported_universes()),
        help="Comma-separated list of universes to evaluate.",
    )
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-test", type=int, default=60)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--lam-frac", type=float, default=0.05)
    p.add_argument("--include-miqp", action="store_true", help="Run MIQP per universe (slow).")
    p.add_argument(
        "--out-json",
        type=Path,
        default=BENCHMARK_DIR / "multi_index.json",
    )
    p.add_argument(
        "--summary-plot",
        type=Path,
        default=PLOTS_DIR / "cross_index_summary.png",
    )
    p.add_argument(
        "--equity-plot",
        type=Path,
        default=PLOTS_DIR / "cross_index_equity.png",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    cfg = MultiIndexConfig(
        indices=[s.strip() for s in args.indices.split(",") if s.strip()],
        start_date=args.start,
        end_date=args.end,
        n_train=args.n_train,
        n_test=args.n_test,
        K=args.K,
        lam_frac=args.lam_frac,
        skip_miqp=not args.include_miqp,
    )
    logger.info("config: %s", asdict(cfg))
    orch = MultiIndexComparison(cfg)
    result = orch.run()
    save_multi_index_result(result, args.out_json)
    logger.info("wrote %s", args.out_json)

    # Plots — defer import so the CLI works in headless test environments.
    from sit.benchmarks.multi_index_plots import (
        plot_cross_index_equity,
        plot_cross_index_summary,
    )

    args.summary_plot.parent.mkdir(parents=True, exist_ok=True)
    args.equity_plot.parent.mkdir(parents=True, exist_ok=True)
    plot_cross_index_summary(result, args.summary_plot)
    plot_cross_index_equity(result, args.equity_plot)
    logger.info("wrote %s + %s", args.summary_plot, args.equity_plot)

    print("\n=== Cross-index summary (ADMM, OOS) ===")
    headers = ["index", "benchmark", "n_active", "OOS R²", "OOS TE", "Sharpe"]
    print("  " + "  ".join(f"{h:>14s}" for h in headers))
    for name, run in result.runs.items():
        if not run.results:
            print(f"  {name:>14s}  (skipped: {run.error or 'no results'})")
            continue
        admm = next((r for r in run.results if r.name == "admm"), None)
        if admm is None:
            continue
        print(
            "  "
            + "  ".join(
                [
                    f"{name:>14s}",
                    f"{run.benchmark:>14s}",
                    f"{run.n_active:>14d}",
                    f"{admm.oos_r2:>14.4f}",
                    f"{admm.oos_te_annual:>14.4f}",
                    f"{admm.oos_sharpe_annual:>14.3f}",
                ]
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "IndexRun",
    "MultiIndexComparison",
    "MultiIndexConfig",
    "MultiIndexResult",
    "SnapshotFactory",
    "save_multi_index_result",
]
