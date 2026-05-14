"""benchmarks/method_comparison.py — Phase-2 head-to-head CLI.

Runs every Phase-2 baseline on a single train/test split and emits:

  benchmarks/_results/method_comparison.json
  plots/method_comparison.png
  plots/sparsity_vs_te_pareto.png

Examples
--------
Quick synthetic smoke run (no network) ::

    python -m benchmarks.method_comparison --data synthetic --K 8

Real S&P 500 snapshot (downloads 502 tickers via yfinance) ::

    python -m benchmarks.method_comparison --data sp500 --K 50 \\
        --start 2025-04-01 --end 2026-03-09 --pareto

Skip MIQP if MOSEK is unavailable ::

    python -m benchmarks.method_comparison --data sp500 --K 50 --skip miqp
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sit.benchmarks import (  # noqa: E402
    ComparisonConfig,
    MethodComparison,
    make_sp500_snapshot,
    make_synthetic_dataset,
    pareto_sweep,
    plot_method_comparison,
    plot_pareto_frontier,
    save_results,
)
from sit.paths import BENCHMARK_DIR, PLOTS_DIR  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-2 head-to-head method comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", choices=("synthetic", "sp500"), default="synthetic", help="Data source."
    )
    p.add_argument(
        "--K", type=int, default=50, help="Cardinality target for OMP/MIQP/naive baselines."
    )
    p.add_argument(
        "--lam-frac",
        type=float,
        default=0.05,
        help="ADMM/FISTA/LASSO penalty as fraction of λ_max.",
    )
    p.add_argument(
        "--miqp-time-limit", type=float, default=60.0, help="Wall-clock cap on MIQP per-fit (s)."
    )
    p.add_argument("--miqp-mip-gap", type=float, default=1e-3, help="MIQP relative MIP gap.")
    p.add_argument(
        "--n-random-seeds",
        type=int,
        default=100,
        help="Seeds for the random-equal-weight ensemble.",
    )
    p.add_argument("--include", default=None, help="Comma-separated method names to *only* run.")
    p.add_argument("--skip", default="", help="Comma-separated method names to skip (e.g. 'miqp').")

    p.add_argument(
        "--pareto",
        action="store_true",
        help="Also generate the (#stocks, OOS TE) Pareto sweep + plot.",
    )
    p.add_argument(
        "--pareto-K-grid",
        default="5,10,20,30,50,75,100,150",
        help="Cardinality grid for the Pareto sweep.",
    )
    p.add_argument(
        "--pareto-lam-points",
        type=int,
        default=20,
        help="Number of λ points on the convex Pareto curves.",
    )
    p.add_argument(
        "--miqp-K-max",
        type=int,
        default=50,
        help="Maximum K to attempt with MIQP in the Pareto sweep.",
    )

    # Synthetic-specific
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-test", type=int, default=60)
    p.add_argument("--p", type=int, default=200, help="Synthetic universe size.")
    p.add_argument("--k-true", type=int, default=8, help="True synthetic sparsity.")
    p.add_argument("--seed", type=int, default=20260511)

    # SP500-specific
    p.add_argument("--start", default="2025-04-01")
    p.add_argument("--end", default="2026-03-09")
    p.add_argument("--benchmark", default="SPY")

    p.add_argument("--out-json", type=Path, default=None, help="Override the JSON output location.")
    p.add_argument(
        "--out-bar", type=Path, default=None, help="Override the bar-chart output location."
    )
    p.add_argument(
        "--out-pareto", type=Path, default=None, help="Override the Pareto-plot output location."
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _build_inputs(args: argparse.Namespace, *, verbose: bool) -> tuple:
    if args.data == "synthetic":
        if verbose:
            print(
                f"[data] synthetic: n_train={args.n_train} n_test={args.n_test} "
                f"p={args.p} k_true={args.k_true} seed={args.seed}"
            )
        inputs, _ = make_synthetic_dataset(
            n_train=args.n_train,
            n_test=args.n_test,
            p=args.p,
            k=args.k_true,
            seed=args.seed,
        )
        meta = {
            "data": "synthetic",
            "n_train": args.n_train,
            "n_test": args.n_test,
            "p": args.p,
            "k_true": args.k_true,
            "seed": args.seed,
        }
        return inputs, meta

    if verbose:
        print(
            f"[data] sp500: start={args.start} end={args.end} "
            f"n_train={args.n_train} n_test={args.n_test} benchmark={args.benchmark}"
        )
    inputs, meta = make_sp500_snapshot(
        start_date=args.start,
        end_date=args.end,
        n_train=args.n_train,
        n_test=args.n_test,
        benchmark=args.benchmark,
    )
    return inputs, {"data": "sp500", **meta}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    verbose = not args.quiet

    # Make matplotlib happy in sandboxes
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

    inputs, dataset_meta = _build_inputs(args, verbose=verbose)
    if verbose:
        print(
            f"[data] X_train_std shape={inputs.X_train_std.shape} "
            f"X_test_raw shape={inputs.X_test_raw.shape}"
        )

    skip = tuple(s.strip() for s in args.skip.split(",") if s.strip())
    include = (
        tuple(s.strip() for s in args.include.split(",") if s.strip()) if args.include else None
    )
    cfg = ComparisonConfig(
        K=args.K,
        lam_frac=args.lam_frac,
        miqp_time_limit=args.miqp_time_limit,
        miqp_mip_gap=args.miqp_mip_gap,
        n_random_seeds=args.n_random_seeds,
        seed=args.seed,
        include_methods=include,
        skip_methods=skip,
    )

    if verbose:
        print(f"\n[run] head-to-head with K={cfg.K}, λ-frac={cfg.lam_frac}")
    t0 = time.perf_counter()
    mc = MethodComparison(cfg)
    results = mc.run(inputs, verbose=verbose)
    if verbose:
        print(f"[run] head-to-head finished in {time.perf_counter() - t0:.1f}s")

    pareto = None
    if args.pareto:
        K_grid = tuple(int(s) for s in args.pareto_K_grid.split(","))
        pareto_methods = ("admm", "lasso", "fista", "omp", "miqp", "topn_cap", "equal_weight_topn")
        if skip:
            pareto_methods = tuple(m for m in pareto_methods if m not in skip)
        if include:
            pareto_methods = tuple(m for m in pareto_methods if m in include)

        if verbose:
            print(f"\n[pareto] sweeping {len(pareto_methods)} methods over K={K_grid}")
        t0 = time.perf_counter()
        pareto = pareto_sweep(
            inputs,
            methods=pareto_methods,
            K_grid=K_grid,
            lam_grid_size=args.pareto_lam_points,
            miqp_K_max=args.miqp_K_max,
            miqp_time_limit=min(args.miqp_time_limit, 30),
            verbose=verbose,
        )
        if verbose:
            print(f"[pareto] finished in {time.perf_counter() - t0:.1f}s")

    out_json = args.out_json or (BENCHMARK_DIR / "method_comparison.json")
    out_bar = args.out_bar or (PLOTS_DIR / "method_comparison.png")
    out_pareto = args.out_pareto or (PLOTS_DIR / "sparsity_vs_te_pareto.png")

    save_results(results, cfg, out_json, pareto=pareto, extra={"dataset": dataset_meta})
    if verbose:
        print(f"\n[out] JSON  → {out_json}")

    plot_method_comparison(
        results,
        out_bar,
        subtitle=(
            f"{dataset_meta.get('data', 'data')} | "
            f"K={cfg.K} | λ-frac={cfg.lam_frac} | "
            f"n_train={inputs.n_train}, n_test={inputs.n_test}, p={inputs.p}"
        ),
    )
    if verbose:
        print(f"[out] bar   → {out_bar}")

    if pareto is not None:
        plot_pareto_frontier(
            pareto,
            out_pareto,
            subtitle=(
                f"{dataset_meta.get('data', 'data')} | "
                f"sweep K∈{args.pareto_K_grid} | λ-grid={args.pareto_lam_points} | "
                f"n_train={inputs.n_train}, p={inputs.p}"
            ),
        )
        if verbose:
            print(f"[out] pareto→ {out_pareto}")

    if verbose:
        print("\n=== Summary ===")
        for r in results:
            print(
                f"  {r.name:22s}: nnz={r.n_active:3d}  OOS R²={r.oos_r2:+.4f}  "
                f"TE={r.oos_te_annual:.4f}  IR={r.oos_ir_annual:+.2f}  "
                f"max_w={r.max_weight:.3f}  HHI={r.hhi:.3f}  t={r.fit_time_s:.2f}s"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
