"""benchmarks/cvxpy_benchmark.py — ADMM vs CVXPY vs FISTA, with timing + plot.

Run with:

    make benchmark
    # or directly:
    python -m benchmarks.cvxpy_benchmark --p-grid 100 200 400 800 --n 120 --reps 3

Outputs (under ``benchmarks/_results/``):
    cvxpy_speedup.json     -- per-(p, method) timing/objective table
    cvxpy_speedup.png      -- log-log timing vs p, all 3 methods

The JSON is consumed by the README to populate the latency table.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from sit.paths import BENCHMARK_DIR
from sit.solvers.admm import SparseTrackerADMM
from sit.solvers.fista import fista_l1_nonneg
from sit.viz.style import apply_style

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None


@dataclass
class TimingRow:
    method: str
    n: int
    p: int
    k_true: int
    lam: float
    rep: int
    elapsed_s: float
    objective: float
    n_active: int


def _make_problem(rng: np.random.Generator, n: int, p: int, k: int, noise_sigma: float):
    X = rng.standard_normal((n, p))
    X /= X.std(axis=0) + 1e-12
    support = rng.choice(p, size=k, replace=False)
    w_true = np.zeros(p)
    w_true[support] = rng.uniform(0.5, 1.5, size=k)
    y = X @ w_true + noise_sigma * rng.standard_normal(n)
    return X, y, w_true, support


def _bench_admm(X, y, lam, *, max_iter, tol) -> TimingRow:
    solver = SparseTrackerADMM(
        lam=lam, max_iter=max_iter, tol=tol, adaptive_rho=True, verbose=False
    )
    t0 = time.perf_counter()
    solver.fit(X, y)
    elapsed = time.perf_counter() - t0
    obj = 0.5 * float(np.linalg.norm(X @ solver.z - y) ** 2) + lam * float(np.sum(solver.z))
    return TimingRow(
        method="admm",
        n=X.shape[0],
        p=X.shape[1],
        k_true=-1,
        lam=lam,
        rep=-1,
        elapsed_s=elapsed,
        objective=obj,
        n_active=int((solver.z > 1e-6).sum()),
    )


def _bench_fista(X, y, lam, *, max_iter, tol) -> TimingRow:
    t0 = time.perf_counter()
    result = fista_l1_nonneg(X, y, lam=lam, max_iter=max_iter, tol=tol)
    elapsed = time.perf_counter() - t0
    w = np.asarray(result["w"])
    obj = 0.5 * float(np.linalg.norm(X @ w - y) ** 2) + lam * float(np.sum(w))
    return TimingRow(
        method="fista",
        n=X.shape[0],
        p=X.shape[1],
        k_true=-1,
        lam=lam,
        rep=-1,
        elapsed_s=elapsed,
        objective=obj,
        n_active=int((w > 1e-6).sum()),
    )


def _bench_cvxpy(X, y, lam) -> TimingRow:
    assert cp is not None
    p = X.shape[1]
    w = cp.Variable(p, nonneg=True)
    objective = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + lam * cp.norm1(w))
    prob = cp.Problem(objective)
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL, verbose=False)
    elapsed = time.perf_counter() - t0
    w_val = np.asarray(w.value).ravel() if w.value is not None else np.zeros(p)
    obj = 0.5 * float(np.linalg.norm(X @ w_val - y) ** 2) + lam * float(np.sum(w_val))
    return TimingRow(
        method="cvxpy",
        n=X.shape[0],
        p=X.shape[1],
        k_true=-1,
        lam=lam,
        rep=-1,
        elapsed_s=elapsed,
        objective=obj,
        n_active=int((w_val > 1e-6).sum()),
    )


def run(args: argparse.Namespace) -> dict:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(args.seed)
    rows: list[TimingRow] = []

    for p in args.p_grid:
        for rep in range(args.reps):
            rng = np.random.default_rng(rng_master.integers(2**31))
            X, y, _w_true, _support = _make_problem(
                rng, n=args.n, p=p, k=args.k, noise_sigma=args.noise
            )
            lam_max = SparseTrackerADMM.compute_lambda_max(X, y)
            lam = args.lam_ratio * lam_max
            print(f"[{p=:>4d}, {rep=}/{args.reps - 1}]  lam={lam:.4f}  lam_max={lam_max:.4f}")

            # ADMM
            r_admm = _bench_admm(X, y, lam, max_iter=args.max_iter, tol=args.tol)
            r_admm.k_true = args.k
            r_admm.rep = rep
            rows.append(r_admm)
            print(
                f"   admm  : {r_admm.elapsed_s:.3f}s  obj={r_admm.objective:.6f}  nnz={r_admm.n_active}"
            )

            # FISTA
            r_fista = _bench_fista(X, y, lam, max_iter=args.max_iter, tol=args.tol)
            r_fista.k_true = args.k
            r_fista.rep = rep
            rows.append(r_fista)
            print(
                f"   fista : {r_fista.elapsed_s:.3f}s  obj={r_fista.objective:.6f}  nnz={r_fista.n_active}"
            )

            # CVXPY (skip when unavailable or when p is too large to be patient)
            if cp is not None and p <= args.cvxpy_max_p:
                r_cvxpy = _bench_cvxpy(X, y, lam)
                r_cvxpy.k_true = args.k
                r_cvxpy.rep = rep
                rows.append(r_cvxpy)
                print(
                    f"   cvxpy : {r_cvxpy.elapsed_s:.3f}s  obj={r_cvxpy.objective:.6f}  nnz={r_cvxpy.n_active}"
                )
            else:
                print("   cvxpy : skipped (p too large or cvxpy missing)")

    # Aggregate
    summary: dict[str, dict] = {}
    for r in rows:
        key = f"{r.method}_p{r.p}"
        bucket = summary.setdefault(key, {"elapsed_s": [], "objective": [], "n_active": []})
        bucket["elapsed_s"].append(r.elapsed_s)
        bucket["objective"].append(r.objective)
        bucket["n_active"].append(r.n_active)

    for key, bucket in summary.items():
        bucket["elapsed_s_mean"] = float(np.mean(bucket["elapsed_s"]))
        bucket["elapsed_s_std"] = float(np.std(bucket["elapsed_s"]))
        bucket["objective_mean"] = float(np.mean(bucket["objective"]))

    payload = {
        "config": vars(args),
        "rows": [asdict(r) for r in rows],
        "summary": summary,
    }
    out_json = BENCHMARK_DIR / "cvxpy_speedup.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n💾 wrote {out_json}")

    # Plot timing vs p
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return payload

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, marker, color in (
        ("admm", "o", "#22C55E"),
        ("fista", "s", "#06B6D4"),
        ("cvxpy", "D", "#F43F5E"),
    ):
        ps = []
        means = []
        stds = []
        for p in args.p_grid:
            key = f"{method}_p{p}"
            if key in summary:
                ps.append(p)
                means.append(summary[key]["elapsed_s_mean"])
                stds.append(summary[key]["elapsed_s_std"])
        if ps:
            ax.errorbar(
                ps,
                means,
                yerr=stds,
                marker=marker,
                color=color,
                linewidth=2,
                label=method.upper(),
                capsize=3,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of features p (n=120)")
    ax.set_ylabel("Wall-clock time (seconds)")
    ax.set_title("ADMM vs FISTA vs CVXPY — solve time vs problem size")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    out_png = BENCHMARK_DIR / "cvxpy_speedup.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 wrote {out_png}")

    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADMM vs CVXPY benchmark")
    parser.add_argument("--n", type=int, default=120, help="number of observations")
    parser.add_argument(
        "--p-grid",
        type=int,
        nargs="+",
        default=[100, 200, 300, 500, 800],
        help="grid of feature dimensions",
    )
    parser.add_argument("--k", type=int, default=15, help="true sparsity")
    parser.add_argument("--lam-ratio", type=float, default=0.05, help="lam / lam_max")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per p")
    parser.add_argument("--noise", type=float, default=0.01, help="noise σ")
    parser.add_argument("--max-iter", type=int, default=5000, help="solver max_iter")
    parser.add_argument("--tol", type=float, default=1e-6, help="solver tolerance")
    parser.add_argument("--cvxpy-max-p", type=int, default=600, help="skip CVXPY for p above this")
    parser.add_argument("--seed", type=int, default=20260511)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(args)


if __name__ == "__main__":
    main()
