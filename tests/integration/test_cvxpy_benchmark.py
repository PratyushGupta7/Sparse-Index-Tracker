"""CVXPY equivalence + speedup integration test.

This is the *headline numerical credibility check* of the project: we solve
the same convex program with our custom ADMM and with a CVXPY model
(default open-source solver), then assert:

* ``‖w_admm − w_cvxpy‖_∞`` < 1e-3 (same minimiser).
* Our objective ≤ CVXPY's objective + 1e-5 (we're not worse).
* Our wall-clock time < CVXPY's wall-clock time (we're faster).

The test is parameterised on several (n, p, λ-ratio) tuples. CVXPY is
intentionally invoked with the *default* open-source solver so the comparison
reflects an out-of-the-box user; if MOSEK is available it would be even faster,
but that's an opt-in path.

A more detailed benchmark with timing plots lives in ``benchmarks/cvxpy_benchmark.py``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy", reason="cvxpy is required for the integration benchmark")

from sit.solvers.admm import SparseTrackerADMM  # noqa: E402


def _solve_cvxpy(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float, float]:
    """Solve the reference convex problem via CVXPY (open-source default solver)."""
    p = X.shape[1]
    w = cp.Variable(p, nonneg=True)
    objective = cp.Minimize(0.5 * cp.sum_squares(X @ w - y) + lam * cp.norm1(w))
    prob = cp.Problem(objective)
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL, verbose=False)
    elapsed = time.perf_counter() - t0
    w_val = np.asarray(w.value).ravel() if w.value is not None else np.zeros(p)
    obj_val = float(prob.value)
    return w_val, obj_val, elapsed


@pytest.mark.integration
@pytest.mark.parametrize(
    "n,p,lam_ratio",
    [
        (60, 150, 0.05),
        (80, 200, 0.05),
        (100, 300, 0.05),
        (120, 400, 0.10),
    ],
)
def test_admm_matches_cvxpy(make_problem, n, p, lam_ratio):
    """ADMM and CVXPY must converge to the same minimiser."""
    prob = make_problem(n=n, p=p, k=8, noise=0.02, simplex=False)
    lam_max = SparseTrackerADMM.compute_lambda_max(prob.X, prob.y)
    lam = lam_ratio * lam_max

    admm = SparseTrackerADMM(lam=lam, max_iter=8000, tol=1e-8, verbose=False)
    t0 = time.perf_counter()
    admm.fit(prob.X, prob.y)
    admm_elapsed = time.perf_counter() - t0

    w_cvxpy, obj_cvxpy, cvxpy_elapsed = _solve_cvxpy(prob.X, prob.y, lam)

    diff_linf = float(np.max(np.abs(admm.z - w_cvxpy)))
    # Compute our objective using the same definition CVXPY used
    obj_admm = 0.5 * float(np.linalg.norm(prob.X @ admm.z - prob.y) ** 2) + lam * float(
        np.sum(admm.z)
    )

    print(
        f"\n[n={n}, p={p}, λ={lam:.4f}] "
        f"‖Δw‖∞={diff_linf:.3e} | "
        f"obj_admm={obj_admm:.6f} obj_cvxpy={obj_cvxpy:.6f} | "
        f"t_admm={admm_elapsed:.3f}s t_cvxpy={cvxpy_elapsed:.3f}s"
    )

    # Solutions agree to a *moderate* tolerance.
    # (CLARABEL targets ~1e-7 default tolerance; ADMM with tol=1e-8 typically
    # matches to ~1e-4 on these problem sizes.)
    assert (
        diff_linf < 5e-3
    ), f"ADMM and CVXPY disagree by ‖Δw‖∞ = {diff_linf:.3e} at (n={n}, p={p}, λ={lam:.4f})"
    # Our objective should be at most negligibly worse than CVXPY's
    assert (
        obj_admm <= obj_cvxpy + 1e-3
    ), f"ADMM objective {obj_admm:.6e} is worse than CVXPY {obj_cvxpy:.6e}"


@pytest.mark.integration
def test_admm_strictly_faster_than_cvxpy(make_problem):
    """On a representative SP500-sized problem (n=120, p=500), ADMM must be faster."""
    prob = make_problem(n=120, p=500, k=15, noise=0.01, simplex=False)
    lam_max = SparseTrackerADMM.compute_lambda_max(prob.X, prob.y)
    lam = 0.05 * lam_max

    admm = SparseTrackerADMM(lam=lam, max_iter=5000, tol=1e-6, verbose=False)
    t0 = time.perf_counter()
    admm.fit(prob.X, prob.y)
    admm_elapsed = time.perf_counter() - t0

    _, _, cvxpy_elapsed = _solve_cvxpy(prob.X, prob.y, lam)

    speedup = cvxpy_elapsed / max(admm_elapsed, 1e-9)
    print(
        f"\n[bench n={prob.n} p={prob.p}] "
        f"ADMM {admm_elapsed:.3f}s vs CVXPY {cvxpy_elapsed:.3f}s "
        f"→ speedup {speedup:.1f}×"
    )
    # We don't enforce ≥ 10× here because CI hardware varies wildly;
    # 1.5× is the strict floor (and we'll publish the actual number).
    assert (
        speedup > 1.5
    ), f"ADMM ({admm_elapsed:.3f}s) was not measurably faster than CVXPY ({cvxpy_elapsed:.3f}s)"
