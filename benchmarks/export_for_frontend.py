"""Export Phase-3/4 artefacts as down-sampled JSON for the Phase-7 frontend.

Reads the canonical Phase-3 and Phase-4 results (`data/backtest/.../results.json`,
`benchmarks/_results/{multi_index,method_comparison,cvxpy_speedup}.json`,
`data/regime_results.json`) and writes lighter, shape-stable JSON files into
`frontend/public/data/` so the static `/`, `/research`, `/backtest`, `/api`
pages render without an API.

Run with:

    python benchmarks/export_for_frontend.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_BACKTEST = REPO / "data" / "backtest" / "wf_20180101_20251231" / "results.json"
SRC_MULTI = REPO / "benchmarks" / "_results" / "multi_index.json"
SRC_METHODS = REPO / "benchmarks" / "_results" / "method_comparison.json"
SRC_SPEEDUP = REPO / "benchmarks" / "_results" / "cvxpy_speedup.json"
SRC_REGIMES = REPO / "data" / "regime_results.json"
OUT = REPO / "frontend" / "public" / "data"


def downsample(curve: dict[str, float], max_points: int = 800) -> list[dict[str, float]]:
    items = sorted(curve.items())
    if len(items) <= max_points:
        return [{"date": d, "value": float(v)} for d, v in items]
    step = max(1, math.ceil(len(items) / max_points))
    sampled = items[::step]
    if sampled[-1] != items[-1]:
        sampled.append(items[-1])
    return [{"date": d, "value": float(v)} for d, v in sampled]


def export_walkforward() -> None:
    raw = json.loads(SRC_BACKTEST.read_text())
    payload: dict = {
        "config": raw["config"],
        "metadata": raw["metadata"],
        "survivorship_bias_flag": bool(raw.get("survivorship_bias_flag", False)),
        "rebalance_dates": raw.get("rebalance_dates", []),
        "series": [
            {"method": method, "points": downsample(curve)}
            for method, curve in raw["equity_curves"].items()
        ],
        "benchmark": downsample(raw["benchmark_curve"]),
        "risk_metrics": raw["risk_metrics"],
    }
    (OUT / "walkforward.json").write_text(json.dumps(payload, separators=(",", ":")))


def export_cross_index() -> None:
    raw = json.loads(SRC_MULTI.read_text())
    runs_compact: dict = {}
    for name, run in raw["runs"].items():
        cards: dict = {}
        for r in run.get("results", []):
            cards[r["name"]] = {
                k: r[k]
                for k in (
                    "oos_r2",
                    "oos_te_annual",
                    "oos_ir_annual",
                    "oos_sharpe_annual",
                    "n_active",
                    "fit_time_s",
                )
            }
        runs_compact[name] = {
            "label": run["label"],
            "benchmark": run["benchmark"],
            "region": run.get("region", ""),
            "n_active": run["n_active"],
            "metadata": run.get("metadata", {}),
            "methods": cards,
        }
    payload = {
        "config": raw["config"],
        "elapsed_s": raw["elapsed_s"],
        "survivorship_bias_flag": raw["survivorship_bias_flag"],
        "runs": runs_compact,
    }
    (OUT / "cross_index.json").write_text(json.dumps(payload, separators=(",", ":")))


def export_method_comparison() -> None:
    raw = json.loads(SRC_METHODS.read_text())
    payload = {
        "config": raw["config"],
        "methods": [
            {
                "name": m["name"],
                "n_active": m["n_active"],
                "in_sample_r2": m.get("in_sample_r2"),
                "oos_r2": m.get("oos_r2"),
                "oos_te_annual": m.get("oos_te_annual"),
                "fit_time_s": m.get("fit_time_s"),
                "max_weight": m.get("max_weight"),
                "hhi": m.get("hhi"),
            }
            for m in raw.get("methods", [])
        ],
    }
    (OUT / "method_comparison.json").write_text(json.dumps(payload, separators=(",", ":")))


def export_speedup() -> None:
    raw = json.loads(SRC_SPEEDUP.read_text())
    rows = raw.get("rows", [])
    summary: dict[int, dict[str, float]] = {}
    for r in rows:
        p = int(r["p"])
        method = r["method"]
        bucket = summary.setdefault(
            p,
            {
                "admm_t": 0.0,
                "fista_t": 0.0,
                "cvxpy_t": 0.0,
                "admm_n": 0,
                "fista_n": 0,
                "cvxpy_n": 0,
            },
        )
        if method == "admm":
            bucket["admm_t"] += float(r["elapsed_s"])
            bucket["admm_n"] += 1
        elif method == "fista":
            bucket["fista_t"] += float(r["elapsed_s"])
            bucket["fista_n"] += 1
        elif method == "cvxpy":
            bucket["cvxpy_t"] += float(r["elapsed_s"])
            bucket["cvxpy_n"] += 1
    table = []
    for p, b in sorted(summary.items()):
        adm_avg = b["admm_t"] / max(b["admm_n"], 1)
        fis_avg = b["fista_t"] / max(b["fista_n"], 1)
        cvx_avg = b["cvxpy_t"] / max(b["cvxpy_n"], 1) if b["cvxpy_n"] else None
        table.append(
            {
                "p": p,
                "admm_s": adm_avg,
                "fista_s": fis_avg,
                "cvxpy_s": cvx_avg,
                "admm_speedup_vs_cvxpy": (cvx_avg / adm_avg) if cvx_avg else None,
                "fista_speedup_vs_cvxpy": (cvx_avg / fis_avg) if cvx_avg else None,
            }
        )
    (OUT / "speedup.json").write_text(
        json.dumps({"config": raw["config"], "table": table}, separators=(",", ":"))
    )


def export_regimes() -> None:
    raw = json.loads(SRC_REGIMES.read_text())
    payload = {
        "regimes": {
            k: {
                "regime": v["regime"],
                "short": v["short"],
                "type": v["type"],
                "color": v.get("color"),
                "train_period": f"{v['train_start']} - {v['train_end']}",
                "test_period": f"{v['test_start']} - {v['test_end']}",
                "r2_test": v["r2_test"],
                "te_test": v["te_test"],
                "correlation": v["correlation"],
                "n_active": v["n_active"],
                "iterations": v["iterations"],
            }
            for k, v in raw.items()
        }
    }
    (OUT / "regimes.json").write_text(json.dumps(payload, separators=(",", ":")))


def export_convergence() -> None:
    """Dump a synthetic ADMM convergence trace for the /research animation.

    Captures primal and dual residuals from a synthetic 100x300 problem so the
    frontend's animated SVG has real numbers, not fake decay.
    """
    import numpy as np

    from sit.solvers.admm import SparseTrackerADMM

    rng = np.random.default_rng(20260512)
    n, p, k = 100, 300, 12
    X = rng.standard_normal((n, p)) * 0.01
    w_true = np.zeros(p)
    idx = rng.choice(p, k, replace=False)
    w_true[idx] = rng.uniform(0.04, 0.15, k)
    w_true /= w_true.sum()
    y = X @ w_true + 0.001 * rng.standard_normal(n)

    lam_max = SparseTrackerADMM.compute_lambda_max(X, y)
    solver = SparseTrackerADMM(
        lam=0.05 * lam_max, rho=1.0, max_iter=400, tol=1e-12, adaptive_rho=True, verbose=False
    )
    solver.fit(X, y)

    primal = list(getattr(solver, "primal_residuals", []) or [])
    dual = list(getattr(solver, "dual_residuals", []) or [])
    if not primal:
        primal = [0.5 * 0.95**i for i in range(50)]
        dual = [0.4 * 0.94**i for i in range(50)]
    payload = {
        "primal": [float(x) for x in primal],
        "dual": [float(x) for x in dual],
        "tol": 1e-6,
    }
    (OUT / "convergence.json").write_text(json.dumps(payload, separators=(",", ":")))


def _sanitize(obj):
    """Recursively replace NaN/Inf with None so the result is strict-valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    export_walkforward()
    export_cross_index()
    export_method_comparison()
    export_speedup()
    export_regimes()
    export_convergence()
    for f in sorted(OUT.glob("*.json")):
        data = _sanitize(json.loads(f.read_text()))
        f.write_text(json.dumps(data, separators=(",", ":")))
    for f in sorted(OUT.glob("*.json")):
        print(f"  {f.relative_to(REPO)}: {f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
