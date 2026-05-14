"""λ-path service: cached regularization path for the interactive frontend slider.

On first call we run a 12-point ``lam_frac`` sweep on a small synthetic
snapshot of the index (using the universes registry) and cache the result
to ``data/lambda_paths/<index>.json``. Subsequent calls return the cached
JSON so the slider stays responsive even on free-tier dynos.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from sit.api.settings import get_settings
from sit.solvers.admm import SparseTrackerADMM

logger = logging.getLogger(__name__)


DEFAULT_LAM_FRAC_GRID = (
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
)


def _path_file(index: str) -> Path:
    settings = get_settings()
    return settings.data_dir / "lambda_paths" / f"{index}.json"


def _load_disk(index: str) -> dict[str, Any] | None:
    path = _path_file(index)
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return dict(json.load(f))
    except Exception:
        return None


def _save_disk(index: str, payload: dict[str, Any]) -> None:
    path = _path_file(index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(payload, f)
    tmp.replace(path)


def _build_synthetic_path(
    n_train: int = 120,
    n_test: int = 60,
    p: int = 80,
    k_true: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic train/test split — used as fallback when offline."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = rng.standard_normal((n, p)) * 0.01
    w_true = np.zeros(p)
    idx = rng.choice(p, size=k_true, replace=False)
    w_true[idx] = rng.uniform(0.04, 0.15, size=k_true)
    w_true /= w_true.sum()
    y = X @ w_true + 0.001 * rng.standard_normal(n)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _build_real_path(
    index: str, n_train: int = 120, n_test: int = 60
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Real-data train/test split via yfinance. Falls back on synthetic on failure."""
    import pandas as pd
    import yfinance as yf

    from sit.data.universes import get_universe

    tickers, benchmark = get_universe(index)
    tickers = [t for t in tickers if t != benchmark][:120]
    end = datetime.today()
    start = end - timedelta(days=(n_train + n_test) * 2 + 30)

    bench = yf.download(
        benchmark,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )
    const = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    bench_prices = bench.get("Adj Close", bench["Close"])
    const_prices = const["Close"]
    by = bench_prices.pct_change().dropna()
    cy = const_prices.pct_change().dropna(how="all")
    merged = pd.merge(by.rename(benchmark), cy, left_index=True, right_index=True, how="inner")
    merged = merged.dropna(axis=1)
    if merged.shape[0] < n_train + n_test:
        raise RuntimeError("not enough trading days")
    df = merged.tail(n_train + n_test)
    cols = [c for c in df.columns if c != benchmark]
    y = np.asarray(df[benchmark].values, dtype=np.float64)
    X = np.asarray(df[cols].values, dtype=np.float64)
    var = X.var(axis=0)
    keep = var > 1e-12
    X = X[:, keep]
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def compute_path(index: str, *, prefer_real: bool = False) -> dict[str, Any]:
    """Compute (or return cached) λ-path JSON for ``index``."""
    cached = _load_disk(index)
    if cached is not None:
        cached["cached"] = True
        return cached

    try:
        if prefer_real:
            X_tr, y_tr, X_te, y_te = _build_real_path(index)
        else:
            X_tr, y_tr, X_te, y_te = _build_synthetic_path()
    except Exception as exc:
        logger.warning("Falling back to synthetic λ-path data (%s).", exc)
        X_tr, y_tr, X_te, y_te = _build_synthetic_path()

    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    X_tr_std = (X_tr - mu) / sigma

    lam_max = SparseTrackerADMM.compute_lambda_max(X_tr_std, y_tr)

    points: list[dict[str, Any]] = []
    for frac in DEFAULT_LAM_FRAC_GRID:
        solver = SparseTrackerADMM(
            lam=float(frac * lam_max),
            rho=1.0,
            max_iter=3000,
            tol=1e-6,
            adaptive_rho=True,
            verbose=False,
        )
        solver.fit(X_tr_std, y_tr)
        w = solver.get_raw_weights(sigma)
        nnz = int(np.sum(w > 1e-6))
        port_in = X_tr @ w
        ss_res = float(np.sum((y_tr - port_in) ** 2))
        ss_tot = float(np.sum((y_tr - y_tr.mean()) ** 2))
        r2_in = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        port_oos = X_te @ w
        oos_te = float(np.std(port_oos - y_te) * np.sqrt(252))
        points.append(
            {
                "lam": float(frac * lam_max),
                "lam_frac": float(frac),
                "nnz": nnz,
                "in_sample_r2": float(r2_in),
                "oos_te": oos_te,
            }
        )

    payload: dict[str, Any] = {
        "index": index,
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "universe_size": int(X_tr.shape[1]),
        "points": points,
        "cached": False,
    }
    _save_disk(index, payload)
    return payload
