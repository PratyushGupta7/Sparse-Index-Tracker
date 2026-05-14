"""sit/backtest/plots.py — visualisations for the Phase-3 walk-forward backtest.

Four plots, all in the brand palette:

1. ``walkforward_equity.png`` — log-y equity curves with shaded drawdown
   underneath each line.
2. ``walkforward_drawdown.png`` — drawdown curves layered with fill_between.
3. ``walkforward_factor_loadings.png`` — rolling 90-day β to MKT / SMB / HML
   for the ADMM portfolio.
4. ``walkforward_turnover.png`` — bar chart of per-rebalance turnover with a
   dashed 95th-percentile line.

Style mirrors ``sit.benchmarks.plots`` so all phases share a visual identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

# Force a headless backend before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from sit.backtest.risk_metrics import drawdown, rolling_factor_betas
from sit.viz.style import BRAND_PALETTE, BRAND_PRIMARY, apply_style

if TYPE_CHECKING:
    from sit.backtest.walkforward import WalkForwardResult
    from sit.data.famafrench import FamaFrenchFactors

_METHOD_COLORS = {
    "admm": BRAND_PALETTE["admm"],
    "lasso": BRAND_PALETTE["lasso"],
    "fista": BRAND_PALETTE["fista"],
    "omp": BRAND_PALETTE["omp"],
    "miqp": BRAND_PALETTE["miqp"],
    "equal_weight_topn": BRAND_PALETTE["equal_weight"],
    "benchmark": BRAND_PALETTE["accent"],
}

_METHOD_PRETTY = {
    "admm": "ADMM (ours)",
    "lasso": "sklearn LASSO",
    "fista": "FISTA",
    "omp": "OMP (greedy)",
    "miqp": "MIQP (MOSEK)",
    "equal_weight_topn": "Equal-weight top-N",
    "benchmark": "Benchmark",
}


def _maybe_log_yaxis(ax) -> None:
    """Set log y-axis only if all data is positive."""
    ax.set_yscale("log")


def plot_walkforward_equity(
    result: WalkForwardResult,
    out_path: Path,
    *,
    title: str = "Walk-forward equity curves",
    subtitle: str | None = None,
    log_y: bool = True,
) -> Path:
    """Equity curves of every method on a shared time axis."""
    apply_style(dark=True)
    fig, ax = plt.subplots(figsize=(13, 7))

    for name, curve in result.equity_curves.items():
        color = _METHOD_COLORS.get(name, "#888")
        lw = 2.4 if name == "admm" else 1.6
        alpha = 0.95 if name in {"admm", "benchmark"} else 0.8
        ax.plot(
            curve.index,
            curve.to_numpy(),
            color=color,
            linewidth=lw,
            alpha=alpha,
            label=_METHOD_PRETTY.get(name, name),
        )

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("NAV (USD)", fontweight="bold")
    if log_y:
        _maybe_log_yaxis(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    fig.suptitle(full_title, fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BRAND_PRIMARY)
    plt.close(fig)
    return out_path


def plot_walkforward_drawdown(
    result: WalkForwardResult,
    out_path: Path,
    *,
    title: str = "Walk-forward drawdowns",
    subtitle: str | None = None,
) -> Path:
    """Drawdown comparison: layered ``fill_between`` style."""
    apply_style(dark=True)
    fig, ax = plt.subplots(figsize=(13, 7))

    for name, curve in result.equity_curves.items():
        dd = drawdown(curve)
        color = _METHOD_COLORS.get(name, "#888")
        alpha = 0.30 if name in {"admm", "benchmark"} else 0.18
        ax.fill_between(
            dd.index,
            dd.to_numpy(),
            0.0,
            color=color,
            alpha=alpha,
            label=_METHOD_PRETTY.get(name, name),
        )
        ax.plot(dd.index, dd.to_numpy(), color=color, linewidth=1.0, alpha=0.9)

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Drawdown", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="lower left", framealpha=0.9, ncol=2)

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    fig.suptitle(full_title, fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BRAND_PRIMARY)
    plt.close(fig)
    return out_path


def plot_walkforward_factor_loadings(
    result: WalkForwardResult,
    factors: FamaFrenchFactors,
    out_path: Path,
    *,
    method: str = "admm",
    window: int = 90,
    title: str = "Rolling 90-day Fama-French betas",
    subtitle: str | None = None,
) -> Path:
    """Rolling factor betas of one method's portfolio against FF3."""
    apply_style(dark=True)
    fig, ax = plt.subplots(figsize=(13, 7))

    eq = result.equity_curves[method]
    rets = eq.pct_change().dropna()
    betas = rolling_factor_betas(rets, factors, window=window).dropna()

    factor_colors = {
        "beta_mkt": BRAND_PALETTE["secondary"],
        "beta_smb": BRAND_PALETTE["accent"],
        "beta_hml": BRAND_PALETTE["regime_stable"],
    }
    factor_pretty = {
        "beta_mkt": "β · MKT-RF",
        "beta_smb": "β · SMB",
        "beta_hml": "β · HML",
    }
    for col, color in factor_colors.items():
        ax.plot(
            betas.index,
            betas[col].to_numpy(),
            color=color,
            linewidth=2.0,
            label=factor_pretty[col],
        )
    ax.axhline(0.0, color="white", linewidth=0.6, alpha=0.4)
    ax.axhline(1.0, color="white", linewidth=0.6, alpha=0.4, linestyle="--")
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Rolling β", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9, ncol=3)

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    fig.suptitle(full_title, fontsize=14, fontweight="bold", y=0.98)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BRAND_PRIMARY)
    plt.close(fig)
    return out_path


def plot_walkforward_turnover(
    result: WalkForwardResult,
    out_path: Path,
    *,
    method: str = "admm",
    title: str = "Per-rebalance turnover",
    subtitle: str | None = None,
) -> Path:
    """Bar chart of per-rebalance turnover with a dashed 95th-percentile line."""
    apply_style(dark=True)
    fig, ax = plt.subplots(figsize=(13, 5))

    series = result.turnover.get(method)
    if series is None or series.empty:
        ax.text(0.5, 0.5, f"No turnover data for {method!r}", ha="center", va="center")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BRAND_PRIMARY)
        plt.close(fig)
        return out_path

    values = series.to_numpy(dtype=float)
    p95 = float(np.quantile(values, 0.95))
    color = _METHOD_COLORS.get(method, BRAND_PALETTE["admm"])
    ax.bar(series.index, values, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axhline(
        p95,
        color=BRAND_PALETTE["accent"],
        linewidth=1.5,
        linestyle="--",
        label=f"95th pct = {p95:.1%}",
    )
    ax.set_ylabel("Turnover (½·L1 of Δw)", fontweight="bold")
    ax.set_xlabel("Rebalance date", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right", framealpha=0.9)

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    fig.suptitle(full_title, fontsize=14, fontweight="bold", y=0.99)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BRAND_PRIMARY)
    plt.close(fig)
    return out_path


__all__ = [
    "plot_walkforward_drawdown",
    "plot_walkforward_equity",
    "plot_walkforward_factor_loadings",
    "plot_walkforward_turnover",
]
