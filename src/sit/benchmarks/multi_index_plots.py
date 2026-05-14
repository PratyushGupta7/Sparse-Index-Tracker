"""sit/benchmarks/multi_index_plots.py — Phase-4 cross-index plots.

Two plots:

* ``cross_index_summary.png`` — grouped bar chart with one bar per universe
  on three axes (OOS R², OOS TE, OOS Sharpe).
* ``cross_index_equity.png`` — normalised hold-out equity curves (one per
  universe), all rebased to 100 at the start of the test window.

Style matches the deep-navy + electric-green + warm-gold palette from
:mod:`sit.viz.style` so all phase-X plots share a visual identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

# Force a headless backend before pyplot is imported (see Phase-2 plot module
# for the rationale).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sit.viz.style import BRAND_PALETTE, BRAND_PRIMARY, apply_style

if TYPE_CHECKING:
    from sit.benchmarks.multi_index import IndexRun, MultiIndexResult


_INDEX_COLOURS: dict[str, str] = {
    "sp500": BRAND_PALETTE["admm"],
    "nasdaq100": BRAND_PALETTE["lasso"],
    "russell2000": BRAND_PALETTE["accent"],
    "nifty50": BRAND_PALETTE["miqp"],
}


def _index_colour(name: str) -> str:
    return _INDEX_COLOURS.get(name, BRAND_PALETTE["neutral"])


def _index_label(run: IndexRun) -> str:
    return f"{run.label} ({run.benchmark})"


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------


def plot_cross_index_summary(result: MultiIndexResult, out_path: Path) -> Path:
    """Grouped bars: 4 universes × {OOS R², OOS TE, OOS Sharpe} for ADMM."""
    apply_style(dark=True)
    runs = [(name, run) for name, run in result.runs.items() if run.results]
    if not runs:
        raise ValueError("Cannot plot summary: no successful runs.")

    metrics = [
        ("oos_r2", "OOS R²", lambda r: r.oos_r2),
        ("oos_te_annual", "OOS Tracking Error", lambda r: r.oos_te_annual),
        ("oos_sharpe_annual", "OOS Sharpe", lambda r: r.oos_sharpe_annual),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=BRAND_PRIMARY)

    for ax, (_key, title, getter) in zip(axes, metrics):
        values: list[float] = []
        labels: list[str] = []
        colours: list[str] = []
        for name, run in runs:
            admm = next((r for r in run.results if r.name == "admm"), None)
            if admm is None:
                continue
            values.append(float(getter(admm)))
            labels.append(_index_label(run))
            colours.append(_index_colour(name))
        ax.bar(
            np.arange(len(values)),
            values,
            color=colours,
            edgecolor=BRAND_PALETTE["text"],
            linewidth=0.4,
        )
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_axisbelow(True)
        for i, v in enumerate(values):
            ax.text(
                i,
                v,
                f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=BRAND_PALETTE["text_muted"],
            )

    fig.suptitle(
        "Sparse Index Replication — cross-index ADMM summary",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        f"{result.config.start_date} - {result.config.end_date}  |  "
        f"K={result.config.K}, lam-frac={result.config.lam_frac}  |  "
        f"n_train={result.config.n_train}, n_test={result.config.n_test}",
        ha="center",
        fontsize=9,
        color=BRAND_PALETTE["text_muted"],
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BRAND_PRIMARY)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Equity curve overlay
# ---------------------------------------------------------------------------


def plot_cross_index_equity(result: MultiIndexResult, out_path: Path) -> Path:
    """Normalised equity curves (rebased to 100) — one panel per universe.

    Plots the ADMM portfolio against its native benchmark for each index in
    a 2×2 grid. Uses the per-method daily-return paths embedded in each
    ``IndexRun.test_curves`` so the plot can be regenerated from the JSON
    artefact alone, without rerunning the comparison.
    """
    apply_style(dark=True)
    runs = [
        (name, run)
        for name, run in result.runs.items()
        if run.results and run.test_curves and "admm" in run.test_curves
    ]
    if not runs:
        raise ValueError("Cannot plot equity: no successful runs with admm test curves.")

    n = len(runs)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(11, 3.4 * rows), facecolor=BRAND_PRIMARY, squeeze=False
    )

    flat_axes = axes.flatten()
    for ax, (name, run) in zip(flat_axes, runs):
        p_returns = np.asarray(run.test_curves["admm"], dtype=np.float64)
        b_returns = np.asarray(run.test_curves["benchmark"], dtype=np.float64)
        p_curve = 100.0 * np.cumprod(1.0 + p_returns)
        b_curve = 100.0 * np.cumprod(1.0 + b_returns)
        days = np.arange(len(p_curve))
        ax.plot(
            days, b_curve, color=BRAND_PALETTE["text_muted"], linewidth=1.6, label=run.benchmark
        )
        ax.plot(
            days,
            p_curve,
            color=_index_colour(name),
            linewidth=2.2,
            label="ADMM sparse replica",
        )
        ax.set_title(_index_label(run), fontweight="bold")
        ax.set_ylabel("NAV (rebased to 100)", fontsize=9)
        ax.set_xlabel("Hold-out trading day", fontsize=9)
        ax.legend(loc="best", fontsize=8)
        ax.set_axisbelow(True)

    for ax in flat_axes[len(runs) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Cross-index hold-out equity — ADMM vs native benchmark",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BRAND_PRIMARY)
    plt.close(fig)
    return out_path


__all__ = [
    "plot_cross_index_equity",
    "plot_cross_index_summary",
]
