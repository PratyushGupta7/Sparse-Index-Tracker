"""sit/benchmarks/plots.py — Phase-2 method-comparison visualisations.

Two plots:

* ``method_comparison.png`` — grouped bar chart with one bar per method on
  three axes (OOS R², OOS TE, fit time on log scale).
* ``sparsity_vs_te_pareto.png`` — Pareto frontier of (#stocks, OOS TE) with
  one curve per method.

Style matches the deep-navy + electric-green + warm-gold palette from
``sit.viz.style`` so all Phase-1 / Phase-2 plots share a visual identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

# Force a headless backend before pyplot is imported.
# This is essential when these plots are generated from a CI runner, the API
# server, or a pytest worker — none of which have a window manager.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sit.viz.style import BRAND_PALETTE, BRAND_PRIMARY, apply_style

if TYPE_CHECKING:
    from sit.benchmarks.comparison import MethodResult


# Per-method colour mapping (consistent across both plots).
# We extend the central BRAND_PALETTE so the two naive baselines share the
# slate-family with `equal_weight` already defined there.
_METHOD_COLORS = {
    "admm": BRAND_PALETTE["admm"],
    "fista": BRAND_PALETTE["fista"],
    "lasso": BRAND_PALETTE["lasso"],
    "omp": BRAND_PALETTE["omp"],
    "miqp": BRAND_PALETTE["miqp"],
    "topn_cap": BRAND_PALETTE["equal_weight"],  # slate-400
    "equal_weight_topn": "#64748B",  # slate-500
    "random_equal_weight": "#CBD5E1",  # slate-300
}

_METHOD_PRETTY = {
    "admm": "ADMM (ours)",
    "fista": "FISTA",
    "lasso": "sklearn LASSO",
    "omp": "OMP (greedy)",
    "miqp": "MIQP (MOSEK)",
    "topn_cap": "Top-N market-cap",
    "equal_weight_topn": "Equal-weight top-N",
    "random_equal_weight": "Random N (ensemble)",
}


def plot_method_comparison(
    results: list[MethodResult],
    out_path: Path,
    *,
    title: str = "Sparse Index Tracker — head-to-head (OOS, hold-out)",
    subtitle: str | None = None,
) -> Path:
    """Grouped bar chart: one row per metric (OOS R², OOS TE, fit time)."""
    apply_style(dark=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    names = [r.name for r in results]
    pretty = [_METHOD_PRETTY.get(n, n) for n in names]
    colors = [_METHOD_COLORS.get(n, "#777") for n in names]
    x = np.arange(len(names))

    # ----- (1) OOS R² -----
    r2s = [r.oos_r2 for r in results]
    bars = axes[0].bar(x, r2s, color=colors, edgecolor="white", linewidth=1.0)
    axes[0].set_ylabel("OOS R²", fontweight="bold")
    axes[0].axhline(0, color="white", linewidth=0.8, alpha=0.3)
    axes[0].set_title("Higher is better (1.0 = perfect tracking)", fontsize=10, color="#94A3B8")
    for b, v in zip(bars, r2s):
        axes[0].annotate(
            f"{v:+.3f}",
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 3 if v >= 0 else -10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="white",
        )

    # ----- (2) OOS Tracking Error (lower is better) -----
    tes = [r.oos_te_annual for r in results]
    bars = axes[1].bar(x, tes, color=colors, edgecolor="white", linewidth=1.0)
    axes[1].set_ylabel("OOS TE (annualised)", fontweight="bold")
    axes[1].set_title("Lower is better", fontsize=10, color="#94A3B8")
    for b, v in zip(bars, tes):
        axes[1].annotate(
            f"{v:.3f}",
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="white",
        )

    # ----- (3) Fit time (log-scale) -----
    times = np.maximum([r.fit_time_s for r in results], 1e-4)
    bars = axes[2].bar(x, times, color=colors, edgecolor="white", linewidth=1.0)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Fit time (s, log scale)", fontweight="bold")
    axes[2].set_title("Wall-clock time per fit", fontsize=10, color="#94A3B8")
    for b, v in zip(bars, times):
        axes[2].annotate(
            f"{v:.3g}s",
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="white",
        )

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(pretty, rotation=25, ha="right")

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    fig.suptitle(full_title, fontsize=14, fontweight="bold", y=0.995)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BRAND_PRIMARY)
    plt.close(fig)
    return out_path


def plot_pareto_frontier(
    pareto: dict[str, list[dict[str, float]]],
    out_path: Path,
    *,
    title: str = "Sparsity vs OOS Tracking Error — Pareto frontier",
    subtitle: str | None = None,
) -> Path:
    """Plot one curve per method on the (#active stocks, OOS TE) plane."""
    apply_style(dark=True)
    fig, ax = plt.subplots(figsize=(11, 7))

    for method, points in pareto.items():
        if not points:
            continue
        n_act = [pt["n_active"] for pt in points]
        te = [pt["oos_te_annual"] for pt in points]
        color = _METHOD_COLORS.get(method, "#777")
        marker = "o" if method in {"admm", "miqp"} else "s"
        lw = 2.4 if method == "admm" else 1.6
        ax.plot(
            n_act,
            te,
            marker=marker,
            markersize=6,
            color=color,
            linewidth=lw,
            label=_METHOD_PRETTY.get(method, method),
            alpha=0.95 if method in {"admm", "miqp"} else 0.8,
        )

    ax.set_xlabel("# active stocks", fontweight="bold")
    ax.set_ylabel("OOS tracking error (annualised)", fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)

    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    fig.suptitle(full_title, fontsize=14, fontweight="bold", y=0.99)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BRAND_PRIMARY)
    plt.close(fig)
    return out_path


__all__ = ["plot_method_comparison", "plot_pareto_frontier"]
