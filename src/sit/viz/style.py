"""sit/viz/style.py — single source of truth for the project's visual identity.

Palette (locked in MASTER_ENHANCEMENT_PLAN.md §9):
  - deep-navy (#0B1220) for backgrounds / titles
  - electric-green (#22C55E) for the *sparse portfolio* lines
  - warm-gold (#F59E0B) for highlights / annotations
  - slate / white neutrals
  - regime accents (red bear, amber volatile, green bull, blue stable)
"""

from __future__ import annotations

from typing import Final

import matplotlib as mpl
import matplotlib.pyplot as plt

BRAND_PRIMARY: Final[str] = "#0B1220"  # deep navy
BRAND_SECONDARY: Final[str] = "#22C55E"  # electric green
BRAND_ACCENT: Final[str] = "#F59E0B"  # warm gold
BRAND_NEUTRAL: Final[str] = "#64748B"  # slate-500

BRAND_PALETTE: Final[dict[str, str]] = {
    "primary": BRAND_PRIMARY,
    "secondary": BRAND_SECONDARY,
    "accent": BRAND_ACCENT,
    "neutral": BRAND_NEUTRAL,
    "background": "#FFFFFF",
    "muted": "#F1F5F9",
    "border": "#E2E8F0",
    "text": "#0F172A",
    "text_muted": "#475569",
    # regime colours (kept consistent with the existing regime tester)
    "regime_bear": "#DC2626",
    "regime_volatile": "#F59E0B",
    "regime_bull": "#16A34A",
    "regime_stable": "#2563EB",
    # method colours (Phase 2 + Phase 3)
    "admm": "#22C55E",
    "lasso": "#3B82F6",
    "omp": "#A855F7",
    "miqp": "#F43F5E",
    "fista": "#06B6D4",
    "equal_weight": "#94A3B8",
    "spy": "#0F172A",
}


def apply_style(*, dark: bool = False) -> None:
    """Apply the project's matplotlib style. Idempotent; call once per script.

    Parameters
    ----------
    dark
        If ``True``, use a dark background suitable for embedding in the
        Next.js frontend's dark mode. Default light theme is print-friendly.
    """
    bg = BRAND_PRIMARY if dark else "#FFFFFF"
    fg = "#F8FAFC" if dark else BRAND_PALETTE["text"]
    grid = "#1E293B" if dark else BRAND_PALETTE["border"]

    cycle = mpl.cycler(
        color=[
            BRAND_PALETTE["admm"],
            BRAND_PALETTE["lasso"],
            BRAND_PALETTE["omp"],
            BRAND_PALETTE["miqp"],
            BRAND_PALETTE["fista"],
            BRAND_PALETTE["regime_bear"],
            BRAND_PALETTE["regime_stable"],
            BRAND_PALETTE["accent"],
        ]
    )

    plt.rcParams.update(
        {
            "figure.facecolor": bg,
            "axes.facecolor": bg,
            "savefig.facecolor": bg,
            "savefig.dpi": 150,
            "figure.dpi": 110,
            "axes.edgecolor": grid,
            "axes.labelcolor": fg,
            "axes.titlecolor": fg,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.prop_cycle": cycle,
            "axes.grid": True,
            "grid.color": grid,
            "grid.alpha": 0.35 if dark else 0.4,
            "grid.linewidth": 0.6,
            "xtick.color": fg,
            "ytick.color": fg,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": fg,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "cm",
            "figure.autolayout": False,
            "lines.linewidth": 2.0,
        }
    )
