"""sit/backtest/transaction_costs.py — pluggable cost models for the walk-forward backtest.

Three implementations ship in Phase 3, all conforming to a single
``TransactionCostModel`` Protocol so the orchestrator in
``sit.backtest.walkforward`` can swap them transparently:

* ``LinearCost(bps_per_side)`` — proportional to dollars-traded. The textbook
  industry default. ``bps_per_side = 5`` ⇒ 10 bps round-trip.
* ``SqrtImpactCost(kappa, daily_vol, participation_rate)`` — Almgren-style
  square-root market-impact. Use this when modelling a small-cap rebalance
  where dollars-per-name is non-trivial vs ADV.
* ``NoCost()`` — for ablation studies / as a control.

All three accept a *signed-trades-in-dollars* ``pd.Series`` (one entry per
ticker; positive = buy, negative = sell) and return the **total cost in
dollars** for that single rebalance. The orchestrator calls ``apply`` once
per rebalance date and subtracts the result from the next-period equity.

Why dollars and not weights?
----------------------------
A walk-forward backtest tracks an evolving portfolio NAV. Costs are expressed
in dollars precisely so that the orchestrator can subtract them from the NAV
in absolute terms and not have to renormalise weight-space arithmetic in the
middle of a rebalance.

References
----------
- Almgren, R. & Chriss, N. (2000). *Optimal execution of portfolio
  transactions.* J. of Risk.
- Tóth, B. et al. (2011). *Anomalous price impact and the critical nature of
  liquidity in financial markets.* PRX 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TransactionCostModel(Protocol):
    """Anything that can quote a dollar cost for a vector of signed trades."""

    def apply(self, trades_dollars: pd.Series) -> float: ...


# ---------------------------------------------------------------------------
# NoCost — control / ablation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoCost:
    """Frictionless trading. Returns zero cost for any trade vector."""

    name: str = "no_cost"

    def apply(self, trades_dollars: pd.Series) -> float:
        del trades_dollars
        return 0.0


# ---------------------------------------------------------------------------
# Linear cost — the industry default
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearCost:
    """Linear-in-volume transaction cost.

    Parameters
    ----------
    bps_per_side
        Cost in basis points (1 bp = 1e-4) charged on every dollar of
        *one-sided* trading volume. The conventional retail / small-fund
        round-trip cost is ``2 * bps_per_side`` since every rebalance both
        buys some names and sells others.

    Examples
    --------
    >>> import pandas as pd
    >>> cost = LinearCost(bps_per_side=5.0)              # 10 bps round-trip
    >>> trades = pd.Series({"AAPL": 1_000.0, "MSFT": -500.0, "GOOG": 0.0})
    >>> cost.apply(trades)                               # 5e-4 * (1000 + 500)
    0.75
    """

    bps_per_side: float = 5.0
    name: str = "linear"

    def __post_init__(self) -> None:
        if self.bps_per_side < 0:
            raise ValueError(f"bps_per_side must be >= 0 (got {self.bps_per_side}).")

    def apply(self, trades_dollars: pd.Series) -> float:
        if trades_dollars.empty:
            return 0.0
        notional = float(np.abs(trades_dollars.to_numpy(dtype=np.float64)).sum())
        return notional * self.bps_per_side * 1e-4


# ---------------------------------------------------------------------------
# Square-root impact — Almgren-style market-impact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SqrtImpactCost:
    """Almgren-style square-root market-impact cost (per name).

    For each traded ticker the cost in dollars is

        ``cost_i = κ · σ_daily,i · sqrt(participation_i) · |notional_i|``

    where ``participation_i = |notional_i| / ADV_i``. The ``√`` shape is the
    canonical empirical regularity (Almgren 2005, Tóth et al. 2011 …).

    This implementation expresses the per-name daily volatility and ADV via a
    single scalar ``daily_vol`` (e.g. an index-level number) plus a single
    ``participation_rate`` knob — sufficient for the ablation we do in Phase 3.
    The orchestrator can be wired with a per-ticker version later if desired.

    Parameters
    ----------
    kappa
        Dimensionless impact coefficient. Real-world numbers are O(0.5–1.5).
    daily_vol
        Average daily return volatility used as the price-impact scale (e.g.
        ``σ_train.mean()``). Annualised volatility is ``daily_vol * √252``.
    participation_rate
        Fraction of daily-volume that the rebalance consumes. Default 0.05
        (= 5 % of ADV).
    """

    kappa: float = 1.0
    daily_vol: float = 0.01
    participation_rate: float = 0.05
    name: str = "sqrt_impact"

    def __post_init__(self) -> None:
        if self.kappa < 0:
            raise ValueError(f"kappa must be >= 0 (got {self.kappa}).")
        if self.daily_vol < 0:
            raise ValueError(f"daily_vol must be >= 0 (got {self.daily_vol}).")
        if not 0 <= self.participation_rate <= 1:
            raise ValueError(
                f"participation_rate must lie in [0, 1] (got {self.participation_rate})."
            )

    def apply(self, trades_dollars: pd.Series) -> float:
        if trades_dollars.empty:
            return 0.0
        notional = float(np.abs(trades_dollars.to_numpy(dtype=np.float64)).sum())
        return self.kappa * self.daily_vol * math.sqrt(self.participation_rate) * notional


__all__ = [
    "LinearCost",
    "NoCost",
    "SqrtImpactCost",
    "TransactionCostModel",
]
