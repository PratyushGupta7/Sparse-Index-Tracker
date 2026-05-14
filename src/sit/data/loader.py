"""sit/data/loader.py — generic index-aware data pipeline.

Originally Phase-1 shipped a hand-rolled :class:`SP500DataLoader` that did
six things in one class: scrape Wikipedia for tickers, download adjusted-
close prices, compute returns, align dates, drop bad columns, and
standardise. Phase 4 needs the same pipeline for *any* index (S&P 500,
Nasdaq-100, Russell 2000, Nifty 50), so we factor the universe-fetching
step out into :mod:`sit.data.universes` and expose a single
:class:`IndexDataLoader` that takes a "universe function" plus a benchmark
ticker.

The legacy :class:`SP500DataLoader` is preserved as a thin back-compat
alias so existing scripts (``app.py``, the regime tester) keep working
unchanged.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

# Importing yfinance can be slow in test environments; do it lazily inside
# methods that actually need the network. Top-level imports stay fast for
# the unit-test path.

warnings.filterwarnings("ignore")


UniverseFn = Callable[..., "tuple[list[str], str]"]


def _default_sp500_universe() -> tuple[list[str], str]:
    """Default universe: today's S&P 500 (SPY benchmark).

    Late-binding the import so ``sit.data.loader`` can be imported in
    network-free environments.
    """
    from sit.data.universes import sp500

    return sp500()


class IndexDataLoader:
    """Generic index-aware Phase-1 pipeline.

    Pulls a constituent list + benchmark ticker via *universe_fn*, downloads
    adjusted-close prices for both, computes simple daily returns, aligns
    on the inner-join date index, drops constituents with any missing
    history (no forward-fill, ever), slices to the last ``n_days`` rows,
    and standardises ``X`` per-column for L1 fairness.

    Parameters
    ----------
    universe_fn
        Callable returning ``(constituents, benchmark_ticker)``. See
        :mod:`sit.data.universes` for the four shipped factories.
    n_days
        Final training-window length in trading days (default 120). The
        loader will raise ``ValueError`` if the available window is shorter.
    benchmark
        Optional override for the benchmark ticker. Defaults to whatever
        ``universe_fn`` returns (which is the right answer in 99 % of
        cases).
    start_date, end_date
        Date bounds passed straight to :func:`yfinance.download`.
    data_dir
        Where ``save()`` writes the resulting NumPy arrays.
    """

    def __init__(
        self,
        universe_fn: UniverseFn = _default_sp500_universe,
        *,
        n_days: int = 120,
        benchmark: str | None = None,
        start_date: str = "2025-06-01",
        end_date: str = "2026-03-01",
        data_dir: str = "data",
    ) -> None:
        self.universe_fn = universe_fn
        self.n_days = n_days
        self._benchmark_override = benchmark
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir

        self.tickers: list[str] | None = None
        self.benchmark: str | None = benchmark
        self.X: np.ndarray | None = None
        self.X_raw: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.X_mean: np.ndarray | None = None
        self.X_std: np.ndarray | None = None
        self.stock_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def fetch_tickers(self) -> list[str]:
        """Resolve constituents + benchmark via the configured universe fn."""
        print("📥 Step 1: Fetching index constituents…")
        constituents, benchmark = self.universe_fn()
        if self._benchmark_override is not None:
            benchmark = self._benchmark_override
        # Ensure the benchmark is excluded from constituents to avoid leakage.
        if benchmark in constituents:
            constituents = [t for t in constituents if t != benchmark]
            print(f"   ⚠️  Removed benchmark '{benchmark}' from constituents.")
        self.tickers = list(constituents)
        self.benchmark = benchmark
        print(f"   ✅ {len(self.tickers)} tickers fetched (benchmark = {benchmark}).")
        return self.tickers

    def download_prices(self) -> tuple[pd.Series, pd.DataFrame]:
        """Download adjusted-close prices for the benchmark + constituents."""
        if self.tickers is None or self.benchmark is None:
            raise RuntimeError("fetch_tickers() must run before download_prices().")
        import yfinance as yf

        print(f"\n📥 Step 2: Downloading prices ({self.start_date} → {self.end_date})…")

        try:
            spy_data = yf.download(
                self.benchmark,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                multi_level_index=False,
                progress=False,
            )
            if spy_data.empty:
                raise ValueError(f"❌ No data returned for benchmark '{self.benchmark}'.")
            spy_prices = spy_data["Adj Close"]
            spy_prices.name = self.benchmark

            print(f"   Downloading {len(self.tickers)} stocks (may take 1-2 min)…")
            constituent_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                progress=True,
            )
            if constituent_data.empty:
                raise ValueError("❌ No constituent data returned from yfinance.")
            constituent_prices = constituent_data["Adj Close"]
        except KeyError as exc:
            print(f"   ⚠️  'Adj Close' not found ({exc}). Falling back to auto-adjusted 'Close'.")
            spy_data = yf.download(
                self.benchmark,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                multi_level_index=False,
                progress=False,
            )
            spy_prices = spy_data["Close"]
            spy_prices.name = self.benchmark
            constituent_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=True,
            )
            constituent_prices = constituent_data["Close"]
        except Exception as exc:  # pragma: no cover (network)
            raise ConnectionError(f"❌ Failed to download price data: {exc}")

        print(f"   ✅ {self.benchmark}: {spy_prices.shape[0]} days")
        print(f"   ✅ Constituents: {constituent_prices.shape}")
        return spy_prices, constituent_prices

    def compute_returns(
        self, spy_prices: pd.Series, constituent_prices: pd.DataFrame
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Convert adjusted-close prices to *simple* daily returns."""
        if self.benchmark is None:
            raise RuntimeError("benchmark unresolved")
        print("\n📊 Step 3: Computing simple returns…")
        spy_ret = spy_prices.pct_change().dropna()
        const_ret = constituent_prices.pct_change().dropna(how="all")
        print(f"   ✅ {self.benchmark} returns: {spy_ret.shape[0]} obs")
        print(f"   ✅ Constituent returns: {const_ret.shape}")
        return spy_ret, const_ret

    def align_and_clean(
        self, spy_ret: pd.Series, const_ret: pd.DataFrame
    ) -> pd.DataFrame:
        """Inner-join on date, drop NaN columns, slice to ``n_days``."""
        if self.benchmark is None:
            raise RuntimeError("benchmark unresolved")
        print("\n🧹 Step 4: Aligning dates and cleaning…")

        merged = pd.merge(
            spy_ret.rename(self.benchmark),
            const_ret,
            left_index=True,
            right_index=True,
            how="inner",
        )
        print(f"   After merge:    {merged.shape[0]} days × {merged.shape[1]} cols")

        spy_col = merged[[self.benchmark]]
        constituents_only = merged.drop(columns=[self.benchmark])
        constituents_clean = constituents_only.dropna(axis=1)
        n_dropped = constituents_only.shape[1] - constituents_clean.shape[1]
        if n_dropped > 0:
            print(f"   ⚠️  Dropped {n_dropped} stock(s) with incomplete history.")

        merged = pd.concat([spy_col, constituents_clean], axis=1)
        print(f"   After NaN drop: {merged.shape[0]} days × {merged.shape[1]} cols")

        if merged.shape[0] < self.n_days:
            raise ValueError(
                f"❌ Only {merged.shape[0]} trading days available, need {self.n_days}. "
                "Extend date range!"
            )
        final = merged.tail(self.n_days)
        print(f"   ✅ Final slice:  {final.shape[0]} days × {final.shape[1]} cols")
        print(
            f"   Date range: {final.index[0].strftime('%Y-%m-%d')} → "
            f"{final.index[-1].strftime('%Y-%m-%d')}"
        )
        return final

    def extract_and_standardize(self, final_df: pd.DataFrame) -> None:
        """Split ``final_df`` into ``X``, ``y`` and standardise ``X``."""
        if self.benchmark is None:
            raise RuntimeError("benchmark unresolved")
        print("\n⚙️  Step 5: Extracting arrays and standardizing X…")
        self.y = final_df[self.benchmark].to_numpy()
        stock_cols = [c for c in final_df.columns if c != self.benchmark]
        self.X_raw = final_df[stock_cols].to_numpy()
        self.stock_names = list(stock_cols)

        self.X_mean = self.X_raw.mean(axis=0)
        self.X_std = self.X_raw.std(axis=0)
        mask = self.X_std > 1e-10
        if not mask.all():
            n_rm = int((~mask).sum())
            print(f"   ⚠️  Removing {n_rm} zero-variance stock(s).")
            self.X_raw = self.X_raw[:, mask]
            self.X_mean = self.X_mean[mask]
            self.X_std = self.X_std[mask]
            self.stock_names = [s for s, m in zip(self.stock_names, mask) if m]

        self.X = (self.X_raw - self.X_mean) / self.X_std
        n, p = self.X.shape
        if p <= n:
            warnings.warn(
                f"This loader was designed for the high-dimensional regime p > n; "
                f"got p={p}, n={n}. Continuing anyway.",
                stacklevel=2,
            )
        print(f"   ✅ y shape: {self.y.shape}")
        print(f"   ✅ X shape: {self.X.shape}  (p={p}, n={n}, ratio={p / max(n, 1):.2f}x)")

    def save(self) -> None:
        """Persist the standardised arrays + tickers to ``data_dir``."""
        if (
            self.X is None
            or self.y is None
            or self.X_raw is None
            or self.X_mean is None
            or self.X_std is None
        ):
            raise RuntimeError("run() must complete before save().")
        os.makedirs(self.data_dir, exist_ok=True)
        np.save(os.path.join(self.data_dir, "X_standardized.npy"), self.X)
        np.save(os.path.join(self.data_dir, "y_spy.npy"), self.y)
        np.save(os.path.join(self.data_dir, "X_raw.npy"), self.X_raw)
        np.save(os.path.join(self.data_dir, "X_mean.npy"), self.X_mean)
        np.save(os.path.join(self.data_dir, "X_std.npy"), self.X_std)
        pd.Series(self.stock_names).to_csv(
            os.path.join(self.data_dir, "stock_names.csv"), index=False
        )
        print(f"\n💾 Data saved to '{self.data_dir}/':")
        for f in sorted(os.listdir(self.data_dir)):
            sz = os.path.getsize(os.path.join(self.data_dir, f)) / 1024
            print(f"   📁 {f:30s} ({sz:.1f} KB)")

    def run(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Execute the full pipeline end-to-end and return ``(X, y, names)``."""
        print("=" * 60)
        print("  PHASE 1: DATA ENGINEERING PIPELINE")
        print("=" * 60)

        self.fetch_tickers()
        spy_prices, const_prices = self.download_prices()
        spy_ret, const_ret = self.compute_returns(spy_prices, const_prices)
        final_df = self.align_and_clean(spy_ret, const_ret)
        self.extract_and_standardize(final_df)
        self.save()

        print("\n" + "=" * 60)
        print("  ✅ PHASE 1 COMPLETE")
        print("=" * 60)
        if self.X is None or self.y is None:
            raise RuntimeError("pipeline did not populate X/y arrays.")
        print(f"  X: {self.X.shape}  |  y: {self.y.shape}")
        print(f"  p/n ratio: {self.X.shape[1] / self.X.shape[0]:.2f}x")
        if self.stock_names is not None:
            print(f"  Stocks surviving: {len(self.stock_names)}")
        print(f"  NaN in X: {np.isnan(self.X).any()}  |  NaN in y: {np.isnan(self.y).any()}")
        print("=" * 60)
        return self.X, self.y, list(self.stock_names or [])


# ---------------------------------------------------------------------------
# Back-compat alias
# ---------------------------------------------------------------------------


class SP500DataLoader(IndexDataLoader):
    """Back-compat alias for the original Phase-1 loader.

    Identical to :class:`IndexDataLoader` with ``universe_fn = sp500`` and a
    legacy default of ``benchmark='SPY'``. Existing imports (``from
    sit.data.loader import SP500DataLoader``) continue to work unchanged.
    """

    def __init__(
        self,
        n_days: int = 120,
        benchmark: str = "SPY",
        start_date: str = "2025-06-01",
        end_date: str = "2026-03-01",
        data_dir: str = "data",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            universe_fn=_default_sp500_universe,
            n_days=n_days,
            benchmark=benchmark,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            **kwargs,
        )


__all__ = [
    "IndexDataLoader",
    "SP500DataLoader",
    "UniverseFn",
]


if __name__ == "__main__":  # pragma: no cover
    from sit.paths import DATA_DIR as _DATA_DIR

    loader = SP500DataLoader(data_dir=str(_DATA_DIR))
    loader.run()
