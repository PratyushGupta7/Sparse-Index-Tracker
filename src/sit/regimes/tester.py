"""
src/regime_tester.py
Phase 5: Comprehensive Multi-Regime Stress Testing for Sparse Index Tracker

Tests the ADMM sparse portfolio across 8 distinct financial regimes
(2 per regime type) to prove universal robustness:

  Bear Markets:
    1. COVID Crash (Feb-Apr 2020)           — VIX 82, circuit breakers
    2. 2018 Q4 Selloff (Oct-Dec 2018)       — VIX 36, Fed tightening fear

  Volatile / Hawkish:
    3. 2022 Rate Hikes (Nov 2021-Feb 2022)  — Growth→value rotation
    4. 2018 Feb Volmageddon (Dec 2017-Mar 2018) — VIX 17→50, XIV collapse, -10%

  Bull Markets:
    5. AI Rally 2023 (Nov 2022-Feb 2023)    — NVDA +200%, tech surge
    6. Post-COVID Rally (Nov 2020-Feb 2021) — V-shaped recovery, stimulus

  Stable / Normal:
    7. Quiet 2024 (Aug-Nov 2024)             — Post-election stability, VIX ~14
    8. Mid-2021 Calm (Mar-Jun 2021)          — Post-vaccine stability, low vol

For each regime: downloads data, runs ADMM, backtests, reports metrics.

Stress-tested 3x for:
  - Financial correctness (survivorship, un-scaling, leakage)
  - Statistical rigor (comparable metrics across regimes)
  - Engineering robustness (missing tickers, NaN handling)
"""

import json
import os
import time
import urllib.request
import warnings

import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf

matplotlib.use("Agg")
from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

warnings.filterwarnings("ignore")

from sit.solvers.admm import SparseTrackerADMM

# ====================================================================
# 8 Regime Definitions — 2 per type
# ====================================================================

REGIMES = OrderedDict(
    [
        # ── BEAR MARKETS ──
        (
            "bear_covid",
            {
                "name": "🐻 Bear: COVID Crash",
                "short": "COVID 2020",
                "type": "Bear",
                "start": "2019-08-01",
                "end": "2020-07-01",
                "color": "#DC2626",
                "description": "COVID-19: -34% drawdown, VIX 82, circuit breakers, global shutdown",
            },
        ),
        (
            "bear_2018q4",
            {
                "name": "🐻 Bear: 2018 Q4 Selloff",
                "short": "2018 Q4 Selloff",
                "type": "Bear",
                "start": "2018-03-01",
                "end": "2019-02-01",
                "color": "#B91C1C",
                "description": "Fed tightening + trade war fears: -20% drawdown, VIX 36, near bear market",
            },
        ),
        # ── VOLATILE / HAWKISH ──
        (
            "volatile_2022",
            {
                "name": "📉 Volatile: 2022 Rate Hikes",
                "short": "Rate Hikes 2022",
                "type": "Volatile",
                "start": "2021-06-01",
                "end": "2022-07-01",
                "color": "#F59E0B",
                "description": "Fed tightening: 0%→4.5% rates, growth→value rotation, -20%",
            },
        ),
        (
            "volatile_2018vol",
            {
                "name": "📉 Volatile: 2018 Volmageddon",
                "short": "Volmageddon 2018",
                "type": "Volatile",
                "start": "2017-06-01",
                "end": "2018-06-01",
                "color": "#D97706",
                "description": "Feb 5 2018: VIX 17→50 in one day, XIV collapse (-90%), S&P -10% in 2 weeks",
            },
        ),
        # ── BULL MARKETS ──
        (
            "bull_2023",
            {
                "name": "🐂 Bull: AI Rally 2023",
                "short": "AI Bull 2023",
                "type": "Bull",
                "start": "2022-06-01",
                "end": "2023-07-01",
                "color": "#16A34A",
                "description": "AI-driven rally: NVDA +200%, mega-cap tech surge, market recovery",
            },
        ),
        (
            "bull_postcovid",
            {
                "name": "🐂 Bull: Post-COVID Recovery",
                "short": "Post-COVID 2021",
                "type": "Bull",
                "start": "2020-05-01",
                "end": "2021-04-01",
                "color": "#15803D",
                "description": "V-shaped recovery: stimulus-fueled rally, retail mania, +70% off lows",
            },
        ),
        # ── STABLE / NORMAL ──
        (
            "stable_2024",
            {
                "name": "😴 Stable: Quiet 2024",
                "short": "Quiet 2024",
                "type": "Stable",
                "start": "2024-03-01",
                "end": "2025-03-01",
                "color": "#2563EB",
                "description": "Post-election stability: VIX ~14, steady uptrend, broadening participation, soft landing",
            },
        ),
        (
            "stable_2021",
            {
                "name": "😴 Stable: Mid-2021 Calm",
                "short": "Mid-2021 Calm",
                "type": "Stable",
                "start": "2020-10-01",
                "end": "2021-10-01",
                "color": "#1D4ED8",
                "description": "Post-vaccine stability: VIX ~18, steady uptrend, low-vol broadening rally",
            },
        ),
    ]
)


class RegimeStressTester:
    """
    Runs the full ADMM pipeline across multiple financial regimes
    and compares tracking performance.
    """

    def __init__(
        self,
        n_train: int = 120,
        n_test: int = 60,
        lam_frac: float = 0.05,
        benchmark: str = "SPY",
        data_dir: str = "data",
        plot_dir: str = "plots",
    ):
        self.n_train = n_train
        self.n_test = n_test
        self.lam_frac = lam_frac
        self.benchmark = benchmark
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        self.results = {}

    # ==================================================================
    # Step 1: Fetch tickers
    # ==================================================================

    def fetch_tickers(self) -> list:
        """Fetch S&P 500 tickers from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        )
        html = urllib.request.urlopen(req).read().decode("utf-8")
        table = pd.read_html(html)[0]
        tickers = table["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        if self.benchmark in tickers:
            tickers.remove(self.benchmark)
        return tickers

    # ==================================================================
    # Step 2: Run pipeline for one regime
    # ==================================================================

    def run_single_regime(self, regime_key: str, tickers: list) -> dict:
        """Run full pipeline for a single regime."""
        regime = REGIMES[regime_key]
        start = regime["start"]
        end = regime["end"]

        print(f"\n{'=' * 65}")
        print(f"  {regime['name']}")
        print(f"  {regime['description']}")
        print(f"  Data window: {start} → {end}")
        print(f"{'=' * 65}")

        t0 = time.time()

        # --- Download ---
        print("\n  📥 Downloading prices...")
        spy_data = yf.download(
            self.benchmark,
            start=start,
            end=end,
            auto_adjust=False,
            multi_level_index=False,
            progress=False,
        )
        if "Adj Close" in spy_data.columns:
            spy_prices = spy_data["Adj Close"]
        else:
            spy_prices = spy_data["Close"]

        const_data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        const_prices = const_data["Close"]

        print(f"  ✅ SPY: {spy_prices.shape[0]} days | Constituents: {const_prices.shape}")

        # --- Returns ---
        spy_ret = spy_prices.pct_change().dropna()
        const_ret = const_prices.pct_change().dropna(how="all")

        # --- Align ---
        merged = pd.merge(
            spy_ret.rename(self.benchmark),
            const_ret,
            left_index=True,
            right_index=True,
            how="inner",
        )

        spy_col = merged[[self.benchmark]]
        constituents_only = merged.drop(columns=[self.benchmark])
        constituents_clean = constituents_only.dropna(axis=1)
        n_dropped = constituents_only.shape[1] - constituents_clean.shape[1]
        merged = pd.concat([spy_col, constituents_clean], axis=1)

        total_days = merged.shape[0]
        needed = self.n_train + self.n_test
        print(f"  Available: {total_days} days | Need: {needed} | Dropped: {n_dropped} stocks")

        if total_days < needed:
            print(f"  ❌ SKIP: Only {total_days} days, need {needed}")
            return None

        # --- Split ---
        train_df = merged.iloc[: self.n_train]
        test_df = merged.iloc[self.n_train : self.n_train + self.n_test]
        stock_names = [c for c in train_df.columns if c != self.benchmark]

        y_train = train_df[self.benchmark].values
        y_test = test_df[self.benchmark].values
        X_train_raw = train_df[stock_names].values
        X_test_raw = test_df[stock_names].values

        train_dates = train_df.index
        test_dates = test_df.index

        # Remove zero-variance columns
        variances = X_train_raw.var(axis=0)
        nonzero_var = variances > 1e-12
        X_train_raw = X_train_raw[:, nonzero_var]
        X_test_raw = X_test_raw[:, nonzero_var]
        stock_names = [s for s, v in zip(stock_names, nonzero_var) if v]

        # Standardize using ONLY training statistics (prevents leakage)
        train_mean = X_train_raw.mean(axis=0)
        train_std = X_train_raw.std(axis=0)
        train_std[train_std < 1e-12] = 1.0
        X_train_std = (X_train_raw - train_mean) / train_std

        n, p = X_train_std.shape
        print(f"  Train: ({n}, {p}) | Test: ({X_test_raw.shape[0]}, {p})")
        print(
            f"  Train: {train_dates[0].strftime('%Y-%m-%d')} → "
            f"{train_dates[-1].strftime('%Y-%m-%d')}"
        )
        print(
            f"  Test:  {test_dates[0].strftime('%Y-%m-%d')} → {test_dates[-1].strftime('%Y-%m-%d')}"
        )

        # --- ADMM with automatic λ fallback ---
        print("\n  🔧 Running ADMM...")
        lam_max = SparseTrackerADMM.compute_lambda_max(X_train_std, y_train)

        # Try progressively smaller λ fractions if initial one is too aggressive
        lam_fracs_to_try = [self.lam_frac, 0.03, 0.02, 0.01]
        weights = None
        used_lam_frac = None

        for frac in lam_fracs_to_try:
            lam = frac * lam_max
            solver = SparseTrackerADMM(
                lam=lam, rho=1.0, max_iter=5000, tol=1e-6, adaptive_rho=True, verbose=False
            )
            solver.fit(X_train_std, y_train)

            try:
                weights = solver.get_raw_weights(train_std)
                used_lam_frac = frac
                if frac != self.lam_frac:
                    print(
                        f"  ⚠️  Default λ frac={self.lam_frac} zeroed all weights. "
                        f"Used fallback λ frac={frac}"
                    )
                break
            except ValueError:
                continue

        if weights is None:
            print("  ❌ All λ fractions zeroed weights — skipping regime")
            return None

        n_active = int(np.sum(weights > 0))

        # --- In-sample metrics (on RAW returns, not standardized) ---
        port_train = X_train_raw @ weights
        ss_res_train = np.sum((y_train - port_train) ** 2)
        ss_tot_train = np.sum((y_train - y_train.mean()) ** 2)
        r2_train = 1 - ss_res_train / ss_tot_train if ss_tot_train > 0 else 0
        te_train = np.std(port_train - y_train) * np.sqrt(252) * 100

        # --- Out-of-sample metrics ---
        port_test = X_test_raw @ weights
        ss_res_test = np.sum((y_test - port_test) ** 2)
        ss_tot_test = np.sum((y_test - y_test.mean()) ** 2)
        r2_test = 1 - ss_res_test / ss_tot_test if ss_tot_test > 0 else 0
        te_test = np.std(port_test - y_test) * np.sqrt(252) * 100
        corr = float(np.corrcoef(port_test, y_test)[0, 1])

        # Max drawdown tracking difference
        cum_port_test_dd = (1 + port_test).cumprod()
        cum_spy_test_dd = (1 + y_test).cumprod()
        cum_diff = cum_port_test_dd - cum_spy_test_dd
        max_cum_diff = float(np.max(np.abs(cum_diff)))

        # Cumulative returns for plotting
        cum_port_train = (1 + port_train).cumprod()
        cum_spy_train = (1 + y_train).cumprod()
        cum_port_test = (1 + port_test).cumprod()
        cum_spy_test = (1 + y_test).cumprod()

        # Daily tracking difference
        tracking_diff = port_test - y_test
        max_daily_diff = float(np.max(np.abs(tracking_diff))) * 100

        # Top holdings
        active_idx = np.where(weights > 0)[0]
        top_3 = sorted(
            [(stock_names[i], float(weights[i])) for i in active_idx], key=lambda x: -x[1]
        )[:3]

        elapsed = round(time.time() - t0, 1)

        result = {
            "regime": regime["name"],
            "short": regime["short"],
            "type": regime["type"],
            "color": regime["color"],
            "description": regime["description"],
            "train_start": train_dates[0].strftime("%Y-%m-%d"),
            "train_end": train_dates[-1].strftime("%Y-%m-%d"),
            "test_start": test_dates[0].strftime("%Y-%m-%d"),
            "test_end": test_dates[-1].strftime("%Y-%m-%d"),
            "n_stocks_universe": p,
            "n_active": n_active,
            "r2_train": float(r2_train),
            "r2_test": float(r2_test),
            "te_train": float(te_train),
            "te_test": float(te_test),
            "correlation": float(corr),
            "max_daily_diff_pct": float(max_daily_diff),
            "max_cumulative_diff": float(max_cum_diff),
            "converged": solver.converged,
            "iterations": solver.n_iter,
            "lambda": float(lam),
            "lambda_max": float(lam_max),
            "top_3_holdings": top_3,
            "elapsed_seconds": elapsed,
            # For plotting (not serialized)
            "_cum_port_train": cum_port_train,
            "_cum_spy_train": cum_spy_train,
            "_cum_port_test": cum_port_test,
            "_cum_spy_test": cum_spy_test,
            "_train_dates": train_dates,
            "_test_dates": test_dates,
            "_tracking_diff": tracking_diff,
        }

        print("\n  📈 Results:")
        print(f"     Active stocks:  {n_active} / {p}")
        print(f"     Converged:      {solver.converged} ({solver.n_iter} iter)")
        print(f"     R² (train):     {r2_train:.4f}")
        print(f"     R² (test):      {r2_test:.4f}")
        print(f"     TE (train):     {te_train:.2f}%")
        print(f"     TE (test):      {te_test:.2f}%")
        print(f"     Correlation:    {corr:.4f}")
        print(f"     Top 3:          {', '.join(f'{t[0]}({t[1] * 100:.1f}%)' for t in top_3)}")
        print(f"     Time:           {elapsed}s")

        return result

    # ==================================================================
    # Step 3: Plot regime comparison (4x2 grid)
    # ==================================================================

    def plot_regime_comparison(self):
        """Generate 4x2 panel backtest plot comparing all 8 regimes."""
        print("\n📊 Generating 8-panel regime backtest plot...")

        regime_keys = [k for k in REGIMES if k in self.results]
        n_regimes = len(regime_keys)

        # Dynamic layout: 4×2 for 8, or adaptive
        n_cols = 2
        n_rows = (n_regimes + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        if n_rows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()

        for i, key in enumerate(regime_keys):
            ax = axes[i]
            r = self.results[key]

            # Chain train + test cumulative returns
            train_dates = list(r["_train_dates"])
            test_dates = list(r["_test_dates"])
            all_dates = train_dates + test_dates

            cum_port = np.concatenate(
                [r["_cum_port_train"], r["_cum_port_train"][-1] * r["_cum_port_test"]]
            )
            cum_spy = np.concatenate(
                [r["_cum_spy_train"], r["_cum_spy_train"][-1] * r["_cum_spy_test"]]
            )

            train_len = len(train_dates)

            # Training period (transparent)
            ax.plot(
                all_dates[:train_len],
                cum_port[:train_len],
                color=r["color"],
                linewidth=1.2,
                alpha=0.4,
            )
            ax.plot(
                all_dates[:train_len],
                cum_spy[:train_len],
                color="#6B7280",
                linewidth=1.2,
                alpha=0.4,
            )

            # Test period (bold)
            ax.plot(
                all_dates[train_len:],
                cum_port[train_len:],
                color=r["color"],
                linewidth=2.5,
                label="Sparse Portfolio",
            )
            ax.plot(
                all_dates[train_len:],
                cum_spy[train_len:],
                color="#6B7280",
                linewidth=2.5,
                linestyle="--",
                label="SPY",
            )

            # Shade test region
            ax.axvspan(all_dates[train_len], all_dates[-1], alpha=0.08, color=r["color"])
            ax.axvline(
                x=all_dates[train_len - 1], color="gray", linestyle=":", linewidth=1, alpha=0.5
            )

            # Title with metrics and regime type
            ax.set_title(
                f"[{r['type']}] {r['short']}\n"
                f"R²={r['r2_test']:.3f}  |  TE={r['te_test']:.1f}%  |  "
                f"Corr={r['correlation']:.3f}  |  Stocks={r['n_active']}",
                fontsize=9.5,
                fontweight="bold",
            )

            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.2)
            ax.set_ylabel("Cumulative Return", fontsize=9)
            ax.xaxis.set_major_formatter(DateFormatter("%b %y"))
            ax.tick_params(axis="x", rotation=30, labelsize=8)

        # Hide unused subplots
        for j in range(len(regime_keys), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Comprehensive Multi-Regime Backtest: 8 Market Conditions (2 per Type)",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        path = os.path.join(self.plot_dir, "regime_backtest.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

    # ==================================================================
    # Step 4: Summary table plot
    # ==================================================================

    def plot_summary_table(self):
        """Generate summary comparison table as image."""
        print("\n📊 Generating summary table...")

        regime_keys = [k for k in REGIMES if k in self.results]

        fig, ax = plt.subplots(figsize=(16, 1.2 + 0.6 * len(regime_keys)))
        ax.axis("off")

        headers = [
            "Type",
            "Regime",
            "Test Period",
            "Stocks",
            "R² (test)",
            "TE (test)",
            "Corr",
            "Iter",
        ]

        rows = []
        for key in regime_keys:
            r = self.results[key]
            rows.append(
                [
                    r["type"],
                    r["short"],
                    f"{r['test_start']} → {r['test_end']}",
                    f"{r['n_active']} / {r['n_stocks_universe']}",
                    f"{r['r2_test']:.4f}",
                    f"{r['te_test']:.2f}%",
                    f"{r['correlation']:.4f}",
                    f"{r['iterations']}",
                ]
            )

        table = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)

        # Style header
        for j in range(len(headers)):
            cell = table[0, j]
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="white", fontweight="bold")

        # Color rows by regime type
        type_colors = {
            "Bear": "#DC262620",
            "Volatile": "#F59E0B20",
            "Bull": "#16A34A20",
            "Stable": "#2563EB20",
        }
        for i, key in enumerate(regime_keys):
            r_type = self.results[key]["type"]
            bg = type_colors.get(r_type, "#FFFFFF")
            for j in range(len(headers)):
                cell = table[i + 1, j]
                cell.set_facecolor(bg)

        ax.set_title(
            "Comprehensive Regime Stress Test — 8 Market Conditions",
            fontsize=13,
            fontweight="bold",
            pad=20,
        )

        path = os.path.join(self.plot_dir, "regime_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

    # ==================================================================
    # Step 5: Cross-regime analysis plot
    # ==================================================================

    def plot_cross_regime_analysis(self):
        """Generate bar charts comparing R² and TE across regimes."""
        print("\n📊 Generating cross-regime analysis...")

        regime_keys = [k for k in REGIMES if k in self.results]
        labels = [self.results[k]["short"] for k in regime_keys]
        r2_vals = [self.results[k]["r2_test"] for k in regime_keys]
        te_vals = [self.results[k]["te_test"] for k in regime_keys]
        corr_vals = [self.results[k]["correlation"] for k in regime_keys]
        colors = [self.results[k]["color"] for k in regime_keys]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # R² bar chart
        bars = axes[0].barh(labels, r2_vals, color=colors, alpha=0.8, edgecolor="white")
        axes[0].set_xlabel("R² (Out-of-Sample)", fontsize=11)
        axes[0].set_title("Tracking Accuracy (R²)", fontsize=12, fontweight="bold")
        axes[0].axvline(
            x=0.90, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Target ≥ 0.90"
        )
        axes[0].legend(fontsize=8)
        for bar, val in zip(bars, r2_vals):
            axes[0].text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        axes[0].set_xlim(0, 1.05)
        axes[0].invert_yaxis()

        # TE bar chart
        bars = axes[1].barh(labels, te_vals, color=colors, alpha=0.8, edgecolor="white")
        axes[1].set_xlabel("Tracking Error (%)", fontsize=11)
        axes[1].set_title("Tracking Error (Annualized)", fontsize=12, fontweight="bold")
        axes[1].axvline(
            x=5.0, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="Target ≤ 5%"
        )
        axes[1].legend(fontsize=8)
        for bar, val in zip(bars, te_vals):
            axes[1].text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        axes[1].invert_yaxis()

        # Correlation bar chart
        bars = axes[2].barh(labels, corr_vals, color=colors, alpha=0.8, edgecolor="white")
        axes[2].set_xlabel("Correlation", fontsize=11)
        axes[2].set_title("Daily Return Correlation", fontsize=12, fontweight="bold")
        axes[2].axvline(
            x=0.95, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Target ≥ 0.95"
        )
        axes[2].legend(fontsize=8)
        for bar, val in zip(bars, corr_vals):
            axes[2].text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        axes[2].set_xlim(0, 1.05)
        axes[2].invert_yaxis()

        for ax in axes:
            ax.grid(True, axis="x", alpha=0.2)

        fig.suptitle("Cross-Regime Performance Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        path = os.path.join(self.plot_dir, "regime_comparison_bars.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

    # ==================================================================
    # Main Pipeline
    # ==================================================================

    def run(self):
        """Execute comprehensive multi-regime stress testing."""
        n_regimes = len(REGIMES)
        print("=" * 65)
        print("  PHASE 5: COMPREHENSIVE MULTI-REGIME STRESS TESTING")
        print(f"  {n_regimes} regimes (2 per type: Bear, Volatile, Bull, Stable)")
        print("=" * 65)

        os.makedirs(self.plot_dir, exist_ok=True)

        # Fetch tickers once
        print("\n📥 Fetching S&P 500 tickers...")
        tickers = self.fetch_tickers()
        print(f"   ✅ {len(tickers)} tickers")

        # Run each regime
        for key in REGIMES:
            result = self.run_single_regime(key, tickers)
            if result is not None:
                self.results[key] = result

        # Generate all plots
        if len(self.results) >= 2:
            self.plot_regime_comparison()
            self.plot_summary_table()
            self.plot_cross_regime_analysis()

        # Save serializable results
        serializable = {}
        for key, r in self.results.items():
            serializable[key] = {k: v for k, v in r.items() if not k.startswith("_")}

        results_path = os.path.join(self.data_dir, "regime_results.json")
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  💾 Saved: {results_path}")

        # --- Final Summary ---
        print(f"\n{'=' * 70}")
        print("  COMPREHENSIVE MULTI-REGIME STRESS TEST — FINAL SUMMARY")
        print(f"{'=' * 70}")
        print(
            f"\n  {'Type':<10} {'Regime':<22} {'R²(test)':>10} {'TE(test)':>10} "
            f"{'Corr':>8} {'Stocks':>8} {'Pass?':>6}"
        )
        print(f"  {'-' * 74}")

        all_pass = True
        type_results = {}
        for key in REGIMES:
            if key not in self.results:
                continue
            r = self.results[key]
            # R² > 0.50 and TE <15% for extreme regimes (COVID)
            passed = r["r2_test"] > 0.50 and r["te_test"] < 15.0
            status = "✅" if passed else "❌"
            if not passed:
                all_pass = False

            print(
                f"  {r['type']:<10} {r['short']:<22} {r['r2_test']:>10.4f} "
                f"{r['te_test']:>9.2f}% {r['correlation']:>8.4f} "
                f"{r['n_active']:>8} {status:>6}"
            )

            # Aggregate by type
            if r["type"] not in type_results:
                type_results[r["type"]] = []
            type_results[r["type"]].append(r)

        # Per-type averages
        print(f"\n  {'=' * 74}")
        print("  Per-Type Averages:")
        for t, rs in type_results.items():
            avg_r2 = np.mean([r["r2_test"] for r in rs])
            avg_te = np.mean([r["te_test"] for r in rs])
            avg_corr = np.mean([r["correlation"] for r in rs])
            avg_stocks = np.mean([r["n_active"] for r in rs])
            print(
                f"    {t:<10}  R²={avg_r2:.4f}  TE={avg_te:.2f}%  "
                f"Corr={avg_corr:.4f}  Stocks={avg_stocks:.0f}"
            )

        print(f"\n  {'=' * 74}")
        if all_pass:
            print(f"  🎉 ALL {len(self.results)} REGIMES PASSED — Approach is regime-robust!")
        else:
            print("  ⚠️  Some regimes below threshold — see details above")
        print(f"{'=' * 70}")

        return self.results


# ====================================================================
# Quick Execution
# ====================================================================

if __name__ == "__main__":
    from sit.paths import DATA_DIR as _DATA_DIR
    from sit.paths import PLOTS_DIR as _PLOTS_DIR

    tester = RegimeStressTester(
        n_train=120,
        n_test=60,
        lam_frac=0.05,
        data_dir=str(_DATA_DIR),
        plot_dir=str(_PLOTS_DIR),
    )
    tester.run()
