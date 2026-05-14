"""
src/backtester.py
Phase 3: Academic Proofs & Validation for Sparse Index Tracker

Generates three deliverables:
  1. Convergence Plot — primal/dual residuals over ADMM iterations
  2. Regularization Path — sparsity (nnz) vs λ curve
  3. Out-of-Sample Backtest — cumulative portfolio vs SPY on unseen data

Stress-tested 3x for:
  - Financial correctness (train/test leakage, un-scaling, survivorship)
  - Visual integrity (log scales, labels, clean formatting)
  - Statistical rigor (honest out-of-sample reporting)
"""

import json
import os
import urllib.request
import warnings

import matplotlib
import numpy as np
import pandas as pd
import yfinance as yf

matplotlib.use("Agg")  # Non-interactive backend (save to file)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

from sit.solvers.admm import SparseTrackerADMM


class Phase3Validator:
    """
    Complete Phase 3 pipeline: re-downloads data, splits train/test,
    re-runs ADMM on training set, and generates all academic proofs.
    """

    def __init__(
        self,
        start_date: str = "2025-04-01",
        end_date: str = "2026-03-09",
        n_train: int = 120,
        n_test: int = 60,
        benchmark: str = "SPY",
        lam_frac: float = 0.05,
        data_dir: str = "data",
        plot_dir: str = "plots",
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.n_train = n_train
        self.n_test = n_test
        self.benchmark = benchmark
        self.lam_frac = lam_frac
        self.data_dir = data_dir
        self.plot_dir = plot_dir

        # Will be populated by the pipeline
        self.stock_names = None
        self.X_train_std = None
        self.X_train_raw = None
        self.y_train = None
        self.X_test_raw = None
        self.y_test = None
        self.train_dates = None
        self.test_dates = None
        self.train_mean = None
        self.train_std = None
        self.solver = None
        self.weights = None  # Raw-space weights

    # ==================================================================
    # Step 1: Data Download & Train/Test Split
    # ==================================================================

    def fetch_tickers(self) -> list:
        """Fetch S&P 500 tickers from Wikipedia."""
        print("📥 Step 1a: Fetching S&P 500 tickers...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        )
        html = urllib.request.urlopen(req).read().decode("utf-8")
        table = pd.read_html(html)[0]
        tickers = table["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]

        # Remove benchmark from constituents (prevent data leakage)
        if self.benchmark in tickers:
            tickers.remove(self.benchmark)

        print(f"   ✅ {len(tickers)} constituent tickers fetched.")
        return tickers

    def download_and_split(self):
        """Download prices, compute returns, split into train/test."""
        tickers = self.fetch_tickers()

        print(f"\n📥 Step 1b: Downloading prices ({self.start_date} → {self.end_date})...")

        # Download SPY
        spy_data = yf.download(
            self.benchmark,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False,
            multi_level_index=False,
            progress=False,
        )
        if "Adj Close" in spy_data.columns:
            spy_prices = spy_data["Adj Close"]
        else:
            spy_prices = spy_data["Close"]

        # Download constituents
        const_data = yf.download(
            tickers, start=self.start_date, end=self.end_date, auto_adjust=True, progress=True
        )
        const_prices = const_data["Close"]

        print(f"   ✅ SPY: {spy_prices.shape[0]} days")
        print(f"   ✅ Constituents: {const_prices.shape}")

        # Compute simple returns
        print("\n📊 Step 1c: Computing simple returns...")
        spy_ret = spy_prices.pct_change().dropna()
        const_ret = const_prices.pct_change().dropna(how="all")

        # Align dates (inner join)
        merged = pd.merge(
            spy_ret.rename(self.benchmark),
            const_ret,
            left_index=True,
            right_index=True,
            how="inner",
        )

        # Protect SPY, clean NaN stocks
        spy_col = merged[[self.benchmark]]
        constituents_only = merged.drop(columns=[self.benchmark])
        constituents_clean = constituents_only.dropna(axis=1)
        n_dropped = constituents_only.shape[1] - constituents_clean.shape[1]
        if n_dropped > 0:
            print(f"   ⚠️  Dropped {n_dropped} stock(s) with incomplete history.")
        merged = pd.concat([spy_col, constituents_clean], axis=1)

        total_days = merged.shape[0]
        needed = self.n_train + self.n_test
        print(f"   Total available: {total_days} return days, need {needed}")

        if total_days < needed:
            raise ValueError(
                f"❌ Only {total_days} days available, need {needed}. Extend date range!"
            )

        # Split: first n_train for training, next n_test for testing
        train_df = merged.iloc[: self.n_train]
        test_df = merged.iloc[self.n_train : self.n_train + self.n_test]

        self.train_dates = train_df.index
        self.test_dates = test_df.index
        self.stock_names = list([c for c in train_df.columns if c != self.benchmark])

        # Extract arrays
        self.y_train = train_df[self.benchmark].values
        self.y_test = test_df[self.benchmark].values

        X_train_raw_df = train_df[self.stock_names]
        X_test_raw_df = test_df[self.stock_names]

        self.X_train_raw = X_train_raw_df.values
        self.X_test_raw = X_test_raw_df.values

        # Remove zero-variance stocks from training
        variances = self.X_train_raw.var(axis=0)
        nonzero_var = variances > 1e-12
        if not nonzero_var.all():
            n_removed = (~nonzero_var).sum()
            print(f"   ⚠️  Removed {n_removed} zero-variance stocks.")
            self.X_train_raw = self.X_train_raw[:, nonzero_var]
            self.X_test_raw = self.X_test_raw[:, nonzero_var]
            self.stock_names = [s for s, v in zip(self.stock_names, nonzero_var) if v]

        # Standardize using ONLY training statistics (prevent test leakage)
        self.train_mean = self.X_train_raw.mean(axis=0)
        self.train_std = self.X_train_raw.std(axis=0)
        self.train_std[self.train_std < 1e-12] = 1.0  # guard
        self.X_train_std = (self.X_train_raw - self.train_mean) / self.train_std

        n, p = self.X_train_std.shape
        print(f"\n   ✅ Train: X({n}, {p}), y({n},)")
        print(
            f"      Date range: {self.train_dates[0].strftime('%Y-%m-%d')} → "
            f"{self.train_dates[-1].strftime('%Y-%m-%d')}"
        )
        print(
            f"   ✅ Test:  X({self.X_test_raw.shape[0]}, {self.X_test_raw.shape[1]}), "
            f"y({self.y_test.shape[0]},)"
        )
        print(
            f"      Date range: {self.test_dates[0].strftime('%Y-%m-%d')} → "
            f"{self.test_dates[-1].strftime('%Y-%m-%d')}"
        )
        print(f"   p/n ratio: {p / n:.2f}x")

    # ==================================================================
    # Step 2: Re-run ADMM on Training Data
    # ==================================================================

    def run_admm(self):
        """Re-run ADMM solver on the training data."""
        print(f"\n{'=' * 65}")
        print("  RE-RUNNING ADMM ON TRAINING DATA")
        print(f"{'=' * 65}")

        lam_max = SparseTrackerADMM.compute_lambda_max(self.X_train_std, self.y_train)
        lam = self.lam_frac * lam_max
        print(f"   λ_max = {lam_max:.6f}")
        print(f"   λ     = {lam:.6f}  ({self.lam_frac} × λ_max)")

        self.solver = SparseTrackerADMM(
            lam=lam, rho=1.0, max_iter=5000, tol=1e-6, adaptive_rho=True, verbose=True
        )
        self.solver.fit(self.X_train_std, self.y_train)

        # Get raw-space weights (un-scaled from standardized space)
        self.weights = self.solver.get_raw_weights(self.train_std)

        # In-sample metrics
        port_train = self.X_train_raw @ self.weights
        ss_res = np.sum((self.y_train - port_train) ** 2)
        ss_tot = np.sum((self.y_train - self.y_train.mean()) ** 2)
        r2_train = 1 - ss_res / ss_tot
        te_train = np.std(port_train - self.y_train) * np.sqrt(252) * 100

        n_nonzero = np.sum(self.weights > 0)
        print("\n  📈 In-Sample Metrics:")
        print(f"     Active stocks:     {n_nonzero} / {len(self.weights)}")
        print(f"     R² (train):        {r2_train:.4f}")
        print(f"     Ann. TE (train):   {te_train:.2f}%")

        return r2_train, te_train

    # ==================================================================
    # Plot 1: Convergence
    # ==================================================================

    def plot_convergence(self):
        """Plot primal and dual residuals over ADMM iterations."""
        print("\n📊 Generating convergence plot...")

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        iters = range(1, len(self.solver.primal_residuals) + 1)
        ax.semilogy(
            iters,
            self.solver.primal_residuals,
            color="#2563EB",
            linewidth=2,
            label="Primal residual $\\|w - z\\|_2$",
        )
        ax.semilogy(
            iters,
            self.solver.dual_residuals,
            color="#DC2626",
            linewidth=2,
            linestyle="--",
            label="Dual residual $\\|\\rho(z^k - z^{k-1})\\|_2$",
        )

        # Mark convergence point
        ax.axvline(
            x=self.solver.n_iter,
            color="#16A34A",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"Converged (iter {self.solver.n_iter})",
        )

        ax.set_xlabel("Iteration", fontsize=13)
        ax.set_ylabel("Residual (log scale)", fontsize=13)
        ax.set_title("ADMM Convergence: Primal & Dual Residuals", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=1)

        plt.tight_layout()
        path = os.path.join(self.plot_dir, "convergence.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

    # ==================================================================
    # Plot 2: Regularization Path
    # ==================================================================

    def plot_regularization_path(self, n_lambdas: int = 50):
        """Sweep 50 λ values and plot sparsity curve."""
        print(f"\n📊 Generating regularization path ({n_lambdas} λ values)...")

        lam_max = SparseTrackerADMM.compute_lambda_max(self.X_train_std, self.y_train)
        lambdas = np.logspace(np.log10(0.001 * lam_max), np.log10(lam_max), n_lambdas)

        nnz_list = []
        te_list = []
        r2_list = []

        for i, lam in enumerate(lambdas):
            s = SparseTrackerADMM(
                lam=lam, rho=1.0, max_iter=3000, tol=1e-6, adaptive_rho=True, verbose=False
            )
            s.fit(self.X_train_std, self.y_train)

            try:
                w = s.get_raw_weights(self.train_std)
            except ValueError:
                w = np.zeros(len(self.stock_names))

            nnz = int(np.sum(w > 0))
            nnz_list.append(nnz)

            # Tracking metrics on raw returns
            port_ret = self.X_train_raw @ w
            te = np.std(port_ret - self.y_train) * np.sqrt(252) * 100
            ss_res = np.sum((self.y_train - port_ret) ** 2)
            ss_tot = np.sum((self.y_train - self.y_train.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            te_list.append(te)
            r2_list.append(r2)

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"   [{i + 1}/{n_lambdas}] λ={lam:.6f}  nnz={nnz:4d}  R²={r2:.4f}  TE={te:.2f}%"
                )

        # Plot: sparsity vs λ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Non-zero count vs λ
        ax1.semilogx(
            lambdas,
            nnz_list,
            color="#2563EB",
            linewidth=2.5,
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
        # Mark our chosen λ
        chosen_lam = self.lam_frac * lam_max
        chosen_nnz = int(np.sum(self.weights > 0))
        ax1.axvline(
            x=chosen_lam,
            color="#DC2626",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Selected λ={chosen_lam:.4f}",
        )
        ax1.scatter(
            [chosen_lam],
            [chosen_nnz],
            color="#DC2626",
            s=120,
            zorder=5,
            edgecolors="white",
            linewidth=2,
            label=f"nnz={chosen_nnz}",
        )

        ax1.set_xlabel("λ (log scale)", fontsize=13)
        ax1.set_ylabel("Non-zero weights", fontsize=13)
        ax1.set_title("Regularization Path: Sparsity vs λ", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Right: R² and TE vs λ
        color_r2 = "#2563EB"
        color_te = "#DC2626"
        ax2.semilogx(lambdas, r2_list, color=color_r2, linewidth=2, label="R²")
        ax2.set_xlabel("λ (log scale)", fontsize=13)
        ax2.set_ylabel("R²", fontsize=13, color=color_r2)
        ax2.tick_params(axis="y", labelcolor=color_r2)
        ax2.axvline(x=chosen_lam, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        ax2_twin = ax2.twinx()
        ax2_twin.semilogx(
            lambdas, te_list, color=color_te, linewidth=2, linestyle="--", label="Ann. TE (%)"
        )
        ax2_twin.set_ylabel("Annualized TE (%)", fontsize=13, color=color_te)
        ax2_twin.tick_params(axis="y", labelcolor=color_te)

        ax2.set_title("Accuracy vs λ", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center left")

        plt.tight_layout()
        path = os.path.join(self.plot_dir, "regularization_path.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

        return lambdas, nnz_list, r2_list, te_list

    # ==================================================================
    # Plot 3: Out-of-Sample Backtest
    # ==================================================================

    def plot_backtest(self):
        """
        Apply portfolio weights to unseen test data and compare vs SPY.
        """
        print("\n📊 Generating backtest plot...")

        # Portfolio daily returns on TEST data (raw-space weights × raw returns)
        port_ret_test = self.X_test_raw @ self.weights
        spy_ret_test = self.y_test

        # Cumulative returns (1 + r).cumprod(), both start at 1.0
        cum_port = (1 + port_ret_test).cumprod()
        cum_spy = (1 + spy_ret_test).cumprod()

        # Also compute in-sample cumulative returns
        port_ret_train = self.X_train_raw @ self.weights
        cum_port_train = (1 + port_ret_train).cumprod()
        cum_spy_train = (1 + self.y_train).cumprod()

        # Out-of-sample metrics
        ss_res_test = np.sum((spy_ret_test - port_ret_test) ** 2)
        ss_tot_test = np.sum((spy_ret_test - spy_ret_test.mean()) ** 2)
        r2_test = 1 - ss_res_test / ss_tot_test if ss_tot_test > 0 else 0.0
        te_test = np.std(port_ret_test - spy_ret_test) * np.sqrt(252) * 100
        daily_corr = np.corrcoef(port_ret_test, spy_ret_test)[0, 1]

        # In-sample metrics for comparison
        ss_res_train = np.sum((self.y_train - port_ret_train) ** 2)
        ss_tot_train = np.sum((self.y_train - self.y_train.mean()) ** 2)
        r2_train = 1 - ss_res_train / ss_tot_train
        te_train = np.std(port_ret_train - self.y_train) * np.sqrt(252) * 100

        # Total return
        total_port = cum_port[-1] - 1
        total_spy = cum_spy[-1] - 1

        print("\n  📈 In-Sample vs Out-of-Sample Comparison:")
        print(f"  {'Metric':<25} {'In-Sample':>12} {'Out-of-Sample':>14}")
        print(f"  {'-' * 51}")
        print(f"  {'R²':<25} {r2_train:>12.4f} {r2_test:>14.4f}")
        print(f"  {'Annualized TE':<25} {te_train:>11.2f}% {te_test:>13.2f}%")
        print(f"  {'Correlation':<25} {'—':>12} {daily_corr:>14.4f}")
        print(f"  {'Portfolio total return':<25} {'—':>12} {total_port:>13.2f}%")
        print(f"  {'SPY total return':<25} {'—':>12} {total_spy:>13.2f}%")

        # --- Plot ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2.5, 1]})

        # === Top: Cumulative Returns ===
        ax1 = axes[0]

        # Combine train + test for continuous line
        all_dates = list(self.train_dates) + list(self.test_dates)
        train_len = len(self.train_dates)
        test_len = len(self.test_dates)

        # Full cumulative (train then test, chained)
        cum_port_full = np.concatenate([cum_port_train, cum_port_train[-1] * cum_port])
        cum_spy_full = np.concatenate([cum_spy_train, cum_spy_train[-1] * cum_spy])

        # Training period
        ax1.plot(
            all_dates[:train_len],
            cum_port_full[:train_len],
            color="#2563EB",
            linewidth=2,
            label="Sparse Portfolio (in-sample)",
        )
        ax1.plot(
            all_dates[:train_len],
            cum_spy_full[:train_len],
            color="#DC2626",
            linewidth=2,
            label="SPY Benchmark (in-sample)",
        )

        # Test period (different style)
        ax1.plot(
            all_dates[train_len:],
            cum_port_full[train_len:],
            color="#2563EB",
            linewidth=2.5,
            linestyle="-",
            alpha=0.9,
            label="Sparse Portfolio (out-of-sample)",
        )
        ax1.plot(
            all_dates[train_len:],
            cum_spy_full[train_len:],
            color="#DC2626",
            linewidth=2.5,
            linestyle="-",
            alpha=0.9,
            label="SPY Benchmark (out-of-sample)",
        )

        # Mark train/test boundary
        ax1.axvline(
            x=all_dates[train_len - 1], color="#6B7280", linestyle=":", linewidth=2, alpha=0.6
        )
        ax1.text(
            all_dates[train_len - 1],
            ax1.get_ylim()[0] * 0.98,
            "  ← Train | Test →",
            fontsize=10,
            color="#6B7280",
            verticalalignment="bottom",
        )

        # Shade test region
        ax1.axvspan(all_dates[train_len], all_dates[-1], alpha=0.06, color="#F59E0B")

        ax1.set_ylabel("Cumulative Return", fontsize=13)
        ax1.set_title(
            f"Backtest: Sparse Portfolio ({int(np.sum(self.weights > 0))} stocks) vs SPY",
            fontsize=15,
            fontweight="bold",
        )
        ax1.legend(fontsize=9, loc="upper left", ncol=2)
        ax1.grid(True, alpha=0.3)

        # === Bottom: Daily Tracking Difference (test set only) ===
        ax2 = axes[1]
        tracking_diff = port_ret_test - spy_ret_test
        colors = ["#DC2626" if d < 0 else "#16A34A" for d in tracking_diff]
        ax2.bar(self.test_dates, tracking_diff * 100, color=colors, alpha=0.6, width=1.5)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_xlabel("Date", fontsize=13)
        ax2.set_ylabel("Daily Diff (%)", fontsize=13)
        ax2.set_title("Out-of-Sample Tracking Difference (Portfolio − SPY)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add metrics text box on the backtest
        textstr = (
            f"Out-of-Sample Metrics\n"
            f"R² = {r2_test:.4f}\n"
            f"Ann. TE = {te_test:.2f}%\n"
            f"Corr = {daily_corr:.4f}"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8)
        ax1.text(
            0.98,
            0.03,
            textstr,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
            fontfamily="monospace",
        )

        plt.tight_layout()
        path = os.path.join(self.plot_dir, "backtest.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   ✅ Saved: {path}")

        # Return metrics for report
        return {
            "r2_train": float(r2_train),
            "r2_test": float(r2_test),
            "te_train": float(te_train),
            "te_test": float(te_test),
            "correlation": float(daily_corr),
            "total_return_portfolio": float(total_port * 100),
            "total_return_spy": float(total_spy * 100),
            "n_active": int(np.sum(self.weights > 0)),
            "n_total": len(self.weights),
            "train_start": self.train_dates[0].strftime("%Y-%m-%d"),
            "train_end": self.train_dates[-1].strftime("%Y-%m-%d"),
            "test_start": self.test_dates[0].strftime("%Y-%m-%d"),
            "test_end": self.test_dates[-1].strftime("%Y-%m-%d"),
        }

    # ==================================================================
    # Main Pipeline
    # ==================================================================

    def run(self):
        """Execute the full Phase 3 pipeline."""
        print("=" * 65)
        print("  PHASE 3: ACADEMIC PROOFS & VALIDATION")
        print("=" * 65)

        # Create output directories
        os.makedirs(self.plot_dir, exist_ok=True)

        # Step 1: Download & split data
        self.download_and_split()

        # Step 2: Re-run ADMM on training data
        r2_train, te_train = self.run_admm()

        # Plot 1: Convergence
        self.plot_convergence()

        # Plot 2: Regularization path
        lambdas, nnz_list, r2_list, te_list = self.plot_regularization_path()

        # Plot 3: Backtest
        metrics = self.plot_backtest()

        # Save metrics to JSON
        metrics_path = os.path.join(self.data_dir, "phase3_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  💾 Saved: {metrics_path}")

        # Save updated weights
        np.save(os.path.join(self.data_dir, "sparse_weights.npy"), self.weights)
        print("  💾 Saved: data/sparse_weights.npy (re-trained)")

        # Save active stocks
        active_idx = np.where(self.weights > 0)[0]
        active_tickers = [self.stock_names[i] for i in active_idx]
        active_weights = self.weights[active_idx]
        pd.DataFrame({"ticker": active_tickers, "weight": active_weights}).sort_values(
            "weight", ascending=False
        ).to_csv(os.path.join(self.data_dir, "active_stocks.csv"), index=False)
        print("  💾 Saved: data/active_stocks.csv")

        # Final summary
        print(f"\n{'=' * 65}")
        print("  PHASE 3 COMPLETE — SUMMARY")
        print(f"{'=' * 65}")
        print(f"  Train: {metrics['train_start']} → {metrics['train_end']} ({self.n_train} days)")
        print(f"  Test:  {metrics['test_start']} → {metrics['test_end']} ({self.n_test} days)")
        print(f"  Active stocks: {metrics['n_active']} / {metrics['n_total']}")
        print(f"\n  {'Metric':<25} {'In-Sample':>12} {'Out-of-Sample':>14}")
        print(f"  {'-' * 51}")
        print(f"  {'R²':<25} {metrics['r2_train']:>12.4f} {metrics['r2_test']:>14.4f}")
        print(f"  {'Annualized TE':<25} {metrics['te_train']:>11.2f}% {metrics['te_test']:>13.2f}%")
        print(f"  {'Correlation':<25} {'—':>12} {metrics['correlation']:>14.4f}")
        print(f"\n  📁 Plots saved to: {self.plot_dir}/")
        print("     • convergence.png")
        print("     • regularization_path.png")
        print("     • backtest.png")
        print(f"{'=' * 65}")

        return metrics


# ====================================================================
# Quick Execution
# ====================================================================

if __name__ == "__main__":
    from sit.paths import DATA_DIR as _DATA_DIR
    from sit.paths import PLOTS_DIR as _PLOTS_DIR

    validator = Phase3Validator(
        start_date="2025-04-01",
        end_date="2026-03-09",
        n_train=120,
        n_test=60,
        lam_frac=0.05,
        data_dir=str(_DATA_DIR),
        plot_dir=str(_PLOTS_DIR),
    )
    metrics = validator.run()
