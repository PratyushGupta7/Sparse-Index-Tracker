"""benchmarks/walkforward.py — Phase-3 walk-forward backtest CLI.

Runs the rolling rebalance backtest configured by a YAML or CLI flags, then
emits

  data/backtest/wf_2018_2025/results.json
  plots/walkforward_equity.png
  plots/walkforward_drawdown.png
  plots/walkforward_factor_loadings.png
  plots/walkforward_turnover.png

Examples
--------
::

    python -m benchmarks.walkforward --config configs/wf_2018_2025.yaml
    python -m benchmarks.walkforward --start 2020-01-01 --end 2022-12-31 \
        --rebalance monthly --tx-bps 10 --K 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:  # pragma: no cover (only matters when run as a script)
    sys.path.insert(0, str(SRC_DIR))

from sit.backtest import (  # noqa: E402
    WalkForwardConfig,
    compute_risk_metrics,
    load_price_panel_yfinance,
    run,
    save_walkforward_result,
)
from sit.backtest.plots import (  # noqa: E402
    plot_walkforward_drawdown,
    plot_walkforward_equity,
    plot_walkforward_factor_loadings,
    plot_walkforward_turnover,
)
from sit.benchmarks.datasets import _fetch_sp500_tickers  # noqa: E402
from sit.data.famafrench import load_famafrench_daily  # noqa: E402
from sit.paths import DATA_DIR, PLOTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("walkforward.cli")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase-3 walk-forward backtest with transaction costs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=None, help="YAML config (overrides CLI flags).")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--rebalance", choices=("weekly", "monthly"), default=None)
    p.add_argument("--tx-bps", type=float, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--lam-frac", type=float, default=None)
    p.add_argument("--initial-capital", type=float, default=None)
    p.add_argument("--cash-proxy", default=None)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--methods", default=None, help="Comma-separated methods.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override the output directory (defaults to data/backtest/wf_<start>_<end>).",
    )
    p.add_argument(
        "--no-factors",
        action="store_true",
        help="Skip the FF3 factor download / regression / rolling-betas plot.",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> WalkForwardConfig:
    payload: dict[str, Any] = {}
    if args.config is not None:
        with open(args.config) as f:
            payload = yaml.safe_load(f) or {}
    overrides = {
        "start_date": args.start,
        "end_date": args.end,
        "rebalance": args.rebalance,
        "tx_bps": args.tx_bps,
        "K": args.K,
        "lam_frac": args.lam_frac,
        "initial_capital": args.initial_capital,
        "cash_proxy_ticker": args.cash_proxy,
        "benchmark_ticker": args.benchmark,
        "methods": [m.strip() for m in args.methods.split(",")] if args.methods else None,
    }
    for k, v in overrides.items():
        if v is not None:
            payload[k] = v
    return WalkForwardConfig(**payload)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    cfg = _build_config(args)
    logger.info("config: %s", cfg.model_dump())

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else DATA_DIR / "backtest" / f"wf_{cfg.start_date}_{cfg.end_date}".replace("-", "")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output directory: %s", out_dir)

    # 1. Universe: today's S&P 500 (Wikipedia).
    logger.info("fetching S&P 500 universe via Wikipedia")
    tickers = _fetch_sp500_tickers(benchmark=cfg.benchmark_ticker)

    # 2. Price panel via yfinance (single download, then split).
    logger.info("downloading prices for %d tickers + benchmark", len(tickers))
    prices, benchmark, cash_returns, active = load_price_panel_yfinance(
        tickers,
        benchmark=cfg.benchmark_ticker,
        start_date="2017-06-01",  # buffer for the first lookback
        end_date=cfg.end_date,
        cash_proxy_ticker=cfg.cash_proxy_ticker,
    )
    logger.info("downloaded %d active tickers, %d trading days", len(active), len(prices))

    # 3. Run the backtest.
    t0 = time.perf_counter()
    result = run(cfg, prices, benchmark, cash_returns=cash_returns)
    logger.info(
        "walk-forward finished in %.1fs (%d rebalances)",
        time.perf_counter() - t0,
        len(result.rebalance_dates),
    )

    # 4. FF3 factors (optional).
    factors = None
    if not args.no_factors:
        try:
            factors = load_famafrench_daily()
        except Exception as exc:  # pragma: no cover (network-dependent)
            logger.warning("FF3 download failed: %r — skipping factor analytics", exc)

    # 5. Risk metrics per method.
    risk_metrics: dict[str, dict[str, float]] = {}
    for method, curve in result.equity_curves.items():
        risk_metrics[method] = compute_risk_metrics(
            curve,
            result.benchmark_curve,
            rebalance_dates=result.rebalance_dates,
            weights_history=result.weights_history.get(method),
            turnover=result.turnover.get(method),
            factors=factors,
        )

    # 6. Persist + plot.
    json_path = out_dir / "results.json"
    save_walkforward_result(
        result,
        json_path,
        risk_metrics=risk_metrics,
        extra={
            "n_active_tickers": len(active),
            "buffer_start": "2017-06-01",
        },
    )
    logger.info("wrote %s", json_path)

    plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    sub = (
        f"{cfg.start_date} → {cfg.end_date} | {cfg.rebalance} | "
        f"{cfg.tx_bps:.1f} bps/side | K={cfg.K}, λ-frac={cfg.lam_frac}"
    )
    plot_walkforward_equity(result, plots_dir / "walkforward_equity.png", subtitle=sub)
    plot_walkforward_drawdown(result, plots_dir / "walkforward_drawdown.png", subtitle=sub)
    plot_walkforward_turnover(
        result,
        plots_dir / "walkforward_turnover.png",
        method="admm",
        subtitle=f"ADMM, {sub}",
    )
    if factors is not None:
        plot_walkforward_factor_loadings(
            result,
            factors,
            plots_dir / "walkforward_factor_loadings.png",
            method="admm",
            window=90,
            subtitle=f"ADMM rolling 90-day β to FF3 factors | {sub}",
        )

    print("\n=== Walk-forward summary ===")
    headers = ["method", "ann_ret", "sharpe", "max_dd", "TE", "IR", "turnover/yr"]
    print("  " + "  ".join(f"{h:>14s}" for h in headers))
    for method, m in risk_metrics.items():
        print(
            "  "
            + "  ".join(
                [
                    f"{method:>14s}",
                    f"{m.get('ann_return', float('nan')):>14.4f}",
                    f"{m.get('sharpe', float('nan')):>14.3f}",
                    f"{m.get('max_drawdown', float('nan')):>14.4f}",
                    f"{m.get('tracking_error_annual', float('nan')):>14.4f}",
                    f"{m.get('information_ratio', float('nan')):>14.3f}",
                    f"{m.get('turnover_annualized', float('nan')):>14.3f}",
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
