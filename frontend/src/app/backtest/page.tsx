import type { Metadata } from "next";
import { BacktestView } from "./BacktestView";
import { getCrossIndex, getMethodComparison, getRegimes, getWalkforward } from "@/lib/datasets";

export const metadata: Metadata = {
  title: "Backtest — walk-forward 2018–2025 across 4 markets",
  description:
    "Sharpe, Sortino, max-DD, tracking error, turnover and Fama-French loadings — net of 10 bps round-trip costs.",
};

export default async function BacktestPage() {
  const [wf, cross, methods, regimes] = await Promise.all([
    getWalkforward(),
    getCrossIndex(),
    getMethodComparison(),
    getRegimes(),
  ]);
  return <BacktestView wf={wf} cross={cross} methods={methods} regimes={regimes} />;
}
