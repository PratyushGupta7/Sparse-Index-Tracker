/**
 * Server-side helpers that read the JSON artefacts shipped under
 * `frontend/public/data/`. Used by Server Components so the static pages
 * (/, /research, /backtest) render fully without a live API.
 */

import fs from "node:fs/promises";
import path from "node:path";

const DATA_DIR = path.join(process.cwd(), "public", "data");

export interface EquityPoint {
  date: string;
  value: number;
}

export interface MethodSeries {
  method: string;
  points: EquityPoint[];
}

export interface RiskMetrics {
  ann_return: number;
  ann_vol: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  ulcer_index: number;
  calmar: number;
  tracking_error_annual: number;
  information_ratio: number;
  beta: number;
  ff3_alpha_annual: number;
  ff3_beta_mkt: number;
  turnover_annualized: number;
  hhi_avg: number;
  effective_n_avg: number;
  [k: string]: number;
}

export interface WalkforwardData {
  config: Record<string, unknown>;
  metadata: Record<string, unknown>;
  survivorship_bias_flag: boolean;
  rebalance_dates: string[];
  series: MethodSeries[];
  benchmark: EquityPoint[];
  risk_metrics: Record<string, RiskMetrics>;
}

export interface CrossIndexMethod {
  oos_r2: number;
  oos_te_annual: number;
  oos_ir_annual: number;
  oos_sharpe_annual: number;
  n_active: number;
  fit_time_s: number;
}

export interface CrossIndexRun {
  label: string;
  benchmark: string;
  region: string;
  n_active: number;
  metadata: { source?: string; [k: string]: unknown };
  methods: Record<string, CrossIndexMethod>;
}

export interface CrossIndexData {
  config: Record<string, unknown>;
  elapsed_s: number;
  survivorship_bias_flag: boolean;
  runs: Record<string, CrossIndexRun>;
}

export interface MethodComparisonRow {
  name: string;
  n_active: number;
  in_sample_r2: number;
  oos_r2: number;
  oos_te_annual: number;
  fit_time_s: number;
  max_weight: number;
  hhi: number;
}

export interface RegimeSummary {
  regime: string;
  short: string;
  type: string;
  color?: string;
  train_period: string;
  test_period: string;
  r2_test: number;
  te_test: number;
  correlation: number;
  n_active: number;
  iterations: number;
}

async function readJson<T>(name: string): Promise<T> {
  const buf = await fs.readFile(path.join(DATA_DIR, name), "utf-8");
  return JSON.parse(buf) as T;
}

export const getWalkforward = () => readJson<WalkforwardData>("walkforward.json");
export const getCrossIndex = () => readJson<CrossIndexData>("cross_index.json");
export const getMethodComparison = () =>
  readJson<{ config: Record<string, unknown>; methods: MethodComparisonRow[] }>(
    "method_comparison.json"
  );
export const getSpeedup = () =>
  readJson<{
    config: Record<string, unknown>;
    table: {
      p: number;
      admm_s: number;
      fista_s: number;
      cvxpy_s: number | null;
      admm_speedup_vs_cvxpy: number | null;
      fista_speedup_vs_cvxpy: number | null;
    }[];
  }>("speedup.json");
export const getConvergence = () =>
  readJson<{ primal: number[]; dual: number[]; tol: number }>("convergence.json");
export const getRegimes = () =>
  readJson<{ regimes: Record<string, RegimeSummary> }>("regimes.json");
