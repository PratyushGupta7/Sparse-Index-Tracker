import { z } from "zod";
import { API_URL } from "./env";

/* ------------------------------------------------------------------ */
/* Schemas mirror src/sit/api/schemas.py                               */
/* ------------------------------------------------------------------ */

const allocationSchema = z.object({
  ticker: z.string(),
  shares: z.number(),
  price: z.number(),
  weight: z.number(),
  allocated: z.number(),
  actual_cost: z.number(),
});

export const investLiveSchema = z.object({
  mode: z.string(),
  index: z.string(),
  benchmark: z.string(),
  capital: z.number(),
  total_invested: z.number(),
  residual_cash: z.number(),
  utilization_pct: z.string(),
  n_stocks_bought: z.number(),
  price_date: z.string(),
  total_time_seconds: z.number(),
  model: z.object({
    train_period: z.string(),
    r2_train: z.number(),
    te_train_pct: z.number(),
    active_stocks: z.number(),
    universe_size: z.number(),
    converged: z.boolean(),
    iterations: z.number(),
    solve_time_ms: z.number(),
    solver_iterations: z.number(),
  }),
  allocations: z.array(allocationSchema),
  warnings: z.array(z.string()).nullable(),
});

export type InvestLiveResponse = z.infer<typeof investLiveSchema>;

export const lambdaPathSchema = z.object({
  index: z.string(),
  n_train: z.number(),
  n_test: z.number(),
  universe_size: z.number(),
  cached: z.boolean(),
  points: z.array(
    z.object({
      lam: z.number(),
      lam_frac: z.number(),
      nnz: z.number(),
      in_sample_r2: z.number(),
      oos_te: z.number(),
    })
  ),
});

export type LambdaPathResponse = z.infer<typeof lambdaPathSchema>;

/* ------------------------------------------------------------------ */
/* Client (uses the proxy on the browser, direct in server components) */
/* ------------------------------------------------------------------ */

function joinPath(base: string, path: string): string {
  return `${base.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
}

export interface ApiClientOpts {
  baseUrl?: string;
  signal?: AbortSignal;
}

export async function fetchInvestLive(
  capital: number,
  index = "sp500",
  opts: ApiClientOpts = {}
): Promise<InvestLiveResponse> {
  const base = opts.baseUrl ?? "/api/proxy";
  const url = `${joinPath(base, "/api/v1/invest_live")}?capital=${encodeURIComponent(
    capital
  )}&index=${encodeURIComponent(index)}`;
  const r = await fetch(url, { method: "GET", signal: opts.signal });
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`API ${r.status}: ${body}`);
  }
  return investLiveSchema.parse(await r.json());
}

export async function fetchLambdaPath(
  index = "sp500",
  opts: ApiClientOpts = {}
): Promise<LambdaPathResponse> {
  const base = opts.baseUrl ?? "/api/proxy";
  const url = `${joinPath(base, "/api/v1/lambda-path")}?index=${encodeURIComponent(index)}`;
  const r = await fetch(url, { signal: opts.signal });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return lambdaPathSchema.parse(await r.json());
}

export const apiServerUrl = API_URL;
