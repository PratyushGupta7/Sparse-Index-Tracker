"use client";

import { useMemo, useState } from "react";
import { Info } from "lucide-react";
import { DrawdownChart, EquityChart } from "@/components/EquityChart";
import { RegimeTabs } from "@/components/RegimeTabs";
import { StatCard } from "@/components/StatCard";
import { REGIME_WINDOWS } from "@/lib/regimes";
import { fmtMultiplier, fmtPct } from "@/lib/format";
import type {
  CrossIndexData,
  MethodComparisonRow,
  RegimeSummary,
  WalkforwardData,
} from "@/lib/datasets";

const ALL_SERIES = ["admm", "benchmark", "lasso", "omp", "equal_weight_topn"] as const;
type SeriesKey = (typeof ALL_SERIES)[number];

interface Props {
  wf: WalkforwardData;
  cross: CrossIndexData;
  methods: { config: Record<string, unknown>; methods: MethodComparisonRow[] };
  regimes: { regimes: Record<string, RegimeSummary> };
}

export function BacktestView({ wf, cross, methods, regimes }: Props) {
  const [active, setActive] = useState<Set<SeriesKey>>(new Set(["admm", "benchmark"]));
  const [regime, setRegime] = useState<string | null>(null);
  const [indexKey, setIndexKey] = useState<string>("sp500");

  type Row = { date: string } & Record<string, number | string>;
  const merged = useMemo<Row[]>(() => {
    const win = regime ? REGIME_WINDOWS.find((r) => r.id === regime) : null;
    const series: Record<string, Record<string, number>> = {};
    const inWindow = (d: string) => !win || (d >= win.start && d <= win.end);
    for (const s of wf.series) {
      for (const p of s.points) {
        if (!inWindow(p.date)) continue;
        series[p.date] ??= {};
        series[p.date][s.method] = p.value;
      }
    }
    for (const p of wf.benchmark) {
      if (!inWindow(p.date)) continue;
      series[p.date] ??= {};
      series[p.date]["benchmark"] = p.value;
    }
    return Object.entries(series)
      .map(([date, vals]) => ({ date, ...vals }) as Row)
      .sort((a, b) => a.date.localeCompare(b.date));
  }, [wf, regime]);

  const drawdown = useMemo<Row[]>(() => {
    const filtered = merged.filter((p) => typeof p["admm"] === "number");
    const result: Row[] = [];
    let peak = -Infinity;
    for (const p of filtered) {
      const v = p["admm"] as number;
      const newPeak = Math.max(peak, v);
      result.push({ date: p.date, admm: (v - newPeak) / newPeak });
      peak = newPeak;
    }
    return result;
  }, [merged]);

  const indexCard = cross.runs[indexKey];
  const indexAdmm = indexCard?.methods?.admm;

  const m = wf.risk_metrics.admm;

  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
      <header className="space-y-2">
        <p className="text-sm font-medium uppercase tracking-widest text-[var(--primary)]">
          Backtest
        </p>
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
          Walk-forward 2018 → 2025 · weekly rebalance · 10 bps round-trip
        </h1>
        <p className="max-w-3xl text-[var(--muted-foreground)]">
          {wf.metadata.n_rebalances ? String(wf.metadata.n_rebalances) : 418} rebalances, 120-day
          rolling look-back, leftover cash parked in BIL. Numbers below are
          <strong> net of transaction costs</strong>.
        </p>
        {wf.survivorship_bias_flag && (
          <p className="inline-flex items-center gap-2 rounded-md bg-[var(--accent)]/10 px-3 py-2 text-sm text-[var(--accent)]">
            <Info className="h-4 w-4" /> Survivorship bias: today&apos;s S&amp;P 500 membership.
          </p>
        )}
      </header>

      <section className="mt-6 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2">
          {ALL_SERIES.map((s) => {
            const on = active.has(s);
            return (
              <button
                key={s}
                type="button"
                onClick={() =>
                  setActive((prev) => {
                    const next = new Set(prev);
                    if (next.has(s)) next.delete(s);
                    else next.add(s);
                    return next;
                  })
                }
                className={`rounded-md border border-[var(--border)] px-3 py-1.5 text-xs font-medium ${
                  on
                    ? "bg-[var(--primary)]/10 border-[var(--primary)] text-[var(--primary)]"
                    : "bg-[var(--card)] text-[var(--muted-foreground)]"
                }`}
                aria-pressed={on}
              >
                {s}
              </button>
            );
          })}
        </div>
        <RegimeTabs value={regime} onChange={setRegime} />
      </section>

      <section className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--card)] p-4">
        <EquityChart data={merged} series={[...active]} />
      </section>

      <section className="mt-4 rounded-xl border border-[var(--border)] bg-[var(--card)] p-4">
        <p className="mb-2 text-sm text-[var(--muted-foreground)]">ADMM drawdown</p>
        <DrawdownChart data={drawdown} series="admm" />
      </section>

      <section className="mt-6 grid gap-3 sm:grid-cols-2 md:grid-cols-4">
        <StatCard label="Ann. return" value={fmtPct(m.ann_return, 2)} accent="primary" />
        <StatCard label="Vol" value={fmtPct(m.ann_vol, 2)} />
        <StatCard label="Sharpe" value={m.sharpe.toFixed(2)} accent="primary" />
        <StatCard label="Sortino" value={m.sortino.toFixed(2)} />
        <StatCard label="Max DD" value={fmtPct(m.max_drawdown, 2)} />
        <StatCard label="Tracking err" value={fmtPct(m.tracking_error_annual, 2)} />
        <StatCard label="Info ratio" value={m.information_ratio.toFixed(2)} />
        <StatCard
          label="Turnover"
          value={fmtMultiplier(m.turnover_annualized, 1)}
          hint="annualised"
        />
      </section>

      <section className="mt-12 space-y-4">
        <div className="flex items-end justify-between">
          <h2 className="text-2xl font-semibold">Cross-market summary</h2>
          <select
            value={indexKey}
            onChange={(e) => setIndexKey(e.target.value)}
            className="rounded-md border border-[var(--border)] bg-[var(--card)] px-3 py-2 text-sm"
            aria-label="Choose index"
          >
            {Object.entries(cross.runs).map(([k, v]) => (
              <option key={k} value={k}>
                {v.label}
              </option>
            ))}
          </select>
        </div>
        {indexAdmm && (
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">
            <StatCard label="OOS R²" value={indexAdmm.oos_r2.toFixed(3)} accent="primary" />
            <StatCard label="OOS TE" value={fmtPct(indexAdmm.oos_te_annual, 2)} />
            <StatCard label="OOS Sharpe" value={indexAdmm.oos_sharpe_annual.toFixed(2)} />
            <StatCard label="# stocks" value={indexAdmm.n_active.toString()} accent="accent" />
          </div>
        )}
        {indexCard?.metadata?.source === "synthetic" && (
          <p className="rounded-md bg-[var(--accent)]/10 px-3 py-2 text-xs text-[var(--accent)]">
            Demo data — re-run with{" "}
            <code className="font-mono">
              python -m sit.benchmarks.multi_index --indices {indexKey}
            </code>{" "}
            for live numbers.
          </p>
        )}
      </section>

      <section className="mt-12 space-y-4">
        <h2 className="text-2xl font-semibold">Head-to-head baselines</h2>
        <div className="overflow-x-auto rounded-xl border border-[var(--border)] bg-[var(--card)]">
          <table className="w-full text-sm">
            <thead className="bg-[var(--card)]/60 text-[var(--muted-foreground)]">
              <tr>
                <th className="px-3 py-2 text-left">Method</th>
                <th className="px-3 py-2 text-right">In-sample R²</th>
                <th className="px-3 py-2 text-right">OOS R²</th>
                <th className="px-3 py-2 text-right">OOS TE</th>
                <th className="px-3 py-2 text-right">Stocks</th>
                <th className="px-3 py-2 text-right">Fit (s)</th>
              </tr>
            </thead>
            <tbody>
              {methods.methods.map((row) => (
                <tr key={row.name} className="border-t border-[var(--border)] font-mono">
                  <td className="px-3 py-2 font-sans font-medium">{row.name}</td>
                  <td className="px-3 py-2 text-right">{row.in_sample_r2.toFixed(3)}</td>
                  <td className="px-3 py-2 text-right">{row.oos_r2.toFixed(3)}</td>
                  <td className="px-3 py-2 text-right">{fmtPct(row.oos_te_annual, 2)}</td>
                  <td className="px-3 py-2 text-right">{row.n_active}</td>
                  <td className="px-3 py-2 text-right">{row.fit_time_s.toExponential(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-12 space-y-4">
        <h2 className="text-2xl font-semibold">8-regime stress test</h2>
        <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">
          {Object.entries(regimes.regimes).map(([k, r]) => (
            <div key={k} className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-4">
              <p className="text-xs uppercase tracking-wide text-[var(--muted-foreground)]">
                {r.type}
              </p>
              <p className="mt-1 font-semibold">{r.short}</p>
              <p className="mt-2 font-mono text-sm">
                R²={r.r2_test.toFixed(3)} · TE={(r.te_test * 100).toFixed(2)}%
              </p>
              <p className="font-mono text-xs text-[var(--muted-foreground)]">
                ρ={r.correlation.toFixed(3)} · {r.n_active} stocks
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
