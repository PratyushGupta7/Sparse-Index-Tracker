"use client";

import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { toast } from "sonner";
import { AllocationTreemap } from "@/components/AllocationTreemap";
import { StatCard } from "@/components/StatCard";
import { fetchInvestLive, type InvestLiveResponse } from "@/lib/api";
import { fmtPct, fmtUsd } from "@/lib/format";

const FormSchema = z.object({
  capital: z.number().min(100, "Min $100").max(10_000_000, "Max $10M"),
  index: z.enum(["sp500", "nasdaq100", "russell2000", "nifty50"]),
});

type FormValues = z.infer<typeof FormSchema>;

const INDICES: { value: FormValues["index"]; label: string }[] = [
  { value: "sp500", label: "S&P 500" },
  { value: "nasdaq100", label: "Nasdaq 100" },
  { value: "russell2000", label: "Russell 2000" },
  { value: "nifty50", label: "Nifty 50" },
];

export function InvestForm() {
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({
    resolver: zodResolver(FormSchema),
    defaultValues: { capital: 100_000, index: "sp500" },
  });

  const [result, setResult] = useState<InvestLiveResponse | null>(null);
  const [progress, setProgress] = useState(0);
  const [iter, setIter] = useState(0);

  useEffect(() => {
    if (!isSubmitting) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setProgress(0);
    setIter(0);
    const start = performance.now();
    const target = 8000;
    const id = setInterval(() => {
      const elapsed = performance.now() - start;
      const p = Math.min(elapsed / target, 0.97);
      setProgress(p);
      setIter(Math.floor(p * 5000));
    }, 80);
    return () => clearInterval(id);
  }, [isSubmitting]);

  const onSubmit = handleSubmit(async (values) => {
    try {
      const res = await fetchInvestLive(values.capital, values.index);
      setResult(res);
      setProgress(1);
      setIter(res.model.solver_iterations);
      toast.success(`Solved in ${res.model.solve_time_ms.toFixed(0)} ms`);
    } catch (err) {
      toast.error(String(err));
    }
  });

  const copy = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    toast.success(`${label} copied`);
  };

  return (
    <div className="mt-8 space-y-8">
      <form
        onSubmit={onSubmit}
        className="grid gap-4 rounded-xl border border-[var(--border)] bg-[var(--card)] p-6 sm:grid-cols-[1fr_1fr_auto]"
      >
        <label className="space-y-1">
          <span className="text-sm font-medium">Capital (USD)</span>
          <input
            type="number"
            step={100}
            {...register("capital", { valueAsNumber: true })}
            className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2"
            aria-invalid={!!errors.capital}
          />
          {errors.capital && <span className="text-xs text-red-500">{errors.capital.message}</span>}
        </label>

        <label className="space-y-1">
          <span className="text-sm font-medium">Index</span>
          <select
            {...register("index")}
            className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2"
          >
            {INDICES.map((i) => (
              <option key={i.value} value={i.value}>
                {i.label}
              </option>
            ))}
          </select>
        </label>

        <button
          type="submit"
          disabled={isSubmitting}
          className="self-end rounded-md bg-[var(--primary)] px-5 py-2.5 font-medium text-[var(--primary-foreground)] disabled:opacity-50"
        >
          {isSubmitting ? "Solving…" : "Get allocations"}
        </button>
      </form>

      {isSubmitting && (
        <div className="space-y-2 rounded-xl border border-[var(--border)] bg-[var(--card)] p-4">
          <div className="flex items-center justify-between text-sm">
            <span>ADMM iteration {iter} / 5000</span>
            <span className="font-mono">{(progress * 100).toFixed(0)}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--background)]">
            <div
              className="h-full bg-[var(--primary)] transition-all"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      )}

      {result && (
        <section className="space-y-6">
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">
            <StatCard label="Invested" value={fmtUsd(result.total_invested, 2)} accent="primary" />
            <StatCard label="Cash residual" value={fmtUsd(result.residual_cash, 2)} />
            <StatCard label="In-sample R²" value={result.model.r2_train.toFixed(3)} />
            <StatCard
              label="OOS TE"
              value={`${result.model.te_train_pct.toFixed(2)}%`}
              accent="accent"
            />
          </div>
          <p className="text-sm text-[var(--muted-foreground)]">
            Trained on <code>{result.model.train_period}</code> · solved in{" "}
            <code>{result.model.solve_time_ms.toFixed(0)} ms</code> · {result.model.iterations}{" "}
            iterations · {result.model.active_stocks} of {result.model.universe_size} stocks active.
          </p>

          <AllocationTreemap
            data={result.allocations.map((a) => ({
              ticker: a.ticker,
              weight: a.weight,
              shares: a.shares,
              cost: a.actual_cost,
            }))}
          />

          <div className="overflow-x-auto rounded-xl border border-[var(--border)] bg-[var(--card)]">
            <table className="w-full text-sm">
              <thead className="text-[var(--muted-foreground)]">
                <tr>
                  <th className="px-3 py-2 text-left">Ticker</th>
                  <th className="px-3 py-2 text-right">Weight</th>
                  <th className="px-3 py-2 text-right">Shares</th>
                  <th className="px-3 py-2 text-right">Price</th>
                  <th className="px-3 py-2 text-right">Cost</th>
                </tr>
              </thead>
              <tbody>
                {result.allocations.map((a) => (
                  <tr key={a.ticker} className="border-t border-[var(--border)] font-mono">
                    <td className="px-3 py-2 font-sans font-semibold">{a.ticker}</td>
                    <td className="px-3 py-2 text-right">{fmtPct(a.weight, 2)}</td>
                    <td className="px-3 py-2 text-right">{a.shares}</td>
                    <td className="px-3 py-2 text-right">{fmtUsd(a.price, 2)}</td>
                    <td className="px-3 py-2 text-right">{fmtUsd(a.actual_cost, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => copy(JSON.stringify(result, null, 2), "JSON")}
              className="rounded-md border border-[var(--border)] bg-[var(--card)] px-4 py-2 text-sm hover:bg-[var(--card)]/80"
            >
              Copy JSON
            </button>
            <button
              type="button"
              onClick={() => {
                const csv = ["ticker,weight,shares,price,cost"]
                  .concat(
                    result.allocations.map(
                      (a) => `${a.ticker},${a.weight},${a.shares},${a.price},${a.actual_cost}`
                    )
                  )
                  .join("\n");
                copy(csv, "CSV");
              }}
              className="rounded-md border border-[var(--border)] bg-[var(--card)] px-4 py-2 text-sm hover:bg-[var(--card)]/80"
            >
              Copy CSV
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
