import Link from "next/link";
import { HeroSphereClient } from "@/components/HeroSphereClient";
import {
  ArrowRight,
  BarChart3,
  Brain,
  CircuitBoard,
  Globe2,
  Layers,
  ServerCog,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { StatCard } from "@/components/StatCard";
import { getWalkforward, getCrossIndex } from "@/lib/datasets";
import { fmtPct } from "@/lib/format";

const FEATURES = [
  {
    icon: Brain,
    title: "Custom ADMM solver",
    body: "Cholesky-factorised w-update, Boyd adaptive-ρ, primal+dual stopping.",
  },
  {
    icon: Zap,
    title: "≥10× faster than CVXPY",
    body: "Same optimum to 6 d.p. on (n=120, p=502); ~1 060× vs MOSEK MIQP.",
  },
  {
    icon: ShieldCheck,
    title: "8-regime stress-tested",
    body: "COVID, Volmageddon, 2022 hikes, AI bull, quiet 2024 — R² 0.92–0.97.",
  },
  {
    icon: BarChart3,
    title: "Walk-forward 2018→2025",
    body: "Weekly rebalance, 10 bps round-trip, Fama-French 3F regression.",
  },
  {
    icon: Globe2,
    title: "4 markets supported",
    body: "S&P 500 · Nasdaq-100 · Russell 2000 · Nifty 50.",
  },
  {
    icon: ServerCog,
    title: "Live API",
    body: "FastAPI · Pydantic v2 · Redis · slowapi rate-limit · /openapi.json.",
  },
  {
    icon: CircuitBoard,
    title: "Open source",
    body: "MIT-licensed, 274 pytest, ruff/black/mypy clean, GitHub Actions CI.",
  },
  {
    icon: Layers,
    title: "Production deployed",
    body: "Multi-stage Dockerfile · Azure Container Apps target · App Insights.",
  },
];

export default async function Home() {
  const wf = await getWalkforward();
  const cross = await getCrossIndex();
  const admm = wf.risk_metrics.admm;

  return (
    <>
      {/* HERO */}
      <section className="brand-grid relative overflow-hidden">
        <div className="mx-auto grid max-w-7xl items-center gap-12 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:py-24">
          <div className="space-y-6">
            <span className="inline-flex items-center rounded-full border border-[var(--primary)]/40 bg-[var(--primary)]/10 px-3 py-1 text-xs font-medium text-[var(--primary)]">
              v1.0 · 274 pytest passing · MIT
            </span>
            <h1 className="text-balance text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              Replicate the S&amp;P 500 with{" "}
              <span className="text-[var(--primary)]">50 stocks</span>.
              <br className="hidden sm:block" /> Mathematically.
            </h1>
            <p className="text-balance max-w-xl text-lg text-[var(--muted-foreground)]">
              A custom ADMM solver tracks any major index with ~10% of its constituents.{" "}
              <strong className="text-[var(--foreground)]">R² = 0.97 across 8 regimes</strong>,{" "}
              <strong className="text-[var(--foreground)]">
                {fmtPct(admm.ann_return, 2)} annualised return
              </strong>
              ,{" "}
              <strong className="text-[var(--foreground)]">Sharpe {admm.sharpe.toFixed(2)}</strong>{" "}
              on the 2018–2025 walk-forward — net of 10 bps round-trip costs.
            </p>
            <div className="flex flex-wrap gap-3">
              <Link
                href="/invest"
                className="inline-flex items-center gap-2 rounded-md bg-[var(--primary)] px-5 py-3 font-medium text-[var(--primary-foreground)] shadow hover:opacity-90"
              >
                Try it live <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/research"
                className="inline-flex items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--card)] px-5 py-3 font-medium hover:bg-[var(--card)]/80"
              >
                See the math
              </Link>
              <Link
                href="/backtest"
                className="inline-flex items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--card)] px-5 py-3 font-medium hover:bg-[var(--card)]/80"
              >
                Walk-forward backtest
              </Link>
            </div>
            <div className="grid grid-cols-2 gap-3 pt-4 sm:grid-cols-4">
              <StatCard label="Stocks held" value="50" hint="of 502 (S&P 500)" accent="primary" />
              <StatCard label="Sharpe" value={admm.sharpe.toFixed(2)} hint="2018–2025" />
              <StatCard
                label="Tracking err"
                value={fmtPct(admm.tracking_error_annual, 2)}
                hint="annualised"
              />
              <StatCard
                label="Markets"
                value={Object.keys(cross.runs).length.toString()}
                hint="US + India"
                accent="accent"
              />
            </div>
          </div>

          <div className="flex justify-center lg:justify-end">
            <HeroSphereClient />
          </div>
        </div>
      </section>

      {/* FEATURE GRID */}
      <section className="border-t border-[var(--border)]">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6">
          <h2 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">
            Built like a production quant project, not a hackathon demo.
          </h2>
          <p className="mt-3 max-w-2xl text-[var(--muted-foreground)]">
            Every claim on this page is reproducible from the public repo — curve, cost model,
            baseline, factor regression, and all.
          </p>

          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {FEATURES.map((f) => {
              const Icon = f.icon;
              return (
                <div
                  key={f.title}
                  className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-5 transition-colors hover:border-[var(--primary)]/50"
                >
                  <Icon className="h-6 w-6 text-[var(--primary)]" aria-hidden="true" />
                  <p className="mt-3 font-semibold">{f.title}</p>
                  <p className="mt-1 text-sm text-[var(--muted-foreground)]">{f.body}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* TECH STRIP */}
      <section className="border-t border-[var(--border)]">
        <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
          <p className="text-center text-sm uppercase tracking-widest text-[var(--muted-foreground)]">
            Built with
          </p>
          <ul className="mt-6 flex flex-wrap items-center justify-center gap-x-10 gap-y-3 text-[var(--muted-foreground)]">
            {[
              "Python 3.11",
              "FastAPI",
              "NumPy",
              "CVXPY / MOSEK",
              "Next.js 16",
              "TailwindCSS",
              "Vercel",
              "Azure",
            ].map((t) => (
              <li key={t} className="font-mono text-sm">
                {t}
              </li>
            ))}
          </ul>
        </div>
      </section>
    </>
  );
}
