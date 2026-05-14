import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "API explorer",
  description: "Public, open API to the sparse index replication engine.",
};

const CURL_EXAMPLES = [
  {
    label: "Health",
    cmd: (base: string) => `curl ${base}/api/v1/health`,
  },
  {
    label: "Portfolio (cached weights)",
    cmd: (base: string) => `curl '${base}/api/v1/portfolio?index=sp500'`,
  },
  {
    label: "Allocate $10k",
    cmd: (base: string) => `curl '${base}/api/v1/invest?capital=10000&index=sp500'`,
  },
  {
    label: "Live retrain on Nifty 50",
    cmd: (base: string) => `curl '${base}/api/v1/invest_live?capital=500000&index=nifty50'`,
  },
  {
    label: "Walk-forward 2020 window",
    cmd: (base: string) =>
      `curl '${base}/api/v1/backtest/walkforward?start=2020-01-01&end=2020-12-31'`,
  },
  {
    label: "λ-path for the slider",
    cmd: (base: string) => `curl '${base}/api/v1/lambda-path?index=sp500'`,
  },
];

export default function ApiPage() {
  const base = "/api/proxy";
  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <header className="space-y-2">
        <p className="text-sm font-medium uppercase tracking-widest text-[var(--primary)]">API</p>
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
          Open API · no auth · slowapi rate-limit only
        </h1>
        <p className="max-w-3xl text-[var(--muted-foreground)]">
          Explore the sparse index replication API through this site&apos;s secure proxy. CORS
          allowlist, Redis caching, telemetry, and rate limits are env-var driven.
        </p>
      </header>

      <section className="mt-8 space-y-4">
        <h2 className="text-xl font-semibold">cURL examples</h2>
        <div className="space-y-3">
          {CURL_EXAMPLES.map((ex) => (
            <div
              key={ex.label}
              className="rounded-lg border border-[var(--border)] bg-[var(--code-bg)] p-4"
            >
              <p className="text-xs font-medium uppercase text-[var(--muted-foreground)]">
                {ex.label}
              </p>
              <pre className="mt-1 overflow-x-auto font-mono text-sm">
                <code>{ex.cmd(base)}</code>
              </pre>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-8 space-y-2 rounded-xl border border-[var(--border)] bg-[var(--card)] p-6">
        <h2 className="text-xl font-semibold">Swagger UI</h2>
        <p className="text-sm text-[var(--muted-foreground)]">
          Use the embedded Swagger UI below to inspect endpoints and schemas.
        </p>
        <iframe
          src={`${base}/docs`}
          title="Swagger UI"
          loading="lazy"
          className="h-[640px] w-full rounded-md border border-[var(--border)] bg-white"
        />
      </section>
    </div>
  );
}
