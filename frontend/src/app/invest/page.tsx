import type { Metadata } from "next";
import { InvestForm } from "./InvestForm";

export const metadata: Metadata = {
  title: "Invest — live ADMM retrain demo",
  description: "Enter capital + index, get a freshly-retrained sparse portfolio with share counts.",
};

export default function InvestPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <header className="space-y-2">
        <p className="text-sm font-medium uppercase tracking-widest text-[var(--primary)]">
          Invest
        </p>
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">Live ADMM retrain</h1>
        <p className="max-w-3xl text-[var(--muted-foreground)]">
          Enter how much you&apos;d allocate. The API will pull the last 120 trading days, retrain
          ADMM from scratch, and return shares to buy at today&apos;s prices. Open{" "}
          <code className="font-mono">localhost:8000</code> first (or set{" "}
          <code className="font-mono">NEXT_PUBLIC_API_URL</code>).
        </p>
      </header>

      <InvestForm />
    </div>
  );
}
