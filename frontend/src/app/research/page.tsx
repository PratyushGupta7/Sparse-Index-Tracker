import { Suspense } from "react";
import { BlockMath, InlineMath } from "react-katex";
import { ConvergenceAnim } from "@/components/ConvergenceAnim";
import { L1L2Geometry } from "@/components/L1L2Geometry";
import { LambdaSlider, type LambdaPoint } from "@/components/LambdaSlider";
import { getConvergence } from "@/lib/datasets";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Research — the math behind sparse index tracking",
  description:
    "ADMM updates, BIC λ-selection, L1 vs L2 geometry, primal/dual residual decay — derived in plain LaTeX.",
};

const FALLBACK_LAMBDA: LambdaPoint[] = [
  { lam_frac: 0.005, nnz: 78, oos_te: 0.012, in_sample_r2: 0.998 },
  { lam_frac: 0.01, nnz: 64, oos_te: 0.014, in_sample_r2: 0.996 },
  { lam_frac: 0.02, nnz: 50, oos_te: 0.017, in_sample_r2: 0.991 },
  { lam_frac: 0.04, nnz: 38, oos_te: 0.022, in_sample_r2: 0.985 },
  { lam_frac: 0.06, nnz: 30, oos_te: 0.027, in_sample_r2: 0.978 },
  { lam_frac: 0.08, nnz: 24, oos_te: 0.033, in_sample_r2: 0.97 },
  { lam_frac: 0.1, nnz: 20, oos_te: 0.038, in_sample_r2: 0.961 },
  { lam_frac: 0.15, nnz: 15, oos_te: 0.045, in_sample_r2: 0.948 },
  { lam_frac: 0.2, nnz: 12, oos_te: 0.052, in_sample_r2: 0.93 },
  { lam_frac: 0.3, nnz: 8, oos_te: 0.064, in_sample_r2: 0.9 },
  { lam_frac: 0.4, nnz: 5, oos_te: 0.082, in_sample_r2: 0.852 },
  { lam_frac: 0.5, nnz: 3, oos_te: 0.097, in_sample_r2: 0.78 },
];

export default async function ResearchPage() {
  const convergence = await getConvergence();

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <header className="space-y-3">
        <p className="text-sm font-medium uppercase tracking-widest text-[var(--primary)]">
          Research
        </p>
        <h1 className="text-balance text-3xl font-bold tracking-tight sm:text-4xl">
          The math behind a sparse, fully-invested, long-only index tracker
        </h1>
        <p className="max-w-3xl text-[var(--muted-foreground)]">
          We solve a high-dimensional convex problem with non-negativity and simplex constraints via
          a custom ADMM splitting. λ controls the bias-variance trade-off between tracking accuracy
          and portfolio sparsity.
        </p>
      </header>

      <section className="mt-12 space-y-6 rounded-xl border border-[var(--border)] bg-[var(--card)] p-6">
        <h2 className="text-2xl font-semibold">The optimisation problem</h2>
        <BlockMath
          math={String.raw`\min_{w \in \mathbb{R}^p}\;\tfrac{1}{2}\|Xw - y\|_2^2 + \lambda\|w\|_1 \quad \text{s.t.}\quad w \ge 0,\; \mathbf{1}^\top w = 1`}
        />
        <p className="text-[var(--muted-foreground)]">
          <InlineMath math="X" /> is an <InlineMath math="n \times p" /> matrix of constituent
          returns, <InlineMath math="y" /> the index return, <InlineMath math="w" /> the desired
          sparse, simplex-constrained portfolio weights.
        </p>
      </section>

      <section className="mt-8 grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-6">
          <h3 className="text-lg font-semibold">L1 vs L2 — why corners give sparsity</h3>
          <p className="mt-2 text-sm text-[var(--muted-foreground)]">
            The L1 ball&apos;s vertices align with coordinate axes — so the constraint hyperplane{" "}
            <InlineMath math="y=Xw" /> almost surely touches it at a vertex (sparse). The L2 ball is
            rotationally symmetric and has no such vertices.
          </p>
          <div className="mt-4 flex justify-center">
            <L1L2Geometry />
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border)] bg-[var(--card)] p-6">
          <h3 className="text-lg font-semibold">ADMM updates (the three steps)</h3>
          <div className="mt-2 space-y-3">
            <BlockMath
              math={String.raw`w^{k+1} = (X^\top X + \rho I)^{-1}\bigl(X^\top y + \rho(z^k - u^k)\bigr)`}
            />
            <BlockMath
              math={String.raw`z^{k+1} = \mathrm{prox}_{(\lambda/\rho)\|\cdot\|_1, +}\!\bigl(w^{k+1} + u^k\bigr)`}
            />
            <BlockMath math={String.raw`u^{k+1} = u^k + w^{k+1} - z^{k+1}`} />
          </div>
          <p className="text-sm text-[var(--muted-foreground)]">
            We Cholesky-factorise <InlineMath math="X^\top X + \rho I" /> once per ρ-update, so each
            iteration costs <InlineMath math="O(p^2)" /> instead of <InlineMath math="O(p^3)" />.
          </p>
        </div>
      </section>

      <section className="mt-8 space-y-4 rounded-xl border border-[var(--border)] bg-[var(--card)] p-6">
        <h2 className="text-2xl font-semibold">Pick your λ — interactive</h2>
        <p className="text-sm text-[var(--muted-foreground)]">
          Slide to see the bias–variance trade-off live: tighter regularisation (right) → fewer
          stocks → higher tracking error.
        </p>
        <Suspense fallback={null}>
          <LambdaSlider points={FALLBACK_LAMBDA} />
        </Suspense>
        <p className="text-xs text-[var(--muted-foreground)]">
          Demo data — when running against a live API, switching to{" "}
          <code>{"/api/proxy/api/v1/lambda-path"}</code> repaints the chart.
        </p>
      </section>

      <section className="mt-8 space-y-4 rounded-xl border border-[var(--border)] bg-[var(--card)] p-6">
        <h2 className="text-2xl font-semibold">Convergence (real residuals)</h2>
        <p className="text-sm text-[var(--muted-foreground)]">
          Primal and dual residuals from a real synthetic 100×300 ADMM solve. Decay is sub-linear at
          the start, geometric near the optimum (Boyd §3.4).
        </p>
        <ConvergenceAnim data={convergence} />
      </section>
    </div>
  );
}
