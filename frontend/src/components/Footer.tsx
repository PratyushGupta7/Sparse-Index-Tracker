export function Footer() {
  return (
    <footer className="border-t border-[var(--border)] bg-[var(--background)]/80">
      <div className="mx-auto flex max-w-7xl flex-col items-start justify-between gap-4 px-4 py-8 text-sm text-[var(--muted-foreground)] sm:flex-row sm:items-center sm:px-6">
        <div>
          <p className="font-medium text-[var(--foreground)]">Sparse Index Tracker</p>
          <p>Custom ADMM solver · 2018–2025 walk-forward · 4 markets · MIT licensed.</p>
        </div>
        <div className="flex flex-wrap gap-4">
          <a
            href="https://github.com/PratyushGupta7/Sparse-Index-Tracker"
            className="hover:text-[var(--primary)]"
          >
            GitHub
          </a>
          <a href="/api" className="hover:text-[var(--primary)]">
            API docs
          </a>
          <a href="/research" className="hover:text-[var(--primary)]">
            Math
          </a>
          <a href="/backtest" className="hover:text-[var(--primary)]">
            Backtest
          </a>
        </div>
      </div>
    </footer>
  );
}
