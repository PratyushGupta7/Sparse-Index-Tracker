import { cn } from "@/lib/format";

export interface StatCardProps {
  label: string;
  value: string;
  hint?: string;
  accent?: "primary" | "accent" | "muted";
  className?: string;
}

export function StatCard({ label, value, hint, accent = "muted", className }: StatCardProps) {
  const valueClass =
    accent === "primary"
      ? "text-[var(--primary)]"
      : accent === "accent"
        ? "text-[var(--accent)]"
        : "text-[var(--foreground)]";
  return (
    <div className={cn("rounded-lg border border-[var(--border)] bg-[var(--card)] p-4", className)}>
      <p className="text-xs font-medium uppercase tracking-wide text-[var(--muted-foreground)]">
        {label}
      </p>
      <p className={cn("mt-1 font-mono text-2xl font-semibold", valueClass)}>{value}</p>
      {hint && <p className="mt-1 text-xs text-[var(--muted-foreground)]">{hint}</p>}
    </div>
  );
}
