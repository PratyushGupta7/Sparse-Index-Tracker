"use client";

import { REGIME_WINDOWS } from "@/lib/regimes";
import { cn } from "@/lib/format";

export interface RegimeTabsProps {
  value: string | null;
  onChange: (id: string | null) => void;
}

export function RegimeTabs({ value, onChange }: RegimeTabsProps) {
  return (
    <div className="flex flex-wrap gap-2" role="tablist" aria-label="Regime windows">
      <button
        type="button"
        onClick={() => onChange(null)}
        className={cn(
          "rounded-md border border-[var(--border)] bg-[var(--card)] px-3 py-1.5 text-xs font-medium",
          value === null && "border-[var(--primary)] text-[var(--primary)]"
        )}
        role="tab"
        aria-selected={value === null}
      >
        Full sample
      </button>
      {REGIME_WINDOWS.map((r) => (
        <button
          key={r.id}
          type="button"
          onClick={() => onChange(r.id)}
          title={r.description}
          className={cn(
            "rounded-md border border-[var(--border)] bg-[var(--card)] px-3 py-1.5 text-xs font-medium",
            value === r.id && "border-[var(--primary)] text-[var(--primary)]"
          )}
          role="tab"
          aria-selected={value === r.id}
        >
          {r.label}
        </button>
      ))}
    </div>
  );
}
