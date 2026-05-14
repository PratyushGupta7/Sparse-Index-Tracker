export const fmtUsd = (n: number, digits = 0): string =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: digits,
  }).format(n);

export const fmtPct = (n: number, digits = 2): string => `${(n * 100).toFixed(digits)}%`;

export const fmtBps = (n: number): string => `${(n * 10_000).toFixed(0)} bps`;

export const fmtNumber = (n: number, digits = 2): string =>
  n.toLocaleString("en-US", { maximumFractionDigits: digits });

export const fmtMultiplier = (n: number, digits = 1): string => `${n.toFixed(digits)}x`;

export const cn = (...parts: (string | undefined | null | false)[]): string =>
  parts.filter(Boolean).join(" ");
