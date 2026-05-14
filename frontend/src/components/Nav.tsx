"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Menu, X, Github } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { cn } from "@/lib/format";

const NAV = [
  { href: "/", label: "Home" },
  { href: "/invest", label: "Invest" },
  { href: "/research", label: "Research" },
  { href: "/backtest", label: "Backtest" },
  { href: "/api", label: "API" },
];

export function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-[var(--border)] bg-[var(--background)]/85 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <span className="grid h-8 w-8 place-items-center rounded-md bg-[var(--primary)] text-[var(--primary-foreground)] text-sm font-bold">
            Σ
          </span>
          <span className="hidden sm:inline">Sparse Index Tracker</span>
          <span className="sm:hidden">SIT</span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex" aria-label="Main">
          {NAV.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname?.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-[var(--primary)]/10 text-[var(--primary)]"
                    : "text-[var(--foreground)]/70 hover:text-[var(--foreground)] hover:bg-[var(--card)]"
                )}
                aria-current={active ? "page" : undefined}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          <a
            href="https://github.com/PratyushGupta7/Sparse-Index-Tracker"
            target="_blank"
            rel="noreferrer"
            className="hidden rounded-md border border-[var(--border)] bg-[var(--card)] p-2 hover:bg-[var(--card)]/80 md:inline-flex"
            aria-label="GitHub repository"
          >
            <Github className="h-4 w-4" />
          </a>
          <ThemeToggle />
          <button
            type="button"
            className="rounded-md border border-[var(--border)] bg-[var(--card)] p-2 md:hidden"
            onClick={() => setOpen((o) => !o)}
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
          >
            {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {open && (
        <nav
          className="border-t border-[var(--border)] bg-[var(--background)] md:hidden"
          aria-label="Mobile"
        >
          <div className="mx-auto flex max-w-7xl flex-col gap-1 px-4 py-3">
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                className="rounded-md px-3 py-3 text-sm font-medium hover:bg-[var(--card)]"
              >
                {item.label}
              </Link>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
}
